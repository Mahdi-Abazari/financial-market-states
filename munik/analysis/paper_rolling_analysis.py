"""Full-resolution rolling daily analysis with disk-backed arrays and resume.

All observations and float64 matrix distances are retained (stride=1).
No quantization, PCA, or random sampling is used. Dates use the notebook's
return-start convention. The 2007-2010 zoom is daily data, not a TAQ replica.
"""

from pathlib import Path
from time import perf_counter
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange, get_num_threads, set_num_threads
from tqdm.auto import tqdm


def rolling_plan(normalized_returns, stride=1):
    z = normalized_returns
    if stride < 1 or int(stride) != stride:
        raise ValueError('stride must be a positive integer')
    if not z.index.is_unique or not z.index.is_monotonic_increasing:
        raise ValueError('Dates must be unique and sorted')
    values = np.ascontiguousarray(z.to_numpy(dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError('Normalized returns must be finite')
    eligible = np.flatnonzero(z.index >= z.index[0] + pd.DateOffset(months=2))[::stride]
    dates = z.index[eligible]
    if len(dates) < 2:
        raise ValueError('Not enough full two-month windows')
    lower_dates = dates - pd.DateOffset(months=2)
    lower = z.index.searchsorted(lower_dates, side='right')
    upper = eligible + 1
    if np.any(upper - lower < 3):
        raise ValueError('Too few observations in a rolling window')
    m, n = len(dates), z.shape[1]
    p = n * (n + 1) // 2
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    digest.update(z.index.asi8.tobytes())
    digest.update(json.dumps(z.columns.tolist()).encode('utf-8'))
    digest.update(f'calendar_months=2;stride={stride};v=1'.encode('ascii'))
    return dict(dates=dates, lower=lower, upper=upper, windows=m, stocks=n,
                features=p, feature_bytes=m*p*8, distance_bytes=m*m*8,
                pairs=m*(m-1)//2, fingerprint=digest.hexdigest(), stride=int(stride))


def _array(path, shape, dtype):
    if path.exists():
        array = np.lib.format.open_memmap(path, mode='r+')
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f'Cached array has wrong shape or dtype: {path}')
        return array
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def build_rolling_dataset(normalized_returns, output_dir, stride=1, progress=True):
    plan = rolling_plan(normalized_returns, stride)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(fingerprint=plan['fingerprint'], windows=plan['windows'],
                  stocks=plan['stocks'], features=plan['features'],
                  stride=plan['stride'], lookback='two calendar months',
                  dtype='float64', dates='return-start labels')
    config_path = folder / 'manifest.json'
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('Input differs from cached data; choose a new output directory')
    else:
        needed = plan['feature_bytes'] + plan['distance_bytes'] + 1_000_000_000
        if shutil.disk_usage(folder).free < needed:
            raise OSError(f'Insufficient disk space; need about {needed / 1e9:.2f} GB free')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    m, n, p = plan['windows'], plan['stocks'], plan['features']
    if (folder / 'feature_rows_done.npy').exists() and (
            not (folder / 'weighted_features.npy').exists()
            or not (folder / 'largest_eigenvalues.npy').exists()):
        raise ValueError('Cached feature files were removed; use a new output directory')
    weighted = _array(folder / 'weighted_features.npy', (m, p), np.float64)
    eigenvalues = _array(folder / 'largest_eigenvalues.npy', (m,), np.float64)
    done_path = folder / 'feature_rows_done.npy'
    was_new = not done_path.exists()
    done = _array(done_path, (m,), np.bool_)
    if was_new:
        done[:] = False
        done.flush()
    row, col = np.triu_indices(n)
    weights = np.where(row == col, 1.0, 2.0) / (n*n)
    values = np.ascontiguousarray(normalized_returns.to_numpy(dtype=np.float64))
    records = pd.DataFrame(dict(date=plan['dates'],
                               first_return_date=normalized_returns.index[plan['lower']],
                               last_return_date=plan['dates'],
                               n_returns=plan['upper']-plan['lower']))
    records.to_csv(folder / 'rolling_windows.csv', index=False)
    pending = np.flatnonzero(~np.asarray(done))
    uncommitted = []
    for i in tqdm(pending, desc='Rolling correlations and eigenvalues', disable=not progress):
        block = values[plan['lower'][i]:plan['upper'][i]]
        centered = block - block.mean(axis=0)
        norms = np.sqrt(np.sum(centered * centered, axis=0))
        if np.any(norms <= 0):
            raise ValueError(f'Constant series in rolling window {plan["dates"][i]}')
        b = centered / norms
        corr = b.T @ b
        weighted[i] = corr[row, col] * weights
        # B.T@B and B@B.T have the same nonzero eigenvalues.
        # The latter is only about 42x42 instead of 296x296.
        eigenvalues[i] = np.linalg.eigvalsh(b @ b.T)[-1]
        uncommitted.append(int(i))
        if len(uncommitted) >= 32:
            weighted.flush()
            eigenvalues.flush()
            done[uncommitted] = True
            done.flush()
            uncommitted.clear()
    if uncommitted:
        weighted.flush()
        eigenvalues.flush()
        done[uncommitted] = True
        done.flush()
    if not np.asarray(done).all():
        raise RuntimeError('Rolling features are incomplete')
    pd.DataFrame(dict(date=plan['dates'], lambda_max=np.asarray(eigenvalues))).to_csv(
        folder / 'largest_eigenvalues.csv', index=False)
    if progress:
        print(f'Rolling dataset ready: {m} windows, {n} stocks; disk-backed float64 arrays', flush=True)
    return dict(features=weighted, eigenvalues=eigenvalues, dates=plan['dates'],
                folder=folder, stocks=n, plan=plan)


@njit(parallel=True)
def _fill_distance_rows(features, distance, first, stop):
    m, p = features.shape
    for i in prange(first, stop):
        for j in range(i + 1, m):
            value = 0.0
            for f in range(p):
                value += abs(features[i, f] - features[j, f])
            distance[i, j] = value
            distance[j, i] = value


def compute_all_rolling_distances(rolling, max_threads=8, chunk_rows=8, progress=True):
    features = rolling['features']
    folder = rolling['folder']
    m = len(features)
    if (folder / 'distance_rows_done.npy').exists() and not (folder / 'matrix_distances.npy').exists():
        raise ValueError('Cached distance file was removed; use a new output directory')
    distance = _array(folder / 'matrix_distances.npy', (m, m), np.float64)
    done_path = folder / 'distance_rows_done.npy'
    was_new = not done_path.exists()
    done = _array(done_path, (m,), np.bool_)
    if was_new:
        done[:] = False
        np.fill_diagonal(distance, 0.0)
        distance.flush()
        done.flush()
    previous_threads = get_num_threads()
    set_num_threads(max(1, min(int(max_threads), previous_threads)))
    pending = np.flatnonzero(~np.asarray(done))
    remaining_pairs = int(np.sum(m - 1 - pending))
    try:
        # Compile on a tiny contiguous array before timing real work.
        toy = np.ascontiguousarray(features[:2, :4])
        _fill_distance_rows(toy, np.zeros((2, 2), dtype=np.float64), 0, 1)
        bar = tqdm(total=remaining_pairs, desc='All daily matrix pairs', disable=not progress)
        completed_pairs = 0
        started = perf_counter()
        estimated = False
        for offset in range(0, len(pending), chunk_rows):
            rows = pending[offset:offset + chunk_rows]
            # In the normal and resumed run these are contiguous. A previous
            # crash can leave gaps, so split only at gaps to avoid redoing rows.
            groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
            for group in groups:
                if len(group):
                    _fill_distance_rows(features, distance, int(group[0]), int(group[-1]) + 1)
            distance.flush()
            done[rows] = True
            done.flush()
            pairs = int(np.sum(m - 1 - rows))
            completed_pairs += pairs
            bar.update(pairs)
            elapsed = perf_counter() - started
            if not estimated and completed_pairs > 0 and elapsed > 0:
                seconds_left = (remaining_pairs - completed_pairs) * elapsed / completed_pairs
                if progress:
                    tqdm.write(f'Initial measured estimate: {seconds_left / 60:.1f} minutes remaining; this may change with disk/cache speed.')
                estimated = True
        bar.close()
    finally:
        set_num_threads(previous_threads)
    # Check all entries without allocating a second whole distance matrix.
    for start in range(0, m, 64):
        stop = min(start + 64, m)
        block = distance[start:stop]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise RuntimeError('Non-finite or negative distances')
        if not np.allclose(block, distance[:, start:stop].T, atol=1e-12, rtol=0):
            raise RuntimeError('Asymmetric distances')
    if not np.allclose(np.diag(distance), 0.0):
        raise RuntimeError('Nonzero diagonal')
    if progress:
        print('All full-resolution matrix distances computed and checked.', flush=True)
    return distance


def _date_ticks(ax, dates, every=2):
    dates = pd.DatetimeIndex(dates)
    years = range(dates[0].year, dates[-1].year + 1, every)
    positions = []
    texts = []
    for year in years:
        index = int(dates.searchsorted(pd.Timestamp(year=year, month=1, day=1)))
        if index < len(dates):
            positions.append(index)
            texts.append(str(year))
    ax.set_xticks(positions)
    ax.set_xticklabels(texts, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels(texts, fontsize=8)


def _save(fig, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), facecolor='white', bbox_inches='tight')


def plot_rolling_similarity(rolling, distance, output_path):
    dates = rolling['dates']
    selected = np.flatnonzero(dates >= pd.Timestamp('2007-01-01'))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)
    first = axes[0].imshow(distance, cmap='Blues', origin='upper', vmin=0,
                            interpolation='nearest', rasterized=True)
    _date_ticks(axes[0], dates, every=2)
    axes[0].set_title('1992-2010 | rolling two-month daily correlations', fontsize=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Date')
    fig.colorbar(first, ax=axes[0], label='Mean absolute matrix difference', shrink=0.83)
    zoom = distance[np.ix_(selected, selected)]
    second = axes[1].imshow(zoom, cmap='Blues', origin='upper', vmin=0,
                             interpolation='nearest', rasterized=True)
    _date_ticks(axes[1], dates[selected], every=1)
    axes[1].set_title('2007-2010 | daily-data zoom', fontsize=10)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Date')
    fig.colorbar(second, ax=axes[1], label='Mean absolute matrix difference', shrink=0.83)
    _save(fig, output_path)
    return fig


def plot_eigenvalue_analysis(rolling, output_path):
    dates = rolling['dates']
    eigenvalues = np.asarray(rolling['eigenvalues'])
    if not np.isfinite(eigenvalues).all():
        raise ValueError('Non-finite largest eigenvalues')
    alternate = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    axes[0].plot(dates, eigenvalues, color='#334b96', lw=0.8)
    axes[0].set_title('Largest eigenvalue of each daily correlation matrix', fontsize=10)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Largest eigenvalue')
    axes[0].grid(alpha=0.2)
    image = axes[1].imshow(alternate, cmap='Blues', origin='upper', vmin=0,
                           interpolation='nearest', rasterized=True)
    _date_ticks(axes[1], dates, every=2)
    axes[1].set_title('Alternative distance: absolute eigenvalue difference', fontsize=10)
    fig.colorbar(image, ax=axes[1], label='Absolute difference in largest eigenvalues', shrink=0.8)
    _save(fig, output_path)
    return fig

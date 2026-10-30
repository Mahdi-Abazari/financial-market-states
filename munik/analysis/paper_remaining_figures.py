"""Diagnostics and daily-data analogues of Figures 4, 5, and 6.

The histogram bin width and exclusion of self-correlations are explicit
implementation choices; the publication does not specify its histogram bins.
Dates follow the notebook's return-start labeling convention throughout.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm.auto import tqdm


def _save(fig, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor='white', bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), facecolor='white', bbox_inches='tight')


def silhouette_from_distances(distance_matrix, labels):
    """Standard silhouette with the existing matrix-distance metric."""
    d = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels)
    states = np.unique(labels)
    if d.shape != (len(labels), len(labels)) or len(states) < 2:
        raise ValueError('Need a square distance matrix and at least two groups')
    values = np.zeros(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        own = np.flatnonzero(labels == labels[i])
        own = own[own != i]
        if len(own) == 0:
            continue
        a = float(d[i, own].mean())
        b = min(float(d[i, labels == state].mean()) for state in states if state != labels[i])
        denominator = max(a, b)
        values[i] = (b - a) / denominator if denominator > 0 else 0.0
    return values


def state_diagnostics(correlations, labels, stock_info, tickers, window_info, distance_matrix=None):
    c = np.asarray(correlations, dtype=np.float64)
    labels = np.asarray(labels)
    sectors = stock_info.loc[list(tickers), 'sector'].to_numpy()
    row, col = np.triu_indices(c.shape[1], k=1)
    same = sectors[row] == sectors[col]
    energy_pair = (sectors[row] == 'E') ^ (sectors[col] == 'E')
    sil = silhouette_from_distances(distance_matrix, labels) if distance_matrix is not None else None
    rows = []
    for state in sorted(np.unique(labels)):
        member = labels == state
        center = c[member].mean(axis=0)
        if not np.allclose(center, center.T, atol=1e-12, rtol=0):
            raise ValueError('Asymmetric state center')
        if not np.allclose(np.diag(center), 1.0, atol=1e-12, rtol=0):
            raise ValueError('Invalid state diagonal')
        values = center[row, col]
        member_info = window_info.loc[member]
        rows.append(dict(
            state=int(state), periods=int(member.sum()),
            first_period=str(pd.to_datetime(member_info.period_start).min().date()),
            last_period=str(pd.to_datetime(member_info.period_end).max().date()),
            mean_correlation=float(values.mean()),
            within_sector_mean=float(values[same].mean()),
            between_sector_mean=float(values[~same].mean()),
            energy_to_other_sectors=float(values[energy_pair].mean()),
            negative_pair_fraction=float((values < 0).mean()),
            mean_silhouette=float(sil[member].mean()) if sil is not None else np.nan,
        ))
    return pd.DataFrame(rows)


def plot_timeline(window_info, labels, threshold, output_path):
    labels = np.asarray(labels)
    if len(labels) != len(window_info):
        raise ValueError('Date/label count mismatch')
    starts = pd.to_datetime(window_info.period_start)
    stops = pd.to_datetime(window_info.period_end) + pd.Timedelta(days=1)
    fig, ax = plt.subplots(figsize=(14, 3.3), constrained_layout=True)
    ax.hlines(labels, mdates.date2num(starts), mdates.date2num(stops),
              color='black', linewidth=2.3)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.set_xlim(starts.min(), stops.max())
    ax.set_ylim(labels.min() - 0.6, labels.max() + 0.6)
    ax.set_yticks(sorted(np.unique(labels)))
    ax.set_ylabel('Market state')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(f'Temporal evolution of calibrated daily-data states | threshold={threshold:.7f}', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    _save(fig, output_path)
    return fig


def rolling_correlation_histograms(normalized_returns, start='2007-09-01', end='2009-03-31', bins=100, progress=True):
    """Daily trailing two-calendar-month correlations; store histograms, not all matrices."""
    z = normalized_returns.sort_index()
    if not z.index.is_unique or not np.isfinite(z.to_numpy()).all():
        raise ValueError('Need unique dates and finite normalized returns')
    dates = z.index[(z.index >= pd.Timestamp(start)) & (z.index <= pd.Timestamp(end))]
    if len(dates) == 0:
        raise ValueError('No observations in requested date range')
    edges = np.linspace(-1.0, 1.0, bins + 1)
    row, col = np.triu_indices(z.shape[1], k=1)
    probabilities = np.zeros((len(dates), bins), dtype=np.float64)
    records = []
    for i in tqdm(range(len(dates)), desc='Daily crisis histograms', disable=not progress):
        date = dates[i]
        lower = date - pd.DateOffset(months=2)
        if z.index[0] > lower:
            raise ValueError(f'Incomplete two-month history at {date}')
        block = z.loc[(z.index > lower) & (z.index <= date)]
        if len(block) < 3 or (block.std(ddof=0) == 0).any():
            raise ValueError(f'Invalid correlation window at {date}')
        corr = np.corrcoef(block.to_numpy(dtype=np.float64), rowvar=False)
        values = corr[row, col]
        if not np.isfinite(values).all():
            raise ValueError(f'Invalid correlation at {date}')
        # Clip only floating-point roundoff at the mathematical endpoints.
        counts, _ = np.histogram(np.clip(values, -1.0, 1.0), bins=edges)
        probabilities[i] = counts / len(values)
        records.append(dict(date=date, n_returns=len(block),
                            first_return_date=block.index[0],
                            last_return_date=block.index[-1],
                            mean_correlation=float(values.mean()),
                            std_correlation=float(values.std(ddof=0)),
                            negative_fraction=float((values < 0).mean()),
                            unique_stock_pairs=len(values)))
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError('Histogram probabilities do not sum to one')
    return dates, edges, probabilities, pd.DataFrame(records)


def plot_crisis_histograms(dates, edges, probabilities, summary, output_path):
    dates = pd.DatetimeIndex(dates)
    midpoints = (edges[:-1] + edges[1:]) / 2
    chosen = []
    for text in ['2008-09-30', '2008-12-31']:
        candidates = np.flatnonzero(dates <= pd.Timestamp(text))
        if len(candidates) == 0:
            raise ValueError(f'No observations for {text}')
        i = int(candidates[-1])
        if dates[i].to_period('M') != pd.Timestamp(text).to_period('M'):
            raise ValueError(f'Missing requested month {text}')
        chosen.append(i)
    fig = plt.figure(figsize=(13, 5.2), facecolor='white')
    ax = fig.add_subplot(121)
    for i, color, style in zip(chosen, ['black', '#cf3333'], ['-', '--']):
        ax.plot(midpoints, probabilities[i], color=color, linestyle=style,
                label=str(dates[i].date()), lw=1.5)
    ax.set_xlabel('Correlation coefficient')
    ax.set_ylabel(f'Probability per bin (width={edges[1] - edges[0]:.2f})')
    ax.set_title('September and December 2008\ntrailing two-month daily-data correlations', fontsize=10)
    ax.set_xlim(-0.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    surface = fig.add_subplot(122, projection='3d')
    x, y = np.meshgrid(midpoints, mdates.date2num(dates))
    pair_count = int(summary.unique_stock_pairs.iloc[0])
    # This floor is for log-axis display only; original zero probabilities stay zero.
    height = np.log10(np.maximum(probabilities, 1.0 / pair_count))
    surface.plot_surface(x, y, height, cmap='Blues', linewidth=0,
                         antialiased=True, rcount=min(180, len(dates)), ccount=len(midpoints))
    date_ticks = pd.to_datetime(['2007-09-01', '2008-03-01', '2008-09-01', '2009-03-01'])
    surface.set_yticks(mdates.date2num(date_ticks))
    surface.set_yticklabels(date_ticks.strftime('%Y-%m'), fontsize=7)
    surface.set_xlabel('Correlation', fontsize=8)
    # Place this label in figure coordinates to avoid 3-D clipping on export.
    surface.set_zlabel('')
    fig.text(0.975, 0.52, 'log10(probability per bin)', rotation=90,
             va='center', ha='center', fontsize=9)
    surface.set_title('Evolution during the financial crisis', fontsize=10)
    surface.tick_params(axis='x', labelsize=7)
    surface.tick_params(axis='z', labelsize=7)
    surface.view_init(elev=28, azim=-55)
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.16, top=0.86, wspace=0.2)
    _save(fig, output_path)
    return fig, summary.iloc[chosen].copy()


def plot_complete_tree(correlations, labels, tree_nodes, threshold, window_info, output_path):
    """All 114 period leaves; annotate the calibrated cut on the fitted tree."""
    nodes = {int(k): v for k, v in tree_nodes.items()}
    c = np.asarray(correlations)
    labels = np.asarray(labels)
    group_to_state = {tuple(np.flatnonzero(labels == state)): int(state) for state in np.unique(labels)}
    centers = {}
    selected = {}

    def collect_cut(node_id):
        node = nodes[node_id]
        if not node['children'] or node['radius'] < threshold:
            key = tuple(sorted(node['members']))
            if key not in group_to_state:
                raise ValueError('State labels do not match the threshold cut')
            selected[node_id] = group_to_state[key]
        else:
            for child in node['children']:
                collect_cut(child)

    collect_cut(0)
    def center(node_id):
        if node_id not in centers:
            centers[node_id] = c[nodes[node_id]['members']].mean(axis=0)
        return centers[node_id]

    leaves = []
    def collect_leaves(node_id):
        if not nodes[node_id]['children']:
            leaves.append(node_id)
        else:
            for child in nodes[node_id]['children']:
                collect_leaves(child)
    collect_leaves(0)
    if len(leaves) != len(c):
        raise ValueError('The tree is incomplete')
    leaf_y = {node: i for i, node in enumerate(leaves)}
    positions = {}
    fig, ax = plt.subplots(figsize=(10, max(10, len(leaves) * 0.135)))

    def draw(node_id, x):
        node = nodes[node_id]
        if not node['children']:
            y = float(leaf_y[node_id])
            index = node['members'][0]
            date = pd.Timestamp(window_info.iloc[index].period_start)
            term = (date.month + 1) // 2
            text = f'{date.year % 100:02d}-{term}'
            ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=5.5)
            extent = x
        else:
            ys = []
            extent = x
            for child in node['children']:
                edge = float(np.mean(np.abs(center(child) - center(node_id))))
                child_y, child_extent = draw(child, x + edge)
                ax.plot([x, x + edge], [child_y, child_y], color='#222222', lw=0.65)
                ys.append(child_y)
                extent = max(extent, child_extent)
            y = float(np.mean(ys))
            ax.plot([x, x], [min(ys), max(ys)], color='#222222', lw=0.65)
        positions[node_id] = (x, y)
        return y, extent

    _, extent = draw(0, 0.0)
    for node_id, state in selected.items():
        x, y = positions[node_id]
        ax.text(x, y, str(state), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='square,pad=0.12', facecolor='white', edgecolor='black', lw=0.7))
    ax.set_ylim(len(leaves) - 0.5, -0.5)
    ax.set_xlim(-0.02, extent + 0.08)
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set_xlabel('Cumulative center-to-center distance')
    ax.set_title(f'Complete fitted tree: {len(leaves)} periods\nboxed labels mark the calibrated eight-state cut', fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.955, bottom=0.035)
    _save(fig, output_path)
    return fig

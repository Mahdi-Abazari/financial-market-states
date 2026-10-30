"""Validate optimized kernels, then run the notebook's exhaustive analysis."""

from itertools import combinations
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
from numba import njit

from market_states_numba import (
    matrix_features, pairwise_distances, binary_step, exhaustive_split,
    centroid_distances, top_down,
)


@njit
def local_normalize(values, window):
    n_days, n_stocks = values.shape
    result = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    for j in range(n_stocks):
        for t in range(window - 1, n_days):
            mean = 0.0
            for q in range(t - window + 1, t + 1):
                mean += values[q, j]
            mean /= window
            variance = 0.0
            for q in range(t - window + 1, t + 1):
                variance += (values[q, j] - mean) ** 2
            variance /= window
            if variance > 0.0:
                result[t, j] = (values[t, j] - mean) / np.sqrt(variance)
    return result


def reference_step(x, weights, labels):
    if np.unique(labels).size < 2:
        raise ValueError('Empty group')
    centers = np.stack([x[labels == group].mean(axis=0) for group in [0, 1]])
    distances = np.sum(np.abs(x[:, None, :] - centers[None, :, :]) * weights, axis=2)
    new = labels.copy()
    new[distances[:, 1] < distances[:, 0]] = 1
    new[distances[:, 0] < distances[:, 1]] = 0
    if new[0]:
        new = 1 - new
    score = np.mean(distances[np.arange(len(x)), labels] ** 2)
    return new, score


def kernel_tests():
    rng = np.random.default_rng(2183)
    matrices = np.stack([np.corrcoef(rng.normal(size=(20, 4)), rowvar=False) for _ in range(8)])
    x, weights = matrix_features(matrices)
    d = pairwise_distances(x, weights, progress=False)
    for i, j in combinations(range(len(x)), 2):
        assert np.isclose(d[i, j], np.mean(np.abs(matrices[i] - matrices[j])), atol=1e-14, rtol=1e-12)
    center = matrices.mean(axis=0)
    assert np.allclose(centroid_distances(x, weights), np.mean(np.abs(matrices - center), axis=(1, 2)))
    best_score = np.inf
    for a, b in combinations(range(len(x)), 2):
        labels = (d[:, b] < d[:, a]).astype(np.uint8)
        if labels[0]:
            labels = 1 - labels
        for _ in range(100):
            new, score = reference_step(x, weights, labels)
            jit_new, jit_score, valid = binary_step(x, weights, labels)
            assert valid and np.array_equal(new, jit_new) and np.isclose(score, jit_score)
            if np.array_equal(new, labels):
                break
            labels = new
        else:
            raise RuntimeError('Test did not converge')
        best_score = min(best_score, score)
    labels, report = exhaustive_split(x, weights, d, progress=False)
    assert np.isclose(best_score, report['best_squared_distance'], rtol=1e-12)
    print('PASS: full-matrix distances, centroid distances, NumPy vs njit, exhaustive cached vs uncached', flush=True)


def main():
    kernel_tests()
    root = Path(__file__).resolve().parent
    output = root / 'notebook_numba_verified'
    output.mkdir(exist_ok=True)
    price_all = pd.read_csv(root / 'output/clean/adjusted_close_wide.csv', index_col='Date', parse_dates=['Date']).sort_index()
    prices = price_all.loc[:, price_all.notna().all(axis=0)]
    p = prices.to_numpy(dtype=np.float64)
    r = (p[1:] - p[:-1]) / p[:-1]
    normalized = local_normalize(np.ascontiguousarray(r), 13)[12:]
    normalized_all = pd.DataFrame(normalized, index=prices.index[:-1][12:], columns=prices.columns)
    excluded = normalized_all.columns[normalized_all.isna().any()].tolist()
    z = normalized_all.loc[:, normalized_all.notna().all(axis=0)]
    matrices = []
    periods = []
    for year in range(1992, 2011):
        for month in [1, 3, 5, 7, 9, 11]:
            start = pd.Timestamp(year=year, month=month, day=1)
            stop = start + pd.DateOffset(months=2)
            block = z.loc[(z.index >= start) & (z.index < stop)]
            matrices.append(np.corrcoef(block.to_numpy(), rowvar=False))
            periods.append(f'{year}-T{(month + 1) // 2}')
    c = np.stack(matrices)
    x, weights = matrix_features(c)
    d = pairwise_distances(x, weights)
    print(f'DATA {c.shape}; excluded {excluded}; root radius {centroid_distances(x, weights).mean():.12f}', flush=True)
    start_time = perf_counter()
    labels, tree = top_down(x, weights, d, threshold=0.1465,
                          checkpoint=lambda df: df.to_csv(output / 'tree_progress.csv', index=False))
    seconds = perf_counter() - start_time
    assignments = pd.DataFrame(dict(period=periods, state=labels))
    assignments.to_csv(output / 'assignments.csv', index=False)
    tree.to_csv(output / 'tree.csv', index=False)
    report = dict(windows=len(c), stocks=c.shape[1], states=int(len(np.unique(labels))),
                  threshold=0.1465, seconds=seconds, excluded_normalization=excluded,
                  interpretation='Eq2 mean absolute matrix distance for assignment; arithmetic centers; mean squared distance for split selection',
                  tested='NumPy vs njit and exhaustive cached vs uncached on synthetic matrices')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

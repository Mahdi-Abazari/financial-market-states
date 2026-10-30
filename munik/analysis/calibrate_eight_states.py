"""Build the complete daily-data tree and find any eight-state threshold interval."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd

from verify_market_states_numba import local_normalize
from market_states_numba import (
    matrix_features, pairwise_distances, complete_tree, cut_at_threshold,
    cut_to_k, threshold_sensitivity, centroid_distances,
)


def main():
    root = Path(__file__).resolve().parent
    out = root / 'notebook_numba_verified'
    all_prices = pd.read_csv(root / 'output/clean/adjusted_close_wide.csv', parse_dates=['Date'], index_col='Date').sort_index()
    prices = all_prices.loc[:, all_prices.notna().all(axis=0)]
    p = prices.to_numpy(dtype=np.float64)
    returns = (p[1:] - p[:-1]) / p[:-1]
    z_all = pd.DataFrame(local_normalize(np.ascontiguousarray(returns), 13)[12:],
                         index=prices.index[:-1][12:], columns=prices.columns)
    z = z_all.loc[:, z_all.notna().all(axis=0)]
    matrices = []
    metadata = []
    for year in range(1992, 2011):
        for month in [1, 3, 5, 7, 9, 11]:
            start = pd.Timestamp(year=year, month=month, day=1)
            stop = start + pd.DateOffset(months=2)
            block = z.loc[(z.index >= start) & (z.index < stop)]
            matrices.append(np.corrcoef(block.to_numpy(), rowvar=False))
            metadata.append(dict(period=f'{year}-T{(month + 1)//2}',
                                 period_start=str(start.date()),
                                 period_end=str((stop-pd.Timedelta(days=1)).date()),
                                 n_returns=len(block)))
    c = np.stack(matrices)
    x, weights = matrix_features(c)
    d = pairwise_distances(x, weights)
    assignments_old = pd.read_csv(out / 'assignments.csv')
    old_tree = pd.read_csv(out / 'tree.csv')
    old_report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    if old_report['windows'] != len(c) or old_report['stocks'] != c.shape[1]:
        raise ValueError('Cached root has a different universe')
    root_labels = (assignments_old.state.to_numpy() - 1).astype(np.uint8)
    if set(root_labels) != {0, 1}:
        raise ValueError('Expected the previously verified two-state root split')
    for node_id, state in [(1, 0), (2, 1)]:
        expected = float(old_tree.loc[old_tree.node.eq(node_id), 'radius'].iloc[0])
        actual = float(centroid_distances(np.ascontiguousarray(x[root_labels == state]), weights).mean())
        if not np.isclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError('Cached root does not match recomputed data')
    root_info = {'initial_pairs': int(old_tree.loc[old_tree.node.eq(0), 'initial_pairs'].iloc[0]),
                 'reused_verified_root': True}
    def save_tree(nodes):
        (out / 'complete_tree_progress.json').write_text(json.dumps(nodes, indent=2), encoding='utf-8')
    start_time = perf_counter()
    nodes = complete_tree(x, weights, d, root_split=(root_labels, root_info), checkpoint=save_tree)
    (out / 'complete_tree.json').write_text(json.dumps(nodes, indent=2), encoding='utf-8')
    sensitivity = threshold_sensitivity(nodes)
    sensitivity.to_csv(out / 'threshold_sensitivity.csv', index=False)
    intervals = sensitivity.loc[sensitivity.states.eq(8)].copy()
    eight = cut_to_k(nodes, 8)
    fixed_threshold = cut_at_threshold(nodes, 0.1465)
    output_assignments = pd.DataFrame(metadata)
    output_assignments['paper_threshold_state'] = fixed_threshold
    output_assignments['fixed_eight_state'] = eight
    if len(intervals):
        # Select the interval closest to the paper's threshold, without claiming
        # this value came from the paper or was independently validated.
        row = intervals.iloc[np.argmin(np.abs(intervals.threshold.to_numpy() - 0.1465))]
        calibrated_threshold = float(row.threshold)
        calibrated = cut_at_threshold(nodes, calibrated_threshold)
        output_assignments['calibrated_eight_state'] = calibrated
    else:
        calibrated_threshold = None
    output_assignments.to_csv(out / 'all_state_assignments.csv', index=False)
    np.savez_compressed(out / 'daily_analysis_arrays.npz', C=c, D=d, tickers=np.array(z.columns.tolist()))
    report = dict(full_tree_nodes=len(nodes), seconds=perf_counter()-start_time,
                  paper_threshold=0.1465, paper_threshold_states=int(np.unique(fixed_threshold).size),
                  fixed_k_states=int(np.unique(eight).size),
                  eight_state_intervals=intervals.to_dict(orient='records'),
                  calibrated_threshold=calibrated_threshold,
                  note='Eight-state threshold was fitted on this dataset, not reproduced at the published threshold.')
    (out / 'eight_state_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

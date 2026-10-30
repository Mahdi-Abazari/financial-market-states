"""Auditable exhaustive binary clustering for the daily-data notebook.

Distance: mean absolute entrywise difference of correlation matrices.
Centers: arithmetic means. Split selection: mean squared matrix distance.
Stopping: mean (not squared) matrix distance to the center < threshold.

Every initial pair is considered. Repeated identical partitions share their
deterministic continuation, so caching changes computation time, not the set
of initializations. Cycles, empty groups, and iteration limits raise errors.
The publication's xi^(L) notation is not defined explicitly in Methods; using
the entrywise distance from Eq. (2) for clustering is an explicit interpretation.
"""

from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit
from tqdm.auto import tqdm


def matrix_features(correlations):
    c = np.asarray(correlations, dtype=np.float64)
    if c.ndim != 3 or c.shape[1] != c.shape[2]:
        raise ValueError("Expected an array of square correlation matrices")
    if not np.isfinite(c).all():
        raise ValueError("Non-finite correlations")
    if not np.allclose(c, c.transpose(0, 2, 1), rtol=0, atol=1e-12):
        raise ValueError("Matrices must be symmetric")
    n = c.shape[1]
    row, col = np.triu_indices(n)
    x = np.ascontiguousarray(c[:, row, col])
    weights = np.where(row == col, 1.0, 2.0) / (n * n)
    return x, np.ascontiguousarray(weights)


@njit
def distances_from_row(x, weights, row):
    m, p = x.shape
    result = np.zeros(m, dtype=np.float64)
    for j in range(row + 1, m):
        value = 0.0
        for f in range(p):
            value += weights[f] * abs(x[row, f] - x[j, f])
        result[j] = value
    return result


def pairwise_distances(x, weights, progress=True):
    m = len(x)
    d = np.zeros((m, m), dtype=np.float64)
    for i in tqdm(range(m), desc="Matrix distances", disable=not progress):
        row = distances_from_row(x, weights, i)
        d[i, i + 1:] = row[i + 1:]
        d[i + 1:, i] = row[i + 1:]
    return d


@njit
def centroid_distances(x, weights):
    m, p = x.shape
    center = np.zeros(p, dtype=np.float64)
    for i in range(m):
        for f in range(p):
            center[f] += x[i, f]
    for f in range(p):
        center[f] /= m
    d = np.zeros(m, dtype=np.float64)
    for i in range(m):
        for f in range(p):
            d[i] += weights[f] * abs(x[i, f] - center[f])
    return d


@njit
def binary_step(x, weights, labels):
    """One deterministic assignment step; ties retain previous membership."""
    m, p = x.shape
    counts = np.zeros(2, dtype=np.int64)
    centers = np.zeros((2, p), dtype=np.float64)
    for i in range(m):
        group = labels[i]
        counts[group] += 1
        for f in range(p):
            centers[group, f] += x[i, f]
    if counts[0] == 0 or counts[1] == 0:
        return labels.copy(), np.inf, False
    for group in range(2):
        for f in range(p):
            centers[group, f] /= counts[group]
    new_labels = labels.copy()
    objective = 0.0
    for i in range(m):
        da = 0.0
        db = 0.0
        for f in range(p):
            da += weights[f] * abs(x[i, f] - centers[0, f])
            db += weights[f] * abs(x[i, f] - centers[1, f])
        own_distance = da if labels[i] == 0 else db
        objective += own_distance * own_distance
        if db < da:
            new_labels[i] = 1
        elif da < db:
            new_labels[i] = 0
    if new_labels[0] != 0:
        for i in range(m):
            new_labels[i] = 1 - new_labels[i]
    return new_labels, objective / m, True


def exhaustive_split(x, weights, distance_matrix, progress=True, max_iter=10000):
    """Evaluate all pairs with exact memoization of partition trajectories."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    m = len(x)
    if m < 2:
        raise ValueError("Need at least two matrices")
    memo = {}
    best = None
    transitions = 0
    started = perf_counter()
    total = m * (m - 1) // 2
    iterator = tqdm(combinations(range(m), 2), total=total,
                    desc=f"All initial pairs (m={m})", disable=not progress)
    for pair_id, (a, b) in enumerate(iterator, 1):
        labels = np.asarray(distance_matrix[:, b] < distance_matrix[:, a], dtype=np.uint8)
        if labels[0]:
            labels = 1 - labels
        path = []
        visiting = set()
        for _ in range(max_iter):
            key = labels.tobytes()
            if key in memo:
                solution = memo[key]
                break
            if key in visiting:
                raise RuntimeError(f"Assignment cycle for initial pair {(a, b)}; no final result accepted")
            visiting.add(key)
            path.append(key)
            new_labels, score, valid = binary_step(x, weights, labels)
            transitions += 1
            if not valid:
                raise RuntimeError(f"Empty group for initial pair {(a, b)}")
            if np.array_equal(new_labels, labels):
                solution = (labels.copy(), float(score))
                break
            labels = new_labels
        else:
            raise RuntimeError(f"Iteration limit for initial pair {(a, b)}; no final result accepted")
        for key in path:
            memo[key] = solution
        if best is None or solution[1] < best[1]:
            best = (solution[0].copy(), solution[1], (a, b))
        if pair_id % 100 == 0:
            iterator.set_postfix(unique_steps=transitions)
    info = dict(initial_pairs=total, unique_steps=transitions,
                best_squared_distance=best[1], initial_pair=best[2],
                seconds=perf_counter() - started)
    return best[0], info


def top_down(x, weights, distance_matrix, threshold=0.1465, progress=True, checkpoint=None):
    """Exhaustive recursive splits; leaves stop on mean matrix distance."""
    pending = [(0, np.arange(len(x), dtype=np.int64))]
    leaves = []
    records = []
    next_node = 1
    while pending:
        node_id, members = pending.pop(0)
        node_x = np.ascontiguousarray(x[members])
        radius = float(centroid_distances(node_x, weights).mean())
        record = dict(node=node_id, size=len(members), radius=radius,
                      stop_threshold=float(threshold))
        if progress:
            print(f"Node {node_id}: {len(members)} windows; mean distance={radius:.6f}", flush=True)
        if radius < threshold or len(members) == 1:
            leaves.append(members)
            record['split'] = False
        else:
            labels, info = exhaustive_split(node_x, weights,
                distance_matrix[np.ix_(members, members)], progress=progress)
            record.update(info)
            record['split'] = True
            left = members[labels == 0]
            right = members[labels == 1]
            record['left'] = next_node
            record['right'] = next_node + 1
            pending.extend([(next_node, left), (next_node + 1, right)])
            next_node += 2
        records.append(record)
        if checkpoint is not None:
            checkpoint(pd.DataFrame(records))
    # Label ordering is chronological, not the labels of the publication.
    leaves.sort(key=lambda indices: int(indices.min()))
    labels = np.zeros(len(x), dtype=np.int64)
    for state, members in enumerate(leaves, 1):
        labels[members] = state
    return labels, pd.DataFrame(records)


def complete_tree(x, weights, distance_matrix, progress=True, root_split=None, checkpoint=None):
    """Build the full binary tree, independent of a stopping threshold.

    root_split can reuse a previously verified exhaustive split of the same X.
    It is a (binary_labels, diagnostic_dictionary) tuple, not a new clustering.
    """
    pending = [(0, np.arange(len(x), dtype=np.int64))]
    nodes = {}
    next_node = 1
    while pending:
        node_id, members = pending.pop(0)
        node_x = np.ascontiguousarray(x[members])
        radius = float(centroid_distances(node_x, weights).mean())
        record = dict(node=node_id, members=members.tolist(), size=len(members), radius=radius)
        if progress:
            print(f"Full tree node {node_id}: {len(members)} windows; radius={radius:.6f}", flush=True)
        if len(members) > 1:
            if node_id == 0 and root_split is not None:
                labels, info = root_split
                labels = np.asarray(labels, dtype=np.uint8)
                if labels.shape != (len(x),) or set(labels.tolist()) != {0, 1}:
                    raise ValueError('Invalid cached root split')
            else:
                labels, info = exhaustive_split(node_x, weights,
                    distance_matrix[np.ix_(members, members)], progress=progress)
            record.update(info)
            left = members[labels == 0]
            right = members[labels == 1]
            record['children'] = [next_node, next_node + 1]
            pending.extend([(next_node, left), (next_node + 1, right)])
            next_node += 2
        else:
            record['children'] = []
        nodes[node_id] = record
        if checkpoint is not None:
            checkpoint(nodes)
    return nodes


def _labels_from_nodes(nodes, selected):
    size = nodes[0]['size']
    selected = sorted(selected, key=lambda node: min(nodes[node]['members']))
    labels = np.zeros(size, dtype=np.int64)
    for state, node_id in enumerate(selected, 1):
        labels[np.asarray(nodes[node_id]['members'], dtype=np.int64)] = state
    if np.any(labels == 0):
        raise RuntimeError('Tree cut did not cover all windows')
    return labels


def cut_at_threshold(nodes, threshold):
    pending = [0]
    selected = []
    while pending:
        node_id = pending.pop()
        node = nodes[node_id]
        if not node['children'] or node['radius'] < threshold:
            selected.append(node_id)
        else:
            pending.extend(node['children'])
    return _labels_from_nodes(nodes, selected)


def threshold_sensitivity(nodes):
    """One representative of each open interval between node radii."""
    radii = sorted(set(float(node['radius']) for node in nodes.values()))
    bounds = [0.0] + [r for r in radii if r > 0.0]
    bounds.append(max(bounds[-1] * 1.1, bounds[-1] + 0.01))
    rows = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        value = (low + high) / 2
        labels = cut_at_threshold(nodes, value)
        rows.append(dict(lower_boundary=low, upper_boundary=high,
                         threshold=value, states=int(np.unique(labels).size)))
    return pd.DataFrame(rows)


def cut_to_k(nodes, k):
    """Explicit fixed-K tree cut: split the active node with largest radius."""
    if not 1 <= k <= nodes[0]['size']:
        raise ValueError('K outside possible range')
    active = [0]
    while len(active) < k:
        splittable = [node for node in active if nodes[node]['children']]
        node = max(splittable, key=lambda n: (nodes[n]['radius'], -n))
        active.remove(node)
        active.extend(nodes[node]['children'])
    return _labels_from_nodes(nodes, active)

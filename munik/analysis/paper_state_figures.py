"""Sector-ordered state matrices and the actual collapsed binary tree.

Figures use the calibrated daily-data states; matching the number of states
does not establish equivalence to the original paper's state memberships.
Overall average is the equally weighted mean of the observed window matrices,
not the equally weighted mean of the eight state centers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


SECTOR_ORDER = ['E', 'M', 'I', 'CD', 'CS', 'H', 'F', 'IT', 'C', 'U']
PAPER_CMAP = LinearSegmentedColormap.from_list(
    'paper_correlation', ['#ef3333', '#ffffff', '#40549d'], N=256)


def make_sector_layout(tickers, stock_info):
    """Return one sector permutation to apply to every displayed matrix."""
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers) or not stock_info.index.is_unique:
        raise ValueError('Ticker identifiers must be unique')
    info = stock_info.loc[tickers, ['sector']].copy()
    if info['sector'].isna().any() or not info['sector'].isin(SECTOR_ORDER).all():
        raise ValueError('Missing or unknown sector identifiers')
    original_position = {ticker: i for i, ticker in enumerate(tickers)}
    ordered = []
    centers = []
    boundaries = [0]
    present = []
    for sector in SECTOR_ORDER:
        group = sorted(info.index[info['sector'].eq(sector)].tolist())
        if group:
            start = len(ordered)
            ordered.extend(group)
            centers.append(start + (len(group) - 1) / 2)
            boundaries.append(len(ordered))
            present.append(sector)
    permutation = np.array([original_position[ticker] for ticker in ordered], dtype=np.int64)
    return dict(tickers=ordered, permutation=permutation, centers=np.array(centers),
                boundaries=np.array(boundaries), sectors=present,
                counts=info.sector.value_counts().reindex(SECTOR_ORDER, fill_value=0))


def compute_state_matrices(correlations, labels):
    c = np.asarray(correlations, dtype=np.float64)
    labels = np.asarray(labels)
    if len(c) != len(labels) or not np.isfinite(c).all():
        raise ValueError('Invalid correlations or labels')
    states = sorted(np.unique(labels).tolist())
    centers = {int(state): c[labels == state].mean(axis=0) for state in states}
    overall = c.mean(axis=0)
    counts = {int(state): int(np.count_nonzero(labels == state)) for state in states}
    # Verify that the pooled average is the size-weighted average of centers.
    weighted = sum(centers[state] * counts[state] for state in centers) / len(c)
    if not np.allclose(weighted, overall, atol=1e-12, rtol=0):
        raise RuntimeError('Incorrect overall-average weighting')
    return centers, overall, counts


def _matrix_panel(ax, matrix, layout, title, fontsize=7):
    p = layout['permutation']
    image = ax.imshow(matrix[np.ix_(p, p)], cmap=PAPER_CMAP,
                      vmin=-1, vmax=1, origin='upper', interpolation='nearest')
    ax.set_xticks(layout['centers'])
    ax.set_yticks(layout['centers'])
    ax.set_xticklabels(layout['sectors'], fontsize=fontsize)
    ax.set_yticklabels(layout['sectors'], fontsize=fontsize)
    ax.tick_params(axis='both', length=2, pad=2)
    for boundary in layout['boundaries'][1:-1]:
        ax.axhline(boundary - 0.5, color='#333333', lw=0.25, alpha=0.35)
        ax.axvline(boundary - 0.5, color='#333333', lw=0.25, alpha=0.35)
    ax.set_title(title, fontsize=10, pad=7)
    return image


def plot_collapsed_tree(ax, correlations, labels, nodes, threshold):
    """Collapse the fitted tree; do not recluster the state centers.

    Every edge length is the Eq. (2) distance of child and parent centers.
    State leaves can therefore end at different horizontal positions.
    """
    nodes = {int(key): value for key, value in nodes.items()}
    c = np.asarray(correlations)
    labels = np.asarray(labels)
    group_to_state = {
        tuple(np.flatnonzero(labels == state).tolist()): int(state)
        for state in np.unique(labels)
    }
    node_centers = {}
    positions = {}
    selected = []

    def center(node_id):
        if node_id not in node_centers:
            node_centers[node_id] = c[nodes[node_id]['members']].mean(axis=0)
        return node_centers[node_id]

    def collect(node_id):
        node = nodes[node_id]
        if not node['children'] or node['radius'] < threshold:
            key = tuple(sorted(node['members']))
            if key not in group_to_state:
                raise ValueError('Tree cut does not match supplied state labels')
            selected.append((node_id, group_to_state[key]))
        else:
            for child in node['children']:
                collect(child)

    collect(0)
    for y, (node_id, state) in enumerate(selected):
        positions[node_id] = float(y)
    leaf_state = dict(selected)

    def draw(node_id, x):
        node = nodes[node_id]
        if node_id in leaf_state:
            y = positions[node_id]
            ax.annotate(f"state {leaf_state[node_id]}", (x, y), xytext=(5, 0),
                        textcoords='offset points', va='center', fontsize=8)
            return y, x
        children = node['children']
        child_positions = []
        farthest = x
        for child in children:
            edge = float(np.mean(np.abs(center(child) - center(node_id))))
            child_x = x + edge
            child_y, extent = draw(child, child_x)
            ax.plot([x, child_x], [child_y, child_y], color='#222222', lw=1.2)
            child_positions.append(child_y)
            farthest = max(farthest, extent)
        ax.plot([x, x], [min(child_positions), max(child_positions)], color='#222222', lw=1.2)
        return float(np.mean(child_positions)), farthest

    _, extent = draw(0, 0.0)
    ax.set_ylim(len(selected) - 0.5, -0.5)
    ax.set_xlim(-max(extent, 0.01) * 0.03, extent * 1.4 + 0.01)
    ax.set_yticks([])
    ax.set_xlabel('Cumulative center-to-center distance', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    for name in ['left', 'right', 'top']:
        ax.spines[name].set_visible(False)
    ax.set_title('(i) fitted clustering tree', fontsize=10, pad=9)


def plot_figure2(correlations, labels, tree_nodes, threshold, layout, output_path):
    centers, overall, counts = compute_state_matrices(correlations, labels)
    if len(centers) != 8:
        raise ValueError('The eight-panel comparison requires eight fitted states')
    fig = plt.figure(figsize=(13.2, 11.8), facecolor='white')
    grid = fig.add_gridspec(3, 4, left=0.055, right=0.93, bottom=0.06,
                           top=0.94, hspace=0.31, wspace=0.23)
    axes = []
    for i, state in enumerate(sorted(centers)):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        _matrix_panel(ax, centers[state], layout,
                      f'({chr(97+i)}) state {state} | {counts[state]} periods')
        axes.append(ax)
    tree_ax = fig.add_subplot(grid[2, :2])
    plot_collapsed_tree(tree_ax, correlations, labels, tree_nodes, threshold)
    overall_ax = fig.add_subplot(grid[2, 2:])
    _matrix_panel(overall_ax, overall, layout, '(j) overall average correlation', fontsize=8)
    cax = fig.add_axes([0.945, 0.11, 0.014, 0.76])
    fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=PAPER_CMAP), cax=cax,
                 label='Correlation; common scale for all matrices')
    fig.suptitle(f'Eight calibrated daily-data states | threshold = {threshold:.7f}', fontsize=13, y=0.98)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), facecolor='white')
    return fig


def plot_figure3(centers, overall, layout, output_path):
    if len(centers) != 8:
        raise ValueError('Expected eight state centers')
    fig = plt.figure(figsize=(13.2, 7.1), facecolor='white')
    grid = fig.add_gridspec(2, 4, left=0.055, right=0.93, bottom=0.07,
                           top=0.89, hspace=0.25, wspace=0.23)
    for i, state in enumerate(sorted(centers)):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        _matrix_panel(ax, centers[state] - overall, layout,
                      f'({chr(97+i)}) state {state} minus overall')
    cax = fig.add_axes([0.945, 0.13, 0.014, 0.66])
    fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=PAPER_CMAP), cax=cax,
                 label='Correlation difference; same scale as Figure 2')
    fig.suptitle('State centers minus the overall average | calibrated daily-data states',
                 fontsize=12, y=0.96)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), facecolor='white')
    return fig

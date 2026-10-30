"""Reproduce the main data pipeline from Muennix et al. (2012).

The script deliberately keeps the original files untouched.  It creates:

* cleaned per-stock OHLCV files and adjusted-close/return matrices;
* a quality report describing every removed or retained data issue;
* rolling correlation matrices on disjoint two-month windows;
* the paper's similarity matrix and an adaptive, top-down clustering;
* comparison plots and CSV summaries for a seminar project.

The default universe is the complete-history universe, which is closest to
the paper's use of stocks that remained in the S&P 500 for the observation
period.  Use ``--universe all`` to retain incomplete-history stocks in the
cleaned files while still using only windows with complete observations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
DEFAULT_WINDOW_LENGTH = 42  # approximately two US trading months
DEFAULT_LOCAL_NORMALIZATION = 13
DEFAULT_CLUSTER_THRESHOLD = 0.1465


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    munik_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=munik_dir / "date set" / "pure",
        help="Folder containing sector subfolders with stock CSV files.",
    )
    parser.add_argument(
        "--calendar-mat",
        type=Path,
        default=munik_dir / "date set" / "price" / "Date.mat",
        help="MAT file containing the reference trading calendar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory for all cleaned data, tables, figures, and logs.",
    )
    parser.add_argument(
        "--price-column",
        choices=["Close", "Adj Close"],
        default="Adj Close",
        help="Price series used to calculate returns. Adj Close is the safer default.",
    )
    parser.add_argument(
        "--universe",
        choices=["complete", "all"],
        default="complete",
        help="complete uses stocks with every reference date; all keeps every ticker.",
    )
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--local-normalization", type=int, default=DEFAULT_LOCAL_NORMALIZATION)
    parser.add_argument("--cluster-threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD)
    parser.add_argument(
        "--initial-pairs",
        type=int,
        default=16,
        help="Maximum initial center pairs evaluated per split; increase for slower, closer searches.",
    )
    parser.add_argument(
        "--exact-pair-limit",
        type=int,
        default=16,
        help="Evaluate all initial pairs for nodes no larger than this size.",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def reference_calendar(calendar_mat: Path) -> pd.DatetimeIndex | None:
    """Read the 4,789-date reference calendar shipped with the project."""
    if not calendar_mat.exists():
        return None
    try:
        data = loadmat(calendar_mat)
        raw = np.asarray(data["Date1"], dtype=object).ravel()
        values = [str(np.asarray(x).ravel()[0]) for x in raw]
        dates = pd.to_datetime(values, errors="coerce")
        dates = pd.DatetimeIndex(dates.dropna()).normalize().sort_values().unique()
        return dates
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(f"Could not read reference calendar {calendar_mat}: {exc}")
        return None


def stock_files(input_root: Path) -> list[Path]:
    """Find real CSVs and ignore macOS resource-fork files such as ._AAPL.csv."""
    return sorted(
        p for p in input_root.rglob("*.csv") if not p.name.startswith("._")
    )


def clean_stock_file(path: Path, calendar: pd.DatetimeIndex | None) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"{path}: missing required columns {missing_columns}")

    df = raw[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_date_rows = int(df["Date"].isna().sum())
    invalid_numeric_rows = int(df[REQUIRED_COLUMNS[1:]].isna().any(axis=1).sum())
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).drop_duplicates(subset=["Date"], keep="last")
    duplicate_date_rows = before - bad_date_rows - invalid_numeric_rows - len(df)

    ohlc_bad = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    nonpositive_price = (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1)
    negative_volume = df["Volume"] < 0
    zero_volume = df["Volume"] == 0

    off_calendar_rows = 0
    if calendar is not None:
        calendar_set = set(calendar)
        off_calendar_rows = int((~df["Date"].isin(calendar_set)).sum())
        df = df[df["Date"].isin(calendar_set)].copy()

    df = df.sort_values("Date").reset_index(drop=True)
    report = {
        "sector": path.parent.name,
        "ticker": path.stem,
        "source_file": str(path),
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "date_min": df["Date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "date_max": df["Date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "bad_date_rows_removed": bad_date_rows,
        "invalid_numeric_rows_removed": invalid_numeric_rows,
        "duplicate_date_rows_removed": max(0, int(duplicate_date_rows)),
        "off_calendar_rows_removed": off_calendar_rows,
        "ohlc_violation_rows_retained_for_review": int(ohlc_bad.sum()),
        "nonpositive_price_rows_retained_for_review": int(nonpositive_price.sum()),
        "negative_volume_rows_retained_for_review": int(negative_volume.sum()),
        "zero_volume_rows_retained": int(zero_volume.sum()),
    }
    return df, report


def clean_all_data(
    input_root: Path, calendar: pd.DatetimeIndex | None, out_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex]:
    clean_dir = out_dir / "clean" / "stocks"
    clean_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    reports: list[dict] = []

    for path in stock_files(input_root):
        df, report = clean_stock_file(path, calendar)
        ticker = path.stem
        sector = path.parent.name
        key = ticker
        if key in frames:
            raise ValueError(f"Duplicate ticker name: {ticker}")
        df = df.copy()
        df.insert(0, "Ticker", ticker)
        df.insert(1, "Sector", sector)
        frames[key] = df
        reports.append(report)
        ticker_dir = clean_dir / sector
        ticker_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(ticker_dir / f"{ticker}.csv", index=False)

    if not frames:
        raise FileNotFoundError(f"No CSV files found below {input_root}")

    if calendar is None:
        calendar = pd.DatetimeIndex(sorted(set().union(*(set(df["Date"]) for df in frames.values()))))

    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    quality = pd.DataFrame(reports).sort_values(["sector", "ticker"]).reset_index(drop=True)
    quality["reference_calendar_rows"] = len(calendar)
    quality["complete_history"] = quality["clean_rows"].eq(len(calendar))
    quality.to_csv(out_dir / "clean" / "stock_quality_report.csv", index=False)

    # Wide adjusted-close and arithmetic-return matrices are the direct simulation inputs.
    prices = pd.DataFrame(
        {ticker: df.set_index("Date")["Adj Close"].reindex(calendar) for ticker, df in frames.items()},
        index=calendar,
    )
    close_prices = pd.DataFrame(
        {ticker: df.set_index("Date")["Close"].reindex(calendar) for ticker, df in frames.items()},
        index=calendar,
    )
    prices.index.name = "Date"
    close_prices.index.name = "Date"
    prices.to_csv(out_dir / "clean" / "adjusted_close_wide.csv")
    close_prices.to_csv(out_dir / "clean" / "close_wide.csv")

    selected = quality.loc[quality["complete_history"]].ticker.tolist()
    if not selected:
        raise ValueError("No complete-history tickers remain after cleaning")
    selected_for_report = selected
    quality["used_by_default_simulation"] = quality["ticker"].isin(selected_for_report)
    quality.to_csv(out_dir / "clean" / "stock_quality_report.csv", index=False)
    with (out_dir / "clean" / "cleaning_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input_root": str(input_root),
                "calendar_source": str(calendar),
                "calendar_rows": len(calendar),
                "raw_csv_count": len(frames),
                "complete_history_count": len(selected),
                "incomplete_tickers": quality.loc[~quality["complete_history"], "ticker"].tolist(),
                "zero_volume_rows_total": int(quality["zero_volume_rows_retained"].sum()),
                "note": "Zero-volume rows are retained because adjusted prices are valid; volume is not used for correlation simulation.",
            },
            fh,
            indent=2,
        )

    px = prices[selected]
    returns = px.pct_change(fill_method=None)
    returns.index.name = "Date"
    returns.to_csv(out_dir / "clean" / "arithmetic_returns_wide.csv")
    return frames, quality, calendar


def choose_universe(prices_path: Path, quality: pd.DataFrame, universe: str) -> tuple[pd.DataFrame, list[str]]:
    prices = pd.read_csv(prices_path, parse_dates=["Date"]).set_index("Date")
    if universe == "complete":
        tickers = quality.loc[quality["complete_history"], "ticker"].tolist()
    else:
        tickers = quality["ticker"].tolist()
    prices = prices[tickers].sort_index()
    # Do not silently impute missing prices.  The all-universe mode uses the largest
    # available complete cross-section in each window; the default is fully complete.
    return prices, tickers


def local_normalize(returns: pd.DataFrame, n: int) -> pd.DataFrame:
    rolling_mean = returns.rolling(n, min_periods=n).mean()
    rolling_std = returns.rolling(n, min_periods=n).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (returns - rolling_mean) / rolling_std
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    return normalized


@dataclass
class WindowData:
    end_dates: list[pd.Timestamp]
    matrices: np.ndarray
    features: np.ndarray
    tickers: list[str]
    used_dates: list[tuple[pd.Timestamp, pd.Timestamp]]


def make_windows(
    prices: pd.DataFrame,
    window_length: int,
    normalization_n: int,
    universe: str,
) -> WindowData:
    returns = prices.pct_change(fill_method=None)
    normalized = local_normalize(returns, normalization_n)
    # Every matrix window must contain actual normalized returns for every
    # selected stock.  Removing all rows with a missing normalized value also
    # makes the first window begin exactly after the 13-point warm-up period.
    normalized = normalized.dropna(how="any")
    dates = normalized.index

    windows: list[np.ndarray] = []
    end_dates: list[pd.Timestamp] = []
    used_dates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for end in range(window_length - 1, len(normalized), window_length):
        block = normalized.iloc[end - window_length + 1 : end + 1]
        if universe == "complete":
            block = block.dropna(axis=1, how="any")
            if block.shape[1] != prices.shape[1]:
                continue
        else:
            block = block.dropna(axis=1, how="any")
        if block.shape[1] < 3 or block.shape[0] < window_length:
            continue
        corr = block.corr().to_numpy(dtype=np.float32)
        if not np.isfinite(corr).all():
            continue
        windows.append(corr)
        end_dates.append(pd.Timestamp(block.index[-1]))
        used_dates.append((pd.Timestamp(block.index[0]), pd.Timestamp(block.index[-1])))

    if not windows:
        raise ValueError("No complete two-month windows could be constructed")
    # In complete mode all windows have one common cross-section.  In all mode,
    # varying ticker sets are not comparable, so report and stop clearly.
    shapes = {x.shape for x in windows}
    if len(shapes) != 1:
        raise ValueError(
            "The all-universe mode has varying cross-sections across windows. "
            "Use --universe complete for a comparable correlation-state analysis."
        )
    matrices = np.stack(windows)
    features = matrices.reshape(len(matrices), -1)
    tickers = prices.columns.tolist()
    return WindowData(end_dates, matrices, features, tickers, used_dates)


def similarity_matrix(features: np.ndarray, dimension: int) -> np.ndarray:
    # Equation (2): mean absolute elementwise difference. The full matrix is
    # used, including zero diagonal differences, to retain the paper's scaling.
    distances = squareform(pdist(features.astype(np.float64), metric="cityblock"))
    return distances / float(dimension * dimension)


def normalized_radius(x: np.ndarray, center: np.ndarray) -> float:
    return float(np.mean(np.abs(x - center)))


def candidate_pairs(indices: np.ndarray, similarity: np.ndarray, max_pairs: int, exact_limit: int) -> list[tuple[int, int]]:
    m = len(indices)
    if m <= 1:
        return []
    if m <= exact_limit or max_pairs >= m * (m - 1) // 2:
        return list(itertools.combinations(indices.tolist(), 2))

    # Deterministic, strong candidates: for each point include its farthest
    # partner, then fill with evenly spaced pairs. This makes the large root
    # split tractable while keeping the paper's best-pair idea.
    selected: set[tuple[int, int]] = set()
    for i in indices:
        row = similarity[i, indices].copy()
        row[np.where(indices == i)[0][0]] = -np.inf
        j = int(indices[int(np.argmax(row))])
        selected.add(tuple(sorted((int(i), j))))
    all_pairs = list(itertools.combinations(indices.tolist(), 2))
    if len(selected) < max_pairs:
        stride = max(1, len(all_pairs) // max(1, max_pairs - len(selected)))
        selected.update(all_pairs[::stride])
    return list(selected)[:max_pairs]


def split_node(
    x: np.ndarray,
    indices: np.ndarray,
    similarity: np.ndarray,
    max_pairs: int,
    exact_limit: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(indices) < 2:
        return None
    candidates = candidate_pairs(indices, similarity, max_pairs, exact_limit)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for center_a_idx, center_b_idx in candidates:
        labels = (similarity[indices, center_b_idx] < similarity[indices, center_a_idx]).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        for _ in range(30):
            group_a = indices[labels == 0]
            group_b = indices[labels == 1]
            center_a = x[group_a].mean(axis=0)
            center_b = x[group_b].mean(axis=0)
            da = np.mean(np.abs(x[indices] - center_a), axis=1)
            db = np.mean(np.abs(x[indices] - center_b), axis=1)
            new_labels = (db < da).astype(np.int8)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            if labels.sum() == 0 or labels.sum() == len(labels):
                break
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        group_a = indices[labels == 0]
        group_b = indices[labels == 1]
        center_a = x[group_a].mean(axis=0)
        center_b = x[group_b].mean(axis=0)
        radius = (
            np.mean(np.abs(x[group_a] - center_a)) * len(group_a)
            + np.mean(np.abs(x[group_b] - center_b)) * len(group_b)
        ) / len(indices)
        if best is None or radius < best[0]:
            best = (float(radius), group_a.copy(), group_b.copy())
    if best is None:
        return None
    return best[1], best[2], best[0]


def adaptive_top_down(
    features: np.ndarray,
    similarity: np.ndarray,
    threshold: float,
    max_pairs: int,
    exact_limit: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Paper-style recursive binary divisions with an L1 matrix distance."""
    nodes: list[np.ndarray] = [np.arange(len(features), dtype=int)]
    final: list[np.ndarray] = []
    diagnostics: list[dict] = []
    while nodes:
        indices = nodes.pop(0)
        center = features[indices].mean(axis=0)
        radius = normalized_radius(features[indices], center)
        if radius <= threshold or len(indices) < 2:
            final.append(indices)
            diagnostics.append({"size": len(indices), "radius": radius, "split": False})
            continue
        split = split_node(features, indices, similarity, max_pairs, exact_limit, random_state)
        if split is None:
            final.append(indices)
            diagnostics.append({"size": len(indices), "radius": radius, "split": False})
            continue
        group_a, group_b, split_radius = split
        nodes.extend([group_a, group_b])
        diagnostics.append(
            {
                "size": len(indices),
                "radius": radius,
                "split": True,
                "child_a": len(group_a),
                "child_b": len(group_b),
                "child_weighted_radius": split_radius,
            }
        )
    labels = np.full(len(features), -1, dtype=int)
    for label, indices in enumerate(final, start=1):
        labels[indices] = label
    return labels, pd.DataFrame(diagnostics)


def fixed_kmeans(features: np.ndarray, k: int, random_state: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=25, random_state=random_state, algorithm="lloyd")
    return model.fit_predict(features.astype(np.float64)) + 1


def save_simulation_outputs(
    out_dir: Path,
    windows: WindowData,
    similarity: np.ndarray,
    adaptive_labels: np.ndarray,
    split_diagnostics: pd.DataFrame,
    fixed_labels: np.ndarray,
    threshold: float,
    price_column: str,
    window_length: int,
    local_n: int,
) -> None:
    sim_dir = out_dir / "simulation"
    fig_dir = sim_dir / "figures"
    sim_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.DataFrame(
        {
            "window_id": np.arange(len(windows.end_dates)),
            "window_start": [x[0].strftime("%Y-%m-%d") for x in windows.used_dates],
            "window_end": [x[1].strftime("%Y-%m-%d") for x in windows.used_dates],
            "adaptive_state": adaptive_labels,
            "fixed_k8_state": fixed_labels,
        }
    )
    dates.to_csv(sim_dir / "market_state_assignments.csv", index=False)
    pd.DataFrame(similarity, index=dates.window_id, columns=dates.window_id).to_csv(
        sim_dir / "similarity_matrix.csv", index_label="window_id"
    )
    np.save(sim_dir / "correlation_matrices.npy", windows.matrices)
    np.save(sim_dir / "correlation_features.npy", windows.features)
    split_diagnostics.to_csv(sim_dir / "top_down_split_diagnostics.csv", index=False)

    state_counts = pd.DataFrame(
        {
            "adaptive_state": pd.Series(adaptive_labels).value_counts().sort_index(),
            "fixed_k8_state": pd.Series(fixed_labels).value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    state_counts.to_csv(sim_dir / "state_counts.csv", index_label="state")
    try:
        adaptive_silhouette = silhouette_score(windows.features, adaptive_labels, metric="manhattan")
    except Exception:
        adaptive_silhouette = float("nan")
    try:
        fixed_silhouette = silhouette_score(windows.features, fixed_labels, metric="euclidean")
    except Exception:
        fixed_silhouette = float("nan")
    summary = {
        "price_column": price_column,
        "window_length_trading_days": window_length,
        "local_normalization_n": local_n,
        "similarity_distance": "mean absolute difference over full correlation matrix",
        "cluster_threshold": threshold,
        "window_count": len(windows.end_dates),
        "stock_count": len(windows.tickers),
        "adaptive_state_count": int(len(np.unique(adaptive_labels))),
        "fixed_k": 8,
        "adaptive_manhattan_silhouette": adaptive_silhouette,
        "fixed_k8_euclidean_silhouette": fixed_silhouette,
        "note": "The paper's original exhaustive pair search is approximated at large nodes by deterministic farthest-pair candidates; all pairs are evaluated for nodes up to the configured exact-pair limit.",
    }
    with (sim_dir / "simulation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=True)

    # Similarity heatmap.
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(similarity, cmap="Blues", origin="lower", aspect="auto")
    ax.set_title("Market-state similarity distance")
    ax.set_xlabel("Two-month window")
    ax.set_ylabel("Two-month window")
    fig.colorbar(image, ax=ax, label="mean |C(t1) - C(t2)|")
    fig.savefig(fig_dir / "similarity_heatmap.png", dpi=180)
    plt.close(fig)

    # State evolution, with paper-style adaptive states and fixed-k comparison.
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.step(dates.window_id, adaptive_labels, where="mid", label="adaptive top-down")
    ax.scatter(dates.window_id, fixed_labels, s=16, alpha=0.6, label="fixed k=8")
    ax.set_title("Temporal evolution of market states")
    ax.set_xlabel("Two-month window")
    ax.set_ylabel("State label")
    ax.legend()
    fig.savefig(fig_dir / "market_state_evolution.png", dpi=180)
    plt.close(fig)

    # Largest eigenvalue alternative mentioned in the paper.
    largest = np.linalg.eigvalsh(windows.matrices.astype(np.float64))[:, -1]
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(dates.window_id, largest, color="darkred")
    ax.set_title("Largest correlation-matrix eigenvalue")
    ax.set_xlabel("Two-month window")
    ax.set_ylabel("lambda_max")
    fig.savefig(fig_dir / "largest_eigenvalue.png", dpi=180)
    plt.close(fig)

    # Average correlation matrix per adaptive state, suitable for seminar figures.
    state_dir = sim_dir / "state_centers"
    state_dir.mkdir(exist_ok=True)
    for state in sorted(np.unique(adaptive_labels)):
        center = windows.matrices[adaptive_labels == state].mean(axis=0)
        np.save(state_dir / f"state_{state}.npy", center)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = ax.imshow(center, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
        ax.set_title(f"Average correlation matrix - state {state}")
        ax.set_xlabel("Stock index")
        ax.set_ylabel("Stock index")
        fig.colorbar(image, ax=ax, label="correlation")
        fig.savefig(state_dir / f"state_{state}.png", dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/6] Reading reference calendar and cleaning stock files...", flush=True)
    calendar = reference_calendar(args.calendar_mat)
    frames, quality, calendar = clean_all_data(args.input_root, calendar, args.output_dir)
    print(f"[2/6] Cleaned {len(frames)} stocks; {int(quality['complete_history'].sum())} are complete-history.", flush=True)
    price_file = "adjusted_close_wide.csv" if args.price_column == "Adj Close" else "close_wide.csv"
    prices, tickers = choose_universe(
        args.output_dir / "clean" / price_file, quality, args.universe
    )
    if args.universe == "all":
        incomplete = prices.columns[prices.isna().any()].tolist()
        if incomplete:
            warnings.warn(
                f"--universe all includes incomplete tickers {incomplete}; "
                "the comparable correlation run will use --universe complete instead."
            )
            raise ValueError("Use --universe complete for the paper-faithful correlation simulation")

    print(f"[3/6] Building normalized {args.window_length}-day correlation windows for {len(tickers)} stocks...", flush=True)
    windows = make_windows(
        prices,
        window_length=args.window_length,
        normalization_n=args.local_normalization,
        universe="complete",
    )
    print(f"[4/6] Computing similarity matrix for {len(windows.end_dates)} windows...", flush=True)
    similarity = similarity_matrix(windows.features, len(windows.tickers))
    print("[5/6] Running adaptive top-down clustering and fixed k=8 comparison...", flush=True)
    adaptive_labels, diagnostics = adaptive_top_down(
        windows.features,
        similarity,
        threshold=args.cluster_threshold,
        max_pairs=args.initial_pairs,
        exact_limit=args.exact_pair_limit,
        random_state=args.random_state,
    )
    fixed_labels = fixed_kmeans(windows.features, k=8, random_state=args.random_state)
    save_simulation_outputs(
        args.output_dir,
        windows,
        similarity,
        adaptive_labels,
        diagnostics,
        fixed_labels,
        args.cluster_threshold,
        args.price_column,
        args.window_length,
        args.local_normalization,
    )
    print("[6/6] Saved cleaned data, simulation tables, and figures.", flush=True)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "cleaned_stocks": int(len(quality)),
        "complete_history_stocks": int(quality["complete_history"].sum()),
        "simulation_stocks": len(windows.tickers),
        "simulation_windows": len(windows.end_dates),
        "adaptive_states": int(len(np.unique(adaptive_labels))),
        "fixed_k8_states": int(len(np.unique(fixed_labels))),
    }, indent=2))


if __name__ == "__main__":
    run(parse_args())

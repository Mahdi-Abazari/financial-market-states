# Financial-market state seminar pipeline

This folder contains a reproducible Python implementation of the main empirical
workflow in *Identifying States of a Financial Market* (Muennix et al., 2012).

## What it does

1. Ignores macOS `._*` resource-fork files.
2. Reads all 300 stock CSVs without modifying the original data.
3. Parses dates and numeric fields, removes invalid/duplicate rows, filters to
   the 4,789-date reference trading calendar, sorts chronologically, and writes
   cleaned per-stock files.
4. Writes adjusted-close and close matrices, arithmetic returns, and a complete
   quality report. Zero-volume rows are retained because the price is usable and
   volume is not used by the paper's correlation analysis.
5. Uses the 297 complete-history tickers by default.
6. Applies 13-point local normalization, forms disjoint 42-trading-day
   (approximately two-month) windows, and computes correlation matrices.
7. Computes the paper's similarity distance as the mean absolute difference of
   the full correlation matrices.
8. Performs adaptive top-down binary clustering at the paper's threshold 0.1465,
   and also reports a fixed `k=8` k-means comparison.
9. Saves assignments, similarity/correlation matrices, state centers, diagnostics,
   and figures.

## Run

From the repository workspace:

```powershell
python .\munik\analysis\run_pipeline.py
```

For a sensitivity run using unadjusted close prices:

```powershell
python .\munik\analysis\run_pipeline.py --price-column Close --output-dir .\munik\analysis\output_close
```

Useful output files are:

- `output/clean/stock_quality_report.csv`
- `output/clean/cleaning_summary.json`
- `output/clean/adjusted_close_wide.csv`
- `output/clean/arithmetic_returns_wide.csv`
- `output/simulation/market_state_assignments.csv`
- `output/simulation/similarity_matrix.csv`
- `output/simulation/simulation_summary.json`
- `output/simulation/figures/`

The original article evaluates all possible initial center pairs at each split.
That is expensive for the 300-stock data. The script evaluates every pair for
small nodes and uses deterministic farthest-pair candidates at large nodes; the
choice is recorded in `simulation_summary.json`. Set `--initial-pairs` higher
and `--exact-pair-limit` higher for a closer but slower approximation.

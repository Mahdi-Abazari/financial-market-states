# Seminar project: current reproducibility result

## Paper method translated into Python

The article defines daily arithmetic returns, applies local normalization with
`n = 13`, computes correlation matrices over disjoint two-month windows, and
measures the distance between two market states with the mean absolute
elementwise difference between their correlation matrices. It then uses adaptive
top-down two-way k-means-style splits with threshold `0.1465`. The paper reports
eight market states for its original S&P 500 survivor universe.

The implementation is in [run_pipeline.py](./run_pipeline.py), with the full
workflow described in [README.md](./README.md).

## Dataset audit

- 300 real stock CSVs were found in 10 sector folders. Hidden `._*` files were
  ignored as macOS metadata.
- The supplied MATLAB calendar contains 4,789 trading dates from 1992-01-02 to
  2010-12-31.
- No invalid dates, nonnumeric values, duplicate dates, or OHLC consistency
  violations were found in the real CSVs.
- 286 zero-volume rows were retained because their prices are valid and volume
  is not used by the paper's correlation model.
- `MAC` has 4,235 usable calendar rows and begins in 1994.
- `PLL` has 4,786 usable calendar rows after 50 holiday rows are removed and
  three dates absent from that file are recognized.
- `ADI` has 4,788 usable rows and is missing 2010-12-31.
- The default paper-faithful universe therefore contains 297 complete-history
  stocks. No missing prices are forward-filled.

## Current run

Using adjusted close, 13-point local normalization, 42 trading-day windows,
297 complete-history stocks, and threshold `0.1465`:

- 113 comparable correlation windows were constructed.
- The adaptive top-down procedure produced 2 states for this supplied dataset.
- A fixed `k = 8` comparison produced 8 states, so the paper's reported state
  count can still be compared directly even though the adaptive threshold did
  not reproduce eight states here.
- Repeating the adaptive run with 128 candidate initial pairs still produced 2
  states, so this result is not caused by the faster 16-pair setting alone.

This difference is an important seminar result: the supplied files are not
guaranteed to be the exact survivor universe or preprocessing version used in
the article. The presentation should separate (a) faithful implementation of
the published method, (b) reproduction with the supplied data, and (c) the
fixed-`k=8` comparison.

## Main outputs

- `output/clean/stock_quality_report.csv`: row-level stock quality summary.
- `output/clean/cleaning_summary.json`: cleaning decisions and incomplete tickers.
- `output/clean/adjusted_close_wide.csv`: cleaned adjusted-close matrix.
- `output/clean/arithmetic_returns_wide.csv`: arithmetic daily returns for the
  297-stock default universe.
- `output/simulation/market_state_assignments.csv`: window dates and adaptive/
  fixed-k labels.
- `output/simulation/similarity_matrix.csv`: paper-style state-distance matrix.
- `output/simulation/simulation_summary.json`: run settings and quality scores.
- `output/simulation/figures/`: heatmap, state evolution, and largest-eigenvalue
  figures for the seminar.

The original files remain unchanged; cleaned copies are under
`output/clean/stocks/`.

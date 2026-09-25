# Historical coverage summary

Coverage is measured through January 1, 2026. `All available points` includes
every observation returned within each window, including shorter-lived
tickers. `Full-period tickers` includes only tickers spanning the complete
window with no missing OHLCV values.

| Period | All available points | Full-period tickers | Points from full-period tickers |
|---|---:|---:|---:|
| 20 years | 15,094,777 | 1,834 | 9,226,853 |
| 25 years | 17,178,404 | 1,525 | 9,587,674 |
| 30 years | 18,842,051 | 1,112 | 8,395,595 |
| Maximum, before 2026 | 21,879,914 | 5,210 | 21,879,914 |

The completed raw database contains 22,817,463 observations across 5,842
tickers through September 10, 2026. The difference from the maximum-history
comparison comes from observations during 2026 and tickers first listed in
2026. The scan found no partially missing OHLCV rows among returned data.

## Raw-history neural windows (2026-09-24)

The scan used `D:/DBs/timeseries_analysis/history_coverage.db` (23,995,736,064
bytes; last modified 2026-09-20 11:33:39 UTC). The new equity cohort contains 2,714 tickers with more than 80 raw rows before
2016-01-01 and a raw row on 2025-12-31. No ten-year training-history minimum
is applied. Rows after 2025-12-31 are excluded. Each target is the next
**recorded observation** for its ticker; absent raw dates are treated like
weekends. Among adjacent raw observations in this cohort, 11 date gaps exceed
seven calendar days, none exceeds 30, and the maximum is nine days.

Daily unannualized Garman-Klass variance is
0.5 × ln(High/Low)² − (2 ln 2 − 1) × ln(Close/Open)², using adjusted OHLC.
A row is invalid for missing or nonfinite OHLCV, nonpositive prices, negative
volume, High < max(Open, Close), Low > min(Open, Close), or negative or
nonfinite variance. A window needs valid positive-variance input rows and a
valid target row. Zero close/open return remains a valid LSTM input. A valid
zero-variance target is changed to f = 3.396062419686545 × 10⁻¹³, the
smallest positive pre-2016 variance in this cohort; positive targets are
unchanged. The raw target is retained in `RawVariance`. Training and all five
reported metrics use the adjusted target, including each ticker's adjusted
pre-2016 MASE scale. Predicted variance has the same training-derived lower
floor. This target adjustment is a project decision, not an estimator identity.

| Input window | Split | Usable windows | Share of this window's total |
|---|---|---:|---:|
| 20 | Train, before 2016 | 8,664,516 | 57.95% |
| 20 | Validation, 2016-2018 | 1,806,548 | 12.08% |
| 20 | Test, 2019-2025 | 4,479,681 | 29.96% |
| 20 | **Total** | **14,950,745** | **100%** |
| 80 | Train, before 2016 | 7,004,011 | 53.91% |
| 80 | Validation, 2016-2018 | 1,692,154 | 13.02% |
| 80 | Test, 2019-2025 | 4,295,773 | 33.06% |
| 80 | **Total** | **12,991,938** | **100%** |

Across usable target windows, 25,519 zero targets are adjusted for window 20
and 5,640 for window 80. The earlier positive-table runs used a different
cohort and observation set (5,416,755 / 1,120,921 / 2,649,257 train / validation /
test windows before the optional consecutive-session filter). Their metrics
cannot be compared directly with these raw-history windows. Counts above came
from `python -m models.raw_neural_coverage D:/DBs/timeseries_analysis/history_coverage.db`.

CBIO and VATE have no valid pre-2016 variance rows. They contribute no training
or validation windows, but 2,134 window-20 and 1,912 window-80 test windows.
Their training-only MASE scale is undefined. Forecast coverage above retains
those windows; the common five-metric test scoring set excludes them, leaving
4,477,547 window-20 and 4,293,861 window-80 test observations. All five
metrics use that same scoring set. This is an evaluation exclusion, not a
change to the selected raw-history cohort or window rule.

A full pre-2016 scan confirmed that all other 2,712 tickers have at least two
finite positive adjusted targets and a strictly positive naive MASE scale.
The full raw loader and `TimeSeriesDataset` independently reproduced every
split count in the table for both window sizes.

# Realized Utility Setup Implementation Plan

## Implementation decisions

1. Use AAPL, AMZN, GOOG, NFLX, and NVDA on their 1,760 identical 2019--2025 test dates: 8,800 ticker-date observations per model.
2. Use the ten-model intersection retained across both MCS block-length runs. Read the saved 20- and 80-window `forecasts.csv` files without refitting.
3. Use each saved positive daily, unannualized adjusted Garman--Klass target as realized variance and each saved positive forecast as expected variance. Require exact ticker-date alignment and identical targets across models.
4. Read adjusted Open and Close from `raw_history` on those exact keys. Use the net simple session return `exp(log(Close/Open)) - 1`.
5. Average returns within each ticker, then across five tickers. Do the same for actual GK variance.
6. Use YCharts' reported 4.26% long-term average 10-year Treasury yield as a fixed annual risk-free proxy; convert it to `(1 + 0.0426)^(1/252) - 1` per trading day.
7. Use one common, unannualized Sharpe input: `(mean return - daily risk-free proxy) / sqrt(mean actual GK variance)`. It is retrospective because the test returns were unknown at forecast time.
8. Adapt Zhang et al. (2022), Section 5.4, Equation (21) with risk aversion `gamma = 2`: `RU = SR^2/gamma * sqrt(actual/predicted) - SR^2/(2*gamma) * actual/predicted`.
9. Average RU within ticker, then across tickers; multiply by 100 to report daily percent of wealth. Do not deduct transaction costs or simulate portfolio returns.

## Dataset and implementation work

- `inference/calculate_realized_utility.py` joins the saved forecast keys to adjusted prices, checks positivity, finiteness, duplicate and missing keys, and writes `imgs/eval/utility/realized_utility.csv`.
- `ref/final_report/thesis_structure.tex` states the four defining calculations, cites the paper and Treasury proxy, and shows one ten-row RU table. Results interpretation remains for the thesis author.

## Verification

- Check RU against hand-calculated cases: actual = predicted = 4 with SR = 1 and gamma = 2 gives 0.25; actual = 4 and predicted = 1 gives 0.
- Require 8,800 exact keys for every model, 1,760 identical dates per ticker, positive finite prices and variances, and no silent exclusions.
- Reconcile the thesis table to the script's CSV, compile the thesis twice with SyncTeX, inspect the table, and obtain a fresh-context cold review.

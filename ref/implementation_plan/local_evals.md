# Local and Global Volatility Evaluation Implementation Plan

## Implementation decisions

1. Evaluate the 14 saved global models and five-stock local counterparts for NVDA, AAPL, NFLX, GOOG, and AMZN. Do not retrain.
2. Use adjusted daily, unannualized Garman-Klass variance: 0.5 × ln(High/Low)² − (2 ln(2) − 1) × ln(Close/Open)². Forecast the next recorded observation within each ticker, on test targets dated 2019-01-01 through 2025-12-31.
3. Score each windowed LOCAL/GLOBAL pair on the same ticker/date keys. Each stock has 1,760 eligible targets for both 20- and 80-observation windows, giving N=8,800 per windowed row.
4. Use test-period MASE: each ticker/window's scale is the mean absolute difference between each eligible test target and its immediately preceding recorded variance. Divide each absolute forecast error by this scale, then average over forecasts. The same scale applies to both scopes. Test targets enter evaluation only. This differs from training-scaled MASE in `losses.md`.
5. Forecast RFSV-full once per stock at 2025-12-31 using all preceding valid positive variance observations. Its MASE scale is that date's absolute variance change from the immediately preceding recorded observation. Its N=5 rows are labeled `final-date` and are not ranked against full-period rows.
6. Preserve saved neural conventions: MLP and Base LSTM use 0.5 log variance and adjusted intraday log return, outputting log variance; SiLU-LSTM uses variance and adjusted intraday log return, outputting variance; HARNet uses variance alone and outputs its floored variance convention.
7. Preserve saved statistical inputs: AR(1) and HAR use preceding GK variance, SARIMA filters GK variance with invalid rows missing, GARCH uses causal adjusted intraday-return residuals, and RFSV-full uses all preceding valid positive GK variance.
8. Apply each saved fit's positive forecast floor; leave actual targets at the source loader's saved zero-target policy. Report floor-hit percentages.
9. Average MAE, MASE, MSE, and QLIKE over observations, and compute RMSE as √(aggregated MSE). With equal per-stock counts, observation weighting equals equal-stock weighting.
10. Label rows with model name and LOCAL/GLOBAL. Leave the historical global CSV unchanged because its population differs.
11. Save the 28-row CSV and PNG table, 140-row per-stock audit, run log, and six existing residual plot types per row under `imgs/eval/local_eval/`. The PNG follows the existing volatility metrics table, shows N, and colors the lowest and second-lowest full-period loss per metric; RFSV rows are separate and unranked. Each scope's residual plots use the same five stocks; RFSV timelines show one date.
12. Compare saved configurations. Global Base LSTM-20 and local Base LSTM-20 had different training budgets, so their difference cannot be attributed solely to scope.

## Evaluation work

- `inference/evaluate_local_global.py` validates source metadata, checkpoints, input/output conventions, floors, scales, and target dates; it stops on missing artifacts or failed rows.
- Reuse the saved-forecast scoring and residual-plot functions; score every local/global pair on identical test targets.

## Verification

- Check test-period MASE, transforms, alignment, floors, and aggregation on a tiny known example.
- Confirm 28 table and 140 audit rows; N=8,800 for windowed rows and N=5 for RFSV rows. Reload outputs and inspect plots.
- Obtain a fresh-context JSON cold review and validate with `python judge/validate.py REVIEW.json`.

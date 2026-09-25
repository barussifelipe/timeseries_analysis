# Raw-history neural data fixes

Implemented 2026-09-24 for the base LSTM, SiLU-LSTM, HARNet-20, and HARNet-80.
No training or test evaluation was launched.

## Before

1. Neural runs read the positive-variance derived table. Zero-variance raw rows were absent, so windows could skip recorded observations and change the target horizon.
2. The older full-period positive-table cohort had 1,525 tickers and 5,416,755 / 1,120,921 / 2,649,257 train / validation / test window-20 forecasts before the optional consecutive-session filter. Those observations differ from the new cohort.
3. The pending consecutive-session filter dropped windows around missing panel dates. It did not restore source rows excluded by the positive table.
4. Actual targets were strictly positive derived-table variances. Earlier metrics and per-ticker MASE scales therefore used that different target and observation set.
5. Two members of the new raw cohort, CBIO and VATE, have no valid pre-2016 variance history, so a training-only ticker MASE scale cannot be defined for them.

## After: implementation decisions

1. `--raw-history` selects exactly the 2,714 equity tickers with more than 80 pre-2016 raw rows and a row on 2025-12-31. There is no ten-year training-history minimum. The LSTMs and HARNet-20 use window 20; HARNet-80 uses window 80.
2. Compute daily, unannualized Garman-Klass variance from adjusted OHLC. Mark rows invalid for missing/nonfinite OHLCV, nonpositive prices, negative volume, inconsistent High or Low, or negative/nonfinite variance.
3. Reject a window when any input row is invalid or has zero variance, or its target row is invalid. Zero close/open return is a valid LSTM feature. The target is the next recorded observation, even across absent raw dates.
4. Adjust each valid zero-variance target to f = 3.396062419686545e-13, the smallest positive pre-2016 cohort variance; leave positive targets unchanged. Keep the original in `RawVariance`. Training and MAE, MASE, MSE, RMSE, and QLIKE use the adjusted target. Predictions have the same lower floor.
5. Split by target date: train before 2016, validation 2016-2018, test 2019-2025. Use adjusted, valid pre-2016 histories for each ticker's MASE scale. All 2,712 tickers other than CBIO and VATE have a strictly positive scale.
6. Keep CBIO and VATE in the cohort and forecast coverage. Their 2,134 window-20 and 1,912 window-80 test forecasts are excluded from the common five-metric scoring set. The scoreable test counts are 4,477,547 and 4,293,861. Later baselines must use the same observations.
7. Save the cohort ticker list, window rule, target-floor policy, and floor in new checkpoint settings; verify cohort and floor at prediction. Preserve the legacy positive-table checkpoint path. Raw mode supports equity Garman-Klass only and disallows row truncation and the consecutive-session filter.
8. Report the raw-date gaps and split counts in `ref/coverage/history_coverage.md`. The full dataset constructor matched the independent scan for both window sizes. The raw path has 11 adjacent date gaps over seven calendar days; none over 30.

## Verification and next step

`python -m models.raw_neural_coverage <database>` reproduces coverage; the saved scan is `ref/coverage/raw_neural_coverage_2026-09-24.json`. The synthetic raw-window, forecast, scoring-exclusion, and legacy focused checks passed with `.venv/Scripts/python.exe`. Pytest is absent from that environment; the system Python has an incompatible PyTorch/NumPy pair. Earlier metrics cannot be compared directly with this new observation set. The fresh-context JSON cold review passed 100/100 with no findings at `judge/reviews/raw_neural_window_review.json`, and `python judge/validate.py` validated it. Next: plan a baseline comparison on the common scoring set; neural training requires a separate request.

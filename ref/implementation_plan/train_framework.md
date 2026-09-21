# Volatility Training Framework Implementation Plan

## Implementation decisions

1. Fit Parkinson and Garman–Klass variance as separate targets. Forecast daily, unannualized variance in each estimator's original units.
2. Treat each ticker's next available observation as the one-step target. Sort by `(Ticker, Date)` and keep all lags and windows within that ticker.
3. Fit on pre-2016 equities. Use 2016–2018 equities for neural validation and 2019–2025 equities for testing. Apply the split to target dates, so a validation or test window may contain earlier observed history.
4. Fit global models with one parameter set shared across eligible equity tickers. Fit local models with only the selected ticker. Weight global objectives by eligible observations, so longer histories contribute more.
5. Use crypto only as an unseen test against the matching Parkinson or Garman–Klass proxy. Analyze crypto realized variance separately.
6. Report the actual eligible train, validation, and test counts and percentages after windowing. The source equity tables' pre-window split is 59.10% / 12.16% / 28.74%.
7. Reject nonfinite or nonpositive variance in the variance dataset and targets. Do not fill invalid variance with zero.
8. Use only preceding proxy-variance observations from the same ticker as neural input. Store input and target tensors as `float32`; do not normalize inputs or targets.
9. Give neural models a raw real-valued final output. Do not use softplus, absolute value, median scaling, mean scaling, or another output transform.
10. Set each fit's forecast floor to the minimum positive pre-2016 variance in its fitting history: across selected training tickers for a global fit, or within the selected ticker for a local fit. Save it with the fit and reuse it unchanged for validation, equity test, and crypto test.
11. Apply the lower floor as `max(raw_prediction, floor)` before loss or scoring. Only predictions below the floor change; actual variance stays unchanged. A neural output below the floor has zero gradient through this clamp, so inspect small-trial diagnostics for stalled fitting.
12. Train neural models with mean QLIKE over eligible windows. Report QLIKE as the primary evaluation score, plus MAE, MASE, MSE, and RMSE where the required training history exists.
13. Return one `Forecast(ticker, target_date, actual_variance, predicted_variance)` per neural prediction. The actual is the unchanged source variance; the prediction is the floored daily variance in the target estimator's units.
14. Compute equity MASE using each ticker's pre-2016 one-step naive variance scale. Omit crypto MASE because unseen crypto tickers have no equity training history.
15. Use unweighted OLS on pooled, valid within-ticker lag rows for global AR(1) and HAR; local fits use the selected ticker's rows. AR(1) uses one lag. HAR and HARNet require at least 20 observations of history.
16. Keep SARIMA order `(1, 0, 1)` and seasonal order `(1, 0, 1, 5)` fixed for this stage. Fit complete within-ticker training series and sum separate ticker log-likelihoods for a global fit.
17. Fit GARCH on zero-mean intraday log returns `ln(Close/Open)` as residuals, without estimating and subtracting a return mean. Sum separate ticker likelihood contributions for a global fit. The chosen variance proxy controls aligned dates, but does not enter the GARCH likelihood.
18. Estimate RFSV H only from fitting-history log volatility through the existing roughness estimator, using up to 20 exact calendar-day lags. Keep the saved full-period H estimates descriptive. Fix ν = 0 and default to one prior observation for its forecast.
19. Keep statistical specifications fixed in this stage; do not use neural validation to change their parameters or select their specifications. Prediction must not refit statistical parameters.
20. Default neural settings to seed 42, hidden width 16, 30-observation windows (20 for HARNet), Adam learning rate 0.001, batch size 128, and at most 20 epochs. Start HARNet from random initialization; its optional HAR-coefficient initializer is not used.
21. Clip neural gradient norm to 1. Save the best validation-QLIKE checkpoint. After five consecutive validation misses, reduce the learning rate once by a factor of ten; stop after another five misses.
22. Enable W&B logging by default, with a command option to disable it. `--limit-rows` keeps the last N training rows and first N validation/test rows per ticker; limited trials can therefore have gaps at split boundaries.
23. Give each model its own `train`, `predict`, and `python -m models.<name>` entry point, with database, estimator, scope, local ticker, run name, and small-trial limits as common arguments. Expose window size where history length is a model choice; SARIMA and GARCH fit complete selected training series.
24. Save model artifacts under `inference/checkpoints/<model>/<estimator>/<scope>/[ticker]/<run-name>/fit.pth`. Record parameters or weights, floor, settings, dates, and data counts. Reusing a run name overwrites its fit file. Keep `base_lstm_vol` separate from existing `base_lstm_returns` artifacts.
25. Keep the existing return experiment as a reference in `inference/inference.py`; retire `inference/training.py` and update affected imports, tests, README, and context notes.
26. Run the required JSON cold review in a fresh, cleared context. A same-context self-check does not count as that review.

## Dataset and training work

- Move `TimeSeriesDataset` into `models/training_blocks.py`, retaining concatenated tensor storage and ticker-safe windows. Make feature columns, target column, target-date split, and window size explicit.
- Put shared QLIKE, variance metrics, W&B setup, floor, forecast records, and checkpoint helpers in `models/training_blocks.py`.
- Give AR(1), HAR, SARIMA, GARCH, RFSV, MLP, HARNet, SiLU-LSTM, and base LSTM volatility their own fitting and prediction paths. Reuse the LSTM fitting loop in `models/base_lstm.py` for neural models.
- Add the verified OLS, [SARIMAX likelihood](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.loglike.html), [GARCH fitting](https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html), and [RFSV](https://arxiv.org/abs/1410.3394) sources to `ref/thesis/references.md`.
- Leave WLS and MSE fitting comparisons for later experiments.

## Verification

- Run one small synthetic end-to-end smoke test for every model's global and local fit, prediction shape, ticker boundaries, next-observation alignment, floor, artifact path, and reload.
- Check that statistical parameters remain fixed during prediction and that neural retry logic saves the best checkpoint.
- Run the repository's cold review in a fresh, cleared context and validate its JSON result.
- Do not train on the project dataset, download data, or run full experiments in this implementation stage.

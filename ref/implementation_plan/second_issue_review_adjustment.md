# Neural Forecast Floor Adjustment

## Implementation decisions

1. Keep input windows and actual targets as raw, daily, unannualized Parkinson or Garman–Klass variance. The horizon is the next available observation within each ticker.
2. Interpret MLP, base LSTM, and SiLU-LSTM outputs as z = log(ŷ), with variance forecast ŷ = exp(z). Do not apply a forecast floor to these three models.
3. Initialize their final bias to log(median of selected pre-2016 training variances). Multiply their Xavier-initialized final weights by 0.01. Keep recurrent gate initialization unchanged.
4. Keep HARNet-20 and HARNet-80 additive variance outputs and matching fitted-HAR initialization. With training-derived floor f, use ŷ = max(raw output, f) and z = log(ŷ). Below f the clamp gives zero gradient.
5. Train and select neural checkpoints by mean QLIKE over forecast observations: r = log(y) − z; QLIKE = mean[expm1(r) − r]. The tensor variance-unit QLIKE uses r = log(y) − log(ŷ). Require finite positive actual and predicted variance and finite loss.
6. Log HARNet floor-hit percentage as 100 × count(raw training forecasts < f) / count(training forecasts) each epoch alongside validation QLIKE. This is an observation-weighted training-batch diagnostic, not a validation statistic.
7. Save `output_convention` in each neural checkpoint and reject a missing or incompatible convention on reload. Statistical models and their forecast floors are unchanged.

## Verification

- Passed with `.venv/Scripts/python.exe -m models.test_neural_forecast_floor`: small-variance QLIKE agreement with NumPy, finite log-output losses and earlier-layer gradients, both HARNet fitted-forecast behavior and below-floor zero gradient/floor-hit logging, and post-reload variance-unit prediction with incompatible-checkpoint rejection.
- Passed with `.venv/Scripts/python.exe -m models.test_training_smoke`: synthetic global/local fit and reload for both estimators, including neural and statistical models.
- Passed with `.venv/Scripts/python.exe -m models.test_models`, direct `test_garch_causal_residuals_and_forecasts()` and `test_harnet_starts_from_fitted_har()`, and `git diff --check`. `pytest` is absent from the virtual environment; the adjustment checks were run directly.
- No project dataset or long training run was used.
- The fresh-context cold review in `judge/reviews/neural_forecast_floor_review.json` passed validation at 100/100 with no findings.

## Remaining limits

The assumption that HARNet floor hits are infrequent has not been checked on project data. Neural widths, windows, convergence, and comparative accuracy remain for later validation trials.

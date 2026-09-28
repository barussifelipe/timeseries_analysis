# Full-History RFSV Fit Implementation Plan

## Implementation decisions

1. Keep the 2,714-ticker raw-history adjusted Garman–Klass cohort and the pre-2016 fitting cutoff. The existing pooled observation-lag H and ν² estimates remain the only fitted parameters; no optimizer or test target enters this fit.
2. Make `RFSV.forward()` accept finite positive adjusted GK **variance** history from one ticker and return one next-recorded-observation variance forecast. The forward pass first approximates the paper's numbered equation (5.1), the conditional mean of log variance, using exact observation-bin kernel weights. It then applies the unnumbered Section 5.2 correction, +2c(H)ν² for horizon Δ = 1 observation, before exponentiating.
3. For the later one-forecast-per-ticker prediction, use all preceding valid positive GK observations available within that ticker, with no fixed 20-observation cap. Invalid and zero-variance source rows cannot enter the logarithm and are omitted. This compresses gaps into observation time, matching the observation-lag parameter convention; it is an approximation to the paper's continuous-time history.
4. In this fit-only step, call `RFSV.forward()` once per ticker on its complete valid positive **pre-2016** history to verify the fitted parameters and full-history computation. Do not score these outputs or use validation/test observations. Tickers with no positive pre-2016 history cannot take part in this forward check; record their count.
5. Save a separate compact fit at `inference/checkpoints/rfsv/garman-klass/global/raw_history_full/fit.json`. Preserve the prior `raw_history_w20/fit.json` and its already reported test result as historical 20-window artifacts; they are not results of the new full-history method.
6. Record source database version, cohort, training-row and ticker counts, positive training floor, parameter source, full-history input rule, and forward-check count. The exact later forecast cutoff and evaluation date set are unresolved because this task is fit-only.
7. The shared CLI still supplies its legacy default `--window-size 20` argument. The full-history RFSV fit does not use it; a nondefault value is rejected so it cannot be mistaken for a forecast cap.

## Fit and model work

- `models/rfsv.py`: share equation (5.1)'s log-space kernel between variance-unit `forward` and the older volatility-unit `forecast` interface; keep the Section 5.2 correction in both outputs without squaring tiny volatility inputs.
- `models/variance_fit.py`: permit the new full-history forecast rule in the saved RFSV parameter metadata, keeping the old 20-window default for prior callers.
- `models/raw_statistical_fit.py`: route the raw RFSV fit-only command to a separate full-history artifact and run the pre-2016 per-ticker `forward` check.

## Verification

- Check `forward` on constant and varying synthetic histories against direct numerical integration of equation (5.1), including the Section 5.2 correction and the full-history tail convention.
- Run the fit-only command; reload its artifact and verify source, parameters, history counts, and positive outputs. Do not score validation/test, retrain neural models, or overwrite the 20-window RFSV artifact.
- Obtain a fresh-context cold review and validate its JSON.

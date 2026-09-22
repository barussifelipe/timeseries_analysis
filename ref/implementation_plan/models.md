# Model implementation plan

## Implementation decisions

1. Define the active models now; defer dataset wiring, project-data fitting runs, neural training, and evaluation until the canonical variance target, units, transformation, horizon, and chronological splits are fixed.
2. Use only one ticker's ordered history for each forecast. In the crypto transfer test, keep parameters fitted on equities fixed and change only the ticker history supplied for prediction.
3. Keep one file per active model in `models/`. Move the unchanged custom `FEBCellLSTM` and `FEBLSTM` to `models/base_lstm.py` so old return checkpoints remain loadable.
4. Keep the VVSI return experiment separate in `inference/`, including its training code, tests, and preserved checkpoints. Run its entry point as `python -m inference.inference`.
5. Define AR(1) as an intercept plus a supplied coefficient times the latest observed variance. This adapts the one-lag case of the [AR baseline in Tang et al.](https://arxiv.org/html/2311.04727) from volatility to this project's variance target; coefficient fitting belongs to the later fitting stage.
6. Define HAR with four coefficients: intercept, latest one-observation variance, mean of the latest 5 observations, and mean of the latest 20 observations. Fit coefficients by least squares on within-ticker history.
7. Use statsmodels SARIMAX for SARIMA. Start with `order=(1,0,1)` and `seasonal_order=(1,0,1,5)`; fit coefficients on equity history and filter each prediction history with those fixed coefficients without refitting.
8. Treat SARIMA orders as initial candidates. Select them later on chronological equity data using information criteria and residual checks.
9. For GARCH, calculate each ticker's within-session log return as `r_t = log(C_t) - log(O_t) = log(C_t/O_t)`, using positive, finite, consistently adjusted open and close for equities. Obtain the residual as `ε_t = r_t - μ_t`, where `μ_t` is the fitted mean log return. Feed those residuals, supplied ω, α, β, and an initial conditional variance to GARCH(1,1); return volatility as the square root of the next conditional variance. Using close/open returns aligns the GARCH return interval with the interval used by the equity Parkinson and reduced Garman–Klass estimators. Crypto realized variance includes the first 15-minute return from the preceding day's 23:45 close, so check that boundary before claiming exact alignment in the transfer evaluation. Assess fitted parameters and residuals later.
10. Require H as an RFSV input and forecast from past log volatility with the cited rough kernel. Use a midpoint approximation over observed history; optional ν controls the paper's multiplicative correction.
11. Treat the existing full-period global equity Garman–Klass H estimate of about 0.03498 as descriptive. Select an H appropriate to the later out-of-sample claim before evaluation.
12. Define the MLP with configurable width and two SiLU hidden layers. Preserve the existing base LSTM architecture and define SiLU-LSTM by replacing only its candidate and cell-output tanh operations with SiLU; its gates remain sigmoid.
13. Implement separate HARNet-20 and HARNet-80 models with hierarchical causal 1/5/20 and 1/5/20/40/80-observation features. Initialize their averaging filters and output layers from matching fitted HAR coefficients so forecasts match those HAR fits on positive variance histories before optimization.
14. Select neural widths and window lengths from later equity validation results; do not treat initial defaults as optimal.
15. Verify model formulas and neural gradients on small synthetic histories, run the moved VVSI tests and entry-point import check, and load an old checkpoint. Do not fit or train on project datasets in this stage.
16. Later compare models on identical valid observations in each selected positive variance estimator's units using the [defined losses](losses.md); record failed fits and excluded observations.

## Scope and prerequisite

This stage implements model computations and fitting interfaces only. Dataset wiring, fitting runs, neural training, and evaluation follow after selecting the canonical variance target and units. Every forecast uses one ticker's history. For crypto transfer, keep equity-fitted parameters fixed and change only the ticker history.

Implement the active models in [the thesis plan](../thesis/plan.md#model-definitions-and-scope). Defer vol-LSTM, TSFMs, and the fine-tuned SLM. First fix the canonical target (Parkinson, Garman-Klass, or crypto realized variance), units, next-period horizon, transformation, and train/validation/test samples. The existing return dataset and checkpoints cannot establish volatility accuracy.

## 1. Separate architectures from experiments

1. Move `model/model.py` to `models/base_lstm.py` without changing `FEBCellLSTM` or `FEBLSTM` behavior.
2. Rename `model/` to `inference/`, retaining `training.py`, `inference.py`, its tests, and existing checkpoints. Update imports to the explicit modules (`models.base_lstm`, `inference.training`) and run the entry point as `python -m inference.inference` from the repository root. Update the tests, README, context, and `.gitignore` paths together.
3. Replace hardcoded `model/checkpoints` paths in training and tests with `inference/checkpoints`. Preserve existing checkpoint files and verify that an old `FEBLSTM` state dictionary still loads after the move. Avoid the current wildcard imports.

## 2. Establish a shared comparison

1. Later, build target-specific, within-ticker lag windows from the selected variance series. Split chronologically; fit normalization and any clipping bounds on training data only. Preserve the VVSI return experiment as a separate runnable path.
2. Define one forecast record with ticker, forecast date, target, prediction, estimator, horizon, and model name. Compare only common valid observations within each estimator. Convert every prediction to that estimator's positive variance units before applying the [defined losses](losses.md); document any positivity transform and back-transformation.
3. Implement HAR fitting and AR(1) coefficient-based forecasting as simple benchmarks. Keep each in its own file under `models/`.

## 3. Add model files incrementally

| File | Implementation focus |
| --- | --- |
| `models/har.py` | Intercept plus latest variance, latest-five mean, and latest-twenty mean; least-squares `fit(history)`. |
| `models/ar1.py` | AR(1): intercept plus supplied coefficient times latest variance. |
| `models/sarima.py` | SARIMAX with initial orders (1,0,1) and seasonal (1,0,1,5); `fit(equity_history)` estimates parameters, while `forecast(ticker_history, fitted_params)` filters without refitting. |
| `models/garch.py` | GARCH(1,1) from within-ticker log close/open return residuals, supplied ω, α, β, and initial variance; output square-root volatility. |
| `models/rfsv.py` | Forecast from past log volatility with required H using the cited rough kernel; optional ν gives the multiplicative correction. |
| `models/mlp.py` | PyTorch network with configurable width and two SiLU hidden layers. |
| `models/silu_lstm.py` | Custom cell variant replacing candidate and cell-output tanh with SiLU; sigmoid gates remain. |
| `models/harnet_20.py`, `models/harnet_80.py` | Implemented separately: hierarchical causal 1/5/20 and 1/5/20/40/80-observation convolutions, initialized from matching fitted HAR coefficients on positive inputs. |

The two HARNet modules are callable as `python -m models.harnet_20` and
`python -m models.harnet_80`. Training fits four or six HAR coefficients by
ordinary least squares on the selected pre-2016 equity histories, then starts
QLIKE optimization from those coefficients. The 80-observation model extends
the 20-observation path with two trainable two-tap convolutions spaced 20 and
40 observations apart; both start with weights of 1/2. Input windows must
contain at least 20 or 80 observations respectively. Checkpoints use separate
`inference/checkpoints/harnet_20/` and `harnet_80/` paths. Synthetic checks
cover initialized HAR equality, short-window errors, checkpoint reload, and
local/global fits for both variance estimators. Project-data fits and model
comparisons remain pending.

Retain `models/base_lstm.py` as the architectural control and its old return checkpoints. The existing full-period global equity Garman–Klass H estimate is about 0.03498; require H as RFSV input and select an out-of-sample appropriate value later. The SARIMA orders are starting choices, not fitted coefficients or proven optima. Compare candidate orders on chronological equity data using information criteria and residual checks per [statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_internet.html). Start GARCH with (1,1) and later assess fitted parameters and residuals per [ARCH](https://arch.readthedocs.io/en/latest/univariate/forecasting.html). Select neural widths and windows on equity validation results, rather than treating initial defaults as optimal.

## 4. Verify and compare

Check synthetic histories for HAR's separate 1/5/20 terms, SARIMA fit and frozen-parameter forecasting, GARCH constraints and volatility units, supplied-H RFSV, neural output shapes and backward passes, and HARNet equality with initialized HAR. Run the moved VVSI tests, entry-point import smoke check, and an old checkpoint load. Do not fit or train on project datasets in this stage. Later compare predictions on identical rows using the [defined losses](losses.md); log failed fits and excluded observations.

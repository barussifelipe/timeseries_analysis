# One-step variance forecast losses

## Implementation decisions

For forecast observation i, let yᵢ be the actual variance, ŷᵢ the predicted variance, and eᵢ = yᵢ − ŷᵢ. Forecasts and actuals must use the same target estimator's variance units; compare estimators separately. All five metrics average over N forecast observations, so tickers with more observations carry more weight.

| Metric | Definition |
| --- | --- |
| MAE | meanᵢ \|eᵢ\| |
| MASE | meanᵢ (\|eᵢ\| / s(tickerᵢ)) |
| MSE | meanᵢ eᵢ² |
| RMSE | √MSE |
| QLIKE | meanᵢ [yᵢ/ŷᵢ − log(yᵢ/ŷᵢ) − 1] |

For ticker t, sₜ = meanⱼ≥2 \|vₜ,ⱼ − vₜ,ⱼ₋₁\|, using only its chronologically ordered training-history variance. This is the in-sample one-step naive MAE scale from [Hyndman and Koehler (2006)](https://robjhyndman.com/papers/mase.pdf). Their paper also defines the scale-dependent MAE, MSE, and RMSE measures.

Require nonempty, aligned forecast vectors; finite positive yᵢ and ŷᵢ; and at least two finite positive training variances per scored ticker with finite positive sₜ. Reject an undefined QLIKE result. The implementation is `variance_metrics` in `inference/training.py`; the signed-return metrics remain separate.

[Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf) identifies QLIKE as a robust variance-forecast comparison loss under its stated assumptions, including a conditionally unbiased proxy for the latent conditional variance. That result does not automatically validate every Parkinson, Garman-Klass, or realized-variance proxy here; check estimator assumptions before interpreting rankings as latent-variance rankings. Use this QLIKE definition as the later volatility LSTM training objective.

For HARNet and the statistical volatility models, apply a strictly positive lower floor to their **variance forecasts** before QLIKE and evaluation. Choose and record the floor from that estimator's training target scale; keep it fixed for validation and test. MLP and volatility LSTMs instead output log variance and use `exp(output)` without a floor, as specified in [the neural adjustment](second_issue_review_adjustment.md). No model has an upper forecast cap. If a later model outputs volatility, apply the equivalent square-root floor in volatility units before converting to variance. Do not silently floor actual targets or forecasts inside `variance_metrics`; the model's forecast transform must be explicit and shared across evaluations.

The main analysis concerns higher volatility, so the lower floor is expected to have limited practical impact. At evaluation, report the share of forecasts affected by the floor and check whether it changes model rankings, including on high-volatility observations.

Winsorization, other clipping bounds, and joint ES/VaR loss are deferred. Volatility-model training and evaluation await the canonical target, horizon, splits, and common forecast observations.

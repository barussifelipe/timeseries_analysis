# Training adjustments

These decisions supersede the conflicting GARCH mean and HARNet initialization choices in `train_framework.md` (items 17 and 20).

## GARCH

For each ticker `k`, order aligned proxy-variance and open/close observations by date. Let `return[k,t] = log(Close[k,t] / Open[k,t])`. The causal running mean starts at `mean[k,0] = 0`; for `t > 0`, `mean[k,t] = sum(return[k,j] for j < t) / t`. Set `error[k,t] = return[k,t] - mean[k,t]`. Update the mean after observing each return. Each ticker has its own state: equity validation (2016-01-01 through 2018-12-31) and test (2019-01-01 through 2025-12-31) carry that ticker's past forward, while each unseen crypto ticker starts from zero.

Fit one `(omega, alpha, beta)` per local or global model using only pre-2016 equity observations. Minimize `0.5 * sum_k sum_t [log(h[k,t]) + error[k,t]^2 / h[k,t]]`, where `h[k,t+1] = omega + alpha * error[k,t]^2 + beta * h[k,t]`, `omega > 0`, `alpha >= 0`, `beta >= 0`, and `alpha + beta < 1`. The running mean has no fitted parameter. Let `s` be the pooled mean of squared training residuals. Start optimization at `(0.1*s, 0.1, 0.8)` and each ticker's likelihood recursion at `max(mean of that ticker's squared training residuals, s*10^-8)`. During prediction, initialize `h[k,0] = omega / (1 - alpha - beta)`, save each day's forecast before incorporating that day's residual, and keep the fitted parameters fixed across equity and crypto evaluation.

## HARNet

Fit pooled HAR coefficients by ordinary least squares on the selected pre-2016 equity histories (or one selected ticker for a local fit), using at least 20 prior variance observations per target. Initialize HARNet through its existing HAR initializer before neural optimization. Positive input histories make the initialized network's output equal the fitted HAR forecast.

## Global roughness

Analyze `log(sqrt(Variance))` for Parkinson and Garman–Klass equity series. For each `q` in `{1, 1.5, 2, 3, 4}` and exact calendar lag `Delta` from 1 through 400 days, pool within-ticker absolute displacements `|log sigma[k,t+Delta] - log sigma[k,t]|^q`, requiring both dates before 2016-01-01. Regress `log(moment)` on `log(Delta)` for each `q` to get an intercept and slope `zeta(q)`, then estimate `H` through the origin-constrained fit `zeta(q) = H*q`.

Existing full-period global figures are under `imgs/roughness_analysis/global/full`; full-period moments, slopes, and H summaries stay under `imgs/roughness_analysis/csv`. Pre-2016 global equity scaling figures, H comparison, moments, per-`q` regressions, and H summaries are under `imgs/roughness_analysis/global/train`. Full-period H values describe the complete dataset; the training-only H values describe data available before model fitting and must be used for training comparisons.

The rebuilt pre-2016 results use 5,447,255 variance observations per equity estimator, ending on 2015-12-31. The global training H estimates are 0.037471 for Parkinson and 0.034440 for Garman–Klass (1,409,045,621 valid calendar-lag displacements per estimator).

| Equity estimator | Full-period H | Training H | Full-period H / training H |
| --- | ---: | ---: | ---: |
| Parkinson | 0.037454 | 0.037471 | 0.99954135 |
| Garman–Klass | 0.034976 | 0.034440 | 1.01554162 |

Ratios use the unrounded H values in the corresponding summary CSVs.

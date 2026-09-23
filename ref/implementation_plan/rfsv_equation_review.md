# RFSV Equation Review

## Implementation decisions (implemented)

1. Replace the six pre-2016 equity outputs in `imgs/roughness_analysis/global/train` with results from within-ticker observation lags 1 through 400. Keep the full-period calendar-lag analysis unchanged. Label the new training outputs as observation-lag results so old calendar-lag values cannot be reused by mistake.
2. For each estimator, take RFSV `H` from the recomputed pre-2016 global training roughness results. Global and local RFSV runs use the same estimator-specific `H`.
3. Estimate `nu^2 = exp(intercept)` from the free-intercept regression of `log(m(2,Delta))` on `log(Delta)` in those same training results. Their moments use `log(sigma) = 0.5*log(Variance)`, so no factor-of-four conversion applies. Keep the separately estimated `q = 2` slope and the multi-`q` estimate of `H` distinct.
4. Implement the paper's Section 5 forecast approximation for `Delta = 1` next observation, including its unnumbered Section 5.2 variance correction. Do not fit the stationary model's `alpha` or `m` from Equations (3.3)-(3.4).
5. Use the 20 most recent observations for each RFSV forecast, comparable to HAR's month of history. For `N` available values, map the newest to the integral bin `[0,1)`, then bins `[1,2)` through `[N-1,N)` for progressively older values; integrate each bin's kernel weight exactly.
6. Assign the remaining kernel mass on `[N,infinity)` to the oldest supplied log-volatility value. This finite-history extension is a project convention, not a rule stated by the paper. The resulting weights sum to one and preserve constant histories when `nu^2 = 0`.
7. Apply the existing training-derived lower floor to the final variance forecast. Record `H`, `nu^2`, the observation-lag convention, the 20-observation forecast window, and the training-result source in each RFSV fit; reject missing or incompatible training results.

## Paper equations (ASCII for PowerShell)

Source: [Gatheral, Jaisson, and Rosenbaum, *Volatility is rough*, Section 5](https://arxiv.org/pdf/1410.3394).

## Model definition versus forecast approximation

[Section 3.1 of the same paper](https://arxiv.org/pdf/1410.3394) defines the stationary log-volatility process `X_t` with the following explicit solution (Equation (3.3)) and volatility transformation (Equation (3.4)):

```text
Equation (3.3): X_t = nu * integral_(s=-infinity)^t
                    exp(-alpha*(t-s)) dW_s^H + m
Equation (3.4): sigma_t = exp(X_t), t in [0,T]
```

Here `W^H` is fractional Brownian motion, `H < 1/2`, and `nu > 0`, `alpha > 0`, and `m` are model parameters. The paper says that when `alpha` is small relative to `1/T`, `X_t` behaves locally like fractional Brownian motion. Its Section 5 forecast uses this approximation; Equation (5.1) is a forecasting formula derived from it, not Equation (3.3) solved or fitted directly. The current repository fits only `H` and uses the Section 5 forecast kernel. It does **not** fit or simulate the full stationary model (3.3)-(3.4). In particular, the code's `nu = 0` means it omits the optional variance correction; it is not a valid claim that a full model with `nu = 0` has been fitted.

The agreed implementation uses Section 5's forecast approximation. A fitted full stationary (3.3)-(3.4) model would require additional estimation and a separate conditional forecast.

## Section 5 forecast equations

Equation (5.1) in Section 5.1 gives the conditional mean of log variance. Here `sigma`, `H`, `Delta`, `t`, `s`, and `F_t` are the paper's variables:

```text
E[log(sigma_(t+Delta)^2) | F_t]
  = (cos(H*pi)/pi) * Delta^(H+1/2)
    * integral_(s=-infinity)^t log(sigma_s^2)
      / ((t-s+Delta)*(t-s)^(H+1/2)) ds
```

The same paper rewrites Equation (5.1) later in Section 5.1 using the variable `u`; this rewrite is **unnumbered**:

```text
E[log(sigma_(t+Delta)^2) | F_t]
  = (cos(H*pi)/pi) * integral_(u=0)^infinity
    log(sigma_(t-Delta*u)^2) / ((u+1)*u^(H+1/2)) du
```

The kernel is `K_H(u) = cos(H*pi) / (pi*(u+1)*u^(H+1/2))`, with total mass 1 for `0 < H < 1/2`. Its mass from `0` to `x` is the regularized incomplete beta function evaluated at `x/(1+x)`, with shape parameters `(1/2-H, 1/2+H)`. This identity allows exact observation-bin weights.

Section 5.2 gives the variance predictor below. It is **unnumbered**. The paper denotes the estimated log variance by a hat over `log(sigma_(t+Delta)^2)` and the variance predictor by a hat over `sigma_(t+Delta)^2`; `c` and `nu` are its symbols:

```text
c = Gamma(3/2-H) / (Gamma(H+1/2)*Gamma(2-2*H))
hat(sigma_(t+Delta)^2)
  = exp(hat(log(sigma_(t+Delta)^2)) + 2*c*nu^2*Delta^(2*H))
```

The project's forecast horizon is `Delta = 1` observation. Its current training path fixes `nu = 0`; the agreed change estimates `nu^2` from training moments. The observation-time adaptation is a project decision, not a claim that the paper uses an equity trading clock.

Section 5.2 estimates `nu^2` as the exponential of the intercept in a regression of `log(m(2,Delta))` on `log(Delta)`. The paper's `m(2,Delta)` is the second moment of **log-volatility** increments (Section 2.1):

```text
m(2,Delta) = E[(log(sigma_(t+Delta)) - log(sigma_t))^2]
log(m(2,Delta)) = log(nu^2) + 2*H*log(Delta)
nu^2 = exp(intercept)
```

The last two lines use the paper's local fractional-Brownian scaling approximation. They are not an empirical result for this project's proxies. If moments are instead computed from `log(V) = 2*log(sigma)`, their second moment is four times larger, so `nu^2 = exp(intercept)/4`. Any future estimate must use only pre-2016 fitting history and the agreed observation-lag clock; proxy smoothing and measurement noise can affect the intercept.

MLP and volatility LSTM outputs also represent log variance, but their training objective is QLIKE on `exp(z)` in variance units. For a positive observed target `V`, the conditional QLIKE optimum is `exp(z) = E[V | inputs]`, or `z = log(E[V | inputs])`. It is **not** generally `E[log(V) | inputs]`. Therefore the RFSV Section 5.2 correction should not be appended to these neural forecasts. It would only be relevant to a separate model explicitly trained to predict conditional mean log variance under an additional distributional assumption; that model is not part of the current implementation.

## Checked findings

1. **Issue 1, missing near-history mass and undefined finite-history tail.** `models/rfsv.py` uses midpoint values `u = 1.5, 2.5, ...`, omits `u` from `0` to `1`, and truncates the tail beyond the supplied history. For `H = 0.1` and 20 observations, current weights sum to `0.298133`; exact kernel mass from `0` to `20` is `0.917906`, of which `0.615908` lies in the first bin. Thus constant volatility `0.01` produces `0.253357`, rather than `0.01`. Simply starting midpoint samples at `u = 0.5` still approximates the near-zero integrable singularity poorly. A later implementation should integrate each observation bin and choose a tail convention. Extending the oldest observed log variance through the tail would preserve constant histories, but that is a **project assumption**, not specified by the paper.
2. **Issue 3, `nu` correction is half the paper's variance correction.** The code adds `c*nu^2/2` to log volatility and then squares, yielding `+c*nu^2` in log variance. Section 5.2 requires `+2*c*nu^2*Delta^(2*H)`, so the one-observation code is short by a factor of 2 when `nu > 0`. Current training uses `nu = 0`, so the factor error has no numerical effect there. However, this also means the current variance output is the exponentiated conditional-mean log variance, without the paper's positive conditional-variance correction. It should not be described as the full Section 5.2 variance predictor for a nondegenerate `nu > 0` model.

## Verification to perform during implementation

- Check observation-lag moments, ticker boundaries, the pre-2016 cutoff, and the `q = 2` intercept on a small known example. Regenerate and inspect the six training outputs without rebuilding variance tables or rerunning full-period analysis.
- Check that integrated bin weights plus the oldest-value tail sum to one; compare varying histories against a numerical reference integral, constant histories against their input value when `nu^2 = 0`, and `nu^2 > 0` against the Section 5.2 variance equation.
- Run focused and synthetic global/local RFSV checks, including fit reload, variance units, and rejection of incompatible training results. Obtain the required fresh-context cold review and validate its JSON.
- Numerical mass and constant-history values above were checked with `scipy.special.betainc` and the current `RFSV.forecast` on a 20-point constant history; they are equation diagnostics, not performance results.
- The pre-2016 global training roughness outputs and RFSV implementation now follow these decisions. No project-data RFSV fit or model comparison has been run.

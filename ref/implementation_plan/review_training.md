# Pre-training model and training review

Reviewed 2026-09-22 in fresh, independent passes against the current code,
[thesis notes](../thesis/thesis_notes.md), [thesis plan](../thesis/plan.md),
and implementation plans. **Hold full project-data training until findings 1
and 2 are resolved.** No project database or long training job was run.
The review covers the current working tree, including the uncommitted
training-loop move and gradient-clipping diagnostic.

## Model definitions and equations

| Model | Review result |
| --- | --- |
| AR(1) | Latest-variance linear forecast and pre-2016 within-ticker OLS match the project definition. |
| HAR | Separate 1/5/20 variance means and OLS match the definition. |
| SARIMA | Fixed (1,0,1) × (1,0,1,5) specification, pooled training likelihood, and fixed-parameter one-step filtering match the definition. Order suitability and real-data convergence remain untested. |
| GARCH(1,1) | Positive constrained recursion, pre-2016 likelihood, causal within-ticker return mean, and variance output match [training adjustments](training_adjustments.md), which supersede the older zero-mean plan. Crypto realized variance includes a preceding-day close while GARCH uses same-day Open/Close; transfer impact remains unverified. |
| RFSV | **Incorrect finite-history kernel; see findings 1, 3, and 4.** |
| MLP | Two SiLU hidden layers and scalar raw output match the definition; training can stall under the floor (finding 2). |
| Base LSTM | Explicit sigmoid gates, tanh candidate/output, and scalar raw output match the definition; training can stall under the floor (finding 2). The earlier return model remains separate. |
| SiLU-LSTM | Sigmoid gates and SiLU candidate/cell output match the definition; training can stall under the floor (finding 2). |
| HARNet-20 / HARNet-80 | Trailing 1/5/20 and 1/5/20/40/80 averages, fitted-HAR initialization, and separate checkpoint paths match the definitions on positive histories. |

Parkinson and reduced Garman–Klass formulas, ticker-safe next-observation
windows, chronological splits, training-only floor, QLIKE, MAE, MASE, MSE,
and RMSE match their current [loss plan](losses.md). This checks the formulas
and synthetic behavior, not empirical forecast quality. The [HARNet paper](https://arxiv.org/abs/2205.07719),
[rough-volatility notes source](https://arxiv.org/abs/2312.01426),
[crypto comparison source](https://arxiv.org/abs/2311.04727),
[Hyndman and Koehler](https://robjhyndman.com/papers/mase.pdf), and
[Patton](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
are already referenced in the thesis notes. The explicit RFSV forecast check
also required [Gatheral, Jaisson, and Rosenbaum, §5](https://arxiv.org/pdf/1410.3394),
which is listed in the thesis bibliography.

## Findings

1. **High — RFSV forecast scale is wrong.** `models/rfsv.py:17-22` starts its midpoint kernel at 1.5 rather than 0.5 and drops the unobserved tail without a level convention. The cited equation integrates from zero over past time. With H = 0.1, ν = 0, and 20 constant volatility observations of 0.01, the implementation forecasts 0.25336 rather than 0.01; squaring gives a variance forecast near 0.064 instead of 0.0001. Correct the near-history bin and finite-history treatment, then test constant-level and varying histories against the cited equation.
2. **High — the forecast floor can stop neural learning; addressed in [the adjustment](second_issue_review_adjustment.md).** MLP and both LSTMs now output log variance, initialize near the training median, and have nonzero gradients in synthetic small-variance checks. Both HARNets retain their fitted-HAR variance output and floor; their per-epoch floor-hit percentage is logged. Project-data behavior remains unverified.
3. **Medium — RFSV optional ν correction is half the paper's variance correction.** `models/rfsv.py:21` adds cν²/2 to log volatility; `models/variance_fit.py:251` squares it, giving +cν² in log variance. [Gatheral et al., §5.2](https://arxiv.org/pdf/1410.3394) gives +2cν² for the one-step variance forecast. The current training path fixes ν = 0, so this does not affect its present forecasts.
4. **Medium — RFSV uses inconsistent time units.** `models/variance_fit.py:156-170` fits H with exact calendar-day lags, while `models/rfsv.py:17-20` weights equally spaced observation lags. Weekend and holiday gaps therefore do not follow the cited time-based kernel. Choose and document a consistent time convention before comparing RFSV with other models.
5. **Medium, separate crypto RV path — completeness check misses irregular bars.** `data/crypto_data_fetching.py:319-340` accepts 96 clean rows without validating every quarter-hour slot. A synthetic day with 12:00 replaced by 12:07 was accepted as complete. This affects crypto realized variance, which is a separate target from the current Parkinson/Garman–Klass training paths.
6. **Low — reproducibility and plan text need alignment.** Neural checkpoints omit fixed seed, width, learning rate, patience, clip norm, and initialization rule from saved settings (`models/variance_neural.py:50-54`). `train_framework.md` still says HARNet starts randomly, although [training adjustments](training_adjustments.md) supersede that choice. Record effective settings and reconcile the stale decision before reporting experiments.

## Verification and limits

Fresh reviewers ran `.venv/Scripts/python.exe -m models.test_models`,
`-m models.test_training_smoke`, direct GARCH/HARNet training-adjustment checks,
the variance-metric and roughness synthetic checks, a checkpoint reload check,
and targeted counterexamples; these passed except where the counterexamples
demonstrated the findings. No project-data eligibility, neural convergence,
SARIMA/GARCH optimizer behavior on real histories, crypto source regularity,
or out-of-sample model ranking was verified. QLIKE's interpretation as a
latent-variance comparison still depends on the proxy assumptions described
by Patton; those assumptions have not been established for these targets.

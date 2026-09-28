# Sigma-LSTM Implementation Plan

## Implementation decisions

### Agreed

1. Use the existing 2,714-ticker raw adjusted-OHLC equity cohort for the eventual GK experiment: more than 80 pre-2016 rows and a row on 2025-12-31. This defines the population, not model eligibility.
2. Define daily unannualized adjusted Garman–Klass variance as V(t) = 0.5 ln(High(t)/Low(t))² − (2 ln 2 − 1) ln(Close(t)/Open(t))². The target remains next-observation log volatility ℓ(t) = 0.5 ln V(t); a valid zero target uses the existing positive floor selected from pre-2016 targets before taking its logarithm. Input windows require positive valid GK variance, as in the raw neural path.
3. Use one scalar z-scored GK log volatility ℓ(t) per observation. Do not use returns as an input to this variant. Sort by (Ticker, Date), keep windows within ticker, and predict the next recorded observation from the preceding window.
4. Fit separate window-20 and window-80 versions with hidden width 128. Target dates split into training before 2016-01-01, validation from 2016-01-01 through 2018-12-31, and test from 2019-01-01 through 2025-12-31.
5. `SigmaLSTMCell` inherits `FEBCellLSTM`, retaining sigmoid input/forget gates, tanh candidate, and the additive memory update from the paper's Equations (4)–(7). Initialize h(0) = 0 and every component of C(0) = 1; the paper does not specify this initialization.
6. Replace the base output gate with a zero-mean Gaussian draw. Its per-component gate variance is softplus(Wₒ[C(t)²]); sample a standard normal tensor and multiply by the square root of that variance plus the dtype's smallest positive normal value. This numerical addition prevents NaN gradients if float32 softplus underflows to zero at an extreme negative score; it is negligible at ordinary scores. Softplus passes a gradient for finite negative scores until floating-point underflow. This map is our choice for the paper's Equation (8), not part of its printed equation. Use tanh for the unspecified φ in Equation (9). It replaces the exp map that overflowed and the ReLU map whose run was stopped at epoch 12 by user request.
7. `SigmaLSTM` inherits `FEBLSTM`. Interpret Wₕh(t) as the next-observation conditional log-volatility estimate. Keep [mean of C(t)]² as the auxiliary variance of log volatility, an adaptation of the paper's Equation (11); it is not the GK variance forecast and has no calibrated uncertainty claim yet.
8. Convert the log-volatility estimate to a GK variance forecast using V̂(t + 1) = exp(2ℓ̂(t + 1)). Apply the existing training-target-derived positive prediction floor before QLIKE or variance metrics, with no upper cap. Train with QLIKE in GK variance units, not the paper's return likelihood: mean[V(t + 1)/V̂(t + 1) − ln(V(t + 1)/V̂(t + 1)) − 1].
9. Use random seed 42 for model initialization, training, and stochastic evaluation. Do not reset it inside each forward call.
10. Compare the 20- and 80-observation versions only on shared eligible target dates, using the same target, split, and metric definitions. The existing raw GK scan reports 8,664,516 / 1,806,548 / 4,479,681 window-20 and 7,004,011 / 1,692,154 / 4,295,773 window-80 train / validation / test windows; the sigma-LSTM loader must reproduce these counts before a controlled comparison.

### Training and evaluation decisions

11. Use one stochastic forward pass per evaluation batch, with seed 42 reset at the start of each epoch's evaluation and again for final checkpoint validation. Preserve training RNG state around epoch evaluation so scoring does not change the training draws or shuffle order. The same checkpoint, observation order, batch size, and environment should reproduce the pass; no Monte Carlo average is used.
12. Agreed: The auxiliary [mean of C(t)]² output is not a training focus. Use no separate auxiliary loss or calibration step. QLIKE supervises only the GK variance forecast from the log-volatility head, and the auxiliary output must not be presented as calibrated uncertainty.
13. Fit one global mean and population standard deviation (ddof = 0) of GK log volatility on positive valid pre-2016 observations from the raw cohort, counting each row once. Standardize inputs as (ℓ(t) − mean)/standard deviation; use the same fitted values for train, validation, test, and checkpoint inference without clipping out-of-range values.
14. Use hidden width 128, Adam learning rate 0.001, batch size 128, at most 20 epochs, and QLIKE training/validation checkpoint selection.
15. Use patience 10 with the inherited one-time tenfold learning-rate retry. Disable gradient clipping for sigma-LSTM, matching the Base LSTM validation runs; this does not address the observed output-gate exponential overflow.
16. Use a learned bias on the log-volatility output head initialized to half the natural logarithm of the pre-2016 median GK variance; shrink its initial weights by 0.01. This project adaptation allows the stochastic zero-mean gate to represent the negative level of daily log volatility.
17. Use compiled CUDA when available and online W&B logging. Run window 20 first, then start window 80 only if window 20 exits successfully. Save separate checkpoints and console logs.
18. Score validation only after each run's best checkpoint. Do not score test data in this training iteration.
19. For the one-off no-clipping failure diagnostic, enable `--memory-diagnostic` on window 20 and log the minimum, maximum, and mean of finite `memory_t` entries over every time step, observation, and hidden component in the failing batch, plus the nonfinite count and first affected time step. This trace does not change the loss or optimizer update; record its online W&B run separately from validation trials.
20. Keep the output and target in log-volatility units; do not scale them. Convert the head output to GK variance with exp(2 × output) for QLIKE, so no inverse target transform is needed.
21. Save the log-volatility input mean, population standard deviation, and training-row count in the checkpoint and W&B configuration. The full-cohort preflight measured 11,153,536 positive valid pre-2016 rows, mean -4.17238373979845, and population standard deviation 0.8209776735142315. Previous min-max extrema do not specify the z-score distribution.
22. Changing only the input values preserves the existing window eligibility and target dates: verified train/validation/test counts remain 8,664,516 / 1,806,548 / 4,479,681 for window 20 and 7,004,011 / 1,692,154 / 4,295,773 for window 80. It changes model inputs, so scores from earlier unscaled-input runs are not an isolated architecture comparison.

## Observed result under the current settings

The z-scored log-volatility-input, window-20 sigma-LSTM completed 20 epochs with hidden width 128, Adam learning rate 0.001, batch size 128, seed 42, no gradient clipping, and variance-unit QLIKE. Its best checkpoint was epoch 10. The final validation pass on 1,806,548 forecasts reported QLIKE **1.2217863932196134** and MASE **2.116889268968172** ([W&B run](https://wandb.ai/personalfeb/timeseries-volatility/runs/aqezti5x); [local log](../../wandb/sigma_lstm_val_training/logs/sigma_lstm_raw_gk_zscorelogvol_w20_h128_lr0p001_20e_qlike_softplus_noclip_online.log)). Lower QLIKE is better and zero is perfect. MASE here is mean absolute error divided by each ticker's positive training-history naive scale, so 2.1169 means the model's validation absolute error averaged about 2.12 times that scale. Validation QLIKE changed only from 1.2220259128882138 after epoch 1 to 1.2217863924241605 at the best epoch.

These values justify describing **this configuration's validation performance as weak**. They do not establish that sigma-LSTM generally underperforms: a naive forecast has not been scored on these exact validation observations, the window-80 z-score run has not been completed, and test data have not been scored.

## Model and data work

- `models/sigma-lstm.py`: retain the additive cell with C(0) = 1 and softplus gate variance, and expose its log-volatility head and auxiliary output through the shared model factory.
- `models/variance_neural.py` and `models/variance_fit.py`: reuse raw adjusted GK validity and ticker-safe windows; fit and persist train-only z-score statistics for the single log-volatility input, while taking only the unchanged log-volatility head for exp(2 × head), prediction flooring, QLIKE, checkpointing, and validation. Keep the auxiliary output separate.
- `wandb/sigma_lstm_val_training/run_queue.py`: launch and log the two online W&B runs in order, stopping the queue if window 20 fails.

## Verification

- Run the focused synthetic cell check: inherited gates, C(0) = 1, finite nonnegative gate variance, deterministic replay with seed 42, output shapes, Equation (11), finite nonzero gradients for moderate negative and positive gate scores, and finite gradients at an extreme negative score must hold.
- Reuse the existing tiny known GK formula fixture; check train-only z-score input statistics, unchanged log-volatility target/output, variance conversion, QLIKE gradient, exact window counts, finite positive targets, ticker boundaries, and date alignment before training.
- Obtain a fresh-context JSON cold review and validate it with `python judge/validate.py` before launch. Do not download data, score test data, commit, or push.

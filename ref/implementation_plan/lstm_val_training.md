# Garman–Klass LSTM validation training

## Raw-history SiLU MSE correction (2026-09-25)

The user corrected the raw-variance SiLU trial to train with MSE on adjusted
daily Garman–Klass variance: mean((ŷ − y)²), where ŷ is the network's direct
raw-variance output and y is the adjusted raw-history variance target. This
choice gives larger absolute variance errors more weight through squaring;
it is an experiment decision, not an established improvement. The raw output
may be negative, so apply the existing training-derived lower floor only when
forming positive variance forecasts for the five reported metrics. Training
MSE uses the unfloored output; checkpoint selection and patience use validation
MSE on floored forecasts. Continue reporting MAE, MASE, MSE, RMSE, and QLIKE
in variance units. The raw-history cohort,
window-20 validity rule, floor, ln(adjusted Close/Open) second feature, 20
epoch cap, width 128, Adam 0.001, batch 128, patience 10, seed 42, clipping
norm 1, compiled CUDA, and validation-only scoring are unchanged. The prior
W&B run `9znlrmc8` used QLIKE by mistake and was stopped during epoch 1 with
no completed epoch or checkpoint. The MSE run must have a new run name and
checkpoint. An initial MSE attempt with an exponential log-variance head,
W&B run `wajx25z9`, failed at batch 1,576: a finite log output of 173.399
gave a finite loss of 3.2017 × 10¹⁴⁸ but nonfinite parameter gradients.
The direct raw-variance head replaces that failed configuration; no softplus
or log-variance target is used. The cited crypto-winter paper uses MSE on
volatility rather than this trial's variance target, so this is not a direct
replication.

## Raw-variance consecutive-session trial (2026-09-24)

Requested next trial: base LSTM, 20 previous Garman–Klass variance observations
and ln(adjusted Close/Open) as two ordered inputs, next-session raw positive
Garman–Klass variance target, hidden width 128, Adam initial rate 0.001,
batch 128, seed 42, at most 20 epochs, patience 10 with the existing one-rate
retry, no gradient clipping, compiled CUDA, validation QLIKE checkpoint
selection, and no test-period scoring. The model outputs log variance and
exponentiates it for variance-unit QLIKE and reported metrics.

The source database is unchanged. `--consecutive-sessions` excludes any
20-input-plus-target window crossing a missing ticker date in the panel's
6,287-date observed market calendar. This excludes 661,479 of 5,416,755
former training windows (12.21%), 69,335 of 1,120,921 validation windows
(6.19%), and 105,290 of 2,649,257 test windows (3.97%). Retained actual
variance values are unchanged. Earlier unfiltered run metrics are not directly
comparable because their observation sets differ. A compiled two-ticker
one-epoch preflight passed; its checkpoint is separate and is not a reported
full-panel result. The full-panel trial has not yet started.

Daily unannualized reduced Garman–Klass variance from adjusted OHLC:
0.5 ln(High/Low)² − (2 ln 2 − 1) ln(Close/Open)². Forecast the next
observation from the previous
20, 40, or 80 within-ticker values. Full 1,525-equity panel; pre-2016 fit,
2016–2018 validation selection. The 2019–2025 test period is not evaluated.
Seed 42, batch 128, QLIKE checkpoint selection, one learning-rate retry.
Original sweep: up to 20 epochs and patience 5. Selected trials use
patience 10 before and after one rate retry. Hidden-16 trials allow up to
50 epochs; raw hidden-64 was extended from 10 to 20, and log-volatility
hidden-64 and single-input hidden-128 trials allow up to 20 epochs;
selected two-input hidden-128 runs allow up to 50 epochs.
`_logvol_` runs use log(sqrt(variance)) inputs and next-day targets.
_intradayret_ LSTM runs add ln(adjusted Close/Open) as the second input.
Their QLIKE objective and reported errors reconstruct variance with exp(2*target).
The original sweep uses gradient norm cap 1; `_noclip` runs disable clipping.
Each metric below comes from the best validation-QLIKE epoch. Lower is better.

| Run | Status | Epochs | Best epoch | QLIKE | MSE | RMSE | MASE | MAE | W&B | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| base_lstm_vol_gk_w20_h16_lr0.001 | complete | 20 | 17 | 0.490038994 | 0.00272631743 | 0.0522141498 | 0.940150411 | 0.000573647212 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/9s1bjp2u) | [base_lstm_vol_gk_w20_h16_lr0.001.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.001.log) |
| base_lstm_vol_gk_w20_h16_lr0.0001 | interrupted | 5 | — | — | — | — | — | — | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/xqmw6910) | [base_lstm_vol_gk_w20_h16_lr0.0001.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001.log) |
| base_lstm_vol_gk_w20_h16_lr0.0001_noclip | interrupted | — | — | — | — | — | — | — | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/26cqhn7k) | [base_lstm_vol_gk_w20_h16_lr0.0001_noclip.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip.log) |
| base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress | interrupted | 13 | 13 | 0.52933654 | 4.2952121e-05 | 0.00655378676 | 0.57909911 | 0.000560915578 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/kezb18pv) | [base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress_resume_compiled_eval.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress_resume_compiled_eval.log) |
| base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress | interrupted | 7 | 5 | 0.484898549 | 3.2502268e-05 | 0.00570107604 | 0.534934573 | 0.00049110775 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/9eq5x5iw) | [base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress.log) |
| base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs | complete | 20 | 20 | 0.515175555 | 1.8392491 | 1.35618918 | 3.75433579 | 0.00480518443 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/wr52wy0r) | [base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs_resume_to20.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs_resume_to20.log) |
| base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs | interrupted | 0 | None | — | — | — | — | — | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/sp26pqr0) | [base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs.log) |
| base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs_rerun | complete | 20 | 19 | 0.475532444 | 3.21442822e-05 | 0.00566959277 | 0.530444171 | 0.000487556407 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/qbvvsiwy) | [base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs_rerun.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs_rerun.log) |
| base_lstm_vol_gk_logvol_w20_h128_lr0.0001_noclip_progress_20epochs | complete | 20 | 18 | 0.474608783 | 3.21410957e-05 | 0.00566931175 | 0.511770506 | 0.00046998702 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/hx9jl4m2) | [base_lstm_vol_gk_logvol_w20_h128_lr0.0001_noclip_progress_20epochs.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h128_lr0.0001_noclip_progress_20epochs.log) |
| base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs | interrupted | 0 | None | — | — | — | — | — | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/vm0hppa8) | [base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_interrupted_epoch1.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_interrupted_epoch1.log) |
| base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_rerun | complete | 50 | 43 | 0.469193006 | 3.23347222e-05 | 0.00568636282 | 0.516666573 | 0.000473455604 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/48e1ho51) | [base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_rerun_resume_to50.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_rerun_resume_to50.log) |
| base_lstm_vol_gk_logvol_intradayret_w20_h4_lr0.0001_noclip_progress_20epochs | complete | 30 | 27 | 0.479066905 | 3.22833252e-05 | 0.00568184171 | 0.530850693 | 0.000492148398 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/fx8nhaho) | [base_lstm_vol_gk_logvol_intradayret_w20_h4_lr0.0001_noclip_progress_20epochs_resume_to30.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h4_lr0.0001_noclip_progress_20epochs_resume_to30.log) |
| base_lstm_vol_gk_logvol_intradayret_w20_h64_lr0.0001_noclip_progress_20epochs | complete | 20 | 19 | 0.473162663 | 3.23554548e-05 | 0.00568818555 | 0.524930105 | 0.000480496376 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/x8t7waqe) | [base_lstm_vol_gk_logvol_intradayret_w20_h64_lr0.0001_noclip_progress_20epochs.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h64_lr0.0001_noclip_progress_20epochs.log) |
| silu_lstm_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_50epochs | failed scoring epoch 18: train MSE overflow | 17 on latest attempt; 16-19 from original attempt superseded | 15 | 0.4740026653 | 0.001849031679 | 0.04300036835 | 0.5482458493 | 0.000534005571 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/qjefnxj6) | [continuation log](../../wandb/lstm_val_training/logs/silu_lstm_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_50epochs_resume_float64_from15_to50.log) |

## Incomplete or failed runs

- base_lstm_vol_gk_w20_h16_lr0.0001: Stopped during epoch 6 to investigate high clipping and slow convergence; partial checkpoint is not a completed trial.
- base_lstm_vol_gk_w20_h16_lr0.0001_noclip: Stopped to restart with live batch and epoch console metrics; partial checkpoint is not a completed trial.
- base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress: Stopped at user-requested epoch 13 boundary; best epoch 13, validation QLIKE 0.5293365402; partial epoch 14 discarded. SiLU handoff cancelled.
- base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress: Stopped at user request during epoch 8; partial epoch discarded.
- base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs: Stopped at user request during epoch 1; elapsed 264.3 seconds.
- base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs: stopped during epoch 1 after temporary cap-change request; no checkpoint

## Validation residual histogram

For the two-input hidden-128 best checkpoint (epoch 19), AAPL has 754
validation forecasts in 2016-2018. Residual is actual Garman-Klass
variance minus predicted variance. The [histogram](../../imgs/lstm_validation/aapl_h128_intradayret_validation_residual_histogram.png)
shows all residuals and a labeled central-98% zoom; the dated
[CSV](../../imgs/lstm_validation/aapl_h128_intradayret_validation_residuals.csv)
contains actual variance, predicted variance, and residual. This is a
single-ticker diagnostic, not a full-panel result.

## Global validation residual histogram

For the two-input hidden-4 best checkpoint (epoch 27 of 30), the 2016-2018
validation set has 1,120,921 forecasts across 1,525 tickers. Residual is
actual minus predicted daily unannualized Garman-Klass variance. The
[histogram](../../imgs/lstm_validation/global_h4_intradayret_validation_residual_histogram.png)
shows the full distribution on a log count axis and a central-98% zoom
(1st-99th percentile).
The [dated residuals](../../imgs/lstm_validation/global_h4_intradayret_validation_residuals.csv.gz)
include ticker, date, actual variance, predicted variance, and residual.
Mean residual is -1.14778e-05 and median residual is -6.64878e-05.
No test-period observations were used.

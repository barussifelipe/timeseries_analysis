# Garman–Klass LSTM validation training

Daily unannualized reduced Garman–Klass variance from adjusted OHLC:
0.5 ln(High/Low)² − (2 ln 2 − 1) ln(Close/Open)². Forecast the next
observation from the previous
20, 40, or 80 within-ticker values. Full 1,525-equity panel; pre-2016 fit,
2016–2018 validation selection. The 2019–2025 test period is not evaluated.
Seed 42, batch 128, QLIKE checkpoint selection, one learning-rate retry.
Original sweep: up to 20 epochs and patience 5. Later selected runs:
up to 50 total epochs and patience 10 before and after the retry, except
the raw-variance hidden-64 trial first capped at 10, then extended to 20.
`_logvol_` runs use log(sqrt(variance)) inputs and next-day targets.
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
| base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs | running | 10 | 10 | 0.533365959 | 0.0286546039 | 0.169276708 | 0.686777729 | 0.00122997526 | [run](https://wandb.ai/personalfeb/timeseries-volatility/runs/wr52wy0r) | [base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs_resume_to20.log](../../wandb/lstm_val_training/logs/base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs_resume_to20.log) |

## Incomplete or failed runs

- base_lstm_vol_gk_w20_h16_lr0.0001: Stopped during epoch 6 to investigate high clipping and slow convergence; partial checkpoint is not a completed trial.
- base_lstm_vol_gk_w20_h16_lr0.0001_noclip: Stopped to restart with live batch and epoch console metrics; partial checkpoint is not a completed trial.
- base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress: Stopped at user-requested epoch 13 boundary; best epoch 13, validation QLIKE 0.5293365402; partial epoch 14 discarded. SiLU handoff cancelled.
- base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress: Stopped at user request during epoch 8; partial epoch discarded.
- base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs: Resumed from best epoch 10; no superseded epochs.

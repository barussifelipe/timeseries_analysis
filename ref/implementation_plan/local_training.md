# Local Volatility Training Implementation Plan

## Implementation decisions

1. Select the five eligible raw-history equities with the highest mean nonnegative daily volume before 2016, retaining only one share class per company: NVDA, AAPL, NFLX, GOOG, and AMZN. GOOG outranks GOOGL; AMZN is next after the duplicate share class.
2. Use adjusted daily, unannualized Garman-Klass variance: `0.5 * ln(High/Low)^2 - (2 * ln(2) - 1) * ln(Close/Open)^2`. The target is the next recorded observation within the same ticker. Valid zero-variance targets use the minimum positive pre-2016 variance of that ticker; input windows require valid positive raw variance.
3. Fit each ticker independently. Training targets end before 2016-01-01; validation targets run from 2016-01-01 through 2018-12-31; test targets run from 2019-01-01 through 2025-12-31. Validation and test windows may use earlier observed inputs. No test targets enter fitting or checkpoint selection.
4. Fit Base LSTM with 20 and 80 input observations separately for each of the five tickers (ten fits). Use GK log volatility and adjusted `ln(Close/Open)` as the two inputs, log variance as the model output, and variance-unit QLIKE as the training and validation objective. Use seed 42, hidden width 128, Adam initial learning rate 0.001, batch size 128, gradient norm cap 1, compiled CUDA when available, patience 10 with the existing one-time tenfold learning-rate reduction, and at most 20 epochs. Save the best validation-QLIKE checkpoint.
5. After the Base LSTM group, fit HARNet-20 and HARNet-80 separately for all five tickers (ten more fits). HARNet's existing single input is raw adjusted GK variance; its fixed one-channel architecture has no configurable hidden width, so the requested width 128 applies to Base LSTM only. Initialize each HARNet from its ticker's pre-2016 HAR coefficients, then train with variance-unit QLIKE, seed 42, Adam 0.001, batch 128, gradient norm cap 1, compiled CUDA when available, patience 10 with the same learning-rate retry, and at most 20 epochs.
6. For the next requested group, fit MLP with 20 and 80 input observations separately for the same five tickers (ten fits). Match the completed global MLP runs: GK log volatility and adjusted `ln(Close/Open)` inputs, log-variance output, variance-unit QLIKE, seed 42, hidden width 128, Adam 0.001, batch 128, gradient norm cap 1, compiled CUDA when available, patience 10 with the one-time learning-rate retry, and at most 20 epochs. Save the best validation-QLIKE checkpoint.
7. For the now-requested SiLU-LSTM group, fit windows 20 and 80 separately for all five tickers (ten fits) with the saved global configuration: raw adjusted GK variance and adjusted `ln(Close/Open)` inputs, direct variance output, unfloored variance-unit MSE for training, seed 42, hidden width 128, Adam 0.001, batch 128, gradient norm cap 1, compiled CUDA when available, patience 10 with the one-time learning-rate retry, and at most 20 epochs. Apply the ticker's training-derived positive floor to validation forecasts when selecting the best validation-MSE checkpoint, as in the global run; do not cap the raw training output.
8. Use one online W&B run and one local checkpoint per model/ticker/window. Include the ticker in each run name: `base_lstm_local_<TICKER>_raw_gk_logvol_return_w<WINDOW>_h128_lr0p001_20e_qlike`, `harnet_<WINDOW>_local_<TICKER>_raw_gk_variance_w<WINDOW>_lr0p001_20e_qlike`, `mlp_local_<TICKER>_raw_gk_logvol_return_w<WINDOW>_h128_lr0p001_20e_qlike`, or `silu_lstm_local_<TICKER>_raw_gk_variance_return_direct_mse_w<WINDOW>_h128_lr0p001_20e`. Save the source database size and modification time in each checkpoint. Preserve global artifacts. Run only the requested model groups; later groups await separate requests.
9. Base LSTM, MLP, and HARNet use QLIKE; SiLU-LSTM uses MSE. Windows 20 and 80 apply where supported. Defer all post-fit test forecasts and model-comparison scores until the scoring request; validation loss during training is necessary for checkpoint selection.
10. These local fits use the same fixed population and source database version as the global raw-history experiments, but each has fewer windows and its own training-derived floor. Later comparison must rescore global and local checkpoints on identical ticker/date keys before averaging five ticker losses into one row per model.

## Eligible target windows

Counts use `load_raw_neural` and `TimeSeriesDataset(..., valid_column='Valid')` on `D:/DBs/timeseries_analysis/history_coverage.db`; they include only valid targets following 20 or 80 consecutive valid positive raw-variance input observations within a ticker. All five have a 2025-12-31 row. These are target-window counts, not raw-row counts.

| Ticker | Train 20 | Val 20 | Test 20 | Train 80 | Val 80 | Test 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVDA | 4,223 | 754 | 1,760 | 4,103 | 754 | 1,760 |
| AAPL | 8,408 | 754 | 1,760 | 7,915 | 754 | 1,760 |
| NFLX | 3,407 | 754 | 1,760 | 3,347 | 754 | 1,760 |
| GOOG | 2,843 | 754 | 1,760 | 2,783 | 754 | 1,760 |
| AMZN | 4,669 | 754 | 1,760 | 4,609 | 754 | 1,760 |
| **Total** | **23,550** | **3,770** | **8,800** | **22,757** | **3,770** | **8,800** |

## Dataset and implementation work

- Reuse the raw-history local neural trainer and fit-only CLI option that skips its separate post-training validation/test prediction pass. Training still measures each model's configured validation loss at every epoch and saves the best checkpoint.
- Verify the five selected tickers and the counts above before launching. The Base LSTM, HARNet, and MLP groups are complete; run the separately requested SiLU-LSTM group with distinct names, logs, and checkpoints. Stop its queue on failure rather than substituting a different ticker or configuration.

## Verification

- Check that fit-only mode does not invoke post-fit prediction, and that local checkpoint metadata identifies its ticker, window, training settings, counts, floor, and source database.
- Run a focused check before full fitting and obtain the repository's fresh-context JSON cold review for implementation changes. Do not score the test set or update the comparison table in this step.

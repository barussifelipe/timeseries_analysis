# Local Volatility Training Implementation Plan

## Implementation decisions

1. Select the five eligible raw-history equities with the highest mean nonnegative daily volume before 2016, retaining only one share class per company: NVDA, AAPL, NFLX, GOOG, and AMZN. GOOG outranks GOOGL; AMZN is next after the duplicate share class.
2. Use adjusted daily, unannualized Garman-Klass variance: `0.5 * ln(High/Low)^2 - (2 * ln(2) - 1) * ln(Close/Open)^2`. The target is the next recorded observation within the same ticker. Valid zero-variance targets use the minimum positive pre-2016 variance of that ticker; input windows require valid positive raw variance.
3. Fit each ticker independently. Training targets end before 2016-01-01; validation targets run from 2016-01-01 through 2018-12-31; test targets run from 2019-01-01 through 2025-12-31. Validation and test windows may use earlier observed inputs. No test targets enter fitting or checkpoint selection.
4. For the current request, fit Base LSTM with 20 and 80 input observations separately for each of the five tickers (ten fits). Use GK log volatility and adjusted `ln(Close/Open)` as the two inputs, log variance as the model output, and variance-unit QLIKE as the training and validation objective. Use seed 42, hidden width 128, Adam initial learning rate 0.001, batch size 128, gradient norm cap 1, compiled CUDA when available, patience 10 with the existing one-time tenfold learning-rate reduction, and at most 20 epochs. Save the best validation-QLIKE checkpoint.
5. Use one online W&B run and one local checkpoint per model/ticker/window. Include the ticker in the run name: `base_lstm_local_<TICKER>_raw_gk_logvol_return_w<WINDOW>_h128_lr0p001_20e_qlike`. Preserve global artifacts. Run only this requested Base LSTM group; later model groups await separate requests.
6. Future local SiLU-LSTM fits use direct variance output and MSE training loss. Other requested neural fits use QLIKE. Windows 20 and 80 apply where supported. Defer all post-fit test forecasts and model-comparison scores until the scoring request; validation QLIKE during training is necessary for checkpoint selection.
7. The ten local fits use the same fixed population and source database version as the global raw-history experiments, but each has fewer windows and its own training-derived floor. Later comparison must rescore global and local checkpoints on identical ticker/date keys before averaging five ticker losses into one row per model.

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

- Reuse the raw-history local neural trainer and add a fit-only CLI option that skips its separate post-training validation/test prediction pass. Training still measures validation QLIKE at each epoch and saves the best checkpoint.
- Verify the five selected tickers and the counts above before launching. Run the ten requested fits with distinct run names, logs, and checkpoint paths, recording failures rather than substituting a different ticker or configuration.

## Verification

- Check that fit-only mode does not invoke post-fit prediction, and that local checkpoint metadata identifies its ticker, window, training settings, counts, floor, and source database.
- Run a focused check before full fitting and obtain the repository's fresh-context JSON cold review for implementation changes. Do not score the test set or update the comparison table in this step.

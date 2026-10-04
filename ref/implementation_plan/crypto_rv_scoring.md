# Crypto RV Transfer Scoring Implementation Plan

## Implementation decisions

1. Keep BTC, ETH, SOL, XRP, and DOGE and each coin's 1,760 retained dates from the earlier GK evaluation. The retained `crypto_realized_variance` and `crypto_garman_klass_variance` keys must agree exactly. This preserves 8,800 target keys per configuration, at both windows.
2. Use daily unannualized realized variance from the existing 15-minute history: the sum of 96 squared close-to-close log returns within each complete UTC day, including the return from the preceding day's 23:45 close to 00:00. Require 96 complete valid bars, finite positive RV, and the retained table value on every target key. Do not recalculate targets from daily OHLC.
3. Feed preceding valid RV observations, sorted by ticker and date, into the saved equity-trained GK global fits. Keep their parameters, transforms, model-specific positive floors, daily OHLC intraday-return feature where used, and 20/80 input lengths fixed. This is a joint transfer across asset class and input/target estimator, not a newly fitted RV model.
4. Omit raw daily rows without valid RV from input series. This preserves all 1,760 target dates per coin; a forecast is for the next recorded valid RV observation. The read-only preflight found 64--65 scored targets per coin whose immediately preceding valid RV record is more than one calendar day earlier; the largest gap is three days. Requiring 80 consecutive raw daily rows with RV would leave only 798--800 targets per coin. No future values enter a window.
5. Use the existing test-date naive per-coin MASE scale, MAE, MASE, MSE, RMSE, QLIKE, and floor-hit definitions in daily variance units. Compute the naive change against the preceding valid RV observation. Leave positive actual RV unchanged. Reject nonfinite forecasts and nonpositive scales.
6. Keep this run separate from GK scores and equity rankings. Compare the two crypto runs only on the verified matching ticker/date keys. Differences combine the change in input proxy and the change in target proxy, so they do not isolate estimator quality. Training and validation populations and fit artifacts remain unchanged.
7. Preserve the stored RV despite a source-feed inconsistency: retained 15-minute closes exceed the vendor daily high or fall below its daily low on 22 BTC, 12 ETH, 72 SOL, 95 XRP, and 27 DOGE target dates (123 dates differ by more than 1%). SOL has 24 RV targets above 1 and a maximum of 6.90, whereas its GK maximum is 0.143. Record these diagnostics in the run log and treat cross-estimator rankings as descriptive until the feeds are reconciled; changing the retained targets would break the requested matched population.

## Dataset and implementation work

- Add `inference/evaluate_crypto_rv.py` reusing the saved-fit checks, target selector, scoring functions, aggregation, metric table, and residual plots. Write `imgs/eval/crypto_rv_eval/metrics.csv`, `metrics.png`, `per_stock.csv`, `run.log`, and 90 residual plots. Expect 15 aggregate rows and 75 per-coin audit rows.
- Record database size and modification time, fit paths and floors, target span, eligible RV-history counts, missing raw-day exclusions, and target gap counts in the run log.

## Verification

- On a small known 15-minute example, check the UTC 96-return RV sum and prior-close boundary. Check target table equality, ticker isolation, prior-only windows, and target count/date equality against the GK evaluation.
- Reconcile every aggregate row with its five audit rows and residual count; inspect the table and representative plots. Obtain a fresh-context JSON cold review using `judge/prompt.md` and validate it with `python judge/validate.py`.

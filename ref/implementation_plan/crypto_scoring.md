# Crypto Global Scoring Implementation Plan

## Implementation decisions

1. Select five assets by the mean of daily close × raw volume on each asset's 1,760 retained crypto Garman–Klass keys. This is estimated dollar turnover, not exchange-reported dollar volume. The altFINS historical OHLC schema gives no explicit quote-currency field; stored BTC closes are on the USD price scale (for example 87,614.59 on 2025-12-31), consistent with altFINS' dollar-denominated market display. USD quotation is inferred rather than guaranteed by the endpoint schema.

   | Rank | Crypto | Mean daily estimated dollar volume |
   |---:|---|---:|
   | 1 | BTC | $7,935,697,962 |
   | 2 | ETH | $5,102,476,851 |
   | 3 | SOL | $1,153,337,658 |
   | 4 | XRP | $983,874,418 |
   | 5 | DOGE | $902,343,829 |

2. Score 15 saved equity-trained global configurations without refitting: nine at window 20 and six at window 80. AR(1), GARCH(1,1), and SARIMA have only window-20 fits.
3. Forecast the next recorded daily crypto Garman–Klass variance from vendor OHLC: 0.5 ln(High/Low)² − (2 ln 2 − 1) ln(Close/Open)². The target is daily, unannualized variance; crypto OHLC is not equity-adjusted. Do not substitute crypto realized variance.
4. Use exactly the same 1,760 retained positive target keys per ticker at both windows. Earlier same-ticker records are history only. Reject duplicate dates, invalid OHLC, nonpositive retained variance, and mismatched retained table values.
5. Keep saved parameters, input transforms, and positive floors fixed. Leave actual variance unchanged. Apply local evaluation's test-date naive per-ticker MASE scale, MAE, MASE, MSE, RMSE, QLIKE, and floor-hit definitions. Reject missing fit artifacts, nonfinite forecasts, or nonpositive scales.
6. Record database size and modification time, fit paths, target dates, counts, and exclusions. Keep crypto scores separate from equity rankings.

## Dataset and implementation work

- Add `inference/evaluate_crypto_global.py` using the existing raw OHLC formula, datasets, scoring functions, and residual plotting. Output `metrics.csv`, `metrics.png`, `per_stock.csv`, `run.log`, and six residual plots for each configuration under `imgs/eval/crypto_eval/`.

## Verification

- Check dollar-volume arithmetic, Garman–Klass formula, target-key alignment, ticker boundaries, and prior-only windows on a small known example.
- Reconcile 15 aggregate rows, 75 ticker rows, 8,800 targets per configuration, saved fit source metadata, floors, and 90 plots. Validate a fresh-context cold review using `judge/validate.py`.

## Source and limit

- [altFINS historical OHLC API](https://altfins.com/crypto-market-and-analytical-data-api/documentation/api/get-history/) defines the fields but does not specify their quote currency. [altFINS' market-data deck](https://altfins.com/wp-content/uploads/2023/10/altFINS-dataAPI-deck-v2.pdf) displays BTC/ETH prices and volume in dollars. Applying that display convention to historical OHLC is an inference. These transfer results cannot establish universal generalization or economic utility.

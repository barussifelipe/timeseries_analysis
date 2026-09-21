# Roughness Analysis Implementation Plan

## Implementation decisions

1. Use the complete 25-year equity population from 2001-01-01 through 2025-12-31: 1,525 tickers and 9,587,674 observations.
2. Include all eligible equities globally. Treat cryptocurrency as an unseen cross-asset population; its panel need not match the equity panel size.
3. Give every retained cryptocurrency its latest 1,760 jointly valid observations ending no later than 2025-12-31. Candidate history begins on 2019-01-01.
4. Use identical retained `(Ticker, Date)` keys for all three crypto estimators.
5. Exclude pegged fiat currencies and stablecoins: `BRL, CAD, DAI, EUR, GBP, JPY, MXN, PAX, TRY, TUSD, USD, USDC, USDT`.
6. Rank the top-ten equities and cryptocurrencies independently by `AVG(Volume) DESC` over retained raw observations.
7. Store each asset-class/estimator combination separately in five `WITHOUT ROWID` tables keyed by `(Ticker, Date)`:
   - `equity_parkinson_variance`
   - `equity_garman_klass_variance`
   - `crypto_parkinson_variance`
   - `crypto_garman_klass_variance`
   - `crypto_realized_variance`
8. Store variance and derive volatility only as `sqrt(Variance)` when needed.
9. Parkinson variance is `log(High/Low)^2 / (4 log 2)`.
10. Reduced Garman-Klass variance is `0.5 log(High/Low)^2 - (2 log 2 - 1) log(Close/Open)^2`.
11. Crypto realized variance is the existing completed 96-bar UTC-day estimator: squared 15-minute close-to-close log returns, including the preceding day's 23:45 close.
12. Preserve ticker-major ordering and never calculate returns, lags, windows, or roughness displacements across ticker boundaries.
13. Pool valid within-ticker displacements with equal observation weight for global roughness.
14. Use `q = {1, 1.5, 2, 3, 4}` and exact calendar lags 1 through 400 days.
15. Save generated figures under `imgs/roughness_analysis/`.
16. Defer VIX, volatility commonality, and log-volume features until evidence shows they are needed.
17. Reject invalid logarithm inputs, inconsistent OHLC rows, non-finite values, negative volume, and non-positive variance. Report rejection counts.

## Dataset and storage work

- Add one focused module using the existing SQLite, NumPy, pandas, and Matplotlib dependencies.
- Build crypto eligibility from the intersection of all three estimator inputs, apply exclusions, require at least 1,760 dates, and retain the latest 1,760.
- Rebuild all five derived tables transactionally and idempotently without changing raw history or `crypto_daily_volatility`.

## Roughness analysis

For variance `V`, analyze log volatility as `log(sqrt(V))`. For every estimator, ticker, and calendar lag:

```text
d(i,t,Delta) = log sigma(i,t+Delta) - log sigma(i,t)
m(q,Delta) = sum_i sum_t |d(i,t,Delta)|^q / sum_i n_i(Delta)
```

- Regress `log m(q,Delta)` on `log Delta` separately for each `q` to estimate `zeta(q)` and regression R-squared.
- Estimate `H` from the origin-constrained relation `zeta(q) = Hq` and report its R-squared.
- Repeat independently for each top-ten local ticker.
- Generate deterministic global/local estimator timelines, distributions, moment-scaling plots, zeta fits, and local Hurst comparisons. Annotate estimator, population, observations, H, and R-squared.

## Verification

- Test both formulas against hand-calculated examples.
- Test crypto exclusions, joint eligibility, latest-1,760 selection, and identical estimator keys.
- Use a two-ticker fixture to prove displacements never cross boundaries and pooled sums/counts match manual results.
- Test non-positive variance filtering and unavailable calendar lags.
- Verify idempotent table rebuilding.
- Run existing data-fetching tests.
- Smoke-check schemas, unique keys, row counts, finite positive variances, ordering, top-ten rankings, and image creation.
- Update `CONTEXT.md` with finalized definitions, populations, outputs, and the next Step 1 task.

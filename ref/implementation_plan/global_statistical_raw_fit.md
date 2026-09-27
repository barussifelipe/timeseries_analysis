# Global Statistical Raw-History Fitting Implementation Plan

## Implementation decisions

1. The cohort is the existing raw-history selection: more than 80 pre-2016 rows and a 2025-12-31 row per ticker.
2. Fitting uses only target dates before 2016-01-01. Validation is 2016-01-01 through 2018-12-31; test is 2019-01-01 through 2025-12-31.
3. The target is adjusted, daily, unannualized Garman–Klass variance in variance units: 0.5 ln(High/Low)² − (2 ln 2 − 1) ln(Close/Open)².
4. The horizon is the next recorded observation after 20 recorded, valid, positive-variance inputs from the same ticker. Calendar gaps are allowed. Invalid source rows and zero-variance inputs break a run.
5. A valid zero-variance target is replaced by the minimum positive pre-2016 variance. The same value is recorded as the later lower forecast floor; it is never an upper cap.
6. All four models use the same eligible window-20 target dates. AR(1) uses the last input, and HAR uses the 1-, 5-, and 20-observation summaries.
7. Global AR(1) and HAR use ordinary least squares with one contribution per eligible training target. The implementation accumulates scaled normal equations by chunk and records their condition number.
8. Global SARIMA(1,0,1) × (1,0,1,5) maximizes the sum of eligible ticker-segment log likelihoods with the existing Powell method and 100-iteration limit.
9. Global GARCH(1,1) minimizes the sum of eligible ticker-segment Gaussian conditional negative log likelihoods with the existing L-BFGS-B bounds and 100-iteration limit. Its residual is adjusted ln(Close/Open) minus the segment's causal preceding-observation mean. It estimates return conditional variance, not Garman–Klass forecast-error coefficients.
10. Valid zero-variance targets close positive-input segments; invalid rows close them without joining across the boundary. Segments require at least 21 observations.
11. The fit-only entry point is `--raw-fit-only --estimator garman-klass --scope global --window-size 20`. It saves compact JSON in a separate `raw_history_w20` checkpoint path without W&B or forecast scoring.
12. The full cohort must match 8,664,516 train, 1,806,548 validation, and 4,479,681 test eligible windows before fitting. Fit-only commands reject ticker and row limits so diagnostic fits cannot overwrite full-cohort artifacts.
13. RFSV waits for H and ν² estimated from this cohort. The existing derived-table commands and RFSV artifact remain separate.
14. The older derived-table path selected 1,525 tickers. Its fitted parameters are not interchangeable with the 2,714-ticker raw-history fits, and no forecast comparison is made at this stage.

## Verification

- Run the two-ticker raw-history fixture for Garman–Klass, floors, boundaries, split keys, streamed OLS, and pooled objective sums.
- Verify the full-cohort counts before running each fit, then reload each completed JSON artifact and check finite parameters and convergence.
- Obtain a fresh-context cold review and validate its JSON before reporting completed fits.

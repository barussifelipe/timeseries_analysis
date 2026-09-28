# Raw-History Garman-Klass Roughness Re-Estimation

## Implementation decisions

1. Select the current raw-history equity cohort: more than 80 rows before 2016-01-01 and a row on 2025-12-31. The database contains 2,714 such tickers and 12,397,872 raw pre-2016 rows, including history back to 1962-01-02.
2. Use only adjusted OHLCV rows that pass `raw_frame` validation and have strictly positive, finite, daily unannualized Garman-Klass variance. Zero and invalid rows cannot enter log volatility. Keep the original target-floor policy separate; it does not repair source exclusions or define roughness observations.
3. Use 0.5 ln(GK variance) as log volatility. Within each ticker, use recorded-observation lags 1–400 on the retained positive rows. Pool displacement moments by observation count for q = {1, 1.5, 2, 3, 4}; do not join ticker boundaries or use validation/test rows.
4. Estimate ζ(q) by a free-intercept regression of ln moment on ln lag. Estimate H through the origin in ζ(q) = Hq. Estimate ν² as exp(the q = 2 regression intercept). These are descriptive proxy estimates for the current cohort, not forecast scores.
5. Replace only the GK rows of the three pre-2016 training CSVs and the GK scaling plot in `imgs/roughness_analysis/global/train/`. Relabel the shared H plot with each estimator's cohort. Leave the Parkinson rows and full-period figures unchanged. The resulting shared pre-2016 CSVs contain two different equity cohorts, marked by `Cohort` in the summary.
6. Preserve the older derived-table RFSV checkpoint. It belongs to the 1,525-ticker sample; a current-cohort RFSV fit is a separate step.

## Verification

- Run the focused synthetic cohort test: excluded ticker, pre-2016 cutoff, within-ticker lag counts, positive ν², and preserved Parkinson summary row.
- Recompute H and ν² from the saved GK moments and compare with the saved summary and `rfsv_fit('garman-klass')` output.
- Check the generated GK scaling plot and shared H plot, saved CSV row counts and metadata, and the git diff. Obtain and validate the required cold review.

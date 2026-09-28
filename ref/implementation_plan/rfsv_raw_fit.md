# Raw-Cohort Garman-Klass RFSV Fit

## Implementation decisions

1. Fit only the global adjusted-equity Garman-Klass model on the existing 2,714-ticker raw-history cohort: more than 80 raw rows before 2016-01-01 and a row on 2025-12-31. The database version is recorded by absolute path, byte size, and modification time.
2. Require the same 20-positive-input, next-recorded-observation target keys as the raw-history ML path: 8,664,516 train, 1,806,548 validation, and 4,479,681 test windows. Invalid and zero-variance inputs break windows; valid zero targets use the existing training-derived floor. No validation/test observations enter parameter estimation.
3. Take H = 0.03375577838376565 and ν² = 0.49527452891226625 from the saved pre-2016 GK observation-lag roughness moments. Require the saved summary to identify the current raw cohort and agree with recomputed moments. RFSV has no separate iterative optimizer.
4. Forecast daily unannualized GK variance one recorded observation ahead from the latest 20 positive same-ticker variances. The existing Section 5 kernel uses exact observation-bin masses, oldest-value tail extension, and the +2cν² log-variance correction for horizon 1; apply the pre-2016 minimum positive variance as a prediction floor.
5. Save a fit-only JSON at `inference/checkpoints/rfsv/garman-klass/global/raw_history_w20/fit.json`, separate from the earlier derived-table `.pth`. Store parameters, roughness source, floor, cohort, split counts, and database version. Do not score validation/test or start W&B.
6. Reject a derived-table RFSV fit that attempts to read these raw-cohort GK moments. Keep the previous derived checkpoint unchanged.

## Verification

- Run the focused raw-history key fixture and RFSV forecast equation check.
- Run the full-cohort count gate, save/reload the JSON, verify finite parameters, floor, source cohort, and expected split counts.
- Obtain a fresh-context cold review, validate its JSON, and fix any failing findings.

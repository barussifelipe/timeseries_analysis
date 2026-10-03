# RFSV Rescoring and Window Matched MCS Implementation Plan

## Implementation decisions

### Agreed decisions

1. Score the saved global and local RFSV fits on every eligible test target using only the preceding 20 or 80 observations of the same ticker. The target is the next recorded adjusted, daily, unannualized Garman–Klass variance: 0.5 × [ln(High/Low)]² − (2 ln 2 − 1) × [ln(Close/Open)]², using adjusted OHLC values.
2. Reuse the saved pre-2016 estimates of H and ν² and their positive prediction floors. Do not refit or change source data.
3. Replace final-date RFSV rows with RFSV-20 and RFSV-80; regenerate MAE, MASE, MSE, RMSE, QLIKE, floor-hit rates, and residual plots.
4. Run separate MCS comparisons for each window, with its matching local and global model variants plus AR(1), GARCH(1,1), and SARIMA: 18 candidates per stock per run.
5. Use blocks of 20 observations for `20_size` and 80 for `80_size`; retain T_max, 10,000 draws, α = 0.10, ticker-specific seeded streams, and QLIKE.
6. Revise only the two thesis result tables, their captions, and numerical color rankings; leave surrounding claims for the user's later edit.

### Defaults to validate

7. Use each window's test-date naive MASE scale for RFSV, matching that window's other models.
8. Remove the nine-ticker exclusion imposed solely by final-date RFSV's zero MASE scale, after checking every restored ticker's full-period scale. Expected global scored counts are 4,479,681 for window 20 and 4,295,773 for window 80; regenerate all model rows on those populations.
9. Keep the root MCS outputs as historical artifacts. Current window-matched results go under `imgs/eval/mcs/20_size/` and `imgs/eval/mcs/80_size/`, each with dated forecasts and losses, `results.json`, and a readable final-set summary.

## Dataset and implementation work

- Extend the shared RFSV scoring path to take either window and call the existing `RFSV.forward()` kernel with only earlier same-ticker observations.
- Update global and five-stock scoring and residual plotting to include both window-matched RFSV rows, while retaining the saved fit identities and floors.
- Regenerate both evaluation outputs and reconcile RFSV rows with dated forecasts. Require identical sorted ticker/date keys and actual values for all 18 MCS candidates in each run.
- Record candidate order, fit identities, floors, source version, block length, seeds, and exclusions in each MCS result.
- Regenerate the two thesis table values, captions, and color rankings from saved CSVs. Update the focused table check and `CONTEXT.md`.

## Verification

- Check both windows against `RFSV.forward()` on small histories, including ticker boundaries, positivity, and no future observations.
- Verify global counts, finite metrics, MASE scales, RMSE identities, floor hits, residual counts, and exact five-stock alignment.
- Require 1,760 dates and 18 candidates per stock in each MCS run; reconcile per-stock mean QLIKE with the new audit. Check both block lengths and deterministic reruns.
- Validate thesis rows against CSVs; compile `ref/final_report/thesis_structure.tex` in place with two `pdflatex -synctex=1` passes.
- Obtain a fresh-context cold review using `judge/prompt.md`; validate its JSON and resolve findings until at least 95/100 with no critical findings.

The thesis prose outside the tables temporarily retains claims about final-date RFSV, as requested.

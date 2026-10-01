# MCS Implementation Plan

Plan the Model Confidence Set (MCS) comparison of saved local and global volatility forecasts. Follow Brini's settings with 20-observation blocks instead of 22. This document does not implement the procedure or report results.

## Implementation decisions

### Agreed choices

1. Use Brini's T_max statistic and matching argmax elimination rule.
2. Use a moving-block bootstrap with fixed block length **20**, replacing Brini's 22.
3. Use **10,000 bootstrap repetitions**.
4. Use **alpha = 0.10**, producing a **90% MCS**.
5. Use **QLIKE** only for this implementation. MSE confidence sets remain a separate future comparison.
6. Build one MCS per stock for **NVDA, AAPL, NFLX, GOOG, and AMZN**.
7. Compare the **26 full-period configurations**, treating model family, window, and LOCAL/GLOBAL scope as distinct candidates in one initial set per stock.
8. Exclude both RFSV-full configurations: each supplies only one forecast per stock, insufficient for a 20-observation bootstrap. Including them later requires matched historical forecasts.

### Project defaults and validation requirements

9. Preserve the existing target: next recorded within-ticker observation of adjusted daily, unannualized Garman-Klass variance, computed as 0.5 × ln(High/Low)² − (2 ln(2) − 1) × ln(Close/Open)².
10. Preserve the existing test period, **2019-01-01 through 2025-12-31**, saved fits, input transformations, and saved positive prediction floors. Do not refit, change targets, or introduce another floor.
11. Require identical sorted `(Ticker, Date)` keys and actual values across all 26 candidates. Expect **1,760 observations per stock**, but verify complete key equality rather than relying on counts.
12. Stop on mismatches, missing candidates, duplicate keys, or non-finite/non-positive actuals and predictions. Report affected observations; do not silently intersect dates or filter candidates.
13. Interpret "20 days" as **20 consecutive retained forecast observations**, following the existing observation-time target. Report calendar gaps; do not fill them or imply blocks span exactly 20 calendar days.
14. Use overlapping, non-circular blocks starting at positions 0 through n − 20. Draw blocks uniformly with replacement, concatenate enough to reach n observations, and truncate the last block. Reject inputs with fewer than 20 observations.
15. Resample the same row indices for every candidate within a stock. Never construct blocks across stock boundaries.
16. Use NumPy's random generator with seed **42**, and deterministic child streams for tickers sorted alphabetically. Reuse each stock's bootstrap samples throughout its elimination sequence; record the generator and library versions.
17. Break argmax ties by lexicographically sorted candidate identifier.
18. Preserve comparisons of saved configurations. Different features and training budgets limit attribution of differences to model scope alone. Existing AR(1) and HAR candidates provide simple statistical baselines; MCS survival does not establish absolute accuracy or economic utility.
19. Treat bootstrap centering as null calibration, not a learned forecasting transformation. It uses evaluation losses without changing forecasts or fitting models.
20. Brini explicitly supplies the statistic, block length, repetitions, loss, significance level, and per-asset/horizon comparison. Seed, tie handling, finite-bootstrap correction, and implementation mechanics above are project defaults, not claims about his code.

## Evaluation and implementation work

- Add a focused MCS module and entry point using existing NumPy and pandas dependencies.
- Reuse verified fit loading and the existing scoring callbacks in `inference/evaluate_local_global.py` and `inference/evaluate_volatility_test.py` to obtain dated actuals and floored predictions. Current summary CSVs lack the observation-level loss history required for bootstrapping.
- Compute dimensionless per-observation QLIKE as actual/prediction − ln(actual/prediction) − 1, using the existing numerically stable log-ratio formulation.
- Keep the existing evaluation artifacts unchanged. Write new outputs beneath `imgs/eval/mcs/`: dated forecasts and QLIKE losses; elimination trace with candidate set, T_max, bootstrap p-value, and removed candidate; final membership and adjusted MCS p-values for each stock/candidate; inclusion percentages across the five stocks; and metadata recording source and fit identities, target, dates, gaps, floors, candidate order, seed, bootstrap settings, and exclusions.
- Record Brini's settings and the deliberate 20-observation departure in thesis notes when implementing; reuse his existing reference.
- Update `CONTEXT.md` after implementation, stating which checks passed, whether project-data evaluation ran, and what remains pending. Adding this plan alone leaves the research plan position unchanged.

## Test statistic and elimination sequence

For each current candidate set M:

1. Calculate each model's mean excess loss relative to the current-set average, including itself in that average.
2. For each bootstrap sample, calculate that same mean excess loss and subtract the original mean excess loss to impose the null.
3. Estimate the variance of each mean excess loss using its 10,000 bootstrap replicates and the sample-variance convention (`ddof=1`).
4. Divide observed excess losses and centered bootstrap excess losses by the corresponding estimated standard errors.
5. Take the maximum standardized excess loss across current candidates for both the observed and bootstrap statistics.
6. Calculate p = (1 + number of bootstrap maxima ≥ observed maximum) / 10,001.
7. If p < 0.10, remove the argmax candidate and recompute for the reduced set. Otherwise, stop.
8. If one candidate remains, retain it without another test.
9. For eliminated candidates, report the running maximum of step p-values as their adjusted MCS p-value; assign surviving candidates 1. Report raw step p-values separately.
10. If an excess-loss series is identically zero, define its observed and bootstrap standardized statistics as zero. Stop with a diagnostic for any other zero or non-finite estimated variance; do not add arbitrary denominator offsets.

## Verification

- Hand-check QLIKE and standardized excess losses on a tiny known example.
- Verify block length, valid starting positions, replacement sampling, final truncation, shared candidate indices, and ticker boundaries.
- Verify null centering, maximum versus argmax, tie handling, finite-bootstrap p-values, rejection, recomputation, singleton stopping, and adjusted p-values.
- Check identical-loss and degenerate-variance cases explicitly.
- Verify repeated runs with the same inputs and seed produce identical outputs.
- Reconcile regenerated per-stock mean QLIKE with the existing audit using relative tolerance 10⁻⁸ and absolute tolerance 10⁻¹²; stop and investigate discrepancies.
- Confirm 130 stock/candidate membership rows, five final sets, and inclusion percentages calculated from stock-level membership. These expected counts are acceptance conditions, not completed results.
- Obtain a fresh-context review using `judge/prompt.md`, validate its JSON with `python judge/validate.py REVIEW.json`, and resolve findings until the score is at least 95/100 with no critical findings.
- For the immediate plan-writing task, verify documentation consistency only. Do not run forecast regeneration, the full bootstrap, training, downloads, commits, or pushes. Project-data evaluation requires a subsequent user request.

## Sources and limits

- [Brini (2026), Section 4.4 and Table 8](https://arxiv.org/html/2607.05291v1#S4.SS4): moving-block bootstrap, 22-day blocks, 10,000 repetitions, QLIKE, T_max, alpha = 0.10, and per-asset/horizon MCS inclusion. The 20-observation block length is the user's project decision.
- [Hansen, Lunde, and Nason (2011), Section 3.1.2](https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Hansen_Lunde_Nason.pdf): standardized mean excess-loss statistic, matching elimination rule, and dependence-aware bootstrap null calibration.
- Survival means insufficient evidence to continue elimination under the chosen loss and settings. It does not prove equal expected loss. The asymptotic coverage guarantee requires the paper's assumptions; choosing a fixed block length alone does not establish those assumptions for this dataset.

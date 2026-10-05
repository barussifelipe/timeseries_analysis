## Thesis moment interpretation bands (2026-10-05)

Section 2 now has bullet interpretations immediately after Equations 2.6 and
2.7. WHO/UNICEF's survey-data rules of thumb supply the descriptive skewness
band [-0.5, 0.5] and excess-kurtosis band [-1, 1]; NIST supplies the left/right
and lighter/heavier-tail readings. The thesis states that these are neither
normality tests nor validated cutoffs for financial data. Following Westfall
(2014), it notes that heavier tails can coincide with a more concentrated
center and lighter tails with a flatter one, but peak height does not follow
from kurtosis alone. The excess-kurtosis bullets now separate the sign from
the screening band: positive values indicate heavier-tail direction and
negative values lighter-tail direction even within [-1, 1]. The sources were
added to the working references and
thesis bibliography and summarized in thesis notes. Two in-place SyncTeX
LaTeX passes rebuilt the PDF, pages 9-10 were inspected, and the technical
evaluation checker passed. Next: interpret residual patterns alongside matched
QLIKE and MCS results, then revisit stale RFSV prose.

## Thesis moment definitions (2026-10-05)

Section 2 now names the mean, variance, skewness, and kurtosis after Equation
2.1. Population skewness and excess kurtosis are Equations 2.6 and 2.7
immediately after variance Equation 2.5; the former skewness definition after
covariance was moved. The text distinguishes these population moments from
the pooled histograms' descriptive sample moments, which use n in their
standardizing variance. Two in-place SyncTeX-enabled LaTeX passes rebuilt the
PDF; page 9 was inspected and both equations are readable. Next: interpret
residual patterns alongside matched QLIKE and MCS results, then revisit stale
RFSV prose.

## Thesis histogram shape statistics (2026-10-05)

The 20 raw/log residual histograms referenced by the ten Residual Structure
variants and both target histograms in Figure 6.2 display pooled full-sample
skewness and excess kurtosis beside mean and population standard deviation.
They use the mean cubed and mean fourth-power standardized observation,
respectively, with 3 subtracted from the fourth moment; a normal distribution
has zero excess kurtosis. Each full and central-99.5% panel shows the same
8,800-observation statistics. The 22 PNGs were regenerated from saved matched
forecast CSVs, without changing forecasts, targets, splits, or values. Shared
plotting code labels future regenerated figures. Focused target and residual
checks passed; two in-place SyncTeX-enabled LaTeX passes rebuilt the PDF. The
Base LSTM-20 prose agrees with the positive raw and log skewness shown in its
figures. Next: interpret residual patterns alongside matched QLIKE and MCS
results, then revisit stale RFSV prose.

## Garman--Klass data figures (2026-10-05)

`data/plot_gk_data.py` reads the existing 2,714-ticker raw-history cohort read-only
and writes 16 adjusted-OHLC GK PNGs plus `imgs/gk_data/counts.json`. Section 3.1
now shows full-history global and five-stock data; Section 4.1 shows their
pre-2016 observations before model inputs. Full-history global rows: 19,220,867
raw, 176,432 invalid, 19,044,435 plotted, including 1,254,289 valid zeros
replaced by the cohort training floor 3.396062419686545e-13. Pre-2016 global:
12,397,872 raw, 165,479 invalid, 12,232,393 plotted, including 1,078,857
floored zeros. Full-history local: 36,654 raw, 16 invalid, 36,638 plotted,
including 28 zeros; pre-2016 local: 24,084 raw, 16 invalid, 24,068 plotted,
including the same 28 zeros. Local floors come from each stock's positive
pre-2016 observations. Timelines show the daily mean and natural log of that
mean; histograms pool observations and fit one normal curve per full plotted
population, with a central 99.5% display zoom. These are descriptive rows,
not eligible forecast windows. The tiny GK formula and daily-mean checks passed;
all 16 PNGs were checked; two in-place SyncTeX-enabled LaTeX passes produced a
readable 56-page PDF and the new pages were inspected. No source data, training,
forecast, or score file changed. The broader data-selection and technical-
evaluation checkers still fail older assertions outside this GK work. Next:
interpret residual patterns alongside matched QLIKE and MCS results, then
revisit stale RFSV prose.

## Thesis residual normal overlays (2026-10-05)

The 20 raw/log histogram PNGs used by the ten Residual Structure variants now
have in-place orange normal overlays. For each variant and residual scale, the
curve uses the unconditional mean and population variance of its 8,800 pooled
five-stock test residuals from the saved window-matched MCS forecasts. The same
full-sample fit appears on the full and central-99.5% panels, scaled to expected
bin counts. `inference/plot_global_residuals.py` now preserves this overlay on
future histogram regeneration; the thesis captions and explanatory prose state
the method and its descriptive limit. The earlier standalone Base LSTM-80 Local
diagnostic files were removed. Forecasts, targets, and timelines did not change.
The technical-evaluation checker and a three-point normal-fit arithmetic check
passed. Two in-place SyncTeX-enabled pdflatex passes produced the 51-page PDF;
the first and last residual pages were visually inspected without new layout
warnings or unresolved references. A fresh-context cold review independently
reproduced all 20 PNGs pixel for pixel from saved forecasts and passed 100/100;
its validated JSON is `judge/reviews/thesis_residual_normal_overlays_review.json`.
Next: interpret residual patterns alongside matched QLIKE and MCS results,
then revisit stale RFSV prose.

## Thesis residual figure numbering (2026-10-04)

Residual Structure now gives each of the ten shared MCS model variants two
separately numbered figures: one with raw/log daily mean timelines and one with
raw/log individual-residual histograms. The same 40 saved PNGs and 8,800 matched
targets per variant are used; forecasts and residual values did not change.
The technical-evaluation checker verifies all 20 figure labels and image pairs.
Two in-place SyncTeX-enabled pdflatex passes produced the 51-page PDF; first
and last residual pages were visually checked, and no unresolved references
were reported. The fresh-context cold review
`judge/reviews/residual_figure_split_review.json` passed 100/100 and validated.
Next: interpret the residual patterns alongside matched QLIKE
and MCS results, then revisit stale RFSV prose.

## Thesis Technical Evaluation residual plots (2026-10-04)

Technical Evaluation now uses subsections in the agreed order. Residual Structure defines individual raw/log errors and the distinct daily cross-ticker raw mean and log-of-means, with variance units and sign interpretation. Four model subsubsections show the ten variants shared by all five final MCS sets under both bootstrap block lengths. Ten four-panel figures reuse 40 saved five-stock residual PNGs; no forecasts or plots were regenerated. The focused technical-evaluation checker passes. Two in-place SyncTeX-enabled pdflatex passes built the 54-page thesis PDF; sampled residual pages show the intended panel order and section numbering, with no new overfull warning. The implementation plan is `ref/implementation_plan/residual_plot.md`. Next: interpret the residual patterns alongside matched QLIKE and MCS results, then revisit stale RFSV prose.

## Thesis cross-window MCS table (2026-10-04)

Section 5.4.3 now reports the five stock-level final-set sizes for both
20- and 80-observation bootstrap blocks in Table 5.6. Per the user's final
preference, it lists only the ten-model intersection shared by all five stocks,
then shows the difference between block-length results in the table.
The only membership difference is GOOG's AR(1) Local, retained with
80-observation blocks. Counts and names were checked against both saved result JSONs; the
technical-evaluation checker now accepts the current subsection order and
passes. Two direct SyncTeX-enabled pdflatex passes rebuilt the 50-page PDF;
the table and adjacent pages were visually inspected. A `\FloatBarrier` after
Table 5.6 keeps it before the Section 5.4.4 heading, verified on PDF page 39.
The separate
data-selection text checker currently fails on its preexisting 96-bar wording
assertion. No MCS result, forecast, or source data changed. Next: interpret
these MCS results alongside matched QLIKE scores and revise stale RFSV prose.

## Cross-window MCS reassessment (2026-10-04)

At the user's request, both MCS block-length runs now include all 30 distinct
local/global candidates, including both 20- and 80-observation forecast variants.
The reassessment reads the two saved window-matched forecast files, verifies
identical 1,760 dates and actuals for every candidate and identical predictions
for six shared candidates, then recomputes QLIKE T_max sets with the original
10,000 draws, alpha 0.10, seed 42, and blocks of 20 or 80 observations. New
outputs are in `imgs/eval/mcs/all_windows/`; original window-only results remain
for comparison. Ten models survive across all five stocks in either run.
Stock-level membership differs only for GOOG (19 members with 20-observation
blocks, 20 with 80-observation blocks). The thesis MCS candidate description
now matches these runs. No fitting, data selection, or forecast regeneration
occurred. Focused MCS tests passed (2), and a repeated run produced byte-identical
result JSONs. Two in-place pdflatex passes rebuilt the thesis PDF and SyncTeX.
The broader technical-evaluation checker currently fails its fixed subsection-order
assertion against the user's new MCS subsection; the fresh-context MCS review
passed 99/100 with validated JSON at
`judge/reviews/cross_window_mcs_review.json`. Next: interpret these MCS results in the Experimental Results section
alongside matched QLIKE scores, then continue the stale RFSV prose revision.

## Thesis Crypto Realized Variance Table (2026-10-04)

Section 5.4.1 (Experimental Results) now includes Table 5.5 (tab:crypto-rv-results)
presenting the transfer evaluation of all 15 saved equity-trained Global fits on
the 15-minute intraday Realized Variance proxy \eqref{eq:realized-variance} over the
same 8,800 retained crypto targets. It follows the exact format of Table 5.4
(longtable layout, 15 model rows grouped into window-20 and window-80 by \hline,
minima in red, second minima in blue, maxima in dark orange). Column scalings
preserve RMSE = sqrt(MSE) identities: MAE (×10⁻³), MASE (unscaled), MSE (×10⁻²),
RMSE (×10⁻¹), and QLIKE (unscaled, with floor-hit values in scientific notation).
Accompanying text notes that under Realized Variance, RFSV-80 achieves the lowest
MSE and RMSE, outperforming all neural and classical models on squared loss, while
Base LSTM-20 minimizes MASE and QLIKE, and SiLU-LSTM-20 minimizes MAE. It also
documents how floor activations on high-variance crypto regimes heavily penalize
QLIKE (reaching 1.34×10¹⁰ for MLP-20), whereas recurrent and statistical models
without floor hits maintain stable QLIKE values between 0.39 and 2.09 (with MLP-80
at 84.66). Two in-place SyncTeX-enabled pdflatex passes rebuilt the 49-page PDF
without errors or undefined citations. Focused checks judge/check_technical_evaluation.py
and judge/check_data_selection_text.py passed. Fresh-context cold review
judge/reviews/crypto_rv_thesis_table_review.json passed 99/100 with validated JSON.
The next thesis task remains revision of stale final-date RFSV prose against
the window-matched results.

## Thesis Crypto RV Validity & Calendar Gaps Before Table 5.4 (2026-10-04)


Section 5.4.1 (Experimental Results) now documents the cryptocurrency Realized
Variance (RV) validity requirements, omitted dates from joint estimator filtering,
and the resulting calendar gaps in the explanatory paragraph immediately preceding
Table 5.4 (tab:crypto-results). It explains that constructing daily Realized
Variance strictly requires 96 complete, clean 15-minute intraday bars per 24-hour
UTC day and the preceding day's 23:45 close for return continuity
(r₁(t) = ln(P_{t, 00:00} / P_{t-1, 23:45})). Dates failing these criteria (due to
vendor collection dropouts, API holes, or missing preceding boundary closes) were
omitted from the joint panel. To ensure a matched, directly comparable evaluation
panel across estimators, candidate dates were filtered by their intersection,
retaining the latest 1,760 jointly valid observations per coin through 2025-12-31.
Consequently, forecasting models target the next recorded valid observation from
preceding valid history, bridging occasional calendar gaps (64--65 targets per coin
have a two- or three-day gap). Two in-place SyncTeX-enabled pdflatex passes rebuilt
the 48-page PDF without errors or undefined citations. Focused checks
judge/check_data_selection_text.py and judge/check_technical_evaluation.py passed.
Fresh-context cold review judge/reviews/crypto_evaluation_data_issues_review.json
passed 100/100 with validated JSON. The next thesis task remains revision of
stale final-date RFSV prose against the window-matched results.

## Thesis Data Section Local & Crypto Selection (2026-10-04)


Section 3.1 (Data) now documents the local equity selection (NVDA, AAPL,
NFLX, GOOG, and AMZN) from the 2,714-ticker cohort, chosen by highest mean
nonnegative daily trading volume before 2016 (retaining one share class per
firm, with GOOG outranking GOOGL). It explains the equity volume rationale:
corporate forward share-split conventions generally keep nominal prices within
comparable retail-accessible bands ($10¹ to $10²), allowing raw volume to
capture major market pillars.

Section 3.1 also documents the out-of-sample cryptocurrency transfer selection
(BTC, ETH, SOL, XRP, and DOGE) across 1,760 aligned dates from 2019 to 2025. It
provides the concise rationale for ranking by mean daily dollar volume
(Close × Volume) rather than raw token volume: cryptocurrency token supplies span
over ten orders of magnitude without standardized denominations; raw volume is
dominated by micro-priced tokens while dropping BTC (which ranks 113th in raw
volume), whereas dollar volume neutralizes denomination bias and reflects market
liquidity. Furthermore, it clarifies that while equities rely solely on daily
OHLC range data, the cryptocurrency dataset provides both the daily
Garman--Klass (GK) estimator and the 15-minute intraday realized variance (RV)
proxy (96 fifteen-minute squared log returns per day), enabling cross-proxy
transfer evaluation. Line 933 in Section 5.2 was streamlined to match this
concise phrasing. Two in-place SyncTeX-enabled pdflatex passes rebuilt the
48-page PDF without undefined references or fatal errors. Focused check
judge/check_data_selection_text.py and judge/check_technical_evaluation.py
passed. Fresh-context review judge/reviews/data_local_crypto_selection_review.json
passed 100/100 with validated JSON. The next thesis task remains revision of
stale final-date RFSV prose against the window-matched results.

## Thesis computational timing (2026-10-04)

Section 5.3 now starts with the documented lower bound of 70 h 52 min of
cumulative experiment-run elapsed time, including global model selection,
final fits, measured failed/interrupted attempts, and local fits. The total is
not calendar or GPU time and excludes runs without final elapsed markers.
The final-global-fit table reports 14 timed fits for 15 model variants because
RFSV-20/80 share one parameter fit; Base LSTM-20 and MLP-80 report only
completed continuation segments. Five-stock local fits total 46 min 42 s
across 40 neural fits and 2 min 23 s across 30 statistical fits. No fitting
or evaluation was rerun. The next thesis task remains revision of stale
final-date RFSV prose against the window-matched results.

## MCS summaries and thesis table layout (2026-10-03)

The current `imgs/eval/mcs/20_size/final_sets.md` and `80_size/final_sets.md`
now list only the intersection across all five stock-level final sets: five
models at window 20 and six at window 80. The superseded
`imgs/eval/mcs/historical_rfsv/` run and root `final_set.md` were removed at
the user's request; root `forecasts.csv` and `results.json` remain historical.
No forecasts, losses, or MCS memberships changed. Thesis Tables 5.1 and 5.2
now span the text width with inline metric scales. MSE is displayed in units of
10⁻⁵ in Table 5.1 and 10⁻⁷ in Table 5.2; RMSE uses 10⁻³ in Table 5.1 and
10⁻⁴ in Table 5.2. There is no
separator before the RFSV rows in Table 5.2. The underlying
metric CSVs are unchanged. Two in-place SyncTeX-enabled `pdflatex` passes
rebuilt the 46-page PDF. The next thesis task remains revision of the stale
final-date RFSV prose against the window-matched results.

## Window-matched RFSV scoring and MCS (2026-10-03)

The current implementation plan is `ref/implementation_plan/rfsv_adjustment_scoring.md`.
Saved pre-2016 global and five local RFSV H, ν², and positive floors now score
the full 2019–2025 eligible test period with preceding 20- or 80-observation
within-ticker histories. The target remains next-recorded adjusted daily,
unannualized Garman–Klass variance. Both RFSV rows use their window's
test-date naive MASE scale. The former nine-ticker final-date exclusion was
removed after both full-period window populations showed positive scales:
4,479,681 forecasts at window 20 and 4,295,773 at window 80, with zero
scale exclusions. The global CSV has 15 rows; the five-stock CSV has 30
local/global rows and 150 per-stock audit rows. Both MCS runs use 18 matching
candidates per stock, 1,760 aligned dates, 10,000 QLIKE bootstrap draws,
and blocks of 20 or 80 observations. Current outputs are under
`imgs/eval/mcs/20_size/` and `imgs/eval/mcs/80_size/`; the root MCS outputs
remain historical. Their dated QLIKE means match the five-stock audit and
repeated loss-matrix bootstraps reproduce the saved elimination traces.
Only the two thesis result tables, captions, and numerical rankings were
updated; surrounding historical final-date claims await the user's edit.
The focused 45-row table check and two adjacent SyncTeX-enabled PDF passes
passed. Global and five-stock residual regeneration completed, with six PNGs
for each of 15 global and 30 local/global model sets; their RFSV counts match
the scored populations. Fresh-context review
`judge/reviews/rfsv_window_mcs_review.json` passed 100/100 with no findings,
and `python judge/validate.py` validated its JSON. The next concrete task is to revise those surrounding result claims
and interpretation against the matched-period scores and MCS outputs.

## Thesis Technical Evaluation (2026-10-02)

Follow-up edits explain the saved nine-ticker exclusions: zero raw GK variance
on the final and preceding dates receives the same floor, leaving a zero
RFSV final-date MASE scale. Tables and figures now use section-based numbers.
The complexity table is flushed before Residual Structure. Its last column
reports the Global / Local combination ratio B^(-2713p), without percentage
scaling, following the user's latest correction.
The plain-ratio follow-up rebuilt the adjacent PDF/SyncTeX with two passes;
the focused check and fresh-context review
`judge/reviews/technical_plain_ratio_review.json` passed 100/100 without
findings, with validated JSON.
The rebuilt PDF has 46 pages; Table 5.5 appears on page 35, before Residual
Structure on page 37. Two direct SyncTeX passes and the extended focused
check pass; no new overflows or unresolved references. Fresh-context follow-up
review `judge/reviews/technical_evaluation_followup_review.json` passed
100/100 with no findings and validated JSON. These edits do not change
scores, data, or plan position.

Technical Evaluation now contains Overall Result, Parameter Complexity, and
Residual Structure in `ref/final_report/thesis_structure.tex`. The agreed
plan is saved as `ref/implementation_plan/technical_evaluation.md`. Four
editable metric tables preserve all 42 saved CSV rows, with separate global
window populations and separate RFSV final-date tables. Global counts come
from the matching full-history log, not the older windowed-RFSV log.

Evaluation Setup now follows saved observation weighting, pooled RMSE, and
test-relative ticker/window naive MASE. This supersedes the equal-weight
RMSE convention below; evaluation code and saved metrics are unchanged.
Training-scaled MASE remains the roadmap convention for future work, and
the reporting departure is explicit. Full-period five-stock extrema select
Local SARIMA, Global AR(1), Global MLP-80, and Local SiLU-LSTM-80 for four
figure groups reusing 16 matched-population residual plots. Log-of-means
timelines are distinguished from individual log residuals, and display
trimming, varying axes, and retrospective test selection are disclosed.

The complexity table uses verified architecture counts and B = 2^64 for
2,714 tickers; its local column is extrapolated from only five fitted local
tickers. Encoding combinations require distinguishability to count functions.
Checkpoint tensors are unavailable; SARIMA's five parameters are checked
from its selected specification because statsmodels is not installed.
Mixed metric/scope winners, differing training budgets, and RFSV's single
date prevent blanket superiority or scope-only attribution. No fitting,
rescoring, downloads, commits, or pushes were run.

The focused check `python judge/check_technical_evaluation.py` verifies
42 rows, counts, floor-hit rates, rounded precision, RMSE identities,
extrema, complexity exponents, figure paths, and subsection order.
Two direct SyncTeX-enabled passes built a 45-page PDF; pages 32-41 were
visually inspected (also page 31 after merge recovery). Existing overflow
warnings outside this task remain;
the added section has no overflow or undefined references. Fresh-context
review `judge/reviews/technical_evaluation_review.json` passed 100/100 with
no findings and validated JSON. The user's Computational Evaluation timing paragraph
was preserved verbatim during recovery from an incorrect merge. The local
loss display and introduction now fit; the matched table stays on one page.
Next concrete thesis task: substantiate Computational Evaluation with saved
timing evidence, then scope matched-observation MCS and unseen-asset
transfer checks; these analyses are not completed by this section.

## Thesis Evaluation Setup (2026-10-02)

Hyperparameters opening now records the user's baseline-LSTM comparison
(window 20; hidden sizes 4, 16, 32, 64, 128; learning rates 0.001 and 0.0001)
and choices 128/0.001. Corrected the first-layer architecture attribution to
Zhang et al., Appendix B, Table B.2, using the existing zhang2022 citation;
the SiLU crypto-winter paper uses smaller hidden widths. No new experiment
was run or ranking independently verified. Research notes record the source.
Two final direct SyncTeX-enabled LaTeX passes regenerated the adjacent
36-page PDF. No new overflow or undefined-reference warnings remain in the
rewritten paragraph; existing user whitespace elsewhere is preserved.
Fresh-context review `judge/reviews/thesis_hyperparameters_opening_review.json`
passed 100/100 with no findings and validated JSON.

Implementation now ends with Evaluation Setup: report the global metric over
all global forecast observations; report the arithmetic mean of the five
ticker-specific local metrics for each model specification, with equal ticker
weight, as requested. Global and local equations reuse the Concepts mean
loss J(theta; Z) and observation loss ell(prediction, actual), with one shared
global theta and ticker-specific local theta_i. Evaluation sets and counts
are defined; M = 5 per model specification. RMSE takes the square root before
averaging local scores. This supersedes the earlier three-subsection layout.
No evaluation code changed. `inference/evaluate_local_global.py::aggregate`
currently uses observation-count weighting and derives RMSE from pooled MSE;
it must be aligned with this reporting convention before using its output as
an equal-weight local summary. Experimental Results remains next.
Two direct SyncTeX-enabled LaTeX passes rebuilt the adjacent 35-page PDF.
No undefined references or new layout warnings occur in the added paragraph;
existing overfull warnings at lines 77-78, 259, and 698-699 remain.
The added source has no trailing whitespace; existing user whitespace remains.
Fresh-context review `judge/reviews/thesis_evaluation_equations_review.json`
passed 100/100 with no findings and validated JSON.

## Thesis Elfwing citation placement (2026-10-02)

Added `\cite{elfwing2017}` immediately after Elfwing, Uchibe, and Doya in
Implementation (line 740), removing the stray possessive. Two direct
SyncTeX-enabled LaTeX passes regenerated the adjacent 35-page PDF; the
citation resolves to reference 25 without undefined-reference warnings.
Existing user wording is preserved elsewhere. The changed line adds no
whitespace error; five existing trailing-space lines remain. Existing
layout warnings are outside the changed paragraph. Fresh-context review
`judge/reviews/thesis_elfwing_citation_review.json` passed 100/100 with no
findings and validated JSON. No temporary judge files were created.
No model or data changes;
research plan position remains Experimental Results next.

## Thesis SARIMA fitting wording (2026-10-02)

Simplified the Implementation SARIMA Fitting bullet: Powell searches parameter
values to maximize the combined training-segment log likelihood; the Kalman
filter evaluates each candidate. Global/local pooling, first-segment starting
values, and the 100-iteration cap are retained. No fitting code changed.
User edits elsewhere in the thesis are preserved. Two direct SyncTeX-enabled
LaTeX passes regenerated the adjacent 35-page PDF; page 29 was visually checked.
Only earlier overfull warnings at lines 77-78 and 259 remain. Fresh-context
review `judge/reviews/thesis_sarima_wording_review.json` passed 100/100 with no
findings and validated JSON. `git diff --check` reports three existing user
trailing-space lines (GARCH/RFSV inputs and Adam learning rate); this task adds
none. Temporary judge diff and preview files were removed after review.
Research plan position is unchanged; Experimental Results remains next.

## Thesis Input data bold labels (2026-10-02)

All nine model bullets in Implementation's Input data now begin with a bold
`Input:` label after the model name. The user's invalid `\bold` command was
replaced with `\textbf`; SARIMA's variance input is explicit again. Existing
user edits to the population and AR description are preserved. Brief GARCH,
LSTM, SiLU, and HARNet phrasing adjustments keep the labels within the margins.
No model, data, or fitting behavior changed. Two direct SyncTeX-enabled LaTeX
passes rebuilt the adjacent 36-page PDF; only the preexisting overfull warnings
at lines 77-78 and 259 remain. Nine-label source checks and `git diff --check`
passed; Input data pages 27-28 were visually inspected. Fresh-context review
`judge/reviews/thesis_input_labels_review.json` passed 100/100 with no findings
and validated JSON. At the user's request, task-created judge previews, focused
diffs, and temporary check notes were deleted after review; review JSONs remain.
Research plan position is unchanged; Experimental Results remains next.

## Thesis Implementation model setup (2026-10-02)

Implementation in `ref/final_report/thesis_structure.tex` now contains Input
data, Hyperparameters, and Fitting in that order, with separate model bullets
and one bullet per hyperparameter. It documents the selected global cohort
and five local tickers, 20/80 variants, units/transforms, training-only
estimation and initialization, validation checkpoint selection, and the
QLIKE/MSE distinction. The SiLU derivative source (Elfwing et al., printed
Equation 10) is in the bibliography, references, and notes; the recurrent
instability mechanism is qualified as a hypothesis, with input/output changes
acknowledged. `ref/implementation_plan/implementation.md` records the full
numbered decisions and verification requirements as additionally requested.

Code verification clarified that selected SARIMA/GARCH evaluation carries
preceding filtering state; window 20 controls target eligibility. The stale
two-layer MLP roadmap entry remains superseded by current 128 -> 128 -> 2
hidden layers. Checkpoints are absent in this workspace, so their contents
remain unverified; selectors, launchers, code, saved reports, and prior handoff
records support the described settings. Methods prose is unchanged. No
model, dataset, interface, fit, or evaluation behavior changed.

Two final direct `pdflatex -synctex=1` passes regenerated the adjacent 36-page
PDF and SyncTeX. Rendered pages 26-32 were inspected. Exact subsection/bullet
checks (9 input, 17 hyperparameter, 9 fitting bullets), unchanged-Methods
comparison, citation/reference log checks, and `git diff --check` passed.
Only preexisting overfull warnings at source lines 77-78 and 259 remain.
Fresh-context cold review `judge/reviews/models_setup_review.json` passed
100/100 with no findings, and its JSON validated. The review corrected an
earlier draft's failure attribution: the documented SiLU QLIKE event rejected
nonfinite or nonpositive validation forecasts; the nonfinite-gradient event
belonged to a separate MSE exponential-head trial. Research plan position is
unchanged; next thesis task is Experimental Results, with matched-observation
and RFSV final-date comparison limits explicit before interpreting scores.

## Thesis equation-block numbering (2026-10-02)

All nine multiline `align` displays now use `aligned` inside `equation`,
with one equation number per block. Former row labels were consolidated
and internal references updated. MLP's scalar readout starts at the same
left edge as its first hidden-layer equation. Mathematical contents and
model/data behavior are unchanged. Two direct `pdflatex -synctex=1` passes
regenerated the adjacent 31-page PDF and SyncTeX. Label/reference checks
passed; rendered activations, metrics, MCS, MLP, and LSTM blocks were checked.
Only the earlier overflow warnings remain (source lines 77-78 and now 259).
Fresh-context review `judge/reviews/thesis_equation_blocks_review.json`
passed 100/100 with no findings and validated JSON.
Research plan position is unchanged; Implementation remains next.

## Thesis variance notation unified (2026-10-02)

At the user's request, variance-valued lowercase `h(t)` is now `v(t)` from
Background Equation 2.6 onward, including ARCH/GARCH, AR/SARIMA forecasts,
and the RFSV historical kernel. Neural hidden-state `h(t)`, convolution
filters, and uppercase `H` retain their meanings. AR/SARIMA use `v(t)` for
the observed proxy and `hat v(t+1)` for its forecast; GARCH explicitly states
that its `v(t)` is conditional return variance. The Methods plan's notation
rows match the source. No model, data, or evaluation behavior changed.
Two direct `pdflatex -synctex=1` passes regenerated the adjacent 31-page PDF
and SyncTeX. Background pages 9-10 and Methods pages 23-25 were visually
checked; only earlier overfull warnings at lines 77-78 and 253 remain.
Fresh-context review `judge/reviews/thesis_variance_v_review.json` passed
100/100 with no findings and validated JSON.
Research plan position is unchanged; Implementation remains next.

## Thesis rebuild and automatic preview configuration (2026-10-01)

Fixed missing math delimiters around `p_{\mathcal M}` in the user's MCS
paragraph, which blocked compilation. Two successful direct
`pdflatex -synctex=1` passes from `ref/final_report/` regenerated the adjacent
32-page PDF and SyncTeX. The final log has no undefined citations/references,
rerun requests, or fatal errors; existing overfull and underfull warnings remain.
Project VS Code settings now enable builds on file changes, LaTeX autosave
after 1500 ms, the internal PDF tab, and no automatic cleanup. The existing
two-pass recipe and Perl-free glob cleanup are preserved. Settings JSON and
final-log checks passed. Live editor refresh could not be exercised from this
terminal; PDF rendering was unavailable because PyMuPDF is not installed.
Research plan position remains unchanged: Implementation is next.

## Thesis Methods drafted (2026-10-01)

Methods in `ref/final_report/thesis_structure.tex` now contains exactly Data
and Models, with AR, SARIMA, GARCH, HAR, RFSV, MLP, LSTM, SiLU-LSTM, and
HARNet in the agreed order. It describes the exchange-specific stock-universe
filters, sequential acquisition defaults, audited raw-history GK cohort and
chronological splits, positive-input windows, zero-target policy, and separate
forecast floor. Counts are saved audit figures: the source database and listing
CSVs are unavailable here. No data behavior, fitting, or metric result changed.
The MLP has three hidden layers in current code; the roadmap's two-layer
description is stale. RFSV uses a full positive-history approximation and its
existing evaluator uses final-date targets, so its results require separate
matched-date treatment. Cryptocurrency transfer remains intended work.

`ref/implementation_plan/methods.md` records every agreed decision, audited
count, default requiring validation, and the notation/code cross-check. Existing
primary-paper citations are reused; `ref/thesis/thesis_notes.md` qualifies
HARNet's implemented hierarchy and the current full-history RFSV variant.
Two successful direct `pdflatex -synctex=1` passes regenerated the adjacent
33-page PDF and SyncTeX after the final source edit. Pages 21-28 were rendered
and visually inspected. Only the earlier overfull warnings at lines 77-78 and
253 remain; references and citations resolve. Focused source/audit checks and
`git diff --check` pass. Fresh-context cold review
`judge/reviews/thesis_methods_review.json` passed 100/100 with no findings,
and its JSON validated. Next thesis-writing task:
Implementation, using saved experiment configurations for fitting procedures,
transforms, model settings, initialization, and reproducibility details.

## Thesis PDF recompiled from user edits (2026-10-01)

Two direct `pdflatex -synctex=1` passes from `ref/final_report/` successfully
regenerated the 25-page `thesis_structure.pdf` and its adjacent SyncTeX file
from the user's latest saved source. No thesis source was edited. MiKTeX exists
at `AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe`, outside PATH;
the sandbox attempt could not access its user configuration, and the approved
retry succeeded. The earlier missing-build blocker below is now resolved.
The log retains overfull warnings at lines 77-78 and 253 and bibliography
underfull warnings. Page 20 was rendered and visually checked: the updated MCS
display fits without overflow. Fresh-context cold review
`judge/reviews/thesis_recompile_review.json` passed 100/100 with no findings,
and its JSON validated. Next thesis-writing task: Methods.

## MCS hypothesis display formatting (2026-10-01)

The MCS display in `ref/final_report/thesis_structure.tex` now places the
bootstrap rejection condition beside the null, separated by horizontal
spacing, and keeps the alternative on its own line. Both equation labels and
statistical definitions are preserved, along with the preexisting local prose
edit below the display. The focused source check and `git diff --check`
passed. Two direct `pdflatex -synctex=1` attempts from `ref/final_report/`
failed because `pdflatex` is unavailable in this environment; the earlier
MiKTeX installation recorded below could not be found. The PDF and SyncTeX
remain unchanged, and rendered legibility and overflow are unverified.
Fresh-context cold review
`judge/reviews/thesis_mcs_inline_rejection_review.json` validated at 90/100,
FAIL, with the missing build and rendered inspection as its critical blocker.
Next: run the required two-pass build, inspect this display, and obtain a
passing rereview when LaTeX is available; Methods remains the next
thesis-writing task.

## Thesis Model Evaluation and MCS (2026-09-30)

Section 2 now ends with a Model Evaluation subsection based on Hansen, Lunde, and Nason (2011). It defines matched per-model losses, pairwise loss differences, the paper's Equation (1) equal-performance null, the unnumbered Section 3.1.2 standardized excess-loss statistic and matching elimination rule, the bootstrap sequence, and the asymptotic coverage interpretation. A subsequent formatting refinement places the alternative hypothesis and bootstrap rejection condition directly below the null. The original paper was added to the thesis bibliography and `ref/thesis/references.md`, with concise findings in `ref/thesis/thesis_notes.md`. No model, data, fit, or evaluation result changed. Two direct MiKTeX passes with SyncTeX regenerated the 25-page PDF; page 20 was visually checked with the new display. The log reports only earlier overfull lines 77-78 and 253. A tiny numerical excess-loss identity check passed. Fresh-context reviews `judge/reviews/thesis_mcs_model_evaluation_review.json` and `judge/reviews/thesis_mcs_hypothesis_review.json` both passed 100/100 with no findings and validated JSON. Next thesis-writing task: develop Methods, including the project-specific MCS bootstrap and aggregation decisions.

## Thesis local and global complexity (2026-09-30)

The Background subsection now defines one fitted parameter vector per series for a local method and one shared vector for a global method, with explicit forecast, parameter-count, and hypothesis-class cardinality equations. It follows only Montero-Manso and Hyndman (2021), separating their Proposition 1 existence result from the Section 3.4 equal-bound complexity comparison and its assumptions. `ref/thesis/thesis_notes.md` records the paper-specific distinctions. No data, fit, or evaluation changed. Two direct MiKTeX passes with SyncTeX from `ref/final_report/` regenerated the 25-page `thesis_structure.pdf` from its `.tex` source and preserved `thesis_structure.synctex.gz`. Pages 19-20 were visually checked in the earlier build; the direct build has only earlier overfull lines at source 77-78 and 253. `AGENTS.md` now requires direct, same-directory thesis builds, and the three empty temporary `.build_*` folders were removed. Fresh-context cold reviews `judge/reviews/thesis_local_global_complexity_review.json` and `judge/reviews/thesis_pdf_workflow_review.json` passed 98/100 and 100/100, respectively; both JSON files validated. Next thesis-writing task: Methods.

## Thesis volatility-proxy concepts (2026-09-30)

The Volatility Proxies subsection now defines latent spot volatility and daily
integrated variance following HARNet, then presents daily unannualized RV,
Parkinson, and reduced Garman--Klass variance equations in that order. It
explains RV consistency as within-day sampling grows. The intraday return now
has its own numbered equation; the Brownian squared-range identity behind
Parkinson and the moment calculation for the reduced Garman--Klass estimator
each have displayed equations. Andersen and Bollerslev (1998) was added to the
thesis bibliography
and `ref/thesis/references.md`; `ref/thesis/thesis_notes.md` records HARNet's
relevant equations. No data, estimator implementation, fit, or comparison was
changed. Two MiKTeX passes generated a 23-page PDF; pages 18-19 were visually
checked, with only older overfull lines 76-77 and 252. Next thesis-writing task:
Local and Global complexity. Cold review remains pending: two fresh review
agents failed when the service reported its usage limit, so no passing review
may be claimed for this section.

## Thesis RFSV predictor and c(H) equations (2026-09-30)

The RFSV Section 5.2 variance predictor and the definition of c(H) now have
separate numbered equations (`eq:rfsv-variance-predictor` and `eq:rfsv-c-h`) in
`ref/final_report/thesis_structure.tex`. This is a formatting change only; no
formula, model fit, data, or evaluation changed. Two MiKTeX passes built the
PDF, and page 13 was visually checked: the equations render as (2.38) and
(2.39). The build log has the earlier unrelated overfull lines at source 76-77
and 252. Next thesis-writing task: Volatility Proxies.
Fresh-context cold review `judge/reviews/thesis_rfsv_c_split_review.json`
passed 100/100 with no findings, and its JSON validated.

## Thesis RFSV Gaussian-moment scale explanation (2026-09-30)

The Statistics Concepts RFSV parameter-estimation passage now states q>=0 for
the empirical moment, defines the fractional Brownian Gaussian absolute moment
K_q, derives the approximate nu-scaled log-volatility increment moment under
small mean reversion, and separates slope qH from intercept log(nu^q K_q).
It explains K_2=1 and the resulting nu-squared estimate from the q=2 intercept;
the existing Section 5.2 variance predictor remains. This is thesis prose only:
no model fit, data, or evaluation changed. Two MiKTeX passes built a 22-page
PDF; pages 13-14 were visually checked. The build logs still report the older
overfull lines at source 76-77 and 252, outside this RFSV edit. Next thesis
writing task: Volatility Proxies.
Fresh-context cold review `judge/reviews/thesis_rfsv_kq_review.json` passed
100/100 with no findings, and its JSON validated.

## Thesis Gaussian NLL objective notation (2026-09-30)

The Statistics Concepts parameter-estimation passage now separates Gaussian
log likelihood with `\theta^*\in\operatorname*{arg\,max}\log L(\theta)` from
negative log likelihood loss with
`\theta^*\in\operatorname*{arg\,min}\mathcal L_{\mathrm{NLL}}(\theta)`.
It explains the optimizer's distribution-based criterion, with GARCH as the
changing-variance example. No data, fit, forecast, or evaluation changed. A
two-pass MiKTeX build produced a 21-page PDF; page 12 was visually checked,
and only the preexisting source 76-77 overfull line remains. The first cold
review was interrupted because the user requested this equation split before
it finished. Next thesis-writing task: Volatility Proxies.

## Thesis parameter-estimation notation revision (2026-09-30)

The Statistics Concepts parameter-estimation passage now defines beta and
qualifies OLS as BLUE under Gauss--Markov assumptions. The Gaussian likelihood
uses sigma-squared and explains minimizing negative log likelihood. The RFSV
moment definition now follows Gatheral et al. Section 2.1's regular-grid
notation, with an explicit N=floor(T/Delta); an added q=2 intercept equation
states nu-squared estimation, and the paper's unnumbered Section 5.2 variance
predictor shows where nu-squared enters. Penn State's Gauss--Markov source was
added to the thesis bibliography and `ref/thesis/references.md`. No data, fit,
forecast, or evaluation changed. The PDF compiled in two MiKTeX passes and
pages 12-13 were visually checked; only the preexisting overfull line at source
76-77 remains. Next thesis-writing task: Volatility Proxies.
The first cold review passed 97/100 but found commas accidentally rendered in
two RFSV estimator superscripts. Those commas were removed, and a new two-pass
PDF build and page-13 visual inspection confirmed clean squared symbols. The
first review covers the superseded draft. Fresh-context final cold review
`judge/reviews/thesis_parameter_revision_final_review.json` passed 100/100 with
no findings, and its JSON validated.

## Thesis Background: statistical parameter estimation (2026-09-30)

Parameter Estimation now appears as a subsubsection of Statistics Concepts in
`ref/final_report/thesis_structure.tex`. It explains OLS for AR/HAR, conditional
Gaussian log likelihood for models including GARCH, the Kalman filter's role in
SARIMA likelihood evaluation, and RFSV moment regressions for H and nu-squared.
It distinguishes fitted coefficients from chosen model orders, constraints, and
windows. Kalman's primary paper was added to the thesis bibliography and
`ref/thesis/references.md`. This is thesis prose only; no data, fit, forecast,
or evaluation changed. Two MiKTeX passes compiled a 21-page PDF, and pages
12-13 were visually checked. The newly introduced overfull equation was fixed;
the remaining overfull line at source 76-77 predates this step. Next thesis
writing task: Volatility Proxies, then Local and Global complexity.
Fresh-context cold review `judge/reviews/thesis_parameter_estimation_review.json`
passed 100/100 with no findings, and its JSON validated.

## Thesis optimizer objective and gradient descent displays (2026-09-30)

The ML Concepts optimization passage now displays the generic training-loss
argmin (`eq:training-objective`) and the gradient descent update
(`eq:gradient-descent`). It defines argmin, the training batch, and learning
rate, while preserving the previous update rule. No fit, target, data, or
metric changed. The thesis PDF was rebuilt in two MiKTeX passes from the
latest saved source; page 14 was visually checked for both equations. The
earlier 100/100 gradient-display review applies to the superseded passage.
Fresh-context cold review `judge/reviews/ml_optimizer_argmin_review.json`
passed 100/100 with no findings, and its JSON validated. Next thesis-writing
task remains Parameter Estimation and Volatility Proxies.

## Thesis PDF metric bullets rendered (2026-09-30)

The ML Concepts source already had five `itemize` entries for MAE, MASE, MSE,
RMSE, and QLIKE, while the checked-in `ref/final_report/thesis_structure.pdf`
was older and still showed the superseded combined paragraph. Two MiKTeX
passes completed in a temporary output directory; the refreshed PDF was copied
to `ref/final_report/thesis_structure.pdf`. Page 13 was inspected visually and
shows all five distinct bullets below the unchanged metric equations. The
current thesis source was not edited for this correction; unrelated artifacts
remain untouched. Fresh-context cold review
`judge/reviews/ml_metric_pdf_review.json` independently rendered page 13 and
passed 100/100 with no findings; its JSON validated. Next thesis-writing task
remains Parameter Estimation and Volatility Proxies.

## Thesis ML Concepts wording revision (2026-09-30)

The ML Concepts text now uses `x` for neural-layer inputs, separates the five
metric effects into bullets, and states the generic training objective as an
argmin of mean per-example loss. Convolution prose defines K as filter size,
calls its terms weighted mappings, and uses the requested Bai et al. TCN
wording. No data, model, target, or score changed. The earlier ML review covers
superseded prose. Next thesis-writing task remains Parameter Estimation and
Volatility Proxies; the separate matched five-stock comparison awaits
interpretation before an equal-budget scope claim.
Focused notation, environment, label, and citation checks passed. Fresh-context
cold review `judge/reviews/ml_concepts_wording_review.json` passed 98/100 with
no critical findings; its JSON validated. MiKTeX initialization remains
blocked under the sandbox, so the revised PDF has not been visually checked.

## Thesis Background: ML Concepts (2026-09-29)

`ref/final_report/thesis_structure.tex` now has four ML subsubsections:
neural networks and layers; losses, backpropagation, and optimization; temporal
convolutions; and model fitting. The metrics use generic observed and predicted
values, with a mean per-example loss equation and a plain-language naive scale;
project-specific scale selection belongs in Implementation. QLIKE's positive
domain and prediction limits are explicit. Convolutions follow HARNet's
ordinary/receptive-field/causal/dilated order, with TCN motivation from Bai et
al. Model fitting covers chronological train/validation/test roles, validation
hyperparameter selection, hidden width, batches, and epochs. This revision is
thesis prose only: no target, data selection, fit, or score changed. The earlier
98/100 cold review applies to the superseded text. Next thesis-writing task:
develop Parameter Estimation and Volatility Proxies; the model-comparison task
remains interpretation of matched five-stock scores before any equal-budget
scope claim.
Focused structure, citation, and QLIKE-limit checks passed. MiKTeX still could
not initialize under the sandbox, so no rendered PDF was checked. Fresh-context
cold review `judge/reviews/ml_concepts_revision_review.json` passed 98/100
with only that minor verification limit; its JSON validated.

## Five-stock local/global volatility evaluation (2026-09-29)

The saved 14 global fits and their five-stock local counterparts were scored
without retraining on NVDA, AAPL, NFLX, GOOG, and AMZN. Each windowed
LOCAL/GLOBAL row uses the same 8,800 eligible 2019-2025 ticker/date targets;
RFSV-full has five separate 2025-12-31 forecasts per scope. Test-date naive
MAE supplies each ticker/window's shared MASE scale, as specified in
`ref/implementation_plan/local_evals.md`; this is test-relative and differs
from the training-scaled convention in `losses.md`. Each fit's saved positive
forecast floor is applied; actual targets are unchanged by evaluation. The
28-row CSV and PNG table, 140-row stock audit, run log, and 168 residual PNGs
are under `imgs/eval/local_eval/`. The PNG follows the existing volatility
metrics figure and shows N per row; its loss colors exclude the five-date
RFSV rows. The historical global CSV remains untouched because
it covers a different population. Global Base LSTM-20 and local Base LSTM-20
have different training budgets, so a scope-only causal interpretation is
unsupported. The focused arithmetic check passed. Next: inspect the matched
five-stock comparison, then decide whether to run a controlled equal-budget
scope experiment; do not rank the five RFSV dates against full-period rows.

## Local statistical raw-history fits (2026-09-29)

AR(1), HAR-20, HAR-80, GARCH(1,1), SARIMA, and RFSV-full were fit
independently for NVDA, AAPL, NFLX, GOOG, and AMZN: 30 pre-2016-only
adjusted GK variance fits, with separate ticker-named JSON and plain-text
logs under `inference/checkpoints/<model>/garman-klass/local/<ticker>/`.
The windowed fits use the existing positive-input segment rules, ticker-specific
training floors, and the plan's window-20 or window-80 target counts. RFSV
uses each ticker's full preceding positive history and its own H and ν² from
pre-2016 observation-lag 1–400 log-volatility moments. The five scaling plots,
five-stock H chart, and summary/moments/zeta CSVs are in
`imgs/roughness_analysis/local/train/`. Local H values are NVDA 0.060615,
AAPL 0.091738, NFLX 0.037446, GOOG 0.078093, and AMZN 0.060418.
The source database was `D:/DBs/timeseries_analysis/history_coverage.db`
(23,995,736,064 bytes; mtime 2026-09-20 11:33:39 UTC). All 30 jobs exited
successfully; reloaded artifacts matched their ticker, source version, split
counts, finite parameters, convergence diagnostics, and local roughness CSVs.
The focused raw-statistical, RFSV, local-roughness, and 11 existing roughness
checks passed by direct `.venv` invocation; `pytest` is not installed. No
W&B run or validation/test score was produced. Global fits and unrelated
thesis edits were preserved. Next: score saved local and global fits on matched
ticker/date keys before comparing models. See
`ref/implementation_plan/local_training.md`.

## Full-history RFSV fit-only revision (2026-09-28)

`RFSV.forward()` now accepts a ticker's positive adjusted GK variance history,
computes the paper's equation (5.1) log-variance prediction with exact
observation-bin weights and oldest-value tail extension, then applies the
unnumbered Section 5.2 +2c(H)ν² correction and returns one variance forecast.
The older volatility-unit `forecast()` shares the same log-space kernel so tiny
positive volatility inputs do not underflow while squaring. The global
parameters are still read from pre-2016 pooled observation-lag roughness:
H=0.03375577838376565 and ν²=0.49527452891226625; no optimizer, new target,
or test outcome was used. The raw RFSV fit-only command now checks `forward()`
on every ticker's complete positive valid pre-2016 history, rather than a
20-observation cap, and saves a distinct artifact at
`inference/checkpoints/rfsv/garman-klass/global/raw_history_full/fit.json`.
The run checked 11,153,536 positive training rows across 2,712 of the 2,714
cohort tickers; two had no positive pre-2016 history. The source DB version,
floor 3.396062419686545e-13, parameters, history rule, and counts were
verified after reload. The prior `raw_history_w20/fit.json` SHA-256 remained
7701F144F059323B5C34AAD4C6C681A71F752F02EF9623DEA013088BE1157FEC.
No validation/test forecast or metric was produced. The existing RFSV-20 row
in `imgs/eval/volatility_test_metrics.csv` is historical and does not represent
the new full-history fit. Positive observations across invalid or zero rows
are compressed into observation time; this is a project approximation.
Focused equation, cohort-guard, fit-route, and tiny-volatility checks passed by
direct `.venv` invocation. Fresh-context cold rereview
`judge/reviews/rfsv_full_history_fit_rereview.json` passed 100/100 with no
findings and its JSON validated. See
`ref/implementation_plan/rfsv_full_history_fit.md`. Next: define the single
forecast cutoff/target per ticker before any full-history evaluation.

## Thesis Background: Statistics Concepts (2026-09-28)

The Background section in `ref/final_report/thesis_structure.tex` now covers
conditional moments, white noise, Brownian and fractional Brownian motion,
roughness, and mean-reverting fractional OU log volatility with citations. The
ML Concepts heading is reserved for a later writing pass. This documentation
change did not alter data, model fits, or forecast comparisons. The thesis
compiled in a temporary directory, and a fresh-context cold review passed
100/100 with no findings. Next thesis-writing task: develop ML Concepts;
the model-comparison task remains matched-date RFSV and baseline scoring.

## Raw-history Garman-Klass RFSV fit (2026-09-28)

The global, fit-only RFSV calibration for the 2,714-ticker raw-history cohort
completed without scoring validation/test or starting W&B. The full raw loader
matched 8,664,516 / 1,806,548 / 4,479,681 eligible window-20 train /
validation / test target dates. It saved a separate compact JSON artifact at
`inference/checkpoints/rfsv/garman-klass/global/raw_history_w20/fit.json`.
The artifact records H=0.03375577838376565, ν²=0.49527452891226625,
observation lags 1–400, forecast window 20, and the pre-2016 adjusted GK
variance floor 3.396062419686545e-13, plus source database version and
cohort/count metadata. These parameters were read from the preceding pre-2016
roughness outputs; RFSV has no separate iterative optimizer. The loader found
176,432 invalid source rows and 1,254,289 valid zero-variance rows across the
loaded period; no raw data or target policy was changed. The JSON was reloaded,
its parameters matched `rfsv_fit`, and a positive 20-value forecast fixture
was finite. The prior derived-table `.pth` checkpoint was preserved; the
derived-table fitting path now rejects a raw-cohort GK parameter source.
Focused raw-fit, forecast-equation, and cohort-guard checks passed. See
`ref/implementation_plan/rfsv_raw_fit.md`. Fresh-context cold review
`judge/reviews/rfsv_raw_fit_review.json` passed 100/100 with no findings.
Next: score RFSV and matched baselines on identical forecast dates.

## Raw-history Garman-Klass H and nu-squared re-estimation (2026-09-28)

The pre-2016 GK roughness and RFSV inputs were recomputed in place from the
current raw-history cohort rather than the earlier 1,525-ticker derived table.
The source database was `D:/DBs/timeseries_analysis/history_coverage.db`
(23,995,736,064 bytes, last modified 2026-09-20 11:33:39 UTC). The cohort rule
selected 2,714 tickers with >80 pre-2016 raw rows and a 2025-12-31 row;
12,397,872 raw pre-2016 rows were scanned, 11,153,536 valid positive adjusted
GK variance rows were retained, and 2,710 tickers contributed displacement
pairs. Within-ticker observation lags 1–400 supplied 4,250,253,253 pooled
displacement pairs across lags. With q={1,1.5,2,3,4}, the origin-constrained
ζ(q)=Hq fit gave H=0.03375577838376565 (uncentered R²=0.9612926244466938).
The q=2 free-intercept log-moment fit gave ν²=0.49527452891226625.
These replace the old GK H=0.03246302891466224 and ν²≈0.44473011 for RFSV
parameter reading; they are different-population estimates, not forecast
improvements. The three pre-2016 training CSVs and GK/global H plots under
`imgs/roughness_analysis/global/train/` were updated. The Parkinson row and
plot retain the earlier derived cohort, labeled on the shared H plot. Full-
period plots, raw data, target floor, and validation/test window sets were not
changed. The old derived-table RFSV checkpoint remains tied to its old cohort.
The focused 11-test roughness suite and saved-artifact/RFSV-reader checks
passed. Fresh-context cold review `judge/reviews/raw_gk_roughness_review.json`
passed 100/100 with no findings. See `ref/implementation_plan/raw_gk_roughness.md`.
Next: fit RFSV on the raw cohort using these parameters before a shared-date
forecast comparison.

## Sigma-LSTM z-scored GK log-volatility input (2026-09-28)

At the user's request, the raw-volatility min-max window-20 online W&B run
`ocpgynhd` was stopped during epoch 2 after epoch 1 validation QLIKE
1.2234348299824669. Window 80 did not start. This interrupted run is an
experiment record, not a completed result. The current sigma-LSTM input is
GK log volatility z-scored using the global mean and population standard
deviation of positive valid pre-2016 rows only;
the next-observation log-volatility target and variance-unit QLIKE are
unchanged. The target and model output are not scaled. Training QLIKE uses
`r = 2 * target - max(2 * output, log(floor))` and averages
`expm1(r) - r`. The focused synthetic training/inference checks passed in `.venv`,
and cold review `judge/reviews/sigma_lstm_zscore_input_review.json` passed 100/100.
The full-cohort preflight found 2,714 tickers, 11,153,536 eligible pre-2016
input rows, mean -4.17238373979845, population standard deviation
0.8209776735142315, and GK variance floor 3.396062419686545e-13. The
window-20 loader reproduced 8,664,516 / 1,806,548 / 4,479,681 train /
validation / test windows. The online W&B run `aqezti5x` completed all 20
epochs at `https://wandb.ai/personalfeb/timeseries-volatility/runs/aqezti5x`.
Its log is `wandb/sigma_lstm_val_training/logs/sigma_lstm_raw_gk_zscorelogvol_w20_h128_lr0p001_20e_qlike_softplus_noclip_online.log`.
The best checkpoint was epoch 10; the final 1,806,548-observation validation
pass scored QLIKE 1.2217863932196134 and MASE 2.116889268968172. This is
weak validation performance for the current configuration, but no matched
naive forecast has yet been scored on the same observations. No window-80
run was started and test data were not scored. An initial
sandboxed launch could not connect to W&B and was terminated before training;
its log is retained with `.sandbox_blocked.log`. Next: compare against a
matched simple baseline before judging the architecture; assess return inputs
only as a separate experiment if requested.

## Historical sigma-LSTM min-max GK log-volatility input (2026-09-28)

The preceding implementation scaled log-volatility input with train-only
min-max bounds inferred from raw-volatility extrema. The user corrected this
to z-score standardization before any training with that input variant.

## Historical sigma-LSTM min-max GK volatility input (2026-09-28)

The user canceled the softplus-gate, unscaled-log-volatility-input queue. Its
window-20 process tree was stopped in epoch 7 after batch 17,000; six epochs
completed, with best validation QLIKE 1.2220036443261122 at epoch 1. The last
logged gradient norm was 36,508.10, and window 80 never started. W&B run
ste59xtc and its logs remain an interrupted trial.

The current sigma-LSTM input is one daily unannualized raw GK volatility
sqrt(RawVariance), scaled by one global training-only min-max transform.
Positive valid pre-2016 source rows supply the minimum and maximum; the
output and target remain next-observation GK log volatility, with the same
variance-unit QLIKE and prediction floor. No clipping is applied to
validation/test inputs outside the training range. The full cohort has
11,153,536 eligible training rows, volatility minimum 5.827574469439704e-7
and maximum 3.863512528781642. The scaled training median is 0.00393596239,
and 87.8041% of training values are below 0.01, so the global maximum
compresses most inputs. Exact train/validation/test window counts remain
8,664,516 / 1,806,548 / 4,479,681 for window 20 and
7,004,011 / 1,692,154 / 4,295,773 for window 80. The transform and
statistics are saved for checkpoint inference. Synthetic training and
inference checks passed, and cold review
`judge/reviews/sigma_lstm_minmax_input_review.json` passed 100/100. At the
user's request, a standalone window-20 online W&B run started under PID 7768
at `https://wandb.ai/personalfeb/timeseries-volatility/runs/ocpgynhd`.
It passed the exact cohort count gate and logged epoch-1 batch 1,000 on
compiled CUDA. Window 80 was not started or scheduled in this run. Next:
monitor window-20 validation and numerical stability.

## Sigma-LSTM softplus gate restart (2026-09-27)

Historical unscaled-input trial; the current input decision is recorded above.

The user stopped the active ReLU-gate window-20 queue at epoch 12 batch 6,000;
11 epochs had completed, with best validation QLIKE 1.2217863924241605 at
epoch 10. The process tree was terminated and window 80 had not started. This
run was online W&B yfyyab8y and remains an interrupted trial, not a completed
20-epoch result.

The current cell now sets gate_variance =
softplus(output_gate(memory_t.square())). Its Gaussian draw uses
sqrt(gate_variance + torch.finfo(dtype).tiny) to avoid sqrt(0) NaN gradients
when float32 softplus underflows at extreme negative scores. Moderate negative
scores still pass gradients. All earlier GK log-volatility, QLIKE, no-clipping,
width-128, seed-42, and sequential window-20/window-80 choices remain.
New run names include softplus so prior runs and checkpoints are preserved.
The user explicitly requested restarting both sequential online W&B runs
after this change. Focused eager and compiled CUDA checks passed; cold review
`judge/reviews/sigma_lstm_softplus_review.json` passed 99/100, and its sole
stale launcher-path finding was corrected afterward. The new queue is active
under PID 25148. Window-20 online W&B run `ste59xtc` is at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/ste59xtc` and had
logged epoch-1 batch 6,000 at last check, beyond the no-clipping exponential
gate's batch-2,002 failure. The queue starts window 80 only if window 20 exits
successfully. Next: monitor both runs and record best validation results.

## Sigma-LSTM ReLU gate experiment (2026-09-27)

Historical ReLU-gate trial; the current softplus gate and input are recorded above.

At the user's request, the current sigma-LSTM gate variance is
`ReLU(output_gate(memory_t.square()))`, sampled with a standard-normal draw
times its square root. Negative raw gate scores have zero gradient through
ReLU; shared weights or upstream memory changes can still move a score across
zero. Positive scores close to zero have a large square-root derivative and
remain a stability risk without gradient clipping. The prior exponential gate produced nonfinite memory in the window-20
no-clipping diagnostic at batch 2,002. The raw GK log-volatility input,
next-observation QLIKE target, no-clipping setting, width 128, and sequential
window-20 then window-80 plan remain. The new W&B run names include `relu` so
they do not overwrite the failed exponential-gate logs or checkpoints.
Focused eager and compiled CUDA gate-gradient checks passed. Cold review
`judge/reviews/sigma_lstm_relu_gate_review.json` passed 97/100 with the
near-zero positive-gradient risk documented above. The sequential online
W&B queue is active under PID 6584. Window-20 run `yfyyab8y` is at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/yfyyab8y`. It
completed epoch 1 without nonfinite values: train QLIKE 1.25549937,
validation QLIKE 1.22201225, validation MASE 2.14319817, and best epoch 1.
This is only an early validation result, not a baseline win. Epoch 2 had
logged batch 19,000 at last check. The queue starts window 80 only after
window 20 exits successfully.

## Sigma-LSTM sequential validation queue (2026-09-27)

Historical exponential-gate trials; the current ReLU gate is recorded above.

The sigma-LSTM uses only adjusted GK log volatility, with window sizes 20 and
80, hidden width 128, QLIKE, Adam 0.001, batch 128, up to 20 epochs, patience
10, seed 42, and one stochastic evaluation pass per split. The full cohort
window-20 count gate passed: 8,664,516 train / 1,806,548 validation /
4,479,681 test windows. The window-80 run has not started.

Online W&B window-20 run 887clrnx failed at epoch 1 batch 5,282 with finite
loss and nonfinite gradients. The gate standard deviation was changed from
sqrt(exp(z)) to the equivalent exp(0.5 z), which removed a float32 underflow
gradient path; cold review sigma_lstm_stablegate_review.json passed 100/100.
Online W&B window-20 run eo7w4ov6 then failed at epoch 1 batch 12,301 before
validation. An offline replay found finite parameters, 127/128 finite model
outputs, and output-gate log-variance ranging from -606 to +350. The positive
extreme makes exp(0.5 z) overflow float32 in the forward pass.

The user has now removed sigma-LSTM gradient clipping; both failed runs used
norm-1 clipping, while the earlier Base LSTM validation scripts disabled it.
The sigma-LSTM entry point now records clip_norm=None, and the sequential
queue passes --no-grad-clip explicitly. Online W&B run m1fbr9n8 failed at
epoch 1 batch 2,187 with finite loss about 1.29e28 and nonfinite gradients;
the pre-update gradient norm was 79.53 at batch 2,000. The queue stopped and
window 80 did not start. Removing clipping worsened stability and did not
resolve the gate issue. Cold review sigma_lstm_noclip_review.json passed
100/100. Next: settle the gate's numerical rule before another full run;
preserve the failed runs and their logs. The
Base LSTM also used an intraday-return input, so its scores do not isolate
architecture effects.

A requested no-clipping memory diagnostic used the same window-20 cohort,
compiled CUDA, and online W&B. The successful diagnostic log is
`wandb/sigma_lstm_val_training/logs/sigma_lstm_memorytrace_noclip_v3_online.log`
(W&B run `9fuxbbhy`). It failed at epoch 1 batch 2,002: across all 20 steps,
128 observations, and 128 hidden components, finite `memory_t` entries had
minimum -10.5705366, maximum 12.8468399, and mean 0.1658676. There were
327,053 finite and 627 nonfinite entries; the first nonfinite entries appeared
at step 16. Model parameters remained finite. The earlier two diagnostic
attempts `qoef6455` and `aulh7zvq` failed before a usable finite-value
summary; their logs are retained. No validation epoch completed.

## Global linear HAR-80 fit (2026-09-27)

After the pooled SARIMA fit converged, a separate `models/har_80.py` was added
for an intercept plus 1-, 5-, 20-, 40-, and 80-observation means of adjusted
daily unannualized Garman-Klass variance. Global OLS used the same 2,714 raw
equity tickers, pre-2016 targets, and positive-input window rule as the ML
path, now with window 80. Eligibility matched 7,004,011 train / 1,692,154
validation / 4,295,773 test target dates, and the pre-2016 floor remained
3.396062419686545e-13. The six coefficients were saved at
`inference/checkpoints/har_80/garman-klass/global/raw_history_w80/fit.json`;
reload confirmed finite parameters and convergence (normal-matrix condition
1495.5121024582847). No W&B run or validation/test scoring was performed.
The direct coefficient fixture and full cohort count check passed. See
`ref/implementation_plan/har_80_raw_fit.md`. This 80-window fit excludes
1,660,505 train, 114,394 validation, and 183,908 test target dates relative
to window 20; compare forecasts only on shared dates. Next: complete the
fresh-context cold review, then estimate H and nu-squared from this cohort
before fitting RFSV.

## Sigma-LSTM model definition (2026-09-27)

Historical original definition; the current ReLU gate is recorded above.

`models/sigma-lstm.py` defines the project-adapted GK log-volatility cell.
`SigmaLSTMCell` inherits `FEBCellLSTM`; `SigmaLSTM` inherits `FEBLSTM`. The
paper's additive cell update remains, with h(0) = 0 and C(0) = 1. The Gaussian
output-gate variance is exp(W_o[C(t)^2]) rather than the earlier softplus
mapping, while its draw still uses the square root of variance. The main head
is interpreted as next-observation log volatility; its future QLIKE path must
convert this to GK variance with exp(2 * head output). The auxiliary squared
mean memory is interpreted as log-volatility variance but is not calibrated.
Window 20 and 80, width 128, seed 42, the target/floor policy, and remaining
evaluation choices are listed separately in `ref/implementation_plan/sigma-LSTM.md`.
This definition was later connected to the raw-history trainer for the queue
described above. The user confirmed no auxiliary loss or calibration.

## Global raw-history statistical fits (2026-09-27)

The fit-only statistical path now uses the 2,714-ticker adjusted raw-history
Garman-Klass cohort and the ML window-20 target keys. Full-data eligibility
matched 8,664,516 train / 1,806,548 validation / 4,479,681 test windows;
the pre-2016 positive floor is 3.396062419686545e-13. The fitted target is
next-recorded-observation daily unannualized variance, with zero valid targets
floored. AR(1) and HAR pooled OLS fits and GARCH(1,1) pooled return-residual
conditional-variance fit completed in separate compact JSON checkpoints under
`inference/checkpoints/<model>/garman-klass/global/raw_history_w20/`. Each
artifact was reloaded and checked for finite parameters and convergence;
GARCH also satisfies alpha + beta < 1. GARCH estimates conditional variance
of adjusted ln(Close/Open) residuals, not Garman-Klass forecast errors.
SARIMA pooled fitting completed and its reloaded artifact has five finite
parameters; Powell reported convergence after 6 iterations with summed
negative log likelihood -35304345.05181905. Its first-series seed had issued
a convergence warning, but the pooled optimizer subsequently converged.
No W&B run or validation/test
scoring was performed. The two-ticker fixture passed, and fresh-context cold
review `judge/reviews/global_statistical_raw_fit_review.json` passed 100/100.
The implementation choices are in
`ref/implementation_plan/global_statistical_raw_fit.md`. The subsequent
linear HAR(1,5,20,40,80) fit is recorded above. Estimate H and nu-squared
from this cohort before any RFSV fit. The interrupted derived-table RFSV
artifact was not changed.

## Window-80 MLP log-volatility validation trial (2026-09-27)

At the user's request, the completed window-20 MLP was rerun with only the
window changed to 80. The same 2,714-ticker raw adjusted daily unannualized
Garman-Klass variance cohort, input features [0.5 ln(adjusted GK variance),
ln(adjusted Close/Open)], next recorded observation target, QLIKE objective,
Adam 0.001, hidden width 128, batch 128, patience 10 with one tenfold-rate
retry, 20 epochs, seed 42, gradient clipping norm 1, compiled CUDA, online
W&B, and validation-only scoring apply. The flattened network input is 160
values and the architecture is 160 -> 128 -> 128 -> 2 -> 1. Its reported
train / validation / test window counts are 7,004,011 / 1,692,154 /
4,295,773, versus 8,664,516 / 1,806,548 / 4,479,681 at window 20;
direct window-length comparison needs a common scoring set. A direct shape
check passed. The visible launcher is
`wandb/mlp_val_training/logs/start_mlp_raw_logvol_w80.ps1`; online W&B run
`lhzx2v36` is at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/lhzx2v36`.
The original run completed epochs 1-11 and was stopped during epoch 12. Its
only saved best checkpoint is epoch 1 (validation QLIKE 0.43494748140134837,
optimizer rate 0.001); epochs 2-11 and partial epoch 12 have no recoverable
model state. At the user's request, training resumed from checkpoint epoch 1
in the same W&B run, so epochs 2-11 are being retrained and their old W&B
history remains visible as superseded. The original PowerShell UTF-16 log is
preserved; the eleven completed epoch rows were copied to UTF-8 JSONL for the
resume reader. An initial resume attempt exited before training because it
read the UTF-16 log as UTF-8. The successful launcher is
`wandb/mlp_val_training/logs/resume_mlp_raw_logvol_w80.ps1` and its live log
ends in `_resume_from_epoch1_retry.log`. It verified the same window counts,
W&B run ID, and compiled CUDA `start_epoch` 2. Next: monitor the resumed run
and record its best validation metrics. No test scoring was requested.

## Window-20 MLP log-volatility validation trial (2026-09-26)

At the user's request, the MLP launched after both window-80 LSTMs finished.
It uses the existing 2,714-ticker global raw-history adjusted daily,
unannualized Garman-Klass variance cohort, 20 valid input observations,
features [0.5 ln(adjusted GK variance), ln(adjusted Close/Open)], and the next
recorded observation target. The network flattens 20 x 2 inputs and follows
40 -> 128 -> 128 -> 2 -> 1 with SiLU between layers, as requested; the
user's intermediate-layer edit was retained and its `nn.linear` typo fixed.
QLIKE is the objective and checkpoint criterion; log-variance output is
exponentiated and floored from training data for positive variance scoring.
Other Base LSTM settings are Adam 0.001, batch 128, patience 10 with one
tenfold-rate retry, 20 epochs max, seed 42, gradient clipping norm 1,
compiled CUDA, online W&B, and validation-only scoring. Its window-20
forecast counts should match 8,664,516 / 1,806,548 / 4,479,681 train /
validation / test; the run log must verify them. Raw-history MLP wiring and
shape/inference checks passed. A fresh-context cold review passed 100/100 in
`judge/reviews/mlp_w20_prelaunch_review.json`. The visible launch script is
`wandb/mlp_val_training/logs/start_mlp_raw_logvol_w20.ps1`; run name is
`mlp_raw_gk_logvol_return_w20_h128_lr0p001_20e_qlike_online`. The process
reported the expected 8,664,516 / 1,806,548 / 4,479,681 windows, connected
to online W&B run `jaj5mioo` at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/jaj5mioo`, and
completed all 20 epochs. It selected epoch 2 by validation QLIKE
0.4601829778717594; independent best-checkpoint validation scoring on
1,806,548 windows gave QLIKE 0.460182977375786, MAE
0.0006497323079140628, MASE 1.019471213974424, MSE
2.6333397513918494e-05, and RMSE 0.005131607692908578. Its exit code was
zero. No test scoring was requested.

## Window-80 LSTM queue (2026-09-25)

After HARNet-80 completed, the user requested two sequential online W&B
validation trials. Base LSTM starts first with its previous two inputs
[0.5 ln(adjusted GK variance), ln(adjusted Close/Open)], log-volatility
target, QLIKE loss, width 128, Adam 0.001, batch 128, patience 10, 20 epochs,
seed 42, gradient clipping norm 1, and compiled CUDA. SiLU-LSTM is queued
after a successful Base run with its previous two inputs [raw adjusted GK
variance, ln(adjusted Close/Open)], raw-variance target and direct head, MSE
loss, and the same other parameters. Both use window 80, global raw history,
next recorded observation target, and validation-only scoring. The 2,714
tickers and split dates are unchanged; eligible train / validation / test
windows are 7,004,011 / 1,692,154 / 4,295,773 rather than 8,664,516 /
1,806,548 / 4,479,681 at window 20. Window lengths need a common scoring
set for direct comparison; Base QLIKE and SiLU MSE also differ in objective.
The raw-history window guard now admits 80 for the two LSTMs. Focused raw
window checks passed, and the fresh-context cold review passed 100/100 in
`judge/reviews/lstm_w80_prelaunch_review.json`. The visible queue launcher is
`wandb/lstm_val_training/logs/start_lstm_w80_queue.ps1`; Base began under
`base_lstm_raw_gk_logvol_return_w80_h128_lr0p001_20e_wandb`, and SiLU is
queued as `silu_lstm_raw_gk_variance_return_direct_mse_w80_h128_lr0p001_20e_wandb`.
Base W&B run `n8v4m22f` is online at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/n8v4m22f`; compiled
CUDA training completed 20 epochs and selected epoch 20 by validation QLIKE
0.4131435726971332. Independent best-checkpoint validation QLIKE on
1,692,154 windows was 0.4131435717333512. SiLU online W&B run `h5ux74yq`
at `https://wandb.ai/personalfeb/timeseries-volatility/runs/h5ux74yq`
completed 20 epochs and selected epoch 4 by validation MSE
1.7940556133998083e-05. Its independent best-checkpoint validation MSE was
1.7940556328516154e-05 and QLIKE 24894.404243868754 on the same window
count. Both exits were zero. No test scoring was requested.

## HARNet-80 raw-variance validation trial (2026-09-25)

At the user's request, HARNet-80 started in a visible PowerShell terminal with
the HARNet-20 trial's settings: global raw adjusted daily unannualized
Garman-Klass variance, next recorded observation target, QLIKE objective,
initial Adam rate 0.001, batch 128, patience 10 with one tenfold-rate retry,
20 epochs max, seed 42, gradient clipping norm 1, compiled CUDA, and
validation-only scoring. Its 80-observation window changes the eligible
forecast counts to 7,004,011 / 1,692,154 / 4,295,773 for train / validation /
test, so direct model comparison requires a common observation set. It starts
from a pooled pre-2016 HAR(1,5,20,40,80) fit on its eligible training windows.
The online run ID is `vyp9im9l` at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/vyp9im9l`;
the live log is
`wandb/harnet_val_training/logs/harnet_80_raw_gk_variance_w80_lr0p001_20e_qlike_online.log`.
It completed all 20 epochs, selecting epoch 18 by validation QLIKE
0.4307484029432806; independent best-checkpoint validation scoring on
1,692,154 windows gave QLIKE 0.43074840161835937, MAE
0.0005985247307848403, MASE 0.8254054735876607, MSE
1.8522101377075283e-05, and RMSE 0.004303731099531577. Earlier launch
attempts stopped before
training because `--wandb-id` requires an existing run for resume; no
checkpoint or epoch score came from them. Test scoring remains deferred.

## HARNet-20 raw-variance validation trial (2026-09-25)

The raw-variance SiLU-LSTM MSE run completed epoch 20 and selected epoch 16
by validation MSE 2.208466703535898e-05. Its independent best-checkpoint
validation score on 1,806,548 windows was MSE 2.2084667117207434e-05,
QLIKE 491809.2414945714, MAE 0.0006713999301386554, MASE
1.1684035882870252, and RMSE 0.0046994326377986775. This result has
12.57% training-batch floor hits in epoch 20 and should not be presented as
an improvement by QLIKE.

After epoch 20, the global HARNet-20 raw-history run started with adjusted,
daily, unannualized Garman-Klass variance as its sole input and next recorded
observation target. It uses the existing 2,714-ticker cohort and positive-input
window rule, with 8,664,516 / 1,806,548 / 4,479,681 train / validation /
test windows. Zero source targets receive the pre-2016 positive floor
3.396062419686545e-13; actual targets are otherwise unchanged. Forecasts
are floored at that value before QLIKE, with no upper cap. HARNet starts from
a pooled pre-2016 HAR(1,5,20) fit on eligible training windows. Settings are
Adam 0.001, batch 128, patience 10 with one tenfold learning-rate retry,
20 epochs max, seed 42, gradient clipping norm 1, compiled CUDA, QLIKE
training and validation checkpoint selection, and validation-only scoring.
The first launch, `harnet_20_raw_gk_variance_w20_lr0p001_20e_qlike_offline`,
logged epoch 1 validation QLIKE 0.5463413274 but stalled before saving a
checkpoint; it was stopped. Its local W&B ID is `kb3cqo84` and its log is
preserved in `wandb/harnet_val_training/logs/`. Online W&B access was verified
outside the sandbox. A fresh run with the same settings is active under
`harnet_20_raw_gk_variance_w20_lr0p001_20e_qlike_online` in a visible
PowerShell terminal. Its live console log is in the same logs directory;
online W&B run `u7ia0lvh` is at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/u7ia0lvh`.
It completed epoch 20 after the configured retry at learning rate 0.0001;
its best checkpoint is epoch 20 with validation QLIKE 0.4714980945508361
on 1,806,548 windows. No test scoring was requested. HARNet's
variance-only feature and QLIKE objective differ from the SiLU MSE trial;
their results are not a controlled architecture comparison. HARNet-80 is
the current training task.

## AAPL validation residual diagnostic (2026-09-25)

The existing AAPL residual histogram and dated CSV in `imgs/lstm_validation/`
were refreshed from the raw-history two-input base LSTM trained through epoch
20. Its saved best checkpoint is epoch 7. The 2016-2018 validation set has 754
AAPL forecasts; residual is adjusted daily, unannualized Garman-Klass variance
minus predicted variance. This is a single-ticker validation diagnostic; no
test scoring or new training was run. The next model task remains monitoring
the raw-variance SiLU MSE trial.

## Raw-variance SiLU MSE trial (2026-09-25)

The first raw-variance SiLU run mistakenly used QLIKE from the prior trials.
At the user's correction, it was stopped in epoch 1 after 43,000 of 67,692
training batches, with no completed epoch or checkpoint. Its W&B run is
`9znlrmc8`; its partial log is in `wandb/lstm_val_training/logs/` under the
`silu_lstm_raw_gk_variance_return_w20_h128_lr0p001_20e_wandb` name. Do not
report it as an MSE result.

The replacement SiLU-LSTM trial uses the same 2,714-ticker raw-history cohort
and window-20 validity rule as the log-volatility trials. The first feature
and target are adjusted daily, unannualized Garman-Klass variance; the second
feature is ln(adjusted Close/Open). The window counts remain 8,664,516 /
1,806,548 / 4,479,681 by train / validation / test, while values change.
The zero-target floor remains 3.396062419686545e-13. The corrected head
outputs raw variance directly. The requested training loss is mean squared
error between raw output and adjusted daily variance; the user's rationale is
to give large absolute errors more weight. Reported forecasts are floored for
positive-variance metrics, and checkpoint selection and patience use their
validation MSE. QLIKE remains a reported metric. This is a
user-directed departure from the roadmap's QLIKE objective; the cited
crypto-winter paper uses MSE on volatility, a different target. Hyperparameters
are hidden 128, Adam 0.001, window 20, batch 128, patience 10, 20 epochs max,
seed 42, gradient clipping norm 1, compiled CUDA, validation-only scoring,
and online W&B. An initial MSE run `wajx25z9` with the old exponential head
failed at batch 1,576 with nonfinite gradients, despite a finite loss; it has
no completed epoch or checkpoint. The direct raw-variance head replaces that
configuration, with no softplus. Focused checks and fresh-context cold review
passed (100/100) in `judge/reviews/silu_direct_mse_prelaunch_review.json`.
The fresh online W&B run `4lo1fsvf` is at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/4lo1fsvf` with logs
under `silu_lstm_raw_gk_variance_return_direct_mse_w20_h128_lr0p001_20e_wandb`.
It reported the expected 8,664,516 / 1,806,548 / 4,479,681 windows and
passed batch 4,000 of epoch 1 with finite raw-variance MSE and gradients.
No test scoring was requested. Next: monitor the run, then record the best
validation MSE and all five reported variance metrics or diagnose failure.

## Sequential raw-history log-volatility LSTM trials (2026-09-25)

The user capped both models at 20 epochs and requested online W&B. The base
LSTM first completed epochs 1 and 2 locally, with validation QLIKE
0.4574488582 and 0.4558720081. The local-only continuation was stopped in
partial epoch 3; its saved best checkpoint is epoch 2. The two complete epoch
rows were copied from local logs into W&B run `chrpzpxq` as explicitly marked
backfilled history. W&B synced them at
`https://wandb.ai/personalfeb/timeseries-volatility/runs/chrpzpxq`.
The online base continuation began at epoch 3 and completed epoch 20; best
checkpoint epoch 7 has validation QLIKE 0.4536128288, independently checked
at 0.4536128275 on 1,806,548 validation windows. The base checkpoint retains
its original `...30e` run-name directory for resume compatibility.
The subsequent SiLU run `advoqi3r` completed two epochs and failed while
scoring epoch 3. Validation QLIKE rose from 0.4671938337 at epoch 1 to
6,601,709.9547 at epoch 2. QLIKE then rejected a nonfinite or nonpositive
variance; the exact forecast was not logged. The best saved checkpoint is
epoch 1. See `wandb/lstm_val_training/logs/silu_lstm_raw_gk_logvol_return_w20_h128_lr0p001_20e_wandb.log`.

Both models use the 2,714-ticker raw-history Garman-Klass cohort, ordered
inputs [0.5 ln(adjusted GK variance), ln(adjusted Close/Open)], adjusted
log-volatility targets, hidden width 128, Adam rate 0.001, window 20,
batch 128, patience 10, seed 42, compiled CUDA, gradient clipping norm 1,
and validation-only reporting. No test scoring is scheduled. The active
launcher is `inference/run_raw_logvol_pair_wandb.py`; queue status is in
`wandb/lstm_val_training/logs/raw_gk_logvol_pair_wandb_queue.log`, and W&B
state is in `wandb/lstm_val_training/logs/raw_gk_logvol_pair_wandb_state.json`.
Subsequent training now logs the first offending forecast during scoring,
including split, ticker, target date, log-variance output, and variance value,
before raising the same validation error. A one-sample overflow check passed;
the old SiLU log cannot recover its missing forecast. The cited crypto-winter
paper's unnumbered Section 3.2 cell equations match this SiLU cell, but its
width, window, scaling, loss, and ensemble differ from this trial. Next:
inspect a new diagnostic run before deciding whether to alter SiLU training;
the test split remains deferred.

## Raw-history neural window data (2026-09-24)

The raw-history data path for base LSTM, SiLU-LSTM, HARNet-20, and HARNet-80
is implemented behind `--raw-history`; existing positive-table checkpoints
retain their path. The cohort is 2,714 tickers with >80 pre-2016 raw rows and
a 2025-12-31 row. Daily unannualized Garman?Klass variance is computed from
adjusted OHLC. Invalid rows or zero-variance input rows reject a window;
valid zero targets are adjusted to the smallest positive pre-2016 cohort
variance, 3.396062419686545e-13. Raw targets remain available for audit.
Target dates are the next recorded observation. Window-20 usable counts are
8,664,516 / 1,806,548 / 4,479,681 and window-80 counts are 7,004,011 /
1,692,154 / 4,295,773 for train / validation / test. Eleven adjacent raw
date gaps exceed seven calendar days; none exceeds 30. The coverage report
is in `ref/coverage/history_coverage.md`, and implementation notes are in
`ref/implementation_plan/data_fixes.md`. New runs use different observations
and adjusted-target metrics from earlier runs. No training was launched.
The full `TimeSeriesDataset` constructor matched the independently scanned
train / validation / test counts for both windows. The synthetic raw-window,
forecast, and legacy focused checks passed in the project virtual environment.
CBIO and VATE have no valid pre-2016 variance history. They have 2,134
window-20 and 1,912 window-80 test forecasts in coverage, but cannot have a
training-only ticker MASE scale. All five test metrics exclude those forecasts,
using 4,477,547 or 4,293,861 common observations. All other 2,712 cohort tickers have strictly positive pre-2016 adjusted
MASE scales. The fresh-context JSON cold review passed 100/100 with no findings at
`judge/reviews/raw_neural_window_review.json`; `python judge/validate.py`
validated it. Next: plan a baseline comparison on this scoring set and
launch neural training only after a separate request.

# Project context

## Raw-variance base LSTM with fixed observed-session horizon (2026-09-24)

At the user's request, the next trial returns to raw daily Garmanâ€“Klass
variance inputs/targets, with hidden width 128, initial Adam rate 0.001,
and at most 20 epochs. The prior two-input window-20, batch-128, patience-10,
no-clipping, compiled-CUDA, validation-QLIKE setup is retained. The model's
forecast head still outputs log variance for positive variance predictions.
Changing to raw inputs alone would not repair missing zero-variance days, so
the new optional `--consecutive-sessions` filter requires every input and
target date to be adjacent in the panel's observed market-session calendar.
It retains 4,755,276 training, 1,051,586 validation, and 2,543,967 test
windows, compared with 5,416,755 / 1,120,921 / 2,649,257 before filtering.
All retained target values remain unchanged. The calendar is saved in the
checkpoint for consistent subset inference; this run's metrics will not be
directly comparable to earlier unfiltered runs. A two-ticker one-epoch
compiled CUDA preflight and focused checks passed. New warning instructions
for data changes are in `AGENTS.md`. Next: fresh cold review, then launch
the full-panel validation-only trial; no test evaluation.

## Equity volatility data audit (2026-09-24)

A read-only audit of the 25-year Garmanâ€“Klass panel found 9,587,674 eligible
raw rows, of which 9,217,433 positive-variance rows are retained across
1,525 tickers. The training tail reaches variance 14.92673; several top
rows have isolated, suspicious adjusted OHLC lows that need source checking.
All 333,609 nonpositive derived variances are exactly zero, with flat OHLC;
none is negative. Removing these zero-variance days creates irregular targets: 13,550
of 5,416,755 training windows span more than seven calendar days and 895
span more than 30. A matched-date, symbol-based subset of 81 tickers from
the 93-stock liquid S&P 500 reference has a far lighter variance tail, but
it is not the paper's actual data or proxy. Full counts, examples, limits,
and next checks are in
`ref/implementation_plan/equity_volatility_data_review_2026-09-24.md`.
No data or model behavior was changed. Next: verify suspicious price rows
independently and settle the target horizon before another SiLU trial.

## Two-input base-LSTM validation results (2026-09-24)

The two-input hidden-128 base LSTM completed the 50-epoch cap. It
resumed from best epoch 25 after a 30-epoch stage, retrained epochs
26-30, and selected best epoch 43 by validation QLIKE 0.4691930072.
An independent validation forecast gave QLIKE 0.4691930059 on
1,120,921 observations. W&B `48e1ho51` and the final log are at
`wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_20epochs_rerun_resume_to50.log`.
The 30-epoch merge history is retained alongside logs. The focused
resume check and fresh-context prelaunch cold reviews passed (100/100).

The two-input SiLU-LSTM full-panel fit started in W&B run `qjefnxj6`
with its own checkpoint and log. It uses the same ordered features
[log(sqrt(Garman-Klass variance)), ln(adjusted Close/Open)], window 20,
hidden width 128, log-volatility target, variance QLIKE, batch 128,
rate 0.0001, patience 10, no clipping, compiled CUDA, and at most
50 total epochs. The log is
`wandb/lstm_val_training/logs/silu_lstm_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_50epochs.log`.
The run failed during epoch 20 after completing 19 epochs: the pre-loss
check rejected either nonfinite log variance or a conversion that was not
finite positive float32 variance. The failing batch output was not logged,
so the exact cause is unknown. The best saved checkpoint is epoch 15,
validation QLIKE 0.4740026653. The failed epoch-20 partial work was not saved.
The shared training helper now checks finite log variance without exponentiating;
forecast conversion still requires finite positive variance. This removes the
unnecessary variance conversion from training, but a future evaluation can still fail if a forecast
cannot be represented in variance units. The continuation started from best
epoch 15 in compiled CUDA mode, with epochs 16-19 superseded, and resumed
the same online W&B run `qjefnxj6`. Its new stdout/stderr logs are
`wandb/lstm_val_training/logs/silu_lstm_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_50epochs_resume_from15_to50_online.log`
and the matching `.err` file. An initial sandboxed start could not connect
to W&B; the network-enabled restart connected successfully. That attempt
completed epoch 16 (validation QLIKE 0.4743042324, worse than the saved
epoch 15) and failed while scoring epoch 17. The float32 exponentiation
used for variance-unit metrics produced a nonpositive or nonfinite value;
the exact offending prediction was not logged. Scoring and inference now
exponentiate finite log-variance outputs in float64, preserving the strict
finite-positive forecast check without imposing a forecast cap.
The prelaunch focused tests, two-ticker compiled CUDA preflight, and
fresh-context cold review passed (100/100). The log-variance check fix passed
focused checks and a separate fresh-context cold review (100/100) at
`judge/reviews/lstm_log_variance_check_review.json`. The float64 conversion
passed focused tests and a fresh-context cold review (100/100) at
`judge/reviews/lstm_float64_scoring_review.json`. The next compiled CUDA
continuation restarted at epoch 16 in the same W&B run; logs are
`wandb/lstm_val_training/logs/silu_lstm_gk_logvol_intradayret_w20_h128_lr0.0001_noclip_progress_50epochs_resume_float64_from15_to50.log`
and the matching `.err` file. It completed epochs 16 and 17 but failed
while scoring epoch 18: `error ** 2` overflowed float64 in the training
MSE calculation, so variance metrics became nonfinite. Epoch 17 already
had train QLIKE 4.7277346739, MAE 7.736279475e43, and MSE
2.168352975e94, while validation QLIKE was 0.4742159823. Best checkpoint
remains epoch 15 (validation QLIKE 0.4740026653). This is a large-forecast
instability, not the earlier float32 exponentiation failure. The run is
stopped; no further restart is scheduled until the instability is addressed.
No test-period evaluation.

The user retained standard variance-unit QLIKE after the proposed logQLIKE
was found undefined on some pre-2019 targets. The hidden-4 two-input run
resumed from its best epoch-19 model and Adam state; the old epoch 20 was
superseded. It completed the 30-epoch cap with patience 10 and compiled CUDA.
Its best checkpoint is epoch 27, validation QLIKE 0.4790669037. W&B run
`fx8nhaho` and the continuation log are preserved under
`wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_intradayret_w20_h4_lr0.0001_noclip_progress_20epochs_resume_to30.log`.

The global 2016-2018 validation residual histogram for that checkpoint and
1,120,921 dated residuals across 1,525 tickers are in `imgs/lstm_validation`
(`global_h4_intradayret_validation_residual_histogram.png` and matching
`*.csv.gz`). Residual is actual minus predicted daily Garman-Klass variance.
The figure shows the full distribution on a log count axis and a central-98% zoom. No test-period
evaluation was run. Next: compare the validation residuals and forecasts
with simple baselines on identical observations before test evaluation.

The full-panel, window-20 base LSTM with ordered inputs
[log(sqrt(Garman-Klass variance)), ln(adjusted Close/Open)] completed
three compiled CUDA trials. Each used batch 128, Adam rate 0.0001,
no gradient clipping, patience 10, and at most 20 epochs. The target
was next-observation log volatility; QLIKE and reported errors are
in variance units. Best validation-QLIKE checkpoints: hidden 128,
epoch 19, 0.4720972054 (W&B 48e1ho51); hidden 4, epoch 19,
0.4805205024 (fx8nhaho); hidden 64, epoch 19, 0.4731626635
(x8t7waqe). The earlier single-input hidden-128 run completed at
epoch 20 with best epoch 18 and QLIKE 0.4746087831 (hx9jl4m2).
One initial two-input hidden-128 launch was interrupted during epoch 1
with no checkpoint after a temporary cap-change request (vm0hppa8).
All three queued reruns finished; results and logs are in
ref/implementation_plan/lstm_val_training.md and wandb/lstm_val_training.
A compiled CUDA two-ticker preflight and fresh-context cold reviews
passed before the queue. No test-period evaluation has been run.

An AAPL 2016-2018 validation histogram of residual = actual variance
- predicted variance from the hidden-128 best checkpoint, plus its
754 dated residuals, are in imgs/lstm_validation. This is one ticker,
so its shape is not a full-panel residual conclusion. Next: inspect
validation residuals and compare models with simple baselines on
identical observations before test evaluation.

## Hidden-64 log-volatility cancellation and run timing (2026-09-23)

The raw-variance base LSTM completed 20 epochs. Its best checkpoint is epoch
20 with validation QLIKE 0.5151755551; the independent final validation
forecast agreed within float rounding. The next base LSTM using
log(sqrt(Garman-Klass variance)) for both inputs and targets, hidden width 64,
and the same selected settings started in W&B run sp26pqr0. The user then
cancelled it during epoch 1. No epoch completed and no checkpoint was saved.
Its log ends with a run_end interruption marker of 264.3 seconds measured
from queued launcher start, including the wait for the previous run.

The shared neural command now emits a final run_end JSON marker with status
and total wall time after normal completion, handled interruption, or failure.
The detached log-volatility launcher also appends a marker when a killed child
cannot emit one. Focused complete/interrupted/failed checks pass, and a
one-epoch real-data command printed its final wall time. Fresh-context review
passed 99/100 in judge/reviews/lstm_run_elapsed_review.json; its only test
coverage finding was fixed. The user then authorized a fresh hidden-64
log-volatility rerun because the stopped attempt had no checkpoint. The new
attempt uses the same window 20, rate 0.0001, batch 128, compiled scoring
and training, no clipping, patience 10, and at most 20 epochs, with its own
checkpoint and W&B identity. The prior raw-variance 20-epoch run is complete
and the dataset identity matches. The fresh-context review passed 100/100
in judge/reviews/lstm_logvol64_rerun_review.json. The rerun started at epoch 1
in compiled CUDA mode with W&B ID qbvvsiwy and console log
wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h64_lr0.0001_noclip_progress_20epochs_rerun.log.
Next: monitor its best-QLIKE checkpoint through the 20-epoch cap. The CLI
will write a final elapsed-time marker on completion, and the launcher can
append one if the child is killed. Test-period evaluation remains deferred.

## Raw-variance LSTM with hidden width 64 (2026-09-23)

At the user's request, the full-panel log-volatility base LSTM was stopped
during epoch 8. Seven epochs completed; its best checkpoint was epoch 5 with
validation QLIKE 0.4848985493. The run `9eq5x5iw`, console log, and checkpoint
were preserved. Its result row is marked interrupted and records the discarded
partial epoch 8. The next selected run uses raw Garman-Klass variance for
inputs and targets, base LSTM hidden width 64, window 20, initial rate 0.0001,
batch 128, no clipping, compiled training/scoring, patience 10, and at most
10 total epochs. A two-ticker compiled CUDA preflight passed; its validation
metric matched a separate prediction pass within float rounding. A fresh
cold review passed 99/100 in `judge/reviews/lstm_raw64_review.json`; its minor
results-header finding was corrected. The full-panel fit started from epoch 1
in W&B run `wr52wy0r` with console log
`wandb/lstm_val_training/logs/base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs.log`.
During epoch 10 the user authorized extending this run to at most 20 total
epochs. Epoch 10 completed with the best validation QLIKE 0.5333659590;
the separately reported validation QLIKE agreed within float rounding.
The continuation reopened W&B run `wr52wy0r` and began in compiled CUDA mode
at epoch 11 from the saved epoch-10 model and Adam state. No epochs were
superseded. Its log is
`wandb/lstm_val_training/logs/base_lstm_vol_gk_raw_w20_h64_lr0.0001_noclip_progress_10epochs_resume_to20.log`.
A fresh cold review passed 99/100 in
`judge/reviews/lstm_raw64_extend_review.json`; the review requested the
restart check, which passed. Next: monitor the continuation through its
20-epoch cap and record the final best checkpoint. The
wider sweep and test evaluation remain deferred.

## Global Garmanâ€“Klass histograms (2026-09-23)

Three descriptive full-period equity figures were generated from the saved
Garmanâ€“Klass variance table under `imgs/roughness_analysis/global/full`:
within-ticker log-volatility increments at exact calendar lags 1, 5, 25, and
125 days, untransformed daily variance, and log daily variance. Each overlays a full-sample normal
maximum-likelihood fit; the displayed histogram ranges are cropped and labeled,
while fitting uses all 9,217,433 variance rows or all eligible lag pairs.
These descriptive plots do not change the pre-2016 observation-lag H estimate
used by RFSV. Next model step remains the small authorized fitting trial.

Last updated: 2026-09-23

## Epoch-13 LSTM stop and log-volatility request (2026-09-23)

The user stopped the current Garman-Klass base LSTM after epoch 13 validation.
The best checkpoint is epoch 13 with validation QLIKE 0.5293365402. The log
`wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress_resume_compiled_eval.log`
ends with a JSON stop marker recording best epoch, best QLIKE, and discarded
partial epoch 14. The queued SiLU handoff was cancelled. The result record is
marked interrupted at 13 epochs. The user next requested a base LSTM with the
same window 20, hidden width 16, learning rate 0.0001, no clipping, compiled
training/scoring, patience 10, and log-volatility data. Both input and target
are log(sqrt(daily variance)); model output z remains log variance. Training
QLIKE and reported errors reconstruct variance from the target. No RFSV
correction is applied because QLIKE directly optimizes the variance forecast.
The shared loop now logs
best epoch and best validation QLIKE at every epoch and on normal completion.
The two-ticker compiled preflight passed with variance-unit train/validation
metrics and an interchangeable checkpoint. A fresh-context review passed
100/100 in `judge/reviews/lstm_logvol_final_review.json`. The full-panel base
LSTM started in its own W&B run `9eq5x5iw` and writes progress to
`wandb/lstm_val_training/logs/base_lstm_vol_gk_logvol_w20_h16_lr0.0001_noclip_progress.log`.
Next: monitor its first epoch and eventual best-QLIKE checkpoint. No SiLU run
has started.

## LSTM continuation handoff (2026-09-23)

The base Garman-Klass no-clipping run `kezb18pv` logged three complete epochs,
then was stopped during epoch 4. Its best saved checkpoint is epoch 2 (validation
QLIKE 0.6233092337); epoch 3 scored 0.6713596033 and will be superseded on
continuation. The checkpoint has Adam state but no RNG state, so the first
continuation is not bit-for-bit equivalent to uninterrupted training. The
original console log and checkpoint remain intact. A sandboxed attempt to
reopen W&B in the same run failed at network initialization before training.
The online continuation then started in compiled CUDA mode from epoch 3 with
`--epochs 50 --patience 10` and is logging batches to
`wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress_resume.log`.
It resumed W&B run `kezb18pv`. The earlier reviewed SiLU handoff was cancelled
when the user stopped the base run at epoch 13.
The first resumed epoch (epoch 3) improved validation QLIKE to 0.5844599048
and became the best checkpoint. All five live curve tables appeared in W&B;
the QLIKE table contains retained epochs 1 and 2 plus the new epoch 3.
After the first continuation started, the shared loop was changed to use its
compiled wrapper for train/validation scoring when compilation is enabled.
At the user's request, the first continuation stopped after epoch 5 metrics:
validation QLIKE was 0.6093654774, so the best checkpoint remained epoch 3.
The base run was relaunched in the same W&B ID from epoch 3 with both training
and scoring compiled; logged epochs 4 and 5 are superseded in its curves.
Its new console log is
`wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress_resume_compiled_eval.log`.
The SiLU handoff was stopped. Warm forward-only checks
at batch 128/window 20/width 16 measured 0.00381 s eager versus 0.00121 s
compiled for base, and 0.00395 versus 0.00109 s for SiLU. These are isolated
forward timings, not end-to-end epoch speedups.
The continuation keeps window
20, width 16, rate 0.0001, no clipping, best-QLIKE selection, one rate retry,
and at most 50 total epochs. The shared loop now supports checkpoint/Adam
resume, optional compilation, and live per-epoch W&B train/validation curves.
RTX 2060 warmup measured 0.0058 s/step compiled versus 0.0215 eager for the
base LSTM, with matching predictions and gradients on the checked batch.
All project tests were moved to `models/tests`; focused resume, curve, and smoke
checks pass. Fresh-context reviews passed 100/100 in
`judge/reviews/lstm_resume_final_review.json` and
`judge/reviews/lstm_selected_pair_queue_review.json`.
Next: follow the epoch-13 handoff above. No test-period evaluation or wider sweep
has been run.

## Garmanâ€“Klass LSTM validation sweep (2026-09-23)

The no-clipping full-panel trial at 09:20 was stopped before completion at
the user's request for visible training progress. It is marked interrupted
and its W&B record/checkpoint were preserved. The rerun uses the same model,
seed, data, and hyperparameters with console progress every 1,000 batches
(plus first/last batch) and all train/validation metrics after each epoch.
The `_noclip_progress` run started as hidden background PID 16400 at
09:37 local time. Its live terminal log is
`wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip_progress.log`
and W&B URL is `https://wandb.ai/personalfeb/timeseries-volatility/runs/kezb18pv`.
The log showed CUDA and batch progress. Synthetic checks and a
limited online preflight passed; fresh-context review
`judge/reviews/lstm_progress_review.json` validated at 100/100. The new
full-panel result remains pending. The original runner marked the stopped
process failed; the continuation handoff above supersedes that status.

The user requested one additional full-panel base LSTM trial at window 20,
width 16, and learning rate 0.0001 with gradient clipping disabled. The
`--no-grad-clip` command records `clip_norm=None` and checks gradient finiteness
without scaling it. A separate `_noclip` run name preserves the interrupted
clipped trial. Focused curve, forecast, and sweep checks passed; a limited
online CUDA preflight logged zero clipped batches and five W&B curves. The
fresh-context review `judge/reviews/lstm_no_clip_review.json` passed 100/100.
The earlier full-panel trial started as hidden background PID 7432 on 2026-09-23 at
09:20 local time. Its console log is
`wandb/lstm_val_training/logs/base_lstm_vol_gk_w20_h16_lr0.0001_noclip.log`.
The online W&B run is
`https://wandb.ai/personalfeb/timeseries-volatility/runs/26cqhn7k`; the
console log confirms CUDA and the full 5,416,755 training windows.
The earlier trial did not complete; its status remains interrupted in the
validation results table.

The sweep was stopped after one completed full-panel run and five logged
epochs of the second. The first base LSTM (window 20, width 16, rate 0.001)
selected epoch 17/20 with validation QLIKE 0.490039. A causal trailing-20
mean baseline on the same 1,120,921 validation observations scored 0.537610.
The first run clipped 97.38% of batches in epoch 1 and 93.98% in its best
epoch; the interrupted second run clipped 93.66â€“99.99% over five epochs.
A 50-batch check at the first checkpoint found median unclipped gradient norm
7.82 against the norm-1 cap, with 92% above the cap. Pre-2016 target variance
ranges from 3.4e-13 to 14.93, with median 0.000236. Large actual/predicted
ratios plausibly drive large QLIKE gradients, but clipping has not been shown
to cause the plateau. A matched exploratory four-epoch pilot on the first 20
tickers, with 500 pre-2016 and 500 validation rows each, gave validation
QLIKE 1.68883 at norm cap 1 and 1.65352 at cap 10; this limited subset does
not establish a full-panel improvement. The first full run improved validation
QLIKE from 0.531095 to 0.490039. The partial second run is marked interrupted
in the results table. Next: test a change on a more representative panel and
review it before deciding whether to restart the full 36-run sweep.

The 36-configuration full-panel validation sweep was launched via
`models/lstm_val_sweep.py`. It reads
the 1,525-equity, 9,217,433-row Garmanâ€“Klass table from
`D:\DBs\timeseries_analysis\history_coverage.db`, fits pre-2016 targets, and
uses 2016â€“2018 validation QLIKE for checkpoint selection. The command uses
CUDA, 20/40/80-observation windows, widths 16/32/64, and initial rates
0.001/0.0001 for base and SiLU LSTMs. It does not evaluate test targets.
Best-epoch metrics, W&B URLs, and log paths are written after each full run
to `ref/implementation_plan/lstm_val_training.md`; progress and failures are
also in `wandb/lstm_val_training/results.json` and `sweep.stdout.log`.
The limited 2-ticker online preflight passed on CUDA with five W&B curves,
a reloadable checkpoint, and validation-only output. Focused checks passed,
and fresh-context review `judge/reviews/lstm_val_sweep_review.json` validated
at 100/100 with no findings. Full-panel outcomes remain incomplete.

## Neural training curves (2026-09-23)

The shared volatility neural fitting loop now evaluates training and validation
windows after each epoch with the final epoch weights, using the established
variance metrics and each ticker's pre-2016 MASE history. It logs all five
metrics for both splits and five overlaid W&B curves when enabled. Validation
QLIKE alone still selects checkpoints and triggers the learning-rate retry and
early stop. `--no-wandb` still fits and selects a checkpoint without charts or
local reports. Synthetic two-ticker curve checks, neural floor checks, and the
training smoke passed; no project-data fit ran. Next: small authorized
project-data fitting trials to inspect neural diagnostics before comparison.

## RFSV observation-lag implementation (2026-09-23)

The six pre-2016 global equity roughness outputs now use within-ticker
observation lags 1--400; full-period calendar-lag outputs were unchanged.
Parkinson H = 0.03638939 and Î½Â² = 0.41077821; Garmanâ€“Klass H = 0.03246303
and Î½Â² = 0.44473011. Î½Â² is exp(the q = 2 free-intercept regression intercept)
from log-volatility moments. Both global and local RFSV fits load the matching
estimator's saved training results, reject incompatible lag metadata, and save
the parameters and source. Each forecast uses the latest 20 observations,
exact Section 5 kernel bin masses, oldest-value tail extension, and the
unnumbered Section 5.2 `2*c*nu^2` log-variance correction before the existing
training-derived variance floor. This is the paper's Section 5 approximation,
not a fit of the full stationary fOU model. Synthetic tests passed, and the
fresh-context cold review in `judge/reviews/rfsv_observation_review.json`
validated at 100/100 with no findings. No project-data RFSV fit or model comparison was run. See
`ref/implementation_plan/rfsv_equation_review.md`. Next: run small authorized
project-data fitting trials and inspect diagnostics before comparison.

## Neural forecast floor adjustment (2026-09-22)

Review finding 2 is addressed in the working tree: MLP, base volatility LSTM,
and SiLU-LSTM now output log daily variance, initialize at the selected
pre-2016 training-target median, and exponentiate for variance-unit forecasts.
Their forecasts have no floor. HARNet-20 and HARNet-80 retain fitted-HAR
initialization and the training-derived variance floor, with floor-hit
percentage logged per epoch. Neural checkpoints record and validate their
output convention. Synthetic checks cover gradients, QLIKE, both HARNets,
global/local smoke fitting, and checkpoint reload. A fresh-context cold review
passed 100/100 with no findings in
`judge/reviews/neural_forecast_floor_review.json`. No project-data fitting or
comparison was run. Next: inspect small authorized project-data fitting
trials and diagnostics.

## HARNet split (2026-09-22)

The volatility training framework now exposes `models.harnet_20` and
`models.harnet_80` as separate commands and checkpoint paths. The first uses
1/5/20-observation summaries; the second adds trainable 40/80-observation
averaging stages. Each starts from an OLS fit of matching HAR terms on the
selected pre-2016 equity histories, before QLIKE optimization. Both remain
model definitions and synthetic checks; no project-data fitting or comparison
was run. Next: inspect small authorized project-data fitting trials and
validation diagnostics before any model comparison.

## Training framework update (2026-09-21)

`models/training_blocks.py` now builds ticker-safe next-day daily variance windows for
Parkinson or Garmanâ€“Klass. The nine active model modules expose global and local
fits, forecasts, and commands. Models fit only pre-2016 equity histories;
neural checkpoints select on 2016â€“2018 validation QLIKE; 2019â€“2025 equities
and matching unseen crypto proxies are evaluation sets. The floor is the
minimum positive fitting-history variance, saved with each fit. Statistical
specifications are fixed at this stage. The return experiment remains a
reference under `inference.inference`, with its old loop at
`inference/returns_training.py`; no project-data training was run for this
framework. The synthetic smoke is `python -m models.tests.test_training_smoke`.
The neural output adjustment above supersedes the earlier all-model raw-output
floor. HARNet outputs below its floor still have zero gradient through the
clamp, so inspect floor-hit rates in small trials.
The existing training-framework JSON is a same-context self-check; a fresh-context
cold review remains required under `AGENTS.md`.
Next: inspect real dataset eligibility and fit diagnostics in small authorized
trials, then choose neural widths/windows and statistical specifications from
the reserved validation period.

## Goal

The work has two parts. **Part I is the VVSI project**, which predicts short-horizon stock returns with a global LSTM. **Part II is the full thesis project** described in `ref/plan.md`: a comparison of statistical/econometric and machine-learning models for realized-volatility forecasting, followed by portfolio-utility evaluation. The full thesis also covers volatility roughness (fBm/fOU and the Hurst exponent) and whether learned behavior generalizes across assets and datasets.

`ref/plan.md` is the current plan. `ref/thesis_notes.md` contains the literature notes supporting it.

## What is implemented

- NASDAQ and NYSE ticker-list ingestion and cleaning, including removal of non-equity instruments and consolidation of share classes.
- Bulk daily OHLCV download from Yahoo Finance for 2006-01-01 through 2026-01-01.
- Resumable, rate-limited maximum-history acquisition, with raw ticker-major
  OHLCV and 20/25/30-year coverage metadata stored under
  `D:\DBs\timeseries_analysis`.
- A resumable altFINS crypto acquisition framework stores daily and 15-minute
  OHLCV in separate SQLite tables, builds daily realized variance from squared
  15-minute close-to-close log returns (including the preceding day's final
  close), and records coverage before and after completeness checks. The live
  download uses `ALTFINS_API_KEY` and logs to `crypto_history_scan.txt`. The
  completed scan covered all 5,145 altFINS symbols and produced 2,055,008
  aligned daily/proxy observations, below the 9,587,674-point stock target;
  details are in `ref/crypto_history_coverage.md`.
- Split/dividend-consistent OHLC adjustment via `auto_adjust=True`.
- Scale-free features: intraday return, overnight return, relative high-low range, and log volume. Daily return was removed because it is determined by intraday and overnight returns.
- SQLite persistence in a `features` table.
- Chronological train/validation/test splits: train before 2016, validation from 2016 through 2018, and test from 2019 onward.
- Per-ticker log-volume z-score statistics fitted only on training data, with safe handling of unseen tickers and invalid standard deviations.
- Per-ticker rolling-window dataset construction. Data is sorted by `(Ticker, Date)`, and windows cannot cross ticker boundaries.
- A PyTorch LSTM implemented from individual gates, with Xavier initialization and a forget-gate bias of one.
- End-to-end return-model training, validation, testing, Weights & Biases logging, best-validation checkpointing, gradient clipping, GPU memory reporting, and `torch.compile` support.
- The VVSI return pipeline now reads the 1,834 tickers with complete 20-year
  coverage directly from the existing raw-history database over 2006-01-01 through 2025-12-31,
  logs exact observation-weighted MSE, RMSE, MAE, bounded sMAPE, and R-squared,
  and produces final split tables, top-15 ticker tables, and full per-ticker
  validation/test loss distributions in W&B.
- Separate one-step variance metrics now define MAE, MASE, MSE, RMSE, and QLIKE in each estimator's variance units. MASE uses each ticker's training-history one-step naive scale; QLIKE requires finite positive variance and is the planned volatility LSTM objective. The signed-return pipeline remains separate.
- HARNet and statistical volatility models use an estimator-specific positive lower floor on variance forecasts, selected from training targets and held fixed for validation/test. MLP and volatility LSTMs instead forecast `exp(log variance)` without a floor. The metric function still rejects nonpositive inputs; no upper forecast cap is planned.
- The three 50-epoch VVSI production runs for window sizes 10, 30, and 100
  completed. Their best-validation epochs were 19, 37, and 18 respectively.
  Future runs use checkpoint filenames containing the best epoch and one
  learning-rate retry: after 10 consecutive non-improving validation epochs,
  the learning rate is reduced by a factor of 0.1; another 10 consecutive
  misses stop training early.
- Several training runs and checkpoints exist locally. The earlier train/validation discrepancy led to fixes for mixed-ticker windows, inconsistent adjusted prices, and feature scaling.
- Literature notes cover HAR/HARNet, GARCH, rough volatility, global versus local models, volatility commonality, TSFMs, evaluation losses, and economic utility.
- The production roughness build retained all 1,525 complete-history equities
  and 9,217,433 jointly valid positive-variance dates per equity estimator
  after rejecting 370,241 rows. It retained 181 floating cryptocurrencies
  with exactly 1,760 shared dates each (318,560 rows per crypto estimator).
- Parkinson, reduced Garman-Klass, and existing 15-minute realized variance are
  stored in five ticker-major `WITHOUT ROWID` tables. Global and top-ten local
  roughness analysis uses log volatility, q={1, 1.5, 2, 3, 4}, exact calendar
  lags 1--400 for full-period outputs and observation lags 1--400 for the
  pre-2016 global training outputs, with only within-ticker displacement pairs. The completed run
  wrote 64 figures plus moment, zeta, and Hurst summary CSVs under
  `imgs/roughness_analysis/`.

## Current plan position

For **Part I / VVSI**, the return-prediction rerun infrastructure and the
three production LSTM runs are complete. The next action is to compare and
report the window-size results.

For **Part II / the full thesis**, Parkinson and Garmanâ€“Klass are separate daily variance targets. The training framework and synthetic checks are implemented; no project-data model fitting or volatility evaluation has run.

The VVSI LSTM still predicts `overnight_returns`; its earlier checkpoints are retained. The new `base_lstm_vol` path is for next-day variance. Estimator tables and initial roughness analysis exist; Parkinson and Garmanâ€“Klass remain separate comparison tracks.

Completed or reusable parts of Step 1:

- equity universe construction;
- daily OHLCV acquisition and adjusted prices;
- chronological splitting, storage, normalization, plotting helper, and safe per-ticker windows.

Still required before substantive comparison:

1. Review fit diagnostics and sample eligibility from small authorized trials.
2. Select neural widths/windows and later statistical specifications using the reserved validation period.
3. Add VIX or volatility commonality only if initial evidence shows they are needed.

After the RFSV equation and observation-lag H changes above are reviewed, the next model-comparison preparation is a small authorized project-data fitting trial, followed by diagnostic review. Saved full-period H estimates remain descriptive.

## Planned later work

- Models under consideration: SARIMA, HAR, HARNet, GARCH, AR(1), MLP, LSTM variants, RFSV, TimesFM, and TinyTimeMixers. The small-language-model idea is explicitly low priority.
- The defined comparison metrics are QLIKE, MSE, MASE, MAE, and RMSE. Winsorization and joint ES/VaR loss are deferred; Patton's proxy-robustness result requires its assumptions and does not automatically cover every estimator here.
- Compare local versus global training and the feature sets RV-only, RV + commonality, and RV + commonality + VIX.
- Statistical comparison with a Model Confidence Set and forecast-efficiency checks with Mincer-Zarnowitz regressions.
- Economic comparison using Sharpe ratio and realized utility, followed by portfolio simulation.

## Repository map

- `data/filtering_stock.py`: ticker-universe cleaning.
- `data/data_fetching.py`: download, feature creation, SQLite storage, splits, normalization, and plotting.
- `models/training_blocks.py`: ticker-safe variance windows, metrics, floors, and artifact helpers.
- `data/roughness_analysis.py`: variance-table construction, retained universe
  selection, top-volume samples, roughness/Hurst estimation, and figures.
- `models/base_lstm.py`: custom LSTM cell and sequence model; other model definitions live beside it.
- `inference/returns_training.py`: VVSI training/evaluation metrics, checkpointing, and diagnostics.
- `inference/inference.py`: current executable VVSI training and test pipeline (`python -m inference.inference`).
- `ref/plan.md`: current research plan.
- `ref/thesis_notes.md`: paper notes and rationale.
- `ref/references.md`: bibliography/source list.
- `ref/progress_report.md`: earlier project report; useful history, but it predates the volatility-focused plan.
- `README.md`: describes the implemented return-prediction prototype; it is not a statement that the revised volatility plan is complete.

## Working-tree note

At the time of this update, `main` matches `origin/main`. `.claude/` is an existing untracked user directory and must not be modified or committed incidentally.
## Raw-history volatility test evaluation (2026-09-28)

Saved raw-history adjusted Garman-Klass models were scored without retraining on
2019-01-01 through 2025-12-31 next-recorded-observation targets. The evaluator
`inference/evaluate_volatility_test.py` verified all 14 checkpoint identities,
source/cohort metadata, training-only floor 3.396062419686545e-13, and full
window counts before scoring. The eight neural rows were MLP, Base LSTM,
direct-variance SiLU-LSTM, and HARNet at windows 20 and 80; the six statistical
rows were AR(1), HAR-20/80, GARCH(1,1), SARIMA, and RFSV-20. The red-marked
SiLU log-volatility trial was excluded. The complete MAE, MASE, MSE, RMSE,
QLIKE, floor-hit share, observation count, excluded count, and artifact paths
are in `imgs/eval/volatility_test_metrics.csv`; the labeled table image is
`imgs/eval/volatility_test_metrics.png`.

The user subsequently replaced the training-history MASE scale with a naive
forecast scale calculated on the exact scored test targets for each ticker and
window. The naive forecast uses the immediately preceding recorded variance;
the ticker scale is its mean absolute test error. MASE averages each model's
absolute test error divided by that ticker's test naive scale, so the naive
forecast scores 1 on the same dates. This is a test-relative evaluation metric,
not the conventional training-scaled MASE in `ref/implementation_plan/losses.md`.
The test outcomes set the scale only after forecasts; they are not used for
fitting or prediction. The earlier pre-2016-scale exclusion of 2,134 window-20
and 1,912 window-80 observations was reversed. Every ticker has a positive
finite test scale: all 4,479,681 window-20 and 4,295,773 window-80 eligible
forecasts enter **all five** metrics, with zero exclusions. This changes MASE
and restores those observations to MAE, MSE, RMSE, and QLIKE as well.

The target and forecasts remain daily unannualized adjusted GK variance, with
the saved training-only forecast floor and no extra test-target flooring.
MAE and RMSE have variance units; MSE has squared variance units; MASE and
QLIKE are dimensionless. SiLU direct-variance models and SARIMA still have
very large QLIKE when some forecasts hit the tiny positive floor; these are
reported negative results. Different window lengths have different target
dates, so they are not ranked against each other. Raw-history neural forecasts
are floored by current code, although `ref/implementation_plan/losses.md`
says MLP/LSTM log-variance outputs should have no forecast floor; this test
matches prior raw-history scoring. The focused synthetic fixture passed by
direct invocation in `.venv`; `pytest` is absent. The earlier 97/100 cold
review applies to the superseded training-scale version. Fresh-context review
`judge/reviews/volatility_test_mase_review.json` passed 97/100 with no critical
findings; its only minor finding is the absence of a known-answer neural/GARCH
fixture. `python judge/validate.py` validated the JSON. Next: assess
matched-date or high-volatility sensitivity before comparative claims.
## Full-history RFSV test row and refreshed metric figure (2026-09-28)

The saved pre-2016 full-history RFSV fit was source- and cohort-verified and
scored through `RFSV.forward()` once per ticker, forecasting the 2025-12-31
adjusted GK variance from all earlier valid positive observations. Its one-date
MASE scale is that ticker's absolute 2025-12-31 versus preceding-recorded-row
variance change. Nine tickers (CMTV, FORTY, GYRO, JBK, KELYB, LBTYB, MGYR,
PYT, SENEB) had zero raw GK variance on both dates and hence zero scale; all
nine were removed from every model's scored dates. The RFSV row has 2,705
forecasts: MAE 0.000691016259044611, MASE 10.292698148970693, MSE
1.1216008087402747e-05, RMSE 0.003349030917654053, QLIKE
0.5644500996145306, and 0% forecast-floor hits. Full rerun counts were
4,479,204 for each window-20 row (477 excluded forecasts) and 4,295,635
for each window-80 row (138 excluded forecasts). The old RFSV-20 artifact and
prior run log remain historical; the delivered table contains RFSV-full in its
place. `imgs/eval/volatility_test_metrics.png` and its CSV omit N and Excluded
columns; red and blue mark the two lowest values in each of the five losses.
These color tags are descriptive only: RFSV has one final-date forecast per
ticker, whereas the other rows average over eligible 2019-2025 dates, and
their test-date MASE scales differ. The full run log records counts and row
values. Focused checks and the full 14-row reevaluation passed. Next:
compare all models on matched 2025-12-31 targets before inferring model
superiority from the RFSV row. Fresh-context cold review
`judge/reviews/rfsv_full_test_eval_review.json` passed 100/100 with no
findings, and its JSON validated.
## Local Base LSTM and HARNet fitting (2026-09-28)

The current request limits fitting to Base LSTM and HARNet at windows 20 and 80
for NVDA, AAPL, NFLX, GOOG, and AMZN; other local models and all post-fit test
scoring are deferred until requested. The five were selected by pre-2016 mean
volume in the eligible raw-history cohort, with GOOG retained over GOOGL and
AMZN taking the fifth company slot. `ref/implementation_plan/local_training.md`
records each ticker's train/validation/test target-window counts and the fixed
run settings. A fit-only CLI option skips post-fit prediction; each run still
uses validation QLIKE to select its best checkpoint and records source database
size and modification time. Focused raw-window, fit-only, and queue checks
passed. Fresh-context cold review
`judge/reviews/local_training_preflight_review.json` passed 100/100 and its
JSON validated. The online, one-at-a-time Base LSTM queue
`python -m inference.run_local_base_lstm` completed all ten ticker/window fits
with zero failures, separate checkpoints, and complete W&B logs; its first W&B
run is `xlnyeeg9`. The subsequently requested HARNet group uses each model's
existing raw-GK-variance input and ticker-specific pre-2016 HAR initialization;
HARNet has no configurable hidden width. Its fit-only online queue
`python -m inference.run_local_harnet` completed all ten ticker/window fits
with zero failures and synced W&B logs. Focused queue checks and
fresh-context HARNet preflight review
`judge/reviews/local_harnet_preflight_review.json` passed 100/100. Queue status
and logs are under `wandb/local_training/`. All twenty checkpoints were reloaded
and checked for the ticker, model, window, loss, inputs, hyperparameters,
source database size and modification time, positive floor, and completed
training status. No test forecast or post-fit score was produced. The next
local-training task awaits the user's requested model group; later comparison
must score saved local and global fits on common ticker/date keys.
Fresh-context final cold review `judge/reviews/local_training_final_review.json`
passed 100/100 with no findings, and its JSON validated.

## Local MLP fitting complete (2026-09-28)

At the user's next request, MLP window-20 and window-80 fits were added for
the same five raw-history GK equities: NVDA, AAPL, NFLX, GOOG, and AMZN.
`ref/implementation_plan/local_training.md` now records the MLP-specific
decision: log-volatility plus adjusted intraday-return inputs, log-variance
output, QLIKE, width 128, seed 42, batch 128, Adam 0.001, patience 10, and
at most 20 epochs. The previously verified ticker/window counts are unchanged.
`inference/run_local_mlp.py` queues only these ten fit-only online W&B runs,
stopping on failure and preserving all prior checkpoints. Its focused command
check passed. Fresh-context preflight cold review
`judge/reviews/local_mlp_preflight_review.json` passed 98/100 with one minor
test-coverage finding and no critical findings; its JSON validated. The online
queue completed all ten ticker/window runs with zero failures. All ten
checkpoints were reloaded and verified against ticker, window, target/input
convention, QLIKE, hyperparameters, source database size and modification
time, positive floor, and the exact counts in the local plan. All ten W&B
logs show successful sync. No post-fit or test scoring was run. Next: await
the user's next requested local model group; later score all models on
matched ticker/date observations before aggregating local losses.
Fresh-context final cold review `judge/reviews/local_mlp_final_review.json`
passed 98/100 with no critical findings; its sole minor finding is absent
launcher failure-branch test coverage, and its JSON validated.

## Local SiLU-LSTM fitting complete (2026-09-29)

The user requested local SiLU-LSTM window-20 and window-80 fits for NVDA,
AAPL, NFLX, GOOG, and AMZN, matching the direct-variance MSE global runs
documented in `ref/implementation_plan/lstm_val_training.md` and saved
checkpoints. `ref/implementation_plan/local_training.md` records the exact
input, target, loss, floor, windows, hyperparameters, and unchanged per-ticker
split counts. `inference/run_local_silu_lstm.py` runs only these ten fits with
ticker-named online W&B runs and fit-only checkpoints. The focused queue check
passed; fresh-context preflight cold review
`judge/reviews/local_silu_preflight_review.json` passed 99/100, and its one
minor plan wording finding was corrected. The queue completed all ten
ticker/window runs with zero failures. All ten checkpoints were reloaded and
checked against ticker, window, raw-variance input/output, MSE, hyperparameters,
source database size and modification time, positive floor, and the exact
counts in the local plan. Every W&B log shows successful sync. No post-fit or
test scoring was run. Next: await the user's next requested local model group;
later compare local and global fits on matched ticker/date observations.
Fresh-context final cold review `judge/reviews/local_silu_final_review.json`
passed 100/100 with no findings, and its JSON validated.

## Global test residual diagnostics (2026-09-29)

The user requested residual plots for the 14 saved global adjusted GK models before local scoring. `inference/plot_global_residuals.py` reuses the verified artifacts, target dates, forecast floor, and shared eligible test observations from `inference/evaluate_volatility_test.py`; it does not refit models. The ticker-level raw residual is actual minus floored predicted daily, unannualized adjusted GK variance; the ticker-level log residual is log(actual variance) minus log(predicted variance), using positive saved floors. Each model has six PNGs in `imgs/eval/global_eval/residuals/<model>/`: raw/log histograms with full and central-99.5% panels, dated raw/log points restricted to each distribution's central 99.5%, and two daily cross-ticker mean timelines. For each date, the means use every eligible ticker and are computed as mean(actual) − mean(predicted) and log(mean(actual)) − log(mean(predicted)); both have the same sign. The user explicitly corrected the earlier mean-of-log-residual definition. Trimming changes display only. All 14 folders contain six nonempty PNGs. The rerun scored 4,479,204 residuals for each window-20 row, 4,295,635 for each window-80 row, and 2,705 for RFSV-full. RFSV-full's timeline is a single 2025-12-31 cross-section, so these plots are diagnostics across different date populations rather than a matched-date ranking. The focused synthetic checks passed using `.venv` Python; the virtual environment lacks pytest, while system Python lacks compatible dependencies. Fresh-context cold review `judge/reviews/global_log_of_means_review.json` validated at 100/100 with no findings; the prior `global_daily_mean_review.json` covers the superseded mean-of-logs definition. Next: follow the user's requested evaluation sequence.
## Thesis Result Table Presentation (2026-10-03)

Evaluation Setup equation 4.2 puts both local-loss forms on one display line. The VS Code LaTeX recipe uses the available `pdflatex` command for two SyncTeX passes; workspace autosave and LaTeX Workshop are configured to rebuild after edits. The nine zero-change RFSV exclusions are stated in two direct sentences with all tickers named. The global and five-stock result tables include RFSV rows and omit N and Floor (%) columns. At the user's correction, each metric column has one red minimum and one blue second minimum across all displayed rows, including RFSV; these colors describe numerical values across different date populations and do not establish comparable model performance. The five-stock RFSV rows have no divider between them. Two direct SyncTeX passes regenerated the adjacent 46-page PDF; the 42-row metric/color check passed, with no new overfull warnings. Fresh-context review `judge/reviews/thesis_result_ranking_review.json` passed 100/100 and validated. Saved scores and evaluation code did not change. Next thesis task remains substantiating Computational Evaluation with saved timing evidence, then scoping matched-observation MCS and unseen-asset transfer checks.
## MCS framework definition (2026-10-03)

`models/mcs_definition.py` now defines finite-positive-variance QLIKE losses and
the model-agnostic T_max MCS elimination sequence with 20-observation
non-circular moving blocks, 10,000 bootstrap draws, alpha 0.10, seed 42,
lexicographic ties, and adjusted p-values. Synthetic checks cover the loss,
block draws, centered statistic, deterministic output, elimination, and invalid
inputs. No saved forecasts were loaded, no stock-specific MCS was calculated,
and no training or model comparison was run. The next task is to connect
verified dated forecast histories, enforce exact per-stock key alignment, and
run the five-stock evaluation requested in the plan. The broader thesis
evaluation position is unchanged.
The focused pytest check passed; fresh-context review
`judge/reviews/mcs_definition_review.json` validated at 100/100 with no
findings.

## Five-stock MCS final sets documented (2026-10-03)

The saved `imgs/eval/mcs/results.json` scores 26 local/global configurations
separately for AAPL, AMZN, GOOG, NFLX, and NVDA, with 1,760 matched
observations per stock, dimensionless QLIKE, and a 90% MCS using 20-observation
moving blocks and 10,000 draws. The five final sets contain 8, 11, 17, 15,
and 16 configurations, respectively. `imgs/eval/mcs/final_set.md` records
all 26 membership rows and the eight-configuration intersection. The
intersection is descriptive, not a test of equal performance or a 90% joint
coverage result. RFSV is excluded because only final-date forecasts are
available. No forecasts or MCS were rerun for this documentation task.
Next: use the saved evaluation for the thesis comparison, preserving the
per-stock interpretation and the differing model features and training budgets;
unseen-asset transfer remains untested.
## Crypto global transfer scoring (2026-10-04)

`ref/implementation_plan/crypto_scoring.md` records the dollar-volume ranking and
fixed scoring choices. BTC, ETH, SOL, XRP, and DOGE lead by mean close × raw
volume on their 1,760 retained GK dates; this is estimated USD turnover, with
historical OHLC quote currency inferred from vendor price scale and dollar
market display because the historical endpoint omits an explicit currency field.
Saved equity-trained global fits were scored without refitting on the same
8,800 crypto target keys per configuration, at their saved 20- or 80-record
window. All 15 aggregate and 75 per-crypto rows, 90 residual plots, table PNG,
and run log are under `imgs/eval/crypto_eval/`. All rows have N=8,800; no
targets or MASE scales were excluded, and saved positive floors had zero hits.
Base LSTM-80 has the lowest QLIKE (0.587240); this is an unseen-asset transfer
result on vendor daily, unannualized GK variance, separate from equity rankings.
The small known-example check passed; source metadata, ranking arithmetic,
row counts, and plot counts were reconciled. Fresh-context review
`judge/reviews/crypto_global_scoring_review.json` passed 100/100 with no
findings and validated JSON. Next: interpret transfer limits before
incorporating these results in the thesis comparison.
## Crypto transfer thesis table (2026-10-04)

Technical Evaluation now places the 15-row equity-trained Global crypto
loss table immediately after the five-stock matched-results table. Table 5.4
uses scaled columns (MAE ×10^-3, MSE ×10^-4, RMSE ×10^-2) and the same
red/blue/orange extrema convention. The nearby prose distinguishes vendor
crypto OHLC from adjusted equity OHLC, states the 8,800-target population and
test-date MASE convention, and names the metric-specific minima. The following
parameter-complexity table automatically becomes Table 5.5. The expanded
focused check validates all 60 displayed metric rows against saved CSVs and
their color rankings. Two in-place SyncTeX-enabled `pdflatex` passes built the
48-page PDF; pages 36-37 were visually inspected, with no new overfull boxes
or unresolved references. No forecasts or metrics were rerun. Next: cold
review this table, then interpret transfer limits for the thesis discussion.
## Crypto realized-variance transfer scoring (2026-10-04)

`ref/implementation_plan/crypto_rv_scoring.md` fixes the second crypto transfer
test. The same saved equity-trained GK Global fits now receive prior valid
15-minute realized-variance history and forecast the next recorded valid daily
RV on exactly the same 1,760 retained GK/RV target dates per BTC, ETH, SOL,
XRP, and DOGE (8,800 per configuration). RV is the daily, unannualized sum of
96 squared 15-minute close-to-close log returns, including the prior UTC day's
final close. The fits, input transforms, and GK-training floors were not
changed. Earlier raw daily rows lacking complete positive RV were omitted
from RV input histories; 64-65 targets per coin follow a two- or three-day
calendar gap. Training/validation populations and prior GK scores did not
change. The target-key set is identical to GK, while both input and target
variance proxies changed, so score differences do not isolate estimator
quality.

`imgs/eval/crypto_rv_eval/` contains 15 aggregate rows, 75 audit rows, a
table image, run log, and 90 residual plots. Every row has N=8,800 with no
target exclusions. Base LSTM-20 has the lowest RV QLIKE (0.392958) and MASE
(0.872923). MLP-20, SiLU-LSTM-20/80, and SARIMA hit their saved floors;
MLP-20's QLIKE is about 1.34e10, a negative result driven by extreme
underpredictions. A source-feed diagnostic found retained 15-minute closes
outside the vendor daily high-low on 22 BTC, 12 ETH, 72 SOL, 95 XRP, and 27
DOGE dates; 123 of these 228 mismatches exceed a 1% margin. SOL has 24 RV
targets above 1 and a maximum of 6.90, versus maximum GK 0.143. Scores keep
the requested matched targets unchanged; the mismatch limits interpretation
of estimator comparisons. The synthetic RV boundary/window checks passed; aggregate
arithmetic, counts, floors, and plots were reconciled. Next: fresh-context
review, then decide how to present the matched GK/RV transfer comparison in
the thesis without implying a causal estimator ranking.
## Five-stock test data before residuals (2026-10-05)

Section 6.4 now opens with raw/log daily mean timelines and raw/log pooled
histograms of the 8,800 adjusted GK test targets shared by the local/global
residual comparisons: five stocks on each of 1,760 dates, 2019-01-02 through
2025-12-31. `data/plot_local_test_gk.py` loads the same eligible targets as
the window-20 and window-80 evaluation and verifies their dates and values
match; it reuses `data/plot_gk_data.py` for the four PNGs in
`imgs/eval/local_eval/test_data/`. The plots use daily unannualized variance,
log of each date's cross-ticker mean for the log timeline, and pooled individual
log variances for the log histogram. The normal overlays use each full pooled
distribution. No data selection, fits, forecasts, or scores changed. The
technical-evaluation check passed; two in-place SyncTeX LaTeX passes produced
the 56-page PDF, and both new pages were visually checked. A fresh-context
cold review validated at 100/100 with no findings in
`judge/reviews/local_test_target_figures_review.json`. Next: interpret
residual patterns alongside matched QLIKE and MCS results, then revisit stale
RFSV prose.

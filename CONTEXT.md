# Project context

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

## Global Garman–Klass histograms (2026-09-23)

Three descriptive full-period equity figures were generated from the saved
Garman–Klass variance table under `imgs/roughness_analysis/global/full`:
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

## Garman–Klass LSTM validation sweep (2026-09-23)

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
epoch; the interrupted second run clipped 93.66–99.99% over five epochs.
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
the 1,525-equity, 9,217,433-row Garman–Klass table from
`D:\DBs\timeseries_analysis\history_coverage.db`, fits pre-2016 targets, and
uses 2016–2018 validation QLIKE for checkpoint selection. The command uses
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
Parkinson H = 0.03638939 and ν² = 0.41077821; Garman–Klass H = 0.03246303
and ν² = 0.44473011. ν² is exp(the q = 2 free-intercept regression intercept)
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
Parkinson or Garman–Klass. The nine active model modules expose global and local
fits, forecasts, and commands. Models fit only pre-2016 equity histories;
neural checkpoints select on 2016–2018 validation QLIKE; 2019–2025 equities
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

For **Part II / the full thesis**, Parkinson and Garman–Klass are separate daily variance targets. The training framework and synthetic checks are implemented; no project-data model fitting or volatility evaluation has run.

The VVSI LSTM still predicts `overnight_returns`; its earlier checkpoints are retained. The new `base_lstm_vol` path is for next-day variance. Estimator tables and initial roughness analysis exist; Parkinson and Garman–Klass remain separate comparison tracks.

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

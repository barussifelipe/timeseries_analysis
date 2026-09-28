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

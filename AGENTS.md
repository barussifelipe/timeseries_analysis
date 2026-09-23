# Repository guidelines for AI models

## Start here

Before making changes, read `CONTEXT.md`, `ref/thesis/plan.md`, and the files involved in the request. Treat `ref/thesis/plan.md` as the research roadmap and `CONTEXT.md` as the current handoff state. If they disagree with the code, report the discrepancy and verify the code rather than guessing.

## Scope and implementation

- Make the smallest change that completes the requested step. Reuse existing code and installed dependencies; do not add speculative abstractions or models.
- Keep the volatility-forecasting objective separate from the earlier return-prediction prototype. Do not present the existing LSTM as a completed volatility model.
- Follow the order in the plan unless the user explicitly reprioritizes it: dataset and target, simple baselines, complex models, then comparison and portfolio utility.
- Preserve unrelated user changes and local artifacts. Never commit `.env`, databases, checkpoints, W&B runs, virtual environments, caches, or `.claude/` unless explicitly requested.
- Update `CONTEXT.md` after a substantive milestone or change in plan position. Record facts, decisions, remaining work, and the next concrete task; do not turn it into a diary.

## Plan writing template

For the next `/plan`, follow `ref/implementation_plan/roughness_analysis.md`. Keep every concrete choice and assumption visible as its own numbered decision or table row; do not compress a prior decision table into summary paragraphs. Separate agreed decisions from defaults that still need validation.

```markdown
# <Task> Implementation Plan

## Implementation decisions

1. <One concrete choice: population, target, units, horizon, or split.>
2. <One concrete choice: method, defaults, validation rule, or artifact.>

## <Dataset and implementation work>

- <Specific file or component and the change to make.>

## Verification

- <Small runnable check and its acceptance condition.>
- <Required review, prohibited runs, and remaining limits.>
```

Add focused sections such as formulas or analysis only when the task needs them. Record the exact values and tradeoffs that affect results, including assumptions introduced by the implementer.

## Data and statistical correctness

- Prevent look-ahead bias. Keep splits chronological and fit every learned transform, threshold, winsorization bound, or statistic on training data only unless the method explicitly requires otherwise.
- Keep all time-series operations within ticker boundaries and sort by `(Ticker, Date)` before creating lags or windows.
- State the exact target, units, horizon, estimator formula, annualization convention, and any transformation such as log variance before training or comparing models.
- Use adjusted OHLC values consistently. Validate positivity and finite values before logarithms, ratios, QLIKE, Parkinson, or Garman-Klass calculations.
- Align market-wide variables such as VIX and commonality by date without forward-looking fills. Document how missing dates and unseen tickers are handled.
- Compare models on identical observations, splits, horizons, feature sets, and metric definitions. Include a naive or simple statistical baseline before claiming improvement.
- For one-step variance forecasts, use the [defined MAE, MASE, MSE, RMSE, and QLIKE](ref/implementation_plan/losses.md) in each estimator's variance units and compare estimators separately. Average over forecast observations; compute MASE with each ticker's finite positive training-history naive scale. Require finite positive actual and predicted variance for QLIKE. Use QLIKE as the later volatility LSTM training objective. These choices follow Hyndman and Koehler (2006) and Patton (2011) under the latter's proxy assumptions; winsorization and joint ES/VaR loss are deferred.
- For the later volatility model, set an estimator-specific positive floor for predicted variance from training targets only, then apply it consistently before QLIKE and evaluation. Record the value and transform; it is a lower limit, not an upper cap. Leave actual targets unchanged and keep `variance_metrics` strict about zero or negative predictions.
- Do not call a result state of the art, causal, universal, or economically useful without evidence that supports that exact claim.

## Research and reproducibility

- Prefer primary papers and official documentation. Add a source to `ref/references.md` and concise findings to `ref/thesis_notes.md` when a new research claim changes the implementation or evaluation.
- Separate sourced facts, project decisions, and hypotheses. Mark assumptions and unresolved choices explicitly.
- Use fixed random seeds where randomness affects comparisons, log configurations and dataset versions, and save enough metadata to reproduce reported results.
- Report failures and negative results. Never invent metrics, experiment outcomes, citations, data availability, or completed work.

## Verification

- Trace all callers before changing shared data or model behavior.
- Add the smallest runnable check for non-trivial logic. For data changes, verify formulas on a tiny known example and check schema, dates, ticker boundaries, missing values, infinities, and leakage.
- Run the narrowest relevant checks before handing off. State exactly what ran and what could not be run.
- After each implementation task, obtain a cold review using `judge/prompt.md`, validate its JSON with `python judge/validate.py REVIEW.json`, and fix findings until it passes (at least 95/100 and no critical findings). Report unresolved blockers without claiming a pass.
- Run the JSON cold reviewer in a fresh, cleared context. Give it the task, plan, diff, relevant files, and check results to inspect directly; do not let an implementer review its own work or count a same-context self-check as a cold review.
- Do not run downloads, long training jobs, destructive operations, commits, or pushes unless the user requests them.

## Commit structure

Only create a commit when the user requests it. Keep each commit limited to one logical change and use:

```text
[model] - area: description of change
```

Replace `model` with the AI model that made the change. Use the narrowest area: `data`, `model`, `training`, `evaluation`, `research`, `docs`, or `chore`. Keep the description concise, explain the outcome rather than the work session, and add a body only when the reason, method, or validation would otherwise be unclear.

Examples:

```text
[codex] - data: add Parkinson volatility estimator
[codex] - model: add HAR baseline
[codex] - evaluation: compute QLIKE on the shared test set
[codex] - research: document volatility commonality assumptions
[codex] - docs: update current plan position
```

Before committing, inspect the diff, run the narrowest relevant checks, and stage only files belonging to that change. Never mix generated data, checkpoints, W&B output, secrets, caches, or unrelated user edits into a commit. Update `CONTEXT.md` in the same commit when the change completes a milestone or moves the project to another plan step.

## Communication

- Lead with the outcome and current plan position. Be concise and distinguish implemented, tested, planned, and speculative work.
- When a request conflicts with the research plan or risks invalidating comparisons, explain the issue before changing direction.
- In prose, write equations with Unicode mathematical notation (for example,
  ζ(q) = Hq, Δ, Σ, and R²), not LaTeX commands or delimiters. For equations
  shown in PowerShell, terminal output, or shell commands, use ASCII-only
  notation so they display without encoding changes. When checking a paper,
  keep its variable names and identify each equation by its printed number;
  explicitly say when the paper leaves an equation unnumbered.

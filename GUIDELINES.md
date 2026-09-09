# Repository guidelines for AI models

## Start here

Before making changes, read `CONTEXT.md`, `ref/plan.md`, and the files involved in the request. Treat `ref/plan.md` as the research roadmap and `CONTEXT.md` as the current handoff state. If they disagree with the code, report the discrepancy and verify the code rather than guessing.

## Scope and implementation

- Make the smallest change that completes the requested step. Reuse existing code and installed dependencies; do not add speculative abstractions or models.
- Keep the volatility-forecasting objective separate from the earlier return-prediction prototype. Do not present the existing LSTM as a completed volatility model.
- Follow the order in the plan unless the user explicitly reprioritizes it: dataset and target, simple baselines, complex models, then comparison and portfolio utility.
- Preserve unrelated user changes and local artifacts. Never commit `.env`, databases, checkpoints, W&B runs, virtual environments, caches, or `.claude/` unless explicitly requested.
- Update `CONTEXT.md` after a substantive milestone or change in plan position. Record facts, decisions, remaining work, and the next concrete task; do not turn it into a diary.

## Data and statistical correctness

- Prevent look-ahead bias. Keep splits chronological and fit every learned transform, threshold, winsorization bound, or statistic on training data only unless the method explicitly requires otherwise.
- Keep all time-series operations within ticker boundaries and sort by `(Ticker, Date)` before creating lags or windows.
- State the exact target, units, horizon, estimator formula, annualization convention, and any transformation such as log variance before training or comparing models.
- Use adjusted OHLC values consistently. Validate positivity and finite values before logarithms, ratios, QLIKE, Parkinson, or Garman-Klass calculations.
- Align market-wide variables such as VIX and commonality by date without forward-looking fills. Document how missing dates and unseen tickers are handled.
- Compare models on identical observations, splits, horizons, feature sets, and metric definitions. Include a naive or simple statistical baseline before claiming improvement.
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

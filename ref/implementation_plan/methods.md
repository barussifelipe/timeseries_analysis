# Thesis Methods Implementation Plan

## Implementation decisions

| # | Agreed decision |
|---|---|
| 1 | Write Methods in English with exactly two subsections: Data and Models. |
| 2 | Focus on equities; introduce cryptocurrencies briefly as intended unseen-ticker transfer evaluation from another asset class. |
| 3 | Describe the NASDAQ/NYSE listing population as the stock universe, not constructed financial indices. |
| 4 | Each sample contains one ticker's chronological history and one next-observation target; global models share one parameter vector across separate samples. |
| 5 | Reserve fitting, parameter updates, batch ordering, initialization, and experiment hyperparameters for Implementation. |
| 6 | Order model subsubsections AR, SARIMA, GARCH, HAR, RFSV, MLP, LSTM, SiLU-LSTM, HARNet; group 20/80 variants within their family. |
| 7 | Reuse established symbols and distinguish observed proxies, latent variance, and forecasts. |
| 8 | Number equations, define symbols, use parenthesized time indices, and explain each mathematical mapping. |
| 9 | Explain the positive zero-target floor through the undefined logarithm of zero and identify it as a project target policy. |
| 10 | Preserve existing thesis edits; restrict substantive changes to Methods, necessary references/research notes, this plan, and the handoff. |

### Data decisions and audited facts

| # | Choice or fact | Evidence / implication |
|---|---|---|
| 1 | Exchange CSVs supply Symbol, Name, and Market Cap. | `data/filtering_stock.py`; no original exchange-level totals are invented. |
| 2 | NASDAQ strips whitespace, replaces slash with hyphen, and deduplicates. | Does not receive NYSE share-class/name filters. |
| 3 | NYSE excludes caret symbols and groups trailing class suffixes by base. | Prefer unsuffixed symbol, then `/A`, then largest Market Cap; name exclusions occur after class selection. |
| 4 | NYSE name exclusions are Fund, Trust, Notes, Warrant, Preferred, Debentures, Senior Floating Rate, Term. | Case-insensitive substring matching; replace slash with hyphen and deduplicate. |
| 5 | Scheduler deduplicates the combined lists. | `data/data_fetching.py`. |
| 6 | Acquire each ticker sequentially with maximum history, adjusted OHLC, threading disabled. | `scan_history_coverage`, distinct from the earlier bulk-download return prototype. |
| 7 | Persist status, attempts, first/last date, last error; skip successful saved tickers on restart. | No acquisition is run for thesis writing. |
| 8 | Recorded acquisition: 5,842 tickers, 22,817,463 observations through 2026-09-10. | `ref/coverage/history_coverage.md`; snapshot, not new computation. |
| 9 | Cohort: 2,714 tickers, more than 80 raw pre-2016 rows, and a row on 2025-12-31. | `models/support_scripts/raw_neural_data.py` and saved audit; end-date criterion introduces survivorship selection. |
| 10 | Retain all available history through 2025-12-31; no ten-year minimum. | Distinct from earlier fixed-period cohorts. |
| 11 | Training target dates precede 2016-01-01. | Chronological split. |
| 12 | Validation target dates are 2016-01-01 through 2018-12-31. | Chronological split. |
| 13 | Test target dates are 2019-01-01 through 2025-12-31. | Chronological split. |
| 14 | Assign split by target date; earlier history may cross split boundaries. | No future inputs enter a forecast. |
| 15 | Use daily unannualized reduced adjusted-OHLC Garman–Klass variance in squared log-return units. | Reuse Background equation `eq:garman-klass-variance`; within-session proxy. |
| 16 | Reject missing/nonfinite OHLCV, nonpositive prices, inconsistent OHLC, negative volume, negative/nonfinite variance. | `raw_frame`; no data behavior changes. |
| 17 | All window inputs must be valid and strictly positive variance. | Invalid and zero inputs block windows. |
| 18 | Valid zero targets become 3.396062419686545 × 10⁻¹³, the smallest positive training-cohort variance. | Positive targets unchanged; originals retained in `RawVariance`; training and metrics use adjusted targets. |
| 19 | Target floor and prediction floor are separate policies. | Neither fills missing dates nor repairs invalid inputs; prediction floor is a lower limit. |
| 20 | Sort by ticker/date; never cross ticker boundaries. | One step is next recorded observation, not necessarily next trading session; no date filling. |
| 21 | All-split adjusted-zero totals: 25,519 (20) and 5,640 (80). | Saved coverage audit. |
| 22 | CBIO and VATE lack valid training histories and training-only MASE scales. | Exclude their 2,134 / 1,912 test windows from common five-metric scoring. |
| 23 | Common scoring population is 4,477,547 (20), 4,293,861 (80). | These are eligibility counts; actual experiment coverage can differ. |
| 24 | Compare window lengths on shared target keys. | Their unequal coverage totals do not establish a matched comparison. |
| 25 | Cryptocurrency evaluation keeps equity parameters fixed on each unseen ticker's history. | Intended evaluation only; compatible units, horizons, and day/session boundaries still require validation. |
| 26 | Crypto acquisition covered 5,145 altFINS symbols, returning 3,119,215 daily rows and 2,055,008 daily/proxy-aligned days during 2019–2025. | `ref/coverage/crypto_history_coverage.md`; distinguish acquired observations from retained evaluation rows. |
| 27 | Retained crypto panel: 181 tickers, latest 1,760 jointly valid positive dates each, 318,560 rows per estimator. | Saved production handoff in `CONTEXT.md`, selection in `data/roughness_analysis.py`; excludes stablecoins/fiat. Source database unavailable. |
| 28 | Crypto table contains only Test: 314,940 windows for 20 observations and 304,080 for 80. | Derived as 181 × (1,760 − w), following `TimeSeriesDataset`; potential coverage, not completed forecasts or recomputed eligibility. All retained dates lie in the equity test period, not necessarily on the same dates as equities. |

| History length | Training | Validation | Test |
|---|---:|---:|---:|
| 20 observations | 8,664,516 | 1,806,548 | 4,479,681 |
| 80 observations | 7,004,011 | 1,692,154 | 4,295,773 |

### Defaults requiring validation

| # | Default or unresolved choice | Validation limit |
|---|---|---|
| 1 | Two seconds minimum between request starts. | Configured scheduler default; available summary does not establish exact acquisition-run settings. |
| 2 | Rate-limit retries after 30, 60, 120, 240 seconds. | Same limit. |
| 3 | Transient-error retries after 5, 15 seconds. | Same limit. |
| 4 | Original exchange-level listing totals. | Original listing inputs/run record unavailable in workspace; omit counts. |
| 5 | Exact experiment-specific input/output transforms, orders, widths, states, and floors. | Describe later in Implementation from saved configuration, rather than assume uniform settings. |
| 6 | Crypto transfer methodology and outcomes. | Deferred; no generalization claim. |

### Notation cross-check and model decisions

| # | Symbol / model | Choice and grounding |
|---|---|---|
| 1 | Observed proxy | Preserve Background's `σ̂²_GK`; introduce `v_i(t)` as a short alias for the original observed proxy. |
| 2 | Latent variance | Preserve `σ²(t)` for spot variance and `IV(t)` for integrated variance, not implied volatility. |
| 3 | Targets | `y_i(t)` is the project-adjusted target; `Y_i(t + 1)` is its scalar sample target. |
| 4 | History and parameters | Extend `X_i` to `X_i(t)`; retain `θ_i` local and `θ` global. |
| 5 | Architectural outputs | `a(t + 1)` precedes variance conversion; `ỹ` precedes prediction flooring; `ŷ` follows it. |
| 6 | Conversion | Identity for variance, exp(a) for log variance, exp(2a) for log volatility; positive training-derived lower forecast floor. |
| 7 | AR | Intercept plus coefficient times latest observed variance, with zero-conditional-mean forecast error. |
| 8 | SARIMA | Lag operator, four polynomial definitions, differencing orders and seasonal period; no deterministic trend, consistent with current SARIMAX construction. |
| 9 | GARCH | Adjusted log(Close/Open), causal within-ticker mean, residual and standardized innovation; ω > 0, α, β ≥ 0, α + β < 1. Scalar `v(t)` is conditional variance; volatility interface returns its square root. |
| 10 | HAR | Trailing means for {1, 5, 20} and {1, 5, 20, 40, 80}; explain project 20 versus Corsi's 22. |
| 11 | RFSV | Preserve `X(t) = log σ(t)`, α, m, ν, Bᴴ, H and c(H); explicitly distinguish scalar X from sample history. Reuse Background derivations. |
| 12 | RFSV implementation | Δ = 1 in observation time, exact bin masses, oldest-value tail extension; full preceding positive history compresses invalid/zero rows and is an exception to fixed windows. Separate matched dates are needed for comparison. |
| 13 | MLP | Three SiLU hidden affine maps, common width in first two, two-unit third, scalar readout. The roadmap's two-hidden-layer statement is stale; current code is authoritative. |
| 14 | LSTM | Concatenate preceding hidden state and current input; sigmoid input/forget/output gates; tanh candidate and cell output; cell recurrence and scalar readout. Neural vector `h(t)` is explicitly distinct from GARCH scalar `v(t)`. |
| 15 | SiLU-LSTM | Replace candidate and cell-output tanh with SiLU, retain sigmoid gates and cell recurrence. Distinct from deferred volatility-inspired LSTM. |
| 16 | HARNet | Single-channel bias-free convolution layers with ReLU, filters (5,4,2,2), dilations (1,5,20,40), receptive fields (1,5,20,40,80), and linear readout of all horizons. Use first two layers for 20, all four for 80. |

## Dataset and implementation work

- Replace only the Methods placeholder in `ref/final_report/thesis_structure.tex` with Data and Models and their numbered equations.
- Reuse existing primary-paper bibliography entries. Verify Gatheral et al. Equation (5.1) and the unnumbered Section 5.2 correction, and HARNet Section 4.1 against the original papers.
- Record code-specific qualifications in the thesis research notes; no new reference is required when the cited papers are already in `ref/thesis/references.md`.
- Create this plan after the Methods implementation, as requested, and update `CONTEXT.md` with the current handoff.

## Verification

- Check exact subsection/model order, equation labels and references, citations, symbol definitions, audit counts, and preservation of source outside Methods against the saved pre-edit source.
- Cross-check mathematical mappings directly against model, loader, and evaluator code. No new data scan: source database is unavailable.
- Compile from `ref/final_report/` with two `pdflatex -synctex=1` passes, producing the adjacent PDF and existing SyncTeX artifact. Inspect rendered Methods pages; fix introduced overflow.
- Obtain a fresh-context review with `judge/prompt.md`; validate its JSON using `python judge/validate.py REVIEW.json`; fix findings until at least 95/100 and no critical findings.
- Run no downloads, training, dataset-behavior changes, commits, or pushes.

Verification results are recorded in `judge/reviews/methods_checks.txt`, the cold-review JSON, and the Methods handoff in `CONTEXT.md`.

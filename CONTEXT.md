# Project context

Last updated: 2026-09-21

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
- The later volatility model will use an estimator-specific positive lower floor on variance forecasts, selected from training targets and held fixed for validation/test. Its numeric value awaits target selection; no upper forecast cap is planned. The metric function still rejects nonpositive inputs, so the forecast transform must be applied explicitly upstream.
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
  lags 1--400, and only within-ticker displacement pairs. The completed run
  wrote 64 figures plus moment, zeta, and Hurst summary CSVs under
  `imgs/roughness_analysis/`.

## Current plan position

For **Part I / VVSI**, the return-prediction rerun infrastructure and the
three production LSTM runs are complete. The next action is to compare and
report the window-size results.

For **Part II / the full thesis**, the revised volatility dataset is still incomplete. Model definitions now exist in `models/`, ahead of target selection; no project-data model fitting or volatility evaluation has run.

The VVSI LSTM still predicts `overnight_returns`; it has not yet been trained for volatility. The volatility estimator datasets and initial roughness analysis exist, but the canonical prediction target and horizon must be chosen before fitting or evaluation. Classical model computations, neural architectures, and synthetic checks are implemented; the VVSI package moved to `inference/` with its old checkpoints.

Completed or reusable parts of Step 1:

- equity universe construction;
- daily OHLCV acquisition and adjusted prices;
- chronological splitting, storage, normalization, plotting helper, and safe per-ticker windows.

Still required to finish Step 1:

1. Review the generated roughness estimates and select the canonical
   prediction target and horizon.
2. Adapt the dataset windows to that target and add lagged realized variance.
3. Add VIX or volatility commonality only if the initial evidence shows they
   are needed.

The next concrete task is to select the canonical variance estimator and forecast horizon, then wire within-ticker windows and fit the defined models on chronological equity data. Validation must select SARIMA orders, neural widths/windows, and an out-of-sample appropriate H.

## Planned later work

- Models under consideration: SARIMA, HAR, HARNet, GARCH, AR(1), MLP, LSTM variants, RFSV, TimesFM, and TinyTimeMixers. The small-language-model idea is explicitly low priority.
- The defined comparison metrics are QLIKE, MSE, MASE, MAE, and RMSE. Winsorization and joint ES/VaR loss are deferred; Patton's proxy-robustness result requires its assumptions and does not automatically cover every estimator here.
- Compare local versus global training and the feature sets RV-only, RV + commonality, and RV + commonality + VIX.
- Statistical comparison with a Model Confidence Set and forecast-efficiency checks with Mincer-Zarnowitz regressions.
- Economic comparison using Sharpe ratio and realized utility, followed by portfolio simulation.

## Repository map

- `data/filtering_stock.py`: ticker-universe cleaning.
- `data/data_fetching.py`: download, feature creation, SQLite storage, splits, normalization, plotting, and window dataset.
- `data/roughness_analysis.py`: variance-table construction, retained universe
  selection, top-volume samples, roughness/Hurst estimation, and figures.
- `models/base_lstm.py`: custom LSTM cell and sequence model; other model definitions live beside it.
- `inference/training.py`: VVSI training/evaluation metrics, checkpointing, and diagnostics.
- `inference/inference.py`: current executable VVSI training and test pipeline (`python -m inference.inference`).
- `ref/plan.md`: current research plan.
- `ref/thesis_notes.md`: paper notes and rationale.
- `ref/references.md`: bibliography/source list.
- `ref/progress_report.md`: earlier project report; useful history, but it predates the volatility-focused plan.
- `README.md`: describes the implemented return-prediction prototype; it is not a statement that the revised volatility plan is complete.

## Working-tree note

At the time of this update, `main` matches `origin/main`. `.claude/` is an existing untracked user directory and must not be modified or committed incidentally.

# Project context

Last updated: 2026-09-10

## Goal

This thesis project is moving from short-horizon stock-return prediction toward a comparison of statistical/econometric and machine-learning models for realized-volatility forecasting. The intended outcome is a fair comparison of local and global models, followed by a portfolio-utility evaluation. The research also covers volatility roughness (fBm/fOU and the Hurst exponent) and whether learned behavior generalizes across assets and datasets.

`ref/plan.md` is the current plan. `ref/thesis_notes.md` contains the literature notes supporting it.

## What is implemented

- NASDAQ and NYSE ticker-list ingestion and cleaning, including removal of non-equity instruments and consolidation of share classes.
- Bulk daily OHLCV download from Yahoo Finance for 2006-01-01 through 2026-01-01.
- Resumable, rate-limited maximum-history acquisition, with raw ticker-major
  OHLCV and 20/25/30-year coverage metadata stored under
  `D:\DBs\timeseries_analysis`.
- Split/dividend-consistent OHLC adjustment via `auto_adjust=True`.
- Scale-free features: intraday return, overnight return, relative high-low range, and log volume. Daily return was removed because it is determined by intraday and overnight returns.
- SQLite persistence in a `features` table.
- Chronological train/validation/test splits: train before 2016, validation from 2016 through 2018, and test from 2019 onward.
- Per-ticker log-volume z-score statistics fitted only on training data, with safe handling of unseen tickers and invalid standard deviations.
- Per-ticker rolling-window dataset construction. Data is sorted by `(Ticker, Date)`, and windows cannot cross ticker boundaries.
- A PyTorch LSTM implemented from individual gates, with Xavier initialization and a forget-gate bias of one.
- End-to-end return-model training, validation, testing, Weights & Biases logging, best-validation checkpointing, gradient clipping, GPU memory reporting, and `torch.compile` support.
- Several training runs and checkpoints exist locally. The earlier train/validation discrepancy led to fixes for mixed-ticker windows, inconsistent adjusted prices, and feature scaling.
- Literature notes cover HAR/HARNet, GARCH, rough volatility, global versus local models, volatility commonality, TSFMs, evaluation losses, and economic utility.

## Current plan position

We are in **Step 1: Build the dataset**, with the original return-prediction infrastructure complete but the revised volatility dataset incomplete.

The current code predicts `overnight_returns`; it does not yet implement the realized-volatility target described in `ref/plan.md`. Therefore the project has not reached the plan's **Define the models** implementation stage, even though an LSTM prototype already exists and candidate models have been identified in the research notes.

Completed or reusable parts of Step 1:

- equity universe construction;
- daily OHLCV acquisition and adjusted prices;
- chronological splitting, storage, normalization, plotting helper, and safe per-ticker windows.

Still required to finish Step 1:

1. Define and compute the daily realized-volatility proxy from OHLC data: Parkinson and reduced Garman-Klass estimators are planned.
2. Decide the canonical prediction target and horizon, then adapt the dataset windows to that target.
3. Add the planned predictors: lagged RV, market-volatility commonality, and VIX.
4. Run local and global descriptive/statistical analysis, including roughness and Hurst-exponent analysis.
5. Complete the running coverage scan and use its 20/25/30-year results to confirm the usable universe/data volume.
6. Prepare the available crypto intraday data later as an unseen universality test; it is not currently in the repository.

The next concrete task is to implement and validate the Parkinson and reduced Garman-Klass series in the data pipeline. After the volatility dataset is fixed, implement a small statistical baseline—preferably HAR/OLS—before adapting the LSTM and adding more complex models.

## Planned later work

- Models under consideration: SARIMA, HAR, HARNet, GARCH, OLS, MLP, LSTM variants, RFSV, TimesFM, and TinyTimeMixers. The small-language-model idea is explicitly low priority.
- Primary comparison metrics: QLIKE, MSE, and MASE, with bounded/winsorized losses where justified.
- Compare local versus global training and the feature sets RV-only, RV + commonality, and RV + commonality + VIX.
- Statistical comparison with a Model Confidence Set and forecast-efficiency checks with Mincer-Zarnowitz regressions.
- Economic comparison using Sharpe ratio and realized utility, followed by portfolio simulation.

## Repository map

- `data/filtering_stock.py`: ticker-universe cleaning.
- `data/data_fetching.py`: download, feature creation, SQLite storage, splits, normalization, plotting, and window dataset.
- `model/model.py`: custom LSTM cell and sequence model.
- `model/training.py`: training/evaluation metrics, checkpointing, and diagnostics.
- `model/inference.py`: current executable training and test pipeline.
- `ref/plan.md`: current research plan.
- `ref/thesis_notes.md`: paper notes and rationale.
- `ref/references.md`: bibliography/source list.
- `ref/progress_report.md`: earlier project report; useful history, but it predates the volatility-focused plan.
- `README.md`: describes the implemented return-prediction prototype; it is not a statement that the revised volatility plan is complete.

## Working-tree note

At the time of this update, `main` matches `origin/main`. `.claude/` is an existing untracked user directory and must not be modified or committed incidentally.

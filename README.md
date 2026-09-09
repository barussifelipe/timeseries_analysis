# Stock Return Prediction — A From-Scratch LSTM on 20 Years of Market Data

A deep learning pipeline that predicts short-horizon equity returns from daily OHLCV data
across the full NASDAQ and NYSE universe. The LSTM is implemented **cell-by-cell in PyTorch**
— no `nn.LSTM` — and trained as a single global model shared across all tickers.

Built as an academic research project (university thesis work, supervised).
**Status: research in progress** — the data pipeline and training loop are complete and
running; the architecture comparison and final Pareto-frontier evaluation are the current
work.

---

## Why this project is worth a look

Most time-series portfolio projects are a Kaggle CSV, a `model.fit()`, and an R² that quietly
leaks the future into the past. This one is the opposite: the interesting engineering is in
the parts that are *easy to get wrong*, and the git history documents each bug being found
and fixed rather than hiding it.

| Concern | How it's handled here |
|---|---|
| **Data leakage** | Normalization statistics are fit on the **training split only** and applied to val/test. Splits are chronological (train `<2016`, val `2016–2018`, test `≥2019`) — never random, so no future bar ever informs a past prediction. |
| **Non-stationarity** | Raw prices and volumes are never fed to the network. Every feature is scale-free and stationary by construction: intraday/overnight returns, relative high-low range, log-volume. |
| **Cross-sectional scale** | Volume is z-scored **per ticker**, so the feature answers "is this unusual *for this stock*" rather than encoding market cap. Tickers absent from train, or with degenerate variance, fall back to `0.0` instead of producing `NaN`/`inf`. |
| **Feature redundancy** | Daily return was deliberately **dropped** — it is fully determined by intraday × overnight return and carries no independent signal. |
| **Windowing correctness** | Rolling windows are built per-ticker on contiguous, `(Ticker, Date)`-sorted series. An earlier `(Date, Ticker)` sort silently mixed rows across different stocks inside a single window — the root cause of a large train/val loss gap, traced and fixed. |

That last row is the one worth reading twice: the bug did not throw an error. It produced a
plausible-looking training curve with a suspicious validation gap, and finding it required
reasoning about the data rather than the model.

---

## Architecture

### `FEBCellLSTM` — the LSTM cell, written out

Implemented from first principles for transparency and control over initialization:

- Input, forget, candidate, and output gates as explicit `nn.Linear` layers over the
  `[h_prev ; x_t]` concatenation (a single concat instead of separate `xWᵀ + hUᵀ` products).
- **Xavier uniform** initialization on all gate weights, suited to the symmetric
  sigmoid/tanh activations.
- **Forget-gate bias initialized to 1** — `sigmoid(1) ≈ 0.73` biases the cell toward
  retaining prior state early in training, a standard remedy for vanishing gradients through
  long unrolls.

```
c_t = f_t · c_prev + i_t · g_t
h_t = o_t · tanh(c_t)
```

### `FEBLSTM` — the sequence model

Unrolls the cell over a 30-day window and projects the final hidden state through a linear
layer to a single-step return prediction. **One global model for every ticker** — it has to
learn return dynamics that generalize across thousands of stocks, not memorize one.

---

## Engineering & MLOps

- **Experiment tracking** — every run logged to **Weights & Biases**: hyperparameters,
  per-epoch train/val metrics, and checkpoint artifacts. Run names encode the full config
  (`bs128_hs64_ws30_lr0.0001_epochs50`) so any result is reproducible from its name.
- **Checkpointing** — best-validation-loss only, saving model + optimizer state + epoch +
  losses, so training is resumable rather than merely saved.
- **Training stability** — gradient clipping (`max_norm=1.0`) applied **before**
  `optimizer.step()`, not after; a subtle ordering bug that makes clipping a no-op.
- **GPU throughput** — `torch.compile` (Triton backend), pinned-memory `DataLoader`s with
  parallel workers, and explicit VRAM instrumentation (allocated / reserved / peak) to size
  batches against real memory limits rather than by trial and error.
- **Metrics** — MAE, RMSE, MAPE and R² tracked side by side, computed under `no_grad` so
  monitoring never perturbs the training graph. On near-zero financial returns MAPE is
  reported but read with care — a case where the metric, not the model, is the trap.

---

## Data pipeline

```
NASDAQ/NYSE listings (CSV)
   └─► ticker cleaning ──► yfinance bulk download (2006–2026, split/dividend adjusted)
          └─► stationary feature engineering ──► SQLite (`features`)
                 └─► chronological split ──► per-ticker z-score (train-fitted)
                        └─► TimeSeriesDataset (30-day windows) ──► DataLoader
```

Details that matter:

- **Ticker universe construction** — share classes are collapsed to one primary ticker per
  company via regex base-symbol extraction, preferring the plain symbol, then Class A, then
  largest market cap. Non-equity instruments (funds, trusts, warrants, preferreds, notes,
  debentures) are filtered out by name.
- **Corrected price adjustment** — the initial download used `auto_adjust=False`, leaving
  Open unadjusted relative to Close across splits and dividends, which corrupted every
  overnight return. Fixed to adjust all price columns consistently.
- **Storage** — features are persisted to SQLite, so experiments re-read a clean table
  instead of re-downloading thousands of tickers on every run.
- **Windowing** — datasets are flattened into contiguous tensors with an explicit
  `valid_indices` list, so a window can never straddle a ticker boundary.

---

## Tech stack

**PyTorch** · **Python** · **pandas** / **NumPy** · **SQLite** · **Weights & Biases** ·
**Triton** / `torch.compile` · **yfinance** · **Matplotlib** · CUDA

---

## Repository layout

```
data/
  filtering_stock.py   # ticker universe: share-class dedup, instrument filtering
  data_fetching.py     # download, feature engineering, splits, z-scoring, Dataset
  src/                 # listing CSVs + SQLite feature store
model/
  model.py             # FEBCellLSTM + FEBLSTM (LSTM implemented from scratch)
  training.py          # train/eval loops, metrics, checkpointing, VRAM instrumentation
  inference.py         # experiment entrypoint: config, W&B init, train, test
ref/                   # project brief, references, progress report, research notes
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch pandas numpy yfinance wandb matplotlib

python -m data.data_fetching   # build the SQLite feature store (one-off, slow)
python model/inference.py      # train + evaluate, logging to W&B
```

Hyperparameters live at the top of `model/inference.py`. A W&B account is needed for logging
(`wandb login`).

---

## Roadmap

- **GARCH-family volatility model** as a statistical benchmark, characterizing volatility
  structure independently of the network.
- **Architecture sweep** — the feed-forward variant on moving-average/slope ratios
  (`MC(x)/O`, `SL(x)/O` across 5–200 day lookbacks) alongside the LSTM.
- **Pareto frontier** over (mean error, error standard deviation) to compare architectures on
  both accuracy *and* consistency — because in finance a model that is occasionally
  catastrophic is worse than one that is uniformly mediocre.

---

## What I'd want a reader to take from this

That I can build an ML system end to end — data acquisition, feature design, model
implementation, training infrastructure, experiment tracking — and that I debug the quiet
failures, not just the loud ones. Financial return prediction is close to the hardest
signal-to-noise problem in applied ML; the honest deliverable is a rigorous, leakage-free
pipeline and a clear-eyed evaluation, not a headline accuracy number.

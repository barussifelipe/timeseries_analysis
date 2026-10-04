# Data Section Local and Crypto Selection Implementation Plan

## Implementation decisions

1. In Section 3.1 (`Data`), document the five-stock local equity selection: NVDA, AAPL, NFLX, GOOG, and AMZN, selected from the 2,714-ticker raw-history cohort by highest mean daily nonnegative trading volume before 2016 (retaining one share class per company, with GOOG outranking GOOGL and AMZN following).
2. Explain the equity volume selection rationale: corporate share-split conventions in U.S. equity markets generally maintain nominal share prices within comparable retail-accessible ranges ($10^1$ to $10^2$), allowing raw share volume to naturally capture prominent market leaders.
3. In Section 3.1 (`Data`), document the five-asset cryptocurrency selection for out-of-sample transfer evaluation: BTC, ETH, SOL, XRP, and DOGE, selected across 1,760 aligned dates from 2019 to 2025.
4. Provide the concise economic and statistical rationale for crypto selection: unlike equities, cryptocurrency token supplies span more than ten orders of magnitude without standardized denominations; selecting by raw token volume captures micro-priced tokens (such as BTT, WIN, LUNA, and TRX) while omitting primary market drivers like BTC (which ranks 113th in raw token volume). Assets are therefore selected by mean daily dollar volume ($\text{Close} \times \text{Volume}$), ensuring representative market liquidity.
5. In Section 3.1 (`Data`), document that while equities rely solely on daily OHLC range data, the cryptocurrency dataset provides both the daily Garman--Klass (GK) estimator in \eqref{eq:garman-klass-variance} and the 15-minute intraday realized variance (RV) proxy in \eqref{eq:realized-variance} (summing 96 fifteen-minute squared log returns per UTC day), enabling cross-proxy transfer evaluation.
6. In Section 5.2 (line 933), streamline the existing verbose cryptocurrency selection sentence to refer concisely to the dollar-volume selection and avoid unit-denomination bias.
7. Compile the thesis document from `ref/final_report/` with two direct `pdflatex -synctex=1` passes, ensuring no undefined references, no new fatal errors or bad overflows, and maintaining existing document structure.

## Dataset and implementation work

- Edit `ref/final_report/thesis_structure.tex` in Section 3.1 (`\subsection{Data}`) to insert paragraphs detailing local equity selection and concise cryptocurrency selection.
- Update line 929 in Section 5.2 of `ref/final_report/thesis_structure.tex` to maintain concise phrasing.
- Run two `pdflatex -synctex=1` passes in `ref/final_report/` to update `thesis_structure.pdf` and `thesis_structure.synctex.gz`.

## Verification

- Run `python judge/check_technical_evaluation.py` to ensure all existing table and structure checks continue to pass.
- Inspect the compilation log for undefined references and layout warnings.
- Obtain an independent cold review in a fresh, cleared context using `judge/prompt.md` and validate its JSON with `python judge/validate.py`.
- Preserve existing model checkpoints, databases, and unrelated files; do not run long training jobs, downloads, or git pushes.

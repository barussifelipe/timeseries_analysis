# Crypto Realized Variance Thesis Table Implementation Plan

## Implementation decisions

1. Add Table 5.5 (`tab:crypto-rv-results`) immediately following the Garman--Klass cryptocurrency evaluation table (Table 5.4, `tab:crypto-results`) and its summary discussion in Section 5.4.1.
2. Format Table 5.5 identically to Table 5.4 using `longtable` with `@{}p{.235\linewidth}>{\raggedleft\arraybackslash}p{\dimexpr .165\linewidth-2\tabcolsep\relax}...@{}`:
   - Columns: `Model`, `MAE ($\times10^{-3}$)`, `MASE`, `MSE ($\times10^{-2}$)`, `RMSE ($\times10^{-1}$)`, `QLIKE`.
   - Rows: 15 equity-trained Global fits on 8,800 retained crypto targets (9 window-20 rows, horizontal rule `\hline`, 6 window-80 rows).
   - Order: `MLP-20`, `Base LSTM-20`, `SiLU-LSTM-20`, `HARNet-20`, `AR(1)`, `HAR-20`, `GARCH(1,1)`, `SARIMA`, `RFSV-20`, `MLP-80`, `Base LSTM-80`, `SiLU-LSTM-80`, `HARNet-80`, `HAR-80`, `RFSV-80`.
   - Colors: Red for minimum, blue for second minimum, dark orange (`\textcolor{orange!65!black}`) for maximum per column across all rows.
3. Scaling factors and precision:
   - MAE: scaled by $10^3$ (`MAE ($\times10^{-3}$)`), 5 decimal places (e.g. `7.03521` to `10.05437`).
   - MASE: unscaled, 6 decimal places (e.g. `0.872923` to `1.142607`).
   - MSE: scaled by $10^2$ (`MSE ($\times10^{-2}$)`), 5 decimal places (e.g. `0.96381` to `2.40041`).
   - RMSE: scaled by $10^1$ (`RMSE ($\times10^{-1}$)`), 5 decimal places (e.g. `0.98174` to `1.54932`). This maintains $\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$ unit consistency.
   - QLIKE: unscaled, standard precision; values exceeding $10^4$ from prediction floor activations formatted in scientific notation (e.g. `$1.34\times10^{10}$`, `$4.63\times10^8$`, `$3.45\times10^6$`, `$5.21\times10^7$`).
4. Accompanied by concise framing text before Table 5.5 explaining the cross-proxy transfer evaluation on 15-minute Realized Variance, and a concise summary paragraph after Table 5.5 highlighting RFSV-80's squared-loss leadership, Base LSTM-20's MASE/QLIKE leadership, and the floor-hit impact on QLIKE.
5. Update `judge/check_technical_evaluation.py` to:
   - Account for the 4th `longtable` and updated header count assertions.
   - Validate all 15 rows of `tab:crypto-rv-results` against `imgs/eval/crypto_rv_eval/metrics.csv` with exact scaling, color rankings, and RMSE-MSE identities.
6. Recompile `ref/final_report/thesis_structure.tex` in place with two passes of `pdflatex -synctex=1` and verify 0 errors, 0 undefined citations, and clean pagination.

## Dataset and implementation work

- Modify `ref/final_report/thesis_structure.tex` to add introductory text, Table 5.5, and summary discussion.
- Update `judge/check_technical_evaluation.py` to test both crypto tables.
- Run `pdflatex` compilation twice and inspect visual layout with `pdftoppm`.

## Verification

- Automated technical evaluation check verifying exact metric values, scales, colors, and row counts across all 4 thesis tables (75 total rows).
- Visual check of rendered PDF pages.
- Independent fresh-context cold review using `judge/prompt.md` validated via `python judge/validate.py`.

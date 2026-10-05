# Technical Evaluation Residual Plots Implementation Plan

## Implementation decisions

1. Promote Overall Result, Crypto Evaluation, Model Statistical Significance, Residual Structure, and Parameter Complexity to subsections in that order.
2. Use the ten-model intersection shared by the final MCS sets of all five stocks under both 20- and 80-observation bootstrap blocks. This is narrower than the union of models retained for at least one stock.
3. Keep the matched five-stock equity panel: NVDA, AAPL, NFLX, GOOG, and AMZN; 1,760 retained target dates and 8,800 ticker/date forecasts per variant.
4. Analyze one-step errors in daily, unannualized adjusted Garman--Klass variance. Define an individual raw residual as actual minus forecast variance and an individual log residual as log actual minus log forecast variance.
5. Define the raw daily timeline as the mean across tickers of actual variance minus the mean forecast variance. Define the log daily timeline as the log of mean actual variance minus the log of mean forecast variance. The latter is generally different from the mean individual log residual.
6. Raw errors retain daily variance units; log errors are dimensionless. Positive errors indicate underprediction. Histograms show all individual residuals and a display-only central 99.5% panel; the saved forecasts and losses are unchanged.
7. Group the variants by Base LSTM, HARNet, MLP, and RFSV. Include Base LSTM-20 Global; Base LSTM-80 Global and Local; HARNet-20 Global; HARNet-80 Global and Local; MLP-20 Global; MLP-80 Global; RFSV-20 Local; and RFSV-80 Local.
8. For each variant, reuse the four saved PNGs under `imgs/eval/local_eval/residuals/<model>/<scope>/`: `timeline_mean_raw.png`, `timeline_mean_log.png`, `histogram_raw.png`, and `histogram_log.png`. Place raw then log timelines side by side above raw then log histograms. Do not regenerate forecasts or plots.

## Thesis and checker work

- Edit `ref/final_report/thesis_structure.tex` to add the residual equations, units, ten bold variant introductions, four model subsubsections, and ten captioned figure groups.
- Update `judge/check_technical_evaluation.py` to verify the new heading hierarchy and the exact 40 ordered image paths, captions, variant names, and figure labels.

## Verification

- Run `python judge/check_technical_evaluation.py`; require all ten groups and saved-image paths to pass.
- Compile `thesis_structure.tex` twice from `ref/final_report/` using `pdflatex -synctex=1`; inspect affected PDF pages for layout, references, and readable plots.
- Obtain a fresh-context review using `judge/prompt.md`, validate its JSON with `python judge/validate.py`, and resolve findings to at least 95/100 with no critical findings.
- Update `CONTEXT.md` with the completed thesis change and next concrete task. No training, download, rescore, commit, or push is part of this step.

# Technical Evaluation Implementation Plan

## Implementation decisions

1. Add Overall Result, Parameter Complexity, and Residual Structure under Technical Evaluation, in that order.
2. Save this agreed plan as `ref/implementation_plan/technical_evaluation.md` during implementation.
3. Preserve saved metrics and reuse existing residual plots. Do not retrain or rescore forecasts.
4. Present global-only results first, then the matched five-stock Local × Global table.
5. Report different winners across metrics and mixed local/global evidence; avoid a blanket global-superiority claim.
6. Separate RFSV's final-date results from full-period rankings.
7. Disclose different eligible populations for global window-20 and window-80 results.
8. Align Evaluation Setup with saved tables: observation-weighted metrics, RMSE = √(pooled MSE), and test-relative MASE. State the departure from training-scaled MASE.
9. Retain the limitation that training configurations differ, preventing attribution solely to local/global scope.
10. Compare parameter combinations using 2,714 tickers and B = 2⁶⁴ possible values per scalar parameter.
11. For p fitted scalar parameters, show Local = B^(2,714p) and Global = B^p. Label Local as extrapolated from the architecture, since only five local tickers were fitted.
12. Explain that parameter combinations equal distinct functions only under the distinguishability assumption; neural symmetries can violate it.
13. Count fitted coefficients and neural weights/biases; exclude optimizer state, filtering state, fixed hyperparameters, and preprocessing metadata.
14. Identify the overall best and worst for each metric across both modalities using matched five-stock full-period results.
15. Use matched five-stock residual plots for those selected models, ensuring the plotted population matches the ranking.
16. Include raw/log histograms and raw/log daily mean timelines; omit individual dated residual scatter plots.
17. Reuse each model's figures across metric discussions rather than duplicating them.
18. Explain the saved nine-ticker exclusions: zero raw variance on both final and preceding dates produces a zero final-date MASE denominator, even after the shared zero replacement.
19. Flush the parameter-combination table before Residual Structure so it cannot float into that section.
20. Number tables and figures within thesis sections, consistently with equations.
21. Add Global/Local as the last complexity-table column: B^(−2,713p), displayed exactly because decimal rounding would yield zero. The user's latest instruction supersedes the earlier percentage and third-column choices.

## Overall Result and Parameter Complexity

- Create editable LaTeX tables from `imgs/eval/volatility_test_metrics.csv` and `imgs/eval/local_eval/metrics.csv`, retaining precision sufficient to distinguish rankings.
- Include MAE, MASE, MSE, RMSE, QLIKE, and available observation-count and floor-hit information.
- Highlight minima within comparable groups; present RFSV separately.
- Verify the following counts against selected code before insertion.

| Model specification | Scalar parameters p |
|---|---:|
| AR(1) | 2 |
| HAR-20 / HAR-80 | 4 / 6 |
| GARCH(1,1) | 3 |
| SARIMA | 5 |
| RFSV-full | 2 |
| MLP-20 / MLP-80 | 22,021 / 37,381 |
| Base LSTM-20 / Base LSTM-80 | 67,201 each |
| SiLU-LSTM-20 / SiLU-LSTM-80 | 67,201 each |
| HARNet-20 / HARNet-80 | 13 / 19 |

## Residual Structure

| Metrics | Overall best | Overall worst |
|---|---|---|
| MAE, MASE | SARIMA Local | AR(1) Global |
| MSE, RMSE | MLP-80 Global | AR(1) Global |
| QLIKE | MLP-80 Global | SiLU-LSTM-80 Local |

- Use existing matched-population plots for these four distinct models and organize discussion around the three comparison groups, with clear figure references.
- Explain positive residuals as underprediction and negative residuals as overprediction.
- Distinguish individual log residuals from daily log-of-means timelines.
- Disclose central-99.5% display trimming and differing existing axis scales.
- Discuss visible bias, asymmetry, tails, and temporal patterns without asserting formal independence or whiteness.
- Label model selection as retrospective illustration from test results.

## Verification

- Check table values, overall extrema, parameter counts, figure paths, and captions against saved artifacts and code.
- Verify the three requested subsubsections and preserve unrelated edits.
- Compile from `ref/final_report/` with two `pdflatex -synctex=1` passes; inspect tables, figures, references, and layout.
- Obtain fresh-context review using `judge/prompt.md`, validate its JSON, and resolve findings until at least 95/100 with no critical findings.
- Update `CONTEXT.md` with the completed section, conventions, limitations, and next task.
- Run no downloads, training jobs, commits, or pushes.

## Implementation assumptions

1. Use the full-history global log to recover counts absent from the global CSV; do not use the older windowed-RFSV log, whose scores differ.
2. Six significant digits and explicit column scale factors retain every reported ranking.
3. B is an idealized scalar encoding budget, not a claim that every bit pattern is a finite admissible coefficient or every vector represents a distinct function.

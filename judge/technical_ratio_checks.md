# Ratio-column verification

- Latest user instructions: remove percentage conversion, retain only Global/Local ratio, and place that column last.
- `python judge/check_technical_evaluation.py`: PASS, including exact ratio exponents 2713p and last-column header, plus original saved-score and placement checks.
- Two direct `pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error thesis_structure.tex` passes: exit 0, 46-page PDF and adjacent SyncTeX regenerated. No new overflow or unresolved-reference warnings; existing unrelated overflows remain at 81-82, 263, 702-703, and 707-708.
- Rendered page 35 inspected: Table 5.5 column order is Model, p, Local (extrapolated), Global, Global/Local. All 12 ratio entries are B^(-2713p), with no factor 100 or percent label. Table fits and remains before Residual Structure.
- Focused source diff `judge/technical_ratio.diff` changes only header/cell order and ratio explanatory wording. Other text and metric values are preserved. Plan and focused check follow the latest user correction.

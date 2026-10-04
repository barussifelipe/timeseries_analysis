# Crypto Evaluation Data Issues Implementation Plan

## Implementation decisions

1. Retain the existing Table 5.4 layout, font scaling, and results without alteration. The table continues to report the 15 equity-trained Global model configurations on the five dollar-volume-selected cryptocurrencies (BTC, ETH, SOL, XRP, DOGE) over their 1,760 retained target dates.
2. In Section 5.4.1 (Experimental Results), expand the explanatory paragraph immediately preceding Table 5.4 to explicitly document the RV validity requirements and the origin of skipped dates:
   - Constructing daily unannualized Realized Variance strictly requires 96 complete, clean 15-minute intraday bars per 24-hour UTC day and the preceding day's 23:45 close for continuity (calculating $r_1(t) = \ln(P(t, \text{00:00}) / P(t-1, \text{23:45}))$).
   - Dates failing these conditions (due to vendor collection dropouts, API holes, or missing preceding boundary closes) were deemed invalid and omitted.
   - To preserve a matched, harmonized evaluation panel across estimators (GK, Parkinson, and RV), candidate dates were filtered by their intersection, retaining the latest 1,760 jointly valid observations per asset through December 31, 2025.
   - Consequently, models forecast the next recorded valid observation from preceding valid observations, with 64--65 scored targets per coin spanning a two- or three-day calendar gap where intermediate raw dates lacked valid intraday coverage.
3. Keep the paragraph concise and tightly fitted within the remaining vertical space on page 36 before Table 5.4, avoiding awkward page-break spills or table displacement.
4. Recompile `ref/final_report/thesis_structure.tex` in place with two passes of `pdflatex -synctex=1` to update `thesis_structure.pdf` and `thesis_structure.synctex.gz`. Verify 0 errors, 0 undefined citations, and no layout regressions.

## Dataset and implementation work

- Edit Section 5.4.1 in `ref/final_report/thesis_structure.tex` immediately before Table 5.4 (`tab:crypto-results`).
- Update `judge/check_data_selection_text.py` or add a targeted check to verify that the RV validity and calendar gap documentation is present and accurate.
- Compile PDF in two passes and inspect the rendered visual layout with `pdftoppm`.

## Verification

- Small runnable test verifying the presence of the RV validity conditions, boundary close explanation, and calendar gap count in the LaTeX source.
- Two-pass LaTeX compilation check with zero errors and no undefined citations.
- Visual inspection of page 36 and 37 layouts to verify no pagination or overflow regressions.
- Independent cold review using `judge/prompt.md` validated via `python judge/validate.py`.

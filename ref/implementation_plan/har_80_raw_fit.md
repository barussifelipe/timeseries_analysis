# HAR-80 Raw-History Fit Implementation Plan

## Implementation decisions

1. Add `models/har_80.py` as a separate linear model with an intercept and 1-, 5-, 20-, 40-, and 80-observation variance averages.
2. Keep the adjusted daily, unannualized Garman–Klass variance target and next recorded observation horizon used by the raw-history statistical fits.
3. Require 80 consecutive recorded, valid, positive-variance input observations from one ticker; a valid zero target is floored with the minimum positive pre-2016 variance and then breaks the input run.
4. Fit global ordinary least squares on eligible targets before 2016-01-01, one contribution per target. The existing streamed normal-equation accumulator uses a variance scale and records its condition number.
5. Expect 7,004,011 train, 1,692,154 validation, and 4,295,773 test target dates. The change from window 20 excludes 1,660,505 train, 114,394 validation, and 183,908 test dates. Direct model comparison needs the shared target-date intersection.
6. Save a compact, fit-only JSON artifact under `inference/checkpoints/har_80/garman-klass/global/raw_history_w80/fit.json`; create no W&B run and perform no validation/test scoring.
7. Keep HAR(1,5,20) and all existing checkpoint paths unchanged. The new model is separate from HARNet-80, which has trainable convolutional stages.

## Verification

- Check the five summaries and streamed coefficients against direct ordinary least squares on two positive ticker histories.
- Confirm full-data eligibility counts and floor before fitting, then reload the fit and check six finite coefficients and convergence.
- Obtain the required fresh-context cold review and validate its JSON.

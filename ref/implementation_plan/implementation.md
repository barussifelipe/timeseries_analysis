# Models Setup Implementation Plan

## Implementation decisions

1. Add exactly Input data, Hyperparameters, and Fitting directly under Implementation, in that order; do not add Models Setup.
2. Use separate model bullets for Input data and Fitting: AR, SARIMA, GARCH, HAR, RFSV, MLP, LSTM, SiLU-LSTM, HARNet.
3. Use one bullet per hyperparameter in Hyperparameters.
4. Global fits pool the 2,714-ticker cohort. Local fits use NVDA, AAPL, NFLX, GOOG, and AMZN separately.
5. Preserve training before 2016, validation in 2016–2018, and testing in 2019–2025; fitting and initialization statistics use training only.
6. Target next-recorded-observation daily unannualized adjusted Garman–Klass variance. Preserve Methods' zero-target adjustment and separate prediction floor; local floors come from each ticker's training history.
7. AR forecasts from one lag but its least-squares samples meet 20-observation eligibility.
8. SARIMA/GARCH fit valid pre-2016 segments of at least 21 observations. Code-validated clarification: selected evaluation carries preceding within-ticker filtering state, with window-20 scoring eligibility, rather than truncating forecast history to 20 observations. SARIMA treats invalid observations as missing; GARCH resets variance and causal mean after invalid or zero-variance inputs.
9. Report 20- and 80-observation MLP, LSTM, SiLU-LSTM, HAR, and HARNet variants.
10. RFSV T counts all preceding valid positive observations within the forecast ticker, excluding target/future observations. Omit invalid/zero inputs; fixed windows require every input to be valid and positive.
11. Reference the existing Methods 20/80 sample-count table without duplicating it; leave Methods content unchanged.
12. AR/SARIMA/HAR/HARNet inputs are variance; GARCH uses causally demeaned adjusted within-session log returns.
13. RFSV receives positive variance, transforms it internally to log variance for forecasting, and estimates roughness from training log volatility.
14. MLP/LSTM use log volatility plus adjusted within-session returns and output log variance.
15. SiLU-LSTM uses variance plus adjusted within-session returns and direct variance outputs. Preserve sigmoid gates; SiLU replaces candidate and cell-output activations.
16. HAR averaging horizons are {1,5,20} and {1,5,20,40,80}.
17. AR order is 1 with intercept.
18. SARIMA orders are (1,0,1)(1,0,1,5).
19. GARCH orders are (1,1), with positive omega, nonnegative alpha/beta, and alpha + beta < 1.
20. Recurrent hidden width is 128.
21. MLP has three SiLU hidden layers with widths 128 → 128 → 2 and a scalar readout. The roadmap's two-layer description is stale; verified current code takes precedence.
22. HARNet uses a single channel: filter lengths (5,4), dilations (1,5), adding lengths (2,2) and dilations (20,40) for 80; no convolution biases and ReLU after each convolution.
23. Adam initial learning rate is 0.001.
24. Neural batch size is 128.
25. Maximum epochs is 20; selected global LSTM-20's historical 30e suffix is not the cap.
26. Patience is 10 consecutive unimproved epochs; improvements reset the counter.
27. Reduce learning rate once by factor 0.1 after the first patience interval, then stop after another interval, subject to the epoch cap.
28. Neural gradient clipping norm is 1.
29. Neural random seed is 42; training batches shuffle while within-window sequence order remains chronological.
30. RFSV moment orders are {1,1.5,2,3,4}.
31. RFSV observation lags are 1–400; H and ν² are fitted, not selected hyperparameters.
32. AR/HAR fit by ordinary least squares on matching eligible training targets.
33. SARIMA uses pooled state-space likelihood, first-segment starting fit, and Powell optimization (100 pooled iterations maximum).
34. GARCH uses constrained Gaussian likelihood and L-BFGS-B (100 iterations maximum). Initialize (omega, alpha, beta) at (0.1s,0.1,0.8) for training residual mean square s; initialize segment variance from its residual mean square with lower bound 1e-8s.
35. RFSV uses training-only log-moment regressions; fit ζ(q) = Hq through the origin and ν² from the exponential of the q=2 intercept.
36. MLP/LSTM/SiLU-LSTM fit with minibatch Adam. MLP Xavier-initializes output weights before scaling by 0.01; recurrent models scale default output weights by 0.01. Output bias is log training median variance for MLP/LSTM, training median variance for SiLU-LSTM.
37. HARNet initializes from its matching training-only HAR fit, then optimizes with Adam.
38. QLIKE is neural training/validation objective except SiLU-LSTM, which trains raw unfloored direct outputs with variance-unit MSE and selects checkpoints with floored validation MSE. Statistical models retain OLS, likelihood, or moment objectives. QLIKE evaluates all models.
39. Save the best configured validation-loss checkpoint; test targets do not select checkpoints or enter fitting.
40. Explain observed earlier SiLU QLIKE instability only as a hypothesis. Cite Elfwing, Uchibe, and Doya, printed Equation (10), https://arxiv.org/pdf/1702.03118: SiLU′(z) approaches 1 for large positive z while tanh′(z) approaches zero. Sensitivity amplification through recurrent weights/gates is conditional, not guaranteed.
41. Successful SiLU configuration changes input transformation and output parameterization as well as loss; do not attribute stability solely to the loss switch.
42. Update thesis bibliography, ref/thesis/references.md, ref/thesis/thesis_notes.md, and CONTEXT.md. No model, dataset, interface, or scoring changes.

## Thesis implementation work

- Fill `ref/final_report/thesis_structure.tex` Implementation with the agreed model inputs, subsets, hyperparameters, initialization, fitting, and qualified loss/stability explanation.
- Keep the source paper's variable names and printed equation number in research notes. Use parenthesized time indices in thesis equations.
- Record this plan in `ref/implementation_plan/implementation.md` after drafting, as additionally requested.

## Verification

- Cross-check selected experiment definitions in `inference/evaluate_volatility_test.py` and `inference/evaluate_local_global.py` against launchers, fitting code, and available metadata/reports. Report missing checkpoints as unverified; do not claim to have reloaded them.
- Check exactly three Implementation subsections, model bullets, hyperparameter bullets, chronological fitting, and units/output conversions.
- Compile from `ref/final_report/` with two `pdflatex -synctex=1` passes, preserving adjacent PDF/SyncTeX. Inspect Implementation pages, citations/references, and overflow.
- Run `git diff --check`; obtain a fresh-context reviewer using `judge/prompt.md` and validate its JSON with `python judge/validate.py`. Require at least 95/100 and no critical findings; fix and rereview if necessary.
- Do not download data, train models, score experiments, commit, or push.

## Defaults requiring validation

1. Current evaluation selectors define the experiments. `inference/checkpoints/` is absent here, so checkpoint contents remain unverified; code and saved reports support the descriptive configuration.
2. Keep Implementation descriptive. RFSV final-date evaluation and matched-observation limitations remain for Experimental Results.
3. Preserve unrelated local edits and Methods prose. The initial working tree was clean.

## Completion evidence

- Implementation text and research notes completed. Exactly three subsections and 9/17/9 bullets checked; Methods unchanged; final two-pass direct build produced the adjacent 36-page PDF and SyncTeX, with only preexisting overfull warnings. Pages 26–32 inspected. `git diff --check` passed.
- Fresh-context `judge/reviews/models_setup_review.json` passed 100/100 with no findings and validated JSON. The review corrected draft failure wording to distinguish the SiLU QLIKE forecast rejection from a separate MSE trial's nonfinite gradients. Checkpoint contents remain unverified. Next thesis task: Experimental Results.

# Price Reconstruction Implementation Plan

## Implementation decisions

### Agreed

1. Place both figures in thesis Section 6.6, Price Reconstruction.
2. Use saved MLP-80 Global forecasts for AAPL, AMZN, GOOG, NFLX, and NVDA, and infer Base LSTM-80 Global forecasts for the full eligible test cohort without refitting.
3. For each ticker-date, define drift as the rolling conditional mean from an intercept-and-slope AR(1) fitted by OLS to the 252 preceding valid adjusted log(Close/Open) returns (251 consecutive lagged pairs); predict from the latest preceding log return. Exclude the target day from the fit. If lagged inputs have no sample variation, use the OLS-equivalent constant fit (slope zero, intercept equal to the dependent-return mean).
4. Simulate one adjusted Close from the target adjusted Open as `Open * exp(drift + sqrt(predicted daily GK variance) * Z)`; this is a one-day post-open adaptation of the log-price diffusion in Equation 2.48, with drift already expressed as a log-return mean.
5. Draw independent standard-normal shocks with NumPy seed 0 in ticker-major, date-minor order across the global eligible forecast keys. Reuse the same shock for any local key.
6. Average actual and simulated adjusted Closes over identical eligible tickers on each date. Use blue for actual and red for simulation, labeled as a simulation.
7. Exclude ticker-dates lacking 252 preceding valid observations, count exclusions, and retain matched observations for both lines. Rolling AR(1) uses information available before each target even on validation or test dates; it is not a once-fitted pre-2016 coefficient.

### Validated default

8. Reuse the saved local CSV and fitted global checkpoint. The existing evaluator emits ticker-level global forecasts in target order; inference needs no training.
9. Show the global daily mean adjusted Close on a logarithmic vertical axis because the retained source price levels span several orders of magnitude; keep the local axis linear. This changes display only, not observations or arithmetic means.

### Agreed follow-up

10. Exclude WHLR from the global adjusted Close and log-mean Close plots because its extreme stored adjusted price level dominates the arithmetic mean and zero-variance days cause it to leave and re-enter the window-80 forecast set. Keep the fitted model, source data, other evaluation artifacts, and five-stock subset unchanged. This removes 1,510 plotted global price ticker-dates, with no training or validation data changes.
11. Add a separate log-Close view for each population using `log(daily arithmetic mean adjusted Close / USD)` for both actual and simulated lines. Do not average individual stock log Closes. Keep the same plotted ticker-dates, dates, and shocks as the corresponding price-level plot.
12. Add two gross-return plots, one per scope. Divide each actual and simulated adjusted Close by that ticker-date's adjusted Open, then average the resulting Close/Open ratios on each date. The lines are gross returns with 1 meaning no intraday change; subtract 1 to obtain net simple returns. Do not divide the cross-stock mean Close by a cross-stock mean Open.
13. Describe the predicted-variance-based red line as a forecast-conditioned single-draw simulation: the variance forecast controls dispersion, while the sampled shock determines one realized scenario. It is not a unique directional point forecast.
14. Add a global gross-return plot using the fitted MLP-80 Global variance model. Match the global Base LSTM-80 return plot's eligible ticker-dates, adjusted Opens, drift, and shocks. This changes only the simulated line's variance forecast.
15. Include WHLR's 1,510 eligible ticker-dates in both global gross-return plots while continuing to exclude it from global adjusted Close and log-mean Close plots. The return plot populations therefore differ from the price plot population; both return models still use identical observations.
16. Apply the rolling AR(1) drift in decision 3 to all seven reconstructed views at the user's request. The variance models, target populations, adjusted prices, and seed-0 shocks stay fixed; simulated prices and gross returns change. This is a forecast-time rolling fit, not a retraining of the saved variance models.

## Plot and thesis work

- Add `data/plot_price_reconstruction.py` and save seven PNGs and seven date-level CSVs under `imgs/data_properties/`.
- Add displayed definitions and seven figure descriptions in Section 6.6. State axes, populations, models, and line meanings without interpretation, and document the WHLR diagnostic exclusion.
- Record the observed exclusion counts and next task in `CONTEXT.md`.

## Verification

- Check a small two-ticker example for rolling AR(1) OLS agreement, target-day exclusion, preceding-only 252-valid-observation windows, ticker boundaries, invalid-price rejection, key alignment, shared shocks, and equal plotted populations.
- Run the script on saved artifacts; reconcile each CSV count with forecast and exclusion counts.
- Check `log(mean Close)` against a known two-stock example and confirm that the two log CSVs retain the price-level date counts.
- Check that mean ticker-level Close/Open differs from mean Close divided by mean Open on a two-stock example, and that the global return CSV counts exceed the global price CSV counts by the eligible WHLR rows.
- Match every date and stock count in the new global MLP-80 gross-return CSV to the global Base LSTM-80 gross-return CSV; confirm their actual lines match and their simulated lines use the same shocks.
- Compile `thesis_structure.tex` twice in place with `pdflatex -synctex=1`.
- Obtain fresh-context cold review, validate its JSON, and resolve findings.

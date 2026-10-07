# Price Reconstruction Implementation Plan

## Implementation decisions

### Agreed

1. Place both figures in thesis Section 6.6, Price Reconstruction.
2. Use saved MLP-80 Global forecasts for AAPL, AMZN, GOOG, NFLX, and NVDA, and infer Base LSTM-80 Global forecasts for the full eligible test cohort without refitting.
3. For each ticker-date, define drift as the arithmetic mean of the 252 preceding valid adjusted log(Close/Open) observations, excluding the target day.
4. Simulate one adjusted Close from the target adjusted Open as `Open * exp(drift + sqrt(predicted daily GK variance) * Z)`; this is a one-day post-open adaptation of the log-price diffusion in Equation 2.48, with drift already expressed as a log-return mean.
5. Draw independent standard-normal shocks with NumPy seed 0 in ticker-major, date-minor order across the global eligible forecast keys. Reuse the same shock for any local key.
6. Average actual and simulated adjusted Closes over identical eligible tickers on each date. Use blue for actual and red for simulation, labeled as a simulation.
7. Exclude ticker-dates lacking 252 preceding valid observations, count exclusions, and retain matched observations for both lines.

### Validated default

8. Reuse the saved local CSV and fitted global checkpoint. The existing evaluator emits ticker-level global forecasts in target order; inference needs no training.
9. Show the global daily mean adjusted Close on a logarithmic vertical axis because the retained source price levels span several orders of magnitude; keep the local axis linear. This changes display only, not observations or arithmetic means.

### Agreed follow-up

10. Exclude WHLR only from the global price-reconstruction plots because its extreme stored adjusted price level dominates the arithmetic mean and zero-variance days cause it to leave and re-enter the window-80 forecast set. Keep the fitted model, source data, other evaluation artifacts, and five-stock subset unchanged. This removes 1,510 plotted global ticker-dates, with no training or validation data changes.
11. Add a separate log-Close view for each population using `log(daily arithmetic mean adjusted Close / USD)` for both actual and simulated lines. Do not average individual stock log Closes. Keep the same plotted ticker-dates, dates, and shocks as the corresponding price-level plot.
12. Add two gross-return plots, one per scope. Divide each actual and simulated adjusted Close by that ticker-date's adjusted Open, then average the resulting Close/Open ratios on each date. The lines are gross returns with 1 meaning no intraday change; subtract 1 to obtain net simple returns. Do not divide the cross-stock mean Close by a cross-stock mean Open.
13. Describe the predicted-variance-based red line as a forecast-conditioned single-draw simulation: the variance forecast controls dispersion, while the sampled shock determines one realized scenario. It is not a unique directional point forecast.
14. Add a global gross-return plot using the fitted MLP-80 Global variance model. Match the existing global Base LSTM-80 plot's eligible ticker-dates, adjusted Opens, drift, and shocks; exclude WHLR as in the other global displays. This changes only the simulated line's variance forecast.

## Plot and thesis work

- Add `data/plot_price_reconstruction.py` and save seven PNGs and seven date-level CSVs under `imgs/data_properties/`.
- Add displayed definitions and seven figure descriptions in Section 6.6. State axes, populations, models, and line meanings without interpretation, and document the WHLR diagnostic exclusion.
- Record the observed exclusion counts and next task in `CONTEXT.md`.

## Verification

- Check a small two-ticker example for formula, preceding-only 252-valid-observation windows, ticker boundaries, invalid-price rejection, key alignment, shared shocks, and equal plotted populations.
- Run the script on saved artifacts; reconcile each CSV count with forecast and exclusion counts.
- Check `log(mean Close)` against a known two-stock example and confirm that the two log CSVs retain the price-level date counts.
- Check that mean ticker-level Close/Open differs from mean Close divided by mean Open on a two-stock example, and that the gross-return CSV counts match the price plots.
- Match every date and stock count in the new global MLP-80 gross-return CSV to the global Base LSTM-80 gross-return CSV; confirm their actual lines match and their simulated lines use the same shocks.
- Compile `thesis_structure.tex` twice in place with `pdflatex -synctex=1`.
- Obtain fresh-context cold review, validate its JSON, and resolve findings.

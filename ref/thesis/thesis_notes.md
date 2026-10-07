# THESIS NOTES 

## Realized utility setup

- Sourced fact: Zhang et al. (2022), Section 5.4, Equation (21), derive frictionless realized utility per unit wealth from a shared Sharpe ratio, risk aversion, realized volatility, and model-expected volatility. Their notation exponentiates log realized volatility; this project substitutes positive daily adjusted GK variance directly into the realized-to-predicted ratio. [Paper](https://arxiv.org/pdf/2202.08962)
- Sourced input: YCharts reports 4.26% as the long-term average 10-year Treasury rate. [YCharts](https://ycharts.com/indicators/10_year_treasury_rate)
- Project decisions: use the 4.26% yield only as a fixed annual proxy, convert it to an effective daily rate with 252 trading days, set risk aversion to 2, and compute one retrospective test-sample Sharpe ratio from five equal-weighted ticker means. The test returns were unavailable at forecast time; no observed cash return or implementable trading strategy is claimed. RU excludes transaction costs.


- Tukey (1977) gives the boxplot's 1.5-times-fourth-spread inner-fence convention. The thesis residual-moment table applies the same multiplier to linearly interpolated quartiles of ten displayed model-level values per metric and transform; these are descriptive flags, not a normality test. The 0.7% normal-population tail rate is a mathematical consequence of the rule, not a documented reason Tukey chose 1.5. [Book record](https://search.worldcat.org/title/3058187); [Walker and Chakraborti (2013)](https://ww2.amstat.org/meetings/proceedings/2013/data/assets/handouts/310115_83897.pdf)
- NIST's Engineering Statistics Handbook defines skewness as asymmetry (negative: longer left tail; positive: longer right tail) and excess kurtosis relative to the normal distribution's zero reference (positive: heavier tails; negative: lighter tails). [NIST Section 1.3.5.11](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)
- WHO/UNICEF (2019, p. 71) offers descriptive screening rules of thumb: skewness outside [-0.5, 0.5] and excess kurtosis outside [-1, 1]. These bands screen magnitude; the sign of excess kurtosis distinguishes lighter from heavier tails even within [-1, 1]. The guidance was developed for anthropometric survey distributions, so the thesis uses the bands only as illustrative guides, not validated normality cutoffs for financial variances or residuals. [WHO/UNICEF report](https://iris.who.int/bitstream/handle/10665/324791/9789241515559-eng.pdf)
- Westfall (2014) shows that excess kurtosis does not determine peak height: heavier tails may accompany a more concentrated center, but distributions with high kurtosis can also have lower peaks. Interpret the statistic through tail extremity and inspect the histogram for central shape. [Publisher article](https://doi.org/10.1080/00031305.2014.917055)
- Hyndman and Koehler (2006), [Another look at measures of forecast accuracy](https://robjhyndman.com/papers/mase.pdf): MASE divides each absolute forecast error by the training-history mean absolute one-step naive change for that series. Here, use each ticker's selected variance estimator history and reject a zero or undefined scale. MAE, MSE, and RMSE remain in that estimator's variance units.
- Patton (2011), [Volatility forecast comparison using imperfect volatility proxies](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf): QLIKE is a robust forecast comparison loss under the paper's assumptions, including conditional unbiasedness of the proxy for latent variance. This does not establish that every project estimator satisfies those assumptions.
- Hansen, Lunde, and Nason (2011), [The model confidence set](https://doi.org/10.3982/ECTA5771): Equation (1) tests zero expected pairwise loss differences for every pair in the current model set. Their unnumbered Section 3.1.2 $T_{\max}$ statistic takes the largest standardized mean excess loss relative to the current-set average; its matching rule removes that model after a bootstrap rejection and repeats until non-rejection. Under the paper's assumptions, the resulting set asymptotically contains all minimum-expected-loss candidates with probability at least $1-\alpha$; finite samples can retain inferior models. Apply separately by variance proxy, horizon, and loss on identical forecast observations, with a dependence-aware bootstrap design specified before evaluation.
- What we are doing is trying to unveil the structure behind the pricing or volatility and predict longer term ranges. What HFT does is basically work on it as it is to pursue imbalances on physical queues in bid and asks. 
- We don't really act. 
- We are trying to find the best structural model, not play with the effects of the structure. 
## NOTES - 1.1 - *HARd to beat: The overlooked impact of rolling windows in the era of machine learning* [https://arxiv.org/abs/2406.08041]
- Basically, the data is already defined. 
- The models that we are going to compare need to be listed as well. 
- So far HAR and LSTM. 
- Ensure we have an appendix with the range of hyperparameters. 
- Need to define what we are estimating, the range High - Low/Open need to have the Q-LIKE loss then, since it properly adjusts for real-case scenarios. It's better to overprotect than to under. 
- Use the utility framework to compare with Sharpe rations and proportionality to the wealth to pay for the model. 
- Maybe add VIX as another feature since it's a direct comparison with S&P500. 
- Continue to use MSE even tho Q-like is more comprehensive. 
---
## NOTES - 1.2 - *Global neural networks and the data scaling effect in financial time series forecasting* [https://arxiv.org/abs/2309.02072]
- I need to define the objectives of our paper. First is comparison between models. What else? 
- I could add digital coins to check if the global assumptiosn regarding stocks are translated to digital coins. 
- Use different amounts of tickers to make the prediction. 
- Global training is different than we are doing here. We are throwing away the chnological order. 
- The approach is interesting since they demean the returns since the returns mean are equal to zero and estimate the variance easily. 
- Need to investigate if the approach of training the model with each ticker being a row is the optimal, what is the objective we are trying to achieve and if training each series individually in the same model would make it better. 
- ES and VAR metrics according to Basel methodology. They use Q-LOSS and Joint Loss 
- We have small data. <1000 tickers. 
- They used a rolling window of 2 years as a baseline and used a TI metric to calculate the temporal importance. 
- Stock-dependency of LSTM 
- NN uncover data pattern from rich data environments. May need to append more data to the data doing two downloads, with a interval. 
- News Impact Curve (NIC) simulation (Engle & Ng, 1993; Liu et al., 2023 Section 4.3.1): The saved tables evaluate all 10 MCS intersection models using return shocks s in {-5, ..., +5} percentage points (s = 100 * log(Close/Open), r = s/100 in [-0.05, +0.05]); Figures 6.25 and 6.26 plot the eight non-RFSV variants. Figure 6.25 uses the saved global pre-2016 variance floor 3.396062419686545e-13 as a shared reference, with log-volatility input 0.5 * log(floor) = -14.35549. Figure 6.26 uses the arithmetic mean of all 12,232,393 valid pre-2016 adjusted GK variance observations in the same global cohort, replacing zero GK estimates by the floor as training does: 9.9822850495058469e-4, with log-volatility input -3.45476417. The local fits retain their own forecast floors, but each NIC input is common to all plotted variants. The floor and mean references are project choices, not prescribed by Liu et al.; the latter vary a return-only zero history.
---
## NOTES - 1.3 - *HARNet: A convolutional neural network for realized volatility forecasting* [https://arxiv.org/abs/2205.07719]
- Corsi (2009), already listed in `references.md`, uses daily, weekly, and monthly realized-volatility components to reproduce persistent volatility dynamics in a parsimonious HAR model. Persistence of high and low volatility is the clustering pattern; it does not imply that signed returns are predictable or that volatility cannot eventually revert.
- Section 2, equations (1)–(3), treats spot volatility as latent, defines daily integrated variance, and gives the intraday squared-return realized-variance proxy. Its consistency statement cites Andersen and Bollerslev (1998) and assumes the sampling interval shrinks within a fixed day.
- Implementation detail: nested (1,5,20) receptive fields use a 5-step averaging convolution followed by a 4-step convolution with dilation 5. Averaging-filter initialization and fitted HAR output coefficients reproduce HAR on nonnegative histories.
- Methods cross-check (2026-10-01): [Section 4.1 of the original HARNet paper](https://arxiv.org/html/2205.07719) motivates multiplicatively nested horizons. The project's single-channel, bias-free filters have lengths (5, 4, 2, 2) and dilations (1, 5, 20, 40), yielding receptive fields 5, 20, 40, 80. HARNet-20 uses the first two layers; HARNet-80 uses all four. These describe the repository architecture, not every architecture evaluated in the paper.
- HARnet perform the same as HAR with its initialization and even better when optimized. 
- Uses dilated convolution filters. Allows to expand the receptive field without increasing the parameters. Allows to increase exponentially the horizon, while linearly increasing the depth. 
- Each layer of HARnet computes features with respect to different time horizons, like HAR. 
- Used OLS and WLS to fit, either for homoscedacity or heteroscedacity. 
- All of the models are defining log price dynamics with the same stochastic differential equation: dp(t) = µ(t)dt + σ(t)dW(t). 
- They chose the weight of WLS as 1/OLS. They used also log transformation to account for heteroscedacity. 
- Interesting to see if can initialize a model using a base statistical model like they did
---
## NOTES - 1.4 - *Rough volatility: Evidence from range volatility estimators* [https://arxiv.org/abs/2312.01426]
- A more statistical approach. Nice to give a comparison between traditional models like ARIMA which assumed that models were driven by WN, which is N(0, stddev) when modern assumption is fBm(fractional Brownian Motion) N(O, stddev^H). 
- A RFSV (Rough Fractional Stochastic Volatility) model assume a fOU motion where the mean reversion is added like in a standard OU process, however, W gets substituted by W^H. 
- Parkinson estimator is only asymptotically unbiased under assumption of Geometric Brownian Motion. 
- Garman-Kass is optimal in mean-variance sense among a certain class of estimator. Can use the reduced form of it. 
- Could show that our data follow a fBm by two ways: either we run the model in parallel for each timeseries and then update the weights or we do as we are already doing. Could do both as well. 
- Final objective: compare RFSV model to Deep Neural Networks. 
---
## NOTES - 4.1 - *Forecasting realized volatility with time series foundation models: A comparison with econometric benchmarks* [https://arxiv.org/abs/2607.05291]
- Using TSFM to predict. Use them to compare. 
- Paper from July 2026. Recent. 
- I'll need to compare the realized volatility with the squared intraday returns with the estimators to prove our point. 
- As small as you observed the intraday returns, more precision you have about the real volatility. 
- Refer to the HAR variations but doens't use them as comparisons. 
- If I'm using TSFM to compare I can use the latest one of Google or I can also talk about the different models. The focus should be a overall view and focusing on the global structure. 
- All of this TSFM are zeroshot. 
- Use the formal model testing. 
- Could use the vol and estimators as a separate section to compare how to performance of the models change. However, it would contain smaller data. 
- Constrain on the least squares so that the parameters doesn't turn negative. 
- Bounding Q-like over min RV and max RV to avoid distortions in loss. Q-LIKE diverges as forecast approaches zero. 
- MZ regression to assess forecast efficiency. 
---
## NOTES - 4.2 - *Volatility-inspired σ-LSTM cell* [https://arxiv.org/abs/2205.07022]
- Checked the paper itself: Equations (4)-(7) use return input, sigmoid forget/input gates, and a tanh candidate; Equation (8) draws a zero-mean Gaussian output gate whose variance depends on squared cell memory; Equation (9) multiplies that draw by an unspecified φ(C); Equations (10)-(11) output a linear return estimate and variance equal to squared mean cell memory. `models/sigma-lstm.py` inherits the existing `FEBCellLSTM` and `FEBLSTM` classes but adapts their input/head to GK log volatility. Its current softplus gate-variance map, tanh φ, and C(0) = 1 are project choices, not specified by the paper. The auxiliary squared-mean cell-state output is only interpreted as log-volatility variance here; this does not establish calibration. The paper's Equation (12) return likelihood is not implemented because the project trains with GK-variance QLIKE. See [sigma-LSTM decisions](../implementation_plan/sigma-LSTM.md).
- The paper recommends min-max input scaling, but the user chose a global training-only z-score of GK log-volatility inputs for this project. The log-volatility target remains unscaled and QLIKE remains in variance units. This is a project adaptation, not the paper's return-prediction target.
- Could use directly the vol-LSTM. Or use both. 
- Standardization before training RNNs 
- The benefits were low, however, were better than the LSTM itself. So good to know. 
---
## NOTES - 5.2 - *Forecasting volatility with machine learning and rough volatility: Example from the crypto-winter* [https://arxiv.org/abs/2311.04727]
- Gatheral, Jaisson, and Rosenbaum (2018), Section 3.1, specify stationary fOU log volatility with restoring drift toward level m and a deliberately long reversion time scale. They distinguish this long-run mechanism from the short-scale rough behavior at H < 1/2; local reversal after an extreme observation alone does not estimate a stable long-run mean.
- Checked Section 3.2 (its LSTM cell equations are unnumbered): sigmoid input, forget, and output gates; SiLU candidate and SiLU cell output. `models/silu_lstm.py` matches these cell operations. The paper's return-input LSTM instead uses one recurrent layer of hidden width 4, windows of 7 or 30 observations, training-set per-coin scaling, MSE on volatility, and an average of 10 seeded fits. This project's width-128, window-20, raw-history log-volatility/return, variance-QLIKE run is a separate experiment; the paper's results do not establish stability for that configuration.
- Implementation detail: the SiLU-LSTM retains sigmoid gates and uses SiLU for the candidate and cell output. In [Gatheral, Jaisson, and Rosenbaum, Sections 3.1 and 5](https://arxiv.org/pdf/1410.3394), Equations (3.3)-(3.4) define stationary fOU log volatility and `sigma_t = exp(X_t)`. Under small mean reversion, Section 5 derives the simpler Equation (5.1) log-variance forecast, whose equivalent `u` form is unnumbered and starts at zero. The unnumbered Section 5.2 variance equation adds `2*c*nu^2*Delta^(2*H)` before exponentiation. The repository uses the Section 5 approximation with estimator-specific pre-2016 observation-lag H and ν², 20 observed forecast values, and oldest-value tail extension. It does not fit the full stationary model; see [the equation review](../implementation_plan/rfsv_equation_review.md).
- Methods cross-check (2026-10-01): the preceding 20-value description records the earlier variant. The current `RFSV.forward()` and full-history fit use all preceding valid positive within-ticker variance observations, exact observation-bin kernel masses, and oldest-value tail extension. Invalid and zero rows are compressed out of that history, unlike fixed-window eligibility. The paper's Equation (5.1) and unnumbered Section 5.2 correction were checked again against the primary PDF; the proxy substitution and observation-time convention remain project approximations. The existing full-history evaluator scores only the final 2025-12-31 target per ticker, so its saved results cannot be ranked against the full-period windowed rows as a matched comparison.
- Bitcoin volatility is rought based on the Takaishi paper. Could implement the framework of testing roughness to the full stock market dataset. 
- Join crypto and stocks in the same dataset to test universality of the model 
- Using SiLU allows to pass the true volatility, without making the gradient become 0. Maybe using it on the vol-LSTM instead of TanH. Need to study more this approach. 
- LSTMs with return inputs but this is high frequency. 
- Using sensitivity parameters to understand the behaviour of vol based on returns and variance to match with the previous studies. 
- Crypto have an inverted assymetric volatility or inverse lerage effected. 
- No asset-specific features 
- RFSV with QRH with lambda at 0.15 in the linear combinatior gives a similar but much more parsimonious model for the universality. 
- Zumbach effect: time reversal symmetric is broken. A trend can give the impulse to a spike in volatility after liquidation of assets, however, the spike in volatility does not follow a future trend. QRH solves this. Garch doesnt. It needs to have a mathemtical component that captures this effect conditioned by time and direction. 
--- 
## NOTES - 6.1 - *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.* [https://www.jstor.org/stable/1912773?origin=crossref&googleloggedin=true&seq=1]
- Lagrange Multiplier to check the distribution of the errors, if they are homoscedastic or not. 
- Engle's ARCH conditional variance responds to past squared shocks: a large innovation raises subsequent conditional variance. This is one model mechanism for volatility clustering, meaning persistence in return magnitudes rather than return signs.
---
## NOTES - 6.2 - *Generalized autoregressive conditional heteroskedasticity.* [https://d1wqtxts1xzle7.cloudfront.net/62739731/GENERALIZED_AUTOREGRESSIVE_CONDITIONAL20200402-79966-1j3fzc2-libre.pdf?1585986363=&response-content-disposition=inline%3B+filename%3DGENERALIZED_AUTOREGRESSIVE_CONDITIONAL_H.pdf&Expires=1786795899&Signature=YPhsB8xaRfELgAUAAK568pgjShVqysSSdKr48N1e40iMPkdSqRk8Xg~61DUJrKDTJ91TbNfPiQfwU10TVsWJI~ikjmL5ImUhjaQ0vGUJ7lnGreQyjERtJy8exmM7SD~nem4bh1eNCfyuYv-JduX8fGanAGA8crg0tlR-p-BOhIHeT76mGN5PYwqScg115g8tuDRugMmYgmcJYKRivafKOgk7YP7ckCijGa5I6svuvnGslij~Gvzk89otS4lXikYxxfRUMu1OCTMa9QZyKjWCKlfbTdfjkZuH94ZIgZvg5UdeHzU0-uEA4vLP7pU36X173O60TUtf9XneFbOFnTL0Lw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA]
- ACF and PCF work for GARCH AND ARCH as well. 
- GARCH(p, q) can be interpret as a ARMA(m, p) in e^2 of orders m = max(p, q) and p. 
- Bollerslev's GARCH also carries forward lagged conditional variance. For GARCH(1,1), under standardized innovations and α + β < 1, the conditional tendency of a deviation from v̄ = ω/(1 − α − β) is multiplied by α + β each step; a coefficient near one allows slow mean reversion and persistent clustering.
---
## NOTES - Engle and Ng (1993) - *Measuring and testing the impact of news on volatility* [https://doi.org/10.1111/j.1540-6261.1993.tb05127.x]
- The news impact curve compares the next conditional-volatility response to positive and negative return shocks at a fixed background state. In their daily Japanese-stock sample, negative shocks generally raise volatility more than equally sized positive shocks. The authors explicitly caution that this asymmetry, commonly called the leverage effect, does not identify changing financial leverage as its cause.
---
## NOTES - 8.1 - *Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting* [https://arxiv.org/pdf/2008.00444]
- Section 2.1 defines a local algorithm as producing one forecasting function per observed series and a global algorithm as producing one function for the set. Proposition 1 says each can reproduce the other's forecasts for finite observed series and a finite horizon; this is an existence result, not a performance guarantee for a fixed architecture.
- Proposition 2 measures complexity by the cardinality of the available forecasting-function classes: the local class has size equal to the product of the series-specific class sizes, while the global class has its own size. Section 3.4's unnumbered equality of these sizes compares worst-case bounds only under equal in-sample error, independent series, equal effective sample sizes, and bounded loss.
- Section 3.5 illustrates parameter counting for autoregressions with one parameter per lag and approximately 2^64 double-precision values per parameter. Under that illustrative count, matching class sizes makes global parameter count equal the sum of local parameter counts. Parameter count is not the paper's general definition of complexity.
- Section 2.2 notes that identical finite input windows can lead to different local forecasts if full series differ; a global function needs enough history to distinguish such inputs. Correlated equity series should not be assumed to satisfy Proposition 2's independence condition.
- Global and Local models can be equals by proposition one. However, if you have different time series and you want a different answer for each time series, you need more input to differentiate between the two of them. If you have the same input for Xi and Xj and you want a different result, you have to increase the window.
- Generalization Error and metrics. We are assuming Ein to be the sample average while Eout to be the expected. Complexity term + Ein >= Eout 
- How much the expected loss might differ on in-sampel and out-sample data points based on the model and the loss function. 
- We can try to find a Global that produces the same in-sample error as the local and therefore understand the complexity terms and if it's within our budget. If it is, we get better generalization than the local alternative and better performance. 
- We aim to equalize this error to control the complexity, which what we can do before seeing the data. 
- Assuming a SoTA local algo, we can assume that the global approach will have the complexity as the sum of all individual algos in the set. 
- Generalize for other datasets such as FUTURES, INDEX per day. 
- Increase the size of the window inside the LSTM. 
- Each window is independent. 
- Could use opening or close prices with MASE as the error. 
- Local model fits their own data better, but overfit when out-of-sample. When another time-series gets inputed to be predicted. 
- Use the average error of all the time series. 

## NOTES - 8.2 - *Universal features of price formation in financial markets: Perspectives from deep learning. Quantitative Finance* [https://arxiv.org/abs/1803.06917] 
- At the microstructural level, it holds stationarity. The structure behind what makes the price is not altered, being the effect larger or not. Which means that no matter the volatility, what happens behind it is the same. We are moving from volatility prediction to price direction based on the laws of price formation. 
- The author assumes that the laws of price are universal. The microstructures behind it. 
- In theory, shouldnt work with daily data. The non-stationarity effects gets too big and the price doesn't reflect the microstructure behind it. 
- We cannot build a universal network that doesn't respect the chnological order if our assumptions depend on causality, which means that future won't affect the past, even if it's from two different stocks. They mantained the chnological order here. 
- Maybe predicting direction might be interesting. If it's going to fall, you don't put. If it's going up, you go, even if it's small changes, you'll never lose. Maybe you will with the comission. Need to understand this. 

## NOTES - 8.3 - *Volatility forecasting with machine learning and intraday commonality* [https://arxiv.org/abs/2202.08962]
- Commonality, which means, the measurement on how the vol of a single assets changes relative to the vol of the market increases the prediction power of the model. 
- Past volatility also provide additional information for forecasting. 
- Using ARFIMA as well. 
- Review on Pearson and Spearman correlations and their differences. 
- Winsorization to compensate the spikes created by anomalies in models like GARCH. We can explore the assumption for a heavy-tailed distribution or jump-diffusion models. 
- Calculating commonality but using the Parkinson estimator. Commonality is the R^2 of the regression using RVm; Therefore R^2 is the explained variance by the market RV. Commonality. 
- Use the utility equation to bring this back into the real world. Then do a portfolio simulation to find the return over week, month, semester, year, 2 years and 5 years. 
- Increase the data to the maximum we can to compare in fully power. 
- Sensitivy analysis. 







## Implementation: SiLU derivative and stability hypothesis (2026-10-02)

- Architecture citation clarification: Zhang, Zhang, Cucuringu, and Qian, arXiv:2202.08962v2, Appendix B, Table B.2 notes, specify widths 128, 128/64, and 128/64/32 for MLP variants and state that LSTM variants have similar meanings. This supports a 128-unit first LSTM layer; it is not a SiLU-LSTM architecture attribution. Source: https://arxiv.org/html/2202.08962#A2. The project's selected width and learning rate remain its own comparison decisions, as reported by the user.

- Source: Elfwing, Uchibe, and Doya, arXiv:1702.03118v3, Section 2.2, printed Equation (10), https://arxiv.org/pdf/1702.03118. In the paper's notation, aₖ(s) = σ(zₖ)(1 + zₖ(1 − σ(zₖ))) is dSiLU, the derivative of SiLU. Its large-positive-input limit is 1; tanh's derivative approaches zero.
- Project hypothesis: repeated recurrent transformations can preserve sensitivity and can amplify it depending on learned weights and gates. The source does not establish the cause of this project's instability. Sigmoid gates remain unchanged; only candidate and cell-output activations switch to SiLU.
- Project decision: selected SiLU-LSTM runs use variance plus within-session returns, direct variance outputs, and raw-output MSE training. MLP/LSTM use log-volatility plus returns, log-variance outputs, and QLIKE. HARNet uses variance and QLIKE. Input transformation and output parameterization also changed in the successful SiLU configuration, precluding a loss-only causal attribution.
- Verification limit: checkpoint files are unavailable locally. Selected configurations were checked against evaluation selectors, local launchers, the LSTM-20 continuation launcher, fitting code, and saved evaluation reports. The `30e` directory suffix is historical; the continuation caps the run at 20 epochs.
- Selected evaluation uses full preceding filtering history for SARIMA/GARCH with 20-observation target eligibility, unlike a truncated 20-input forecast call. RFSV final-date scoring and matched-observation limitations belong in Experimental Results; no evaluation or data behavior changed here.

## MCS framework (2026-10-03)

- Source: Hansen, Lunde, and Nason (2011), Section 3.1.2, as cited in `ref/thesis/references.md`: T_max tests the maximum standardized mean excess loss and eliminates its maximizing candidate after rejection. Brini (2026), Section 4.4, as cited there, uses QLIKE, a moving-block bootstrap, 10,000 repetitions, and a 90% confidence set per asset/horizon.
- Project decisions in `ref/implementation_plan/mcs_implementation.md`: use 20 consecutive retained forecast observations per non-circular block, seed 42, and the finite-bootstrap p-value correction. The framework in `models/mcs_definition.py` accepts aligned single-asset QLIKE losses; forecast extraction, key checks, per-stock streams, and project-data evaluation remain pending.
## Price reconstruction diagnostic (2026-10-06)

- Source fact: WHLR's 2024 Form 10-K reports a one-for-10 reverse split in August 2023 and several later reverse splits. This supports treating its large stored historical adjusted price levels with care; it does not prove that any particular vendor adjustment is erroneous. See `ref/thesis/references.md`.
- Project-data finding: WHLR's stored adjusted Close exceeds USD 10^11 on 2022-08-03. It contributes about 99.60% of the plotted global arithmetic mean that day, 99.69% on 2022-11-28, and 99.81% on 2023-08-08. Zero-GK days on 2022-08-03, 2022-11-30, and later dates cause its 80-observation model eligibility to switch off and later on. The sharp level jumps in the original global price figure reflect this changing cross-section, not an equity-wide price crash or recovery.
- Project decision: omit WHLR's 1,510 otherwise eligible ticker-dates only from Section 6.6 global price plots. Keep fitted models, saved forecast evaluation, and the five-stock plot unchanged. For the added log views, transform the arithmetic daily mean with `log(mean Close / USD)`, preserving its date population; do not average ticker-level log prices.

## XXII price-mean diagnostic (2026-10-07)

- Source fact: XXII's documented reverse splits from July 2023 through June 2026 have a cumulative 1-for-223,560,000 ratio. These later corporate actions explain the scale of earlier split-adjusted per-share prices; they do not establish an as-traded 2021 price of over USD 1 billion. Sources are in `ref/thesis/references.md`.
- Project-data finding: on 2021-04-28, XXII's stored adjusted Close is USD 1,256,407,168, equivalent to approximately USD 5.62 before those later splits. Its contribution is 87.91% of the 2,470-stock plotted mean on that date. Across all 1,760 plotted dates, its contribution is 64.13% of the time-averaged daily mean. Removing its one eligible row per date changes the mean of the plotted daily means from USD 174,046.43 to USD 62,465.64.
- Project decision: omit XXII's 1,760 test ticker-dates from the global adjusted Close mean and its log-of-mean view, alongside WHLR. Both views now contain 4,292,159 ticker-dates on the same 1,760 dates. Training, validation, model fits, scored test evaluations, and the global gross-return views retain XXII. Other extreme adjusted price levels still influence the remaining price mean.

## Eight-stock global price display (2026-10-07)

- Source fact: Yahoo Finance applies reverse-split multipliers to prices before the split. A 1-for-100 reverse split therefore displays an earlier USD 0.50 Close as about USD 50 on the new share basis; repeated splits multiply the historical per-share level. SEC filings document reverse splits for NUWE, ZNB, PPCB, JAGX, XTIA, and CETX; see `ref/thesis/references.md`. This is a standard retrospective adjustment, not evidence of a vendor error or an as-traded historical price.
- Project decision: exclude those six stocks alongside WHLR and XXII only from the descriptive global price and log-of-mean Close figures. Preserve the saved variance models, all evaluation populations, the five-stock plots, and the global gross-return figures.
- Project-data finding: the six newly excluded stocks contribute 8,992 eligible stock-dates. Figure 6.30 now averages 4,283,167 stock-dates over the same 1,760 dates; its observed time-average daily arithmetic price mean falls from USD 62,465.64 to USD 2,299.76 before taking the logarithm. The changed sample makes this descriptive line incomparable in level with the previous version of Figure 6.30.

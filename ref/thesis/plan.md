# NAME: Volatility Forecasting: A statistical and Deep Learning approach 

## QUESTIONS
1. What are the changes you'd do to this plan? Is this good enough for a thesis? Is there any paper you'd like me to read? 
2. Can we define the cronogram in the following way: I finish this implementation in a week or week and a half. I write it it in a week? And then final revision? I need to deliver by 09 of October and send the request of laurea on the 20th of September, where you'd need to sign it. 
3. Is the name ok or do you have a suggestion? 
4. How will the VVSI project be graded? I'd need it graded and registered until the 20th of September since I need to send my request. 

## NOTES 
1. Adjust the MAPE. 
2. Table of losses. 
3. In the validation/testing, use the network to compute for a single ticket. Make the distribution of the error. 
4. Use the full data. 
5. Separate the code for VVSI. 


## OBJECTIVE
1. Provide a comprehensive comparison between statistical/econometric and ML (Machine Learning) models in order to determine which is more efficient in predicting Realized Volatility (RV), as a proxy of Implied Volatility (IV), as a step to build a model in order to optimize a portfolio construction. [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)
2. Understand the underlying mathematical mechanism of our data, regarding the price itself and the data roughness. Explore fractional Brownian Motion (fBm) and fractional Ornstein-Uhlenbeck (fOU) analysis and estimation of the Hurst exponent. [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426) [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727)
3. Explore the universality of the models, exploring the concept of complexity between local and global models and how DNN (Deep Neural Network) can leverage high data quantity for generalization. [8.1](thesis_notes.md#notes---81---principles-and-algorithms-for-forecasting-groups-of-time-series-locality-and-globality-international-journal-of-forecasting-httpsrobjhyndmancompapersglobal-modelspdf) [8.2](thesis_notes.md#notes---82---universal-features-of-price-formation-in-financial-markets-perspectives-from-deep-learning-quantitative-finance-httpsarxivorgabs180306917) [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072)

## STEPS
### Build the dataset:
- Explore the optimum quantity regarding data points. We need the dataset as big as possible. [1.2](thesis_notes. md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962). Ok 
- Fetch all data from Yfinance using a scheduler to avoid rate limits. [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072). Ok
- I've access to intraday data from crypto. We can use it to test universality on unseen data. If it generalizes, we can compare estimators. [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727) [8.2](thesis_notes.md#notes---82---universal-features-of-price-formation-in-financial-markets-perspectives-from-deep-learning-quantitative-finance-httpsarxivorgabs180306917). Ok
- Use Parkinson estimator as a RV proxy instead of intraday squared returns since we don't have this data available. It was proved that is asymptotically unbiased under Geometric Brownian Motion. If needed, explore the bias statement based on the data. [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)
- Use Garman-Klass since it's optimal in mean-variance. Use the reduced form of it. [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426)
- Define the volatility proxies so GK, PK for tickets. GK, PK, RV for crypto. Ok
- Plot the dataset and identify visual characteristics. Ok
- Perform a statistical analysis to understand the dataset characteristics, locally and globally. Data roughness analysis. [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426) [8.1](thesis_notes.md#notes---81---principles-and-algorithms-for-forecasting-groups-of-time-series-locality-and-globality-international-journal-of-forecasting-httpsrobjhyndmancompapersglobal-modelspdf). We will do the same game, we will display log mean displacement over loq value with q = 1 q =1.5 q = 2 q = 3 and q = 4. We would then find the slope of the zeta sub q x q to find the H regarding our data. Globally. And then locally, but locally, we would need to understand which individual elements we might want to have a look.  Ok.
- Features: RV, Commonality, VIX (CBOE Volatility Index), log(volume) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962) [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041). Deferred
- Training: < 2016-01-01; Validation: 2016-01-01 -> 2019-01-01; Testing: >= 2019-01-01 [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041). Ok.
--- 
### Define the models
- SARIMA (Seasonal AutoRegressive Integrated Moving Average) [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426)
- HAR (Heterogeneous AutoRegressive) [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041) [1.3](thesis_notes.md#notes---13---harnet-a-convolutional-neural-network-for-realized-volatility-forecasting-httpsarxivorgabs220507719) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)
- HARNet (Heterogeneous AutoRegressive Network) [1.3](thesis_notes.md#notes---13---harnet-a-convolutional-neural-network-for-realized-volatility-forecasting-httpsarxivorgabs220507719)
- GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) [6.1](thesis_notes.md#notes---61---autoregressive-conditional-heteroscedasticity-with-estimates-of-the-variance-of-united-kingdom-inflation-httpswwwjstororgstable1912773origincrossrefgoogleloggedintrueseq1) [6.2](thesis_notes.md#notes---62---generalized-autoregressive-conditional-heteroskedasticity-httpsd1wqtxts1xzle7cloudfrontnet62739731generalizedautoregressiveconditional20200402-79966-1j3fzc2-librepdf1585986363response-content-dispositioninline3bfilename3dgeneralizedautoregressiveconditionalhpdfexpires1786795899signatureyphsb8xarfelgauaak568pgjshvqysssdkr48n1e40impkdsqrk8xg61dujrkdtj91tbnfpiqfwu10tvswjiikjml5imuhjaq0vguj7lngreqyjertjy8exmm7sdnem4bh1encfyuyv-jdux8fganaga8crg0tlr-p-bohihet76mgn5pywqscg115g8tudrugmmygmcjykrivafkogk7yp7ckcijga5i6svuvngslijgvzk89ots4lxikyxxfrumu1octma9qzykjwcklfbtdfjkzuh94zigzvg5udehzu0-uea4vlp7pu36x173o60tutf9xnefbofntl0lwkey-pair-idapkajlohf5ggslrbv4za)
- AR(1) (one-lag autoregressive baseline, adapted to variance) [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727)
- MLP (Multi-Layer Perceptron) [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072)
- LSTM (Long Short-Term Memory) [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041) [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072) [4.2](thesis_notes.md#notes---42---volatility-inspired-σ-lstm-cell-httpsarxivorgabs220507022)
- vol-LSTM (volatility-inspired LSTM) [4.2](thesis_notes.md#notes---42---volatility-inspired-σ-lstm-cell-httpsarxivorgabs220507022)
- SiLU-LSTM (using SiLU (Sigmoid Linear Unit) as activation function) [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727)
- RFSV (Rough Fractional Stochastic Volatility) [1.4](thesis_notes.md#notes---14---rough-volatility-evidence-from-range-volatility-estimators-httpsarxivorgabs231201426) [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727)
- TSFMs (Time Series Foundation Models: TimesFM (Time Series Foundation Model, Google) and TinyTimeMixers (TTM)) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)
- Fine-tuning a SLM (Small Language Model) (*Not priority, but can be interesting for checking purposes*) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)

#### Model definitions and scope

All active models forecast the next period using history from one ticker. In the crypto transfer test, equity-fitted parameters stay fixed while the input ticker history changes. The exact variance estimator, forecast horizon, units, and transformation must be fixed before fitting or comparing them. A model already trained on overnight returns is an architectural starting point, not a completed volatility forecast.

| Model | Definition for this project | Status |
| --- | --- | --- |
| AR(1) | Intercept plus supplied coefficient times the latest variance; fit coefficients on equity data later. | Defined |
| HAR | Intercept plus separate latest variance, latest-five mean, and latest-twenty mean terms; fit by least squares. | Defined |
| SARIMA | SARIMAX with initial orders (1,0,1) and seasonal (1,0,1,5); fit coefficients on equity history, then filter a ticker's history and forecast with fixed coefficients. Choose orders later from chronological validation, information criteria, and residual checks. | Defined |
| GARCH | GARCH(1,1) conditional variance from supplied ω, α, β, return residuals, and initial variance; return volatility as the square root. Assess fit and residuals later. | Defined |
| RFSV | Forecast from past log volatility using a supplied H and the cited rough kernel. The current full-period equity Garman–Klass H estimate is about 0.03498; select a valid out-of-sample H later. | Defined |
| MLP | PyTorch network with configurable width and two SiLU hidden layers. | Defined |
| LSTM | Existing custom `FEBLSTM` PyTorch cell and sequence architecture; retain old return checkpoints and retrain later for volatility. | Defined; volatility training pending |
| SiLU-LSTM | Same custom PyTorch sequence contract and sigmoid gates, with SiLU replacing both tanh candidate and cell-output operations. | Defined |
| HARNet | Hierarchical causal convolutional 1/5/20-observation features, initialized from fitted HAR coefficients to reproduce HAR on nonnegative inputs. | Defined |
| vol-LSTM | Return-oriented variant outside the current volatility-target comparison. | Deferred |
| TSFMs: TimesFM, TinyTimeMixers | External foundation-model comparison, to be scoped after independent review. | Deferred for separate review |
| Fine-tuned SLM | Exploratory language-model comparison. | Deferred |

Each active model has one file in `models/`; VVSI training and experiment entry points live in `inference/`. Select neural widths and window lengths from later equity validation results. Dataset wiring, fitting runs, neural training, and evaluation remain pending. See [the implementation plan](../implementation_plan/models.md).

### Define the losses 
1. QLIKE (Quasi-Likelihood) (Penalizes underestimations more than overestimation, good correspondence for real-life scenarios) [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041) [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072)
2. MSE (Mean Squared Error) [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041)
3. MASE (Mean Absolute Scaled Error) (Standardized) [8.1](thesis_notes.md#notes---81---principles-and-algorithms-for-forecasting-groups-of-time-series-locality-and-globality-international-journal-of-forecasting-httpsrobjhyndmancompapersglobal-modelspdf)
- Ensure the losses are bounded. We are going to winsorize them. [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962)
- We could expand to use the Joint-Loss based on ES (Expected Shortfall) and VaR (Value at Risk). [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072)

### Train & Compare the models 
- Model testing with MCS (Model Confidence Set) [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)
- Comparison table with QLIKE, MSE and MASE and checking who is SoTA (State of the Art). Comparing models trained Locally, Globally and using RV-only, RV-Commonality and RV-Commonality-VIX [8.1](thesis_notes.md#notes---81---principles-and-algorithms-for-forecasting-groups-of-time-series-locality-and-globality-international-journal-of-forecasting-httpsrobjhyndmancompapersglobal-modelspdf) [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962)
- Utility comparison table, calculating Sharpe Ratio and estimating risk aversion, to estimate RU (Realized Utility). The comparison rows would be the same, would only change the Column to be RU. [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962)

### Appendix
- Hyperparameters, table with the considerated hyperparameters and the Q-LIKE loss for each of them. [1.1](thesis_notes.md#notes---11---hard-to-beat-the-overlooked-impact-of-rolling-windows-in-the-era-of-machine-learning-httpsarxivorgabs240608041)
- Sensitivy of model predictors to understand which lag was more significant. [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727) [8.3](thesis_notes.md#notes---83---volatility-forecasting-with-machine-learning-and-intraday-commonality-httpsarxivorgabs220208962)
- We want to identify the mean-reversing trend and the leverage effect (if we are using crypto, we would need to extend this for them as well). We can identify this visually or we can check if models who doesn't have that considered can recover these facts based only on the data. Using NIC (News Impact Curve). [1.2](thesis_notes.md#notes---12---global-neural-networks-and-the-data-scaling-effect-in-financial-time-series-forecasting-httpsarxivorgabs230902072) [5.2](thesis_notes.md#notes---52---forecasting-volatility-with-machine-learning-and-rough-volatility-example-from-the-crypto-winter-httpsarxivorgabs231104727)
- MZ (Mincer-Zarnowitz) regression to assess forecast efficiency. [4.1](thesis_notes.md#notes---41---forecasting-realized-volatility-with-time-series-foundation-models-a-comparison-with-econometric-benchmarks-httpsarxivorgabs260705291)

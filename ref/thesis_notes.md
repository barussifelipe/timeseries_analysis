# THESIS NOTES 

## NOTES - 1.1 - *HARd to beat: The overlooked impact of rolling windows* [https://arxiv.org/abs/2406.08041]
in the era of machine learning*
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
- Using NIC and the way it simulated to check if our model is following the leverage effect. Could check for mean reversion and volatility clustering. 
---
## NOTES - 1.3 - *HARNet: A convolutional neural network for realized volatility forecasting* [https://arxiv.org/abs/2205.07719]
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
## NOTES - 6.1 - *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.* [https://www.jstor.org/stable/1912773?origin=crossref&googleloggedin=true&seq=1]
- Lagrange Multiplier to check the distribution of the errors, if they are homoscedastic or not. 
---
## NOTES - 6.2 - *Generalized autoregressive conditional heteroskedasticity.* [https://d1wqtxts1xzle7.cloudfront.net/62739731/GENERALIZED_AUTOREGRESSIVE_CONDITIONAL20200402-79966-1j3fzc2-libre.pdf?1585986363=&response-content-disposition=inline%3B+filename%3DGENERALIZED_AUTOREGRESSIVE_CONDITIONAL_H.pdf&Expires=1786795899&Signature=YPhsB8xaRfELgAUAAK568pgjShVqysSSdKr48N1e40iMPkdSqRk8Xg~61DUJrKDTJ91TbNfPiQfwU10TVsWJI~ikjmL5ImUhjaQ0vGUJ7lnGreQyjERtJy8exmM7SD~nem4bh1eNCfyuYv-JduX8fGanAGA8crg0tlR-p-BOhIHeT76mGN5PYwqScg115g8tuDRugMmYgmcJYKRivafKOgk7YP7ckCijGa5I6svuvnGslij~Gvzk89otS4lXikYxxfRUMu1OCTMa9QZyKjWCKlfbTdfjkZuH94ZIgZvg5UdeHzU0-uEA4vLP7pU36X173O60TUtf9XneFbOFnTL0Lw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA]
- ACF and PCF work for GARCH AND ARCH as well. 
- GARCH(p, q) can be interpret as a ARMA(m, p) in e^2 of orders m = max(p, q) and p. 








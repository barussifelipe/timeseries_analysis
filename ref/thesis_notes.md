# THESIS NOTES 

## NOTES - 1.1
Basically, the data is already defined. 
The models that we are going to compare need to be listed as well. 
So far HAR and LSTM. 
Ensure we have an appendix with the range of hyperparameters. 
Need to define what we are estimating, the range High - Low/Open need to have the Q-LIKE loss then, since it properly adjusts for real-case scenarios. It's better to overprotect than to under. 
Use the utility framework to compare with Sharpe rations and proportionality to the wealth to pay for the model. 
Maybe add VIX as another feature since it's a direct comparison with S&P500. 
Continue to use MSE even tho Q-like is more comprehensive. 

## NOTES - 1.2
I need to define the objectives of our paper. First is comparison between models. What else? 
I could add digital coins to check if the global assumptiosn regarding stocks are translated to digital coins. 
Use different amounts of tickers to make the prediction. 
Global training is different than we are doing here. We are throwing away the chnological order. 
The approach is interesting since they demean the returns since the returns mean are equal to zero and estimate the variance easily. 
Need to investigate if the approach of training the model with each ticker being a row is the optimal, what is the objective we are trying to achieve and if training each series individually in the same model would make it better. 
ES and VAR metrics according to Basel methodology. They use Q-LOSS and Joint Loss 
We have small data. <1000 tickers. 
They used a rolling window of 2 years as a baseline and used a TI metric to calculate the temporal importance. 
Stock-dependency of LSTM 
NN uncover data pattern from rich data environments. May need to append more data to the data doing two downloads, with a interval. 
Using NIC and the way it simulated to check if our model is following the leverage effect. Could check for mean reversion and volatility clustering. 


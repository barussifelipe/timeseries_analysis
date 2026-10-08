# References

Citation style: APA 7th edition, the convention used in the financial econometrics and
ML-for-finance literature. Entries are grouped thematically for working purposes; flatten
to a single alphabetical list when exporting to the thesis bibliography.

---

## 1. Panel / cross-sectional volatility forecasting (closest to this thesis)

Bollerslev, T., Hood, B., Huss, J., & Pedersen, L. H. (2018). Risk everywhere:
Modeling and managing volatility. *The Review of Financial Studies, 31*(7),
2729--2773. https://doi.org/10.1093/rfs/hhy041

Audrino, F., & Chassot, J. (2024). *HARd to beat: The overlooked impact of rolling windows
in the era of machine learning* (arXiv:2406.08041). arXiv.
https://arxiv.org/abs/2406.08041

Liu, C., Tran, M.-N., Wang, C., Gerlach, R., & Kohn, R. (2023). *Global neural networks and
the data scaling effect in financial time series forecasting* (arXiv:2309.02072). arXiv.
https://arxiv.org/abs/2309.02072

Reisenhofer, R., Bayer, X., & Hautsch, N. (2022). *HARNet: A convolutional neural network
for realized volatility forecasting* (arXiv:2205.07719). arXiv.
https://arxiv.org/abs/2205.07719

Mouti, S. (2023). *Rough volatility: Evidence from range volatility estimators*
(arXiv:2312.01426). arXiv. https://arxiv.org/abs/2312.01426


## 2. GARCH–neural network hybrids

Xu, Z., Liechty, J., Benthall, S., Skar-Gislinge, N., & McComb, C. (2024). *GARCH-informed
neural networks for volatility prediction in financial markets* (arXiv:2410.00288). arXiv.
https://arxiv.org/abs/2410.00288

Wei, J., Yang, S., & Cui, Z. (2025). *Unified GARCH-recurrent neural network in financial
volatility forecasting* (arXiv:2504.09380). arXiv. https://arxiv.org/abs/2504.09380

Liu, C., Wang, C., Tran, M.-N., & Kohn, R. (2023). *Deep learning enhanced realized GARCH*
(arXiv:2302.08002). arXiv. https://arxiv.org/abs/2302.08002

Zhao, P., Zhu, H., Ng, W. S. H., & Lee, D. L. (2024). *From GARCH to neural network for
volatility forecast* (arXiv:2402.06642). arXiv. https://arxiv.org/abs/2402.06642

## 3. Methodological critique and evaluation discipline

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022). *Are transformers effective for time series
forecasting?* (arXiv:2205.13504). arXiv. https://arxiv.org/abs/2205.13504

Cortesi, F. V., Iannone, G., Crippa, G., Poggio, T., & Beneventano, P. (2026). *Same error,
different function: The optimizer as an implicit prior in financial time series*
(arXiv:2603.02620). arXiv. https://arxiv.org/abs/2603.02620

## 4. Architectures and current state of the art

Brini, A. (2026). *Forecasting realized volatility with time series foundation models: A
comparison with econometric benchmarks* (arXiv:2607.05291). arXiv.
https://arxiv.org/abs/2607.05291

Rodikov, G., & Antulov-Fantulin, N. (2022). *Volatility-inspired σ-LSTM cell*
(arXiv:2205.07022). arXiv. https://arxiv.org/abs/2205.07022

Vennerød, C. B., Kjærran, A., & Bugge, E. S. (2021). *Long short-term memory RNN*
(arXiv:2105.06756). arXiv. https://arxiv.org/abs/2105.06756

## 5. Regime switching and stress periods

Blake, A. C., Gandhi, N. A., & Jakkula, A. R. (2025). *Improving S&P 500 volatility
forecasting through regime-switching methods* (arXiv:2510.03236). arXiv.
https://arxiv.org/abs/2510.03236

Tang, S. H., Rosenbaum, M., & Zhou, C. (2023). *Forecasting volatility with machine learning
and rough volatility: Example from the crypto-winter* (arXiv:2311.04727). arXiv.
https://arxiv.org/abs/2311.04727

## 6. Foundational works (non-arXiv, required citations)

### 6.1 ARIMA / SARIMA

statsmodels developers. (2025). *SARIMAX: Model selection, missing data*.
https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_internet.html

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time
series analysis: Forecasting and control* (5th ed.). John Wiley & Sons.
ISBN 978-1-118-67502-1.

### 6.2 ARCH / GARCH family

ARCH developers. (2024). *Forecasting: ARCH univariate volatility models*.
https://arch.readthedocs.io/en/latest/univariate/forecasting.html

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the
variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1007.
https://doi.org/10.2307/1912773

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal
of Econometrics, 31*(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1

Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach.
*Econometrica, 59*(2), 347–370. https://doi.org/10.2307/2938260

Engle, R. F., & Ng, V. K. (1993). Measuring and testing the impact of news on volatility.
*The Journal of Finance, 48*(5), 1749–1778. https://doi.org/10.1111/j.1540-6261.1993.tb05127.x

Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the
expected value and the volatility of the nominal excess return on stocks. *The Journal of
Finance, 48*(5), 1779–1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x

### 6.3 HAR and realized volatility

Müller, U. A., Dacorogna, M. M., Davé, R. D., Olsen, R. B., Pictet, O. V., & von Weizsäcker,
J. E. (1997). Volatilities of different time resolutions — Analyzing the dynamics of market
components. *Journal of Empirical Finance, 4*(2–3), 213–239.
https://doi.org/10.1016/S0927-5398(97)00007-8

Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and
forecasting realized volatility. *Econometrica, 71*(2), 579–625.
https://doi.org/10.1111/1468-0262.00418

Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard
volatility models do provide accurate forecasts. *International Economic Review,
39*(4), 885–905. https://doi.org/10.2307/2527343

Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up: Including jump
components in the measurement, modeling, and forecasting of return volatility. *The Review
of Economics and Statistics, 89*(4), 701–720. https://doi.org/10.1162/rest.89.4.701

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of
Financial Econometrics, 7*(2), 174–196. https://doi.org/10.1093/jjfinec/nbp001

### 6.4 Estimators, evaluation, and ML in asset pricing

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy.
*International Journal of Forecasting, 22*(4), 679–688.
https://doi.org/10.1016/j.ijforecast.2006.03.001

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
return. *The Journal of Business, 53*(1), 61–65. https://doi.org/10.1086/296071

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies.
*Journal of Econometrics, 160*(1), 246–256.
https://doi.org/10.1016/j.jeconom.2010.03.034

Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
*Econometrica, 79*(2), 453–497. https://doi.org/10.3982/ECTA5771

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The
Review of Financial Studies, 33*(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

## 7. Baseline comparison (HMM literature)

Hassan, M. R., & Nath, B. (2005). Stock market forecasting using hidden Markov model: A new
approach. *5th International Conference on Intelligent Systems Design and Applications
(ISDA'05)*, 192–196. https://doi.org/10.1109/ISDA.2005.85

Kuinchtner, D., & Madalozzo, G. A. (2018). *Predição do mercado de ações usando Hidden
Markov Model* [Undergraduate thesis]. Universidade de Passo Fundo.

---

## 8. Cross-asset pooling and global forecasting models
- Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting, 37(4), 1632–1653. https://arxiv.org/pdf/2008.00444

- Sirignano, J., & Cont, R. (2019). Universal features of price formation in financial markets: Perspectives from deep learning. Quantitative Finance, 19(9), 1449–1459. https://arxiv.org/abs/1803.06917

- Zhang, C., Zhang, Y., Cucuringu, M., & Qian, Z. (2022). Volatility forecasting with machine learning and intraday commonality (arXiv:2202.08962). arXiv. https://arxiv.org/abs/2202.08962

## Open-access copies

Several Section 6 papers have author- or institution-hosted PDFs, useful without a journal
subscription:

- Nelson (1991): https://web.pdx.edu/~crkl/readings/Nelson91.pdf
- Parkinson (1980): https://www.cmegroup.com/trading/fx/files/michael_parkinson.pdf
- Andersen, Bollerslev, Diebold & Labys (2003):
  https://users.ssc.wisc.edu/~behansen/718/Anderson2003.pdf
- Patton (2011): https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf
- Gu, Kelly & Xiu (2020): https://dachxiu.chicagobooth.edu/download/ML.pdf
  (also NBER Working Paper 25398: https://www.nber.org/papers/w25398)

---

**Verification note:** Sections 1–5 were retrieved from the arXiv API and Section 6 was
verified against publisher records, so IDs, titles, authors, volumes, pages, and DOIs are as
published. One caveat: Engle (1982) is cited in the literature as both 987–1007 and
987–1008; the Econometric Society and JSTOR records give 987–1007, used here.

Section 7 was compiled from the bibliography of Kuinchtner & Madalozzo and is not
independently verified; confirm the thesis year and publication details before submission.

## Training implementation sources

- Elfwing, S., Uchibe, E., and Doya, K. (2017). *Sigmoid-weighted linear units for neural network function approximation in reinforcement learning*. arXiv:1702.03118v3. Equation (10) defines the SiLU derivative (dSiLU): https://arxiv.org/pdf/1702.03118

- Pennsylvania State University, *STAT 508, Lesson 4: Linear Regression* (Gauss--Markov theorem and the BLUE qualification): https://online.stat.psu.edu/stat508/Lesson04.html

- Kalman, R. E. (1960). *A new approach to linear filtering and prediction problems*. *Journal of Basic Engineering*, 82(1), 35–45. https://doi.org/10.1115/1.3662552

- NumPy, `numpy.linalg.lstsq` (pooled AR(1) and HAR ordinary least squares): https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
- statsmodels, `SARIMAX.loglike` (sum per-ticker likelihoods for shared SARIMA parameters): https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.loglike.html
- `arch` documentation, *ARCH Modeling* (GARCH(1,1) likelihood and fixed-parameter forecasting): https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html
- Gatheral, J., Jaisson, T., & Rosenbaum, M., *Volatility is rough* (RFSV roughness motivation): https://arxiv.org/abs/1410.3394

## Realized utility sources

- Zhang, C., Zhang, Y., Cucuringu, M., & Qian, Z. (2022). *Volatility forecasting with machine learning and intraday commonality*, Section 5.4, Equation (21). https://arxiv.org/pdf/2202.08962
- YCharts. (n.d.). *10 Year Treasury Rate*, reported long-term average 4.26%. https://ycharts.com/indicators/10_year_treasury_rate

## Price reconstruction source

- Yahoo Finance Help. (n.d.). *What is the adjusted close?* (reverse-split multiplier applied to pre-split prices): https://help.yahoo.com/kb/SLN28256.html

- Wheeler Real Estate Investment Trust, Inc. (2025). *2024 Form 10-K*, Reverse Stock Splits note (August 2023, May/June/September/November 2024, and January 2025 splits): https://www.sec.gov/Archives/edgar/data/1527541/000152754125000038/whlr-20241231.htm

- 22nd Century Group, Inc. (2025). *2024 Form 10-K*, Reverse Stock Split note (July 2023; April and December 2024): https://www.sec.gov/Archives/edgar/data/1347858/000155837025003345/xxii-20241231x10k.htm

- 22nd Century Group, Inc. (2026). *2025 Form 10-K*, Reverse Stock Split note (April and December 2024; June 2025; January 2026): https://www.sec.gov/Archives/edgar/data/1347858/000110465926034814/xxii-20251231x10k.htm

- Nasdaq Trader (2026). *Equity Corporate Actions Alert #2026-396* (XXII 1-for-20 reverse split effective June 12, 2026): https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-396

- Nuwellis, Inc. (2024). *Form 8-K*, 1-for-35 reverse split effective June 27, 2024: https://www.sec.gov/Archives/edgar/data/1506492/000114036124031389/ef20031638_8k.htm
- Zeta Network Group (2025). *Prospectus*, 1-for-25 reverse split and ADD-to-ZNB ticker change effective August 22, 2025: https://www.sec.gov/Archives/edgar/data/1747661/000121390025097615/ea0260776-424b5_zeta.htm
- Propanc Biopharma, Inc. (2025). *Prospectus*, 1-for-60,000 reverse split processed January 29, 2025: https://www.sec.gov/Archives/edgar/data/1517681/000164117225024617/form424b3.htm
- Jaguar Health, Inc. (2025). *2024 Form 10-K*, 1-for-60 in May 2024 and 1-for-25 in March 2025: https://www.sec.gov/Archives/edgar/data/1585608/000095017025047184/jagx-20241231.htm
- XTI Aerospace, Inc. (2025). *2024 Form 10-K*, 1-for-100 in March 2024 and 1-for-250 in January 2025: https://www.sec.gov/Archives/edgar/data/1529113/000121390025032213/ea0235307-10k_xtiaerospace.htm
- Cemtrex, Inc. (2025). *2025 Form 10-K*, 1-for-60 and 1-for-35 in 2024 and 1-for-15 in September 2025: https://www.sec.gov/Archives/edgar/data/1435064/000149315225029383/form10-k.htm

## Descriptive shape statistics

- Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley. https://search.worldcat.org/title/3058187
- National Institute of Standards and Technology. (n.d.). *Measures of skewness and kurtosis*. Engineering Statistics Handbook, Section 1.3.5.11. https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
- World Health Organization & United Nations Children's Fund. (2019). *Recommendations for data collection, analysis and reporting on anthropometric indicators in children under 5 years old* (p. 71). https://iris.who.int/bitstream/handle/10665/324791/9789241515559-eng.pdf
- Westfall, P. H. (2014). Kurtosis as peakedness, 1905--2014. R.I.P. *The American Statistician, 68*(3), 191--195. https://doi.org/10.1080/00031305.2014.917055

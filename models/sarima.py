import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMA:
    """Fit coefficients once; filter each ticker's history with those coefficients."""

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 5)):
        self.order = order
        self.seasonal_order = seasonal_order

    def _model(self, history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("history must be a nonempty finite one-dimensional series")
        return SARIMAX(values, order=self.order, seasonal_order=self.seasonal_order)

    def fit(self, history):
        return np.asarray(self._model(history).fit(disp=False).params)

    def forecast(self, history, fitted_params):
        model = self._model(history)
        params = np.asarray(fitted_params, dtype=float)
        if params.shape != (len(model.param_names),) or not np.isfinite(params).all():
            raise ValueError("fitted_params must match this SARIMA specification")
        return float(model.filter(params).forecast(steps=1)[0])

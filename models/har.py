import numpy as np


class HAR:
    """Linear one-step forecast using 1, 5, and 20 observation variance summaries."""

    def __init__(self, coefficients=None):
        self.coefficients = None if coefficients is None else np.asarray(coefficients, dtype=float)
        if self.coefficients is not None and self.coefficients.shape != (4,):
            raise ValueError("HAR needs four coefficients")

    @staticmethod
    def features(history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
            raise ValueError("HAR needs at least 20 finite observations")
        return np.array([1., values[-1], values[-5:].mean(), values[-20:].mean()])

    def fit(self, history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or len(values) < 21 or not np.isfinite(values).all():
            raise ValueError("HAR fitting needs at least 21 finite observations")
        x = np.stack([self.features(values[:i]) for i in range(20, len(values))])
        self.coefficients = np.linalg.lstsq(x, values[20:], rcond=None)[0]
        return self.coefficients.copy()

    def forecast(self, history, coefficients=None):
        params = self.coefficients if coefficients is None else np.asarray(coefficients, dtype=float)
        if params is None or params.shape != (4,) or not np.isfinite(params).all():
            raise ValueError("four finite HAR coefficients are required")
        return float(self.features(history) @ params)

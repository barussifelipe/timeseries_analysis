import numpy as np


class AR1:
    """One-step AR(1) variance forecast with supplied coefficients."""

    def __init__(self, intercept, coefficient):
        self.intercept = float(intercept)
        self.coefficient = float(coefficient)

    def forecast(self, history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("history must be a nonempty finite one-dimensional series")
        return self.intercept + self.coefficient * values[-1]

import math
import numpy as np


class RFSV:
    """One-step log-volatility forecast using a discrete rough-volatility kernel."""

    def __init__(self, H, nu=0.0):
        if not math.isfinite(H) or not 0 < H < 0.5 or not math.isfinite(nu) or nu < 0:
            raise ValueError("require 0 < H < 0.5 and nu >= 0")
        self.H, self.nu = H, nu

    def forecast(self, volatility_history):
        values = np.asarray(volatility_history, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("volatility_history must contain positive finite observations")
        lag = np.arange(1, len(values) + 1, dtype=float)
        # Midpoint rule for the paper's integral, truncated to observed history.
        u = lag + 0.5
        weights = math.cos(math.pi * self.H) / math.pi / ((u + 1) * u ** (self.H + 0.5))
        correction = self.nu**2 * math.gamma(1.5 - self.H) / (2 * math.gamma(self.H + 0.5) * math.gamma(2 - 2 * self.H))
        return math.exp(correction + float(weights @ np.log(values[::-1])))

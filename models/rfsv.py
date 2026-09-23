import math
import numpy as np
from scipy.special import betainc


class RFSV:
    """Section 5 one-observation volatility forecast from past volatility."""

    def __init__(self, H, nu_squared=0.0):
        if not math.isfinite(H) or not 0 < H < 0.5 or not math.isfinite(nu_squared) or nu_squared < 0:
            raise ValueError("require 0 < H < 0.5 and nu_squared >= 0")
        self.H, self.nu_squared = H, nu_squared

    def forecast(self, volatility_history):
        values = np.asarray(volatility_history, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("volatility_history must contain positive finite observations")
        boundaries = np.arange(len(values) + 1, dtype=float)
        mass = betainc(.5 - self.H, .5 + self.H, boundaries / (1 + boundaries))
        weights = np.diff(mass)
        weights[-1] += 1 - mass[-1]  # Extend the oldest observation through the tail.
        c = math.gamma(1.5 - self.H) / (math.gamma(self.H + .5) * math.gamma(2 - 2 * self.H))
        log_variance = float(weights @ (2 * np.log(values[::-1]))) + 2 * c * self.nu_squared
        return math.exp(log_variance / 2)


def train(args):
    from models.variance_fit import statistical_train
    return statistical_train('rfsv', args)

def predict(fit, frame, split='test', residuals=None):
    from models.variance_fit import statistical_predict
    return statistical_predict('rfsv', fit, frame, split, residuals)

if __name__ == '__main__':
    from models.variance_fit import main
    main('rfsv')

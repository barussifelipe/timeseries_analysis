import math
import numpy as np


class GARCH:
    """GARCH(1,1) volatility from log close/open return residuals and variance."""

    def __init__(self, omega, alpha, beta):
        if not all(math.isfinite(x) for x in (omega, alpha, beta)) or omega <= 0 or min(alpha, beta) < 0 or alpha + beta >= 1:
            raise ValueError("require omega > 0, alpha and beta >= 0, alpha + beta < 1")
        self.omega, self.alpha, self.beta = omega, alpha, beta

    def forecast(self, residual_history, initial_variance):
        residuals = np.asarray(residual_history, dtype=float)
        if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("residual_history must be a nonempty finite series")
        if not math.isfinite(initial_variance) or initial_variance <= 0:
            raise ValueError("initial_variance must be positive and finite")
        variance = float(initial_variance)
        for residual in residuals:
            variance = self.omega + self.alpha * residual**2 + self.beta * variance
        return math.sqrt(variance)


def train(args):
    from models.variance_fit import statistical_train
    return statistical_train('garch', args)

def predict(fit, frame, split='test', residuals=None):
    from models.variance_fit import statistical_predict
    return statistical_predict('garch', fit, frame, split, residuals)

if __name__ == '__main__':
    from models.variance_fit import main
    main('garch')

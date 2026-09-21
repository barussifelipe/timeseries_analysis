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


def train(args):
    from models.variance_fit import statistical_train
    return statistical_train('ar1', args)

def predict(fit, frame, split='test', residuals=None):
    from models.variance_fit import statistical_predict
    return statistical_predict('ar1', fit, frame, split, residuals)

if __name__ == '__main__':
    from models.variance_fit import main
    main('ar1')

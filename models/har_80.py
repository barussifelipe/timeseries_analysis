"""Linear HAR variance forecast with 1, 5, 20, 40, and 80 observation means."""

import numpy as np


class HAR80:
    def __init__(self, coefficients=None):
        self.coefficients = None if coefficients is None else np.asarray(coefficients, dtype=float)
        if self.coefficients is not None and self.coefficients.shape != (6,):
            raise ValueError('HAR-80 needs six coefficients')

    @staticmethod
    def features(history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or len(values) < 80 or not np.isfinite(values).all():
            raise ValueError('HAR-80 needs at least 80 finite observations')
        return np.array([1., *(values[-n:].mean() for n in (1, 5, 20, 40, 80))])

    def fit(self, history):
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or len(values) < 81 or not np.isfinite(values).all():
            raise ValueError('HAR-80 fitting needs at least 81 finite observations')
        x = np.stack([self.features(values[:i]) for i in range(80, len(values))])
        self.coefficients = np.linalg.lstsq(x, values[80:], rcond=None)[0]
        return self.coefficients.copy()

    def forecast(self, history, coefficients=None):
        params = self.coefficients if coefficients is None else np.asarray(coefficients, dtype=float)
        if params is None or params.shape != (6,) or not np.isfinite(params).all():
            raise ValueError('six finite HAR-80 coefficients are required')
        return float(self.features(history) @ params)


def train(args):
    from models.variance_fit import statistical_train
    return statistical_train('har_80', args)


def predict(fit, frame, split='test', residuals=None):
    from models.variance_fit import statistical_predict
    return statistical_predict('har_80', fit, frame, split, residuals)


if __name__ == '__main__':
    from models.variance_fit import main
    main('har_80')

"""Focused checks for the Section 5 finite-history forecast convention."""

import math

import numpy as np
from scipy.integrate import quad

from models.rfsv import RFSV


def test_rfsv_observation_forecast():
    h = .1
    history = np.linspace(.01, .03, 20)
    kernel = lambda u: math.cos(math.pi * h) / math.pi / ((1 + u) * u ** (h + .5))
    expected_log = sum(
        2 * math.log(history[-1 - i]) * quad(kernel, i, i + 1)[0]
        for i in range(20)
    ) + 2 * math.log(history[0]) * quad(kernel, 20, np.inf)[0]
    actual = RFSV(h).forecast(history) ** 2
    assert math.isclose(math.log(actual), expected_log, abs_tol=1e-9)
    assert math.isclose(RFSV(h).forecast(np.full(20, .01)), .01, abs_tol=1e-15)
    nu_squared = .02
    c = math.gamma(1.5 - h) / (math.gamma(h + .5) * math.gamma(2 - 2 * h))
    corrected = RFSV(h, nu_squared).forecast(history) ** 2
    assert math.isclose(math.log(corrected / actual), 2 * c * nu_squared, abs_tol=1e-12)


if __name__ == '__main__':
    test_rfsv_observation_forecast()

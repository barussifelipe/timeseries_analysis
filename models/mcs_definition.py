"""Model Confidence Set for aligned, single-asset QLIKE loss histories.

Callers supply one column per candidate and one chronological row per shared
forecast observation. Forecast loading, key alignment, and per-stock seed
management belong to the later evaluation entry point.
"""

import numpy as np


def qlike_losses(actual, predicted):
    """Dimensionless QLIKE for positive actual and predicted variances."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("actual and predicted must have the same nonempty shape")
    if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError("QLIKE requires finite positive actual and predicted variance")
    log_ratio = np.log(actual) - np.log(predicted)
    losses = np.expm1(log_ratio) - log_ratio
    if not np.isfinite(losses).all():
        raise ValueError("QLIKE loss must be finite")
    return losses


def model_confidence_set(losses, candidates, *, block_length=20,
                         repetitions=10_000, alpha=0.10, seed=42):
    """Run the T_max MCS with a non-circular moving-block bootstrap.

    ``losses`` has shape (observations, candidates). Rows must already be
    aligned, sorted, and restricted to one asset; this function cannot verify
    forecast keys from a loss matrix alone. Returns a trace and adjusted MCS
    p-values keyed by candidate identifier. No forecasts are fitted or read.
    """
    values = np.asarray(losses, dtype=np.float64)
    names = tuple(candidates)
    if (values.ndim != 2 or values.shape[1] != len(names)
            or values.shape[1] < 2 or not np.isfinite(values).all()):
        raise ValueError("losses must be finite with at least two candidate columns")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("candidate identifiers must be unique nonempty strings")
    if not isinstance(block_length, int) or block_length < 1 or values.shape[0] < block_length:
        raise ValueError("block length must be positive and at most the observation count")
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("at least two bootstrap repetitions are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")

    order = np.argsort(names)
    names = tuple(names[i] for i in order)
    values = values[:, order]
    n = values.shape[0]
    blocks, remainder = divmod(n, block_length)
    draws = blocks + bool(remainder)
    starts = np.random.default_rng(seed).integers(0, n - block_length + 1,
                                                  size=(repetitions, draws))
    prefix = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0)))
    boot_sums = (prefix[starts[:, :blocks] + block_length]
                 - prefix[starts[:, :blocks]]).sum(axis=1)
    if remainder:
        boot_sums += prefix[starts[:, -1] + remainder] - prefix[starts[:, -1]]
    boot_means = boot_sums / n
    remaining = list(range(len(names)))
    trace = []
    adjusted = {name: 1.0 for name in names}
    running_p = 0.0
    while len(remaining) > 1:
        current = values[:, remaining]
        excess = current - current.mean(axis=1, keepdims=True)
        observed = excess.mean(axis=0)
        bootstrap = boot_means[:, remaining]
        bootstrap -= bootstrap.mean(axis=1, keepdims=True)
        zero_series = np.all(excess == 0, axis=0)
        bootstrap[:, zero_series] = 0
        standard_errors = bootstrap.std(axis=0, ddof=1)
        constant_nonzero = np.all(excess == excess[:1], axis=0) & ~zero_series
        if (constant_nonzero.any() or np.any((standard_errors == 0) & ~zero_series)
                or not np.isfinite(standard_errors).all()):
            raise ValueError("nonzero excess loss has zero or non-finite bootstrap variance")
        standardized = np.divide(observed, standard_errors,
                                 out=np.zeros_like(observed), where=standard_errors > 0)
        centered = bootstrap - observed
        bootstrap_statistics = np.divide(centered, standard_errors,
                                         out=np.zeros_like(centered), where=standard_errors > 0)
        statistic = float(standardized.max())
        maxima = bootstrap_statistics.max(axis=1)
        p_value = (1 + np.count_nonzero(maxima >= statistic)) / (repetitions + 1)
        removed_index = int(np.argmax(standardized))  # names are sorted: first wins ties
        removed = names[remaining[removed_index]] if p_value < alpha else None
        trace.append({"candidates": tuple(names[i] for i in remaining),
                      "t_max": statistic, "p_value": float(p_value), "removed": removed})
        if removed is None:
            break
        running_p = max(running_p, p_value)
        adjusted[removed] = float(running_p)
        remaining.pop(removed_index)

    return {"members": tuple(names[i] for i in remaining),
            "adjusted_p_values": adjusted, "trace": trace,
            "settings": {"block_length": block_length, "repetitions": repetitions,
                         "alpha": alpha, "seed": seed, "generator": "numpy.default_rng",
                         "numpy_version": np.__version__}}

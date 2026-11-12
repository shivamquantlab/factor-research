"""Factor calculations with explicit signal/forward-return alignment."""

import numpy as np


def _paired(scores: np.ndarray, forward_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(forward_returns, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("scores and forward_returns must be equal-length vectors with at least 2 assets")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    if np.var(x) == 0:
        raise ValueError("scores must have non-zero cross-sectional variance")
    return x, y


def factor_slope(scores: np.ndarray, forward_returns: np.ndarray) -> float:
    """OLS slope with an intercept: exposures at t versus returns after t."""
    x, y = _paired(scores, forward_returns)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float(centered_x @ centered_y / (centered_x @ centered_x))


def long_short_return(
    scores: np.ndarray,
    forward_returns: np.ndarray,
    *,
    quantile: float = 0.2,
) -> float:
    """Equal-weight top-minus-bottom score bucket return after the signal date."""
    x, y = _paired(scores, forward_returns)
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    count = max(1, int(np.floor(x.size * quantile)))
    order = np.argsort(x, kind="stable")
    return float(y[order[-count:]].mean() - y[order[:count]].mean())


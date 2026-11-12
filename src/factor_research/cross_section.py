"""Factor calculations with explicit signal/forward-return alignment."""

import numpy as np


def _paired(scores: np.ndarray, forward_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(forward_returns, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("scores and forward_returns must be equal-length vectors with at least 2 assets")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    score_scale = float(np.max(np.abs(x)))
    if score_scale == 0.0 or np.ptp(x / score_scale) == 0.0:
        raise ValueError("scores must have non-zero cross-sectional variance")
    return x, y


def factor_slope(scores: np.ndarray, forward_returns: np.ndarray) -> float:
    """OLS slope with an intercept: exposures at t versus returns after t."""
    x, y = _paired(scores, forward_returns)
    score_scale = float(np.max(np.abs(x)))
    return_scale = float(np.max(np.abs(y)))
    scaled_x = x / score_scale
    centered_x = scaled_x - scaled_x.mean()
    if return_scale == 0.0:
        return 0.0
    scaled_y = y / return_scale
    centered_y = scaled_y - scaled_y.mean()
    slope_scaled = float(centered_x @ centered_y / (centered_x @ centered_x))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        slope = slope_scaled * (return_scale / score_scale)
    if not np.isfinite(slope):
        raise ValueError("factor slope exceeds the finite float range")
    return float(slope)


def _boundary_weighted_mean(
    scores: np.ndarray,
    returns: np.ndarray,
    count: int,
    *,
    top: bool,
) -> float:
    order = np.argsort(scores, kind="stable")
    ranked = order[-count:] if top else order[:count]
    boundary = scores[ranked[0]] if top else scores[ranked[-1]]
    strict = scores > boundary if top else scores < boundary
    tied = scores == boundary
    remaining = count - int(strict.sum())
    weights = strict.astype(float) + tied.astype(float) * (remaining / int(tied.sum()))
    scale = float(np.max(np.abs(returns)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.dot(returns / scale, weights) / count)


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
    spread = _boundary_weighted_mean(x, y, count, top=True) - _boundary_weighted_mean(
        x, y, count, top=False
    )
    if not np.isfinite(spread):
        raise ValueError("long-short return spread exceeds the finite float range")
    return float(spread)


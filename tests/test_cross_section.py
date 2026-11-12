import numpy as np
import pytest

from factor_research.cross_section import factor_slope, long_short_return


def test_factor_slope_recovers_known_linear_relationship():
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    forward_returns = 0.004 + 0.025 * scores
    assert factor_slope(scores, forward_returns) == pytest.approx(0.025)


def test_long_short_bucket_has_expected_return_spread():
    scores = np.arange(10.0)
    forward_returns = scores / 100.0
    assert long_short_return(scores, forward_returns, quantile=0.2) == pytest.approx(0.08)


def test_invalid_quantile_is_rejected():
    with pytest.raises(ValueError, match="quantile"):
        long_short_return(np.array([0, 1]), np.array([0, 1]), quantile=0.6)


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


def test_tied_bucket_boundary_is_fractional_and_order_invariant():
    scores = np.array([0.0, 0.0, 0.0, 2.0])
    returns = np.array([0.01, 0.02, 0.03, 0.10])
    expected = 0.10 - np.mean([0.01, 0.02, 0.03])
    assert long_short_return(scores, returns, quantile=0.25) == pytest.approx(expected)
    permutation = np.array([2, 3, 0, 1])
    assert long_short_return(scores[permutation], returns[permutation], quantile=0.25) == pytest.approx(expected)


def test_slope_is_scale_stable_for_large_finite_scores():
    scores = np.array([-2e307, -1e307, 1e307, 2e307])
    forward_returns = 0.004 + 0.025 * scores
    assert factor_slope(scores, forward_returns) == pytest.approx(0.025)


def test_constant_scores_are_rejected_without_variance_overflow():
    with pytest.raises(ValueError, match="non-zero"):
        factor_slope(np.array([1e308, 1e308]), np.array([0.01, 0.02]))


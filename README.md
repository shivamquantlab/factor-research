# Factor Research

## Research question
Given cross-sectional signals observable at time t, what return spread is associated with higher scores over the following period?

## Methodology and formulas
For centered score x_i and next-period return y_i, the OLS factor slope with an intercept is beta_hat = sum((x_i-mean(x))*(y_i-mean(y))) / sum((x_i-mean(x))^2). The illustrative portfolio statistic is the equal-weight mean return in the top score quantile minus the bottom quantile. Every synthetic score is generated before its paired forward return.

## Assumptions and limitations
- The demo creates 40 synthetic assets per period with a planted score-return relation and independent noise using NumPy seed 31.
- Returns are gross of fees, market impact, and borrow costs; the result is not investment performance or evidence of a real factor premium.
- There is no point-in-time security master, delisting-return treatment, corporate-action history, or survivorship-bias correction.
- The simple cross-sectional slope is descriptive; it does not control for correlated factors, sector structure, multiple testing, or parameter selection.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and factor-demo --seed 31. The command saves reports/factor_spread.png and reports/summary.json; all observations are synthetic and generated locally.


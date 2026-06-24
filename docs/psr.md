# probabilistic_sharpe_ratio

## Function

Probability that the true Sharpe ratio exceeds a benchmark, accounting
for sample size, skewness and kurtosis of returns.

## API

```python
from oos_lab import probabilistic_sharpe_ratio

probabilistic_sharpe_ratio(returns, benchmark_sr=0.0) -> float
```

- `returns` : 1-D sequence of per-period returns. Need >= 4 obs.
- `benchmark_sr` : per-period Sharpe benchmark (default 0).

## Formula

\\[
\\text{PSR}(SR^*) = \\Phi\\!\\left(\\frac{(\\hat{SR} - SR^*)\\sqrt{n-1}}{\\sqrt{1 - \\gamma_3 \\hat{SR} + (\\gamma_4 - 1)/4 \\cdot \\hat{SR}^2}}\\right)
\\]

where \\(\\hat{SR}\\) is per-period sample Sharpe, \\(\\gamma_3\\) is skewness,
\\(\\gamma_4\\) is non-excess kurtosis.

## Limitations

- Asymptotic; less accurate for `n < 24`.
- Sharpe and benchmark must be in the same time units (per period).

## Relationships

- `deflated_sharpe_ratio` plugs `expected_max_sharpe` into PSR as the
  benchmark.

## References

- Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk*, 15(2), 3-44.

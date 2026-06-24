# sharpe_ratio

## Function

Annualised sample Sharpe ratio of a return series.

## API

```python
from oos_lab import sharpe_ratio

sharpe_ratio(returns, periods_per_year=1, risk_free=0.0) -> float
```

- `returns` : 1-D sequence of per-period returns. NaNs dropped.
- `periods_per_year` : integer multiplier. Pass 252 for daily stocks,
  365 for daily crypto, 1 for per-period.
- `risk_free` : constant subtracted from every return before computing.

## Definition

Returns `mean(r - risk_free) / std(r - risk_free, ddof=1) * sqrt(periods_per_year)`.

## Limitations

- Assumes IID returns; biased for autocorrelated or heteroscedastic series.
- Single annualisation factor; no per-period weighting.

## Relationships

- `probabilistic_sharpe_ratio` computes a confidence statement around
  this Sharpe.
- `deflated_sharpe_ratio` corrects this Sharpe for selection bias across
  trials.

## References

- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business*,
  39(1), 119-138.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts
  Journal*, 58(4), 36-52.

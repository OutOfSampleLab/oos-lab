# probability_of_backtest_overfit

## Function

Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation.

## API

```python
from oos_lab import probability_of_backtest_overfit, PBOResult

probability_of_backtest_overfit(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
) -> PBOResult
```

- `returns_matrix` : `(n_obs, n_variants)` per-variant return matrix.
- `n_partitions` : even integer >= 4. Number of equal blocks.

## Returns

`PBOResult` dataclass:
- `pbo` : fraction of splits where the in-sample winner underperforms the
  out-of-sample median.
- `logits` : per-combination relative-rank logits.
- `in_sample_sharpes`, `out_of_sample_sharpes` : per-combination picks.
- `performance_degradation_slope`, `_intercept` : OLS regression of OOS
  on IS Sharpe across combinations.

## Interpretation

- PBO < 0.2 : edge survives selection.
- PBO ~ 0.5 : indistinguishable from noise.
- PBO > 0.6 : likely overfit.

## Limitations

- Computational cost is `C(n_partitions, n_partitions/2)` (peaks at
  n_partitions=16, 12870 splits).
- Assumes equal-length partitions; uneven sampling can bias the result.

## References

- Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014).
  The Probability of Backtest Overfitting. *Journal of Computational
  Finance*, 20(4), 39-70.

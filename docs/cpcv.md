# CombinatorialPurgedKFold

## Class

Combinatorial Purged Cross-Validation with embargo (Lopez de Prado AFML
Ch.12.4).

## API

```python
from oos_lab import CombinatorialPurgedKFold

CombinatorialPurgedKFold(n_splits: int,
                         n_test_splits: int,
                         embargo_pct: float = 0.0)

cv.split(n, t1) -> Iterator[(train_idx, test_idx)]
cv.n_splits_total -> int  # C(n_splits, n_test_splits)
cv.paths -> int           # C(n_splits-1, n_test_splits-1)
```

- `n_splits` : total contiguous partitions of the time axis.
- `n_test_splits` : number of partitions used as test in each combo.
- `embargo_pct` : fraction of `n` to embargo after each test block.
- `t1` : per-observation label-end index (use `np.arange(1, n+1)` for
  unit-length labels).

## Algorithm

1. Partition `[0, n)` into `n_splits` contiguous groups.
2. Enumerate every choice of `n_test_splits` test groups.
3. Test = union of selected groups.
4. Training = complement minus purged observations (label ends in a test
   group) minus embargo (bars after a test block).

## Limitations

- Simplified purge that only checks the label-end index. Full AFML purge
  also considers the label-start side.
- Computational cost is `C(n_splits, n_test_splits)` strategy fits.

## Relationships

- Outputs feed `probability_of_backtest_overfit` for PBO.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
  Ch. 12. Wiley.

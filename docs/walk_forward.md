# WalkForward

## Class

Anchored or rolling walk-forward index splitter for time-series data.

## API

```python
from oos_lab import WalkForward

WalkForward(train_size: int, test_size: int,
            step: int = 0, anchored: bool = False)

wf.split(n) -> Iterator[(train_idx, test_idx)]
wf.n_splits(n) -> int
```

- `anchored=True` expands training from index 0.
- `anchored=False` slides a fixed-size training window forward.
- `step=0` defaults to `test_size` (non-overlapping test windows).

## Limitations

- No purge or embargo. Use `CombinatorialPurgedKFold` when labels overlap.
- Yields integer index arrays only; bring your own backtester.

## Relationships

- Simpler counterpart to `CombinatorialPurgedKFold`.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
  Ch. 7. Wiley.

# haircut_sharpe

## Function

Harvey-Liu (2015) multiple-testing haircut on a collection of Sharpe
ratios. Implements Holm-Bonferroni (FWER) and BHY (FDR under arbitrary
dependence).

## API

```python
from oos_lab import haircut_sharpe, HaircutResult

haircut_sharpe(
    sharpes,
    n_obs: int,
    method: Literal["holm", "bhy"] = "holm",
    alpha: float = 0.05,
) -> HaircutResult
```

- `sharpes` : per-strategy Sharpe estimates (same time units as `n_obs`).
- `n_obs` : sample size used to estimate each Sharpe.
- `method` : `"holm"` for Holm-Bonferroni FWER control; `"bhy"` for
  Benjamini-Hochberg-Yekutieli FDR control under arbitrary dependence.
- `alpha` : significance threshold for the survives mask.

## Returns

`HaircutResult` dataclass with `raw_sharpes`, `haircut_sharpes`,
`raw_pvalues`, `adjusted_pvalues`, `survives_at_alpha`, `method`.

## Implementation notes

- Adjusted p-values are clipped to `[0, 1]` and floored at the raw
  p-value to guarantee `haircut <= raw`.
- Round-trip through `scipy.stats.norm.ppf` introduces sub-1e-6 numerical
  noise; haircut is also capped at raw to absorb it.

## Limitations

- Assumes Sharpe estimator is asymptotically normal; correction is
  conservative for short samples.
- Holm and BHY are not strictly ordered at every rank; pick the procedure
  whose error type you care about.

## References

- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio
  Management*, 42(1), 13-28.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test
  Procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Benjamini, Y., & Yekutieli, D. (2001). The Control of the False
  Discovery Rate under Dependency. *Annals of Statistics*, 29(4),
  1165-1188.

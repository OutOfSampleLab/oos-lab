# oos-lab overview

A research-grade validation toolkit for systematic trading strategies.

## Modules

| Module | Public API | Purpose |
| --- | --- | --- |
| `metrics.sharpe` | `sharpe_ratio` | Annualised sample Sharpe with `ddof=1`. |
| `metrics.psr` | `probabilistic_sharpe_ratio` | PSR (Bailey & Lopez de Prado 2012). |
| `metrics.deflated_sharpe` | `deflated_sharpe_ratio`, `expected_max_sharpe` | DSR + False Strategy Theorem (Bailey & Lopez de Prado 2014). |
| `cv.walk_forward` | `WalkForward` | Anchored or rolling time-series CV. |
| `cv.cpcv` | `CombinatorialPurgedKFold` | Combinatorial purged CV with embargo. |
| `overfit.pbo` | `probability_of_backtest_overfit`, `PBOResult` | PBO via CSCV. |
| `overfit.haircut` | `haircut_sharpe`, `HaircutResult` | Holm-Bonferroni and BHY haircuts. |

## Module dependencies

```
metrics.returns  -- internal helpers (to_array, log_returns)
   |
   +--> metrics.sharpe
   +--> metrics.psr
   +--> metrics.deflated_sharpe

cv.walk_forward  -- standalone splitter
cv.cpcv          -- standalone splitter

overfit.pbo      -- uses numpy + scipy
overfit.haircut  -- uses scipy.stats for normal CDF/PPF
```

## Versioning

Semantic versioning. Public API is everything re-exported from
`oos_lab/__init__.py`. Breaking changes bump major; new features bump
minor; bug fixes bump patch.

## License

MIT.

## References

See per-module docs and `oos-lab/README.md` for the full bibliography.

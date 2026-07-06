# oos-lab

Research-grade validation toolkit for systematic trading strategies.

Most backtests are fantasies. `oos-lab` gives you the statistical tools to
tell whether yours is real, with no buy/sell signals and no performance
claims of its own. Open source, MIT licensed.

## What is in v0.4

**Metrics**

- `sharpe_ratio` — annualised sample Sharpe with `ddof=1`.
- `probabilistic_sharpe_ratio` — Bailey & López de Prado (2012) PSR. The
  probability that the true Sharpe exceeds a benchmark, accounting for the
  skew and kurtosis of returns.
- `deflated_sharpe_ratio` — Bailey & López de Prado (2014) DSR. PSR with the
  benchmark raised to the expected maximum across `n_trials` strategy
  variants under the false strategy theorem. The first defence against
  selection bias.
- `expected_max_sharpe` — the underlying closed form, useful on its own when
  you want to see how high a "lucky max" Sharpe gets without any real edge.

**Cross-validation**

- `WalkForward` — anchored or rolling walk-forward index splitter for
  time-series strategies.
- `CombinatorialPurgedKFold` — combinatorial purged cross-validation with
  embargo (López de Prado 2018), the leakage-aware splitter for financial
  series.

**Overfitting**

- `probability_of_backtest_overfit` — Probability of Backtest Overfitting
  via Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de
  Prado & Zhu 2014). Returns the PBO scalar plus the performance-degradation
  regression.
- `haircut_sharpe` — Harvey & Liu (2015) multiple-testing haircut with
  Holm-Bonferroni and Benjamini-Hochberg-Yekutieli adjustments.

Every public symbol is re-exported from `oos_lab/__init__.py`. The package
depends only on `numpy` and `scipy`. 56 unit tests pass.

## Install

```
pip install oos-lab
```

## Quickstart

```python
import numpy as np
from oos_lab import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    WalkForward,
    CombinatorialPurgedKFold,
    probability_of_backtest_overfit,
    haircut_sharpe,
)

rng = np.random.default_rng(0)
returns = rng.normal(0.0008, 0.012, size=1000)

print("Sharpe annualised:", sharpe_ratio(returns, periods_per_year=252))
print("PSR vs zero      :", probabilistic_sharpe_ratio(returns))
print("DSR over 1000    :", deflated_sharpe_ratio(returns,
                                                  n_trials=1000,
                                                  var_sharpe_across_trials=0.04))
print("Expected max SR  :", expected_max_sharpe(1000, 0.04))

# PBO needs a matrix of variant returns: shape (n_obs, n_variants)
variants = rng.normal(0.0002, 0.012, size=(1000, 20))
result = probability_of_backtest_overfit(variants, n_partitions=16)
print("PBO              :", result.pbo)
print("Degradation slope:", result.performance_degradation_slope)
```

## What this is NOT

- Not a backtester. Bring your own engine.
- Not a strategy. There is no buy or sell logic anywhere.
- Not financial advice, and not a personalised recommendation. Educational
  and methodological use only. `oos-lab` is an impersonal tool that runs the
  same way for everyone; it makes no claim about your account, your
  positions, or any future result.

## A note on any performance numbers

`oos-lab` computes statistics; it never shows a track record of its own. If
you use it to present simulated or backtested results to others, attach the
standard hypothetical-performance disclaimer (CFTC Rule 4.41): simulated
results have inherent limitations, do not represent actual trading, and are
designed with the benefit of hindsight, so no representation is made that any
account will achieve similar profits or losses.

## Documentation

Per-module write-ups live in [`docs/`](docs/): `sharpe`, `psr`,
`deflated_sharpe`, `walk_forward`, `cpcv`, `pbo`, `haircut`, plus an
[`overview`](docs/overview.md).

## Pro notebooks

The core library is free and always will be. If you want guided, worked
applications of this stack, there is a set of paid Jupyter notebooks that
apply it end to end — a Deflated Sharpe + PBO diagnostic walkthrough, a CPCV
practitioner pack, a Harvey-Liu multiple-testing haircut pack, and
regime-conditional validation — available individually or as a bundle:

**[OutOfSampleLab on Polar](https://polar.sh/outofsamplelab)**

Educational and methodological content only. No signals, no advice, no
performance claims.

## Roadmap

- v0.5 — bootstrap confidence intervals for the deflated Sharpe.
- v0.6 — minimum backtest length and minimum track record length helpers.

## References

- Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *The Journal of Portfolio Management*, 40(5), 94-107.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). The
  Probability of Backtest Overfitting. *Journal of Computational Finance*,
  20(4), 39-70.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *The Journal of Portfolio
  Management*, 42(1), 13-28.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## License

MIT. See [LICENSE](LICENSE).

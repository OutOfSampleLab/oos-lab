# deflated_sharpe_ratio + expected_max_sharpe

## Functions

PSR with the benchmark raised to the expected maximum across `n_trials`
independent backtests under the False Strategy Theorem null.

## API

```python
from oos_lab import deflated_sharpe_ratio, expected_max_sharpe

expected_max_sharpe(n_trials: int,
                    var_sharpe_across_trials: float) -> float

deflated_sharpe_ratio(returns,
                      n_trials: int,
                      var_sharpe_across_trials: float) -> float
```

- `n_trials` : number of parameter variants tested.
- `var_sharpe_across_trials` : sample variance of Sharpes across trials.

## Formula

\\[
E[\\max SR] = \\sigma_{SR} \\big[(1 - \\gamma_{em}) \\Phi^{-1}(1 - 1/N) + \\gamma_{em} \\Phi^{-1}(1 - 1/(Ne))\\big]
\\]

where \\(\\gamma_{em}\\) is the Euler-Mascheroni constant.

DSR = PSR(returns, benchmark_sr = expected_max_sharpe(...)).

## Limitations

- Asymptotic; needs `n_trials >= ~10` for the closed form.
- Estimating `var_sharpe_across_trials` from a hand-picked subset
  understates the benchmark.

## Relationships

- Builds on `probabilistic_sharpe_ratio`.
- Complementary to `probability_of_backtest_overfit` (CSCV-based).

## References

- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*, 40(5), 94-107.

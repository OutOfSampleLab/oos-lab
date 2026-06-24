# Changelog

All notable changes to oos-lab are listed here. This project follows
Semantic Versioning; minor versions add functionality without breaking
the public API.

## 0.4.0 — Harvey-Liu multiple-testing haircut

- Added `haircut_sharpe` and `HaircutResult` in `oos_lab.overfit.haircut`.
- Holm-Bonferroni (family-wise error rate) and
  Benjamini-Hochberg-Yekutieli (false discovery rate under arbitrary
  dependence) corrections.
- Adjusted p-values are inverted back to a haircut Sharpe at the same
  sample size, providing a direct comparable to the raw estimate.

## 0.3.0 — Probability of Backtest Overfitting

- Added `probability_of_backtest_overfit` and `PBOResult` in
  `oos_lab.overfit.pbo`.
- Combinatorially Symmetric Cross-Validation per Bailey, Borwein, Lopez
  de Prado & Zhu (2014).
- Returns PBO scalar, per-combination logits, in/out-of-sample Sharpes
  of the selected variant, and OLS performance-degradation coefficients.

## 0.2.0 — Combinatorial Purged Cross-Validation

- Added `CombinatorialPurgedKFold` in `oos_lab.cv.cpcv`.
- Implements purge of training observations whose label end falls
  inside a test group, plus configurable embargo after each test block,
  per Lopez de Prado AFML Ch.12.4.
- Exposes `n_splits_total = C(n_splits, n_test_splits)` and
  `paths = C(n_splits - 1, n_test_splits - 1)`.

## 0.1.0 — Initial release

- `sharpe_ratio` annualised sample Sharpe with `ddof=1`.
- `probabilistic_sharpe_ratio` per Bailey & Lopez de Prado (2012),
  accounting for skewness and kurtosis of returns.
- `deflated_sharpe_ratio` and `expected_max_sharpe` per Bailey & Lopez
  de Prado (2014) False Strategy Theorem.
- `WalkForward` anchored or rolling time-series splitter.

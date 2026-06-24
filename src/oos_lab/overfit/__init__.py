"""Overfitting detection and multiple-testing corrections."""

from oos_lab.overfit.pbo import (
    probability_of_backtest_overfit,
    PBOResult,
)
from oos_lab.overfit.haircut import (
    haircut_sharpe,
    HaircutResult,
)

__all__ = [
    "probability_of_backtest_overfit",
    "PBOResult",
    "haircut_sharpe",
    "HaircutResult",
]

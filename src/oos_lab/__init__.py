"""oos-lab: research-grade validation toolkit for systematic trading."""

from oos_lab.metrics.sharpe import sharpe_ratio
from oos_lab.metrics.psr import probabilistic_sharpe_ratio
from oos_lab.metrics.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from oos_lab.cv.walk_forward import WalkForward
from oos_lab.cv.cpcv import CombinatorialPurgedKFold
from oos_lab.overfit.pbo import (
    probability_of_backtest_overfit,
    PBOResult,
)
from oos_lab.overfit.haircut import (
    haircut_sharpe,
    HaircutResult,
)

__version__ = "0.4.0"

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "WalkForward",
    "CombinatorialPurgedKFold",
    "probability_of_backtest_overfit",
    "PBOResult",
    "haircut_sharpe",
    "HaircutResult",
]

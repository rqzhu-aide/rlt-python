"""rlt: Reinforcement Learning Trees for Python.

sklearn-compatible random forests for regression, classification, and
survival analysis, ported from the RLT R package.
"""

from .estimators import RLT_reg, RLT_cla, RLT_surv
from .bands import get_surv_band

__version__ = "0.2.0"
__all__ = ["RLT_reg", "RLT_cla", "RLT_surv",
           "get_surv_band"]

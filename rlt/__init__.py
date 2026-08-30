"""rlt: Reinforcement Learning Trees for Python.

sklearn-compatible random forests for regression, classification, and
survival analysis, ported from the RLT R package.
"""

from .estimators import RLTRegressor, RLTClassifier, RLTSurvivalForest

__version__ = "0.1.0"
__all__ = ["RLTRegressor", "RLTClassifier", "RLTSurvivalForest"]

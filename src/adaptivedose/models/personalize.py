from __future__ import annotations
from typing import Dict, List
import numpy as np

FEATURES: List[str] = ["age", "weight", "bmi", "sex_male", "asa"]

class RequirementModel:
    """Ordinary least squares: covariates -> propofol maintenance requirement."""

    def __init__(self):
        self.coef_ = None  # coef_[0] is the intercept

    def fit(self, X, y) -> "RequirementModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.column_stack([np.ones(len(X)), X])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        A = np.column_stack([np.ones(len(X)), X])
        return A @ self.coef_

    def coefficients(self, feature_names: List[str]) -> Dict[str, float]:
        names = ["intercept"] + list(feature_names)
        return {n: float(c) for n, c in zip(names, self.coef_)}

def leave_one_out(X, y) -> np.ndarray:
    """Return one out-of-sample prediction per row (fit on all others)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        preds[i] = RequirementModel().fit(X[mask], y[mask]).predict(X[i])[0]
    return preds

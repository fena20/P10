"""Simple ensembling utilities.

Addresses reviewer suggestion to use ensembles to mitigate residual biases.

The ensemble here is intentionally lightweight:
- Weighted average of multiple fitted base estimators
- Optional non-negative clipping

For nested CV usage, fit each base estimator on the same training data and
use predictions from each model to build the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.utils.helpers import clip_predictions


@dataclass
class EnsembleMember:
    name: str
    estimator: BaseEstimator
    weight: float = 1.0


class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    """A simple weighted-average ensemble of sklearn-style regressors."""

    def __init__(self, members: Sequence[EnsembleMember], clip_non_negative: bool = True):
        self.members = list(members)
        self.clip_non_negative = clip_non_negative
        self.fitted_members_: List[EnsembleMember] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.fitted_members_ = []
        for m in self.members:
            est = clone(m.estimator)
            if sample_weight is not None:
                try:
                    est.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    est.fit(X, y)
            else:
                est.fit(X, y)
            self.fitted_members_.append(EnsembleMember(name=m.name, estimator=est, weight=float(m.weight)))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.fitted_members_:
            raise RuntimeError("Ensemble is not fitted")

        weights = np.array([m.weight for m in self.fitted_members_], dtype=float)
        if np.sum(weights) <= 0:
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)

        preds = None
        for w, m in zip(weights, self.fitted_members_):
            p = m.estimator.predict(X)
            preds = p * w if preds is None else preds + p * w

        if self.clip_non_negative:
            preds, _, _ = clip_predictions(preds, min_val=0.0, report=False)
        return preds

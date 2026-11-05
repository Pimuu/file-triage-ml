"""
Model zoo for the triage experiment.

All models are small scikit-learn estimators that train in seconds on a
laptop CPU. A transparent rule-based baseline is included so the learned
models can be compared against what an analyst might write by hand.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HeuristicRuleBaseline(ClassifierMixin, BaseEstimator):
    """
    Hand-written triage rule (no learning):
        flag if double_extension
             or has_space_padding
             or (extension is script/shortcut/macro-document)
             or (extension is executable and suspicious_keyword).
    predict_proba returns a coarse 3-level score so ROC-AUC is defined.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X):
        strong = (X["double_extension"] | X["has_space_padding"] | X["ext_cat_script"]
                  | X["ext_cat_shortcut"] | X["ext_cat_macro_document"]
                  | (X["ext_cat_executable"] & X["suspicious_keyword"]))
        weak = X["ext_cat_executable"] | X["suspicious_keyword"]
        return np.where(strong == 1, 1.0, np.where(weak == 1, 0.5, 0.0))

    def predict_proba(self, X):
        s = self._score(X)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self._score(X) >= 1.0).astype(int)


def build_models(seed: int = 42) -> dict:
    """Return {display_name: unfitted estimator}."""
    return {
        "Rule baseline": HeuristicRuleBaseline(),
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed,
        ),
    }

import numpy as np


def _select_columns(X, feature_columns=None):
    if feature_columns is None or not hasattr(X, "loc"):
        return X
    return X.loc[:, feature_columns]


class EncodedClassifierModel:
    def __init__(self, estimator, classes, feature_columns=None):
        self.estimator = estimator
        self.classes_ = np.array(classes)
        self.feature_columns = list(feature_columns) if feature_columns else None

    def predict_proba(self, X):
        selected = _select_columns(X, self.feature_columns)
        return self.estimator.predict_proba(selected)

    def predict(self, X):
        selected = _select_columns(X, self.feature_columns)
        predictions = self.estimator.predict(selected)
        predictions = np.asarray(predictions).reshape(-1).astype(int)
        return self.classes_[predictions]

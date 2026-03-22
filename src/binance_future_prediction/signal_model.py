import numpy as np


def _select_columns(X, feature_columns=None):
    if feature_columns is None or not hasattr(X, "loc"):
        return X
    return X.loc[:, feature_columns]


class ConstantProbabilityModel:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        probs = np.asarray(probabilities, dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(self.classes_), dtype=float)
        self.probabilities_ = probs / probs.sum()

    def predict_proba(self, X):
        rows = len(X)
        return np.tile(self.probabilities_, (rows, 1))

    def predict(self, X):
        winner = int(np.argmax(self.probabilities_))
        return np.repeat(self.classes_[winner], len(X))


class TwoStageSignalModel:
    def __init__(self, trade_estimator, direction_estimator, trade_feature_columns=None, direction_feature_columns=None):
        self.trade_estimator = trade_estimator
        self.direction_estimator = direction_estimator
        self.trade_feature_columns = list(trade_feature_columns) if trade_feature_columns else None
        self.direction_feature_columns = list(direction_feature_columns) if direction_feature_columns else None
        self.classes_ = np.array([-1, 0, 1])

    def _positive_probability(self, estimator, X, positive_class, feature_columns=None):
        selected = _select_columns(X, feature_columns)
        probabilities = estimator.predict_proba(selected)
        classes = np.asarray(estimator.classes_)
        positive_index = int(np.where(classes == positive_class)[0][0])
        return probabilities[:, positive_index]

    def predict_proba(self, X):
        trade_probability = self._positive_probability(self.trade_estimator, X, 1, self.trade_feature_columns)
        long_probability_given_trade = self._positive_probability(self.direction_estimator, X, 1, self.direction_feature_columns)
        short_probability = trade_probability * (1.0 - long_probability_given_trade)
        long_probability = trade_probability * long_probability_given_trade
        no_trade_probability = 1.0 - trade_probability
        probabilities = np.column_stack([short_probability, no_trade_probability, long_probability])
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums

    def predict(self, X):
        probabilities = self.predict_proba(X)
        winners = probabilities.argmax(axis=1)
        return self.classes_[winners]


class BlendedSignalModel:
    def __init__(self, models, weights):
        self.models = list(models)
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones(len(self.models), dtype=float)
        self.weights = weight_array / weight_array.sum()
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        combined = None
        for weight, model in zip(self.weights, self.models):
            weighted = model.predict_proba(X) * weight
            combined = weighted if combined is None else combined + weighted
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return combined / row_sums

    def predict(self, X):
        probabilities = self.predict_proba(X)
        winners = probabilities.argmax(axis=1)
        return self.classes_[winners]

import numpy as np


class WeightedEnsembleModel:
    def __init__(self, estimators: dict, weights: dict[str, float]):
        self.estimators = estimators
        self.weights = weights
        self.classes_ = next(iter(estimators.values())).classes_

    def predict_proba(self, X):
        total_weight = sum(self.weights.values())
        probabilities = None

        for name, estimator in self.estimators.items():
            weighted = estimator.predict_proba(X) * self.weights[name]
            probabilities = weighted if probabilities is None else probabilities + weighted

        return probabilities / total_weight

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

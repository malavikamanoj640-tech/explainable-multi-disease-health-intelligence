import numpy as np


class AdaptiveEnsemble:
    """Sample-adaptive probability ensemble.

    Base estimators must implement predict_proba on the same raw feature frame.
    Global model weights come from validation performance, while per-sample
    weights are adjusted by each model's confidence for that sample.
    """
    def __init__(self, estimators=None, model_weights=None, confidence_power=2.0):
        self.estimators = estimators or []
        self.model_weights = model_weights
        self.confidence_power = confidence_power
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        if y is not None:
            for _, estimator in self.estimators:
                estimator.fit(X, y)
        if self.model_weights is None:
            self.model_weights = np.ones(len(self.estimators), dtype=float)
        self.model_weights = np.asarray(self.model_weights, dtype=float)
        self.model_weights = np.maximum(self.model_weights, 1e-8)
        return self

    def _probabilities(self, X):
        return [np.asarray(est.predict_proba(X)) for _, est in self.estimators]

    def predict_proba(self, X):
        probs = self._probabilities(X)
        rows = []
        base_weights = self.model_weights / self.model_weights.sum()
        for i in range(len(X)):
            sample_probs = np.array([p[i] for p in probs])
            confidence = np.max(sample_probs, axis=1)
            adaptive = base_weights * np.power(confidence, self.confidence_power)
            adaptive = adaptive / adaptive.sum()
            rows.append(np.sum(sample_probs * adaptive[:, None], axis=0))
        return np.asarray(rows)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def model_details(self, X):
        """Return transparent per-model probabilities and adaptive weights.

        This is an explanatory view only; it does not alter the ensemble result.
        """
        probs = self._probabilities(X)
        base_weights = self.model_weights / self.model_weights.sum()
        output = []
        for i in range(len(X)):
            rows = []
            sample_probs = np.array([p[i] for p in probs])
            confidence = np.max(sample_probs, axis=1)
            adaptive = base_weights * np.power(confidence, self.confidence_power)
            adaptive = adaptive / adaptive.sum()
            for (name, _), p_row, w in zip(self.estimators, sample_probs, adaptive):
                pred = int(self.classes_[int(np.argmax(p_row))])
                rows.append({
                    "model": name,
                    "predicted_class": pred,
                    "low_risk_probability": round(float(p_row[0]) * 100, 2),
                    "high_risk_probability": round(float(p_row[1]) * 100, 2),
                    "adaptive_weight": round(float(w) * 100, 2),
                })
            output.append(rows)
        return output

    def adaptive_weights(self, X):
        probs = self._probabilities(X)
        base_weights = self.model_weights / self.model_weights.sum()
        output = []
        for i in range(len(X)):
            confidence = np.array([np.max(p[i]) for p in probs])
            adaptive = base_weights * np.power(confidence, self.confidence_power)
            adaptive = adaptive / adaptive.sum()
            output.append({name: round(float(w), 6) for (name, _), w in zip(self.estimators, adaptive)})
        return output

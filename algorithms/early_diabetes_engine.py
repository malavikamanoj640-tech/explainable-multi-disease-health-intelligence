"""Early Diabetes risk prediction engine.

This is an additive restoration of the Early Diabetes analysis that existed in
Diabetes_model.zip. It uses the supplied 520-row early-diabetes dataset and
keeps the same research-oriented ensemble/XAI style as the current Diabetes
module.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

try:
    import shap
except Exception:
    shap = None

EARLY_FIELDS = [
    "age", "gender", "polyuria", "polydipsia", "sudden_weight_loss",
    "weakness", "polyphagia", "genital_thrush", "visual_blurring",
    "itching", "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity"
]
EARLY_COLUMNS = [
    "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]
YES_NO_FIELDS = set(EARLY_FIELDS[2:])


def _factories():
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, min_samples_leaf=3, random_state=42),
        "SVM": lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    }


def _pipeline(factory):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", factory())])


def _encode_frame(df):
    out = df.copy()
    out["Gender"] = out["Gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
    for col in EARLY_COLUMNS[2:]:
        out[col] = out[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out[EARLY_COLUMNS].apply(pd.to_numeric, errors="coerce")


def train_early_diabetes_engine(base_dir: Path):
    path = base_dir / "dataset" / "diabetes_data_upload.csv"
    if not path.exists():
        raise FileNotFoundError(f"Early Diabetes dataset not found: {path}")
    df = pd.read_csv(path)
    X = _encode_frame(df)
    y = df["class"].astype(str).str.strip().str.lower().map({"positive": 1, "negative": 0})
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, cv_metrics, test_metrics, raw_weights = {}, {}, {}, {}
    for name, factory in _factories().items():
        model = _pipeline(factory)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cv_metrics[name] = {"balanced_accuracy_mean": float(scores.mean()), "balanced_accuracy_std": float(scores.std())}
        test_metrics[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "brier": float(brier_score_loss(y_test, prob)),
        }
        raw_weights[name] = max(float(scores.mean()), 0.01)
        models[name] = model
    total = sum(raw_weights.values())
    weights = {k: v / total for k, v in raw_weights.items()}
    return EarlyDiabetesEngine(models, weights, cv_metrics, test_metrics, X_train, base_dir)


class EarlyDiabetesEngine:
    def __init__(self, models, weights, cv_metrics, test_metrics, X_train, base_dir):
        self.models = models
        self.weights = weights
        self.cv_metrics = cv_metrics
        self.test_metrics = test_metrics
        self.feature_names = EARLY_FIELDS
        self.model_columns = EARLY_COLUMNS
        self.X_train = X_train
        self.base_dir = base_dir
        self._shap_background = None

    def _frame(self, data):
        values = []
        values.append(float(data.get("age")) if data.get("age") not in (None, "") else np.nan)
        gender = str(data.get("gender", "")).strip().lower()
        values.append(1.0 if gender == "male" else 0.0 if gender == "female" else np.nan)
        for field in EARLY_FIELDS[2:]:
            value = str(data.get(field, "")).strip().lower()
            values.append(1.0 if value == "yes" else 0.0 if value == "no" else np.nan)
        return pd.DataFrame([values], columns=self.model_columns)

    def predict(self, data):
        X = self._frame(data)
        probabilities, confidences = {}, {}
        for name, model in self.models.items():
            p = float(model.predict_proba(X)[0, 1])
            probabilities[name] = p
            confidences[name] = max(p, 1.0 - p)
        adjusted = {n: self.weights[n] * (confidences[n] ** 2.0) for n in self.models}
        total = sum(adjusted.values()) or 1.0
        adjusted = {n: v / total for n, v in adjusted.items()}
        high_prob = float(np.clip(sum(adjusted[n] * probabilities[n] for n in self.models), 0, 1))
        prediction = int(high_prob >= 0.5)
        details = [{
            "model": name,
            "predicted_class": int(probabilities[name] >= 0.5),
            "low_risk_probability": round((1 - probabilities[name]) * 100, 2),
            "high_risk_probability": round(probabilities[name] * 100, 2),
            "adaptive_weight": round(adjusted[name] * 100, 2),
        } for name in self.models]
        return {
            "prediction": prediction,
            "high_risk_probability": round(high_prob * 100, 2),
            "predicted_class_probability": round((high_prob if prediction else 1 - high_prob) * 100, 2),
            "model_probabilities": probabilities,
            "adaptive_weights": adjusted,
            "model_details": details,
        }

    def shap_explanation(self, data):
        if shap is None:
            return {"available": False, "error": "SHAP is not installed in this environment."}
        try:
            pipe = self.models["Random Forest"]
            imputer, scaler, rf = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["model"]
            X = self._frame(data)
            X_scaled = scaler.transform(imputer.transform(X))
            if self._shap_background is None:
                bg = scaler.transform(imputer.transform(self.X_train.iloc[:min(40, len(self.X_train))]))
                self._shap_background = bg
            explainer = shap.TreeExplainer(rf, self._shap_background)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                vals = np.asarray(sv[1][0], dtype=float)
            else:
                arr = np.asarray(sv)
                vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
            items = []
            for label, value, raw in zip(self.feature_names, vals, X.iloc[0].tolist()):
                items.append({
                    "feature": label.replace("_", " ").title(),
                    "label": label.replace("_", " ").title(),
                    "input": raw,
                    "value": raw,
                    "shap_value": round(float(value), 5),
                    "magnitude": abs(float(value)),
                    "description": f"{label.replace('_', ' ').title()} is supplied to the Early Diabetes machine-learning model.",
                })
            items.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return {"available": True, "method": "TreeExplainer (Random Forest explanation model)", "summary": "Local feature attribution describes which submitted inputs moved the model toward or away from the higher-risk class. This explains model behavior and does not establish clinical causation.", "features": items}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

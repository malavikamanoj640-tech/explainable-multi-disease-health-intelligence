"""Production diabetes prediction engine for the MultiDiseasePrediction app.

Uses the diabetes dataset supplied with the project and exposes the same
research-oriented interface used by the heart module: ensemble prediction,
model transparency, and local XAI.
"""
from pathlib import Path
import json
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix

try:
    import shap
except Exception:
    shap = None


DIABETES_FIELDS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age"
]
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]


def _factories():
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, min_samples_leaf=4, random_state=42),
        "SVM": lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    }


def _pipeline(factory):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", factory()),
    ])


def train_diabetes_engine(base_dir: Path):
    dataset_path = base_dir / "dataset" / "diabetes_Dataset_cleaned.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Diabetes dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    X = df[DIABETES_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Outcome"], errors="coerce").astype("Int64")
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, cv_metrics, test_metrics, raw_weights = {}, {}, {}, {}

    for name, factory in _factories().items():
        model = _pipeline(factory)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cv_metrics[name] = {
            "balanced_accuracy_mean": float(scores.mean()),
            "balanced_accuracy_std": float(scores.std()),
        }
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
    ensemble_prob = np.zeros(len(X_test), dtype=float)
    for name, model in models.items():
        ensemble_prob += weights[name] * model.predict_proba(X_test)[:, 1]
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_metrics = {
        "accuracy": float(accuracy_score(y_test, ensemble_pred)),
        "precision": float(precision_score(y_test, ensemble_pred, zero_division=0)),
        "recall": float(recall_score(y_test, ensemble_pred, zero_division=0)),
        "f1": float(f1_score(y_test, ensemble_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, ensemble_prob)),
        "brier": float(brier_score_loss(y_test, ensemble_prob)),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred).tolist(),
    }

    return DiabetesEngine(
        models=models, weights=weights, cv_metrics=cv_metrics,
        test_metrics=test_metrics, ensemble_metrics=ensemble_metrics,
        X_train=X_train, y_train=y_train, base_dir=base_dir,
    )


class DiabetesEngine:
    def __init__(self, models, weights, cv_metrics, test_metrics, ensemble_metrics, X_train, y_train, base_dir):
        self.models = models
        self.weights = weights
        self.cv_metrics = cv_metrics
        self.test_metrics = test_metrics
        self.ensemble_metrics = ensemble_metrics
        self.feature_names = DIABETES_FIELDS
        self.model_columns = DIABETES_COLUMNS
        self.base_dir = base_dir
        self.X_train = X_train
        self.y_train = y_train
        self._shap_background = None

    def _frame(self, user_data):
        vals = []
        for f in self.feature_names:
            v = user_data.get(f)
            vals.append(float(v) if v is not None and str(v).strip() != "" else np.nan)
        return pd.DataFrame([vals], columns=self.model_columns)

    def predict(self, user_data):
        X = self._frame(user_data)
        probs = {}
        sample_conf = {}
        for name, model in self.models.items():
            p = float(model.predict_proba(X)[0, 1])
            probs[name] = p
            sample_conf[name] = max(p, 1.0-p)
        # Sample-adaptive adjustment around the globally validated weights.
        adjusted = {n: self.weights[n] * (sample_conf[n] ** 2.0) for n in self.models}
        total = sum(adjusted.values()) or 1.0
        adjusted = {n: v / total for n, v in adjusted.items()}
        high_prob = float(np.clip(sum(adjusted[n] * probs[n] for n in self.models), 0, 1))
        prediction = int(high_prob >= 0.5)
        details = []
        for name in self.models:
            details.append({
                "model": name,
                "predicted_class": int(probs[name] >= 0.5),
                "low_risk_probability": round((1-probs[name])*100, 2),
                "high_risk_probability": round(probs[name]*100, 2),
                "adaptive_weight": round(adjusted[name]*100, 2),
            })
        return {
            "prediction": prediction,
            "high_risk_probability": round(high_prob*100, 2),
            "predicted_class_probability": round((high_prob if prediction == 1 else 1-high_prob)*100, 2),
            "model_probabilities": probs,
            "adaptive_weights": adjusted,
            "model_details": details,
            "input_frame": X,
        }

    def shap_explanation(self, user_data):
        if shap is None:
            return {"available": False, "error": "SHAP is not installed in this environment."}
        try:
            rf = self.models["Random Forest"]
            pipe_imputer = rf.named_steps["imputer"]
            pipe_scaler = rf.named_steps["scaler"]
            rf_model = rf.named_steps["model"]
            X = self._frame(user_data)
            X_imp = pipe_imputer.transform(X)
            X_scaled = pipe_scaler.transform(X_imp)
            if self._shap_background is None:
                bg = pipe_scaler.transform(pipe_imputer.transform(self.X_train.iloc[:min(40, len(self.X_train))]))
                self._shap_background = bg
            explainer = shap.TreeExplainer(rf_model, self._shap_background)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                vals = np.asarray(sv[1][0], dtype=float)
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    vals = arr[0, :, 1]
                else:
                    vals = arr[0]
            items = []
            for label, val, raw in zip(self.feature_names, vals, X.iloc[0].tolist()):
                items.append({
                    "feature": label.replace("_", " ").title(),
                    "label": label.replace("_", " ").title(),
                    "input": raw,
                    "value": raw,
                    "shap_value": round(float(val), 5),
                    "magnitude": abs(float(val)),
                    "description": f"{label.replace('_', ' ').title()} is supplied to the diabetes machine-learning model."
                })
            items.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return {
                "available": True,
                "method": "TreeExplainer (Random Forest explanation model)",
                "summary": "Local feature attribution shows which submitted inputs moved the Random Forest explanation toward or away from the higher-risk class. This explains model behavior and does not establish clinical causation.",
                "features": items,
                "baseline": round(float(getattr(explainer, "expected_value", [0])[1] if np.ndim(getattr(explainer, "expected_value", 0)) else getattr(explainer, "expected_value", 0)), 5)
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("dataset/clean_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =====================================
# Split Dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# Feature Scaling
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# Base Models
# =====================================

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

gb = GradientBoostingClassifier(
    random_state=42
)

# =====================================
# Adaptive Ensemble
# =====================================

ensemble = VotingClassifier(
    estimators=[
        ("DecisionTree", dt),
        ("RandomForest", rf),
        ("GradientBoosting", gb)
    ],
    voting="hard"
)

ensemble.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================

y_pred = ensemble.predict(X_test)

# =====================================
# Evaluation
# =====================================

print("=" * 60)
print("Adaptive Ensemble Results")
print("=" * 60)

print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1 Score :", round(f1_score(y_test, y_pred), 3))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================
# Save Model
# =====================================

joblib.dump(ensemble, "models/adaptive_ensemble.pkl")

print("\nAdaptive Ensemble Model Saved Successfully!")
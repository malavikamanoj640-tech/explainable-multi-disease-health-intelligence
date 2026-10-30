import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Base Models
# -----------------------------
estimators = [
    ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )),

    ("gb", GradientBoostingClassifier(
        random_state=42
    )),

    ("lr", LogisticRegression(
        max_iter=1000
    ))
]

# -----------------------------
# Stacking Model
# -----------------------------
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# -----------------------------
# Train
# -----------------------------
stack_model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = stack_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models/heart", exist_ok=True)

joblib.dump(
    stack_model,
    "models/heart/ensemble_model.pkl"
)

print("\nAdaptive Stacking Ensemble Model Saved Successfully!")
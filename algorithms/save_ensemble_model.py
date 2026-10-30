import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")

# Features and Target
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Train-Test Split
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
lr = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

gb = GradientBoostingClassifier(
    random_state=42
)

# -----------------------------
# Adaptive Ensemble Model
# -----------------------------
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)

# Train
ensemble.fit(X_train, y_train)

# -----------------------------
# Create folder automatically
# -----------------------------
os.makedirs("models/heart", exist_ok=True)

# Save Model
joblib.dump(ensemble, "models/heart/ensemble_model.pkl")

print("Adaptive Ensemble Model Saved Successfully!")
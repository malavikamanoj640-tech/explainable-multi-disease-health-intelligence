import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

# =====================================
# Features and Target
# =====================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =====================================
# Train-Test Split
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
# Train Random Forest
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================

print("=" * 60)
print("Random Forest Results")
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

joblib.dump(model, "models/random_forest.pkl")

print("\nRandom Forest Model Saved Successfully!")
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("dataset/clean_diabetes.csv")

X = df.drop("Outcome", axis=1)

# Load Best Model
model = joblib.load("models/decision_tree.pkl")

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP Values
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(shap_values, X)

plt.show()
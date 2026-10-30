import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Clean Dataset
# -----------------------------
df = pd.read_csv("dataset/clean_diabetes.csv")

print("=" * 50)
print("Dataset Shape:", df.shape)
print("=" * 50)

print(df.head())

# -----------------------------
# Class Distribution
# -----------------------------
print("\nOutcome Distribution")
print(df["Outcome"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Class Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Histograms
# -----------------------------
df.hist(figsize=(15,10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Boxplots
# -----------------------------
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.show()

print("\nEDA Completed Successfully!")
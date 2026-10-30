import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================
# Load Dataset
# ============================================

file_path = "dataset/diabetes_Dataset_cleaned.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
except FileNotFoundError:
    print(f"Error: Cannot find {file_path}")
    exit()

# ============================================
# Display Dataset
# ============================================

print("=" * 60)
print("First 5 Rows")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("Column Names")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)
df.info()

# ============================================
# Missing Values
# ============================================

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)
print(df.isnull().sum())

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(exclude=np.number).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ============================================
# Duplicate Rows
# ============================================

duplicates = df.duplicated().sum()

print("\n" + "=" * 60)
print("Duplicate Rows")
print("=" * 60)
print("Duplicate Rows:", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# ============================================
# Statistics
# ============================================

print("\n" + "=" * 60)
print("Statistical Summary")
print("=" * 60)
print(df.describe())

# ============================================
# Feature Scaling
# ============================================

if "Outcome" in df.columns:

    print("\nTarget Column Found: Outcome")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    clean_df = pd.DataFrame(X_scaled, columns=X.columns)
    clean_df["Outcome"] = y.values

else:

    print("\n'Outcome' column not found.")
    print("Skipping Feature Scaling.")

    clean_df = df.copy()

# ============================================
# Save Clean Dataset
# ============================================

output_path = "dataset/clean_diabetes.csv"

clean_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("Clean Dataset Saved Successfully")
print("=" * 60)
print(output_path)
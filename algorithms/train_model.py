""""
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())
"""





import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df["DEATH_EVENT"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Count plot of target variable
plt.figure(figsize=(6,4))
sns.countplot(x="DEATH_EVENT", data=df)
plt.title("Distribution of Death Event")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

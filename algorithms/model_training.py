import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")

# Features (Input)
X = df.drop("DEATH_EVENT", axis=1)

# Target (Output)
y = df["DEATH_EVENT"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


from sklearn.model_selection import train_test_split

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
"""
Why stratify=y?

It ensures that the proportion of the target classes (survived and died) remains similar 
in both the training and testing sets, which leads to a more reliable evaluation.
"""
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
lr_model = LogisticRegression(max_iter=1000)

# Train model
lr_model.fit(X_train, y_train)

print("Logistic Regression Model Trained Successfully!")


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on test data
y_pred = lr_model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
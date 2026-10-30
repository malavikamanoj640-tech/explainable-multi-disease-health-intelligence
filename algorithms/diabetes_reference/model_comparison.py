import pandas as pd

results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "Gradient Boosting"
    ],
    "Accuracy": [
        71.43,
        79.22,
        75.32,
        75.32,
        75.32
    ],
    "Precision": [
        0.609,
        0.704,
        0.660,
        0.660,
        0.667
    ],
    "Recall": [
        0.519,
        0.704,
        0.611,
        0.611,
        0.593
    ],
    "F1 Score": [
        0.560,
        0.704,
        0.635,
        0.635,
        0.627
    ]
}

df = pd.DataFrame(results)

print("\nModel Comparison")
print("=" * 70)
print(df)

best_model = df.loc[df["Accuracy"].idxmax()]

print("\nBest Model")
print("=" * 70)
print(best_model)
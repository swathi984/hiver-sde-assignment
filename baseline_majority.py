import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load Golden Set
df = pd.read_csv("data/golden/golden_set_200.csv")

# Majority-class prediction
majority_intent = df["intent"].value_counts().idxmax()
df["prediction"] = majority_intent

# Metrics
accuracy = accuracy_score(df["intent"], df["prediction"])

print("===== MAJORITY CLASS BASELINE =====")
print(f"Total examples: {len(df)}")
print(f"Majority intent: {majority_intent}")
print(f"Accuracy: {accuracy:.3f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)
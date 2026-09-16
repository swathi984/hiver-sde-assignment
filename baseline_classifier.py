import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# LOAD HUMAN-LABELED GOLDEN SET
# ============================================================

DATA_PATH = "data/golden/golden_set_200.csv"

df = pd.read_csv(DATA_PATH)

# Remove rows with missing text or intent
df = df.dropna(subset=["text", "intent"])

X = df["text"]
y = df["intent"]


# ============================================================
# DATASET INFORMATION
# ============================================================

print(f"Total labeled samples: {len(df)}")

print("\nIntent distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# BASELINE MODEL
# TF-IDF + LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# BASELINE RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== BASELINE CLASSIFIER =====")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Accuracy: {accuracy:.3f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n===== CONFUSION MATRIX =====")

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df.to_string())
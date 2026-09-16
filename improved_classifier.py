import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

DATA_PATH = "data/golden/golden_set_200.csv"

df = pd.read_csv(DATA_PATH)
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
# SAME TRAIN/TEST SPLIT AS BASELINE
# This makes the comparison fair.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# IMPROVED FEATURE EXTRACTION
# WORD + CHARACTER TF-IDF
# ============================================================

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
    )
])


# ============================================================
# IMPROVED CLASSIFIER
# ============================================================

from sklearn.pipeline import Pipeline

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            C=1.0
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== IMPROVED CLASSIFIER =====")
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


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n===== BASELINE COMPARISON =====")
print("Baseline accuracy : 0.525")
print(f"Improved accuracy : {accuracy:.3f}")
print(f"Improvement       : {(accuracy - 0.525):+.3f}")
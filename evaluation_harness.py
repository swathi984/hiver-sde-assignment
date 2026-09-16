import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/golden/golden_set_200.csv"

RESULT_PATH = "data/golden/evaluation_results.csv"
CONFUSION_PATH = "data/golden/ml_confusion_matrix.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD GOLDEN SET
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["text", "intent"]
)

X = df["text"].astype(str)
y = df["intent"].astype(str)

print("=" * 70)
print("HIVER CLASSIFIER EVALUATION HARNESS")
print("=" * 70)

print(f"\nTotal labeled samples: {len(df)}")

print("\nIntent distribution:")
print(y.value_counts())


# ============================================================
# FIXED STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# BASELINE 1: MAJORITY CLASS
# ============================================================

majority_intent = y_train.value_counts().idxmax()

majority_predictions = [
    majority_intent
] * len(X_test)


# ============================================================
# BASELINE 2: KEYWORD / RULE-BASED CLASSIFIER
# ============================================================

def keyword_predict(text):

    text = str(text).lower()

    # Refund
    if any(word in text for word in [
        "refund",
        "money back",
        "refunded",
        "refunds"
    ]):
        return "refund_issue"

    # Return
    if any(word in text for word in [
        "return",
        "returning",
        "send back"
    ]):
        return "return_issue"

    # Cancellation
    if any(word in text for word in [
        "cancel",
        "cancelled",
        "canceled",
        "cancellation"
    ]):
        return "cancellation_issue"

    # Payment
    if any(word in text for word in [
        "payment",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "card",
        "billing"
    ]):
        return "payment_issue"

    # Delivery
    if any(word in text for word in [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "tracking",
        "arrive",
        "arrived",
        "late",
        "delay",
        "shipping"
    ]):
        return "delivery_issue"

    # Prime
    if "prime" in text:
        return "prime_issue"

    # Seller
    if any(word in text for word in [
        "seller",
        "vendor"
    ]):
        return "seller_issue"

    # Account
    if any(word in text for word in [
        "account",
        "password",
        "login",
        "logged in",
        "email address"
    ]):
        return "account_issue"

    # Complaint / escalation
    if any(word in text for word in [
        "complaint",
        "complain",
        "escalate",
        "manager",
        "supervisor",
        "unacceptable"
    ]):
        return "complaint_escalation"

    # General order
    if "order" in text:
        return "order_issue"

    return "other_or_unclear"


keyword_predictions = [
    keyword_predict(text)
    for text in X_test
]


# ============================================================
# BASELINE 3: TF-IDF + LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([
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

print("\nTraining Logistic Regression baseline...")

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)


# ============================================================
# OUR MODEL: WORD + CHARACTER TF-IDF + LINEAR SVM
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

improved_model = Pipeline([
    (
        "features",
        features
    ),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            C=1.0
        )
    )
])

print("Training improved model...")

improved_model.fit(
    X_train,
    y_train
)

improved_predictions = improved_model.predict(
    X_test
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(name, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Macro F1:    {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# RUN ALL EVALUATIONS
# ============================================================

results = []

results.append(
    evaluate(
        "BASELINE 1: MAJORITY CLASS",
        majority_predictions
    )
)

results.append(
    evaluate(
        "BASELINE 2: KEYWORD / RULES",
        keyword_predictions
    )
)

results.append(
    evaluate(
        "BASELINE 3: TF-IDF + LOGISTIC REGRESSION",
        logistic_predictions
    )
)

results.append(
    evaluate(
        "OUR MODEL: WORD + CHARACTER TF-IDF + LINEAR SVM",
        improved_predictions
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# CONFUSION MATRIX FOR OUR MODEL
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    improved_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_PATH
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(f"- {RESULT_PATH}")
print(f"- {CONFUSION_PATH}")
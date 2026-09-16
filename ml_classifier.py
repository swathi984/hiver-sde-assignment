import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load golden set
df = pd.read_csv("data/golden/golden_set_200.csv")

# First 100 = training
train_df = df[df["id"] <= 100].copy()

# Second 100 = held-out test set
test_df = df[df["id"] > 100].copy()

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000
)

X_train = vectorizer.fit_transform(
    train_df["text"].astype(str)
)

X_test = vectorizer.transform(
    test_df["text"].astype(str)
)

y_train = train_df["intent"]

# Train classifier
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


def predict_intent(text):
    """
    Predict the intent of a new customer message.
    """

    text_vector = vectorizer.transform([str(text)])

    prediction = model.predict(text_vector)[0]

    return prediction


if __name__ == "__main__":

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    predictions = model.predict(X_test)

    print("===== TF-IDF + LOGISTIC REGRESSION =====")
    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")

    print(
        f"Accuracy: "
        f"{accuracy_score(test_df['intent'], predictions):.3f}"
    )

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            test_df["intent"],
            predictions,
            zero_division=0
        )
    )
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def classify_intent(text):
    text = str(text).lower()

    # Higher-priority rules first
    if any(word in text for word in [
        "refund", "money back", "refund back", "refunded"
    ]):
        return "refund_issue"

    if any(word in text for word in [
        "return", "send back", "sending back"
    ]):
        return "return_issue"

    if any(word in text for word in [
        "cancel", "cancelled", "canceled", "cancellation"
    ]):
        return "cancellation_issue"

    if any(word in text for word in [
        "charge", "charged", "payment", "pay", "credit card",
        "debit card", "card", "bank"
    ]):
        return "payment_issue"

    if any(word in text for word in [
        "delivery", "delivered", "deliver", "courier",
        "package", "parcel", "arrive", "arrived", "late",
        "delay", "delayed", "tracking", "shipment"
    ]):
        return "delivery_issue"

    if any(word in text for word in [
        "prime", "amazon prime"
    ]):
        return "prime_issue"

    if any(word in text for word in [
        "seller", "merchant"
    ]):
        return "seller_issue"

    if any(word in text for word in [
        "password", "login", "logged in", "account",
        "sign in", "signin", "security"
    ]):
        return "account_issue"

    if any(word in text for word in [
        "complaint", "escalate", "escalation", "manager",
        "unhappy", "not satisfied", "no resolution",
        "no help", "useless", "disgusting", "fed up"
    ]):
        return "complaint_escalation"

    if "order" in text:
        return "order_issue"

    return "other_or_unclear"


# Load Golden Set
df = pd.read_csv("data/golden/golden_set_200.csv")

# Predict using rules
df["prediction"] = df["text"].apply(classify_intent)

# Metrics
accuracy = accuracy_score(df["intent"], df["prediction"])

print("===== KEYWORD / RULE BASELINE =====")
print(f"Total examples: {len(df)}")
print(f"Accuracy: {accuracy:.3f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)
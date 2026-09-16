import pandas as pd
import os

from reply_generator import generate_reply
from ml_classifier import predict_intent


# ============================================================
# LOAD HELD-OUT GOLDEN EXAMPLES
# ============================================================

df = pd.read_csv("data/golden/golden_set_200.csv")

# Use the held-out 100 examples
test_df = df[df["id"] > 100].copy()

# Select 20 examples for reply-quality evaluation.
# Fixed seed makes the evaluation reproducible.
evaluation_df = test_df.sample(
    n=20,
    random_state=42
).copy()


# ============================================================
# GENERATE REPLIES
# ============================================================

rows = []

for _, row in evaluation_df.iterrows():

    customer_message = row["text"]

    intent = predict_intent(
        customer_message
    )

    result = generate_reply(
        customer_message
    )

    rows.append({
        "id": row["id"],
        "customer_message": customer_message,
        "gold_intent": row["intent"],
        "predicted_intent": intent,
        "reply": result["reply"],
        "similarity": round(
            result["similarity"],
            3
        ),
        "grounded": result["grounded"],
        "human_overall_score": "",
        "human_notes": ""
    })


# ============================================================
# SAVE HUMAN EVALUATION FILE
# ============================================================

output_path = (
    "data/golden/reply_quality_evaluation.csv"
)

os.makedirs(
    "data/golden",
    exist_ok=True
)

output_df = pd.DataFrame(rows)

output_df.to_csv(
    output_path,
    index=False
)

print(
    f"Created {len(output_df)} reply evaluation examples."
)

print(
    f"Saved: {output_path}"
)

print("\nColumns requiring human evaluation:")

print(
    "- human_overall_score: rate from 1 to 5"
)

print(
    "- human_notes: briefly explain the score"
)
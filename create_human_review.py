import pandas as pd

INPUT_PATH = "data/golden/llm_judge_results.csv"
OUTPUT_PATH = "data/golden/human_llm_agreement.csv"

df = pd.read_csv(INPUT_PATH)

review = df[
    [
        "id",
        "customer_message",
        "reply",
        "llm_overall",
        "llm_unsupported_claim"
    ]
].copy()

review["human_overall"] = ""
review["human_unsupported_claim"] = ""
review["human_notes"] = ""

review.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("HUMAN REVIEW TEMPLATE CREATED")
print("=" * 70)
print(f"Examples: {len(review)}")
print(f"Saved: {OUTPUT_PATH}")
print()
print("Open the CSV and fill these columns:")
print("  human_overall: 1-5")
print("  human_unsupported_claim: True/False")
print("  human_notes: brief explanation")
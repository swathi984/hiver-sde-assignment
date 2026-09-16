import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/golden_set.csv")
OUTPUT_PATH = Path("data/golden_set.csv")

INTENTS = [
    "delivery_issue",
    "order_issue",
    "refund_issue",
    "return_issue",
    "payment_issue",
    "prime_issue",
    "account_issue",
    "seller_issue",
    "cancellation_issue",
    "complaint_escalation",
    "other_or_unclear",
]

df = pd.read_csv(INPUT_PATH)

# Treat missing intents as unlabeled
df["intent"] = df["intent"].fillna("")

unlabeled = df[df["intent"] == ""]

print("=" * 70)
print("GOLDEN SET LABELING")
print("=" * 70)
print(f"Total examples: {len(df)}")
print(f"Already labeled: {len(df) - len(unlabeled)}")
print(f"Remaining: {len(unlabeled)}")

print("\nINTENTS:")
for i, intent in enumerate(INTENTS, 1):
    print(f"{i}. {intent}")

print("\nEnter the number corresponding to the intent.")
print("Enter 'q' to save and quit.\n")

for index in df.index:
    if df.at[index, "intent"] != "":
        continue

    print("-" * 70)
    print(f"Example {index + 1} / {len(df)}")
    print(f"\n{df.at[index, 'text']}\n")

    while True:
        choice = input("Intent number: ").strip()

        if choice.lower() == "q":
            df.to_csv(OUTPUT_PATH, index=False)
            print("\nProgress saved.")
            print(f"Labeled so far: {(df['intent'] != '').sum()} / {len(df)}")
            raise SystemExit

        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            selected_intent = INTENTS[int(choice) - 1]
            df.at[index, "intent"] = selected_intent
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Saved: {selected_intent}")
            break

        print("Invalid choice. Enter a number from 1 to 11.")

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("LABELING COMPLETE")
print("=" * 70)
print(df["intent"].value_counts())
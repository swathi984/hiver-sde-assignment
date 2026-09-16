import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")
OUTPUT_PATH = Path("data/amazon_help_conversations.csv")

BRAND = "AmazonHelp"

print("Finding AmazonHelp conversations...")

# First pass: collect AmazonHelp tweet IDs
amazon_ids = set()

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["tweet_id", "author_id"],
    chunksize=100_000,
    low_memory=False
):
    brand_rows = chunk[chunk["author_id"] == BRAND]
    amazon_ids.update(brand_rows["tweet_id"].astype(str))

print("AmazonHelp tweets found:", len(amazon_ids))

# Second pass: collect AmazonHelp tweets + tweets
# that directly respond to AmazonHelp
parts = []

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=100_000,
    low_memory=False
):
    chunk["tweet_id_str"] = chunk["tweet_id"].astype(str)
    chunk["in_response_str"] = chunk["in_response_to_tweet_id"].fillna(-1).astype(int).astype(str)

    mask = (
        (chunk["author_id"] == BRAND)
        | (chunk["in_response_str"].isin(amazon_ids))
    )

    selected = chunk[mask].copy()

    if not selected.empty:
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

df.drop(columns=["tweet_id_str", "in_response_str"], inplace=True)

df.drop_duplicates(subset=["tweet_id"], inplace=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== AMAZONHELP CONVERSATION DATA =====")
print("Rows:", len(df))

print("\n===== INBOUND / OUTBOUND =====")
print(df["inbound"].value_counts())

print("\n===== AUTHORS =====")
print(df["author_id"].value_counts().head(10))

print("\nSaved to:")
print(OUTPUT_PATH)
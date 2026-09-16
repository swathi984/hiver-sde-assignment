import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/amazon_help_conversations.csv")
OUTPUT_PATH = Path("data/amazon_help_threads.csv")

print("Loading AmazonHelp conversations...")

df = pd.read_csv(INPUT_PATH)

# Make IDs consistent
df["tweet_id"] = df["tweet_id"].astype(str)

df["in_response_to_tweet_id"] = (
    df["in_response_to_tweet_id"]
    .fillna("")
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

print("Tweets loaded:", len(df))

# Create a lookup from tweet ID -> row
tweet_lookup = df.set_index("tweet_id")

# Build conversation/thread ID by following parent messages
def find_root(tweet_id):
    visited = set()
    current = tweet_id

    while current and current not in visited:
        visited.add(current)

        if current not in tweet_lookup.index:
            break

        parent = tweet_lookup.loc[current, "in_response_to_tweet_id"]

        if not parent or parent == "nan":
            break

        if parent not in tweet_lookup.index:
            break

        current = parent

    return current


print("Building conversation threads...")

df["thread_id"] = df["tweet_id"].apply(find_root)

# Sort conversations chronologically
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

df = df.sort_values(
    ["thread_id", "created_at"]
)

# Remove accidental duplicates
df = df.drop_duplicates(subset=["tweet_id"])

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== THREAD RESULTS =====")
print("Tweets:", len(df))
print("Unique threads:", df["thread_id"].nunique())

thread_sizes = df.groupby("thread_id").size()

print("\n===== THREAD SIZE DISTRIBUTION =====")
print(thread_sizes.describe())

print("\n===== EXAMPLE THREAD =====")

example_thread = thread_sizes.sort_values(ascending=False).index[0]

example = df[df["thread_id"] == example_thread]

for _, row in example.iterrows():
    speaker = "CUSTOMER" if row["inbound"] else "AMAZONHELP"

    print(f"\n[{speaker}]")
    print(row["text"])

print("\nSaved to:")
print(OUTPUT_PATH)
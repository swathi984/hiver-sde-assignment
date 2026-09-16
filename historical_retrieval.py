import ast
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/amazon_help_conversations.csv"

TOP_K = 3


# ============================================================
# LOAD DATA
# ============================================================

print("Loading AmazonHelp conversations...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["tweet_id", "text"])

print(f"Total rows: {len(df)}")


# ============================================================
# SEPARATE CUSTOMER AND AMAZONHELP MESSAGES
# ============================================================

customer_df = df[df["inbound"] == True].copy()
agent_df = df[df["inbound"] == False].copy()

print(f"Customer messages: {len(customer_df)}")
print(f"AmazonHelp messages: {len(agent_df)}")


# ============================================================
# BUILD RESPONSE LOOKUP
#
# AmazonHelp outbound messages contain the customer's
# tweet ID in `in_response_to_tweet_id`.
# ============================================================

agent_df["in_response_to_tweet_id"] = pd.to_numeric(
    agent_df["in_response_to_tweet_id"],
    errors="coerce"
)

response_lookup = (
    agent_df
    .dropna(subset=["in_response_to_tweet_id"])
    .groupby("in_response_to_tweet_id")["text"]
    .apply(list)
    .to_dict()
)


# ============================================================
# CREATE CUSTOMER -> RESPONSE PAIRS
# ============================================================

pairs = []

for _, row in customer_df.iterrows():

    tweet_id = row["tweet_id"]

    responses = response_lookup.get(tweet_id, [])

    if not responses:
        continue

    # Combine multiple agent messages into one historical resolution
    response_text = " ".join(
        str(response).strip()
        for response in responses
        if pd.notna(response)
    )

    customer_text = str(row["text"]).strip()

    if not customer_text or not response_text:
        continue

    pairs.append({
        "tweet_id": tweet_id,
        "customer_message": customer_text,
        "historical_response": response_text
    })


pairs_df = pd.DataFrame(pairs)

print(f"Customer-response pairs: {len(pairs_df)}")


# ============================================================
# SAVE RETRIEVAL DATASET
# ============================================================

OUTPUT_PATH = "data/golden/historical_response_pairs.csv"

pairs_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print(f"Saved pairs to: {OUTPUT_PATH}")


# ============================================================
# BUILD TF-IDF INDEX
# ============================================================

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

message_matrix = vectorizer.fit_transform(
    pairs_df["customer_message"]
)

print(
    f"TF-IDF matrix shape: {message_matrix.shape}"
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_similar_messages(query, top_k=TOP_K):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        message_matrix
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "similarity": float(similarities[index]),
            "customer_message": pairs_df.iloc[index]["customer_message"],
            "historical_response": pairs_df.iloc[index]["historical_response"]
        })

    return results


# ============================================================
# DEMO
# ============================================================

if len(pairs_df) > 0:

    test_message = (
        "My package is delayed and I still haven't received it."
    )

    print("\n===== RETRIEVAL DEMO =====")
    print(f"Customer message: {test_message}")

    results = retrieve_similar_messages(
        test_message,
        top_k=TOP_K
    )

    for i, result in enumerate(results, start=1):

        print(f"\n--- Match {i} ---")
        print(f"Similarity: {result['similarity']:.3f}")

        print(
            f"Historical customer message: "
            f"{result['customer_message']}"
        )

        print(
            f"Historical AmazonHelp response: "
            f"{result['historical_response']}"
        )
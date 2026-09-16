import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load historical customer-support pairs
df = pd.read_csv("data/processed/historical_pairs.csv")

# TF-IDF representation of historical customer messages
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000,
    stop_words="english"
)

matrix = vectorizer.fit_transform(
    df["customer_text"].astype(str)
)


def detect_delivery_subtype(text):
    """
    Identify important delivery-related patterns.
    This prevents opposite delivery situations from
    being treated as identical.
    """

    text = text.lower()

    if (
        ("not delivered" in text)
        or ("didn't deliver" in text)
        or ("did not deliver" in text)
        or ("hasn't arrived" in text)
        or ("has not arrived" in text)
        or ("not arrived" in text)
        or ("where is my package" in text)
        or ("where's my package" in text)
    ):
        return "not_received"

    if (
        ("stamped as delivered" in text)
        or ("marked as delivered" in text)
        or ("shows delivered" in text)
        or ("says delivered" in text)
    ):
        return "marked_delivered"

    if (
        ("late" in text)
        or ("delayed" in text)
        or ("delay" in text)
        or ("when will it arrive" in text)
        or ("when will it deliver" in text)
    ):
        return "delayed"

    return "general"


def compatible_delivery_case(query, historical_text):
    """
    Check whether two delivery messages describe
    compatible situations.
    """

    query_type = detect_delivery_subtype(query)
    historical_type = detect_delivery_subtype(historical_text)

    # Opposite situations should not be treated as a strong match.
    if query_type == "not_received" and historical_type == "marked_delivered":
        return False

    if query_type == "marked_delivered" and historical_type == "not_received":
        return False

    return True


def retrieve_similar(query, top_k=3):
    """
    Retrieve historically similar customer-support cases.

    Delivery cases are filtered to avoid obvious
    semantic conflicts such as:
    'not delivered' vs 'marked as delivered'.
    """

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    query_lower = query.lower()

    # Check whether this is clearly a delivery issue
    delivery_words = [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "arrive",
        "arrived",
        "late",
        "delay",
        "shipping"
    ]

    is_delivery_query = any(
        word in query_lower for word in delivery_words
    )

    candidates = []

    for index, similarity in enumerate(similarities):

        historical_text = str(
            df.iloc[index]["customer_text"]
        )

        # Avoid obvious delivery contradictions
        if is_delivery_query:

            if not compatible_delivery_case(
                query,
                historical_text
            ):
                continue

        candidates.append(
            (index, similarity)
        )

    # Sort by similarity
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for index, similarity in candidates[:top_k]:

        results.append({
            "customer_text": df.iloc[index]["customer_text"],
            "amazon_response": df.iloc[index]["amazon_response"],
            "similarity": float(similarity)
        })

    return results


if __name__ == "__main__":

    query = input("\nEnter a customer message: ")

    results = retrieve_similar(query)

    print("\n===== SIMILAR HISTORICAL CASES =====")

    for i, result in enumerate(results, 1):

        print(f"\n--- Result {i} ---")

        print(
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print("\nCustomer:")
        print(result["customer_text"])

        print("\nAmazon response:")
        print(result["amazon_response"])
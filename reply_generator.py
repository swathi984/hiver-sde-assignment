import re
from historical_retriever import retrieve_similar


def clean_response(response):
    """Clean an historical Amazon response."""

    response = str(response).strip()

    # Remove leading Twitter username
    response = re.sub(r"^@\w+\s*", "", response)

    # Remove agent initials/signature
    response = re.sub(r"\s*\^?[A-Za-z]{1,5}\s*$", "", response)

    return response.strip()


def is_low_confidence(customer_message, similarity):
    """
    Decide whether retrieval is reliable enough
    to directly use historical evidence.
    """

    text = customer_message.lower()

    # Very specific delivery problem
    delivery_problem = any(
        phrase in text
        for phrase in [
            "not delivered",
            "not arrived",
            "hasn't arrived",
            "has not arrived",
            "didn't arrive",
            "did not arrive",
            "where is my package",
            "where's my package"
        ]
    )

    # Require stronger evidence for these cases
    if delivery_problem and similarity < 0.65:
        return True

    # General low similarity
    if similarity < 0.30:
        return True

    return False


def generate_reply(customer_message, top_k=3):

    results = retrieve_similar(
        customer_message,
        top_k=top_k
    )

    if not results:

        return {
            "reply": (
                "I'm sorry you're having trouble. "
                "Please contact Amazon Customer Service "
                "so they can review your issue and help you further."
            ),
            "grounded": False,
            "similarity": 0.0,
            "evidence": []
        }

    best_match = results[0]

    similarity = best_match["similarity"]

    # Conservative response when retrieval is uncertain
    if is_low_confidence(
        customer_message,
        similarity
    ):

        return {
            "reply": (
                "I'm sorry you're having trouble with your delivery. "
                "Please contact Amazon Customer Service through phone "
                "or chat so they can check the latest delivery status "
                "of your order."
            ),
            "grounded": False,
            "similarity": similarity,
            "evidence": results
        }

    reply = clean_response(
        best_match["amazon_response"]
    )

    return {
        "reply": reply,
        "grounded": True,
        "similarity": similarity,
        "evidence": results
    }


if __name__ == "__main__":

    customer_message = input(
        "\nEnter customer message: "
    )

    result = generate_reply(customer_message)

    print("\n===== GENERATED REPLY =====")
    print(result["reply"])

    print("\n===== RETRIEVAL INFORMATION =====")
    print(
        f"Similarity: "
        f"{result['similarity']:.3f}"
    )

    print(
        f"Grounded: "
        f"{result['grounded']}"
    )

    print("\n===== SUPPORTING EVIDENCE =====")

    for i, evidence in enumerate(
        result["evidence"],
        1
    ):

        print(f"\n--- Evidence {i} ---")

        print(
            f"Similarity: "
            f"{evidence['similarity']:.3f}"
        )

        print(
            f"Customer: "
            f"{evidence['customer_text']}"
        )

        print(
            f"Historical response: "
            f"{evidence['amazon_response']}"
        )
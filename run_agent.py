from ml_classifier import predict_intent
from reply_generator import generate_reply
from escalation import decide_escalation


def run_agent(customer_message):
    # 1. Predict intent
    intent = predict_intent(customer_message)

    # 2. Generate grounded reply
    reply_result = generate_reply(customer_message)

    # 3. Decide auto-handle vs escalation
    decision = decide_escalation(
        intent=intent,
        similarity=reply_result["similarity"],
        grounded=reply_result["grounded"],
        customer_message=customer_message
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "reply": reply_result["reply"],
        "similarity": reply_result["similarity"],
        "grounded": reply_result["grounded"],
        "decision": decision["decision"],
        "reason": decision["reason"],
        "evidence": reply_result["evidence"]
    }


if __name__ == "__main__":

    customer_message = input(
        "\nEnter customer message: "
    )

    result = run_agent(customer_message)

    print("\n" + "=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 60)

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nIntent:")
    print(result["intent"])

    print("\nDraft Reply:")
    print(result["reply"])

    print("\nRetrieval Similarity:")
    print(f"{result['similarity']:.3f}")

    print("\nGrounded:")
    print(result["grounded"])

    print("\nDecision:")
    print(result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("\n" + "=" * 60)
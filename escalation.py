def decide_escalation(
    intent,
    similarity,
    grounded,
    customer_message
):
    """
    Decide whether the AI should auto-handle the case
    or escalate it to a human agent.
    """

    text = customer_message.lower()

    # 1. Explicit complaints or escalation requests
    if intent == "complaint_escalation":
        return {
            "decision": "ESCALATE",
            "reason": (
                "The customer explicitly indicates an unresolved "
                "complaint or requests escalation."
            )
        }

    # 2. Sensitive issues requiring account/order verification
    if intent in [
        "payment_issue",
        "refund_issue",
        "account_issue"
    ]:
        return {
            "decision": "ESCALATE",
            "reason": (
                "This issue may require account-specific or "
                "transaction-specific verification."
            )
        }

    # 3. Low retrieval confidence
    if not grounded:
        return {
            "decision": "ESCALATE",
            "reason": (
                "Historical evidence is not sufficiently similar "
                "to the customer's issue, so automated handling "
                "may be unreliable."
            )
        }

    # 4. Very low similarity
    if similarity < 0.30:
        return {
            "decision": "ESCALATE",
            "reason": (
                "The system found weak historical evidence "
                "for this issue."
            )
        }

    # 5. Very short or unclear messages
    if len(text.split()) < 3:
        return {
            "decision": "ESCALATE",
            "reason": (
                "There is not enough information in the customer "
                "message to safely resolve the issue."
            )
        }

    # Otherwise, allow automated handling
    return {
        "decision": "AUTO_HANDLE",
        "reason": (
            "The issue has sufficiently similar historical "
            "support evidence and does not require immediate "
            "human intervention."
        )
    }


if __name__ == "__main__":

    result = decide_escalation(
        intent="delivery_issue",
        similarity=0.75,
        grounded=True,
        customer_message="My package is delayed."
    )

    print("\n===== ESCALATION DECISION =====")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
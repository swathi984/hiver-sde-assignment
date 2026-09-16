import pandas as pd

OUTPUT_PATH = "data/decision_log.csv"

decisions = [
    {
        "decision": "Selected AmazonHelp as the target brand",
        "reason": "AmazonHelp had a large number of customer-support interactions, providing enough data for intent classification and historical response retrieval."
    },
    {
        "decision": "Used inbound customer tweets for intent labeling",
        "reason": "Inbound messages represent the customer requests that the support agent needs to classify."
    },
    {
        "decision": "Built conversation threads using response relationships",
        "reason": "Thread relationships provide context and allow customer messages to be connected with historical support responses."
    },
    {
        "decision": "Created a manually labeled golden set",
        "reason": "A hand-labeled evaluation set provides a fixed reference for measuring classifier performance."
    },
    {
        "decision": "Used a small set of interpretable support intents",
        "reason": "A limited intent taxonomy reduces ambiguity and makes the routing decisions easier to inspect."
    },
    {
        "decision": "Included an other_or_unclear intent",
        "reason": "Some customer tweets are too short, vague, or informational to map reliably to a specific support intent."
    },
    {
        "decision": "Used TF-IDF features for the baseline classifier",
        "reason": "TF-IDF is simple, fast, interpretable, and provides a strong classical NLP baseline."
    },
    {
        "decision": "Combined word and character TF-IDF features",
        "reason": "Character features can improve robustness to spelling variations, usernames, URLs, abbreviations, and noisy social-media text."
    },
    {
        "decision": "Used Linear SVM for the main intent classifier",
        "reason": "Linear SVM works well with sparse high-dimensional text features and is computationally efficient."
    },
    {
        "decision": "Used historical response retrieval before generation",
        "reason": "Retrieval grounds generated responses in previously observed customer-support resolutions instead of relying only on model knowledge."
    },
    {
        "decision": "Used top-3 historical matches",
        "reason": "Multiple retrieved examples provide more context while keeping the generation prompt relatively small."
    },
    {
        "decision": "Added an intent-confidence threshold",
        "reason": "Low-confidence classifications should not be automatically handled because incorrect routing can produce unsafe or irrelevant responses."
    },
    {
        "decision": "Added a retrieval-similarity threshold",
        "reason": "The agent should escalate when it cannot find sufficiently similar historical support evidence."
    },
    {
        "decision": "Escalated complaint and unclear cases",
        "reason": "Highly emotional or ambiguous cases are better routed to human support rather than handled automatically."
    },
    {
        "decision": "Evaluated generated replies using an LLM judge",
        "reason": "Intent accuracy alone does not measure whether the generated customer response is relevant, grounded, actionable, and appropriately written."
    }
]

df = pd.DataFrame(decisions)

df.insert(0, "decision_id", range(1, len(df) + 1))

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("DECISION LOG")
print("=" * 70)
print(f"Total decisions: {len(df)}")
print(f"Saved: {OUTPUT_PATH}")

for _, row in df.iterrows():
    print(f"\n{row['decision_id']}. {row['decision']}")
    print(f"   Reason: {row['reason']}")
import pandas as pd

INPUT_PATH = "data/golden/llm_judge_results.csv"
OUTPUT_PATH = "data/golden/failure_modes.csv"

df = pd.read_csv(INPUT_PATH)

failure_modes = [
    {
        "failure_mode": "Unsupported assumptions about customer context",
        "example_id": 184,
        "example": "Customer only said: '@AmazonHelp thank you'",
        "observed_failure": "Reply assumes the customer reported an issue.",
        "hypothesis": "The response generator may overgeneralize from historical support conversations instead of responding directly to the current message."
    },
    {
        "failure_mode": "Incorrectly assuming the issue is resolved",
        "example_id": 145,
        "example": "Customer says the issue must be fixed or they will leave.",
        "observed_failure": "Reply says: 'I'm glad it's fixed.'",
        "hypothesis": "The retrieval/generation pipeline may select historical resolutions without verifying whether the current customer message indicates resolution."
    },
    {
        "failure_mode": "Ignoring an important detail in the customer message",
        "example_id": 146,
        "example": "Customer specifically complains about the replacement phone being sent through the wrong courier.",
        "observed_failure": "Reply does not address the courier concern.",
        "hypothesis": "Similarity-based retrieval may find generally related delivery examples while missing the most important entity or constraint in the message."
    },
    {
        "failure_mode": "Weak or incomplete acknowledgement",
        "example_id": 154,
        "example": "Customer says the previous support team did not provide a proper solution.",
        "observed_failure": "Reply gives an email follow-up but does not directly acknowledge the unresolved issue.",
        "hypothesis": "Historical responses may prioritize routing customers to another support channel rather than explicitly acknowledging the customer's frustration."
    },
    {
        "failure_mode": "Over-reliance on historical response wording",
        "example_id": 171,
        "example": "Customer sends a positive thank-you message in Japanese.",
        "observed_failure": "The generated response contains a highly stylized historical-response-like tone and multiple emoticons.",
        "hypothesis": "Retrieval can reproduce stylistic patterns from historical tweets that are not always necessary for the current interaction."
    }
]

result = pd.DataFrame(failure_modes)
result.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("TOP 5 FAILURE MODES")
print("=" * 70)

for i, row in result.iterrows():
    print(f"\n{i + 1}. {row['failure_mode']}")
    print(f"   Example ID: {row['example_id']}")
    print(f"   Failure: {row['observed_failure']}")
    print(f"   Hypothesis: {row['hypothesis']}")

print()
print(f"Saved: {OUTPUT_PATH}")
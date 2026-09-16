import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/golden/reply_quality_evaluation_clean.csv"
OUTPUT_PATH = "data/golden/llm_judge_results.csv"

GEMINI_MODEL = "gemini-3.5-flash"

# Start with 5 examples because of the current API quota.
MAX_ROWS = 5


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Add GEMINI_API_KEY to your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

print("Loading reply-quality evaluation data...")

df = pd.read_csv(INPUT_PATH)

df = df.dropna(
    subset=[
        "customer_message",
        "reply"
    ]
)

df = df.head(MAX_ROWS).copy()

print(f"Examples to judge: {len(df)}")


# ============================================================
# LLM JUDGE
# ============================================================

def judge_reply(
    customer_message,
    gold_intent,
    predicted_intent,
    reply,
    similarity,
    grounded
):

    prompt = f"""
You are evaluating an AI customer-support reply.

Evaluate the reply using ONLY the information provided below.

CUSTOMER MESSAGE:
{customer_message}

GOLD INTENT:
{gold_intent}

PREDICTED INTENT:
{predicted_intent}

RETRIEVAL SIMILARITY:
{similarity}

SYSTEM GROUNDING FLAG:
{grounded}

AI-GENERATED REPLY:
{reply}


Evaluate the reply on these five dimensions.

1. relevance
Does the reply address the customer's actual message?

2. groundedness
Does the reply avoid unsupported claims and remain consistent
with the available information?

3. actionability
Does it provide a useful next step when one is appropriate?

4. tone
Is it professional, clear, concise, and appropriate for customer support?

5. overall
Considering all dimensions, how good is the response overall?

For each dimension, give a score from 1 to 5:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Also determine whether the reply contains an unsupported claim.

Return ONLY valid JSON in exactly this format:

{{
  "relevance": 1,
  "groundedness": 1,
  "actionability": 1,
  "tone": 1,
  "overall": 1,
  "unsupported_claim": false,
  "reason": "Short explanation of the judgment."
}}

Do not use markdown.
"""


    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        store=False
    )

    text = response.output_text.strip()

    # Remove accidental markdown code fences.
    text = text.replace("```json", "")
    text = text.replace("```", "").strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:

        print("Warning: Could not parse judge response:")
        print(text)

        result = {
            "relevance": None,
            "groundedness": None,
            "actionability": None,
            "tone": None,
            "overall": None,
            "unsupported_claim": None,
            "reason": text
        }

    return result


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

for index, row in df.iterrows():

    current_number = len(results) + 1

    print(
        f"Judging {current_number}/{len(df)} "
        f"(ID {row['id']})..."
    )

    try:

        judgment = judge_reply(
            customer_message=row["customer_message"],
            gold_intent=row.get(
                "gold_intent",
                ""
            ),
            predicted_intent=row.get(
                "predicted_intent",
                ""
            ),
            reply=row["reply"],
            similarity=row.get(
                "similarity",
                0
            ),
            grounded=row.get(
                "grounded",
                False
            )
        )

        results.append({
            "id": row["id"],

            "customer_message":
                row["customer_message"],

            "gold_intent":
                row.get(
                    "gold_intent",
                    ""
                ),

            "predicted_intent":
                row.get(
                    "predicted_intent",
                    ""
                ),

            "reply":
                row["reply"],

            "similarity":
                row.get(
                    "similarity",
                    0
                ),

            "grounded":
                row.get(
                    "grounded",
                    False
                ),

            "human_overall_score":
                row.get(
                    "human_overall_score",
                    ""
                ),

            "human_notes":
                row.get(
                    "human_notes",
                    ""
                ),

            "llm_relevance":
                judgment.get(
                    "relevance"
                ),

            "llm_groundedness":
                judgment.get(
                    "groundedness"
                ),

            "llm_actionability":
                judgment.get(
                    "actionability"
                ),

            "llm_tone":
                judgment.get(
                    "tone"
                ),

            "llm_overall":
                judgment.get(
                    "overall"
                ),

            "llm_unsupported_claim":
                judgment.get(
                    "unsupported_claim"
                ),

            "llm_reason":
                judgment.get(
                    "reason",
                    ""
                )
        })

        # Save after every successful example.
        # This prevents losing completed evaluations.
        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False
        )

    except Exception as e:

        print("\nGemini request stopped:")
        print(e)

        print(
            "\nSaving completed results..."
        )

        if results:

            pd.DataFrame(
                results
            ).to_csv(
                OUTPUT_PATH,
                index=False
            )

        break

    # Small delay between requests.
    time.sleep(1)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("LLM-AS-JUDGE COMPLETE")
print("=" * 70)

print(
    f"Successfully judged: "
    f"{len(results_df)}"
)

print(
    f"Saved: {OUTPUT_PATH}"
)


# ============================================================
# AVERAGE SCORES
# ============================================================

if len(results_df) > 0:

    print("\nAverage LLM scores:")

    score_columns = [
        "llm_relevance",
        "llm_groundedness",
        "llm_actionability",
        "llm_tone",
        "llm_overall"
    ]

    for column in score_columns:

        value = pd.to_numeric(
            results_df[column],
            errors="coerce"
        ).mean()

        print(
            f"{column}: {value:.2f}"
        )


# ============================================================
# HUMAN VS LLM AGREEMENT
# ============================================================

if len(results_df) > 0:

    comparison = results_df[
        [
            "human_overall_score",
            "llm_overall"
        ]
    ].copy()

    comparison["human_overall_score"] = pd.to_numeric(
        comparison["human_overall_score"],
        errors="coerce"
    )

    comparison["llm_overall"] = pd.to_numeric(
        comparison["llm_overall"],
        errors="coerce"
    )

    comparison = comparison.dropna()

    if len(comparison) > 0:

        exact_agreement = (
            comparison["human_overall_score"]
            ==
            comparison["llm_overall"]
        ).mean()

        print(
            "\nHuman vs LLM exact agreement: "
            f"{exact_agreement:.2%}"
        )

        mean_absolute_difference = (
            comparison[
                "human_overall_score"
            ]
            -
            comparison[
                "llm_overall"
            ]
        ).abs().mean()

        print(
            "Mean absolute score difference: "
            f"{mean_absolute_difference:.2f}"
        )
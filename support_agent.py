import os
import pandas as pd

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = "data/golden/golden_set_200.csv"
RETRIEVAL_PATH = "data/golden/historical_response_pairs.csv"

GEMINI_MODEL = "gemini-3.5-flash"

TOP_K = 3

# Conservative thresholds for automatic handling
MIN_INTENT_SCORE = 0.20
MIN_RETRIEVAL_SIMILARITY = 0.25


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# LOAD GOLDEN DATA
# ============================================================

print("Loading labeled data...")

golden_df = pd.read_csv(GOLDEN_PATH)

golden_df = golden_df.dropna(
    subset=["text", "intent"]
)

X = golden_df["text"]
y = golden_df["intent"]


# ============================================================
# TRAIN INTENT CLASSIFIER
# SAME SPLIT AS IMPROVED CLASSIFIER
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# WORD + CHARACTER TF-IDF + LINEAR SVM
# ============================================================

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
    )
])


classifier = Pipeline([
    ("features", features),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            C=1.0
        )
    )
])


print("Training intent classifier...")

classifier.fit(X_train, y_train)


# ============================================================
# LOAD HISTORICAL RESPONSE PAIRS
# ============================================================

print("Loading historical response pairs...")

history_df = pd.read_csv(RETRIEVAL_PATH)

history_df = history_df.dropna(
    subset=[
        "customer_message",
        "historical_response"
    ]
)

print(
    f"Historical response pairs: {len(history_df)}"
)


# ============================================================
# BUILD RETRIEVAL INDEX
# ============================================================

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    history_df["customer_message"]
)


# ============================================================
# RETRIEVE HISTORICAL RESPONSES
# ============================================================

def retrieve_similar_messages(
    query,
    top_k=TOP_K
):
    query_vector = retrieval_vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        retrieval_matrix
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "similarity": float(
                similarities[index]
            ),
            "customer_message":
                history_df.iloc[index][
                    "customer_message"
                ],
            "historical_response":
                history_df.iloc[index][
                    "historical_response"
                ]
        })

    return results


# ============================================================
# INTENT PREDICTION
# ============================================================

def predict_intent(text):

    predicted_intent = classifier.predict(
        [text]
    )[0]

    decision_scores = classifier.decision_function(
        [text]
    )[0]

    class_names = classifier.classes_

    score_map = dict(
        zip(class_names, decision_scores)
    )

    best_score = score_map[
        predicted_intent
    ]

    return predicted_intent, float(best_score)


# ============================================================
# ESCALATION DECISION
# ============================================================

def decide_handling(
    intent,
    intent_score,
    retrieval_similarity
):

    # Explicit complaints/escalations
    if intent == "complaint_escalation":
        return (
            "ESCALATE",
            "The message indicates an unresolved "
            "customer-support complaint or escalation."
        )

    # Ambiguous messages should not be automated
    if intent == "other_or_unclear":
        return (
            "ESCALATE",
            "The customer's intent is unclear, "
            "so automatic handling could be unsafe."
        )

    # Weak classifier confidence
    if intent_score < MIN_INTENT_SCORE:
        return (
            "ESCALATE",
            "Intent confidence is too low for "
            "automatic handling."
        )

    # Weak historical grounding
    if retrieval_similarity < MIN_RETRIEVAL_SIMILARITY:
        return (
            "ESCALATE",
            "No sufficiently similar historical "
            "resolution was found."
        )

    return (
        "AUTO-HANDLE",
        "The intent is sufficiently confident and "
        "a similar historical resolution is available."
    )


# ============================================================
# GEMINI GROUNDED RESPONSE
# ============================================================

def generate_reply(
    customer_message,
    intent,
    handling_decision,
    retrieved_results
):

    # Use only the strongest historical examples
    evidence = []

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):
        evidence.append(
            f"""
Historical example {i}:
Customer:
{result['customer_message']}

AmazonHelp response:
{result['historical_response']}

Similarity:
{result['similarity']:.3f}
"""
        )

    evidence_text = "\n".join(evidence)

    if handling_decision == "ESCALATE":

        instruction = """
The case should be escalated to a human agent.

Write a short acknowledgement that:
- recognizes the customer's issue,
- does not invent a resolution,
- does not promise something you cannot verify,
- tells the customer that further assistance is needed.
"""

    else:

        instruction = """
Write a concise customer-support response grounded
ONLY in the historical responses provided.

Do not invent policies, refunds, delivery dates,
credits, guarantees, or actions.

Use the historical responses as guidance, but
rewrite them naturally for the current customer.

Do not mention that you are using historical data.
"""

    prompt = f"""
You are an Amazon customer-support assistant.

Customer message:
{customer_message}

Predicted intent:
{intent}

Handling decision:
{handling_decision}

Historical support evidence:
{evidence_text}

{instruction}

Return only the proposed customer-facing reply.
"""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        store=False
    )

    return interaction.output_text.strip()


# ============================================================
# END-TO-END SUPPORT AGENT
# ============================================================

def support_agent(customer_message):

    # --------------------------------------------------------
    # 1. Predict intent
    # --------------------------------------------------------

    intent, intent_score = predict_intent(
        customer_message
    )

    # --------------------------------------------------------
    # 2. Retrieve historical examples
    # --------------------------------------------------------

    retrieved = retrieve_similar_messages(
        customer_message,
        TOP_K
    )

    best_similarity = (
        retrieved[0]["similarity"]
        if retrieved
        else 0.0
    )

    # --------------------------------------------------------
    # 3. Decide auto-handle or escalate
    # --------------------------------------------------------

    decision, reason = decide_handling(
        intent,
        intent_score,
        best_similarity
    )

    # --------------------------------------------------------
    # 4. Generate grounded reply
    # --------------------------------------------------------

    reply = generate_reply(
        customer_message,
        intent,
        decision,
        retrieved
    )

    # --------------------------------------------------------
    # 5. Return complete result
    # --------------------------------------------------------

    return {
        "customer_message": customer_message,
        "intent": intent,
        "intent_score": intent_score,
        "retrieval_similarity": best_similarity,
        "decision": decision,
        "reason": reason,
        "reply": reply,
        "retrieved_examples": retrieved
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)

    test_messages = [
        "My package is delayed and I still haven't received it.",
        "I want my money back. When will I get my refund?",
        "I have been contacting support for days and nobody is helping me.",
        "Thanks, that worked!"
    ]

    for message in test_messages:

        print("\n" + "-" * 70)
        print(f"Customer: {message}")

        result = support_agent(message)

        print(
            f"Intent: {result['intent']}"
        )

        print(
            f"Intent score: "
            f"{result['intent_score']:.3f}"
        )

        print(
            f"Retrieval similarity: "
            f"{result['retrieval_similarity']:.3f}"
        )

        print(
            f"Decision: {result['decision']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

        print(
            f"Reply: {result['reply']}"
        )
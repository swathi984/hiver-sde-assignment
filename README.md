# Hiver SDE Intern — AI Customer Support Agent

## 1. Problem Framing

This project builds an AI-assisted customer support agent using real-world customer-support conversations from Twitter.

The system focuses on three tasks:

1. **Intent classification** — identify the type of customer issue.
2. **Grounded reply generation** — retrieve relevant historical support responses and use them as evidence for drafting a reply.
3. **Handling decision** — decide whether the message can be auto-handled or should be escalated, with a reason.

### What "good" means

A good support agent should:

* identify the customer's intent correctly;
* use historical support interactions as grounding evidence;
* avoid unsupported assumptions;
* provide a useful and concise response;
* escalate ambiguous or sensitive cases instead of confidently answering incorrectly.

### What is not built

This project intentionally does not attempt to build:

* live Amazon order/account lookup;
* authentication or customer identity verification;
* a full production chatbot;
* real-time integration with Amazon support systems;
* payment processing or refund execution;
* production deployment and monitoring.

The scope is limited to intent routing, historical-response retrieval, grounded response generation, and conservative escalation.

---

## 2. Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle:

`thoughtvector/customer-support-on-twitter`

The dataset contains approximately 2.8 million tweets and includes fields such as:

* `tweet_id`
* `author_id`
* `inbound`
* `created_at`
* `text`
* `response_tweet_id`
* `in_response_to_tweet_id`

### Brand Selection

I selected **AmazonHelp** because it provided a large number of customer-support interactions.

The AmazonHelp subset contains approximately:

* 169,840 AmazonHelp tweets
* 270,344 tweets after combining relevant inbound and outbound conversation messages
* 98,875 conversation threads

The full dataset was not repeatedly processed during development. The project uses prepared AmazonHelp subsets and evaluation files to keep the pipeline reproducible and efficient.

---

## 3. Data Processing

The raw AmazonHelp data was processed into conversation threads using the response relationships available in the dataset.

Customer messages were separated from brand responses using the `inbound` field.

Historical customer-response pairs were then extracted and used as the retrieval knowledge base.

The resulting historical response dataset contains approximately **66,347 customer-response pairs**.

---

## 4. Intent Taxonomy

A small intent taxonomy was created based on recurring AmazonHelp customer-support topics.

The final intents are:

* `delivery_issue`
* `other_or_unclear`
* `complaint_escalation`
* `payment_issue`
* `order_issue`
* `account_issue`
* `seller_issue`
* `prime_issue`
* `refund_issue`
* `cancellation_issue`
* `return_issue`

`other_or_unclear` is used when a message is ambiguous or does not contain enough information to confidently assign a specific intent.

---

## 5. Golden Evaluation Set

A **200-example hand-labelled evaluation set** was created from AmazonHelp customer messages.

Each message was manually assigned one intent from the project's taxonomy.

Messages that were ambiguous, lacked enough context, or did not clearly fit a specific support category were assigned to `other_or_unclear`.

The golden set was kept separate from the historical response data used for retrieval.

File:

```text
data/golden/golden_set_200.csv
```

Columns:

```text
id
text
intent
```

### Golden Set Distribution

| Intent               | Examples |
| -------------------- | -------: |
| other_or_unclear     |       63 |
| delivery_issue       |       45 |
| complaint_escalation |       36 |
| payment_issue        |       17 |
| account_issue        |        9 |
| order_issue          |        7 |
| refund_issue         |        7 |
| prime_issue          |        6 |
| seller_issue         |        5 |
| return_issue         |        3 |
| cancellation_issue   |        2 |
| **Total**            |  **200** |

---

## 6. Classification Approach

The classifier combines:

* Word-level TF-IDF features
* Character-level TF-IDF features
* Linear Support Vector Machine

Word-level features capture meaningful support phrases, while character-level features help handle spelling variations, abbreviations, and noisy social-media text.

The classifier uses a stratified 80/20 evaluation split with a fixed random seed.

---

## 7. Classification Results

The final classifier was compared against simple baselines.

| Model                                    |  Accuracy |  Macro F1 | Weighted F1 |
| ---------------------------------------- | --------: | --------: | ----------: |
| Majority Class                           |     32.5% |     0.045 |       0.159 |
| Keyword / Rules                          |     55.0% |     0.371 |       0.476 |
| TF-IDF + Logistic Regression             |     52.5% |     0.326 |       0.525 |
| **Word + Character TF-IDF + Linear SVM** | **62.5%** | **0.382** |   **0.593** |

The final model achieved **62.5% accuracy** on the held-out intent-classification split.

Macro F1 is also reported because the intent distribution is imbalanced.

---

## 8. Historical Response Retrieval

After predicting the customer's intent, the system searches historical customer-support interactions for similar customer messages.

The retrieval process:

1. Converts historical customer messages into TF-IDF vectors.
2. Converts the incoming message into the same vector space.
3. Calculates cosine similarity.
4. Selects the top 3 historical matches.
5. Uses the retrieved responses as grounding evidence for reply generation.

File:

```text
data/golden/historical_response_pairs.csv
```

The system does not directly return a historical response as the final answer. Instead, the retrieved examples are provided as evidence to the response-generation step.

---

## 9. Response Generation

The response generator uses Gemini to draft a concise customer-support reply.

The prompt instructs the model to:

* use the retrieved historical responses as grounding;
* address the customer's actual message;
* avoid inventing customer details;
* avoid claiming an issue is resolved unless supported by the message;
* keep the response concise and professional.

The model is accessed through the Google Gemini SDK.

The API key is loaded from an environment variable rather than being hard-coded in the source code.

---

## 10. Auto-Handle vs Escalation

The system makes a conservative handling decision.

A message is escalated when:

* the predicted intent is `complaint_escalation`;
* the predicted intent is `other_or_unclear`;
* the intent confidence is below the configured threshold;
* historical retrieval similarity is too weak.

Otherwise, the system can mark the message as:

```text
AUTO-HANDLE
```

or:

```text
ESCALATE
```

The decision includes a reason so that the behavior is explainable.

Current thresholds:

```text
MIN_INTENT_SCORE = 0.20
MIN_RETRIEVAL_SIMILARITY = 0.25
TOP_K = 3
```

---

## 11. Reply Quality Evaluation

Generated replies were evaluated using an LLM-as-judge approach.

The judge evaluates:

* relevance;
* groundedness;
* actionability;
* tone;
* overall quality;
* unsupported claims.

The current exploratory judge evaluation contains **5 examples**.

### LLM Judge Results

| Metric        | Average Score |
| ------------- | ------------: |
| Relevance     |      3.00 / 5 |
| Groundedness  |      3.00 / 5 |
| Actionability |      3.00 / 5 |
| Tone          |      3.00 / 5 |
| Overall       |      2.80 / 5 |

The unsupported-claim rate in this 5-example sample was **60%**.

This number should **not** be interpreted as the overall unsupported-claim rate of the complete system because the judge sample is very small and exploratory.

Results are stored in:

```text
data/golden/llm_judge_results.csv
data/golden/evaluation_summary.csv
```

---

## 12. Human vs LLM Judge Agreement

To check whether the LLM judge's assessment was reasonably aligned with human review, 5 generated replies were manually reviewed using the same overall-quality and unsupported-claim criteria.

| Metric                        |   Result |
| ----------------------------- | -------: |
| Examples manually reviewed    |        5 |
| Overall score exact agreement |    60.0% |
| Mean absolute error           | 0.40 / 4 |
| Unsupported-claim agreement   |   100.0% |

The agreement check is an exploratory validation rather than a statistically representative study because only five examples were manually reviewed.

Files:

```text
data/golden/human_llm_agreement.csv
data/golden/judge_agreement_summary.csv
```

---

## 13. What Is Misleading About My Headline Number?

The **62.5% accuracy** headline refers only to the held-out **intent-classification task**.

It does not mean that 62.5% of complete customer-support interactions are automatically resolved correctly.

The complete system also includes:

* intent classification;
* historical retrieval;
* response generation;
* unsupported-claim avoidance;
* escalation decisions.

In addition, the golden set is imbalanced, with `other_or_unclear`, `delivery_issue`, and `complaint_escalation` representing a large portion of the examples.

Therefore, macro F1 and qualitative reply evaluation are also important when interpreting the result.

---

## 14. Top 5 Failure Modes

### 1. Unsupported assumptions about customer context

**Example — ID 184**

Customer:

```text
@AmazonHelp thank you
```

Generated reply assumed that the customer had reported an issue.

**Hypothesis:** The generator can overgeneralize from historical support conversations and insert an assumed support context.

---

### 2. Incorrectly assuming that an issue is resolved

**Example — ID 145**

Customer expressed strong dissatisfaction, but the generated reply said:

```text
I'm glad it's fixed.
```

The customer's message did not establish that the issue had been fixed.

**Hypothesis:** Retrieval can surface historical resolution language that the generator applies without verifying the current conversation state.

---

### 3. Ignoring an important customer detail

**Example — ID 146**

The customer specifically complained that a replacement phone was being sent through the wrong courier.

The generated reply did not address the courier issue.

**Hypothesis:** Similarity retrieval can find generally related delivery examples while missing important entities or constraints in the customer's message.

---

### 4. Weak or incomplete acknowledgement

**Example — ID 154**

The response provided an email follow-up but did not directly acknowledge the customer's complaint about not receiving a proper solution.

**Hypothesis:** Historical responses may prioritize routing customers to another support channel rather than explicitly acknowledging the problem.

---

### 5. Over-reliance on historical response style

**Example — ID 171**

A Japanese thank-you message received a highly stylized response containing multiple emoticons.

**Hypothesis:** Retrieval-based generation can reproduce stylistic patterns from historical responses even when a simpler response would be more appropriate.

---

## 15. What I Would Do With One More Week

If given another week, I would focus on the following improvements:

### 1. Improve the golden evaluation set

Increase the manually labelled set and ensure better coverage of minority intents and ambiguous cases.

### 2. Improve intent classification

Experiment with:

* better text normalization;
* class weighting;
* threshold calibration;
* stronger embedding-based classifiers;
* additional hard-negative examples.

### 3. Improve retrieval

Add:

* semantic embeddings;
* intent-aware retrieval;
* entity/keyword matching;
* reranking of retrieved examples.

This would help preserve important details such as order numbers, courier names, product names, and specific requests.

### 4. Add stronger safety checks

Before generating a response, detect claims such as:

* "your issue is fixed";
* "we received your details";
* "your refund has been processed".

These claims should only be generated when supported by the current conversation.

### 5. Expand evaluation

Increase the LLM-judge and human-review samples and compare multiple judge prompts/models to make the reply-quality evaluation more reliable.

---

## 16. Decision Log

The following non-obvious implementation decisions were recorded during development.

| #  | Decision                                         | Reason                                                          |
| -- | ------------------------------------------------ | --------------------------------------------------------------- |
| 1  | Selected AmazonHelp                              | Large number of available interactions                          |
| 2  | Used inbound customer tweets for intent labeling | Represents incoming customer requests                           |
| 3  | Built conversation threads                       | Preserves customer-support context                              |
| 4  | Created a manually labelled golden set           | Provides an independent evaluation set                          |
| 5  | Used a small intent taxonomy                     | Keeps routing practical and interpretable                       |
| 6  | Added `other_or_unclear`                         | Handles ambiguous/noisy messages conservatively                 |
| 7  | Used TF-IDF as a baseline                        | Simple and reproducible text baseline                           |
| 8  | Combined word + character TF-IDF                 | Helps with noisy social-media language                          |
| 9  | Used Linear SVM                                  | Strong baseline for sparse text classification                  |
| 10 | Retrieved historical responses before generation | Grounds generated replies in observed support behavior          |
| 11 | Retrieved top 3 examples                         | Provides multiple evidence candidates without excessive context |
| 12 | Added an intent-confidence threshold             | Prevents low-confidence automatic handling                      |
| 13 | Added a retrieval-similarity threshold           | Prevents generation from weak historical matches                |
| 14 | Escalated complaints and unclear cases           | Reduces risk from inappropriate automation                      |
| 15 | Added an LLM judge                               | Evaluates generated replies beyond classification accuracy      |

The complete decision log is available at:

```text
data/decision_log.csv
```

---

## 17. Project Structure

```text
hiver-sde-assignment/
│
├── data/
│   ├── golden/
│   │   ├── golden_set_200.csv
│   │   ├── historical_response_pairs.csv
│   │   ├── evaluation_results.csv
│   │   ├── evaluation_summary.csv
│   │   ├── reply_quality_evaluation.csv
│   │   ├── reply_quality_evaluation_clean.csv
│   │   ├── llm_judge_results.csv
│   │   ├── human_llm_agreement.csv
│   │   ├── judge_agreement_summary.csv
│   │   └── failure_modes.csv
│   │
│   └── amazon_help_conversations.csv
│
├── baseline_majority.py
├── baseline_keyword.py
├── baseline_classifier.py
├── improved_classifier.py
├── ml_classifier.py
├── evaluation_harness.py
├── evaluation_summary.py
├── llm_judge.py
├── agreement_check.py
├── failure_modes.py
├── decision_log.py
├── support_agent.py
├── run_agent.py
├── requirements.txt
├── README.md
└── .env
```

---

## 18. Reproduction

### Step 1 — Create and activate the virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3 — Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or expose the API key publicly.

### Step 4 — Run the evaluation

Run:

```powershell
python evaluation_harness.py
```

This evaluates the intent classifier against the prepared golden set and baselines.

### Step 5 — Run the support agent

```powershell
python support_agent.py
```

The program demonstrates:

* customer message;
* predicted intent;
* intent confidence;
* historical retrieval similarity;
* auto-handle/escalate decision;
* generated support response.

### Step 6 — Run the judge evaluation

```powershell
python llm_judge.py
```

The current configured judge evaluation uses a small sample to control API usage.

### Step 7 — Check human/LLM agreement

```powershell
python agreement_check.py
```

---

## 19. Headline Results

The main measured classification result is:

**62.5% intent-classification accuracy**

on the held-out evaluation split.

Additional results:

* Macro F1: **0.382**
* Weighted F1: **0.593**
* LLM judge overall reply score: **2.80 / 5** on a 5-example exploratory sample
* Human/LLM exact overall-score agreement: **60.0%** on 5 manually reviewed examples
* Human/LLM unsupported-claim agreement: **100.0%** on those 5 examples

These metrics measure different parts of the system and should not be interpreted as a single end-to-end accuracy number.

---

## 20. Limitations

The project has several limitations:

* The golden set contains 200 examples and is not representative of every possible customer-support message.
* Intent classes are imbalanced.
* The LLM reply-quality evaluation currently uses only 5 examples.
* The human-vs-LLM agreement check also uses only 5 examples.
* The system does not have access to live customer/account/order information.
* Historical Twitter responses may contain outdated, inconsistent, or stylistically noisy support behavior.
* The generated response can still make unsupported assumptions, as shown in the failure analysis.

The results should therefore be treated as an evaluation of a prototype support-agent pipeline rather than a production-ready customer-support system.

---

## 21. Conclusion

This project demonstrates an end-to-end prototype for converting noisy customer-support conversations into an AI-assisted support workflow.

The system combines:

```text
Customer Message
       ↓
Intent Classification
       ↓
Historical Response Retrieval
       ↓
Grounded Response Generation
       ↓
Safety / Confidence Checks
       ↓
AUTO-HANDLE or ESCALATE
```

The evaluation shows that the combined word- and character-level TF-IDF + Linear SVM classifier improves over the tested simple baselines on the held-out intent-classification split.

The failure analysis also shows that classification accuracy alone is not sufficient for evaluating a support agent. Grounding, response quality, unsupported claims, and conservative escalation are important parts of the overall system.

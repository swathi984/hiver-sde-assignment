import pandas as pd

INPUT_PATH = "data/golden/llm_judge_results.csv"
OUTPUT_PATH = "data/golden/evaluation_summary.csv"

df = pd.read_csv(INPUT_PATH)

summary = {
    "examples_judged": len(df),
    "avg_relevance": df["llm_relevance"].mean(),
    "avg_groundedness": df["llm_groundedness"].mean(),
    "avg_actionability": df["llm_actionability"].mean(),
    "avg_tone": df["llm_tone"].mean(),
    "avg_overall": df["llm_overall"].mean(),
    "unsupported_claim_rate": df["llm_unsupported_claim"].astype(str).str.lower().eq("true").mean(),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

for key, value in summary.items():
    if key == "examples_judged":
        print(f"{key}: {value}")
    elif key == "unsupported_claim_rate":
        print(f"{key}: {value:.2%}")
    else:
        print(f"{key}: {value:.2f}")

print()
print(f"Saved: {OUTPUT_PATH}")
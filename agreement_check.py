import pandas as pd

path = "data/golden/human_llm_agreement.csv"
df = pd.read_csv(path)

overall_agreement = (df["human_overall"] == df["llm_overall"]).mean() * 100
mae = (df["human_overall"] - df["llm_overall"]).abs().mean()
unsupported_agreement = (
    df["human_unsupported_claim"] == df["llm_unsupported_claim"]
).mean() * 100

print("=" * 70)
print("HUMAN vs LLM JUDGE AGREEMENT")
print("=" * 70)

print(f"Examples reviewed: {len(df)}")
print(f"Overall score exact agreement: {overall_agreement:.1f}%")
print(f"Mean absolute error: {mae:.2f} / 4")
print(f"Unsupported-claim agreement: {unsupported_agreement:.1f}%")

summary = pd.DataFrame([{
    "examples_reviewed": len(df),
    "overall_exact_agreement_pct": overall_agreement,
    "overall_mae": mae,
    "unsupported_claim_agreement_pct": unsupported_agreement
}])

summary.to_csv(
    "data/golden/judge_agreement_summary.csv",
    index=False
)

print("\nSaved: data/golden/judge_agreement_summary.csv")
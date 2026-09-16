import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=["author_id", "inbound"],
    low_memory=False
)

print("\n===== TOTAL ROWS =====")
print(len(df))

print("\n===== TOP AUTHOR ACCOUNTS =====")
print(df["author_id"].value_counts().head(30))

print("\n===== INBOUND / OUTBOUND =====")
print(df["inbound"].value_counts())
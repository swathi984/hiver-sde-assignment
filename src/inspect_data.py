import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")

print("Reading dataset sample...")

df = pd.read_csv(
    DATA_PATH,
    nrows=1000
)

print("\n===== SHAPE OF SAMPLE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head().to_string())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
import csv
import pandas as pd

INPUT_PATH = "data/golden/reply_quality_evaluation.csv"
OUTPUT_PATH = "data/golden/reply_quality_evaluation_clean.csv"

print("Reading original CSV...")

rows = []

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.reader(f)

    header = next(reader)

    print("Original header:")
    print(header)

    for row in reader:

        # Remove the unwanted trailing empty column
        if len(row) == 10 and row[-1] == "":
            row = row[:-1]

        if len(row) != 9:
            print(
                f"Skipping malformed row with "
                f"{len(row)} fields"
            )
            continue

        rows.append(row)


clean_df = pd.DataFrame(
    rows,
    columns=header
)

clean_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCSV repaired successfully.")
print(f"Rows: {len(clean_df)}")
print(f"Columns: {list(clean_df.columns)}")
print(f"Saved: {OUTPUT_PATH}")
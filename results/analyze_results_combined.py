# Analyze the results overall

import pandas as pd

# Load the CSV file
df = pd.read_csv("results/run_1/results_annotated.csv")  # CHANGE FOLDER NAME AS NEEDED


# Clean column names
df.columns = df.columns.str.strip()

# EXPECTED LANGUAGE
# Count 0s and 1s in expected_lang
lang_counts = df["expected_lang"].value_counts().sort_index()

print("-" * 40)
print("EXPECTED_LANG")
print("Counts")
print(lang_counts)

# Percentage breakdown
print("\nPercentage")
expected_lang_percent = (lang_counts / len(df) * 100).round(2)
print(expected_lang_percent)
print("-" * 40)
print()

# CUT OFF
# Count 0s and 1s in cut_off
cutoff_counts = df["cut_off"].value_counts().sort_index()

print("-" * 40)
print("CUT_OFF")
print("Counts")
print(cutoff_counts)

# Percentage breakdown
print("\nPercentage")
cutoff_percent = (cutoff_counts / len(df) * 100).round(2)
print(cutoff_percent)
print("-" * 40)
print()

score_columns = ["correctness", "fluency", "relevance", "helpfulness", "conciseness"]

for col in score_columns:
    print("-" * 40)
    print(col.upper())

    # Value counts
    print("Counts")
    print(df[col].value_counts().sort_index())

    # Mean and std
    mean = df[col].mean().round(2)
    std = df[col].std().round(2)
    print(f"\nMean: {mean}")
    print(f"Std: {std}")
    print("-" * 40)
    print()

# TOTAL SAMPLE COUNT
df["correctness"].value_counts()
print(f"Total samples: {len(df)}")  # should be 240

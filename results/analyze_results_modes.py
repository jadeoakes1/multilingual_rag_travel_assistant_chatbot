# Analyze the results based on chatbot mode

import pandas as pd

# Load the CSV file
df = pd.read_csv("results/run_1/results_annotated.csv")  # CHANGE FOLDER NAME AS NEEDED

# Clean column names
df.columns = df.columns.str.strip()

# List of score columns
score_columns = ["correctness", "fluency", "relevance", "helpfulness", "conciseness"]

# Modes to group by
modes = df["mode"].unique()

print("=" * 60)
print("OVERALL SUMMARY BY MODE")
print("=" * 60)

for mode in sorted(modes):
    print(f"\n### MODE: {mode.upper()} ###")
    mode_df = df[df["mode"] == mode]

    print("-" * 40)
    print("EXPECTED_LANG")
    lang_counts = mode_df["expected_lang"].value_counts().sort_index()
    print("Counts:")
    print(lang_counts)
    print("\nPercentage:")
    print((lang_counts / len(mode_df) * 100).round(2))
    print("-" * 40)

    print("CUT_OFF")
    cutoff_counts = mode_df["cut_off"].value_counts().sort_index()
    print("Counts:")
    print(cutoff_counts)
    print("\nPercentage:")
    print((cutoff_counts / len(mode_df) * 100).round(2))
    print("-" * 40)

    for col in score_columns:
        print(f"{col.upper()}")
        print("Counts:")
        print(mode_df[col].value_counts().sort_index())
        mean = mode_df[col].mean().round(2)
        std = mode_df[col].std().round(2)
        print(f"\nMean: {mean}")
        print(f"Std: {std}")
        print("-" * 40)

    print(f"Total samples in {mode}: {len(mode_df)}")
    print("=" * 60)

# Overall total
print(f"\nTotal samples across all modes: {len(df)}")

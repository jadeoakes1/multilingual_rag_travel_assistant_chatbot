# Analyze the results based on generation setting

import pandas as pd

# Load the CSV
df = pd.read_csv("results/run_1/results_annotated.csv")  # CHANGE FOLDER NAME AS NEEDED
df.columns = df.columns.str.strip()

# Settings to compare (update this list if needed)
settings = df["setting"].unique()
score_columns = ["correctness", "fluency", "relevance", "helpfulness", "conciseness"]

print("=" * 60)
print("SUMMARY BY GENERATION SETTING")
print("=" * 60)

for setting in sorted(settings):
    print(f"\n### SETTING: {setting.upper()} ###")
    setting_df = df[df["setting"] == setting]

    # Language expectation
    lang_counts = setting_df["expected_lang"].value_counts().sort_index()
    print("-" * 40)
    print("EXPECTED_LANG")
    print("Counts:")
    print(lang_counts)
    print("Percentage:")
    print((lang_counts / len(setting_df) * 100).round(2))

    # Cut-off rate
    cutoff_counts = setting_df["cut_off"].value_counts().sort_index()
    print("-" * 40)
    print("CUT_OFF")
    print("Counts:")
    print(cutoff_counts)
    print("Percentage:")
    print((cutoff_counts / len(setting_df) * 100).round(2))

    # Evaluation scores
    for col in score_columns:
        print("-" * 40)
        print(col.upper())
        print("Counts:")
        print(setting_df[col].value_counts().sort_index())
        print(f"Mean: {setting_df[col].mean().round(2)}")
        print(f"Std: {setting_df[col].std().round(2)}")

    print(f"Total samples: {len(setting_df)}")
    print("=" * 60)

# Total
print(f"\nTotal samples across all settings: {len(df)}")

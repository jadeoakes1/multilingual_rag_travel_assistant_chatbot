# Saves results from JSONL file to CSV file

import os
import json
import pandas as pd

# Set the directory containing JSONL results files
jsonl_dir = "results"
output_csv = "results/run_1/initial_results_summary.csv"  # CHANGE FOLDER NAME AS NEEDED

# Collect all JSONL files from the directory
jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]

# Parse each file and build rows for the dataframe
rows = []

for file in sorted(jsonl_files):
    file_path = os.path.join(jsonl_dir, file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            base_row = {
                "file": file,
                "query": entry.get("query", "").strip(),
                "answer": entry.get("final_answer", "").strip(),
                "setting": entry.get("setting", ""),
                "mode": entry.get("mode", ""),
                "language": entry.get("language", ""),
                "prompt_id": entry.get("prompt_id", ""),
                "context_count": len(entry.get("context_used", []))
            }

            # Add individual context columns
            context_list = entry.get("context_used", [])
            for i in range(5):
                base_row[f"context_{i+1}"] = context_list[i] if i < len(context_list) else ""

            rows.append(base_row)

# Create a dataframe and save it to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"Saved summary CSV to: {output_csv}")
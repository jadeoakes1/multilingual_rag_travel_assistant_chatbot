# Jade Oakes
# May 2, 2025
# Runs all experiments

import os
from run_prompt_experiments import run_prompt_experiments
from prompt_loader import load_prompt_templates

# Create output folder
os.makedirs("results/run_1", exist_ok=True)  # CHANGE FOLDER NAME FOR NEW RUN

# Load prompt templates
prompt_templates = load_prompt_templates("chatbot/prompt_templates.json")  # add _version# if needed

# How many lines = “safe” to skip (e.g. each experiment generates 10+ entries)
MIN_VALID_LINES = 10

# Run for all mode/prompt_id combinations
for mode, prompts in prompt_templates.items():
    for prompt_id in prompts:
        # Format filename
        safe_prompt_id = prompt_id.replace("/", "_")
        output_path = f"results/run_1/{mode}_{safe_prompt_id}.jsonl"  # CHANGE FOLDER NAME FOR NEW RUN

        # Skip if file exists and has at least MIN_VALID_LINES lines
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)
            if num_lines >= MIN_VALID_LINES:
                print(f"Skipping {mode} | {prompt_id} — {num_lines} lines already written")
                continue
            else:
                print(f"Found incomplete file for {mode} | {prompt_id} — only {num_lines} lines. Re-running...")

        print(f"\nRunning {mode} | {prompt_id}")
        run_prompt_experiments(mode, prompt_id, output_path)
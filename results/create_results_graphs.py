# Create graphs for the results

import matplotlib.pyplot as plt
import numpy as np

# BY MODE

# Define the data
modes = ["general", "travel", "no_retrieval"]
metrics = ["Correctness", "Fluency", "Relevance", "Helpfulness", "Conciseness"]

# Mean scores from evaluation
scores = {
    "Correctness": [2.7, 3.35, 2.41],
    "Fluency":     [4.85, 5.0, 4.88],
    "Relevance":   [3.05, 3.48, 2.62],
    "Helpfulness": [2.55, 3.32, 2.33],
    "Conciseness": [4.47, 4.72, 4.38]
}

# Color Universal Design palette (color-blind safe)
cud_colors = ["#E69F00", "#009E73", "#0072B2"]  # general, travel, no_retrieval

# Plot settings
x = np.arange(len(metrics))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each mode using CUD colors
for i, (mode, color) in enumerate(zip(modes, cud_colors)):
    values = [scores[metric][i] for metric in metrics]
    ax.bar(x + i * bar_width, values, width=bar_width, label=mode, color=color)

# Format the plot
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Mean Score")
ax.set_title("Evaluation Scores by Mode")
ax.set_xticks(x + bar_width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 5.5)
ax.legend(title="Mode")

plt.tight_layout()
plt.savefig("results/graphs/mode_comparison_bar_chart.png", dpi=300)
plt.show()

#####################################################################################

# CUD colors
colors = ["#E69F00", "#0072B2", "#009E73"]

# CUT-OFF
cut_off_percent = {
    "general": 22.5,
    "no_retrieval": 62.5,
    "travel": 35.0
}

modes = list(cut_off_percent.keys())
values = list(cut_off_percent.values())

plt.figure(figsize=(6, 4))
plt.bar(modes, values, color=colors)
plt.ylabel("Cut-Off Rate (%)")
plt.title("Cut-Off Rate by Mode")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("results/graphs/cutoff_rate_by_mode.png", dpi=300)
plt.show()

# EXPECTED LANGUAGE
expected_lang_percent = {
    "general": 65.0,
    "no_retrieval": 83.75,
    "travel": 100.0
}

modes = list(expected_lang_percent.keys())
values = list(expected_lang_percent.values())

plt.figure(figsize=(6, 4))
plt.bar(modes, values, color=colors)
plt.ylabel("Expected Language Match (%)")
plt.title("Expected Language Accuracy by Mode")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("results/graphs/expected_lang_match_by_mode.png", dpi=300)
plt.show()

######################################################################################

# BY GENERATION SETTING

# Generation settings
settings = ["Balanced", "Deterministic"]
metrics = ["Correctness", "Fluency", "Relevance", "Helpfulness", "Conciseness"]

# Mean scores for each setting
scores = {
    "Correctness":   [2.69, 2.54],
    "Fluency":       [4.92, 4.87],
    "Relevance":     [2.88, 2.79],
    "Helpfulness":   [2.63, 2.42],
    "Conciseness":   [4.46, 4.44],
}

# Color Universal Design (CUD) palette
colors = ["#E69F00", "#0072B2"]  # Balanced = orange, Deterministic = blue

# Plot setup
x = np.arange(len(metrics))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
for i, (setting, color) in enumerate(zip(settings, colors)):
    values = [scores[metric][i] for metric in metrics]
    ax.bar(x + i * bar_width, values, width=bar_width, label=setting, color=color)

# Formatting
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Mean Score")
ax.set_title("Evaluation Scores by Generation Setting")
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 5.5)
ax.legend(title="Generation Setting")

plt.tight_layout()
plt.savefig("results/graphs/generation_setting_comparison.png", dpi=300)
plt.show()

# Jade Oakes
# April 15, 2025
# Get stats from JSONL file on sentence length

import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Load tokenizer (multilingual)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

file_path = "data/sentence_pairs_metadata.jsonl"  # add _version# if needed

# Store token lengths
en_lengths = []
es_lengths = []

# Read and tokenize
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing lines"):
        item = json.loads(line)
        en_len = len(tokenizer.tokenize(item["en"]))
        es_len = len(tokenizer.tokenize(item["es"]))
        en_lengths.append(en_len)
        es_lengths.append(es_len)

def summarize_lengths(name, lengths):
    """Summarize lengths"""
    print(f"\n{name} Sentence Token Stats:")
    print(f"  Total: {len(lengths)}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median (50th %ile): {np.percentile(lengths, 50):.0f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")
    print(f"  99th percentile: {np.percentile(lengths, 99):.0f}")

# Print stats
summarize_lengths("EN", en_lengths)
summarize_lengths("ES", es_lengths)
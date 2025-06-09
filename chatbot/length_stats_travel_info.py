# Get stats from JSONL file on chunk text length

import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Load multilingual tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Path to chunked travel metadata
file_path = "data/chunked_travel_info_orig_data.jsonl"  # add _version# if needed

# Store token lengths and associated chunks
chunk_lengths = []
chunk_texts = []

# Read and tokenize
total_words = 0
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing chunks..."):
        item = json.loads(line)
        text = item.get("text", "")
        if text:
            # Count words
            total_words += len(text.split())

            # Tokenize and store length info
            token_len = len(tokenizer.tokenize(text))
            chunk_lengths.append(token_len)
            chunk_texts.append((token_len, text.strip()))

chunk_texts_sorted = sorted(chunk_texts, key=lambda x: x[0])
chunk_lengths_sorted = [length for length, _ in chunk_texts_sorted]

def summarize_lengths(name, lengths):
    """Summarize lengths"""
    print(f"\n{name} Chunk Token Stats:")
    print(f"  Total: {len(lengths)}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median (50th %ile): {np.percentile(lengths, 50):.0f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")
    print(f"  99th percentile: {np.percentile(lengths, 99):.0f}")

# Print stats
print(f"\nTotal words: {total_words}")

summarize_lengths("\nTravel chunk", chunk_lengths)

# Show examples
print("\nExample Chunks")

print("\nShortest Chunk:")
print(f"Tokens: {chunk_texts_sorted[0][0]}")
print(chunk_texts_sorted[0][1])

print("\nLongest Chunk:")
print(f"Tokens: {chunk_texts_sorted[-1][0]}")
print(chunk_texts_sorted[-1][1])

# Show median-ish chunk
mid_index = len(chunk_texts_sorted) // 2
print("\nMedian-ish Chunk:")
print(f"Tokens: {chunk_texts_sorted[mid_index][0]}")
print(chunk_texts_sorted[mid_index][1])
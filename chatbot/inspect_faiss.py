# Check FAISS vectors

import faiss
import json

# Load FAISS index
index_path = "data/rag_index.faiss"
print(f"Loading index from: {index_path}")
index = faiss.read_index(index_path)
print("FAISS index loaded")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)

# Load metadata
metadata_path = "data/rag_metadata.jsonl"
print(f"\nLoading metadata from: {metadata_path}")

metadata = []
with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

# Check that lengths match
assert len(metadata) == index.ntotal, "Metadata count does not match index vectors!"

print(f"Loaded {len(metadata)} metadata entries")

# Inspect a few entries
print("\nPreviewing 5 FAISS vectors and their metadata:\n")

vectors = index.reconstruct_n(0, 5)

for i in range(5):
    meta = metadata[i]
    print(f"  Index {i}")
    print(f"  Vector[:8]: {vectors[i][:8]}...")  # Show first 8 dimensions
    print(f"  EN: {meta['en']}")
    print(f"  ES: {meta['es']}")
    print(f"  Source: {meta['source']}\n")


# Jade Oakes
# April 21, 2025
# Use this to display samples of the FAISS files

import faiss
import numpy as np

# Path to the FAISS index (UNCOMMENT which one to test)
# FAISS_INDEX_FILE = "data/sentence_pairs_index.faiss"  # add _version# if needed
FAISS_INDEX_FILE = "data/chunked_travel_info_index.faiss"  # add _version# if needed

# Load the index
index = faiss.read_index(FAISS_INDEX_FILE)

# Basic info
print(f"Index loaded: {type(index)}")
print(f"Number of vectors: {index.ntotal}")
print(f"Vector dimension: {index.d}")

# Display the first 5 vectors
print("\nFirst 5 vectors (truncated to 5 dimensions):")
vectors = index.reconstruct_n(0, min(5, index.ntotal))
for i, vec in enumerate(vectors):
    print(f"Vector {i}: {np.round(vec[:5], 4)} ...")

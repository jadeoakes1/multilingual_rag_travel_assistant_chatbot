# Jade Oakes
# April 28, 2025
# Vector index for retrieval - embed and index data with FAISS

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Parameters
CHUNKED_FILE = "data/chunked_travel_info_orig_data.jsonl"  # add _version# if needed
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_FILE = "data/chunked_travel_info_index.faiss"  # add _version# if needed
METADATA_FILE = "data/chunked_travel_info_metadata.jsonl"  # add _version# if needed

# Load chunked records
print("Loading chunked records...")
chunked_records = []
with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading chunks"):
        chunked_records.append(json.loads(line))

# Load multilingual embedding model
print("Loading model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Encode all chunks
texts = [record["text"] for record in chunked_records]
print("Encoding texts...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# Create FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))

# Save FAISS index
os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
faiss.write_index(faiss_index, FAISS_INDEX_FILE)
print(f"Saved FAISS index to {FAISS_INDEX_FILE}")

# Save metadata to JSON file for easy lookups
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    for record in chunked_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved metadata to {METADATA_FILE}")
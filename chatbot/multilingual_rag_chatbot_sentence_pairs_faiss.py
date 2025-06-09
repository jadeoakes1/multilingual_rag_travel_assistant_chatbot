# Jade Oakes
# April 15, 2025
# Vector index for retrieval - embed and index data with FAISS
# Convert multilingual sentence pairs into vector embeddings,
# index them for fast similarity search, and store metadata for
# later use by the chatbot

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import torch

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')
    device = 'cuda'
    print("Using GPU for embedding")
else:
    device = 'cpu'
    print("GPU not available, using CPU")

# Load data
data_path = "data/combined_sentence_pairs_300k_each.en-es.jsonl"  # add _version# if needed
entries = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Loading data", unit='line', dynamic_ncols=True):
        entries.append(json.loads(line))

# Prepare texts to embed (English)
texts = [entry["en"] for entry in entries]

# Encode texts into dense vectors
print("Encoding texts...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=device
)

embeddings = np.array(embeddings).astype('float32')  # to be extra safe

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, 'data/rag_index.faiss')
print("Saved FAISS index")

# Save associated metadata (Spanish sentences + source)
with open('data/rag_metadata.jsonl', 'w', encoding='utf-8') as f:
    for entry in tqdm(entries, desc="Writing metadata", unit='entry', dynamic_ncols=True):
        f.write(json.dumps({
            "en": entry["en"],
            "es": entry["es"],
            "source": entry["source"]
        }, ensure_ascii=False) + "\n")
print("Saved metadata")

print(f"Indexed {len(embeddings)} entries with dimension {dimension}")
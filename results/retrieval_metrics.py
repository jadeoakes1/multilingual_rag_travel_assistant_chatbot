# Creates a histogram for the retrieval metrics

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv("results/results_annotated.csv")
df["answer"] = df["answer"].fillna("").str.strip()

# Load the SentenceTransformer model (MiniLM-L12)
model = SentenceTransformer("all-MiniLM-L12-v2")

# Identify context columns
context_cols = [col for col in df.columns if col.startswith("context_")]

# Compute max cosine similarity between answer and context passages
def max_similarity(row):
    answer = row["answer"]
    answer_emb = model.encode(answer, convert_to_tensor=True)
    sims = []
    for col in context_cols:
        ctx = str(row[col]).strip()
        if ctx and ctx.lower() != "nan":
            ctx_emb = model.encode(ctx, convert_to_tensor=True)
            sim = util.cos_sim(answer_emb, ctx_emb).item()
            sims.append(sim)
    return max(sims) if sims else 0

df["max_similarity"] = df.apply(max_similarity, axis=1)

# Show summary and histogram
print("Average max similarity:", df["max_similarity"].mean())
print("Min similarity:", df["max_similarity"].min())
print("Max similarity:", df["max_similarity"].max())

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df["max_similarity"], bins=20, color="cornflowerblue", edgecolor="black")
plt.title("Distribution of Max Semantic Similarity\n(Answer vs. Retrieved Context)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.show()

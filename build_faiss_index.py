import faiss
import pickle
import numpy as np

print("BUILDING FAISS INDEX FOR FAST SEARCH")

# Load embeddings
print("\n1. Loading embeddings...")
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    valid_indices = data['valid_indices']

print(f" Loaded {len(embeddings)} embeddings")
print(f"   Dimension: {embeddings.shape[1]}")

# Create FAISS index
print("\n2. Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add embeddings to index
index.add(embeddings.astype('float32'))

print(f" Added {index.ntotal} vectors to index")

# Save index
print("\n3. Saving FAISS index...")
faiss.write_index(index, "faiss_index.bin")
print(" Index saved to: faiss_index.bin")

# Test search
print("\n4. Testing search...")
distances, indices = index.search(embeddings[:1].astype('float32'), k=3)
print(f"Test search successful!")
print(f"   Top 3 similar indices: {indices[0]}")
print(f"   Distances: {distances[0]}")

print("\n" + "="*70)
print("RAG SYSTEM READY!")
print("NEXT STEP: Run 'python rag_analyzer.py'")

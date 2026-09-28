from sentence_transformers import SentenceTransformer
from PIL import Image
import json
import numpy as np
import pickle
import os

print("BUILDING IMAGE EMBEDDINGS FOR RAG")

# Load knowledge base
print("\n1. Loading knowledge base...")
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    kb_data = json.load(f)

print(f"Loaded {len(kb_data)} examples")

# Load CLIP embedding model
print("\n2. Loading CLIP embedding model...")
embed_model = SentenceTransformer('clip-ViT-B-32')
print("CLIP model loaded")

# Generate embeddings
print(f"\n3. Generating embeddings for {len(kb_data)} images...")

embeddings = []
valid_indices = []

for idx, example in enumerate(kb_data):
    image_path = example['image_path']
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Generate embedding
        embedding = embed_model.encode(image)
        embeddings.append(embedding)
        valid_indices.append(idx)
        
        print(f"   {idx+1}/{len(kb_data)}: {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"   {idx+1}/{len(kb_data)}: {os.path.basename(image_path)} - Error: {e}")

# Convert to numpy array
embeddings_array = np.array(embeddings)

# Save embeddings
print(f"\n4. Saving embeddings...")
with open("embeddings.pkl", "wb") as f:
    pickle.dump({
        'embeddings': embeddings_array,
        'valid_indices': valid_indices
    }, f)

print(f"Embeddings saved to: embeddings.pkl")
print(f"   Shape: {embeddings_array.shape}")
print(f"   Valid examples: {len(valid_indices)}/{len(kb_data)}")

print("NEXT STEP: Run 'python build_faiss_index.py'")

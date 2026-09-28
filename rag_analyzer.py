import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import json
import pickle
import numpy as np
import os
from qwen_vl_utils import process_vision_info

print("RAG-POWERED IMAGE ANALYZER")

# Load Qwen model
print("\n1. Loading Qwen model...")
model_path = "./qwen2-vl-2b-local"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)
print(" Qwen loaded")

# Load CLIP for embeddings
print("\n2. Loading CLIP model...")
embed_model = SentenceTransformer('clip-ViT-B-32')
print("CLIP loaded")

# Load knowledge base
print("\n3. Loading knowledge base...")
with open("knowledge_base.json", "r") as f:
    kb_data = json.load(f)
print(f" Knowledge base loaded ({len(kb_data)} examples)")

# Load FAISS index
print("\n4. Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")
print(f" FAISS index loaded ({index.ntotal} vectors)")

# Load embeddings metadata
with open("embeddings.pkl", "rb") as f:
    emb_data = pickle.load(f)
    valid_indices = emb_data['valid_indices']

print("RAG SYSTEM READY!")


def find_similar_examples(image_path, top_k=3):
    """Find similar images from knowledge base"""
    
    # Load and encode query image
    image = Image.open(image_path)
    query_embedding = embed_model.encode(image)
    
    # Search in FAISS
    distances, indices = index.search(
        np.array([query_embedding]).astype('float32'),
        k=top_k
    )
    
    # Get examples
    similar_examples = []
    for idx in indices[0]:
        kb_idx = valid_indices[idx]
        similar_examples.append(kb_data[kb_idx])
    
    return similar_examples


def analyze_with_rag(image_path, use_rag=True, top_k=3):
    """Analyze image using RAG"""
    
    print(f"\n {os.path.basename(image_path)}")
    print("="*70)
    
    # Load image
    image = Image.open(image_path)
    print(f"Loaded: {image.size[0]}x{image.size[1]}")
    
    results = {}
    
    # Find similar examples
    if use_rag:
        print(f"\n Finding {top_k} similar examples...")
        similar = find_similar_examples(image_path, top_k=top_k)
        
        for i, ex in enumerate(similar, 1):
            print(f"   {i}. {os.path.basename(ex['image_path'])} ({'FLAGGED' if ex['is_flagged'] else 'SAFE'})")
        
        results['similar_examples'] = similar
    else:
        similar = []
        results['similar_examples'] = []
    
    # Build prompt with examples
    print("\n Generating interpretation...")
    
    if similar:
        examples_text = "\n\n".join([
            f"Example {i}:\n{ex['interpretation']}"
            for i, ex in enumerate(similar, 1)
        ])
        
        interp_prompt = f"""Here are examples of similar images and their interpretations:

{examples_text}

Now analyze THIS image in the same thoughtful style. Provide a 2-3 short sentence description about deeper meaning, intent, emotion, and message."""
    else:
        interp_prompt = "Analyze this image for deeper meaning. Provide a 2-3 short sentence interpretation."
    
    # Generate interpretation
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": interp_prompt}
        ],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    interpretation = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    results['interpretation'] = interpretation
    
    # Check flagging based on similar examples
    if similar:
        flagged_count = sum(1 for ex in similar if ex['is_flagged'])
        # If majority of similar examples are flagged, flag this one too
        results['rag_suggested_flag'] = flagged_count >= (top_k / 2)
        results['flag_confidence'] = f"{flagged_count}/{top_k} similar examples flagged"
    else:
        results['rag_suggested_flag'] = False
        results['flag_confidence'] = "No RAG data"
    
    return results


def main():
    """Main function"""
    
    while True:
        print("\n\nOptions:")
        print("1. Analyze with RAG")
        print("2. Analyze without RAG (compare)")
        print("3. Exit")
        
        choice = input("\nChoice (1/2/3): ").strip()
        
        if choice == "1":
            img_path = input("\nImage path: ").strip().replace('"', '').replace("'", '')
            
            if not os.path.exists(img_path):
                print(" File not found")
                continue
            
            results = analyze_with_rag(img_path, use_rag=True, top_k=3)
            
            # Display
            print(" INTERPRETATION:")
            print(results['interpretation'])
            
            print("\n RAG FLAGGING SUGGESTION:")
            if results['rag_suggested_flag']:
                print(f" SUGGEST FLAG - {results['flag_confidence']}")
            else:
                print(f"SUGGEST SAFE - {results['flag_confidence']}")
        
        elif choice == "2":
            img_path = input("\nImage path: ").strip().replace('"', '').replace("'", '')
            
            if not os.path.exists(img_path):
                print(" File not found")
                continue
            
            results = analyze_with_rag(img_path, use_rag=False)
            
            print("INTERPRETATION (No RAG):")
            print(results['interpretation'])
        
        elif choice == "3":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
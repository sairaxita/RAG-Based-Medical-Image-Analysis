from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import os

print("Downloading Qwen2-VL-2B-Instruct model...")
print("size ~4GB ")
print("="*70)

model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Download processor (tokenizer + image processor)
print("\n1. Downloading processor...")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor downloaded")

# Download model weights
print("\n2. Downloading model")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # CPU mode
    device_map="cpu"
)
print(" Model downloaded")

# Save locally
print("\n3. Saving model locally...")
processor.save_pretrained("./qwen2-vl-2b-local")
model.save_pretrained("./qwen2-vl-2b-local")
print("Model saved to: ./qwen2-vl-2b-local")

print("\n" + "="*70)
print("DOWNLOAD COMPLETE!")
print("Model location:", os.path.abspath("./qwen2-vl-2b-local"))
print("="*70)
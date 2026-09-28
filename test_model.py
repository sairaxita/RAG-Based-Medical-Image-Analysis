from mcp_integration import save_to_google_drive, send_gmail_alert, ALERT_EMAIL
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from qwen_vl_utils import process_vision_info


# Load model
model_path = "./qwen2-vl-2b-local"

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Model loaded!\n")

# Sensitive keywords
SENSITIVE_KEYWORDS = [
    'blood', 'bleeding', 'bloody', 'bloodied',
    'wound', 'wounds', 'wounded',
    'cut', 'cuts', 'cutting', 'sliced',
    'injury', 'injured', 'injuries',
    'flesh', 'skin tear', 'laceration',
    'bone', 'fracture', 'broken',
    'gore', 'graphic', 'gruesome',
    'surgery', 'surgical', 'operation',
    'trauma', 'severe', 'deep cut',
    'open wound', 'gash', 'amputation'
]


def check_keywords(text):
    """Check for sensitive keywords"""
    text_lower = text.lower()
    matched = [kw for kw in SENSITIVE_KEYWORDS if kw in text_lower]
    return len(matched) > 0, list(set(matched))


def auto_analyze(image_path):
    """
    Automatically analyze image for BOTH:
    1. Human-like interpretation
    2. Sensitive content flagging
    """
    
    print(f"📷 {os.path.basename(image_path)}")
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"Loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    print("Processing\n")
    
    results = {}
    
    # ─────────────────────────────────────────────────────────────
    # PART 1: HUMAN-LIKE INTERPRETATION
    # ─────────────────────────────────────────────────────────────
    try:
        print("Getting interpretation")
        
        interp_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": """Analyze this image like a human would. Just describe what you see in simple language.

 Provide a 2-3 short sentence description  that captures:

- The intent or purpose of this image
- The emotional message or feeling it conveys
- What the creator wants the viewer to understand

Think deeply and interpret like a human. The text should be like human written"""}
            ],
        }]
        
        text = processor.apply_chat_template(interp_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(interp_messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        interpretation = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        results['interpretation'] = interpretation
        
    except Exception as e:
        print(f" Interpretation failed: {e}")
        results['interpretation'] = None
    
    # ─────────────────────────────────────────────────────────────
    # PART 2: FLAGGING CHECK
    # ─────────────────────────────────────────────────────────────
    try:
        print("Checking for sensitive content...")
        
        flag_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe what you see. Be specific about any visible injuries, wounds, blood, cuts, or medical conditions and the exact location of them on the human body. If there are none, just describe normally."}
            ],
        }]
        
        text = processor.apply_chat_template(flag_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(flag_messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        description = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        # Check keywords
        is_flagged, keywords = check_keywords(description)
        
        results['flagged'] = is_flagged
        results['keywords'] = keywords
        results['description'] = description
        
    except Exception as e:
        print(f"  Flagging check failed: {e}")
        results['flagged'] = False
        results['keywords'] = []
        results['description'] = None
    
    return results


def display_results(image_path, results):
    """Display results in nice format"""
    
  
    
    # Interpretation
    if results.get('interpretation'):
        print("HUMAN-LIKE INTERPRETATION:")
        print(results['interpretation'])
    
    # Flagging status
    print("CONTENT FLAGGING:")
    
    if results.get('flagged'):
        print(" STATUS: FLAGGED - SENSITIVE CONTENT DETECTED")
        print(f" Keywords: {', '.join(results['keywords'])}")
       ## print(f" Details: {results['description'][:100]}...")
    else:
        print("STATUS: SAFE - No sensitive content detected")
    


def main():
    """Main function"""
    
    print("AUTOMATIC INTERPRETATION + FLAGGING SYSTEM")
    print("\nThis system automatically:")
    print("  1. Interprets images for deeper meaning")
    print("  2. Flags sensitive medical/injury content")
    
    while True:
        print("\n\nOptions:")
        print("1. Analyze single image")
        print("2. Analyze multiple images (batch)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            # Single image
            image_path = input("\nEnter image path (or drag & drop): ").strip()
            image_path = image_path.replace('"', '').replace("'", '')
            
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            
            # Analyze
            results = auto_analyze(image_path)
            
            if results:
                display_results(image_path, results)

                # ── MCP: Save to Google Drive ──────────────────────
                image_name = os.path.basename(image_path)
                save_to_google_drive(
                    image_name=image_name,
                    interpretation=results.get('interpretation', 'N/A'),
                    flag_status=" FLAGGED" if results.get('flagged') else "SAFE",
                    keywords=results.get('keywords', [])
                )

                # ── MCP: Gmail alert if flagged ────────────────────
                if results.get('flagged'):
                    print(f"\n Flagged image detected — creating Gmail alert draft...")
                    send_gmail_alert(
                        image_name=image_name,
                        interpretation=results.get('interpretation', 'N/A'),
                        keywords=results.get('keywords', [])
                    )
        
        elif choice == "2":
            # Batch
            folder_path = input("\nEnter folder path: ").strip()
            folder_path = folder_path.replace('"', '').replace("'", '')
            
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue
            
            # Find images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            image_files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_files:
                print("No images found")
                continue
            
            print(f"\nFound {len(image_files)} images")
            print("Starting batch analysis...\n")
            
            # Track
            all_results = []
            flagged_count = 0
            
            for idx, img_path in enumerate(image_files, 1):
                print(f"IMAGE {idx}/{len(image_files)}")
                
                results = auto_analyze(img_path)
                
                if results:
                    display_results(img_path, results)

                    # ── MCP: Save to Google Drive ──────────────────
                    image_name = os.path.basename(img_path)
                    save_to_google_drive(
                        image_name=image_name,
                        interpretation=results.get('interpretation', 'N/A'),
                        flag_status="FLAGGED" if results.get('flagged') else " SAFE",
                        keywords=results.get('keywords', [])
                    )

                    # ── MCP: Gmail alert if flagged ────────────────
                    if results.get('flagged'):
                        print(f"\nFlagged image detected — creating Gmail alert draft...")
                        send_gmail_alert(
                            image_name=image_name,
                            interpretation=results.get('interpretation', 'N/A'),
                            keywords=results.get('keywords', [])
                        )

                    all_results.append({
                        'filename': os.path.basename(img_path),
                        'path': img_path,
                        'results': results
                    })
                    
                    if results.get('flagged'):
                        flagged_count += 1
                
                # Pause between images
                if idx < len(image_files):
                    input("\n  Press Enter for next image...")
            
            # Summary
            print("\n\n" + "╔" + "═"*68 + "╗")
            print("║" + " "*22 + "BATCH SUMMARY" + " "*33 + "║")
            print("╚" + "═"*68 + "╝")
            print(f"\nTotal analyzed: {len(all_results)}")
            print(f"Flagged: {flagged_count}")
            print(f"Safe: {len(all_results) - flagged_count}")
            
            if flagged_count > 0:
                print("\n  FLAGGED IMAGES:")
                print("-"*70)
                for item in all_results:
                    if item['results'].get('flagged'):
                        print(f"  📷 {item['filename']}")
                        print(f"     Keywords: {', '.join(item['results']['keywords'])}")
                print("-"*70)
        
        elif choice == "3":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice")


if __name__ == "__main__":
    main()


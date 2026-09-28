# RAG-Based Medical Image Analysis System

A locally deployable, privacy-preserving medical image analysis pipeline that combines a vision-language model with Retrieval-Augmented Generation (RAG) to detect and flag sensitive medical content in images — with no dependency on external AI APIs.

Built during an internship at **Nushift Technologies** under the Nushift Connect Programme.

---

## What It Does

- Takes a medical image as input
- Retrieves visually similar reference cases from a curated knowledge base using CLIP embeddings and FAISS
- Passes the image and retrieved context to **Qwen2-VL-2B-Instruct** (a vision-language model) for analysis
- Produces two outputs:
  - A natural-language interpretation of the image
  - A structured clinical flag across 6 categories: Blood, Open Wound, Fracture/Amputation, Graphic Trauma, Bruising, and a composite Clinical Flag
- Uploads analysis reports to **Google Drive** for audit trail
- Creates **Gmail alert drafts** for flagged images

---

## How It Works

```
Input Image
     │
     ▼
CLIP ViT-B-32 Embedding
     │
     ▼
FAISS Nearest-Neighbor Search (top 3 similar cases)
     │
     ▼
Qwen2-VL-2B-Instruct (Two-Pass Inference)
     ├── Pass 1: Natural language interpretation
     └── Pass 2: Structured clinical flag (yes/no checklist)
     │
     ▼
Safety Logic (Parse-error escalation + RAG neighbor override)
     │
     ▼
Google Drive Upload + Gmail Alert Draft (if flagged)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vision-Language Model | Qwen2-VL-2B-Instruct |
| Image Embedding | CLIP ViT-B-32 (SentenceTransformers) |
| Vector Search | FAISS (IndexFlatL2) |
| Cloud Storage | Google Drive API v3 |
| Alerting | Gmail API v1 |
| Deep Learning | PyTorch 2.12.1 |
| Model Loading | Hugging Face Transformers 5.12.1 |

---

## Project Structure

```
├── rag_analyzer.py          # Main entry point — full RAG pipeline with CLI menu
├── test_model.py            # Baseline analyzer without RAG
├── utils.py                 # Clinical flagging and parsing logic
├── mcp_integration.py       # Google Drive and Gmail integration
├── build_knowledge_base.py  # Build the knowledge base JSON
├── build_embeddings.py      # Generate CLIP embeddings for knowledge base
├── build_faiss_index.py     # Build FAISS index from embeddings
├── eval_harness.py          # Evaluation harness for accuracy measurement
├── model_download.py        # Download Qwen2-VL model from Hugging Face
├── kb_images/               # Knowledge base reference images
├── knowledge_base.json      # Curated labeled examples (44 entries)
├── requirements.txt         # Python dependencies
└── .gitignore
```


## Safety Design

The system is built to be **conservative** — it would rather flag something for review than miss it.

- **Structured parsing** — extracts yes/no answers from the model's clinical checklist
- **Keyword fallback** — scans for 14 sensitive medical terms if structured parsing fails
- **Parse-error escalation** — defaults to FLAGGED (not safe) if the model output is unreadable
- **RAG neighbor override** — if the model says SAFE but all 3 nearest knowledge base neighbors are FLAGGED, the system overrides and escalates for human review

---

## Author

**Kovvali Sai Raxita** (22BCE8965)  
B.Tech Computer Science and Engineering (Data Analytics)  
VIT-AP University  
Internship at Nushift Technologies, 2026

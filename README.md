# 📄 Smart Contract Assistant

> An end-to-end **Retrieval-Augmented Generation (RAG)** application for analysing legal contracts, insurance policies, and long-form agreements — built with LangChain, FAISS, FastAPI, and Gradio.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **Document Upload** | Supports PDF, DOCX, TXT — with PyMuPDF + pdfplumber fallback |
| 🔢 **Smart Chunking** | Contract-aware `RecursiveCharacterTextSplitter` with clause-level separators |
| 🗄️ **Vector Store** | FAISS (default, in-process) or Chroma (persistent) |
| 🧠 **Multi-Provider LLM** | OpenAI · NVIDIA NIM · HuggingFace (local, offline) |
| 🔍 **Semantic Retrieval** | Top-k similarity search + `LongContextReorder` |
| 💬 **Conversational Chat** | Full message history, streamed responses |
| 📎 **Source Citations** | Every answer references exact page numbers |
| 📝 **Auto-Summary** | Structured 7-section contract summary on demand |
| 🛡️ **Guardrails** | Embedding-based relevance check + harmful content filter |
| ⚖️ **Legal Disclaimer** | Appended to every answer automatically |
| 📊 **LLM-as-a-Judge** | Synthetic QA evaluation with preference score |
| 🌐 **REST API** | FastAPI with `/ingest`, `/chat`, `/summarise`, `/evaluate` |
| 🔌 **LangServe** | Optional `/retriever` and `/generator` endpoints |

---

## 🏗️ Architecture

```
PDF/DOCX/TXT Upload
        │
        ▼
┌─────────────────────────────────────────────┐
│           Ingestion Pipeline                │
│  loader.py → chunker.py → embedder.py       │
│  (load)      (split)       (embed + store)  │
└─────────────────────┬───────────────────────┘
                       │
                       ▼ FAISS / Chroma
┌─────────────────────────────────────────────┐
│           Query Processing                  │
│                                             │
│  1. Relevance Guardrail (cosine sim check)  │
│  2. Semantic Retriever (top-k chunks)       │
│  3. LongContextReorder (best-first edges)   │
│  4. LLM Generator (streamed, with history)  │
│  5. Source Citations + Legal Disclaimer     │
└─────────────────────┬───────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│           Evaluation Pipeline               │
│  1. Sample random chunks                   │
│  2. Synthetic QA generation                │
│  3. RAG answer generation                  │
│  4. LLM-as-a-Judge pairwise scoring        │
│  5. Preference score (% [2] votes)          │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/smart-contract-assistant.git
cd smart-contract-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API key
```

Minimum required setting:
```env
# Option A – OpenAI (recommended)
LLM_PROVIDER=openai
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Option B – Fully offline (no API key needed)
LLM_PROVIDER=huggingface
EMBED_PROVIDER=sentence_transformers
HF_MODEL_ID=google/flan-t5-base
ST_MODEL_NAME=all-MiniLM-L6-v2
```

### 3. Launch

**Option A – Gradio UI only (recommended for demos)**
```bash
python run_ui.py
# Opens at http://localhost:7860
```

**Option B – FastAPI server only**
```bash
python run_server.py
# API docs at http://localhost:8000/docs
```

**Option C – Both simultaneously**
```bash
# Terminal 1
python run_server.py

# Terminal 2
python run_ui.py --port 7861
```

---

## 📁 Project Structure

```
smart-contract-assistant/
│
├── app/
│   ├── config.py                 # All settings (env-configurable)
│   ├── main.py                   # FastAPI app + all REST endpoints
│   │
│   ├── ingestion/
│   │   ├── loader.py             # PDF/DOCX/TXT document loading
│   │   ├── chunker.py            # Contract-aware text splitting
│   │   └── embedder.py           # Embedding creation & vector store management
│   │
│   ├── retrieval/
│   │   └── rag_chain.py          # Full RAG pipeline (retrieval + generation)
│   │
│   ├── guardrails/
│   │   └── safety.py             # Relevance check + harmful content filter
│   │
│   ├── evaluation/
│   │   └── evaluator.py          # LLM-as-a-Judge evaluation pipeline
│   │
│   └── ui/
│       └── gradio_app.py         # 5-tab Gradio interface
│
├── tests/
│   └── test_ingestion.py         # pytest unit tests
│
├── notebooks/                    # Jupyter exploration notebooks (optional)
├── docstore_index/               # Generated vector store (gitignored)
├── run_ui.py                     # Launch Gradio standalone
├── run_server.py                 # Launch FastAPI standalone
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🌐 API Reference

After starting `run_server.py`, visit `http://localhost:8000/docs` for the interactive Swagger UI.

### `GET /health`
Returns service status and current configuration.

### `POST /ingest`
Upload a contract document.
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/contract.pdf"
```

### `POST /chat`
Ask a question about the ingested document.
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the payment terms?",
    "history": [],
    "add_disclaimer": true
  }'
```

### `POST /summarise`
Generate a structured summary.
```bash
curl -X POST http://localhost:8000/summarise
```

### `POST /evaluate`
Run the LLM-as-a-Judge evaluation.
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"num_questions": 5}'
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration Reference

All settings can be set in `.env` or as environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `nvidia` / `huggingface` |
| `OPENAI_API_KEY` | – | Required for OpenAI provider |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `NVIDIA_API_KEY` | – | Required for NVIDIA NIM |
| `NVIDIA_LLM_MODEL` | `meta/llama-3.1-8b-instruct` | NVIDIA LLM |
| `NVIDIA_EMBED_MODEL` | `nvidia/llama-3.2-nv-embedqa-1b-v2` | NVIDIA embedder |
| `EMBED_PROVIDER` | `openai` | `openai` / `nvidia` / `sentence_transformers` |
| `ST_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` / `chroma` |
| `VECTOR_STORE_PATH` | `docstore_index` | Where to save/load the index |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Top-k chunks to retrieve |
| `REORDER_LONG_CONTEXT` | `true` | Apply LongContextReorder |
| `GUARDRAIL_THRESHOLD` | `0.25` | Min cosine similarity to allow query |
| `EVAL_NUM_QUESTIONS` | `5` | Synthetic questions for evaluation |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | FastAPI port |
| `GRADIO_PORT` | `7860` | Gradio UI port |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload size |

---

## 🔌 NVIDIA NIM Setup

To use NVIDIA's LLM and embedding models (as used in the NVIDIA DLI course):

```env
LLM_PROVIDER=nvidia
EMBED_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_LLM_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
```

And install the additional dependency:
```bash
pip install langchain-nvidia-ai-endpoints
```

---

## 📊 Evaluation Methodology

The evaluation implements the **LLM-as-a-Judge pairwise scoring** pattern:

1. **Sample** two random document chunks from the vector store
2. **Generate** a synthetic ground-truth Q&A pair using those chunks
3. **Query** the RAG pipeline with the same question
4. **Judge** both answers with an LLM, scoring:
   - `[1]` = RAG answer is worse / inconsistent
   - `[2]` = RAG answer is better or equivalent
5. **Aggregate** into a **Preference Score** (% of `[2]` votes)

A score ≥ 0.8 indicates the pipeline performs excellently on the document.

---

## ⚠️ Legal Disclaimer

This tool is for **informational and educational purposes only**. It does not constitute legal advice. Always consult a qualified legal professional before acting on any information extracted from a contract.

---

## 📝 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path


class Settings:
    # ── LLM ────────────────────────────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_llm_model: str = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct")
    nvidia_embed_model: str = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    hf_model_id: str = os.getenv("HF_MODEL_ID", "google/flan-t5-base")

    # ── Embeddings ─────────────────────────────────────────────────────────
    embed_provider: str = os.getenv("EMBED_PROVIDER", "sentence_transformers")
    st_model_name: str = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")

    # ── Vector store ───────────────────────────────────────────────────────
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "docstore_index")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "contracts")

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", 5))
    reorder_long_context: bool = os.getenv("REORDER_LONG_CONTEXT", "true").lower() == "true"

    # ── Guard-rails ────────────────────────────────────────────────────────
    guardrail_similarity_threshold: float = float(os.getenv("GUARDRAIL_THRESHOLD", 0.25))
    off_topic_response: str = (
        "⚠️ I'm a Smart Contract Assistant. I can only answer questions "
        "related to the uploaded document. Please upload a contract and "
        "ask questions about its contents."
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_num_questions: int = int(os.getenv("EVAL_NUM_QUESTIONS", 5))

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    gradio_port: int = int(os.getenv("GRADIO_PORT", 7860))

    # ── Misc ───────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", 20))


settings = Settings()
BASE_DIR = Path(__file__).resolve().parent.parent

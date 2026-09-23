"""Application settings, loaded from the .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Gemini LLM
    gemini_api_key: str
    llm_model: str = "gemini-3.6-flash"

    # Gemini embeddings
    embed_model: str = "gemini-embedding-001"
    embed_batch_size: int = 10          # chunks per embedding request
    embed_pause_seconds: float = 1.0    # pause between batches

    # storage
    chroma_path: str = "data/chroma"

    # arXiv / ranking
    arxiv_max_results: int = 10         # candidates fetched per search
    top_k_papers: int = 3               # shortlist size from the topic explorer

    # chunking and retrieval
    chunk_size: int = 700               # tokens per chunk
    chunk_overlap: int = 100            # tokens shared between neighbouring chunks
    retrieval_top_k: int = 5            # chunks returned per question
    similarity_cutoff: float = 0.70     # tuned in Stage 5 (off-topic ~0.63, answerable ~0.80)


settings = Settings()
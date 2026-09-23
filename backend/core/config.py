"""Application settings, loaded from the .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Gemini
    gemini_api_key: str
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    # storage
    chroma_path: str = "data/chroma"

    # arXiv / ranking
    arxiv_max_results: int = 10
    top_k_papers: int = 3

    # chunking and retrieval
    chunk_size: int = 700            # tokens per chunk
    chunk_overlap: int = 100         # tokens shared between neighbouring chunks
    retrieval_top_k: int = 5         # chunks returned per question
    similarity_cutoff: float = 0.5   # tuned in Stage 5


settings = Settings()
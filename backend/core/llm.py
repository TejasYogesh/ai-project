"""Configure the Gemini LLM and embedding model for LlamaIndex."""
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from backend.core.config import settings


def init_llm() -> None:
    """Set the global LLM and embedding model used by all LlamaIndex components."""
    LlamaSettings.llm = GoogleGenAI(
        model=settings.llm_model,
        api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    LlamaSettings.embed_model = GoogleGenAIEmbedding(
        model_name=settings.embed_model,
        api_key=settings.gemini_api_key,
        embed_batch_size=10,
    )
"""Configure the Gemini LLM and embedding model, plus a retrying helper for LLM calls."""
import logging

from google.genai import errors as genai_errors
from llama_index.core import PromptTemplate
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.google_genai import GoogleGenAI
from tenacity import (before_sleep_log, retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)

from backend.core.config import settings
from backend.core.embeddings import RobustGeminiEmbedding

logger = logging.getLogger(__name__)


def init_llm() -> None:
    """Set the global LLM and embedding model used by all LlamaIndex components."""
    LlamaSettings.llm = GoogleGenAI(
        model=settings.llm_model,
        api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    LlamaSettings.embed_model = RobustGeminiEmbedding(
        model_name=settings.embed_model,
        api_key=settings.gemini_api_key,
        batch_size=settings.embed_batch_size,
        pause_seconds=settings.embed_pause_seconds,
    )


def _is_retryable(error: BaseException) -> bool:
    """Retry on 429 (resource exhausted) and 503 (model overloaded)."""
    return isinstance(error, genai_errors.APIError) and error.code in (429, 503)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=30),   # 2s, 4s, 8s ... max 30s
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def structured_llm_call(output_cls, prompt: PromptTemplate, **prompt_args):
    """Ask the LLM to fill in `output_cls`, retrying on temporary Gemini errors.
    All structured LLM calls in the project go through here."""
    return LlamaSettings.llm.structured_predict(output_cls, prompt, **prompt_args)
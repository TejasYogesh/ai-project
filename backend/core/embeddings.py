"""Gemini embedding model for LlamaIndex, with small batches and retry on rate limits."""
import logging
import time

from google import genai
from google.genai import errors, types
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from tenacity import (before_sleep_log, retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)

logger = logging.getLogger(__name__)


def _is_retryable(error: BaseException) -> bool:
    """Retry on 429 (resource exhausted) and 503 (service unavailable)."""
    return isinstance(error, errors.APIError) and error.code in (429, 503)


class RobustGeminiEmbedding(BaseEmbedding):
    """Gemini embeddings that survive temporary capacity errors."""

    pause_seconds: float = 1.0
    _client: genai.Client = PrivateAttr()

    def __init__(self, model_name: str, api_key: str, batch_size: int = 10,
                 pause_seconds: float = 1.0, **kwargs):
        super().__init__(model_name=model_name, embed_batch_size=batch_size,
                         pause_seconds=pause_seconds, **kwargs)
        self._client = genai.Client(api_key=api_key)

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),   # 2s, 4s, 8s ... max 60s
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """One API call for a batch of texts."""
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in result.embeddings]

    # --- methods LlamaIndex calls ---

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query], "RETRIEVAL_QUERY")[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_DOCUMENT")[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed(texts, "RETRIEVAL_DOCUMENT")
        time.sleep(self.pause_seconds)          # be gentle between batches
        return vectors

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
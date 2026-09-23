from llama_index.core import Settings

from backend.core.llm import init_llm

init_llm()
print("LLM:", Settings.llm.complete("Explain what arXiv is in one sentence."))
vector = Settings.embed_model.get_text_embedding("KV cache compression")
print("Embedding length:", len(vector))
print("First 5 numbers:", vector[:5])
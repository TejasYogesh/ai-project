"""Stage 3: chunk a parsed paper, embed it in memory, and retrieve chunks for questions."""
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from backend.core.config import settings
from backend.core.llm import init_llm
from backend.services.arxiv_client import search
from backend.services.pdf_parser import parse_pdf

init_llm()

# 1. Parse the paper (Stage 1 + 2)
paper = search(id_list="1706.03762")[0]
parsed = parse_pdf(paper)

# 2. One Document per section, with metadata for citations
documents = [
    Document(
        text=section.text,
        metadata={"arxiv_id": paper.arxiv_id, "section": section.title, "page": section.page_start},
        excluded_embed_metadata_keys=["arxiv_id", "page"],
    )
    for section in parsed.sections
]

# 3. Split sections into chunks
splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
nodes = splitter.get_nodes_from_documents(documents)
print(f"{len(parsed.sections)} sections -> {len(nodes)} chunks\n")
for node in nodes[:8]:
    print(f"  {len(node.text):>5} chars | {node.metadata['section']}")

# 4. Embed every chunk and build the index (this calls Gemini), cached so reruns don't burn quota
# ponytail: delete data/s3_index after changing chunking; Chroma replaces this in a later stage
persist_dir = Path("data/s3_index")
if persist_dir.exists():
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(persist_dir)))
else:
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(persist_dir))
retriever = index.as_retriever(similarity_top_k=settings.retrieval_top_k)

# 5. Ask questions
questions = [
    "How many layers does the encoder have?",
    "What optimizer did they use for training?",
    "What BLEU score did the model get on English-to-German?",
    "What is the capital of France?",          # off-topic on purpose
]
for question in questions:
    print(f"\nQ: {question}")
    for result in retriever.retrieve(question):
        preview = result.text[:90].replace("\n", " ")
        print(f"  {result.score:.3f} | {result.metadata['section']} (p.{result.metadata['page']}) | {preview}")
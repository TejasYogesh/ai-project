"""Chunk parsed papers, embed them, and store / retrieve them with ChromaDB."""
import logging

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from backend.core.config import settings
from backend.core.exceptions import PaperNotIndexed
from backend.models.schemas import PaperMeta, ParsedPaper

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Create the Chroma client once and reuse it."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def collection_name(arxiv_id: str) -> str:
    """'1706.03762' -> 'paper_1706_03762' (Chroma names cannot contain dots)."""
    return f"paper_{arxiv_id.replace('.', '_')}"


def _to_documents(paper: PaperMeta, parsed: ParsedPaper) -> list[Document]:
    """One Document per section; metadata is inherited by every chunk."""
    return [
        Document(
            text=section.text,
            metadata={"arxiv_id": paper.arxiv_id, "section": section.title,
                      "page": section.page_start},
            excluded_embed_metadata_keys=["arxiv_id", "page"],
        )
        for section in parsed.sections
    ]


def is_indexed(arxiv_id: str) -> bool:
    """True if this paper already has chunks stored in Chroma."""
    try:
        return _get_client().get_collection(collection_name(arxiv_id)).count() > 0
    except Exception:
        return False


def delete_index(arxiv_id: str) -> None:
    """Remove a paper's collection (e.g. after changing chunk settings)."""
    try:
        _get_client().delete_collection(collection_name(arxiv_id))
    except Exception:
        pass  # nothing to delete


def build_index(paper: PaperMeta, parsed: ParsedPaper, rebuild: bool = False) -> str:
    """Chunk, embed and store a paper. Skips the work if it is already indexed.
    Returns the collection name."""
    name = collection_name(paper.arxiv_id)
    if rebuild:
        delete_index(paper.arxiv_id)
    if is_indexed(paper.arxiv_id):
        logger.info("Paper %s already indexed; reusing it.", paper.arxiv_id)
        return name

    splitter = SentenceSplitter(chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(_to_documents(paper, parsed))

    collection = _get_client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    try:
        VectorStoreIndex(nodes, storage_context=storage)      # embeds + writes to Chroma
    except Exception:
        delete_index(paper.arxiv_id)                          # never leave a half-built index
        raise

    logger.info("Indexed %s: %d chunks into %s", paper.arxiv_id, len(nodes), name)
    return name


def load_index(arxiv_id: str) -> VectorStoreIndex:
    """Open an existing paper index without re-embedding anything."""
    if not is_indexed(arxiv_id):
        raise PaperNotIndexed(f"Paper {arxiv_id} has not been indexed yet.")
    collection = _get_client().get_collection(collection_name(arxiv_id))
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))


def retrieve(arxiv_id: str, question: str, top_k: int | None = None) -> list[NodeWithScore]:
    """Return the top-k chunks most similar to the question, with scores."""
    index = load_index(arxiv_id)
    retriever = index.as_retriever(similarity_top_k=top_k or settings.retrieval_top_k)
    return retriever.retrieve(question)
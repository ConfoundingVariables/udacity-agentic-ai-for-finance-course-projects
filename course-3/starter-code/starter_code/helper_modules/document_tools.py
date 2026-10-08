"""
Document Tools Module - one RAG tool per company 10-K filing.

Each company's 10-K PDF is loaded, chunked, embedded into a VectorStoreIndex,
and exposed as a QueryEngineTool named {SYMBOL}_10k_filing_tool.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.tools import QueryEngineTool

from .models import CONFIG, active_provider, configure_models

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "10k_documents"
INDEX_DIR = Path(__file__).resolve().parent.parent / "runtime" / "indexes"  # one folder per company + embedding model
COMPANY_NAMES = {"AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc. (Google)", "TSLA": "Tesla Inc."}


class DocumentToolsManager:
    """Builds the 10-K document analysis tools."""

    def __init__(self, companies: list[str] | None = None, verbose: bool = False):
        """Args: companies to index (default AAPL, GOOGL, TSLA); verbose prints progress."""
        self.companies = companies or list(COMPANY_NAMES)
        self.verbose = verbose
        self.document_tools: list[QueryEngineTool] = []
        self._configure_settings()

    def _configure_settings(self):
        """gpt-3.5-turbo + text-embedding-ada-002 via the active provider's api_base (see models.py)."""
        configure_models()

    def _tool(self, symbol: str) -> QueryEngineTool | None:
        try:
            tool = QueryEngineTool.from_defaults(
                query_engine=_index(symbol, active_provider().embedding).as_query_engine(
                    similarity_top_k=CONFIG["documents"]["similarity_top_k"]),
                name=f"{symbol}_10k_filing_tool",
                description=CONFIG["tools"]["document"].format(name=COMPANY_NAMES[symbol], symbol=symbol),
            )
            if self.verbose:
                print(f"   ✅ {symbol}_10k_filing_tool ready")
            return tool
        except Exception as e:  # missing PDF or embedding/API failure: skip this company
            print(f"   ❌ Could not build {symbol} tool: {e}")

    def build_document_tools(self) -> list[QueryEngineTool]:
        """One QueryEngineTool per company (each index embeds its batches concurrently)."""
        self.document_tools = [t for t in map(self._tool, self.companies) if t]
        return self.document_tools


def _embed_and_save(symbol: str, saved: Path):
    """Chunk the 10-K, embed the chunks in concurrent batches, save texts.json + vectors.npy."""
    cfg = CONFIG["documents"]
    docs = SimpleDirectoryReader(input_files=[str(DOCS_DIR / f"{symbol}_10K_2024.pdf")]).load_data()
    splitter = SentenceSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    texts = [node.get_content() for node in splitter.get_nodes_from_documents(docs)]
    size = CONFIG["models"]["embed_batch_size"]
    with ThreadPoolExecutor(CONFIG["models"]["embed_workers"]) as pool:
        batches = pool.map(Settings.embed_model.get_text_embedding_batch, [texts[i:i + size]
                                                                           for i in range(0, len(texts), size)])
        vectors = [v for batch in batches for v in batch]
    saved.mkdir(parents=True)
    (saved / "texts.json").write_text(json.dumps(texts))
    np.save(saved / "vectors.npy", np.array(vectors, dtype=np.float32))


@cache  # one load per process: the notebook builds tools twice (manager demo + coordinator)
def _index(symbol: str, embedding: str) -> VectorStoreIndex:
    """In-memory index from this embedding model's saved vectors (built on first use).

    Vectors live in a .npy file because LlamaIndex's JSON vector store takes ~15s per 10-K to parse.
    """
    saved = INDEX_DIR / f"{symbol}_{embedding.replace('/', '_')}"
    if not (saved / "vectors.npy").exists():
        _embed_and_save(symbol, saved)
    texts = json.loads((saved / "texts.json").read_text())
    metadata = {"symbol": symbol, "company": COMPANY_NAMES[symbol], "document_type": "10-K"}
    return VectorStoreIndex([TextNode(text=t, metadata=metadata, embedding=v.tolist())
                             for t, v in zip(texts, np.load(saved / "vectors.npy"), strict=True)])

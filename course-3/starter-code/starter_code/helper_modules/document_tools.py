"""
Document Tools Module - one RAG tool per company 10-K filing.

Each company's 10-K PDF is loaded, chunked, embedded into a VectorStoreIndex,
and exposed as a QueryEngineTool named {SYMBOL}_10k_filing_tool.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool

from .models import CONFIG, active_provider, configure_models

INDEX_DIR = Path(__file__).resolve().parent.parent / "runtime" / "indexes"  # cached per embedding model
COMPANY_NAMES = {"AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc. (Google)", "TSLA": "Tesla Inc."}


class DocumentToolsManager:
    """Builds the 10-K document analysis tools."""

    def __init__(self, companies: list[str] | None = None, verbose: bool = False):
        """Args: companies to index (default AAPL, GOOGL, TSLA); verbose prints progress."""
        self.companies = companies or list(COMPANY_NAMES)
        self.verbose = verbose
        self.documents_dir = Path(__file__).resolve().parent.parent / "data" / "10k_documents"
        self.document_tools: list[QueryEngineTool] = []
        self._configure_settings()

    def _configure_settings(self):
        """gpt-3.5-turbo + text-embedding-ada-002 via the active provider's api_base (see models.py)."""
        configure_models()

    def _index(self, symbol: str) -> VectorStoreIndex:
        """Load the company's index from disk, or build it (embedding batches concurrently) and save it."""
        cache = INDEX_DIR / f"{symbol}_{active_provider().embedding.replace('/', '_')}"
        if cache.exists():
            return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(cache)))  # type: ignore[return-value]
        cfg = CONFIG["documents"]
        docs = SimpleDirectoryReader(input_files=[str(self.documents_dir / f"{symbol}_10K_2024.pdf")]).load_data()
        for doc in docs:
            doc.metadata.update({"symbol": symbol, "company": COMPANY_NAMES[symbol], "document_type": "10-K"})
        splitter = SentenceSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        nodes = splitter.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, use_async=True)
        index.storage_context.persist(persist_dir=str(cache))
        return index

    def _tool(self, symbol: str) -> QueryEngineTool | None:
        try:
            tool = QueryEngineTool.from_defaults(
                query_engine=self._index(symbol).as_query_engine(similarity_top_k=CONFIG["documents"]["similarity_top_k"]),
                name=f"{symbol}_10k_filing_tool",
                description=CONFIG["tools"]["document"].format(name=COMPANY_NAMES[symbol], symbol=symbol),
            )
            if self.verbose:
                print(f"   ✅ {symbol}_10k_filing_tool ready")
            return tool
        except Exception as e:  # missing PDF or embedding/API failure: skip this company
            print(f"   ❌ Could not build {symbol} tool: {e}")

    def build_document_tools(self) -> list[QueryEngineTool]:
        """One QueryEngineTool per company; the companies are indexed in parallel threads."""
        with ThreadPoolExecutor() as pool:
            self.document_tools = [t for t in pool.map(self._tool, self.companies) if t]
        return self.document_tools

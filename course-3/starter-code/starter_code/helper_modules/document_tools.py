"""
Document Tools Module - one RAG tool per company 10-K filing.

Each company's 10-K PDF is loaded, chunked, embedded into a VectorStoreIndex,
and exposed as a QueryEngineTool named {SYMBOL}_10k_filing_tool.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # course-3/.env for local runs

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
        """Point LlamaIndex at Vocareum's OpenAI-compatible endpoint via api_base."""
        base_url = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_base=base_url)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_base=base_url)

    def build_document_tools(self) -> list[QueryEngineTool]:
        """Load, chunk and index each 10-K PDF; return one QueryEngineTool per company."""
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        self.document_tools = []
        for symbol in self.companies:
            name = COMPANY_NAMES[symbol]
            pdf = self.documents_dir / f"{symbol}_10K_2024.pdf"
            try:
                docs = SimpleDirectoryReader(input_files=[str(pdf)]).load_data()
                for doc in docs:
                    doc.metadata.update({"symbol": symbol, "company": name, "document_type": "10-K"})
                index = VectorStoreIndex(splitter.get_nodes_from_documents(docs))
                self.document_tools.append(
                    QueryEngineTool.from_defaults(
                        query_engine=index.as_query_engine(similarity_top_k=3),
                        name=f"{symbol}_10k_filing_tool",
                        description=(
                            f"Answers questions about {name} ({symbol}) from its 2024 SEC 10-K "
                            "filing: business segments, products, revenue drivers, risk factors, "
                            "competition and financial disclosures."
                        ),
                    )
                )
                if self.verbose:
                    print(f"   ✅ {symbol}_10k_filing_tool built")
            except Exception as e:  # missing PDF or embedding/API failure: skip this company
                print(f"   ❌ Could not build {symbol} tool: {e}")
        return self.document_tools

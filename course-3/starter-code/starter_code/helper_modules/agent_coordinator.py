"""
Agent Coordinator Module - the single entry point of the financial agent.

query() -> _route_query() (LLM picks tools) -> run tools -> mask PII in database
results -> return the single result, or synthesize several with the LLM.
"""

import ast
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .function_tools import detect_pii_fields

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # course-3/.env for local runs

ROUTING_GUIDELINES = """- Customers, portfolios, holdings, accounts, investment profiles -> database_query_tool
- Current/real-time stock price, change, trading volume -> finance_market_search_tool
- A company's business, products, segments, risk factors or revenue from its 10-K -> that company's *_10k_filing_tool
- Questions spanning several of these need several tools."""


class AgentCoordinator:
    """Routes questions across 3 document tools and 3 function tools."""

    def __init__(self, companies: list[str] | None = None, verbose: bool = False):
        """Args: companies for document tools (default AAPL, GOOGL, TSLA); verbose prints routing."""
        self.companies = companies or ["AAPL", "GOOGL", "TSLA"]
        self.verbose = verbose
        self.document_tools: list[QueryEngineTool] = []
        self.function_tools: list[FunctionTool] = []
        self._configure_settings()

    def _configure_settings(self):
        """LLM for routing and synthesis, using Vocareum's api_base."""
        base_url = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        self.llm = Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_base=base_url)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_base=base_url)

    def setup(self):
        """Build all tools (also called automatically by the first query())."""
        self._create_tools()
        if self.verbose:
            print(f"✅ Ready: {len(self.document_tools)} document tools, "
                  f"{len(self.function_tools)} function tools")

    def _create_tools(self):
        from .document_tools import DocumentToolsManager
        from .function_tools import FunctionToolsManager

        self.document_tools = DocumentToolsManager(self.companies, self.verbose).build_document_tools()
        self.function_tools = FunctionToolsManager(self.verbose).create_function_tools()

    def _execute_tool(self, tool, query: str) -> str:
        """Document tools are queried through their engine; function tools are called directly."""
        try:
            if isinstance(tool, QueryEngineTool):
                return str(tool.query_engine.query(query))
            return str(tool.fn(query))
        except Exception as e:
            return f"Error executing {tool.metadata.name}: {e}"

    def _detect_pii_fields(self, field_names: list) -> set:
        return detect_pii_fields(field_names)

    def _check_and_apply_pii_protection(self, tool_name: str, result: str) -> str:
        """Mask database results whose COLUMNS line names PII fields; pass everything else through."""
        match = re.search(r"COLUMNS: (\[.*\])", result)
        if tool_name != "database_query_tool" or not match:
            return result
        pii = self._detect_pii_fields(ast.literal_eval(match.group(1)))
        if not pii:
            return result
        if self.verbose:
            print(f"   🔒 PII detected {sorted(pii)} — applying protection")
        pii_tool = next(t for t in self.function_tools if t.metadata.name == "pii_protection_tool")
        return pii_tool.fn(result, match.group(1))

    def _route_query(self, query: str) -> list[tuple[str, str]]:
        """Ask the LLM which tools to use, run them, and return [(tool_name, result)]."""
        tools = [t for t in self.document_tools + self.function_tools if t.metadata.name != "pii_protection_tool"]
        listing = "\n".join(f"{i}: {t.metadata.name} - {t.metadata.description}" for i, t in enumerate(tools))
        prompt = (f"Select the tool(s) needed to fully answer the user's query.\n\nGuidelines:\n{ROUTING_GUIDELINES}"
                  f"\n\nTools:\n{listing}\n\nQuery: {query}\n\n"
                  "Respond with ONLY comma-separated tool indices, e.g. 0 or 1,4.")
        picked = dict.fromkeys(int(i) for i in re.findall(r"\d+", str(self.llm.complete(prompt))))
        selected = [tools[i] for i in picked if i < len(tools)]
        if self.verbose:
            print(f"🧭 Routed to: {[t.metadata.name for t in selected]}")
        results = []
        for tool in selected:
            name = tool.metadata.name or ""
            results.append((name, self._check_and_apply_pii_protection(name, self._execute_tool(tool, query))))
        return results

    def _synthesize_results(self, query: str, results: list[tuple[str, str]]) -> str:
        sources = "\n\n".join(f"[Source: {name}]\n{text}" for name, text in results)
        return str(self.llm.complete(
            "You are a financial analyst. Using ONLY the tool outputs below, write one clear, "
            "well-organized answer to the question. Integrate the sources, keep masked PII masked, "
            f"and do not invent data.\n\nQuestion: {query}\n\nTool outputs:\n{sources}\n\nAnswer:"
        )).strip()

    def query(self, question: str, verbose: bool | None = None) -> str:
        """Answer a question end to end: route, run tools, protect PII, synthesize if needed."""
        if verbose is not None:
            self.verbose = verbose
        if not self.function_tools:
            self.setup()
        results = self._route_query(question)
        if not results:
            return "No tool matched this question. Ask about a 10-K, customer portfolios or stock prices."
        if len(results) == 1:  # single source: no synthesis overhead
            return results[0][1]
        return self._synthesize_results(question, results)

    def list_available_tools(self) -> list[str]:
        return [t.metadata.name or "" for t in self.document_tools + self.function_tools]

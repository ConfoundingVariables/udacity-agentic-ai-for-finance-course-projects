"""
Agent Coordinator Module - the single entry point of the financial agent.

query() -> _route_query() (DSPy router picks tools + confidence) -> run tools -> mask PII in database
results -> return the single result, or synthesize several with the LLM -> record metrics.
"""

import ast
import json
import logging
import re
import time
from collections import Counter
from statistics import mean

from llama_index.core.tools import FunctionTool, QueryEngineTool

from .function_tools import detect_pii_fields
from .models import CONFIG, TOKEN_COUNTER, active_provider, configure_models
from .router import RUNTIME, ToolRouter

logger = logging.getLogger("financial_agent")


def _llama_tokens() -> tuple[int, int, int]:
    c = TOKEN_COUNTER
    return c.prompt_llm_token_count, c.completion_llm_token_count, c.total_embedding_token_count

class AgentCoordinator:
    """Routes questions across 3 document tools and 5 function tools."""

    def __init__(self, companies: list[str] | None = None, verbose: bool = False):
        """Args: companies for document tools (default AAPL, GOOGL, TSLA); verbose prints routing."""
        self.companies = companies or ["AAPL", "GOOGL", "TSLA"]
        self.verbose = verbose
        self.document_tools: list[QueryEngineTool] = []
        self.function_tools: list[FunctionTool] = []
        self.traces: list[dict] = []
        self.last_trace: dict = {}
        self._configure_settings()

    def _configure_settings(self):
        """gpt-3.5-turbo via the active provider's api_base (Vocareum, else OpenRouter; see models.py)."""
        self.llm = configure_models()

    def setup(self):
        """Build all tools (also called automatically by the first query())."""
        self._create_tools()
        catalog = "\n".join(f"- {n}: {t.metadata.description}" for n, t in self._routable_tools().items())
        self.router = ToolRouter(catalog)
        if self.verbose:
            print(f"✅ Ready: {len(self.document_tools)} document tools, "
                  f"{len(self.function_tools)} function tools")

    def _create_tools(self):
        from .document_tools import DocumentToolsManager
        from .function_tools import FunctionToolsManager

        self.document_tools = DocumentToolsManager(self.companies, self.verbose).build_document_tools()
        self.function_tools = FunctionToolsManager(self.verbose).create_function_tools()

    def _routable_tools(self) -> dict:
        """Name -> tool for everything the router may pick (PII protection is applied automatically instead)."""
        return {str(t.metadata.name): t for t in self.document_tools + self.function_tools
                if t.metadata.name != "pii_protection_tool"}

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
        """DSPy router picks tools (with a confidence score); run them and return [(tool_name, result)]."""
        tools = self._routable_tools()
        self.route = self.router(query)
        selected = [n for n in dict.fromkeys(self.route.tools) if n in tools]
        if self.verbose:
            print(f"🧭 Routed to {selected} (confidence {self.route.confidence:.2f})")
        return [(n, self._check_and_apply_pii_protection(n, self._execute_tool(tools[n], query))) for n in selected]

    def _synthesize_results(self, query: str, results: list[tuple[str, str]]) -> str:
        sources = "\n".join(f'<source name="{name}">\n{text}\n</source>' for name, text in results)
        return str(self.llm.complete(CONFIG["prompts"]["synthesis"].format(query=query, sources=sources))).strip()

    def query(self, question: str, verbose: bool | None = None) -> str:
        """Answer a question end to end: route, run tools, protect PII, synthesize if needed, record metrics."""
        if verbose is not None:
            self.verbose = verbose
        if not self.function_tools:
            self.setup()
        start, tokens_before = time.perf_counter(), _llama_tokens()
        results = self._route_query(question)
        if not results:
            answer = "No tool matched this question. Ask about a 10-K, customer portfolios or stock prices."
        elif len(results) == 1:  # single source: no synthesis overhead
            answer = results[0][1]
        else:
            answer = self._synthesize_results(question, results)
        confidence = float(self.route.confidence)
        if confidence < CONFIG["routing"]["low_confidence_threshold"]:
            answer = f"⚠️ Low routing confidence ({confidence:.2f}): verify this answer.\n\n{answer}"
        self._record(question, [name for name, _ in results], confidence, time.perf_counter() - start, tokens_before)
        return answer

    def _record(self, question, tools, confidence, latency, tokens_before):
        """Monitoring: per-query latency, tokens (LlamaIndex + DSPy) and estimated cost -> runtime/metrics.jsonl."""
        llm_in, llm_out, embed = (a - b for a, b in zip(_llama_tokens(), tokens_before, strict=True))
        for usage in (self.route.get_lm_usage() or {}).values():
            llm_in, llm_out = llm_in + usage.get("prompt_tokens", 0), llm_out + usage.get("completion_tokens", 0)
        price = CONFIG["pricing"]
        cost = (llm_in * price["llm_input"] + llm_out * price["llm_output"] + embed * price["embedding"]) / 1e6
        self.last_trace = {"question": question, "provider": active_provider().name, "tools": tools,
                           "confidence": round(confidence, 2), "reasoning": self.route.reasoning,
                           "latency_s": round(latency, 2), "llm_prompt_tokens": llm_in,
                           "llm_completion_tokens": llm_out, "embedding_tokens": embed, "cost_usd": round(cost, 6)}
        self.traces.append(self.last_trace)
        RUNTIME.mkdir(exist_ok=True)
        with (RUNTIME / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(self.last_trace) + "\n")
        logger.info("%s | %s | conf %.2f | %.2fs | $%.5f", question[:50], tools, confidence, latency, cost)

    def metrics_summary(self) -> dict:
        t = self.traces
        return {"queries": len(t), "avg_latency_s": round(mean(x["latency_s"] for x in t), 2) if t else 0,
                "total_tokens": sum(x["llm_prompt_tokens"] + x["llm_completion_tokens"] + x["embedding_tokens"]
                                    for x in t),
                "total_cost_usd": round(sum(x["cost_usd"] for x in t), 5),
                "avg_confidence": round(mean(x["confidence"] for x in t), 2) if t else 0,
                "tool_usage": dict(Counter(n for x in t for n in x["tools"]))}

    def feedback(self, question: str, correct_tools: list[str]):
        """Record the tools that should have been used; optimize_router() learns from these."""
        self.router.record_feedback(question, correct_tools)

    def optimize_router(self) -> dict:
        return self.router.optimize()

    def list_available_tools(self) -> list[str]:
        return [t.metadata.name or "" for t in self.document_tools + self.function_tools]

"""End-to-end checks against the live LLM, embeddings, SQLite DB and Yahoo Finance.

Run from starter_code/: python -m pytest tests/test_e2e.py
"""

import re

import pytest
from helper_modules.agent_coordinator import AgentCoordinator


@pytest.fixture(scope="module")
def agent():
    coordinator = AgentCoordinator(companies=["AAPL"])  # one 10-K keeps indexing cost low
    coordinator.setup()
    return coordinator


def tools_used(agent, question):
    return [name for name, _ in agent._route_query(question)]


def test_document_question_uses_10k(agent):
    assert tools_used(agent, "What risk factors does Apple describe in its 10-K?") == ["AAPL_10k_filing_tool"]
    assert "iphone" in agent.query("What are Apple's main products according to its 10-K?").lower()


def test_customer_pii_is_masked(agent):
    answer = agent.query("List the first name, last name and email of every customer")
    assert "🔒 PII Protection Applied" in answer
    assert not re.search(r"[\w.]+@", answer.replace("***@", ""))


def test_market_price(agent):
    answer = agent.query("What is Tesla's current stock price?")
    assert re.search(r"TSLA: \$[\d,]+\.\d\d", answer)


def test_multi_source_question_routes_to_both_tools(agent):
    question = "Compare Tesla's current stock price with what our customers paid for their Tesla shares"
    assert {"finance_market_search_tool", "database_query_tool"} <= set(tools_used(agent, question))

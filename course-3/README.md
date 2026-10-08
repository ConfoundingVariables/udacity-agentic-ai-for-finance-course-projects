# Course 3 — Financial Multi-Tool Agent

An LLM-routed agent that answers financial questions using six tools across three data sources: SEC 10-K filings, a SQLite portfolio database and the Yahoo Finance API. Customer PII in database results is masked automatically.

Code: [`starter-code/starter_code/`](starter-code/starter_code/) · Walkthrough: [`financial_agent_walkthrough.ipynb`](starter-code/starter_code/financial_agent_walkthrough.ipynb)

## Architecture

```mermaid
flowchart TD
    U["User query"] --> Q["AgentCoordinator.query()"]
    Q --> R["_route_query()<br/>LLM selects tool indices"]
    R --> D["{AAPL,GOOGL,TSLA}_10k_filing_tool<br/>RAG over 10-K PDFs"]
    R --> DB["database_query_tool<br/>NL → SQL → SQLite"]
    R --> M["finance_market_search_tool<br/>Yahoo Finance, DB fallback"]
    DB --> P{"COLUMNS contain PII?"}
    P -->|yes| PII["pii_protection_tool"]
    P -->|no| A["Results"]
    PII --> A
    D --> A
    M --> A
    A --> S{"How many results?"}
    S -->|one| OUT["Returned as-is"]
    S -->|several| SYN["_synthesize_results()<br/>LLM merges sources"]
```

| Module | Responsibility |
|---|---|
| `helper_modules/document_tools.py` | Load each 10-K PDF, chunk (1024/100), embed, one `QueryEngineTool` per company |
| `helper_modules/function_tools.py` | Database (NL→SQL, one retry with the error as context), market data, PII masking as `FunctionTool`s |
| `helper_modules/agent_coordinator.py` | Configure models, build tools, route, coordinate PII, synthesize |

All modules use **gpt-3.5-turbo** and **text-embedding-ada-002** with Vocareum's `api_base` (`OPENAI_API_BASE`), as the rubric requires.

## Implementation choices

- **Routing** is a single LLM call that returns tool indices. The PII tool is never routed; the coordinator applies it.
- **PII trigger:** the database tool ends every result with `COLUMNS: [...]`. The coordinator masks the result only when those columns include `name`, `email`, `phone`, `address` or `ssn`.
- **Masking:** email → `***@domain.com`, phone → `***-***-1234`, SSN → `***-**-****`, address → `*** [address masked] ***`, names → `****`. Dict rows are masked by column. Free text is masked by pattern. A notice lists the masked fields.
- **Market data** falls back to the latest stored close in `market_data` when Yahoo is unreachable or rate-limited. The output labels the source.
- **Paths** resolve from `__file__`, and `course-3/.env` loads automatically, so the code runs from any working directory.

## Database (`data/financial.db`)

```mermaid
erDiagram
    customers ||--o{ portfolio_holdings : owns
    companies ||--o{ portfolio_holdings : "held as"
    companies ||--o{ market_data : has
```

Seed data: 10 customers, 3 companies, 23 holdings, 90 market rows. Schema details are in `DB_SCHEMA` in `function_tools.py`.

## Running

```bash
# from course-3/
uv venv .venv && uv pip install --python .venv -r starter-code/starter_code/requirements.txt pytest
cp .env.template .env   # set OPENAI_API_KEY and OPENAI_API_BASE

# from course-3/starter-code/starter_code/
../../.venv/bin/python data/build_database.py
../../.venv/bin/python -m pytest tests/test_e2e.py
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace financial_agent_walkthrough.ipynb
```

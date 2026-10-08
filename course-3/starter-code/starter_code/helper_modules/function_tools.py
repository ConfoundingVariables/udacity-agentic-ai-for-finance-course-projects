"""
Function Tools Module - database, market data and PII protection tools.

1. database_query_tool        natural language -> SQL -> SQLite rows (+ COLUMNS line)
2. finance_market_search_tool live quotes from Yahoo Finance, falling back to stored prices
3. pii_protection_tool        masks emails, phones, names, addresses and SSNs
"""

import ast
import os
import re
import sqlite3
from pathlib import Path

import requests
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # course-3/.env for local runs

SYMBOLS = {"apple": "AAPL", "aapl": "AAPL", "tesla": "TSLA", "tsla": "TSLA",
           "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL"}
PII_MARKERS = ("name", "email", "phone", "address", "ssn", "social_security")

DB_SCHEMA = """SQLite schema:
customers(id PK, first_name, last_name, email, phone, investment_profile -- conservative/moderate/aggressive,
          risk_tolerance -- low/medium/high, account_balance REAL, total_portfolio_value REAL)
portfolio_holdings(id PK, customer_id FK -> customers.id, symbol, shares REAL, purchase_price REAL, current_value REAL)
companies(id PK, symbol, name, sector, market_cap REAL)
market_data(id PK, symbol FK -> companies.symbol, close_price REAL, volume INTEGER, market_cap REAL, date TEXT)

Tips: join portfolio_holdings to customers for names, to companies for company names/sectors,
to market_data for prices. market_data has many dates: use the latest (ORDER BY date DESC / MAX(date)).
Symbols are 'AAPL', 'GOOGL', 'TSLA'."""


def detect_pii_fields(field_names) -> set:
    """Column names that look like PII (email, phone, *name, address, ssn)."""
    return {f for f in field_names if any(m in str(f).lower() for m in PII_MARKERS)}


def mask_value(field: str, value) -> str:
    """Field-specific masking: ***@domain, ***-***-1234, ***-**-****, ****."""
    v, f = str(value), field.lower()
    if "email" in f:
        return "***@" + v.split("@")[-1]
    if "phone" in f:
        return "***-***-" + re.sub(r"\D", "", v)[-4:]
    if "ssn" in f or "social" in f:
        return "***-**-****"
    if "address" in f:
        return "*** [address masked] ***"
    return "****"


def mask_free_text(text: str) -> str:
    """Pattern-based masking for lines that are not dict rows."""
    text = re.sub(r"[\w.+-]+@([\w-]+\.[\w.-]+)", r"***@\1", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", text)  # SSN before phone
    return re.sub(r"\b(?:\d{3}[-.\s])?\d{3}[-.\s](\d{4})\b", r"***-***-\1", text)


class FunctionToolsManager:
    """Builds the database, market data and PII protection function tools."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.db_path = Path(__file__).resolve().parent.parent / "data" / "financial.db"
        self.function_tools: list[FunctionTool] = []
        self._configure_settings()

    def _configure_settings(self):
        """LLM for SQL generation, using Vocareum's api_base."""
        base_url = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        self.llm = Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_base=base_url)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_base=base_url)

    def _run_sql(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [d[0] for d in cursor.description or []], cursor.fetchall()

    def database_query_tool(self, query: str) -> str:
        """Answer a natural-language question about customers, holdings or companies via SQL.

        Generates SQL with the LLM; if it fails, retries once with the error as context.
        Returns the SQL, one dict per row, and a final `COLUMNS: [...]` line used for PII detection.
        """
        sql = error = ""
        for _ in range(2):
            prompt = (f"{DB_SCHEMA}\n\nWrite ONE SQLite SELECT statement answering: {query}\n"
                      + (f"The previous attempt failed with: {error}\n" if error else "")
                      + "Return only the SQL, no markdown or explanation.")
            sql = re.sub(r"```(sql)?", "", str(self.llm.complete(prompt)), flags=re.I).strip().split(";")[0]
            try:
                columns, rows = self._run_sql(sql)
                body = "\n".join(str(dict(zip(columns, r, strict=False))) for r in rows) or "(no rows)"
                return f"SQL Query: {sql}\n\nDatabase Results:\n{body}\nCOLUMNS: {columns}"
            except sqlite3.Error as e:
                error = str(e)
                if self.verbose:
                    print(f"   ⚠️  SQL failed ({error}), retrying with error context")
        return f"Database query error after retry: {error}\nSQL attempted: {sql}"

    def _quote(self, symbol: str) -> str:
        """One formatted quote line; stored DB price if Yahoo is down or rate-limited."""
        try:
            resp = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
            m = resp.json()["chart"]["result"][0]["meta"]
            price, prev = m["regularMarketPrice"], m["chartPreviousClose"]
            change = price - prev
            return (f"{symbol}: ${price:,.2f} | Change: {change:+.2f} ({change / prev * 100:+.2f}%)"
                    f" | Volume: {m.get('regularMarketVolume') or 0:,} [source: Yahoo Finance]")
        except Exception as e:
            _, rows = self._run_sql("SELECT close_price, volume, date FROM market_data "
                                    "WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,))
            if not rows:
                return f"{symbol}: market data unavailable ({e})"
            price, volume, date = rows[0]
            return (f"{symbol}: ${price:,.2f} | Volume: {volume:,} "
                    f"[source: database close on {date}; live API unavailable: {type(e).__name__}]")

    def finance_market_search_tool(self, query: str) -> str:
        """Current price, change, % change and volume for Apple, Google and/or Tesla named in the query."""
        symbols = dict.fromkeys(s for k, s in SYMBOLS.items() if k in query.lower()) or ["AAPL", "GOOGL", "TSLA"]
        return "Real-Time Market Data:\n" + "\n".join(self._quote(s) for s in symbols)

    def pii_protection_tool(self, database_results: str, column_names: str = "") -> str:
        """Mask PII in database results and append a notice listing the masked fields.

        Dict rows are masked by column name (from column_names, or each row's keys);
        other lines are masked by email/phone/SSN patterns.
        """
        pii = detect_pii_fields(ast.literal_eval(column_names)) if column_names else set()
        masked: set = set()
        lines = []
        for line in database_results.split("\n"):
            if line.startswith("{") and line.endswith("}"):
                row = ast.literal_eval(line)
                fields = (pii or detect_pii_fields(row)) & row.keys()
                row.update({f: mask_value(f, row[f]) for f in fields})
                masked |= fields
                line = str(row)
            elif not line.startswith("COLUMNS:"):
                new = mask_free_text(line)
                if new != line:
                    masked.add("pattern-matched contact data")
                line = new
            lines.append(line)
        notice = f"\n\n🔒 PII Protection Applied — masked fields: {sorted(masked)}" if masked else ""
        return "\n".join(lines) + notice

    def create_function_tools(self) -> list[FunctionTool]:
        """Wrap the three tool methods as LlamaIndex FunctionTools."""
        self.function_tools = [
            FunctionTool.from_defaults(
                fn=self.database_query_tool, name="database_query_tool",
                description="Query the customer and portfolio database in natural language: customers, "
                            "holdings, account balances, investment profiles, company records."),
            FunctionTool.from_defaults(
                fn=self.finance_market_search_tool, name="finance_market_search_tool",
                description="Real-time stock price, change and trading volume for Apple (AAPL), "
                            "Google (GOOGL) and Tesla (TSLA) from Yahoo Finance."),
            FunctionTool.from_defaults(
                fn=self.pii_protection_tool, name="pii_protection_tool",
                description="Mask personal data (names, emails, phones, addresses, SSNs) in database results."),
        ]
        return self.function_tools

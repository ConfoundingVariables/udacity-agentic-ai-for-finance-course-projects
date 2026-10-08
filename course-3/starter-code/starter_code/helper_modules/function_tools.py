"""
Function Tools Module - database, market data and PII protection tools.

1. database_query_tool        natural language -> SQL -> SQLite rows (+ COLUMNS line)
2. finance_market_search_tool live quotes from Yahoo Finance, falling back to stored prices
3. pii_protection_tool        masks emails, phones, names, addresses and SSNs
4. financial_analysis_tool    period return, volatility, 20/50-day SMAs, trend, max drawdown
5. portfolio_analysis_tool    cost basis, market value, unrealized P&L, weights, concentration (no PII)
"""

import ast
import math
import re
import sqlite3
import statistics
from pathlib import Path

import requests
from llama_index.core.tools import FunctionTool

from .models import CONFIG, configure_models

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
        """gpt-3.5-turbo via the active provider's api_base (Vocareum, else OpenRouter; see models.py)."""
        self.llm = configure_models()

    def _run_sql(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [d[0] for d in cursor.description or []], cursor.fetchall()

    def _fetch_chart(self, symbol: str) -> dict:
        """Fetch Yahoo Finance chart JSON for `symbol` over the configured history range; raises on error."""
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": CONFIG["analysis"]["history_range"], "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["chart"]["result"][0]

    def database_query_tool(self, query: str) -> str:
        """Answer a natural-language question about customers, holdings or companies via SQL.

        Generates SQL with the LLM; if it fails, retries once with the error as context.
        Returns the SQL, one dict per row, and a final `COLUMNS: [...]` line used for PII detection.
        """
        sql = error = ""
        for _ in range(2):
            prompts = CONFIG["prompts"]
            prompt = prompts["sql"].format(schema=DB_SCHEMA, query=query,
                                           error=prompts["sql_retry"].format(error=error) if error else "")
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
            chart = self._fetch_chart(symbol)
            m = chart["meta"]
            # chartPreviousClose is the close before the whole history range, so take yesterday's daily close.
            closes = [c for c in chart["indicators"]["quote"][0]["close"] if c is not None]
            price, prev = m["regularMarketPrice"], closes[-2]
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

    def financial_analysis_tool(self, query: str) -> str:
        """Period return, annualized volatility, 20/50-day SMAs, trend and max drawdown for queried symbols."""
        symbols = dict.fromkeys(s for k, s in SYMBOLS.items() if k in query.lower()) or ["AAPL", "GOOGL", "TSLA"]
        lines = ["Financial Analysis:"]
        for sym in symbols:
            try:
                result = self._fetch_chart(sym)
                raw = result["indicators"]["quote"][0]["close"]
                closes = [c for c in raw if c is not None]
                source = f"Yahoo Finance ({CONFIG['analysis']['history_range']})"
            except Exception:
                _, rows = self._run_sql(
                    "SELECT close_price FROM market_data WHERE symbol=? ORDER BY date", (sym,))
                closes = [r[0] for r in rows]
                source = "database"
            if len(closes) < 2:
                lines.append(f"\n{sym}: insufficient price data")
                continue
            period_ret = (closes[-1] - closes[0]) / closes[0] * 100
            daily = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
            vol = statistics.stdev(daily) * math.sqrt(252) * 100
            sma20 = statistics.mean(closes[-20:]) if len(closes) >= 20 else None
            sma50 = statistics.mean(closes[-50:]) if len(closes) >= 50 else None
            price = closes[-1]
            sma_str = " | ".join(
                f"{lbl}={val:,.2f}" for lbl, val in (("SMA20", sma20), ("SMA50", sma50)) if val is not None
            ) or "SMAs: n/a"
            trend = ", ".join(
                ("above" if price > val else "below") + f" {lbl}"
                for lbl, val in (("SMA20", sma20), ("SMA50", sma50)) if val is not None
            ) or "n/a"
            peak, max_dd = closes[0], 0.0
            for c in closes:
                peak = max(peak, c)
                max_dd = min(max_dd, (c - peak) / peak * 100)
            lines.append(
                f"\n{sym} [{source}]:\n"
                f"  Period Return: {period_ret:+.2f}%\n"
                f"  Annualized Volatility: {vol:.2f}%\n"
                f"  {sma_str}\n"
                f"  Trend: {trend}\n"
                f"  Max Drawdown: {max_dd:.2f}%"
            )
        return "\n".join(lines)

    def portfolio_analysis_tool(self, query: str) -> str:
        """Holdings analysis at live prices: cost basis, market value, P&L, weights, concentration. No PII."""
        m = re.search(r"(?:customer|client|id)\s*#?\s*(\d+)", query, re.I)
        cid = int(m.group(1)) if m else None
        sql = "SELECT id, investment_profile, risk_tolerance FROM customers" + (" WHERE id=?" if cid else "")
        _, customers = self._run_sql(sql, (cid,) if cid else ())
        if not customers:
            return "No customers found."

        _price_cache: dict[str, tuple] = {}

        def _live(sym: str) -> tuple:
            if sym not in _price_cache:
                try:
                    _price_cache[sym] = (self._fetch_chart(sym)["meta"]["regularMarketPrice"], "Yahoo Finance")
                except Exception:
                    _, rows = self._run_sql(
                        "SELECT close_price, date FROM market_data WHERE symbol=? ORDER BY date DESC LIMIT 1",
                        (sym,))
                    _price_cache[sym] = (rows[0][0], f"DB {rows[0][1]}") if rows else (None, "unavailable")
            return _price_cache[sym]

        max_wts = CONFIG["analysis"]["max_position_weight"]
        lines = ["Portfolio Analysis:"]
        for customer_id, inv_profile, risk_tol in customers:
            _, holdings = self._run_sql(
                "SELECT symbol, shares, purchase_price FROM portfolio_holdings WHERE customer_id=?",
                (customer_id,))
            if not holdings:
                lines.append(f"\nCustomer {customer_id} ({inv_profile}, {risk_tol} risk): no holdings")
                continue
            rows_data = []
            total_cost = total_value = 0.0
            for sym, shares, pp in holdings:
                cost = shares * pp
                price, src = _live(sym)
                if price is None:
                    rows_data.append((sym, shares, cost, None, None, None, src))
                    total_cost += cost
                    continue
                value = shares * price
                pnl = value - cost
                rows_data.append((sym, shares, cost, value, pnl, pnl / cost * 100 if cost else 0.0, src))
                total_cost += cost
                total_value += value

            max_wt = max_wts.get(risk_tol, 1.0)
            detail: list[str] = []
            alerts: list[str] = []
            for sym, shares, cost, value, pnl, pnl_pct, src in rows_data:
                if value is None:
                    detail.append(f"  {sym}: {shares:.2f} shares | Cost ${cost:,.2f} | Value N/A ({src})")
                    continue
                wt = value / total_value if total_value else 0.0
                if wt > max_wt:
                    alerts.append(
                        f"  ALERT: {sym} {wt:.0%} of portfolio — over-concentrated"
                        f" for {risk_tol} risk tolerance (max {max_wt:.0%})")
                detail.append(
                    f"  {sym}: {shares:.2f} sh @ ${cost / shares:,.2f}/sh paid, now ${value / shares:,.2f}/sh"
                    f" | Total cost ${cost:,.2f} | Value ${value:,.2f} ({src})"
                    f" | P&L {pnl:+,.2f} ({pnl_pct:+.2f}%) | Weight {wt:.1%}")

            total_pnl = total_value - total_cost
            total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0.0
            lines.append(
                f"\nCustomer {customer_id} ({inv_profile}, {risk_tol} risk):\n"
                + "\n".join(detail)
                + f"\n  TOTAL: Cost ${total_cost:,.2f} | Value ${total_value:,.2f}"
                  f" | P&L {total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
            lines.extend(alerts)
        return "\n".join(lines)

    def create_function_tools(self) -> list[FunctionTool]:
        """Wrap the five tool methods as LlamaIndex FunctionTools."""
        t = CONFIG["tools"]
        self.function_tools = [
            FunctionTool.from_defaults(
                fn=self.database_query_tool, name="database_query_tool",
                description=t["database_query_tool"]),
            FunctionTool.from_defaults(
                fn=self.finance_market_search_tool, name="finance_market_search_tool",
                description=t["finance_market_search_tool"]),
            FunctionTool.from_defaults(
                fn=self.pii_protection_tool, name="pii_protection_tool",
                description=t["pii_protection_tool"]),
            FunctionTool.from_defaults(
                fn=self.financial_analysis_tool, name="financial_analysis_tool",
                description=t["financial_analysis_tool"]),
            FunctionTool.from_defaults(
                fn=self.portfolio_analysis_tool, name="portfolio_analysis_tool",
                description=t["portfolio_analysis_tool"]),
        ]
        return self.function_tools

# SWIFT Transaction Processing System

A multi-agent system that processes SWIFT payment messages (MT103 / MT202)
through four agentic workflow patterns, screens them for fraud, and produces
operational reports including the **Straight-Through-Processing (STP) rate**.

The system is built to run **online** (against an OpenAI-compatible endpoint such
as the Vocareum proxy) and to **degrade gracefully offline** — when no API key is
configured, every pattern falls back to deterministic heuristics so the pipeline
always completes and still produces reports.

---

## Pipeline overview

```
generate → 1. Evaluator-Optimizer → 2. Parallelization → 3. Prompt Chaining → 4. Orchestrator-Worker → reports
             (validate & correct)     (fraud screening)     (deep investigation)   (task delegation)
```

| Stage | Pattern | Module | Purpose |
|------|---------|--------|---------|
| 1 | Evaluator-Optimizer | `agents/evaluator_optimizer.py` | Validate each message against SWIFT standards; iteratively correct issues via `SwiftCorrectionAgent`. |
| 2 | Parallelization | `agents/parallelization.py` | Run four fraud agents concurrently per message and aggregate their verdicts. |
| 3 | Prompt Chaining | `agents/prompt_chaining.py` | Escalate only *suspicious* messages through a 5-stage investigation chain. |
| 4 | Orchestrator-Worker | `agents/orchestrator_worker.py` | Decompose the goal into tasks and delegate to capability-scoped workers. |

Only suspicious transactions reach stage 3, and each report set is produced from
a named filter — both are deliberate performance choices that mirror a real
fraud desk.

---

## Agents (fraud screening)

Defined in `agents/workflow_agents/base_agents.py`:

- **FraudAmountDetectionAgent** — rule-based amount heuristics (large / round / odd precision).
- **FraudPatternDetectionAgent** — BIC and remittance-keyword patterns (TEST/FAKE BICs, same sender/receiver, "urgent"/"secret").
- **GeographicRiskAgent** — jurisdiction risk from the BIC country code (chars 5-6) plus cross-border corridors.
- **AIAnomalyDetectionAgent** — *custom AI-driven agent*. Computes a cheap statistical signal (Benford leading-digit rarity, round-number bias, extreme magnitude) and **only escalates borderline cases to the LLM**, blending the heuristic and AI judgement. Falls back to the pure heuristic when offline.
- **FraudAggAgent** — averages the agents' risk scores and applies the fraud threshold.

---

## Reliability & performance (`services/`)

`services/llm_client.py` centralises **all** LLM access so these concerns live in one place:

- **Caching** — identical prompts are cached on disk (`.llm_cache/`, SHA-256 keyed).
- **Retry** — transient failures retry with exponential backoff; non-transient 4xx errors are not retried.
- **JSON-mode capability detection** — `response_format={"type":"json_object"}` is only sent to models that support it, and is auto-disabled at runtime if the endpoint rejects it (the Vocareum `gpt-4` model does). Responses are parsed with a tolerant JSON extractor.
- **Offline fallback** — with no usable key (or when the API is unreachable) each caller supplies a deterministic fallback so the pipeline never crashes.

`services/reporting.py` computes metrics (STP rate = *valid AND clean* / total) and
writes JSON + text reports to `reports/`.

---

## Setup

```bash
cd course-2/starter-code/project
python3 -m venv .venv
./.venv/bin/python -m pip install \
    faker numpy openai pandas pydantic scipy python-dotenv pytest

# Configure credentials (copy the template and fill in a real key)
cp .env.template .env
#   OPENAI_API_KEY=<your key>
#   OPENAI_BASE_URL=https://openai.vocareum.com/v1
#   OPENAI_MODEL=gpt-4
```

`.env` is git-ignored; only `.env.template` is committed. With a placeholder key
the system runs fully offline on deterministic heuristics.

---

## Usage

```bash
# Generate sample data -> data/swift_messages.csv
./.venv/bin/python generate_swift_messages.py --count 40

# Run the full pipeline (two+ report sets + STP rate)
./.venv/bin/python main.py

# Score a completed run against weighted criteria
./.venv/bin/python evaluator.py reports/pipeline_report.json

# Adaptable runtime: scenarios, single-message screening, or a REPL
./.venv/bin/python process_chat.py --scenario high_value
./.venv/bin/python process_chat.py --message '{"amount":"9000000.00","currency":"USD","sender_bic":"TESTRU33XXX","receiver_bic":"TESTRU33XXX","remittance_info":"urgent secret transfer"}'
./.venv/bin/python process_chat.py --interactive

# Tests (hermetic — forced offline, no API calls)
./.venv/bin/python -m pytest -q
```

Running `main.py` writes three report sets to `reports/`: the full
`pipeline_report.*` plus one file per requested filter
(`non_fraudulent_report.*`, `high_value_report.*`).

---

## Layout

```
project/
├── main.py                    # pipeline entry point (STP rate + report sets)
├── config.py                  # env-driven config, JSON-mode & offline flags
├── evaluator.py               # weighted scorecard for a completed run
├── process_chat.py            # scenario / single-message / interactive runtime
├── generate_swift_messages.py # writes data/swift_messages.csv
├── agents/
│   ├── evaluator_optimizer.py
│   ├── parallelization.py
│   ├── prompt_chaining.py
│   ├── orchestrator_worker.py
│   └── workflow_agents/base_agents.py
├── services/
│   ├── llm_client.py          # caching + retry + JSON-mode + offline fallback
│   ├── llm_service.py
│   ├── reporting.py
│   └── swift_generator.py
├── models/                    # SWIFTMessage, BankRegistry
└── tests/                     # pytest suite (offline)
```

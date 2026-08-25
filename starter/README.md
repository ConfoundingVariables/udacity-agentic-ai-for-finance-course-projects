# Educational AML/SAR Review System

A complete educational suspicious-activity review workflow built with Python 3.11, Pydantic, pandas, OpenAI-compatible agents, FastAPI, and vanilla HTML/CSS/JavaScript.

The system uses synthetic course data. It does not submit filings to FinCEN, restrict accounts, or provide legal advice.

## Architecture

```text
customers.csv + accounts.csv + transactions.csv
                     |
               validated CaseData
                     |
              RiskAnalystAgent
                     |
              human risk gate
                     |
          ComplianceOfficerAgent
                     |
           human compliance gate
                     |
           SAR JSON + audit outputs
```

Human approval is authoritative at both gates. Compliance generation cannot run before risk approval, and SAR JSON cannot be created before final compliance approval.

## Project structure

```text
starter/
├── data/                         Synthetic CSV source data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_development.ipynb
│   └── 03_workflow_integration.ipynb
├── outputs/
│   ├── audit_logs/               Append-only JSONL and summaries
│   ├── filed_sars/               Human-approved educational SAR JSON
│   └── metrics/                  Timing and staged-cost evidence
├── src/
│   ├── foundation_sar.py         Schemas, loader, audit logger
│   ├── risk_analyst_agent.py     Five-step risk analysis
│   ├── compliance_officer_agent.py  ReACT narrative generation
│   └── sar_workflow.py           State machine and human gates
└── tests/                        Supplied evaluator tests

sar_app/
├── server.py                     Thin FastAPI adapter
└── index.html                    Self-contained review interface
```

## Setup

Use the project environment described by the course workspace, or install the requirements:

```bash
python -m pip install -r requirements.txt
```

Optional OpenAI-compatible configuration:

```text
OPENAI_API_KEY=<credential>
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4
SAR_AGENT_MODE=auto
```

Credentials remain in `.env` and are never logged. Set `SAR_AGENT_MODE=deterministic` for a fully local demonstration. In `auto` mode, a configured OpenAI client is used with deterministic recovery after API or structured-output failure.

## Run the application

From the project root:

```bash
python -m uvicorn sar_app.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Workflow:

1. Choose a customer and inspect the validated case facts.
2. Run the five-step Risk Analyst review.
3. Record a named human approval or rejection with rationale.
4. If approved, review the factual Compliance Officer narrative and citations.
5. Record the final human decision.
6. Approved simulations create SAR JSON and update audit and metrics artifacts.

## Agent controls

### Risk Analyst

- Required reasoning sequence: Data Review, Pattern Recognition, Regulatory Mapping, Risk Quantification, Classification Decision.
- Allowed classifications: `Structuring`, `Sanctions`, `Fraud`, `Money_Laundering`, `Other`.
- Confidence is constrained to 0–1; risk level is constrained to Low/Medium/High/Critical.
- Deterministic evidence excludes synthetic transaction-ID hints.
- Sanctions classification requires confirmed authoritative OFAC Sanctions List Service evidence.

### Compliance Officer

- Uses a ReACT Reasoning/Action/Conclusion contract.
- Includes factual who, what, when, where, and why elements.
- Uses calibrated language and preserves material unknowns.
- Enforces the course simulation limit of 120 words.
- Replaces model-proposed citations with a source-controlled classification-specific set.

Primary-source research and regulatory limitations are documented in [`../regulatory_research.md`](../regulatory_research.md).

## Notebooks

All three notebooks are complete, executable, and saved with outputs:

- `01_data_exploration.ipynb`: data quality, relationships, pattern exploration, validation, and audit logging.
- `02_agent_development.ipynb`: five-step Risk Analyst and ReACT Compliance Officer demonstrations.
- `03_workflow_integration.ipynb`: approved and rejected human-gate paths, SAR generation, audit evidence, timing, and staged cost metrics.

Execute any notebook from the project root:

```bash
jupyter nbconvert --to notebook --execute --inplace starter/notebooks/03_workflow_integration.ipynb
```

## Verification

```bash
python -m ruff check .
python -m pytest -q
python scripts/smoke_api.py
```

The smoke script expects the application to be running in deterministic mode at `http://127.0.0.1:8000`.

## Final submission evidence

| Checklist criterion | Evidence |
|---|---|
| Validated foundation schemas | `src/foundation_sar.py`; foundation tests |
| Five risk classifications | `src/risk_analyst_agent.py`; risk tests and notebook 02 |
| Narratives no longer than 120 words | `src/compliance_officer_agent.py`; compliance tests and notebooks 02–03 |
| CSV-to-SAR end-to-end execution | notebook 03 and `scripts/smoke_api.py` |
| Human compliance oversight | `src/sar_workflow.py`; FastAPI gate endpoints; interface controls |
| Complete audit trails | `outputs/audit_logs/*.jsonl` and `outputs/audit_logs/summary.json` |
| Complete SAR documents | `outputs/filed_sars/*.json` |
| Cost and efficiency metrics | notebook 03 and `outputs/metrics/workflow_metrics.json` |
| Documentation, type hints, and comments | `src/`, this README, and focused tests |

## Limitations

- The data and filings are synthetic educational artifacts.
- The project does not perform live OFAC screening; it accepts only explicitly supplied confirmed screening evidence.
- Regulatory triggers depend on institution type and facts. The source-controlled citations do not replace legal review.
- Metrics report educational cost estimates, not provider billing records.
- Workflow state is in memory; generated audit, metric, and SAR artifacts persist on disk.

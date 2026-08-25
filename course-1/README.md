# Educational AML/SAR Review System

An suspicious-activity review workflow built with Python, Pydantic, pandas, FastAPI, OpenAI-compatible agents, and a vanilla HTML interface.

## Workflow

```text
CSV data → validated case → Risk Analyst → human risk gate
         → Compliance Officer → human compliance gate → SAR JSON + audit outputs
```

Human approval is authoritative at both gates. Compliance generation cannot run before risk approval, and a SAR cannot be created before final compliance approval.

## Repository layout

- `starter/data/` — synthetic customer, account, and transaction CSV files
- `starter/src/` — schemas, data loading, agents, and workflow state machine
- `starter/sar_app/` — FastAPI adapter and self-contained review interface
- `starter/notebooks/` — data exploration, agent development, and integration demonstrations
- `starter/tests/` — supplied evaluator tests
- `tests/` — project-level regression and submission checks
- `scripts/smoke_api.py` — end-to-end API smoke test
- `outputs/` — generated audit, SAR, and metrics artifacts

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r starter/requirements.txt
```

Optional OpenAI-compatible configuration can be placed in `starter/.env`:

```text
OPENAI_API_KEY=<credential>
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4
SAR_AGENT_MODE=auto
```

Use `SAR_AGENT_MODE=deterministic` for a fully local demonstration. Credentials are not logged.

## Run the application

```bash
SAR_AGENT_MODE=deterministic python -m uvicorn sar_app.server:app --app-dir starter --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, then:

1. Select a customer and inspect the validated case facts.
2. Run the five-step Risk Analyst review.
3. Record a named human risk decision and rationale.
4. If approved, review the Compliance Officer narrative and citations.
5. Record the final human compliance decision.
6. Inspect generated SAR, audit, and metrics artifacts.

## Agent contracts

### Risk Analyst

- Follows Data Review, Pattern Recognition, Regulatory Mapping, Risk Quantification, and Classification Decision.
- Supports `Structuring`, `Sanctions`, `Fraud`, `Money_Laundering`, and `Other` classifications.
- Constrains confidence to 0–1 and risk level to Low, Medium, High, or Critical.
- Requires confirmed authoritative evidence for a Sanctions classification.

### Compliance Officer

- Uses a ReACT Reasoning/Action/Conclusion contract.
- Includes factual who, what, when, where, and why elements.
- Preserves material unknowns and uses calibrated language.
- Enforces the 120-word course-simulation limit.
- Applies source-controlled, classification-specific citations.

## Verification

```bash
python -m ruff check .
python -m pytest -q
```

With the application running in deterministic mode, run:

```bash
python scripts/smoke_api.py
```

Execute the integration notebook from the repository root with:

```bash
jupyter nbconvert --to notebook --execute --inplace starter/notebooks/03_workflow_integration.ipynb
```

## Limitations

- Data and filings are synthetic educational artifacts.
- Live OFAC screening is not performed; only explicitly supplied confirmed evidence is accepted.
- Regulatory triggers depend on institution type and facts; citations do not replace legal review.
- Metrics are educational cost estimates, not provider billing records.
- Workflow state is in memory; generated audit, metric, and SAR artifacts persist on disk.

Further regulatory context is documented in [`regulatory_research.md`](regulatory_research.md).
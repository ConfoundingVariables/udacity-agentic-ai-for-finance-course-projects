from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from starter.src import create_vocareum_openai_client
from starter.src.compliance_officer_agent import ComplianceOfficerAgent
from starter.src.foundation_sar import DataLoader, ExplainabilityLogger
from starter.src.risk_analyst_agent import RiskAnalystAgent
from starter.src.sar_workflow import HumanDecisionInput, SarWorkflow

_store: DataLoader | None = None
_workflow: SarWorkflow | None = None
_agent_mode = "deterministic"


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global _agent_mode, _store, _workflow
    root = Path(__file__).resolve().parents[1]
    starter = root / "starter"
    load_dotenv(starter / ".env")

    client = None
    _agent_mode = "deterministic"
    configured_mode = os.getenv("SAR_AGENT_MODE", "auto").lower()
    if configured_mode != "deterministic" and os.getenv("OPENAI_API_KEY"):
        try:
            client = create_vocareum_openai_client()
            _agent_mode = "openai_with_deterministic_fallback"
        except (ImportError, ValueError):
            _agent_mode = "deterministic"

    output_root = Path(os.getenv("SAR_OUTPUT_ROOT", str(starter / "outputs")))
    audit_path = output_root / "audit_logs" / "workflow.jsonl"
    logger = ExplainabilityLogger(audit_path)
    _store = DataLoader.load(starter / "data")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    _workflow = SarWorkflow(
        _store,
        RiskAnalystAgent(
            client,
            logger,
            model=model,
            fallback_on_error=True,
        ),
        ComplianceOfficerAgent(
            client,
            logger,
            model=model,
            fallback_on_error=True,
        ),
        logger,
        output_root,
    )
    logger.log(
        component="Application",
        action="startup",
        case_id="SYSTEM",
        output_data={"agent_mode": _agent_mode, **_store.summary().model_dump()},
    )
    yield
    _workflow = None
    _store = None


app = FastAPI(title="AML/SAR Review", version="1.0.0", lifespan=lifespan)


def store() -> DataLoader:
    if _store is None:
        raise HTTPException(503, "dataset is not loaded")
    return _store


def workflow() -> SarWorkflow:
    if _workflow is None:
        raise HTTPException(503, "workflow is not loaded")
    return _workflow


def workflow_record(case_id: str):
    try:
        return workflow().get_record(case_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/summary")
def summary():
    return {
        **store().summary().model_dump(mode="json"),
        "agent_mode": _agent_mode,
        "workflow_metrics": workflow().metrics().model_dump(mode="json"),
    }


@app.get("/api/cases")
def cases(limit: int = Query(150, ge=1, le=500)):
    return store().list_cases(limit)


@app.get("/api/cases/{customer_id}")
def case(customer_id: str):
    try:
        return store().get_case(customer_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(404, f"case not found: {customer_id}") from exc


@app.post("/api/cases/{customer_id}/analyze")
def analyze_case(customer_id: str):
    try:
        return workflow().analyze_customer(customer_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(404, f"case not found: {customer_id}") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/workflows/{case_id}")
def get_workflow(case_id: str):
    return workflow_record(case_id).model_dump(mode="json")


@app.post("/api/workflows/{case_id}/risk-decision")
def review_risk(case_id: str, decision: HumanDecisionInput):
    workflow_record(case_id)
    try:
        return workflow().review_risk(case_id, decision).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/workflows/{case_id}/retry-compliance")
def retry_compliance(case_id: str):
    workflow_record(case_id)
    try:
        return workflow().retry_compliance(case_id).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/workflows/{case_id}/compliance-decision")
def review_compliance(case_id: str, decision: HumanDecisionInput):
    workflow_record(case_id)
    try:
        return workflow().review_compliance(case_id, decision).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/metrics")
def metrics():
    return workflow().metrics().model_dump(mode="json")


@app.get("/api/audit")
def audit(limit: int = Query(100, ge=1, le=1000)):
    return workflow().audit_entries(limit)


@app.get("/api/outputs")
def outputs():
    return workflow().list_outputs()

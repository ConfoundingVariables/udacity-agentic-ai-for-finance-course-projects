"""Capture deterministic offline feedback paths without claiming live LLM work."""

import json
from pathlib import Path

from config import Config
from main import SWIFTProcessingSystem


def main():
    Config.OPENAI_API_KEY = "real-api-key"  # Recognized offline placeholder.
    Config.ALLOW_OFFLINE_FALLBACK = True
    Config.LLM_CACHE_ENABLED = False
    system = SWIFTProcessingSystem()
    invalid = {
        "message_id": "CORRECTION001", "message_type": "MT103",
        "reference": "REFERENCE-TOO-LONG-FOR-SWIFT", "amount": "100.00",
        "currency": "USD", "sender_bic": "CHASUS33XXX",
        "receiver_bic": "DEUTDEFFXXX",
    }
    corrected = system.process_with_evaluator_optimizer([dict(invalid)])[0]
    assert corrected["validation_status"] == "INVALID"
    assert corrected["correction_evidence"]["attempted"]
    assert not corrected["correction_evidence"]["applied"]
    suspicious = {
        "message_id": "CHAIN001", "message_type": "MT103",
        "reference": "CHAIN001", "amount": "9000000.00", "currency": "USD",
        "sender_bic": "TESTRU33XXX", "receiver_bic": "TESTRU33XXX",
        "remittance_info": "urgent secret transfer",
    }
    screened = system.process_with_parallelization([suspicious])
    assert screened[0]["fraud_status"] == "FRAUDULENT"
    chain = system.process_with_prompt_chaining(screened)
    stages = ["initial_screening", "technical_analysis", "risk_assessment",
              "compliance_review", "final_review"]
    assert chain["escalation_evidence"]["completed_stages"] == stages
    grouped = system.orchestrator_worker.process_with_orchestrator([
        {"message_id": "GROUP001", "amount": "100.00", "currency": "USD"},
        {"message_id": "GROUP002", "amount": "200.00", "currency": "EUR"},
        {"message_id": "GROUP003", "amount": "50.00", "currency": "USD"},
    ])
    memberships = [mid for g in grouped["orchestrator_analysis"]["groups"]
                   for mid in g["message_ids"]]
    assert sorted(memberships) == ["GROUP001", "GROUP002", "GROUP003"]
    assert all(g["status"] == "completed" for g in grouped["group_results"])
    path = Path(__file__).parent / "reports" / "feedback_evidence.json"
    evidence = json.loads(path.read_text()) if path.exists() else {}
    evidence["runner"] = {
        "command": ".venv/bin/python capture_feedback_evidence.py",
        "mode": "offline", "live_attempts_repeated": False,
        "note": "Existing live-attempt evidence is retained; this command makes no API calls.",
    }
    evidence["offline"] = {
        "correction_input": invalid, "correction_output": corrected,
        "screened_messages": screened, "chain_results": chain,
        "grouping_result": grouped,
    }
    path.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(f"Evidence saved: {path}")
    print("Verified: offline correction no-op; fraudulent screening; five chain stages; exact group partition.")


if __name__ == "__main__":
    main()

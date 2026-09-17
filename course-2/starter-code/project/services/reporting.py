"""
Reporting service for the SWIFT processing system.

Computes the Straight-Through-Processing (STP) rate and other operational
metrics from a processed batch, and persists both machine-readable (JSON) and
human-readable (text) reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


def _amount_of(message: Dict) -> float:
    try:
        return float(''.join(c for c in str(message.get('amount', '0'))
                             if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0


class ReportGenerator:
    """Generates structured and textual reports for a processed batch."""

    def __init__(self, output_dir: str = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = output_dir or os.path.join(base, "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Compute validation, fraud and STP metrics for a batch."""
        total = len(messages)
        valid = sum(1 for m in messages if m.get('validation_status') == 'VALID')
        invalid = total - valid
        fraudulent = sum(1 for m in messages if m.get('fraud_status') == 'FRAUDULENT')
        clean = sum(1 for m in messages if m.get('fraud_status') == 'CLEAN')

        # STP: transactions that passed validation AND cleared fraud screening
        # (i.e. processed with zero manual intervention).
        stp_count = sum(
            1 for m in messages
            if m.get('validation_status') == 'VALID'
            and m.get('fraud_status') == 'CLEAN'
        )
        stp_rate = round((stp_count / total) * 100, 2) if total else 0.0

        return {
            "total_messages": total,
            "valid_messages": valid,
            "invalid_messages": invalid,
            "fraudulent_messages": fraudulent,
            "clean_messages": clean,
            "stp_count": stp_count,
            "stp_rate_pct": stp_rate,
            "total_value": round(sum(_amount_of(m) for m in messages), 2),
        }

    def build_report(
        self,
        report_name: str,
        messages: List[Dict],
        chain_results: Dict = None,
        orchestrator_sets: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Assemble the full report dictionary."""
        metrics = self.compute_metrics(messages)

        per_message = [
            {
                "message_id": m.get('message_id'),
                "message_type": m.get('message_type'),
                "amount": m.get('amount'),
                "currency": m.get('currency'),
                "validation_status": m.get('validation_status'),
                "fraud_status": m.get('fraud_status'),
                "fraud_score": m.get('fraud_score'),
                "fraud_decision": m.get('fraud_decision'),
                "fraud_reasons": m.get('fraud_reasons', []),
            }
            for m in messages
        ]

        chain_summary = {}
        if chain_results:
            final = chain_results.get('final_review', {})
            chain_summary = final.get('batch_summary', {}) if isinstance(final, dict) else {}

        orchestrator_summaries = []
        for os_set in (orchestrator_sets or []):
            orchestrator_summaries.append({
                "filter": os_set.get('filter'),
                "message_count": os_set.get('message_count'),
                "summary": (os_set.get('result') or {}).get('summary'),
                "task_count": len((os_set.get('result') or {}).get('task_results', [])),
            })

        return {
            "report_name": report_name,
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "prompt_chain_summary": chain_summary,
            "orchestrator_sets": orchestrator_summaries,
            "transactions": per_message,
        }

    def save(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Persist the report as JSON and a text summary. Returns file paths."""
        name = report.get("report_name", "report")
        json_path = os.path.join(self.output_dir, f"{name}.json")
        txt_path = os.path.join(self.output_dir, f"{name}.txt")

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(self._render_text(report))

        return {"json": json_path, "txt": txt_path}

    @staticmethod
    def _render_text(report: Dict[str, Any]) -> str:
        m = report["metrics"]
        lines = [
            "=" * 60,
            f"SWIFT PROCESSING REPORT: {report['report_name']}",
            f"Generated: {report['generated_at']}",
            "=" * 60,
            "",
            "OPERATIONAL METRICS",
            "-" * 60,
            f"Total messages       : {m['total_messages']}",
            f"Valid / Invalid      : {m['valid_messages']} / {m['invalid_messages']}",
            f"Clean / Fraudulent   : {m['clean_messages']} / {m['fraudulent_messages']}",
            f"STP count            : {m['stp_count']}",
            f"STP RATE             : {m['stp_rate_pct']}%",
            f"Total value          : ${m['total_value']:,.2f}",
            "",
        ]

        if report.get("prompt_chain_summary"):
            lines.append("PROMPT-CHAIN FINAL REVIEW")
            lines.append("-" * 60)
            for k, v in report["prompt_chain_summary"].items():
                lines.append(f"{k}: {v}")
            lines.append("")

        if report.get("orchestrator_sets"):
            lines.append("ORCHESTRATOR REPORT SETS")
            lines.append("-" * 60)
            for s in report["orchestrator_sets"]:
                lines.append(
                    f"[{s['filter']}] {s['message_count']} msgs, "
                    f"{s['task_count']} tasks -> {s['summary']}"
                )
            lines.append("")

        lines.append("TRANSACTIONS")
        lines.append("-" * 60)
        for t in report["transactions"]:
            lines.append(
                f"{t['message_id']} | {t['message_type']} | "
                f"{t['amount']} {t['currency']} | "
                f"val={t['validation_status']} | fraud={t['fraud_status']} "
                f"({t['fraud_score']}) | decision={t['fraud_decision']}"
            )
        lines.append("")
        return "\n".join(lines)

    def generate(
        self,
        report_name: str,
        messages: List[Dict],
        chain_results: Dict = None,
        orchestrator_sets: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Build, persist and return a report plus its file paths."""
        report = self.build_report(report_name, messages, chain_results, orchestrator_sets)
        paths = self.save(report)
        report["_paths"] = paths
        return report

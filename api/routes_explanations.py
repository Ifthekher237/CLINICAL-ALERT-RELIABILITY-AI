"""Rule-based alert explanation endpoint for the local simulation API.

This route intentionally does not call an LLM. LLM explanations are later
roadmap work; Step 17 uses transparent fields from existing artifacts.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes_alerts import SIMULATION_ONLY_NOTE, build_alert_detail


router = APIRouter(tags=["explanations"])


@router.get("/explain-alert/{alert_id}")
def explain_alert(alert_id: str) -> dict[str, Any]:
    """Generate a simple rule-based explanation for one simulated alert."""
    alert = build_alert_detail(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")

    explanation = _generate_rule_based_explanation(alert)
    return {
        "alert_id": alert.get("alert_id"),
        "patient_id": alert.get("patient_id"),
        "severity": alert.get("severity"),
        "alert_type": alert.get("alert_type"),
        "trigger_reason": alert.get("trigger_reason"),
        "audit_status": alert.get("audit_status"),
        "guardrail_status": alert.get("guardrail_decision"),
        "fatigue_status": alert.get("final_alert_status"),
        "fatigue_action": alert.get("fatigue_action"),
        "simulated_response": alert.get("simulated_response"),
        "explanation": explanation,
        "safety_note": (
            "This is a rule-based explanation of simulated data, not clinical advice."
        ),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def _generate_rule_based_explanation(alert: dict[str, Any]) -> str:
    """Create a compact explanation from existing alert/audit/workflow fields."""
    pieces: list[str] = []
    severity = str(alert.get("severity") or "unknown")
    alert_type = str(alert.get("alert_type") or "unknown")
    pieces.append(f"This simulated {severity} alert was categorized as {alert_type}.")

    trigger_reason = alert.get("trigger_reason")
    if trigger_reason:
        pieces.append(f"It was triggered because: {trigger_reason}.")

    audit_status = alert.get("audit_status")
    if audit_status:
        pieces.append(f"The audit layer labeled it as {audit_status}.")

    final_status = alert.get("final_alert_status")
    fatigue_action = alert.get("fatigue_action")
    if final_status or fatigue_action:
        pieces.append(
            f"Fatigue review marked the alert as {final_status or 'unknown'} "
            f"with action {fatigue_action or 'unknown'}."
        )

    simulated_response = alert.get("simulated_response")
    if simulated_response:
        pieces.append(f"The simulated workflow response was {simulated_response}.")

    if alert.get("critical_flag") in {True, 1, "True", "true"}:
        pieces.append("The alert carried a critical flag in the simulation.")

    return " ".join(pieces)

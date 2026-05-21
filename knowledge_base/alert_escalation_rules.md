# Alert Escalation Rules

These rules describe simulated alert escalation behavior for the project. They are not clinical instructions.

## Critical And Immediate Alerts

- Critical alerts require immediate escalation in the simulated workflow.
- Alerts with `critical_flag=True` must remain active and must not be suppressed.
- Immediate safety-priority alerts require urgent human review.
- Any ignored critical or immediate alert should trigger reliability review.

## High Severity Alerts

- High severity alerts usually require urgent review.
- High alerts may be escalated or retained, but should not be grouped away or delayed by fatigue reduction.
- Worsening repeated patterns should be escalated rather than treated as low-value repetition.

## Medium And Low Alerts

- Medium and low alerts can be monitored, grouped, delayed, or reviewed depending on risk score, actionability, repeated patterns, and guardrail status.
- Repeated low-value alerts may be grouped only when safety guardrails allow it.
- Low or medium alerts with unclear evidence, noisy signals, or anomaly-only triggers may require manual verification.

## Uncertainty And Manual Verification

- Uncertain alerts should preserve a human-review path.
- Sensor noise, missing values, drift, low confidence, and weak trigger reasons should be explained as uncertainty sources.
- Escalation recommendations are simulated workflow labels, not treatment recommendations.

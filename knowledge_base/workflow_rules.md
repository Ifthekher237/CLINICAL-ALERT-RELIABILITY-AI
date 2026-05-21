# Workflow Rules

These rules define the simulated alert workflow used by the project. They do not describe real clinical operations.

## Workflow Stages

- Alerts may move through triage queue, nurse review, clinician review, escalated review, and closed states.
- Critical and high-priority alerts should move quickly toward clinician or escalated review in the simulation.
- Low or medium alerts may be monitored or grouped only when safety checks are satisfied.

## Alert Burden

- Repeated low-value alerts can increase alert fatigue and workflow burden.
- Fatigue reduction may group, delay, or downgrade low-risk repeated alerts while keeping an audit trail.
- Critical alerts and safety-sensitive alerts must remain active.
- Alert volume and repeated patterns should be monitored over time.

## Response Timing

- Response delays matter because delayed review can reduce simulated workflow reliability.
- Night-time or high-burden periods may produce slower simulated responses.
- Ignored, delayed, and marked-false responses should feed reliability monitoring.

## Documentation

- Workflow decisions should include human-readable reasons.
- The simulation should preserve alert rows rather than deleting them.
- Explanations should describe workflow status without implying clinical care decisions.

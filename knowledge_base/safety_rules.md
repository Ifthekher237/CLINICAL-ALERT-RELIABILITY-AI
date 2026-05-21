# Safety Rules

These rules define the safety boundary for the simulated clinical alert reliability project. They are internal project rules for software engineering experiments only.

## Simulation Boundary

- The system is a simulated healthcare AI engineering prototype.
- The system is not clinically validated and is not a medical device.
- The system must not be used for real patient monitoring, diagnosis, treatment, triage, or clinical decision-making.
- All outputs are simulated software signals and require cautious interpretation.

## No Diagnosis Or Treatment

- Explanations must not diagnose a patient or imply a patient has a disease.
- Explanations must not recommend medication, treatment, procedures, or clinical interventions.
- LLM-generated text must explain system outputs only.
- Any uncertainty or safety-sensitive result must preserve human review.

## Human Review

- Critical, immediate, high-risk, or uncertain alerts require human review in the simulation workflow.
- Sensor instability or noisy signals should be flagged for manual verification rather than ignored.
- Critical alerts must never be suppressed by fatigue-reduction or threshold-adjustment logic.
- No automated model, rule, or LLM output replaces clinician judgment.

## Auditability

- Alert changes, grouping, delay, escalation, and simulated model-update recommendations should be logged.
- Explanations should include uncertainty and simulation-only safety notes.
- Unsafe wording should be replaced with constrained project-language.

# System Limitations

This project is a simulated healthcare AI reliability platform for portfolio and engineering practice.

## Simulated Data Only

- Data is simulated and does not represent real patient records.
- Model outputs, alerts, workflow responses, and explanations are synthetic project artifacts.
- The project is not clinically validated and should not be used in patient care.

## Model And Monitoring Limits

- Baseline models and rule-based logic are educational prototypes.
- Drift detection, reliability scores, and fatigue-reduction metrics demonstrate engineering patterns, not clinical safety.
- Alert thresholds and update recommendations are simulation-only and require human review.
- Severe drift, high alert burden, or ignored critical alerts should reduce confidence in the simulated system.

## LLM Limits

- LLM outputs can be incomplete, overconfident, or unsafe if unconstrained.
- LLMs must be constrained to explain project outputs and internal rules only.
- LLMs must not diagnose, recommend treatment, or replace clinicians.
- Retrieval-augmented context should come from local project documents only.

## Presentation Boundary

- The system should be described as a pre-deployment reliability and safety-thinking prototype.
- It should not be presented as a working clinical decision support system.
- Any demo should include simulation-only and human-review disclaimers.

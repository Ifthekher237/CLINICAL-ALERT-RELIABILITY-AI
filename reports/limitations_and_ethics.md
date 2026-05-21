# Ethics, Safety, and Limitations Report

## 1. Simulation-Only Disclaimer

`clinical-alert-reliability-ai` is a simulated research and engineering prototype. It is designed to demonstrate healthcare AI systems engineering concepts such as alert reliability, workflow simulation, drift monitoring, alert fatigue reduction, and safety-aware review logic.

The project is not intended for real patient care. All datasets, alerts, workflow responses, outcomes, failure modes, scenario tests, explanations, recommendations, metrics, API responses, and dashboard views are simulated engineering artifacts.

## 2. Not Clinically Validated

This project is not clinically validated and is not a medical device. It has not been evaluated in a hospital environment, reviewed through a clinical safety process, tested prospectively, or validated with real-world patient outcomes.

The results should be interpreted only as evidence that the software pipeline can simulate and inspect alert-reliability behavior. They should not be interpreted as evidence of real-world medical safety, effectiveness, or readiness for clinical deployment.

## 3. Simulated Data Limitations

The patient-monitoring data is synthetic. It was generated to exercise engineering logic such as missing values, noisy sensor readings, deterioration-like patterns, alerts, workflow delays, and drift.

Synthetic data is useful for prototyping, but it cannot capture the full complexity of real patient physiology, care-team behavior, sensor variability, documentation practices, hospital workflows, or clinical context. Model performance, alert rates, and workflow metrics in this project may not transfer to real settings.

## 4. No Real Patient Data Used

No real patient data is used in this repository. The project does not include protected health information, patient identifiers, clinical notes, hospital records, or real monitoring feeds.

This reduces privacy risk for a portfolio project, but it also means the system has not been evaluated against real-world variation, demographic diversity, data quality issues, or operational constraints.

## 5. Human-in-the-Loop Requirement

The project is designed around human review. Safety-sensitive alerts, uncertain alerts, severe drift, degraded reliability, critical alert conditions, and ambiguous LLM/RAG outputs should require human review in the simulated workflow.

Any future non-simulated version would need formal clinical governance, institutional review, safety analysis, usability testing, and qualified human oversight before it could be considered for any operational setting.

## 6. Alert Fatigue Safety Risks

Alert fatigue is a central risk explored by the project. Excessive low-value, repeated, unclear, or noisy alerts can increase workflow burden and make it harder for reviewers to prioritize important signals.

The fatigue-reduction module groups, delays, or downgrades selected low-risk simulated alerts while preserving rows for auditability. This is an engineering demonstration only. In real settings, alert reduction can create safety risks if relevant signals are hidden, delayed, or deprioritized incorrectly.

## 7. Critical Alert Preservation Rule

The simulated system is safety-first: critical alerts, `critical_flag=True` alerts, immediate-priority alerts, and immediate-escalation alerts must not be suppressed by fatigue-reduction logic.

This rule is included to demonstrate conservative safety thinking. It does not prove that the system correctly identifies all critical situations. It only shows that once an alert is labeled critical inside the simulation, downstream reduction logic is designed to preserve it.

## 8. LLM/RAG Limitations

The LLM client, explanation generator, RAG engine, and action recommender are constrained to support text and workflow context. They do not provide clinical judgment, medical advice, or autonomous decisions.

The RAG engine retrieves from local project knowledge-base documents such as safety rules, workflow rules, escalation rules, and system limitations. It does not retrieve external clinical guidelines, live medical literature, or hospital-specific protocols.

## 9. Risk of Hallucination or Incomplete Explanations

LLM-generated or fallback explanations can be incomplete, overly general, or misleading if interpreted outside the simulation boundary. Even rule-based explanations may omit relevant context because they only use available project artifacts.

For this reason, explanations are labeled as simulation-only support text. They should be reviewed as engineering summaries, not as authoritative medical reasoning.

## 10. Drift and Reliability Limitations

The reliability monitor and drift detector use transparent simulated metrics and time-window comparisons. They can flag changes in alert volume, response behavior, vital-sign distributions, reliability scores, and drift scores.

These modules do not guarantee that all important failures will be detected. Thresholds and scoring rules are simplified, and severe drift only indicates that review is needed in the simulated system. It does not automatically identify a root cause or justify automatic model changes.

## 11. Privacy Considerations

Because this project uses synthetic data only, it avoids handling real patient information. If future versions use public de-identified datasets or institutional data, privacy controls would need to be strengthened.

Future work with real or de-identified health data should consider data governance, access controls, retention rules, audit logging, de-identification quality, consent or institutional approval requirements, and secure storage practices.

## 12. Bias and Fairness Limitations

The current simulator does not model demographic groups, social determinants, device differences, care-unit variation, comorbidity patterns, or subgroup-specific workflow differences. As a result, the project cannot evaluate fairness or subgroup performance.

Future versions would need explicit subgroup definitions, fairness metrics, representative datasets, and careful review of whether alerts or workflow recommendations behave differently across populations or care contexts.

## 13. Future Validation Needs

Before any real-world use could be considered, a substantially different validation process would be required. This would include:

- Testing with appropriate de-identified real-world datasets.
- Calibration and threshold analysis.
- Prospective workflow evaluation in a safe non-clinical or shadow-mode setting.
- Review by qualified clinical, safety, privacy, and ethics stakeholders.
- Bias and subgroup performance analysis.
- Human factors and usability evaluation.
- Robust monitoring, logging, failure response, and governance processes.

This project does not complete those validation steps.

## 14. Safe Future Work

Safe future work should keep the project framed as a pre-deployment engineering and reliability prototype. Useful next directions include:

- Adding model cards and data cards.
- Improving calibration and threshold-review documentation.
- Testing with public de-identified datasets where appropriate.
- Expanding scenario tests and failure-mode simulations.
- Improving dashboard clarity for reviewers.
- Adding CI for automated tests.
- Evaluating assumptions with clinical collaborators in a safe, non-clinical setting.
- Strengthening privacy, fairness, and governance documentation before using any real-world health data.

The project should continue to avoid clinical claims, autonomous decision-making, or any suggestion that simulated outputs are suitable for real patient care.

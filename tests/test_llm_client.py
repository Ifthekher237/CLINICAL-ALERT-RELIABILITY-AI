"""Focused tests for Step 18 safe LLM client."""

from __future__ import annotations

from dataclasses import fields

from src.llm.llm_client import (
    LLMClient,
    LLMResponse,
    create_llm_client,
    get_default_healthcare_ai_system_prompt,
)


def test_default_client_can_be_created() -> None:
    client = create_llm_client()

    assert isinstance(client, LLMClient)
    assert client.model_name == "llama3.1"
    assert client.backend == "ollama"
    assert client.allow_fallback is True


def test_safe_system_prompt_contains_no_doctor_guardrails() -> None:
    prompt = get_default_healthcare_ai_system_prompt().lower()

    assert "simulated healthcare ai" in prompt
    assert "must not diagnose" in prompt
    assert "must not recommend treatment" in prompt
    assert "must not replace clinicians" in prompt
    assert "human review" in prompt


def test_build_safe_prompt_includes_safety_constraints() -> None:
    client = LLMClient()
    safe_prompt = client.build_safe_prompt("Explain this simulated alert.").lower()

    assert "explain only system outputs" in safe_prompt
    assert "do not diagnose" in safe_prompt
    assert "do not" in safe_prompt and "recommend treatment" in safe_prompt
    assert "human review" in safe_prompt
    assert "explain this simulated alert" in safe_prompt


def test_fallback_response_returns_failure_and_fallback_used() -> None:
    client = LLMClient()
    response = client.fallback_response("test prompt", error_message="connection failed")

    assert response.success is False
    assert response.fallback_used is True
    assert response.backend == "fallback"
    assert response.error_message == "connection failed"
    assert "cannot diagnose" in response.response_text.lower()


def test_generate_does_not_crash_when_ollama_unavailable() -> None:
    client = LLMClient(base_url="http://127.0.0.1:9", timeout_seconds=1)
    response = client.generate("Explain why human review may be needed.", max_tokens=20)

    assert isinstance(response, LLMResponse)
    assert response.success is False
    assert response.fallback_used is True
    assert response.backend == "fallback"


def test_llm_response_contains_required_fields() -> None:
    response_fields = {field.name for field in fields(LLMResponse)}

    assert {
        "prompt",
        "response_text",
        "model_name",
        "backend",
        "success",
        "fallback_used",
        "safety_note",
        "error_message",
    }.issubset(response_fields)


def test_invalid_backend_falls_back_safely() -> None:
    client = LLMClient(backend="paid_api_that_is_not_supported", timeout_seconds=1)
    response = client.generate("Explain a simulated monitoring output.")

    assert response.success is False
    assert response.fallback_used is True
    assert response.backend == "fallback"
    assert "unsupported" in str(response.error_message).lower()


def test_malformed_ollama_response_falls_back_safely(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self) -> dict:
            return {"unexpected": "missing response text"}

    class FakeRequests:
        @staticmethod
        def post(*args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "src.llm.llm_client._import_requests",
        lambda: FakeRequests,
    )

    client = LLMClient(timeout_seconds=1)
    response = client.generate("Explain a simulated alert.")

    assert response.success is False
    assert response.fallback_used is True
    assert "missing" in str(response.error_message).lower()


def test_successful_mocked_ollama_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"response": "Safe explanation."}'

        def json(self) -> dict:
            return {"response": "Safe explanation."}

    class FakeRequests:
        @staticmethod
        def post(*args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "src.llm.llm_client._import_requests",
        lambda: FakeRequests,
    )

    client = LLMClient(timeout_seconds=1)
    response = client.generate("Explain a simulated alert.")

    assert response.success is True
    assert response.fallback_used is False
    assert response.backend == "ollama"
    assert response.response_text == "Safe explanation."


def test_no_paid_api_key_is_required(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_llm_client()

    assert client.backend == "ollama"
    assert not hasattr(client, "api_key")

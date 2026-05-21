"""Safe local LLM client wrapper for simulated healthcare AI explanations.

Step 18 only builds the reusable client. It can call a local Ollama model when
available and falls back safely when Ollama, the model, or the requests library
is unavailable. The client must not act as a doctor or provide treatment advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_MODEL_NAME = "llama3.1"
DEFAULT_BACKEND = "ollama"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 30
SAFETY_NOTE = (
    "Simulation-only healthcare AI support text. Not a diagnosis, not treatment "
    "advice, and not a substitute for clinician review."
)


@dataclass
class LLMResponse:
    """Structured response from the local LLM wrapper."""

    prompt: str
    response_text: str
    model_name: str
    backend: str
    success: bool
    fallback_used: bool
    safety_note: str
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly response metadata."""
        return asdict(self)


class LLMClient:
    """Safe wrapper around a local Ollama LLM with reliable fallback behavior."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.allow_fallback = allow_fallback

    def is_available(self) -> bool:
        """Return whether the configured local backend appears reachable."""
        if self.backend != "ollama":
            return False
        try:
            requests = _import_requests()
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.timeout_seconds, 3),
            )
            if response.status_code != 200:
                return False
            payload = response.json()
            models = payload.get("models", [])
            if not models:
                return True
            model_names = {
                str(model.get("name", "")).split(":")[0]
                for model in models
                if isinstance(model, dict)
            }
            return self.model_name.split(":")[0] in model_names or not model_names
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 300,
    ) -> LLMResponse:
        """Generate constrained support text or return a safe fallback response."""
        safe_prompt = self.build_safe_prompt(prompt, system_prompt)
        if self.backend != "ollama":
            return self.fallback_response(
                safe_prompt,
                error_message=f"Unsupported LLM backend: {self.backend}",
            )

        try:
            requests = _import_requests()
            payload = {
                "model": self.model_name,
                "prompt": safe_prompt,
                "stream": False,
                "options": {"num_predict": max(int(max_tokens), 1)},
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                return self.fallback_response(
                    safe_prompt,
                    error_message=f"Ollama returned HTTP {response.status_code}: {response.text}",
                )

            data = response.json()
            response_text = data.get("response")
            if not isinstance(response_text, str) or not response_text.strip():
                return self.fallback_response(
                    safe_prompt,
                    error_message="Ollama response was missing non-empty 'response' text.",
                )

            return LLMResponse(
                prompt=safe_prompt,
                response_text=response_text.strip(),
                model_name=self.model_name,
                backend="ollama",
                success=True,
                fallback_used=False,
                safety_note=SAFETY_NOTE,
                error_message=None,
            )
        except Exception as exc:
            return self.fallback_response(safe_prompt, error_message=str(exc))

    def build_safe_prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Combine caller prompt with healthcare AI safety constraints."""
        base_system_prompt = system_prompt or get_default_healthcare_ai_system_prompt()
        return (
            f"{base_system_prompt.strip()}\n\n"
            "User request for simulated system-support explanation:\n"
            f"{user_prompt.strip()}\n\n"
            "Answer only as a support explanation of system outputs. Mention uncertainty "
            "and human review when relevant. Do not diagnose, prescribe, or recommend treatment."
        )

    def fallback_response(
        self,
        prompt: str,
        error_message: str | None = None,
    ) -> LLMResponse:
        """Return a deterministic safe fallback response for reliable demos/tests."""
        if not self.allow_fallback:
            return LLMResponse(
                prompt=prompt,
                response_text=(
                    "LLM generation was unavailable and fallback mode is disabled. "
                    "Human review is required for any interpretation."
                ),
                model_name=self.model_name,
                backend=self.backend,
                success=False,
                fallback_used=False,
                safety_note=SAFETY_NOTE,
                error_message=error_message,
            )

        return LLMResponse(
            prompt=prompt,
            response_text=(
                "A local LLM response is unavailable, so this safe fallback is being used. "
                "For this simulated healthcare AI project, the system can only explain "
                "prototype outputs and safety rules. It cannot diagnose, recommend treatment, "
                "or replace clinician review. Any uncertain or safety-sensitive alert should "
                "be reviewed by a qualified human reviewer in the simulation workflow."
            ),
            model_name=self.model_name,
            backend="fallback",
            success=False,
            fallback_used=True,
            safety_note=SAFETY_NOTE,
            error_message=error_message,
        )


def get_default_healthcare_ai_system_prompt() -> str:
    """Return the default safety prompt for simulated healthcare AI explanations."""
    return (
        "You are a support assistant for a simulated healthcare AI engineering project. "
        "You must not diagnose patients. You must not recommend treatment. You must not "
        "replace clinicians or claim clinical validity. Explain only system outputs, "
        "prototype monitoring signals, uncertainty, and safety-review considerations. "
        "When risk or uncertainty is present, mention that human review is required."
    )


def create_llm_client(model_name: str = DEFAULT_MODEL_NAME) -> LLMClient:
    """Create the default local Ollama client for later explanation steps."""
    return LLMClient(model_name=model_name)


def run_llm_client_demo() -> None:
    """Run a tiny local/fallback demo without requiring Ollama."""
    client = LLMClient(timeout_seconds=3)
    available = client.is_available()
    response = client.generate(
        "Explain why a simulated alert might require human review.",
        max_tokens=120,
    )
    print(f"Ollama available: {available}")
    print(f"Backend used: {response.backend}")
    print(f"Model name: {response.model_name}")
    print(f"Success: {response.success}")
    print(f"Fallback used: {response.fallback_used}")
    print("Response text:")
    print(response.response_text)
    print("Safety note:")
    print(response.safety_note)
    if response.error_message:
        print("Error message:")
        print(response.error_message)


def _import_requests() -> Any:
    """Import requests lazily so tests still pass without optional local setup."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("The requests library is unavailable.") from exc
    return requests


if __name__ == "__main__":
    run_llm_client_demo()

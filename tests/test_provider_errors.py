"""The provider-error classifier names the cause and never repeats the text."""

from __future__ import annotations

import httpx
import pytest

from mira.exceptions import LLMError, NonRetriableLLMError, ProviderError
from mira.llm.provider_errors import ProviderErrorKind, classify_provider_error
from mira.runner_client import _safe_error

KEY = "sk-ant-api03-secretsecretsecretsecretsecret"


def _api_error(status: int, body: str) -> LLMError:
    return NonRetriableLLMError("api_error", status=status, body=body)


def _wrapped(cause: Exception) -> LLMError:
    """How the LLM providers report a failed call: the raw error as the cause."""
    try:
        raise cause
    except Exception as exc:
        try:
            raise LLMError("completion_failed", model="glm-5.2", error=exc) from exc
        except LLMError as wrapped:
            return wrapped


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_api_error(401, f'{{"error": "Invalid API key: {KEY}"}}'), "AUTH"),
        (_api_error(401, '{"error":{"message":"User not found","code":401}}'), "AUTH"),
        (
            NonRetriableLLMError("claude_auth_failed", detail=f"OAuth token has expired {KEY}"),
            "AUTH",
        ),
        (_api_error(403, "Forbidden"), "AUTH"),
        (_api_error(429, '{"type":"rate_limit_error"}'), "QUOTA"),
        (_api_error(402, "Insufficient Balance"), "QUOTA"),
        # A billing refusal sent as 401 is still billing, not a bad key.
        (_api_error(401, "Your credit balance is too low"), "QUOTA"),
        (NonRetriableLLMError("claude_rate_limited", detail="hit your session limit"), "QUOTA"),
        (
            _api_error(404, '{"error":{"code":"model_not_found","message":"no such"}}'),
            "MODEL",
        ),
        (_api_error(400, "The model `glm-9` does not exist"), "MODEL"),
        (
            _api_error(400, "This model's maximum context length is 128000 tokens"),
            "CONTEXT",
        ),
        (_api_error(400, "prompt is too long: 210000 tokens > 200000"), "CONTEXT"),
        (LLMError("api_error", status=502, body="<html>Bad Gateway</html>"), "NETWORK"),
        (LLMError("api_error", status=529, body='{"type":"overloaded_error"}'), "NETWORK"),
        (_wrapped(httpx.ConnectError("[Errno 111] Connection refused")), "NETWORK"),
    ],
)
def test_provider_errors_are_classified(exc: Exception, kind: str) -> None:
    assert classify_provider_error(exc) is ProviderErrorKind[kind]


@pytest.mark.parametrize(
    "exc",
    [
        # GitHub's and the Mira server's refusals are not about the model key.
        ProviderError("GitHub PATCH https://api.github.com/check-runs/1 → 401: Bad credentials"),
        RuntimeError("status: 429"),
        # A SHA or count that holds a status code is not a status.
        LLMError("claude_reported_error", exit_code=1, detail="commit 4290af1 has 401 lines"),
    ],
)
def test_other_errors_are_not_classified(exc: Exception) -> None:
    assert classify_provider_error(exc) is None


def test_the_safe_error_leads_with_the_cause_and_omits_the_provider_text() -> None:
    exc = _api_error(401, f'{{"error": "Invalid API key: {KEY}", "key": "{KEY}"}}')

    error = _safe_error(exc, "review")

    assert error == (
        "Review failed: the model provider rejected its credentials (invalid, expired or "
        "revoked key) (NonRetriableLLMError: LLM API error 401)"
    )
    assert KEY not in error and "Invalid API key" not in error


def test_an_unclassified_error_keeps_the_type_only() -> None:
    assert _safe_error(RuntimeError(f"boom {KEY}"), "review") == "RuntimeError"


def test_the_safe_error_names_a_job_that_is_not_a_review() -> None:
    error = _safe_error(_api_error(401, "{}"), "learning")

    assert error.startswith("Learning failed: the model provider rejected its credentials")

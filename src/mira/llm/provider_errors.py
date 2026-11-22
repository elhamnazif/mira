"""Name the cause of a model provider's refusal in words a PR author can act on.

The patterns are adapted from Pullfrog's ``utils/providerErrors.ts`` and
``utils/apiKeys.ts`` (MIT, Copyright (c) 2026 Pullfrog, Inc.).

An exception's text can carry anything the provider sent back, key included,
so it is only matched, never shown. Callers get one of the fixed
descriptions below.
"""

from __future__ import annotations

import re
from enum import StrEnum

import httpx

from mira.exceptions import LLMError


class ProviderErrorKind(StrEnum):
    AUTH = "the model provider rejected its credentials (invalid, expired or revoked key)"
    QUOTA = "the model provider's rate limit, quota or spending cap was reached"
    MODEL = "the model provider does not serve this model, or the endpoint URL is wrong"
    CONTEXT = "the prompt is longer than the model's context window"
    NETWORK = "the model provider or its gateway could not be reached or failed"


_AUTH, _QUOTA, _MODEL, _CONTEXT, _NETWORK = ProviderErrorKind

# Error codes Mira raises once it has already told the cases apart.
_CODES = {
    "claude_auth_failed": _AUTH,
    "claude_rate_limited": _QUOTA,
    "bedrock_access_denied": _AUTH,
    "bedrock_model_not_found": _MODEL,
}

# First match wins. Context comes before quota because "too many tokens" is
# not a rate limit, and billing and quota come before auth because providers
# answer 401 or 403 for an empty wallet.
_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), kind)
    for pattern, kind in [
        (r"context[_ ]length|context window|maximum context|request_too_large", _CONTEXT),
        (r"(?:prompt|input) is too long|too many (?:input )?tokens", _CONTEXT),
        (
            r"\bCreditsError\b|\bFreeUsageLimitError\b|insufficient (?:balance|credits|funds)",
            _QUOTA,
        ),
        (r"no credits remaining|out of credits|credit balance is too low", _QUOTA),
        (r"requires more credits|spending (?:cap|limit)|used all available credits", _QUOTA),
        (
            r"reached your specified API usage limits|usage limit|hit your \w+(?:-hour)? limit",
            _QUOTA,
        ),
        (r"\bRESOURCE_EXHAUSTED\b|quota|\brate[_ ]limit|too many requests", _QUOTA),
        (r"invalid (?:x-)?api[_ -]?key|incorrect api key|invalid bearer token", _AUTH),
        (r"authentication_error|invalid authentication|no auth credentials", _AUTH),
        (r"\bUser not found\b|not logged in|failed to authenticate|token refresh failed", _AUTH),
        (
            r"(?:authentication|oauth access) token (?:has |is |was )?(?:expired|been revoked)",
            _AUTH,
        ),
        (r"(?:expired|invalidated|revoked) (?:oauth|authentication) token", _AUTH),
        (r"model_not_found|unknown model|invalid model|no such model|no endpoints found", _MODEL),
        (r"model\b[^\n]{0,80}\b(?:does not exist|not found|is not (?:a valid|supported))", _MODEL),
        (r"\boverloaded\b|bad gateway|service unavailable|gateway time-?out", _NETWORK),
        (
            r"model is at capacity|connection (?:refused|reset|error)|ECONN(?:REFUSED|RESET)",
            _NETWORK,
        ),
        (r"ETIMEDOUT|ENOTFOUND|getaddrinfo|name or service not known", _NETWORK),
    ]
]

# A status counts only next to a status key, so a commit SHA or a version
# string that holds "429" does not.
_STATUS = re.compile(
    r"\b(?:status[_ ]?code|http[_ ]?status|api_error_status|status|LLM API error|API Error"
    r"|Error code)[\"']?\s*[:=]?\s*[\"']?(\d{3})\b",
    re.IGNORECASE,
)

_STATUS_KINDS = {
    401: _AUTH,
    403: _AUTH,
    404: _MODEL,
    413: _CONTEXT,
    429: _QUOTA,
    500: _NETWORK,
    502: _NETWORK,
    503: _NETWORK,
    504: _NETWORK,
    529: _NETWORK,
}


def _chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_provider_error(exc: BaseException) -> ProviderErrorKind | None:
    """Why the model provider refused, or None for an error that is not an
    LLM call's. Looks through the exception's causes. A GitHub or Mira
    server error is never one: their 401s are not about the model's key."""
    chain = _chain(exc)
    llm_errors = [each for each in chain if isinstance(each, LLMError)]
    if not llm_errors:
        return None
    for each in llm_errors:
        if each.code in _CODES:
            return _CODES[each.code]
    text = "\n".join(str(each) for each in chain)
    for pattern, kind in _PATTERNS:
        if pattern.search(text):
            return kind
    statuses = [int(m.group(1)) for m in _STATUS.finditer(text)]
    statuses += [e.response.status_code for e in chain if isinstance(e, httpx.HTTPStatusError)]
    for status in statuses:
        if status in _STATUS_KINDS:
            return _STATUS_KINDS[status]
    if any(isinstance(each, httpx.TransportError | ConnectionError) for each in chain):
        return _NETWORK
    return None

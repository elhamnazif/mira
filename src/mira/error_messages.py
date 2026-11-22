"""Centralized LLM error message catalog.

Each error has two templates:
- ``full``: the detailed message used at raise time (includes model names, underlying errors)
- ``safe``: the sanitized message shown to users (no model names, no internal errors)

New LLM error types MUST add both templates here. The ``safe`` template is what
appears in PR comments; the full traceback stays in server logs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorMessage:
    """A pair of full and safe error message templates."""

    full: str
    safe: str


# Maps error code → (full_template, safe_template)
# Templates use {placeholder} syntax for .format() — not f-strings — so they
# can be reused at both raise and display sites without evaluation.
LLM_ERROR_MESSAGES: dict[str, ErrorMessage] = {
    "no_api_key": ErrorMessage(
        full=(
            "No API key found. Set {api_key_env} (or OPENROUTER_API_KEY / "
            'OPENAI_API_KEY) in the environment, or set llm.api_key_env: "" in '
            "your config for a local endpoint that needs no auth."
        ),
        safe="No API key found: {api_key_env} is not set",
    ),
    "api_error": ErrorMessage(
        full="LLM API error {status}: {body}",
        safe="LLM API error {status}",
    ),
    "model_unavailable": ErrorMessage(
        full="LLM API error {status}, model missing or unavailable: {body}",
        safe="LLM API error {status}: model missing or unavailable",
    ),
    "no_tool_call": ErrorMessage(
        full="Model returned neither tool call nor content ({detail})",
        safe="Model returned neither tool call nor content",
    ),
    "empty_response": ErrorMessage(
        full="Model returned no usable completion text ({detail})",
        safe="Model returned no usable completion text",
    ),
    "completion_blocked": ErrorMessage(
        full="Model output was refused or filtered ({protocol})",
        safe="Model output was refused or filtered",
    ),
    "structured_output": ErrorMessage(
        full=(
            "Model {model} returned prose instead of a `{tool}` call (response excerpt: {excerpt})"
        ),
        safe="Model returned prose instead of the required structured tool call",
    ),
    "no_tools": ErrorMessage(
        full="tools list must not be empty",
        safe="tools list must not be empty",
    ),
    "completion_failed": ErrorMessage(
        full="LLM completion failed with {model}: {error}",
        safe="LLM completion failed",
    ),
    "agentic_failed": ErrorMessage(
        full="LLM agentic call failed with {model}: {error}",
        safe="LLM agentic call failed",
    ),
    "tool_call_failed": ErrorMessage(
        full="LLM tool-call failed with {model}: {error}",
        safe="LLM tool-call failed",
    ),
    "both_models_failed": ErrorMessage(
        full="Both primary ({primary_model}) and fallback ({fallback_model}) models failed: {error}",
        safe="Both primary and fallback models failed",
    ),
    # Bedrock-specific errors
    "bedrock_no_boto3": ErrorMessage(
        full="boto3 is required for the Bedrock provider. Install with: pip install mira-reviewer[bedrock]",
        safe="boto3 is required for the Bedrock provider",
    ),
    "bedrock_access_denied": ErrorMessage(
        full="Bedrock access denied for model {model}. Ensure your IAM role/user has bedrock:InvokeModel permission and the model is enabled in your account.",
        safe="Bedrock access denied",
    ),
    "bedrock_model_not_found": ErrorMessage(
        full="Bedrock model not found: {model}. Check the model ID and ensure it's available in your region.",
        safe="Bedrock model not found",
    ),
    "bedrock_validation_error": ErrorMessage(
        full="Bedrock validation error for {model}: {error}",
        safe="Bedrock validation error",
    ),
    "bedrock_api_error": ErrorMessage(
        full="Bedrock API error for {model}: {error}",
        safe="Bedrock API error",
    ),
    "bedrock_call_failed": ErrorMessage(
        full="Bedrock call failed with {model}: {error}",
        safe="Bedrock call failed",
    ),
    "bedrock_no_tool_call": ErrorMessage(
        full="Bedrock model returned neither tool call nor content",
        safe="Bedrock model returned neither tool call nor content",
    ),
    # Codex CLI provider errors
    "codex_auth_file_missing": ErrorMessage(
        full="Codex auth file not found: {path}",
        safe="Codex auth file not found",
    ),
    "codex_command_not_found": ErrorMessage(
        full="Codex CLI command not found: {command!r}. Install @openai/codex or set llm.codex_command.",
        safe="Codex CLI command not found — install @openai/codex or set llm.codex_command",
    ),
    "codex_timeout": ErrorMessage(
        full="Codex CLI timed out after {seconds}s",
        safe="Codex CLI timed out",
    ),
    "codex_exit_failed": ErrorMessage(
        full="Codex CLI failed with exit {exit_code}: {detail}",
        safe="Codex CLI failed",
    ),
    "codex_empty_response": ErrorMessage(
        full="Codex CLI returned an empty response",
        safe="Codex CLI returned an empty response",
    ),
    "codex_non_object_json": ErrorMessage(
        full="Codex CLI response must be a JSON object, got {type}",
        safe="Codex CLI response must be a JSON object",
    ),
    "codex_malformed_json": ErrorMessage(
        full="Codex CLI returned malformed JSON: {error}: {excerpt}",
        safe="Codex CLI returned malformed JSON",
    ),
    "codex_no_json_object": ErrorMessage(
        full="Codex CLI response did not contain a JSON object: {excerpt}",
        safe="Codex CLI response did not contain a JSON object",
    ),
    # Claude Code CLI provider errors
    "claude_command_not_found": ErrorMessage(
        full="Claude Code CLI command not found: {command!r}. Install @anthropic-ai/claude-code or set llm.claude_command.",
        safe="Claude Code CLI command not found — install @anthropic-ai/claude-code or set llm.claude_command",
    ),
    "claude_auth_missing": ErrorMessage(
        full="Claude Code provider needs one of these environment variables: {keys}",
        safe="Claude Code credentials are not set: set one of {keys}",
    ),
    "claude_auth_failed": ErrorMessage(
        full="Claude Code rejected its credentials: {detail}",
        safe="Claude Code rejected its credentials",
    ),
    "claude_rate_limited": ErrorMessage(
        full="Claude Code hit a usage or rate limit: {detail}",
        safe="Claude Code hit a usage or rate limit",
    ),
    "claude_timeout": ErrorMessage(
        full="Claude Code CLI timed out after {seconds}s",
        safe="Claude Code CLI timed out",
    ),
    "claude_no_first_event": ErrorMessage(
        full="Claude Code CLI produced no output in its first {seconds}s",
        safe="Claude Code CLI produced no output",
    ),
    "claude_stalled": ErrorMessage(
        full="Claude Code CLI went silent for {seconds}s",
        safe="Claude Code CLI stalled",
    ),
    "claude_exit_failed": ErrorMessage(
        full="Claude Code CLI failed with exit {exit_code}: {detail}",
        safe="Claude Code CLI failed",
    ),
    "claude_reported_error": ErrorMessage(
        full="Claude Code CLI reported an error (exit {exit_code}): {detail}",
        safe="Claude Code CLI reported an error",
    ),
    "claude_empty_response": ErrorMessage(
        full="Claude Code CLI returned an empty response",
        safe="Claude Code CLI returned an empty response",
    ),
    "claude_no_json_object": ErrorMessage(
        full="Claude Code CLI response did not contain a JSON object: {excerpt}",
        safe="Claude Code CLI response did not contain a JSON object",
    ),
    "claude_workspace_missing": ErrorMessage(
        full="Claude Code workspace review was given a missing directory: {path}",
        safe="Claude Code workspace review was given a missing directory",
    ),
    # OpenCode CLI provider errors
    "opencode_command_not_found": ErrorMessage(
        full="OpenCode CLI command not found: {command!r}. Install opencode-ai or set llm.opencode_command.",
        safe="OpenCode CLI command not found — install opencode-ai or set llm.opencode_command",
    ),
    "opencode_model_invalid": ErrorMessage(
        full="OpenCode needs a model as <provider>/<model>, got {model!r}",
        safe="OpenCode needs a model as <provider>/<model>",
    ),
    "opencode_auth_missing": ErrorMessage(
        full="OpenCode provider {provider!r} needs {keys}",
        safe="OpenCode credentials are not set: set {keys}",
    ),
    "opencode_auth_content_invalid": ErrorMessage(
        full="OPENCODE_AUTH_CONTENT is not a JSON object of OpenCode credentials",
        safe="OPENCODE_AUTH_CONTENT is not a JSON object of OpenCode credentials",
    ),
    "opencode_endpoint_invalid": ErrorMessage(
        full="The openai-compatible endpoint needs {problem}",
        safe="The openai-compatible endpoint needs {problem}",
    ),
    "opencode_auth_failed": ErrorMessage(
        full="OpenCode's provider rejected its credentials: {detail}",
        safe="OpenCode's provider rejected its credentials",
    ),
    "opencode_rate_limited": ErrorMessage(
        full="OpenCode's provider hit a usage or rate limit: {detail}",
        safe="OpenCode's provider hit a usage or rate limit",
    ),
    "opencode_timeout": ErrorMessage(
        full="OpenCode CLI timed out after {seconds}s",
        safe="OpenCode CLI timed out",
    ),
    "opencode_no_first_event": ErrorMessage(
        full="OpenCode CLI produced no output in its first {seconds}s",
        safe="OpenCode CLI produced no output",
    ),
    "opencode_stalled": ErrorMessage(
        full="OpenCode CLI went silent for {seconds}s",
        safe="OpenCode CLI stalled",
    ),
    "opencode_exit_failed": ErrorMessage(
        full="OpenCode CLI failed with exit {exit_code}: {detail}",
        safe="OpenCode CLI failed",
    ),
    "opencode_reported_error": ErrorMessage(
        full="OpenCode CLI reported an error (exit {exit_code}): {detail}",
        safe="OpenCode CLI reported an error",
    ),
    "opencode_empty_response": ErrorMessage(
        full="OpenCode CLI returned an empty response",
        safe="OpenCode CLI returned an empty response",
    ),
    "opencode_no_json_object": ErrorMessage(
        full="OpenCode CLI response did not contain a JSON object: {excerpt}",
        safe="OpenCode CLI response did not contain a JSON object",
    ),
    "opencode_workspace_missing": ErrorMessage(
        full="OpenCode workspace review was given a missing directory: {path}",
        safe="OpenCode workspace review was given a missing directory",
    ),
}


def get_error_message(code: str, **kwargs: object) -> tuple[str, str]:
    """Return (full_message, safe_message) for the given error code.

    Raises:
        KeyError: If the error code is not in the catalog.
    """
    entry = LLM_ERROR_MESSAGES[code]
    return entry.full.format(**kwargs), entry.safe.format(**kwargs)

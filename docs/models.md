# Model slugs

A model slug names a model family as `provider/alias`, such as
`anthropic/claude-sonnet` or `zai/glm`. Mira resolves it to a concrete model
id and picks the harness that runs it from the credentials it finds. Set one
anywhere a model goes:

```yaml
llm:
  model: anthropic/claude-sonnet
```

On GitHub Actions, set the `MIRA_RUNNER_MODEL` variable to the slug, such as
`zai/glm`. A plain model id such as `anthropic/claude-sonnet-4-6` or `glm-5.3`
is not a slug and works as before.

## Slugs

| Slug | Model | Effort levels | Cheap sibling | Fallback |
|---|---|---|---|---|
| `anthropic/claude-fable` | `anthropic/claude-fable-5` | low, medium, high, xhigh, max | `anthropic/claude-sonnet` | `anthropic/claude-opus` |
| `anthropic/claude-opus` | `anthropic/claude-opus-4-8` | low, medium, high, xhigh, max | `anthropic/claude-sonnet` | `anthropic/claude-sonnet` |
| `anthropic/claude-sonnet` | `anthropic/claude-sonnet-4-6` | low, medium, high, xhigh, max | `anthropic/claude-haiku` | none |
| `anthropic/claude-haiku` | `anthropic/claude-haiku-4-5` | low, medium, high, xhigh, max | none | none |
| `openai/gpt` | `openai/gpt-5.2` | none, low, medium, high, xhigh | `openai/gpt-mini` | `openai/gpt-mini` |
| `openai/gpt-mini` | `openai/gpt-5-mini` | minimal, low, medium, high | `openai/gpt-nano` | none |
| `openai/gpt-nano` | `openai/gpt-5-nano` | minimal, low, medium, high | none | none |
| `google/gemini-pro` | `google/gemini-3.1-pro-preview` | low, medium, high | `google/gemini-flash` | `google/gemini-flash` |
| `google/gemini-flash` | `google/gemini-3-flash-preview` | low, medium, high | none | none |
| `zai/glm` | `zai/glm-5.3` | low, high, max | `zai/glm-flash` | `zai/glm-flash` |
| `zai/glm-flash` | `zai/glm-5.3-flash` | none listed | none | none |
| `deepseek/deepseek-pro` | `deepseek/deepseek-v4-pro` | low, high, max | `deepseek/deepseek-flash` | `deepseek/deepseek-flash` |
| `deepseek/deepseek-flash` | `deepseek/deepseek-v4-flash` | low, high, max | none | none |

The slugs live in `src/mira/llm/slugs.json`. Endpoint details such as base
URLs, key variables and model prefixes stay in `providers.json`.

## Which harness runs a slug

Mira checks these rules in order and takes the first that applies:

1. **An explicit provider.** `llm.provider` in the deployment config, or
   `MIRA_RUNNER_LLM_PROVIDER` on Actions. `claude-code` runs Claude Code,
   `openai` uses the slug's own API endpoint, and a `providers.json` name
   such as `openrouter` uses that endpoint.
2. **The vendor's own harness.** An Anthropic slug runs on Claude Code when
   `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` is set.
3. **A Claude Code gateway.** With `ANTHROPIC_BASE_URL` and
   `ANTHROPIC_AUTH_TOKEN` both set, every slug runs on Claude Code. This is
   how a Z.ai GLM Coding Plan key runs `zai/glm`.
4. **The API endpoint.** Otherwise the slug's provider endpoint: OpenRouter
   for Anthropic slugs, and the vendor's own API for the others, with its key
   in that provider's variable, such as `ZAI_API_KEY` or `DEEPSEEK_API_KEY`.
   An explicit `llm.base_url` keeps its endpoint and gets the resolved model.

`opencode` is never chosen on its own. Set it as the provider to run a slug,
or any `<provider>/<model>` id, on [OpenCode](https://opencode.ai).

## OpenCode

`llm.provider: opencode` (`MIRA_RUNNER_LLM_PROVIDER=opencode` on Actions)
runs the model through the OpenCode CLI, so any provider OpenCode supports
works, including subscription plans. The model is `<provider>/<model>`: a
slug such as `zai/glm`, or a plain id such as `openrouter/qwen/qwen3-coder`.

The credential is one of these, checked in order:

- `OPENCODE_AUTH_CONTENT`: the contents of an OpenCode `auth.json`, for a
  subscription such as a ChatGPT plan. Only the model's provider's entry
  reaches the CLI.
- The provider's API key: the key variable of the `providers.json` profile
  with that name (`ZAI_API_KEY` for `zai`, `OPENAI_API_KEY` for `openai`),
  else `<PROVIDER>_API_KEY` (`ANTHROPIC_API_KEY`).
- For `openai-compatible/<model>`: `OPENAI_COMPATIBLE_BASE_URL`,
  `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_CONTEXT` and
  `OPENAI_COMPATIBLE_MAX_OUTPUT`. The two token limits are required.

`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GEMINI_BASE_URL`, `XAI_BASE_URL`,
`DEEPSEEK_BASE_URL`, `MOONSHOT_BASE_URL` and `KIMI_BASE_URL` point that
provider at a gateway instead. A subscription is never sent to a gateway.

The effort level becomes OpenCode's `--variant`. OpenCode has no fallback
model, so `fallback_model` does not apply.

OpenCode loads plugins from the `.opencode` directory of its working
directory, so Mira never runs it inside the pull request's checkout. It runs
in an empty directory, with only read, grep, glob and list allowed, and only
inside the checkout. `opencode_command` and `opencode_timeout_seconds` are
deployment settings.

## Effort, cheap siblings and fallback

**Effort.** Mira's review effort (`low` to `max`) is a point on a scale, and
each model's own levels divide that scale. A level the model has runs as
itself. Any other lands on the nearest level, ties going up, so on `zai/glm`
(`low`, `high`, `max`) `medium` runs as `high`, as the plain `zai` profile
does, and `xhigh` runs as `max`. Levels that turn
reasoning off (`none`, `minimal`) are never picked, and neither are levels
the Claude Code CLI does not take.

**Cheap sibling.** When `indexing_model` is unset, indexing runs on the slug's
cheap sibling. The security pass never does: it stays on the review model
unless `security_model` names another.

**Fallback.** When the endpoint answers that the model does not exist, or the
model stays unavailable after retries, Mira tries the slug's fallback once for
that call and logs it. On Claude Code the CLI's `--fallback-model` does this
for a busy or unavailable model. An explicit `llm.fallback_model` replaces the
slug's fallback.

Adapted from [Pullfrog](https://github.com/pullfrog/pullfrog)'s model
registry, MIT License, Copyright (c) 2026 Pullfrog, Inc.

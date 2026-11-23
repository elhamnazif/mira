# Reviewing on GitHub Actions

With `review.runner: actions`, the Mira server runs no model. For each PR
push it records a review job, shows a queued **Mira** check, and dispatches
the repo's `.github/workflows/mira.yml`. That workflow calls Mira's reusable
`review.yml`, which installs Mira at the workflow's own commit, claims the
job from the server with the run's OIDC token, reviews the PR, and sends the
result back. Comments come from the Mira GitHub App, as in local mode.

The job's `runs-on` comes from the server: `review.runs_on` (default
`["ubuntu-latest"]`), or a repo's runner override such as
`actions:self-hosted,sora`.

## Set up a repo

1. **GitHub App permissions.** The Mira App needs **Actions: write** to
   dispatch the workflow, on top of its review permissions. Accept the new
   permission on each installation.
2. **Add the workflow.** Copy [`mira.yml`](mira.yml) to
   `.github/workflows/mira.yml` on the repo's default branch. GitHub only
   dispatches workflows that exist on the default branch.
3. **Choose the model.** Nothing to do when the server hands runners its
   keys and model (see [Keys from the server](#keys-from-the-server)).
   Otherwise, or to use a different model on this repo, set the Actions
   secret and variables for one of the setups below.

## Choose the model

A repo can pick its own model with Actions secrets and variables, which
override the server's. A key set this way stays in the repo's Actions run and
never reaches the Mira server. Leave every
setting a setup does not list unset. A missing secret fails the job at once,
and the PR's **Mira** check names the secret to set.

`MIRA_RUNNER_MODEL` sets the model for every review pass. Without it, the
runner uses the server's configured models, which may not exist on the
repo's endpoint.

**Claude subscription.** Run `claude setup-token`.

| Kind | Name | Value |
|---|---|---|
| Secret | `CLAUDE_CODE_OAUTH_TOKEN` | the token |
| Variable | `MIRA_RUNNER_MODEL` | optional: `sonnet` or `opus` |

**Z.ai API key** (pay as you go, through Z.ai's OpenAI-compatible API).

| Kind | Name | Value |
|---|---|---|
| Secret | `ZAI_API_KEY` | the key |
| Variable | `MIRA_RUNNER_LLM_PROVIDER` | `zai` |
| Variable | `MIRA_RUNNER_MODEL` | `glm-5.3` |

**Z.ai GLM Coding Plan key, on Mira's own engine.** The same, with
`MIRA_RUNNER_LLM_PROVIDER` set to `zai-coding`, which uses Z.ai's Coding Plan
endpoint (`https://api.z.ai/api/coding/paas/v4`).

**Z.ai GLM Coding Plan through Claude Code.** Z.ai limits Coding Plan keys
to supported coding tools, so Mira runs it through the Claude Code CLI, set
up as in [Z.ai's Claude Code guide](https://docs.z.ai/devpack/tool/claude).

| Kind | Name | Value |
|---|---|---|
| Secret | `ANTHROPIC_AUTH_TOKEN` | the Coding Plan key |
| Variable | `ANTHROPIC_BASE_URL` | `https://api.z.ai/api/anthropic` |
| Variable | `ANTHROPIC_DEFAULT_OPUS_MODEL` | `glm-5.3[1m]` |
| Variable | `ANTHROPIC_DEFAULT_SONNET_MODEL` | `glm-5.3[1m]` |
| Variable | `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `glm-5.3-flash[1m]` |
| Variable | `API_TIMEOUT_MS` | `3000000` |
| Variable | `MIRA_RUNNER_MODEL` | `sonnet` |

With `ANTHROPIC_BASE_URL` set, Mira never sends `CLAUDE_CODE_OAUTH_TOKEN`
to the gateway.

A gateway that needs extra request headers, such as its own auth header,
takes them from the `ANTHROPIC_CUSTOM_HEADERS` secret, one `Name: value`
per line. It is a secret because headers often carry credentials, and Mira
redacts it from review output like the key. Mira passes it to Claude Code
only with `ANTHROPIC_BASE_URL` set.

**Any OpenAI-compatible endpoint**, such as OpenRouter, DeepSeek, Together,
vLLM or Ollama.

| Kind | Name | Value |
|---|---|---|
| Secret | `MIRA_LLM_API_KEY` | the endpoint's key |
| Variable | `MIRA_RUNNER_LLM_BASE_URL` | the endpoint's base URL, e.g. `https://openrouter.ai/api/v1` |
| Variable | `MIRA_RUNNER_MODEL` | a model id the endpoint serves |
| Variable | `MIRA_RUNNER_LLM_API_STYLE` | optional: `chat` (default) or `responses` |

With a base URL set, `MIRA_RUNNER_LLM_PROVIDER` is ignored. An endpoint on
the runner's own network, such as `http://localhost:11434/v1` for Ollama on a
self-hosted runner, may go without `MIRA_LLM_API_KEY`. Any other endpoint
needs it.

**OpenCode**, with any provider OpenCode knows.

| Kind | Name | Value |
|---|---|---|
| Variable | `MIRA_RUNNER_LLM_PROVIDER` | `opencode` |
| Variable | `MIRA_RUNNER_MODEL` | `<provider>/<model>`, e.g. `zai/glm-5.3` or `deepseek/deepseek-chat` |
| Secret | `MIRA_LLM_API_KEY` | the provider's key (`ZAI_API_KEY` also works for `zai`) |
| Secret | `OPENCODE_AUTH_CONTENT` | instead of a key: the contents of an OpenCode `auth.json`, for a subscription |

For an OpenAI-compatible endpoint OpenCode doesn't know, use the model
`openai-compatible/<model>` with the key in `MIRA_LLM_API_KEY` and the
variables `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_CONTEXT` and
`OPENAI_COMPATIBLE_MAX_OUTPUT` (the model's context and output limits in
tokens). OpenCode runs outside the checkout and reads it by absolute path,
so a reviewed repo's `.opencode` plugins, tools and MCP servers never load.

## Configure the server

In the server's deployment config:

```yaml
review:
  runner: actions                 # or set it per repo from the dashboard
  runs_on: ["ubuntu-latest"]
  runner_server_url: https://mira.example.com   # where runners reach /runner/*
  runner_workflow_refs:
    - elhamnazif/mira/.github/workflows/review.yml@refs/heads/main
```

`runner_server_url` must be reachable from the runner, and only the
`/runner/*` routes need to be public. Without it, an Actions review fails its
check with a message that names the key.

`runner_workflow_refs` lists the exact reusable workflows the server
trusts, matched against the OIDC token's `job_workflow_ref` claim. An
empty list rejects every runner.

### Keys from the server

GitHub Free gives private repos no access to org-level secrets, so a key set
once for every repo has to come from the server instead. List the env var
names, set their values in the server's environment, and name a default
model:

```yaml
review:
  runner_credentials: [CLAUDE_CODE_OAUTH_TOKEN, ZAI_API_KEY]
  runner_llm_provider: zai        # used when the repo sets no MIRA_RUNNER_* variables
  runner_model: glm-5.3
```

The server sends these values with the job when the runner claims it. A
claim needs the job's OIDC token from a workflow in `runner_workflow_refs`,
so no other workflow can get them. The runner masks each value in the run's
log. A repo secret of the same name takes precedence. `runner_credentials`
cannot name the server's own `MIRA_*` secrets, except `MIRA_LLM_API_KEY`.

## Self-hosted runners

`review.yml` needs no sudo and writes only to `$RUNNER_TEMP` and the
runner's tool cache. It needs network access to GitHub, PyPI, the npm
registry and the Mira server. uv, Python and Node come from its own setup
steps, so the machine needs none of them.

## Not yet on Actions

On a repo that reviews on Actions, `@mira review`, `@mira review-rest`,
questions and free-form thread replies are skipped and logged by the
server. `help`, `pause`, `resume` and `reject` still work.

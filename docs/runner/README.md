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
3. **Choose the model.** Set the Actions secret and variables for one of the
   setups below, on the repo or its org.

## Choose the model

Each repo picks its model with Actions secrets and variables. The key stays
in the repo's Actions run and never reaches the Mira server. Leave every
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

## Self-hosted runners

`review.yml` needs no sudo and writes only to `$RUNNER_TEMP` and the
runner's tool cache. It needs network access to GitHub, PyPI, the npm
registry and the Mira server. uv, Python and Node come from its own setup
steps, so the machine needs none of them.

## Not yet on Actions

On a repo that reviews on Actions, `@mira review`, `@mira review-rest`,
questions and free-form thread replies are skipped and logged by the
server. `help`, `pause`, `resume` and `reject` still work.

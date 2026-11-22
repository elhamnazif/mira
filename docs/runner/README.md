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
3. **Add the Claude secret.** Run `claude setup-token` and save the token as
   the Actions secret `CLAUDE_CODE_OAUTH_TOKEN`, on the repo or its org.
4. Optional: set the Actions variable `MIRA_RUNNER_MODEL` to use one model
   for every review pass instead of the server's configured models.

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

import { Loader2 } from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { api, type RepoRunnerView } from "@/lib/api"

const INHERIT = "inherit"
const GITHUB_HOSTED = "actions:ubuntu-latest"

type RunnerMode = "inherit" | "local" | "github" | "labels"

function runnerMode(runner: string): RunnerMode {
  if (!runner) return "inherit"
  if (runner === "local") return "local"
  return runner === GITHUB_HOSTED ? "github" : "labels"
}

function describeRunner(runner: string): string {
  if (runner === "local") return "on the Mira server"
  if (runner === GITHUB_HOSTED) return "on GitHub-hosted Actions runners"
  return `on Actions runners labelled ${runner.replace(/^actions:/, "")}`
}

function describeLlm(llm: string): string {
  if (!llm) return "the runner's own default (Claude Code)"
  const [provider, model] = llm.split(":")
  return model ? `${model} via ${provider}` : `${provider}'s default model`
}

export function RunnerSettingsCard({
  owner,
  repo,
  canEdit,
}: {
  owner: string
  repo: string
  canEdit: boolean
}) {
  const [view, setView] = useState<RepoRunnerView | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [mode, setMode] = useState<RunnerMode>("inherit")
  const [labels, setLabels] = useState("")
  const [provider, setProvider] = useState(INHERIT)
  const [model, setModel] = useState("")
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState("")

  const show = (v: RepoRunnerView) => {
    setView(v)
    setMode(runnerMode(v.runner))
    setLabels(
      runnerMode(v.runner) === "labels" ? v.runner.replace(/^actions:/, "") : ""
    )
    const [p, m] = v.llm ? v.llm.split(":") : [INHERIT, ""]
    setProvider(p)
    setModel(m ?? "")
  }

  useEffect(() => {
    // The page keys this card by repo, so each repo starts from a fresh state.
    let cancelled = false
    api
      .getRunnerSettings(owner, repo)
      .then((v) => !cancelled && show(v))
      .catch(() => !cancelled && setLoadError(true))
    return () => {
      cancelled = true
    }
  }, [owner, repo])

  const runnerValue = (): string => {
    if (mode === "inherit") return ""
    if (mode === "local") return "local"
    if (mode === "github") return GITHUB_HOSTED
    const list = labels
      .split(",")
      .map((l) => l.trim())
      .filter(Boolean)
    return list.length ? `actions:${list.join(",")}` : ""
  }

  const save = async () => {
    setError("")
    if (mode === "labels" && !runnerValue()) {
      setError("Name at least one runner label, e.g. self-hosted, sora")
      return
    }
    setSaving(true)
    try {
      const llm = provider === INHERIT ? "" : `${provider}:${model.trim()}`
      show(
        await api.saveRunnerSettings(owner, repo, {
          runner: runnerValue(),
          llm,
        })
      )
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Runner</CardTitle>
        <CardDescription>
          Where this repo's reviews run, and the model they use there. Takes
          effect on the next review.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {!view ? (
          <p
            className={
              loadError
                ? "text-sm text-destructive"
                : "text-sm text-muted-foreground"
            }
          >
            {loadError
              ? "Could not load the runner settings for this repo. Reload the page to retry."
              : "Loading…"}
          </p>
        ) : (
          <>
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="runner-mode">
                Runs on
              </label>
              <Select
                value={mode}
                onValueChange={(v) => setMode(v as RunnerMode)}
                disabled={!canEdit}
              >
                <SelectTrigger id="runner-mode" className="max-w-md">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="inherit">
                    Inherit ({describeRunner(view.default_runner)})
                  </SelectItem>
                  <SelectItem value="github">
                    GitHub Actions, GitHub-hosted
                  </SelectItem>
                  <SelectItem value="labels">
                    GitHub Actions, self-hosted runner
                  </SelectItem>
                  <SelectItem value="local">The Mira server</SelectItem>
                </SelectContent>
              </Select>
              {mode === "labels" && (
                <Input
                  className="max-w-md font-mono text-sm"
                  placeholder="self-hosted, sora"
                  value={labels}
                  onChange={(e) => setLabels(e.target.value)}
                  disabled={!canEdit}
                  aria-label="Runner labels"
                />
              )}
              <p className="text-xs text-muted-foreground">
                {mode === "labels"
                  ? "The runs-on labels, comma-separated. A job waits up to 24 hours for a matching runner to come online."
                  : mode === "local"
                    ? "Reviews run inside the Mira server process, with the server's own model settings."
                    : "Reviews run in the repo's .github/workflows/mira.yml."}
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="runner-provider">
                Model
              </label>
              <div className="flex max-w-md gap-2">
                <Select
                  value={provider}
                  onValueChange={setProvider}
                  disabled={!canEdit || mode === "local"}
                >
                  <SelectTrigger id="runner-provider" className="w-48 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={INHERIT}>Inherit</SelectItem>
                    {view.providers.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  className="font-mono text-sm"
                  placeholder={
                    provider === INHERIT ? "" : "model, e.g. glm-5.3"
                  }
                  value={provider === INHERIT ? "" : model}
                  onChange={(e) => setModel(e.target.value)}
                  disabled={
                    !canEdit || provider === INHERIT || mode === "local"
                  }
                  aria-label="Model id"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {provider === INHERIT
                  ? `Inherits ${describeLlm(view.default_llm)}.`
                  : "A provider is an agent CLI or a providers.json profile. Leave the model empty for the provider's default."}{" "}
                A repo's MIRA_RUNNER_* Actions variables, if set, take
                precedence.
              </p>
            </div>

            {canEdit && (
              <div className="flex items-center gap-3">
                <Button size="sm" onClick={save} disabled={saving}>
                  {saving && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
                  Save runner
                </Button>
                {saved && (
                  <span className="text-xs text-muted-foreground">Saved</span>
                )}
                {error && <p className="text-xs text-destructive">{error}</p>}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

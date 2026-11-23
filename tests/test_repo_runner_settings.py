"""Per-repo runner and runner model: DB columns, dashboard API, claim."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import mira.config as mira_config
from mira.dashboard.api import RepoRunnerSettings
from mira.dashboard.db import AppDatabase
from mira.dashboard.routers.repos import get_runner_settings, set_runner_settings
from mira.runner import RunnerLLMChoice, parse_runner_llm


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    app_db = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr("mira.dashboard.api._app_db", app_db)
    review = {
        "runner": "actions",
        "runs_on": ["ubuntu-latest"],
        "runner_llm_provider": "zai-coding",
        "runner_model": "glm-5.3",
    }
    monkeypatch.setattr(mira_config, "_global_defaults", {"review": review})
    app_db.register_repo("o", "r")
    return app_db


def _admin() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(user=SimpleNamespace(is_admin=True)))


def test_a_new_repo_inherits_and_shows_the_defaults(db: AppDatabase) -> None:
    view = get_runner_settings("o", "r")
    assert (view.runner, view.llm) == ("", "")
    assert view.default_runner == "actions:ubuntu-latest"
    assert view.default_llm == "zai-coding:glm-5.3"
    assert {"claude-code", "opencode", "zai", "zai-coding"} <= set(view.providers)


def test_saving_stores_both_and_reads_back(db: AppDatabase) -> None:
    body = RepoRunnerSettings(runner="actions:self-hosted,sora", llm="claude-code:sonnet")
    set_runner_settings("o", "r", body, _admin())  # type: ignore[arg-type]
    record = db.get_repo("o", "r")
    assert record is not None
    assert (record.runner, record.runner_llm) == ("actions:self-hosted,sora", "claude-code:sonnet")
    view = get_runner_settings("o", "r")
    assert (view.runner, view.llm) == ("actions:self-hosted,sora", "claude-code:sonnet")
    # Back to inherit.
    set_runner_settings("o", "r", RepoRunnerSettings(), _admin())  # type: ignore[arg-type]
    record = db.get_repo("o", "r")
    assert record is not None and (record.runner, record.runner_llm) == ("", "")


@pytest.mark.parametrize(
    "body",
    [
        RepoRunnerSettings(runner="actions:"),
        RepoRunnerSettings(runner="somewhere"),
        RepoRunnerSettings(llm="glm-5.3"),
        RepoRunnerSettings(llm="no-such-provider:m"),
    ],
)
def test_bad_settings_are_refused_and_nothing_is_stored(
    db: AppDatabase, body: RepoRunnerSettings
) -> None:
    with pytest.raises(HTTPException) as exc:
        set_runner_settings("o", "r", body, _admin())  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    record = db.get_repo("o", "r")
    assert record is not None and (record.runner, record.runner_llm) == ("", "")


def test_only_an_admin_may_save(db: AppDatabase) -> None:
    viewer = SimpleNamespace(state=SimpleNamespace(user=SimpleNamespace(is_admin=False)))
    with pytest.raises(HTTPException) as exc:
        set_runner_settings("o", "r", RepoRunnerSettings(runner="local"), viewer)  # type: ignore[arg-type]
    assert exc.value.status_code == 403


def test_an_unregistered_repo_is_not_found(db: AppDatabase) -> None:
    with pytest.raises(HTTPException) as exc:
        set_runner_settings("ghost", "r", RepoRunnerSettings(runner="local"), _admin())  # type: ignore[arg-type]
    assert exc.value.status_code == 404


def test_parse_runner_llm() -> None:
    assert parse_runner_llm("") is None
    assert parse_runner_llm("zai-coding:glm-5.3") == RunnerLLMChoice("zai-coding", "glm-5.3")
    assert parse_runner_llm("claude-code:") == RunnerLLMChoice("claude-code", "")
    for bad in ("glm-5.3", ":glm", "zai coding:glm"):
        with pytest.raises(ValueError):
            parse_runner_llm(bad)

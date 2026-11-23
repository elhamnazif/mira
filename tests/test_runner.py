"""Runner choice (config, per-repo override) and the review_jobs record."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mira.config import ReviewConfig, load_config
from mira.dashboard.db import AppDatabase
from mira.exceptions import ConfigError
from mira.runner import (
    IllegalJobTransition,
    JobKind,
    JobStatus,
    ReviewJob,
    ReviewRestTask,
    ReviewTask,
    RunnerKind,
    RunnerTarget,
    ThreadReply,
    ThreadReplyTask,
    parse_runner_override,
    resolve_runner_target,
)

LOCAL = RunnerTarget(RunnerKind.LOCAL)
HOSTED = RunnerTarget(RunnerKind.ACTIONS, ("ubuntu-latest",))
SORA = RunnerTarget(RunnerKind.ACTIONS, ("self-hosted", "sora"))


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    return AppDatabase(url="", admin_password="admin")


class TestResolve:
    def test_default_is_local(self):
        assert resolve_runner_target("", ReviewConfig()) == LOCAL

    def test_deployment_actions_default_uses_runs_on(self):
        review = ReviewConfig(runner="actions", runs_on=["self-hosted", "sora"])
        assert resolve_runner_target("", review) == SORA

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ("local", LOCAL),
            ("actions:self-hosted,sora", SORA),
            ("actions:self-hosted, sora", SORA),
        ],
    )
    def test_repo_override_beats_actions_default(self, override: str, expected: RunnerTarget):
        review = ReviewConfig(runner="actions", runs_on=["ubuntu-latest"])
        assert resolve_runner_target(override, review) == expected

    def test_repo_override_beats_local_default(self):
        assert resolve_runner_target("actions:ubuntu-latest", ReviewConfig()) == HOSTED

    @pytest.mark.parametrize("bad", ["actions", "actions:", "actions:a,,b", "remote", "local:x"])
    def test_malformed_override_raises(self, bad: str):
        with pytest.raises(ValueError):
            parse_runner_override(bad)

    def test_target_rejects_mismatched_labels(self):
        with pytest.raises(ValueError):
            RunnerTarget(RunnerKind.ACTIONS)
        with pytest.raises(ValueError):
            RunnerTarget(RunnerKind.LOCAL, ("ubuntu-latest",))


class TestConfig:
    def test_deployment_config_sets_runner(self, tmp_path: Path):
        path = tmp_path / "deployment.yaml"
        path.write_text("review:\n  runner: actions\n  runs_on: [self-hosted, sora]\n")
        review = load_config(path, trust_execution_settings=True, use_dashboard=False).review
        assert review.runner is RunnerKind.ACTIONS
        assert review.runs_on == ["self-hosted", "sora"]

    def test_repo_mira_yaml_cannot_set_runner(self, tmp_path: Path):
        path = tmp_path / ".mira.yaml"
        path.write_text(
            "review:\n  runner: actions\n  runs_on: [self-hosted, sora]\n  walkthrough: false\n"
        )
        review = load_config(path, use_dashboard=False).review
        assert review.runner is RunnerKind.LOCAL
        assert review.runs_on == ["ubuntu-latest"]
        assert review.walkthrough is False  # other review keys still apply

    @pytest.mark.parametrize("runs_on", ["[]", "['']", "['a,b']"])
    def test_runs_on_rejects_bad_labels(self, tmp_path: Path, runs_on: str):
        path = tmp_path / "deployment.yaml"
        path.write_text(f"review:\n  runner: actions\n  runs_on: {runs_on}\n")
        with pytest.raises(ConfigError):
            load_config(path, trust_execution_settings=True, use_dashboard=False)


class TestRepoRunnerColumn:
    def test_defaults_to_inherit(self, db: AppDatabase):
        db.register_repo("o", "r")
        record = db.get_repo("o", "r")
        assert record is not None
        assert record.runner == ""

    def test_set_and_read_back(self, db: AppDatabase):
        db.register_repo("o", "r")
        db.set_repo_runner("o", "r", "actions:self-hosted,sora")
        record = db.get_repo("o", "r")
        assert record is not None
        assert parse_runner_override(record.runner) == SORA

    def test_rejects_malformed_value(self, db: AppDatabase):
        db.register_repo("o", "r")
        with pytest.raises(ValueError):
            db.set_repo_runner("o", "r", "actions:")
        record = db.get_repo("o", "r")
        assert record is not None
        assert record.runner == ""

    def test_migration_adds_column(self, tmp_path: Path):
        raw = tmp_path / "legacy.db"
        conn = sqlite3.connect(raw)
        conn.executescript(
            """
            CREATE TABLE repos (
                platform TEXT NOT NULL DEFAULT 'github',
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                index_mode TEXT NOT NULL DEFAULT 'full',
                files_indexed INTEGER NOT NULL DEFAULT 0,
                file_count_estimate INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                installation_id INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                last_indexed_at REAL NOT NULL DEFAULT 0,
                conventions TEXT NOT NULL DEFAULT '',
                private INTEGER,
                base_branch_policy TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (platform, owner, repo)
            );
            INSERT INTO repos (owner, repo) VALUES ('legacy', 'repo');
            """
        )
        conn.commit()
        conn.close()

        db = AppDatabase(url=str(raw), admin_password="admin")
        record = db.get_repo("legacy", "repo")
        assert record is not None
        assert record.runner == ""


def _job(job_id: str = "j1", head_sha: str = "sha1") -> ReviewJob:
    return ReviewJob(
        id=job_id,
        owner="o",
        repo="r",
        pr_number=7,
        head_sha=head_sha,
        installation_id=184691606,
        runs_on=("self-hosted", "sora"),
        created_at=1.0,
    )


class TestReviewJobs:
    def test_round_trip(self, db: AppDatabase):
        db.create_review_job(_job())
        db.set_review_job_check_run("j1", 51_234_567_890)
        db.set_review_job_run("j1", 45_038_116_508, "https://github.com/o/r/actions/runs/1")
        job = db.get_review_job("j1")
        assert job is not None
        assert job.runs_on == ("self-hosted", "sora")
        assert job.status is JobStatus.DISPATCHED
        assert job.check_run_id == 51_234_567_890
        assert job.run_id == 45_038_116_508
        assert job.run_url == "https://github.com/o/r/actions/runs/1"

    @pytest.mark.parametrize(
        "task",
        [
            ReviewTask(),
            ReviewRestTask(paths=("a.py", "b.py")),
            ThreadReplyTask(
                reply=ThreadReply(comment_id=10, body="why?", comment_node_id="C_10", line=4),
                may_resolve=True,
            ),
        ],
    )
    def test_a_jobs_task_round_trips(self, db: AppDatabase, task):
        job = _job()
        job.task = task
        db.create_review_job(job)
        stored = db.get_review_job("j1")
        assert stored is not None
        assert stored.task == task
        assert db._rows("SELECT kind FROM review_jobs") == [(task.kind.value,)]

    def test_active_jobs_exclude_finished(self, db: AppDatabase):
        db.create_review_job(_job("old"))
        db.create_review_job(_job("new", head_sha="sha2"))
        db.transition_review_job("old", JobStatus.DISPATCHED, JobStatus.SUPERSEDED)
        assert [j.id for j in db.list_active_review_jobs("o", "r", 7)] == ["new"]
        assert db.list_active_review_jobs("o", "r", 8) == []

    def test_transition_stamps_and_records_error(self, db: AppDatabase):
        db.create_review_job(_job())
        assert db.transition_review_job("j1", JobStatus.DISPATCHED, JobStatus.CLAIMED)
        assert db.transition_review_job(
            "j1", JobStatus.CLAIMED, JobStatus.FAILED, error="runner crashed"
        )
        job = db.get_review_job("j1")
        assert job is not None
        assert job.status is JobStatus.FAILED
        assert job.error == "runner crashed"
        assert job.claimed_at > 0
        assert job.finished_at > 0

    def test_transition_loses_race_when_status_moved(self, db: AppDatabase):
        db.create_review_job(_job())
        assert db.transition_review_job("j1", JobStatus.DISPATCHED, JobStatus.SUPERSEDED)
        assert not db.transition_review_job("j1", JobStatus.DISPATCHED, JobStatus.CLAIMED)
        job = db.get_review_job("j1")
        assert job is not None
        assert job.status is JobStatus.SUPERSEDED

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (JobStatus.DISPATCHED, JobStatus.SUCCEEDED),
            (JobStatus.SUCCEEDED, JobStatus.FAILED),
            (JobStatus.SUPERSEDED, JobStatus.CLAIMED),
            (JobStatus.FAILED, JobStatus.DISPATCHED),
            (JobStatus.EXPIRED, JobStatus.CLAIMED),
        ],
    )
    def test_illegal_transition_raises(self, db: AppDatabase, old: JobStatus, new: JobStatus):
        with pytest.raises(IllegalJobTransition):
            db.transition_review_job("j1", old, new)

    def test_result_usage_is_recorded_once(self, db: AppDatabase):
        db.create_review_job(_job())
        assert db.get_review_job("j1").usage is None  # type: ignore[union-attr]
        assert db.record_review_job_result("j1", {"model": "opus", "input_tokens": 10})
        assert not db.record_review_job_result("j1", {"model": "other"})
        job = db.get_review_job("j1")
        assert job is not None
        assert job.usage == {"model": "opus", "input_tokens": 10}

    def test_migration_adds_usage_and_task_columns(self, tmp_path: Path):
        raw = tmp_path / "pr2.db"
        conn = sqlite3.connect(raw)
        conn.executescript(
            """
            CREATE TABLE review_jobs (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                head_sha TEXT NOT NULL,
                installation_id INTEGER NOT NULL DEFAULT 0,
                runs_on TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                check_run_id INTEGER,
                run_id INTEGER,
                run_url TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0,
                claimed_at REAL NOT NULL DEFAULT 0,
                finished_at REAL NOT NULL DEFAULT 0
            );
            INSERT INTO review_jobs (id, owner, repo, pr_number, head_sha, status)
            VALUES ('old', 'o', 'r', 7, 'sha', 'dispatched');
            """
        )
        conn.commit()
        conn.close()

        db = AppDatabase(url=str(raw), admin_password="admin")
        job = db.get_review_job("old")
        assert job is not None
        assert job.usage is None
        # Every job before job kinds was a review.
        assert job.kind is JobKind.REVIEW and job.task == ReviewTask()
        assert db.record_review_job_result("old", {})

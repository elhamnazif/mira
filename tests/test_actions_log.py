"""The runner's logging: debug from the Actions env, and grouped LLM passes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from click.testing import CliRunner

from mira import actions_log, cli, runner_client
from mira.actions_log import in_log_group, setup_runner_logging

log = logging.getLogger("mira.test")


@pytest.fixture
def bare_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the root logger that a test hands to ``setup_runner_logging``."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(actions_log, "_grouping_handler", None)
    for key in ("GITHUB_ACTIONS", "RUNNER_DEBUG", "ACTIONS_STEP_DEBUG"):
        monkeypatch.delenv(key, raising=False)
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _fresh_root() -> None:
    """No root handlers, as in a fresh ``mira runner``. Pytest adds its own
    for each test's call, so this runs inside the test."""
    logging.getLogger().handlers.clear()


@pytest.mark.parametrize(
    ("env", "level"),
    [
        ({}, logging.INFO),
        ({"RUNNER_DEBUG": "1"}, logging.DEBUG),
        ({"ACTIONS_STEP_DEBUG": "true"}, logging.DEBUG),
    ],
)
def test_the_runs_debug_setting_turns_on_verbose(
    bare_root: None, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], level: int
) -> None:
    async def run_job(*_: Any, **__: Any) -> int:
        return 0

    monkeypatch.setattr(runner_client, "run_job", run_job)
    _fresh_root()

    result = CliRunner().invoke(
        cli.main, ["runner", "--job", "j", "--server", "https://mira.example"], env=env
    )

    assert result.exit_code == 0, result.output
    assert logging.getLogger().level == level


async def _two_passes() -> None:
    async def chunk(n: int) -> int:
        log.info("chunk %d starts", n)
        await asyncio.sleep(0)
        log.info("chunk %d ends", n)
        return n

    results = await asyncio.gather(
        in_log_group("chunk 1/2", chunk(1)), in_log_group("chunk 2/2", chunk(2))
    )
    assert results == [1, 2]
    log.info("all done")


async def test_each_pass_prints_as_one_group_under_actions(
    bare_root: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    _fresh_root()
    setup_runner_logging(verbose=False)

    await _two_passes()

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "::group::chunk 1/2",
        "mira.test INFO: chunk 1 starts",
        "mira.test INFO: chunk 1 ends",
        "::endgroup::",
        "::group::chunk 2/2",
        "mira.test INFO: chunk 2 starts",
        "mira.test INFO: chunk 2 ends",
        "::endgroup::",
        "mira.test INFO: all done",
    ]


async def test_outside_actions_lines_print_as_they_come(
    bare_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    _fresh_root()
    setup_runner_logging(verbose=False)

    await _two_passes()

    assert capsys.readouterr().out.splitlines() == [
        "mira.test INFO: chunk 1 starts",
        "mira.test INFO: chunk 2 starts",
        "mira.test INFO: chunk 1 ends",
        "mira.test INFO: chunk 2 ends",
        "mira.test INFO: all done",
    ]


async def test_a_failed_pass_still_prints_its_group(
    bare_root: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    _fresh_root()
    setup_runner_logging(verbose=False)

    async def fail() -> None:
        log.warning("about to fail")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await in_log_group("security pass", fail())

    assert capsys.readouterr().out.splitlines() == [
        "::group::security pass",
        "mira.test WARNING: about to fail",
        "::endgroup::",
    ]

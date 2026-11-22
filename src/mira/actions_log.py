"""GitHub Actions log groups for the review's LLM passes.

Once ``mira runner`` calls :func:`setup_runner_logging` under Actions
(``GITHUB_ACTIONS=true``), :func:`in_log_group` holds back the log lines
of one pass and prints them as one collapsible ``::group::`` when the pass
ends. The passes run concurrently, and Actions groups cannot nest or
interleave, so lines printed as they come could not be grouped. Everywhere
else it changes nothing.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Awaitable
from contextlib import nullcontext
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")

# The current pass's held-back lines. Each asyncio task has its own copy.
_held: ContextVar[list[str] | None] = ContextVar("mira_actions_log_group", default=None)
# The runner's stdout handler, when its lines are grouped.
_grouping_handler: _GroupingHandler | None = None


def setup_runner_logging(verbose: bool) -> None:
    """Log to stdout, in groups under Actions. Debug logs are on with
    ``verbose`` or when the run asked for them: "Re-run with debug logging"
    sets ``RUNNER_DEBUG=1``, and the ``ACTIONS_STEP_DEBUG`` secret does too."""
    global _grouping_handler
    debug = os.environ.get("RUNNER_DEBUG") == "1" or os.environ.get("ACTIONS_STEP_DEBUG") == "true"
    handler = _GroupingHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG if verbose or debug else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
        handlers=[handler],
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        _grouping_handler = handler


async def in_log_group(label: str, awaitable: Awaitable[T]) -> T:
    """Await ``awaitable`` with its log lines grouped under ``label``. A
    group inside another one joins it, and a group with no lines is not
    printed."""
    handler = _grouping_handler
    if handler is None or _held.get() is not None:
        return await awaitable
    lines: list[str] = []
    token = _held.set(lines)
    try:
        return await awaitable
    finally:
        _held.reset(token)
        if lines:
            with handler.lock or nullcontext():
                handler.stream.write("".join([f"::group::{label}\n", *lines, "::endgroup::\n"]))
                handler.flush()


class _GroupingHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """A stream handler that holds a record back while a log group is open."""

    def emit(self, record: logging.LogRecord) -> None:
        lines = _held.get()
        if lines is None:
            super().emit(record)
            return
        try:
            lines.append(self.format(record) + "\n")
        except Exception:
            self.handleError(record)

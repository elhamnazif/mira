"""What a PR's own code changed between two reviewed versions.

A compare diff from the last reviewed head to the new one carries the base
branch's changes after a rebase. Here each version's patch is taken over its
own merge base instead, the old patch is replayed onto the new base, and the
result is diffed against the new head, so whatever the new base brought in
drops out.

The idea is Pullfrog's range-diff (utils/rangeDiff.ts, MIT,
https://github.com/pullfrog/pullfrog). Pullfrog replays with
``git merge-tree`` and compares with ``git range-diff``. Mira can review
without a checkout, so this does both through the provider API and keeps the
new head's line numbers, which review comments anchor to.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import unidiff

from mira.dashboard.db import ReviewedVersion
from mira.exceptions import ProviderError
from mira.models import PRInfo
from mira.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class _NoReplay(Exception):
    """The old version's patch does not apply to the new base."""


@dataclass(frozen=True)
class _FilePatch:
    text: str  # this file's section of the unified diff, headers included
    file: unidiff.PatchedFile


async def incremental_diff(
    provider: BaseProvider, pr_info: PRInfo, last: ReviewedVersion, pr_diff: str
) -> str:
    """The diff to review in round 2+: what changed in the PR since ``last``.

    ``pr_diff`` is the PR's current diff over its merge base. Falls back to
    the compare diff from the last reviewed head, and says why, when the last
    review recorded no base or the old version's diff can't be fetched.
    """
    if not last.base_sha:
        reason = "the last review recorded no base SHA"
    else:
        try:
            old_diff = await provider.get_compare_diff(pr_info, last.base_sha, last.head_sha)
            if old_diff.strip():
                fetches = asyncio.Semaphore(8)

                async def read_head(path: str) -> str:
                    async with fetches:
                        return await provider.get_file_content(pr_info, path, pr_info.head_sha)

                return await range_diff(old_diff, pr_diff, read_head)
            reason = "the old version's diff is empty"
        except (ProviderError, unidiff.UnidiffParseError) as exc:
            reason = str(exc)
    logger.info(
        "Range-diff unavailable on %s, using the compare diff from %s: %s",
        pr_info.url,
        last.head_sha[:8],
        reason,
    )
    return await provider.get_compare_diff(pr_info, last.head_sha, pr_info.head_sha)


async def range_diff(
    old_diff: str, new_diff: str, read_head: Callable[[str], Awaitable[str]]
) -> str:
    """The files of ``new_diff`` whose change differs from ``old_diff``'s,
    each as a diff from the old change replayed onto the new base to the
    new head. ``read_head`` returns a file's content at the new head."""
    old = _split(old_diff)
    new = _split(new_diff)
    parts = await asyncio.gather(
        *(_file_delta(path, old.get(path), patch, read_head) for path, patch in new.items())
    )
    return "".join(parts)


async def _file_delta(
    path: str,
    old: _FilePatch | None,
    new: _FilePatch,
    read_head: Callable[[str], Awaitable[str]],
) -> str:
    if old is None:
        return new.text
    if not len(old.file) or not len(new.file):
        # Binary, a pure rename or a mode change: no hunks to compare.
        return "" if old.text == new.text else new.text
    if _body(old.file) == _body(new.file):
        return ""
    if new.file.is_removed_file:
        return new.text
    try:
        head = _lines(await read_head(path))
        replayed = _apply(old.file, _unapply(new.file, head))
    except (ProviderError, _NoReplay) as exc:
        logger.info(
            "Range-diff: reviewing all of %s, its old version does not replay: %s", path, exc
        )
        return new.text
    return _unified(path, replayed, head)


def _split(diff: str) -> dict[str, _FilePatch]:
    files: dict[str, _FilePatch] = {}
    for section in re.split(r"(?m)^(?=diff --git )", diff):
        for patched in unidiff.PatchSet(section):
            files[patched.path] = _FilePatch(section, patched)
    return files


def _body(file: unidiff.PatchedFile) -> list[tuple[str, str]]:
    """The hunks' lines without their positions, which a rebase shifts."""
    return [(line.line_type, line.value) for hunk in file for line in hunk]


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


def _sides(hunk: unidiff.Hunk) -> tuple[list[str], list[str]]:
    source: list[str] = []
    target: list[str] = []
    for line in hunk:
        value = line.value.removesuffix("\n")
        if line.is_context:
            source.append(value)
            target.append(value)
        elif line.is_removed:
            source.append(value)
        elif line.is_added:
            target.append(value)
    return source, target


def _index(start: int, length: int) -> int:
    """A hunk side's first line, 0-based. An empty side names the line before it."""
    return start - 1 if length else start


def _unapply(file: unidiff.PatchedFile, head: list[str]) -> list[str]:
    """The base ``file``'s patch was taken over, rebuilt from the head it produced."""
    base: list[str] = []
    pos = 0
    for hunk in file:
        source, target = _sides(hunk)
        at = _index(hunk.target_start, hunk.target_length)
        if at < pos or head[at : at + len(target)] != target:
            raise _NoReplay(f"the head does not match hunk {hunk.target_start}")
        base += head[pos:at] + source
        pos = at + len(target)
    return base + head[pos:]


def _apply(file: unidiff.PatchedFile, base: list[str]) -> list[str]:
    """``file``'s patch applied to ``base``. Each hunk lands where its old
    lines match exactly, nearest to where the hunk says they were."""
    out: list[str] = []
    pos = 0
    for hunk in file:
        source, target = _sides(hunk)
        at = _find(base, source, pos, _index(hunk.source_start, hunk.source_length))
        if at is None:
            raise _NoReplay(f"hunk {hunk.source_start} has no match on the new base")
        out += base[pos:at] + target
        pos = at + len(source)
    return out + base[pos:]


def _find(lines: list[str], block: list[str], start: int, near: int) -> int | None:
    if not block:
        return min(max(near, start), len(lines))
    matches = [
        i for i in range(start, len(lines) - len(block) + 1) if lines[i : i + len(block)] == block
    ]
    return min(matches, key=lambda i: abs(i - near), default=None)


def _unified(path: str, old: list[str], new: list[str], context: int = 3) -> str:
    # difflib.unified_diff can't turn off autojunk, which misaligns files
    # over 200 lines that repeat a line often (blank lines, closing braces).
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    out: list[str] = []
    for group in matcher.get_grouped_opcodes(context):
        first, last = group[0], group[-1]
        out.append(f"@@ -{_range(first[1], last[2])} +{_range(first[3], last[4])} @@")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out += [" " + line for line in old[i1:i2]]
                continue
            out += ["-" + line for line in old[i1:i2]]
            out += ["+" + line for line in new[j1:j2]]
    if not out:
        return ""
    header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
    return header + "\n".join(out) + "\n"


def _range(start: int, stop: int) -> str:
    length = stop - start
    first = start + 1 if length else start
    return str(first) if length == 1 else f"{first},{length}"

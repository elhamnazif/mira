"""Round 2+ incremental diffs over a real git repo whose base branch moves
under the PR. The fake provider answers the compare and contents API calls
the way GitHub does: a compare diff is taken over the merge base."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
import unidiff

from mira.core.range_diff import incremental_diff
from mira.dashboard.db import ReviewedVersion
from mira.exceptions import ProviderError
from mira.models import PRInfo

APP = [f"app line {n}" for n in range(1, 41)]
LIB = [f"lib line {n}" for n in range(1, 11)]


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        )
        return done.stdout

    def commit(self, files: dict[str, list[str] | None], message: str = "c") -> str:
        for path, lines in files.items():
            if lines is None:
                self.git("rm", "-q", path)
                continue
            (self.root / path).write_text("".join(line + "\n" for line in lines))
            self.git("add", path)
        self.git("commit", "-q", "-m", message)
        return self.head()

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()


class GitProvider:
    """The two provider calls incremental_diff makes, answered from a repo."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    async def get_compare_diff(self, pr_info: PRInfo, base_sha: str, head_sha: str) -> str:
        try:
            return self.repo.git("diff", f"{base_sha}...{head_sha}")
        except subprocess.CalledProcessError as exc:
            raise ProviderError(f"compare {base_sha[:8]}...{head_sha[:8]}: 404") from exc

    async def get_file_content(self, pr_info: PRInfo, path: str, ref: str) -> str:
        try:
            return self.repo.git("show", f"{ref}:{path}")
        except subprocess.CalledProcessError as exc:
            raise ProviderError(f"{path}@{ref[:8]}: 404") from exc


def edit(lines: list[str], changes: dict[int, str]) -> list[str]:
    return [changes.get(n, line) for n, line in enumerate(lines, start=1)]


def changes(diff: str) -> set[tuple[str, str, int, str]]:
    """Every added or removed line: path, sign, its line number, its text."""
    out = set()
    for file in unidiff.PatchSet(diff):
        for hunk in file:
            for line in hunk:
                if line.is_added:
                    out.add((file.path, "+", line.target_line_no, line.value.rstrip("\n")))
                elif line.is_removed:
                    out.add((file.path, "-", line.source_line_no, line.value.rstrip("\n")))
    return out


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    repo = Repo(tmp_path)
    repo.commit({"app.py": APP, "lib.py": LIB}, "base")
    return repo


async def review(repo: Repo, last: ReviewedVersion) -> str:
    """What round 2 reviews, given the PR branch checked out at its new head."""
    head = repo.head()
    pr = PRInfo(
        title="t",
        description="",
        base_branch="main",
        head_branch="pr",
        url="https://github.com/o/r/pull/1",
        number=1,
        owner="o",
        repo="r",
        head_sha=head,
    )
    pr_diff = repo.git("diff", f"main...{head}")
    return await incremental_diff(GitProvider(repo), pr, last, pr_diff)  # type: ignore[arg-type]


def open_v1(repo: Repo) -> ReviewedVersion:
    """Branch the PR off main and push its first version."""
    base = repo.head()
    repo.git("checkout", "-q", "-b", "pr")
    head = repo.commit(
        {"app.py": edit(APP, {5: "pr line 5"}), "lib.py": edit(LIB, {3: "pr lib 3"})}, "v1"
    )
    return ReviewedVersion(head_sha=head, base_sha=base)


def move_main(repo: Repo) -> None:
    """Land unrelated work on main, in a file the PR also touches."""
    repo.git("checkout", "-q", "main")
    repo.commit({"app.py": edit(APP, {30: "main line 30"}), "other.py": ["main only"]}, "main")
    repo.git("checkout", "-q", "pr")


async def test_a_plain_push_reviews_the_same_lines_as_the_compare_diff(repo: Repo) -> None:
    v1 = open_v1(repo)
    move_main(repo)
    repo.commit({"app.py": edit(APP, {5: "pr line 5", 20: "pr line 20"})}, "v2")

    compare = repo.git("diff", f"{v1.head_sha}...HEAD")
    assert (
        changes(await review(repo, v1))
        == changes(compare)
        == {
            ("app.py", "-", 20, "app line 20"),
            ("app.py", "+", 20, "pr line 20"),
        }
    )


async def test_a_rebase_onto_a_moved_base_with_no_pr_change_reviews_nothing(repo: Repo) -> None:
    v1 = open_v1(repo)
    move_main(repo)
    repo.git("rebase", "-q", "main")

    # The compare diff from the old head carries main's change in app.py.
    compare = repo.git("diff", f"{v1.head_sha}...HEAD")
    assert ("app.py", "+", 30, "main line 30") in changes(compare)
    assert await review(repo, v1) == ""


async def test_a_rebase_with_a_real_change_reviews_only_that_change(repo: Repo) -> None:
    v1 = open_v1(repo)
    move_main(repo)
    repo.git("rebase", "-q", "main")
    repo.commit({"app.py": edit(APP, {5: "pr line 5", 12: "pr line 12", 30: "main line 30"})})

    assert changes(await review(repo, v1)) == {
        ("app.py", "-", 12, "app line 12"),
        ("app.py", "+", 12, "pr line 12"),
    }


async def test_a_force_push_dropping_a_commit_reviews_only_the_drop(repo: Repo) -> None:
    open_v1(repo)
    first = repo.head()
    repo.commit({"app.py": edit(APP, {5: "pr line 5", 20: "pr line 20"}), "extra.py": ["x"]})
    v1 = ReviewedVersion(head_sha=repo.head(), base_sha=repo.git("rev-parse", "main").strip())
    repo.git("reset", "-q", "--hard", first)

    # app.py line 20 goes back to main's; extra.py is no longer the PR's.
    assert changes(await review(repo, v1)) == {
        ("app.py", "-", 20, "pr line 20"),
        ("app.py", "+", 20, "app line 20"),
    }


async def test_a_file_whose_old_change_does_not_replay_is_reviewed_whole(repo: Repo) -> None:
    v1 = open_v1(repo)
    # main edits a line inside the context of the PR's lib.py hunk.
    repo.git("checkout", "-q", "main")
    repo.commit({"lib.py": edit(LIB, {5: "main lib 5"})})
    repo.git("checkout", "-q", "pr")
    repo.git("rebase", "-q", "main")

    # lib.py's PR change, without main's line in it; app.py is unchanged.
    assert changes(await review(repo, v1)) == {
        ("lib.py", "-", 3, "lib line 3"),
        ("lib.py", "+", 3, "pr lib 3"),
    }


async def test_with_no_base_recorded_the_compare_diff_is_reviewed(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
    v1 = open_v1(repo)
    move_main(repo)
    repo.git("rebase", "-q", "main")
    old = ReviewedVersion(head_sha=v1.head_sha)

    with caplog.at_level(logging.INFO, logger="mira.core.range_diff"):
        diff = await review(repo, old)
    assert diff == repo.git("diff", f"{v1.head_sha}...HEAD")
    assert "recorded no base SHA" in caplog.text


async def test_a_missing_old_head_falls_back_to_the_compare_diff(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
    v1 = open_v1(repo)
    repo.commit({"app.py": edit(APP, {5: "pr line 5", 20: "pr line 20"})})
    gone = ReviewedVersion(head_sha="0" * 40, base_sha=v1.base_sha)

    # The compare from the old head fails the same way; the engine then
    # reviews the full diff, as it did before range-diffs.
    with caplog.at_level(logging.INFO, logger="mira.core.range_diff"), pytest.raises(ProviderError):
        await review(repo, gone)
    assert "Range-diff unavailable" in caplog.text and "404" in caplog.text

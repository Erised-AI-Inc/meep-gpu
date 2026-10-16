"""The Metal round's tree check reads ``git status --porcelain`` line by line.

``metal_round.Round.tree_is_the_commit`` refuses a checkout that is not the frozen
commit or that holds local changes, and lets the route stage find exactly one change:
the Metal ledger the bind rewrites after the fleet. A porcelain line is two status
columns, a space and the path, and an unstaged change starts with a space
(``" M <path>"``). A strip of the whole output removed that space from the first line,
so the path was read one character short and the one allowed change was refused: the
state the bind leaves, which the runbook's route step expects. These tests drive the
check on a throwaway repository in the four states that showed it (host-only; needs
``git``).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROUND = Path(__file__).resolve().parent / "rounds" / "1.0"
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import metal_round  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

LEDGER = "meep_gpu/metal_kernels/fingerprints.json"
OTHER = "meep_gpu/cuda_kernels/fingerprints.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture()
def checkout(tmp_path):
    """A committed repository holding the Metal ledger and a path that sorts before it,
    and a Round pointed at it (logs outside the repository)."""
    repo = tmp_path / "repo"
    for rel in (LEDGER, OTHER):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text('{"committed": true}\n', encoding="utf-8")
    _git(repo.parent, "init", "-q", str(repo))
    for key, value in (("user.email", "round@test.invalid"), ("user.name", "round"),
                       ("commit.gpgsign", "false"), ("core.autocrlf", "false")):
        _git(repo, "config", key, value)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "frozen")
    head = _git(repo, "rev-parse", "HEAD").strip()
    args = SimpleNamespace(repo=str(repo), stamp="test", results=str(tmp_path / "results"),
                           logs=str(tmp_path / "logs"), python=sys.executable, expect=head)
    return repo, metal_round.Round(args)


def _status(repo: Path) -> str:
    return _git(repo, "status", "--porcelain")


def test_the_four_states_read_as_the_route_rule_documents(checkout):
    repo, rnd = checkout
    observed = []
    assert _status(repo) == ""
    observed.append(rnd.tree_is_the_commit("route"))
    (repo / LEDGER).write_text('{"rebound": true}\n', encoding="utf-8")
    # The state the bind leaves: the Metal ledger modified and not staged.
    assert _status(repo) == f" M {LEDGER}\n"
    observed.append(rnd.tree_is_the_commit("route"))
    (repo / OTHER).write_text('{"another": true}\n', encoding="utf-8")
    assert _status(repo) == f" M {OTHER}\n M {LEDGER}\n"
    observed.append(rnd.tree_is_the_commit("route"))
    _git(repo, "checkout", "--", OTHER)
    _git(repo, "add", LEDGER)
    assert _status(repo) == f"M  {LEDGER}\n"
    observed.append(rnd.tree_is_the_commit("route"))
    assert [why is None for why in observed] == [True, True, False, True], observed
    # The refusal names the stray path whole, and only it.
    assert observed[2] == f"the checkout has local changes: [' M {OTHER}']"


def test_a_stage_other_than_the_route_allows_no_change_at_all(checkout):
    repo, rnd = checkout
    (repo / LEDGER).write_text('{"rebound": true}\n', encoding="utf-8")
    for stage in ("preflight", "fleet"):
        assert rnd.tree_is_the_commit(stage) == (
            f"the checkout has local changes: [' M {LEDGER}']")
    assert rnd.tree_is_the_commit("route") is None


def test_an_untracked_file_and_another_commit_are_refused_on_the_route(checkout):
    repo, rnd = checkout
    (repo / "stray.txt").write_text("x\n", encoding="utf-8")
    assert rnd.tree_is_the_commit("route") == "the checkout has local changes: ['?? stray.txt']"
    (repo / "stray.txt").unlink()
    rnd.args.expect = "0" * 40
    assert rnd.tree_is_the_commit("route").startswith("HEAD is ")


def test_the_route_allowance_is_the_metal_ledger_alone():
    assert metal_round.ROUTE_STAGE_LEDGERS == (LEDGER,)

"""The public export (``tools/release/export_public.py``, RELEASING.md), end to end on
two small repositories built here: a development repository with a release commit
and a public repository with one release on ``main``.

What is pinned is behaviour, not code:

* a green export writes ONE commit whose only parent is the public ``main``, whose
  tree is the release commit's minus the manifest's exclusions, byte for byte, and an
  annotated tag on it, in the scratch clone only: the development and the public
  repositories are unchanged, and nothing is pushed;
* the release commit's own hygiene guard runs on the EXPORTED tree (an excluded file
  is invisible to it), and a red guard, or no guard, refuses the export;
* every other refusal fires on its own defect and commits nothing: a required public
  file that is absent, a public-only file the export would drop, a reference to or a
  digest of an excluded file, a version or date that disagrees, a tag or version the
  public repository already has, and a message out of the release form.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release" / "export_public.py"
AT = "@"
DOMAIN = "example.invalid"
OWNER = ("Ada", "Lovelace", "ada" + AT + DOMAIN)
SECOND = ("Grace", "Hopper")
NO_REPLY = "1+grace" + AT + "users.noreply.github.com"
PRIVATE = "rounds/handoff_runbook.html"
GUARD = '''import json, subprocess, sys
names = ["LEAK" + "-MARK"]
if "--names-file" in sys.argv:
    names += json.load(open(sys.argv[sys.argv.index("--names-file") + 1]))
files = [f for f in subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\\0") if f]
leaks = [f for f in files if any(n in open(f, encoding="utf-8").read() for n in names)]
print(f"scanned {len(files)} files, {len(leaks)} with a leak: {leaks}")
sys.exit(1 if leaks else 0)
'''


def _load():
    spec = importlib.util.spec_from_file_location("export_public_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_public = _load()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path, monkeypatch):
    """No global or system git configuration reaches the repositories built here."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    identity = {"GIT_AUTHOR_NAME": "builder", "GIT_COMMITTER_NAME": "builder",
                "GIT_AUTHOR_EMAIL": "builder" + AT + DOMAIN,
                "GIT_COMMITTER_EMAIL": "builder" + AT + DOMAIN}
    for name, value in identity.items():
        monkeypatch.setenv(name, value)


def write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        if text is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def init_main(repo: Path) -> None:
    """An empty repository whose first commit lands on ``main``. ``git init -b`` needs
    git 2.28 or newer; pointing HEAD at the branch by name works on any git."""
    git(repo, "init", "-q")
    git(repo, "symbolic-ref", "HEAD", "refs/heads/main")


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A", "-f", ".")
    git(repo, "commit", "-q", "--no-verify", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def manifest(**changes) -> str:
    document = {
        "schema": "meep_gpu.public_export", "version": 1, "release_time": "12:00:00 -0700",
        "excluded": [{"path": PRIVATE, "reason": "a hand-off page for one cluster"}],
        "required_public_files": [{"path": ".github/workflows/publish.yml",
                                   "reason": "the package index's publisher names it"}],
        "guard": {"path": "tools/ci/check_hygiene.py", "arguments": [],
                  "names_file_option": "--names-file"},
    }
    document.update(changes)
    return json.dumps(document, indent=2) + "\n"


def citation(version: str = "2.0.0", date: str = "2026-01-02") -> str:
    given, family, email = OWNER
    return (f'cff-version: 1.2.0\ntitle: "demo"\nauthors:\n'
            f'  - family-names: "{family}"\n    given-names: "{given}"\n    email: "{email}"\n'
            f'  - family-names: "{SECOND[1]}"\n    given-names: "{SECOND[0]}"\n'
            f'contact:\n  - family-names: "{family}"\n    given-names: "{given}"\n'
            f'    email: "{email}"\nversion: "{version}"\ndate-released: "{date}"\n')


def release_files(version: str = "2.0.0", date: str = "2026-01-02") -> dict:
    return {
        "pyproject.toml": f'[project]\nname = "demo-pkg"\nversion = "{version}"\n',
        "CITATION.cff": citation(version, date),
        "CHANGELOG.md": f"# Changelog\n\n## {version} — the second release ({date})\n\n- more\n",
        "README.md": "The demo package.\n",
        ".gitignore": "*.log\n",
        "kept.log": "a tracked file that the ignore rule would hide from a plain add\n",
        ".github/workflows/publish.yml": "name: publish\n",
        "tools/release/export_manifest.json": manifest(),
        "tools/ci/check_hygiene.py": GUARD,
        PRIVATE: "a page for one cluster: LEAK-MARK\n",
    }


@pytest.fixture
def repos(tmp_path):
    """(development repository, public repository, scratch directory, message file)."""
    public = tmp_path / "public"
    public.mkdir()
    init_main(public)
    write(public, {"pyproject.toml": '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n',
                   "README.md": "The demo package.\n", "retired.py": "OLD = 1\n",
                   ".github/workflows/publish.yml": "name: publish\n"})
    commit_all(public, "Summary: demo-pkg 1.0.0")
    git(public, "tag", "-a", "v1.0.0", "-m", "demo-pkg 1.0.0 — the first release")

    dev = tmp_path / "dev"
    dev.mkdir()
    init_main(dev)
    write(dev, {"retired.py": "OLD = 1\n", "README.md": "The demo package.\n"})
    commit_all(dev, "the first release's files")
    write(dev, {"retired.py": None, **release_files()})
    commit_all(dev, "the second release")

    message = tmp_path / "message.txt"
    message.write_text(
        "Summary: demo-pkg 2.0.0, the second release\n\nDescription:\n1. More.\n\n"
        f"Co-authored-by: {' '.join(SECOND)} <{NO_REPLY}>\n", encoding="utf-8")
    return dev, public, tmp_path / "scratch", message


def state(repo: Path) -> str:
    return "\n".join([git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
                      git(repo, "status", "--porcelain", "--untracked-files=all"),
                      git(repo, "count-objects", "-v")])


def run(repos, capsys, *extra: str):
    dev, public, scratch, message = repos
    before = state(dev), state(public)
    status = export_public.main(["--dev-repo", str(dev), "--dev-ref", "HEAD",
                                 "--public-repo", str(public), "--message-file", str(message),
                                 "--scratch", str(scratch), "--report",
                                 str(scratch.parent / "report.json"), *extra])
    out = capsys.readouterr().out
    assert (state(dev), state(public)) == before, "the export wrote to a source repository"
    report = json.loads((scratch.parent / "report.json").read_text())
    return status, out, report


def refused(repos, capsys, check: str, needle: str):
    status, out, report = run(repos, capsys)
    assert status == 1, out
    assert report["result"] == "refused"
    assert not report["checks"][check]["passed"], out
    assert any(needle in reason for reason in report["checks"][check]["reasons"]), out
    clone = repos[2] / "public"
    assert git(clone, "rev-parse", "main") == git(repos[1], "rev-parse", "main")
    assert git(clone, "tag", "--list", "v2.0.0") == ""
    return report


def amend(repos, files: dict) -> None:
    write(repos[0], files)
    commit_all(repos[0], "a change")


# ---------------------------------------------------------------------------


def test_a_green_export_builds_one_commit_on_public_main_and_a_tag_and_pushes_nothing(
        repos, capsys):
    dev, public, scratch, message = repos
    status, out, report = run(repos, capsys)
    assert status == 0, out
    assert report["result"] == "built"
    assert all(check["passed"] for check in report["checks"].values())
    clone = scratch / "public"
    new = git(clone, "rev-parse", "main")
    assert new == report["commit"]
    assert git(clone, "rev-parse", f"{new}^@").split() == [git(public, "rev-parse", "main")]
    assert git(clone, "rev-list", "--count", "main") == "2"
    assert git(clone, "remote") == "", "the clone keeps no remote to push to"
    # The tree is the release commit's, minus the excluded file, entry for entry.
    expected = [line for line in git(dev, "ls-tree", "-r", "HEAD").splitlines()
                if not line.endswith("\t" + PRIVATE)]
    assert git(clone, "ls-tree", "-r", "main").splitlines() == expected
    assert any(line.endswith("\tkept.log") for line in expected)
    # The annotated tag, its title from the CHANGELOG heading.
    assert git(clone, "cat-file", "-t", "v2.0.0") == "tag"
    assert git(clone, "rev-parse", "v2.0.0^{commit}") == new
    assert git(clone, "tag", "-l", "--format=%(contents:subject)", "v2.0.0") == \
        "demo-pkg 2.0.0 — the second release"
    # The identity is the CITATION.cff contact; the date is the release date at noon.
    given, family, email = OWNER
    assert git(clone, "log", "-1", "--format=%an <%ae>|%cn <%ce>|%aI|%cI", "main") == \
        f"{given} {family} <{email}>|{given} {family} <{email}>|" \
        "2026-01-02T12:00:00-07:00|2026-01-02T12:00:00-07:00"
    assert git(clone, "tag", "-l", "--format=%(taggername) %(taggeremail)", "v2.0.0") == \
        f"{given} {family} <{email}>"
    raw = git(clone, "cat-file", "commit", new) + "\n"
    assert raw.split("\n\n", 1)[1] == message.read_text()
    # The removal of a file the development history deleted is accounted for.
    assert report["dropped_from_public_main"] == ["retired.py"]
    # It prints the owner's push commands, the tag last.
    assert out.index("push origin main") < out.index("push origin v2.0.0")


def test_the_guard_runs_on_the_exported_tree_and_a_red_guard_refuses(repos, capsys):
    # The excluded file carries the leak marker and the green export above passes,
    # so the guard sees the exported files only. A leak in an exported file refuses.
    amend(repos, {"docs/notes.md": "a note with LEAK-MARK in it\n"})
    report = refused(repos, capsys, "guard", "hygiene guard is red")
    assert any("docs/notes.md" in line for line in report["guard_output_tail"])


def test_a_local_names_file_reaches_the_guard_and_only_adds_names(repos, capsys, tmp_path):
    amend(repos, {"docs/notes.md": "a note that names zebrahost\n"})
    status, out, _ = run(repos, capsys)
    assert status == 0, out
    names = tmp_path / "names.json"
    names.write_text(json.dumps(["zebrahost"]))
    import shutil
    shutil.rmtree(repos[2])
    status, out, report = run(repos, capsys, "--guard-names-file", str(names))
    assert status == 1 and not report["checks"]["guard"]["passed"], out
    assert (repos[2] / "guard_output.txt").read_text().count("docs/notes.md") == 1


def test_a_names_file_the_manifest_has_no_option_for_cannot_run(repos, capsys, tmp_path):
    document = json.loads(manifest())
    del document["guard"]["names_file_option"]
    amend(repos, {"tools/release/export_manifest.json": json.dumps(document)})
    names = tmp_path / "names.json"
    names.write_text("[]")
    status, out, report = run(repos, capsys, "--guard-names-file", str(names))
    assert status == 2 and "names_file_option" in out


def test_no_guard_in_the_release_commit_refuses(repos, capsys):
    amend(repos, {"tools/ci/check_hygiene.py": None})
    refused(repos, capsys, "guard", "carries no hygiene guard")


def test_a_required_public_file_that_is_absent_refuses(repos, capsys):
    amend(repos, {".github/workflows/publish.yml": None})
    refused(repos, capsys, "required", ".github/workflows/publish.yml")


def test_a_public_only_file_the_export_would_drop_refuses(repos, capsys):
    dev, public, _, _ = repos
    write(public, {"PUBLIC_ONLY.md": "on the public repository only\n"})
    commit_all(public, "Summary: a public-only file")
    report = refused(repos, capsys, "removals", "PUBLIC_ONLY.md")
    assert not any("retired.py" in r for r in report["checks"]["removals"]["reasons"])


def test_a_listed_drop_from_public_is_accepted(repos, capsys):
    dev, public, _, _ = repos
    write(public, {"PUBLIC_ONLY.md": "on the public repository only\n"})
    commit_all(public, "Summary: a public-only file")
    amend(repos, {"tools/release/export_manifest.json": manifest(dropped_from_public=[
        {"path": "PUBLIC_ONLY.md", "reason": "retired with this release"}])})
    status, out, _ = run(repos, capsys)
    assert status == 0, out


def test_a_file_that_names_an_excluded_file_refuses(repos, capsys):
    amend(repos, {"docs/index.md": "See rounds/handoff_runbook.html for the steps.\n"})
    refused(repos, capsys, "references", "docs/index.md:1")


def test_a_file_that_carries_an_excluded_files_digest_refuses(repos, capsys):
    import hashlib
    digest = hashlib.sha256(release_files()[PRIVATE].encode()).hexdigest()
    amend(repos, {"records.json": json.dumps({"pins": {"page": digest[:16]}}) + "\n"})
    refused(repos, capsys, "digests", "records.json")


@pytest.mark.parametrize("files, needle", [
    ({"CITATION.cff": citation(version="2.0.1")}, "CITATION.cff's version"),
    ({"CHANGELOG.md": "# Changelog\n\n## Unreleased\n"}, "CHANGELOG.md has no heading"),
    ({"CITATION.cff": citation(date="2026-01-03")}, "date-released"),
], ids=["version", "heading", "date"])
def test_release_facts_that_disagree_refuse(repos, capsys, files, needle):
    amend(repos, files)
    refused(repos, capsys, "version", needle)


def test_a_version_the_public_main_already_carries_refuses(repos, capsys):
    amend(repos, {**release_files(version="1.0.0")})
    report = refused(repos, capsys, "version", "already")
    reasons = " ".join(report["checks"]["version"]["reasons"])
    assert "tag v1.0.0" in reasons and "already carries version 1.0.0" in reasons


@pytest.mark.parametrize("edit, needle", [
    (lambda m, dev: m.replace("Summary: ", "Title: "), "Summary:"),
    (lambda m, dev: m.split("\n\nCo-authored-by")[0] + "\n", "trailers"),
    (lambda m, dev: m.replace("Grace Hopper", "Someone Else"), "not an author"),
    (lambda m, dev: m.replace("1. More.", "1. More, from " + git(dev, "rev-parse",
                                                                  "--short", "HEAD~1") + "."),
     "development commit"),
    (lambda m, dev: m.replace("1. More.", "1. More, see /" + "home/somebody/x."),
     "local path"),
    (lambda m, dev: m.replace("1. More.", "1. Mail " + "someone" + AT + DOMAIN + "."),
     "address"),
    (lambda m, dev: m + "\n", "stripspace"),
], ids=["summary", "trailers", "author", "commit", "path", "address", "stripspace"])
def test_a_message_out_of_the_release_form_refuses(repos, capsys, edit, needle):
    dev, _, _, message = repos
    message.write_text(edit(message.read_text(), dev), encoding="utf-8")
    refused(repos, capsys, "message", needle)


def test_a_scratch_directory_that_is_not_empty_cannot_run(repos, capsys):
    scratch = repos[2]
    scratch.mkdir()
    (scratch / "left.txt").write_text("from an earlier run\n")
    status, out, report = run(repos, capsys)
    assert status == 2 and report["result"] == "cannot run" and "not empty" in out

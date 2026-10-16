#!/usr/bin/env python
"""Build the public release commit from the development repository, in a scratch clone.

    python tools/release/export_public.py \\
        --dev-repo DEV --dev-ref REF --public-repo PUBLIC \\
        --message-file MESSAGE --scratch DIRECTORY \\
        [--version VERSION] [--date YYYY-MM-DD] [--report REPORT.json] \\
        [--guard-names-file NAMES]

Development happens on the private development repository, and the public
repository receives one squashed commit per release (RELEASING.md). That commit's
tree is the development tree at the release commit REF with the files the export
manifest excludes removed, and nothing else changed: no file is edited for the
public tree. A text that would be false or dangling without the excluded files is
fixed in the development repository, so the two trees never say different things.

WHAT IT DOES, in order. Each step prints one flushed line as it finishes.

  1. Clones PUBLIC into ``DIRECTORY/public`` (no checkout, and no remote kept, so
     nothing can be pushed from it by accident) and archives REF of DEV into
     ``DIRECTORY/tree`` with ``git archive``, so the tree holds tracked files only:
     no ignored cache, build or editor file can enter it.
  2. Reads the export manifest from the archived tree
     (``tools/release/export_manifest.json``: the manifest of the release commit,
     never a local copy) and removes the files it excludes.
  3. Runs every check below, prints each one's verdict, and refuses if any fails:
       required    a file the manifest requires on the public repository is absent
                   (the publish workflow is one; it is verified, never copied in);
       removals    the public main carries a file the export drops, which is
                   neither excluded, nor deleted in REF's history, nor listed under
                   ``dropped_from_public``: a public-only file would be lost;
       references  an exported file names an excluded file, by path or file name;
       digests     an exported file carries the sha256 (or, for Python, the code
                   digest) of an excluded file: a record that binds it would break;
       shape       the ``ls-tree -r`` of the exported tree differs from REF's by
                   anything other than the excluded files;
       version     the version is not the same in ``pyproject.toml``,
                   ``CITATION.cff`` and a ``## <version> — <title> (<date>)``
                   heading of ``CHANGELOG.md``; the date disagrees with
                   ``CITATION.cff``; the tag exists on PUBLIC; or the public main
                   already carries this version (an index never takes it twice);
       message     the message is not in the release form (``Summary:``,
                   ``Description:``, ``Co-authored-by`` trailers naming authors of
                   ``CITATION.cff`` and nothing after them), names a commit of DEV,
                   carries a local path or an address that is not a credit, or is
                   changed by ``git stripspace``;
       guard       the exported tree's own hygiene guard (the manifest's ``guard``)
                   is absent, or exits non-zero on the exported tree. Its whole
                   report is kept as ``DIRECTORY/guard_output.txt``. A local names
                   file (``--guard-names-file``) can only add names to it.
     Nothing is committed when one fails.
  4. Writes ONE commit with the exported tree and the public main as its only
     parent, moves ``main`` of the scratch clone to it, and makes the annotated tag
     ``v<version>``. Author, committer and tagger are the contact of
     ``CITATION.cff``; the dates are the release date at the manifest's time of day.
  5. Prints the commands that push the commit and the tag. It never pushes, and it
     never writes to DEV or PUBLIC.

Exit status 0 when the commit and the tag are built, 1 on a refusal, 2 when the
command cannot run (bad arguments, an unreadable repository or manifest, a scratch
directory that is not empty).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MANIFEST = "tools/release/export_manifest.json"
MANIFEST_SCHEMA = "meep_gpu.public_export"
CODE_IDENTITY = "meep_gpu/code_identity.py"
PUBLIC_BRANCH = "main"
TAG_PREFIX = "v"
#: sha256 prefixes searched for in the exported text (the lengths records quote).
DIGEST_PREFIXES = (64, 16, 12)
HEADING = re.compile(r"^## (?P<version>\S+) — (?P<title>.+) \((?P<date>\d{4}-\d{2}-\d{2})\)\s*$")
TRAILER = re.compile(r"^Co-authored-by: (?P<name>[^<>]+?) <(?P<email>[^<>\s]+)>$")
HEX_TOKEN = re.compile(r"\b[0-9a-f]{7,40}\b")
TIME_OF_DAY = re.compile(r"^\d{2}:\d{2}:\d{2} [+-]\d{4}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
#: A GitHub no-reply address, the form a co-author trailer credits without a mailbox.
NO_REPLY = re.compile(r"^\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com$")
#: Home and scratch directories of a machine. ``/home/user`` is the placeholder the
#: shipped records use, so it is not a machine's path.
LOCAL_PATH = re.compile(r"(?:^|[\s(`'\"=:])(?:/Users/|/home/(?!user\b)|/private/|/tmp/|~/)")
CHECKS = ("required", "removals", "references", "digests", "shape", "version", "message",
          "guard")


class Unusable(Exception):
    """The command cannot run at all (exit status 2)."""


_START = time.monotonic()


def say(text: str) -> None:
    print(f"[{time.monotonic() - _START:6.1f} s] {text}", flush=True)


def git(*arguments: str, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
        input_text: Optional[str] = None, check: bool = True) -> str:
    done = subprocess.run(["git", *arguments], cwd=cwd, env=env, input=input_text,
                          capture_output=True, text=True)
    if check and done.returncode != 0:
        raise Unusable(f"git {' '.join(arguments)} failed ({done.returncode}): "
                       f"{done.stderr.strip()}")
    return done.stdout


def git_ok(*arguments: str, cwd: Optional[Path] = None) -> bool:
    return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# Reading the release facts from the exported tree
# ---------------------------------------------------------------------------


def _entries(document: dict, key: str, required: bool) -> List[dict]:
    entries = document.get(key, [] if not required else None)
    if not isinstance(entries, list):
        raise Unusable(f"{MANIFEST}: {key!r} must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("reason"):
            raise Unusable(f"{MANIFEST}: every {key!r} entry needs a path and a reason "
                           f"({entry!r})")
        if entry["path"].startswith("/") or ".." in Path(entry["path"]).parts:
            raise Unusable(f"{MANIFEST}: {entry['path']!r} is not a path inside the tree")
    return entries


def read_manifest(tree: Path) -> dict:
    path = tree / MANIFEST
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Unusable(f"the export manifest {MANIFEST} cannot be read at the release "
                       f"commit: {error}") from error
    if document.get("schema") != MANIFEST_SCHEMA or document.get("version") != 1:
        raise Unusable(f"{MANIFEST} is not a version-1 {MANIFEST_SCHEMA} document")
    _entries(document, "excluded", True)
    _entries(document, "required_public_files", True)
    _entries(document, "dropped_from_public", False)
    if not TIME_OF_DAY.match(str(document.get("release_time", ""))):
        raise Unusable(f"{MANIFEST}: release_time must read 'HH:MM:SS +HHMM'")
    guard = document.get("guard")
    if (not isinstance(guard, dict) or not isinstance(guard.get("path"), str)
            or not isinstance(guard.get("arguments", []), list)
            or not all(isinstance(a, str) for a in guard.get("arguments", []))):
        raise Unusable(f"{MANIFEST}: 'guard' must name a path and a list of arguments")
    if not isinstance(guard.get("names_file_option", ""), str):
        raise Unusable(f"{MANIFEST}: the guard's 'names_file_option' must be a string")
    return document


def project_field(text: str, field: str) -> Optional[str]:
    """``[project] <field> = "..."`` of a pyproject.toml text, read without a TOML parser
    (the standard library has none before Python 3.11)."""
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if section == "[project]":
            match = re.match(rf'^{field}\s*=\s*"([^"]+)"\s*$', stripped)
            if match:
                return match.group(1)
    return None


def citation(tree: Path) -> Dict[str, object]:
    """The parts of CITATION.cff the release uses: version, date, authors, contact.

    CITATION.cff is YAML; the parse here covers the block form the file is written
    in (top-level keys, and lists of ``- key: "value"`` people) and nothing else."""
    top: Dict[str, str] = {}
    people: Dict[str, List[Dict[str, str]]] = {"authors": [], "contact": []}
    key = None
    try:
        text = (tree / "CITATION.cff").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            name, _, value = line.partition(":")
            key = name.strip()
            value = value.strip().strip('"')
            if value:
                top[key] = value
            continue
        if key in people:
            stripped = line.strip()
            if stripped.startswith("- "):
                people[key].append({})
                stripped = stripped[2:]
            if people[key] and ":" in stripped:
                name, _, value = stripped.partition(":")
                people[key][-1][name.strip()] = value.strip().strip('"')
    return {"version": top.get("version"), "date": top.get("date-released"),
            "authors": people["authors"], "contact": people["contact"]}


def full_name(person: Dict[str, str]) -> str:
    return " ".join(part for part in (person.get("given-names"), person.get("family-names"))
                    if part)


def changelog_heading(tree: Path, version: str) -> Optional[Tuple[str, str]]:
    try:
        text = (tree / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        match = HEADING.match(line)
        if match and match.group("version") == version:
            return match.group("title"), match.group("date")
    return None


# ---------------------------------------------------------------------------
# Checks on the exported tree. Each returns a list of refusal reasons.
# ---------------------------------------------------------------------------


def text_files(tree: Path) -> Dict[str, str]:
    texts = {}
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            texts[path.relative_to(tree).as_posix()] = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return texts


def check_required(tree: Path, manifest: dict) -> List[str]:
    return [f"the public repository needs {entry['path']} ({entry['reason']}), and the "
            f"release commit does not carry it: land it in the development repository"
            for entry in manifest["required_public_files"]
            if not (tree / entry["path"]).is_file()]


def check_removals(removed: Sequence[str], manifest: dict, deleted_in_history: Sequence[str],
                   required: Sequence[str]) -> List[str]:
    """Every file the public main carries and the export drops must be accounted for."""
    excluded = {entry["path"] for entry in manifest["excluded"]}
    dropped = {entry["path"] for entry in manifest.get("dropped_from_public", [])}
    accounted = excluded | dropped | set(deleted_in_history) | set(required)
    return [f"the public {PUBLIC_BRANCH} carries {path}, which the export drops, and the "
            f"release commit's history never deleted it: a public-only file would be "
            f"lost (land it in the development repository, or list it under "
            f"'dropped_from_public' with the reason)"
            for path in removed if path not in accounted]


def check_references(texts: Dict[str, str], excluded: Sequence[str]) -> List[str]:
    """An exported file that names an excluded file would point at nothing."""
    needles = sorted({name for path in excluded for name in (path, Path(path).name)},
                     key=len, reverse=True)
    if not needles:
        return []
    pattern = re.compile(r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in needles)
                         + r")(?![\w-])")
    reasons = []
    for rel, text in texts.items():
        if rel == MANIFEST:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            hit = pattern.search(line)
            if hit:
                reasons.append(f"{rel}:{number} names the excluded file {hit.group(0)!r}: "
                               f"reword it in the development repository")
    return reasons


def code_digest_function(tree: Path):
    """The tree's own ``code_identity.code_digest`` (standard library only), loaded by
    path so the package itself is not imported."""
    path = tree / CODE_IDENTITY
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_export_code_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.code_digest


def check_digests(tree: Path, texts: Dict[str, str],
                  excluded_bytes: Dict[str, bytes]) -> List[str]:
    """An exported file that quotes an excluded file's digest binds it."""
    code_digest = None
    if any(path.endswith(".py") for path in excluded_bytes):
        code_digest = code_digest_function(tree)
    reasons = []
    for path, data in excluded_bytes.items():
        digests = {"sha256": hashlib.sha256(data).hexdigest()}
        if path.endswith(".py") and code_digest is not None:
            try:
                digests["code digest"] = code_digest(data.decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                pass
        for rel, text in texts.items():
            for kind, digest in digests.items():
                for length in DIGEST_PREFIXES:
                    if digest[:length] in text:
                        reasons.append(f"{rel} carries the {kind} of the excluded file "
                                       f"{path} (first {length} hex digits): a record "
                                       f"binds it")
                        break
    return reasons


def check_tree_shape(dev_entries: List[str], exported_entries: List[str],
                     excluded: Sequence[str]) -> List[str]:
    """The exported tree is REF's tree minus the excluded files, byte for byte."""
    gone = set(excluded)
    expected = [line for line in dev_entries if line.split("\t", 1)[1] not in gone]
    reasons = []
    missing = sorted(set(expected) - set(exported_entries))
    extra = sorted(set(exported_entries) - set(expected))
    for line in missing[:20]:
        reasons.append(f"ls-tree: the exported tree lacks {line}")
    for line in extra[:20]:
        reasons.append(f"ls-tree: the exported tree carries {line}, which the release "
                       f"commit does not")
    if len(missing) + len(extra) > 40:
        reasons.append(f"ls-tree: {len(missing)} entries missing and {len(extra)} extra "
                       f"in all")
    return reasons


def check_message(message: str, citation_facts: Dict[str, object], dev_repo: Path) -> List[str]:
    reasons = []
    if git("stripspace", input_text=message) != message:
        reasons.append("the message changes under git stripspace (trailing blanks, "
                       "doubled blank lines or a missing final newline)")
    lines = message.rstrip("\n").split("\n")
    if not lines[0].startswith("Summary: ") or len(lines[0]) <= len("Summary: "):
        reasons.append("the message's first line is not 'Summary: <one line>'")
    if "Description:" not in lines:
        reasons.append("the message has no 'Description:' line")
    trailers: List[Tuple[str, str]] = []
    for line in reversed(lines):
        match = TRAILER.match(line)
        if not match:
            break
        trailers.append((match.group("name"), match.group("email")))
    if not trailers:
        reasons.append("the message does not end with Co-authored-by trailers")
    elif len(lines) <= len(trailers) or lines[-len(trailers) - 1] != "":
        reasons.append("the Co-authored-by trailers are not separated from the body by "
                       "one blank line")
    authors = {full_name(person) for person in citation_facts["authors"]}
    for name, _ in trailers:
        if name not in authors:
            reasons.append(f"the trailer names {name!r}, who is not an author in "
                           f"CITATION.cff")
    credited = {person.get("email") for person in
                [*citation_facts["authors"], *citation_facts["contact"]]} - {None}
    trailer_addresses = {email for _, email in trailers}
    for address in sorted(set(ADDRESS.findall(message))):
        if address in credited or (address in trailer_addresses and NO_REPLY.match(address)):
            continue
        reasons.append(f"the message carries the address {address!r}, which is neither "
                       f"a CITATION.cff address nor a co-author's no-reply address")
    for number, line in enumerate(lines, 1):
        if LOCAL_PATH.search(line):
            reasons.append(f"the message's line {number} carries a local path")
    for token in sorted(set(HEX_TOKEN.findall(message))):
        if git_ok("cat-file", "-e", f"{token}^{{commit}}", cwd=dev_repo):
            reasons.append(f"the message names the development commit {token}: the public "
                           f"history does not carry it")
    return reasons


def run_guard(tree: Path, manifest: dict, git_env: Dict[str, str],
              names_file: Optional[Path] = None) -> Tuple[List[str], str]:
    """The exported tree's own hygiene guard, run on the exported tree.

    It runs with the exported tree as its working directory and the export's index
    as its git index, so ``git ls-files`` inside it lists exactly the exported files.
    It runs after ``write-tree``: nothing it writes can enter the commit. A names
    file reaches it through the option the manifest declares for one, which only adds
    names; the guard's own arguments come from the release commit alone."""
    guard = manifest["guard"]
    extra: List[str] = []
    if names_file is not None:
        option = guard.get("names_file_option")
        if not option:
            raise Unusable(f"{MANIFEST}: the guard declares no 'names_file_option', so "
                           f"--guard-names-file cannot reach it")
        extra = [option, str(names_file)]
    path = tree / guard["path"]
    if not path.is_file():
        return [f"the release commit carries no hygiene guard ({guard['path']}): a release "
                f"is refused unless the guard is green on the exported tree"], ""
    env = dict(git_env, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-B", str(path), *guard.get("arguments", []),
                           *extra],
                          cwd=tree, env=env, capture_output=True, text=True)
    output = (done.stdout + done.stderr).strip()
    if done.returncode != 0:
        tail = "\n".join("      " + line for line in output.splitlines()[-25:])
        return [f"the hygiene guard is red on the exported tree (exit {done.returncode}); "
                f"its report ends:\n{tail}"], output
    return [], output


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-repo", required=True, type=Path,
                        help="the development repository (read only)")
    parser.add_argument("--dev-ref", required=True,
                        help="the release commit in the development repository")
    parser.add_argument("--public-repo", required=True,
                        help="the public repository, a path or a URL (read only)")
    parser.add_argument("--message-file", required=True, type=Path,
                        help="the commit message, written from the release's CHANGELOG section")
    parser.add_argument("--scratch", required=True, type=Path,
                        help="an empty or absent directory; the clone and the tree go here")
    parser.add_argument("--version", help="the release version; default: pyproject.toml's")
    parser.add_argument("--date", help="the release date, YYYY-MM-DD; default: the date of "
                                       "the version's CHANGELOG heading")
    parser.add_argument("--guard-names-file", type=Path,
                        help="an untracked local file of names for the hygiene guard, "
                             "passed through the option the manifest declares; it only "
                             "adds names")
    parser.add_argument("--report", type=Path,
                        help="also write the verdict of every check, and the built commit, "
                             "as JSON to this file")
    return parser.parse_args(argv)


def _archive(dev_repo: Path, commit: str, tree: Path) -> None:
    archive = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=dev_repo,
                               stdout=subprocess.PIPE)
    extracted = subprocess.run(["tar", "-x", "-f", "-", "-C", str(tree)], stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() != 0 or extracted.returncode != 0:
        raise Unusable(f"git archive {commit} could not be extracted")


def _local_url(repository: str) -> str:
    """The URL a push of the public repository goes to: its ``origin`` when it is a
    local checkout that has one, otherwise the repository as given."""
    if Path(repository).is_dir():
        url = git("remote", "get-url", "origin", cwd=Path(repository), check=False).strip()
        return url or repository
    return repository


def export(arguments: argparse.Namespace) -> Tuple[int, dict]:
    dev_repo = arguments.dev_repo.resolve()
    scratch = arguments.scratch.resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise Unusable(f"the scratch directory {scratch} is not empty")
    if arguments.date is not None and not DATE.match(arguments.date):
        raise Unusable(f"--date must read YYYY-MM-DD, not {arguments.date!r}")
    try:
        message = arguments.message_file.read_text(encoding="utf-8")
    except OSError as error:
        raise Unusable(f"the message file cannot be read: {error}") from error
    if not message.strip():
        raise Unusable("the message file is empty")
    commit = git("rev-parse", "--verify", f"{arguments.dev_ref}^{{commit}}", cwd=dev_repo).strip()
    dev_tree = git("rev-parse", f"{commit}^{{tree}}", cwd=dev_repo).strip()

    tree = scratch / "tree"
    clone = scratch / "public"
    tree.mkdir(parents=True)
    git("clone", "--quiet", "--no-checkout", "--no-hardlinks", "--branch", PUBLIC_BRANCH,
        arguments.public_repo, str(clone))
    git("remote", "remove", "origin", cwd=clone)
    for key, value in (("core.autocrlf", "false"), ("gc.auto", "0")):
        git("config", key, value, cwd=clone)
    parent = git("rev-parse", "--verify", f"refs/heads/{PUBLIC_BRANCH}", cwd=clone).strip()
    say(f"cloned the public repository: {PUBLIC_BRANCH} = {parent}")

    _archive(dev_repo, commit, tree)
    say(f"archived the release commit {commit} (tree {dev_tree})")

    manifest = read_manifest(tree)
    excluded = [entry["path"] for entry in manifest["excluded"]]
    required = [entry["path"] for entry in manifest["required_public_files"]]
    excluded_bytes = {}
    for path in excluded:
        target = tree / path
        if target.is_file():
            excluded_bytes[path] = target.read_bytes()
            target.unlink()
            say(f"excluded {path}")
        else:
            say(f"excluded {path}: not at the release commit, nothing removed")

    reasons: Dict[str, List[str]] = {name: [] for name in CHECKS}
    reasons["required"] = check_required(tree, manifest)
    say(f"required public files: {len(required)} checked")

    index = clone / ".git" / "index_export"
    git_env = dict(os.environ, GIT_DIR=str(clone / ".git"), GIT_WORK_TREE=str(tree),
                   GIT_INDEX_FILE=str(index))
    # -f: every archived file is added, whatever a .gitignore in the tree says; the
    # shape check then proves the tree is the release commit's minus the exclusions.
    git("add", "-A", "-f", ".", cwd=tree, env=git_env)
    exported_tree = git("write-tree", cwd=tree, env=git_env).strip()
    exported_files = git("ls-tree", "-r", "--name-only", exported_tree, cwd=clone).splitlines()
    reasons["shape"] = check_tree_shape(git("ls-tree", "-r", commit, cwd=dev_repo).splitlines(),
                                        git("ls-tree", "-r", exported_tree,
                                            cwd=clone).splitlines(), excluded)
    say(f"exported tree {exported_tree}: {len(exported_files)} files, compared with the "
        f"release commit's ls-tree")

    removed = sorted(set(git("ls-tree", "-r", "--name-only", parent, cwd=clone).splitlines())
                     - set(exported_files))
    # --no-renames: a file renamed in the development history counts as deleted there.
    deleted = sorted(set(git("--literal-pathspecs", "log", "--no-renames", "--diff-filter=D",
                             "--name-only", "--format=", commit, "--", *removed,
                             cwd=dev_repo).splitlines()) - {""}) if removed else []
    reasons["removals"] = check_removals(removed, manifest, deleted, required)
    say(f"against public {PUBLIC_BRANCH}: the export drops {len(removed)} files, "
        f"{len([p for p in removed if p in deleted])} of them deleted in the release "
        f"commit's history")

    texts = text_files(tree)
    reasons["references"] = check_references(texts, excluded)
    reasons["digests"] = check_digests(tree, texts, excluded_bytes)
    say(f"references and digests of the {len(excluded)} excluded files: "
        f"{len(texts)} text files searched")

    facts = citation(tree)
    pyproject = (tree / "pyproject.toml").read_text(encoding="utf-8") \
        if (tree / "pyproject.toml").is_file() else ""
    package = project_field(pyproject, "name")
    tree_version = project_field(pyproject, "version")
    version = arguments.version or tree_version
    title = date = None
    found: List[str] = []
    if not version or not package:
        found.append("pyproject.toml has no [project] name and version")
    else:
        if tree_version != version:
            found.append(f"pyproject.toml's version is {tree_version!r}, not {version!r}")
        if facts["version"] != version:
            found.append(f"CITATION.cff's version is {facts['version']!r}, not {version!r}")
        heading = changelog_heading(tree, version)
        if heading is None:
            found.append(f"CHANGELOG.md has no heading '## {version} — <title> (<date>)'")
        else:
            title = heading[0]
        date = arguments.date or (heading[1] if heading else None)
        if heading and heading[1] != date:
            found.append(f"the CHANGELOG heading is dated {heading[1]}, not {date}")
        if date and facts["date"] != date:
            found.append(f"CITATION.cff's date-released is {facts['date']!r}, not {date!r}")
        if git_ok("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG_PREFIX}{version}",
                  cwd=clone):
            found.append(f"the public repository already has the tag {TAG_PREFIX}{version}")
        public_pyproject = git("show", f"{parent}:pyproject.toml", cwd=clone, check=False)
        if project_field(public_pyproject, "version") == version:
            found.append(f"the public {PUBLIC_BRANCH} already carries version {version}: "
                         f"a package index takes a version once")
    if not facts["contact"] or not facts["contact"][0].get("email"):
        found.append("CITATION.cff names no contact with an e-mail address: the release "
                     "identity is its contact")
    reasons["version"] = found
    say(f"version {version!r}, date {date!r}, title {title!r}")

    reasons["message"] = check_message(message, facts, dev_repo)
    say("message checked")

    names_file = arguments.guard_names_file.resolve() if arguments.guard_names_file else None
    if names_file is not None and not names_file.is_file():
        raise Unusable(f"the guard names file {names_file} cannot be read")
    reasons["guard"], guard_output = run_guard(tree, manifest, git_env, names_file)
    (scratch / "guard_output.txt").write_text(guard_output + "\n", encoding="utf-8")
    say(f"hygiene guard {manifest['guard']['path']}: "
        f"{'red or absent' if reasons['guard'] else 'green'}")

    report: dict = {"release_commit": commit, "release_tree": dev_tree,
                    "public_parent": parent, "exported_tree": exported_tree,
                    "exported_files": len(exported_files), "excluded": excluded,
                    "dropped_from_public_main": removed, "version": version, "date": date,
                    "checks": {name: {"passed": not reasons[name], "reasons": reasons[name]}
                               for name in CHECKS},
                    "guard_output_tail": guard_output.splitlines()[-40:]}
    failed = [name for name in CHECKS if reasons[name]]
    print("", flush=True)
    for name in CHECKS:
        print(f"  {name:<10} {'FAIL' if reasons[name] else 'pass'}"
              + (f" ({len(reasons[name])})" if reasons[name] else ""), flush=True)
    if failed:
        total = sum(len(reasons[name]) for name in failed)
        print(f"\nREFUSED: {len(failed)} of {len(CHECKS)} checks failed, {total} reason(s); "
              f"nothing was committed or tagged.", flush=True)
        for name in failed:
            for reason in reasons[name]:
                print(f"  - [{name}] {reason}", flush=True)
        report["result"] = "refused"
        return 1, report

    contact = facts["contact"][0]
    when = f"{date}T{manifest['release_time'].replace(' ', '')}"
    identity = dict(os.environ, GIT_AUTHOR_NAME=full_name(contact),
                    GIT_AUTHOR_EMAIL=contact["email"], GIT_COMMITTER_NAME=full_name(contact),
                    GIT_COMMITTER_EMAIL=contact["email"], GIT_AUTHOR_DATE=when,
                    GIT_COMMITTER_DATE=when)
    new = git("commit-tree", exported_tree, "-p", parent, "-F", "-", cwd=clone, env=identity,
              input_text=message).strip()
    git("update-ref", f"refs/heads/{PUBLIC_BRANCH}", new, parent, cwd=clone)
    tag = f"{TAG_PREFIX}{version}"
    git("tag", "-a", tag, "-m", f"{package} {version} — {title}", new, cwd=clone, env=identity)
    tag_object = git("rev-parse", f"refs/tags/{tag}", cwd=clone).strip()
    count = git("rev-list", "--count", PUBLIC_BRANCH, cwd=clone).strip()
    if git("rev-parse", f"{new}^@", cwd=clone).split() != [parent] or \
            git("rev-parse", f"{new}^{{tree}}", cwd=clone).strip() != exported_tree or \
            git("rev-parse", f"{tag}^{{commit}}", cwd=clone).strip() != new:
        raise Unusable("the built commit or tag does not read back as written")
    index.unlink()
    say(f"built {PUBLIC_BRANCH} = {new} (parent {parent}, {count} commits) and tag {tag} = "
        f"{tag_object}")
    report.update(result="built", commit=new, tag=tag, tag_object=tag_object,
                  commits_on_main=int(count))

    url = _local_url(arguments.public_repo)
    local = Path(arguments.public_repo).is_dir()
    print(f"\nBUILT, NOT PUSHED. Nothing was written to the development or the public "
          f"repository.\n"
          f"  commit {new}\n  parent {parent}\n  tree   {exported_tree}\n"
          f"  tag    {tag} ({tag_object}): {package} {version} — {title}\n"
          f"  author {full_name(contact)} <{contact['email']}>, {when}\n"
          f"The exported tree is {tree}; run the suites on it before pushing "
          f"(RELEASING.md).\n", flush=True)
    if local:
        print(f"To publish, the owner runs, in order:\n"
              f"  git -C {arguments.public_repo} pull --ff-only {clone} {PUBLIC_BRANCH}\n"
              f"  git -C {arguments.public_repo} fetch {clone} tag {tag}\n"
              f"  git -C {arguments.public_repo} push origin {PUBLIC_BRANCH}"
              f"      # starts the tests workflow; read that run\n"
              f"  git -C {arguments.public_repo} push origin {tag}"
              f"       # starts the publish workflow: uploads {version} to the package index",
              flush=True)
    else:
        print(f"To publish, the owner runs, in order:\n"
              f"  git -C {clone} push {url} {new}:refs/heads/{PUBLIC_BRANCH}"
              f"      # starts the tests workflow; read that run\n"
              f"  git -C {clone} push {url} refs/tags/{tag}"
              f"       # starts the publish workflow: uploads {version} to the package index",
              flush=True)
    print(f"A version reaches the package index once: push the tag only after the "
          f"{PUBLIC_BRANCH} run is green.", flush=True)
    return 0, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    try:
        status, report = export(arguments)
    except Unusable as problem:
        print(f"CANNOT RUN: {problem}", flush=True)
        status, report = 2, {"result": "cannot run", "reason": str(problem)}
    if arguments.report is not None:
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())

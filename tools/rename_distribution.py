#!/usr/bin/env python
"""Change the distribution and repository name everywhere, in one command.

    python tools/rename_distribution.py --name NEW-NAME                 # dry run
    python tools/rename_distribution.py --name NEW-NAME --apply
    python tools/rename_distribution.py --name NEW-NAME --repository-owner ORG --apply
    python tools/rename_distribution.py --repository-owner ORG --apply  # owner only
    python tools/rename_distribution.py --list-placeholders

WHAT IT CHANGES. The distribution name, which is also the repository name: the
``name`` of ``pyproject.toml``, the extras that refer to the distribution by name,
the titles of the README, the citation files and the manual, the install lines and
every repository URL. The current name is read from ``[project] name`` and the
current repository from ``[project.urls] Repository``; nothing is hard-coded here.

WHAT IT NEVER CHANGES. The import name. ``import meep_gpu`` stays as it is, and so
does every ``MEEP_GPU_*`` environment variable: both are part of the certification
ledgers' path keys and of every user's scripts. The tool does not open any file
under the package directory or the certification harness, whose bytes the ledgers
pin by digest, and it does not open the licence texts or the examples' recorded
output. It reports how often the old name occurs in the trees it did not open.

HOW IT MATCHES. The old name is replaced where it stands as a word, and where it
leads or ends a hyphenated name derived from it, such as the name of a conda
environment. It is not replaced inside a longer run of letters, digits or
underscores. A file or directory whose own name carries the old name is renamed
with the text that refers to it.

It is a dry run unless ``--apply`` is given. After writing it checks its own work
and exits 1 if a check fails:

    * the old name no longer occurs in any file, or file name, it may edit;
    * the number of occurrences of the import name is what it was;
    * ``pyproject.toml`` parses, and its ``name`` is the new name;
    * every extra that was declared is still declared.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

IMPORT_NAME = "meep_gpu"
PLACEHOLDER = "OWNER-PLACEHOLDER"

#: Trees whose bytes are pinned by the certification ledgers, and files that are
#: records. Never opened for writing, whatever they contain.
PROTECTED_TREES = (IMPORT_NAME, "parity")
PROTECTED_FILES = ("LICENSE",)
PROTECTED_DIRECTORIES = ("LICENSES", "examples/expected")
#: Never entered at all.
PRUNED = frozenset({".git", "__pycache__", "site", "build", "dist", "results",
                    "_scratch", "node_modules", ".pytest_cache", ".ruff_cache",
                    ".mypy_cache", ".venv", "venv"})

WORD_BEFORE = r"(?<![A-Za-z0-9_])"
WORD_AFTER = r"(?![A-Za-z0-9_])"
VALID_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
VALID_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
REPOSITORY_URL = re.compile(r"^(?P<scheme>https?)://(?P<host>[^/]+)/(?P<owner>[^/]+)/"
                            r"(?P<name>[^/]+?)/?$")


def normalized(name: str) -> str:
    """The form in which package indexes compare distribution names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def word(name: str) -> re.Pattern:
    return re.compile(WORD_BEFORE + re.escape(name) + WORD_AFTER)


def read_text(path: Path) -> str | None:
    """The file as text with its newlines untouched, or None if it is not text."""
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def is_protected(relative: Path) -> bool:
    parts = relative.parts
    if parts and parts[0] in PROTECTED_TREES:
        return True
    text = relative.as_posix()
    if text in PROTECTED_FILES:
        return True
    return any(text == directory or text.startswith(directory + "/")
               for directory in PROTECTED_DIRECTORIES)


def walk(root: Path):
    """(relative path, protected) for every regular file under ``root``, sorted."""
    stack = [root]
    found = []
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir()):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in PRUNED:
                    stack.append(entry)
            elif entry.is_file():
                relative = entry.relative_to(root)
                found.append((relative, is_protected(relative)))
    return sorted(found)


def current_names(root: Path) -> dict:
    path = root / "pyproject.toml"
    if not path.exists():
        raise SystemExit(f"no pyproject.toml under {root}: give --root")
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    url = project.get("urls", {}).get("Repository", "")
    match = REPOSITORY_URL.match(url)
    return {
        "name": project["name"],
        "extras": sorted(project.get("optional-dependencies", {})),
        "url": url,
        "host": match.group("host") if match else None,
        "owner": match.group("owner") if match else None,
    }


def rename_paths(root: Path, pattern: re.Pattern, new_name: str, apply: bool):
    """Rename every unprotected file and directory whose own name carries the name.

    Deepest first, so a renamed directory never invalidates a path still to come.
    Returns the (old, new) relative paths, as they were before any renaming.
    """
    candidates = set()
    for relative, protected in walk(root):
        if protected:
            continue
        for depth in range(1, len(relative.parts) + 1):
            candidates.add(Path(*relative.parts[:depth]))
    renamed = []
    for relative in sorted(candidates, key=lambda path: len(path.parts), reverse=True):
        if not pattern.search(relative.name):
            continue
        target = relative.with_name(pattern.sub(new_name, relative.name))
        renamed.append((relative, target))
        if apply:
            (root / relative).rename(root / target)
    return renamed


def count_import_name(root: Path, files) -> int:
    total = 0
    for relative, protected in files:
        if protected:
            continue
        text = read_text(root / relative)
        if text is not None:
            total += text.count(IMPORT_NAME)
    return total


def list_placeholders(root: Path) -> int:
    found = 0
    for relative, _protected in walk(root):
        if relative.as_posix() == "tools/rename_distribution.py":
            continue
        text = read_text(root / relative)
        if text is None or PLACEHOLDER not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if PLACEHOLDER in line:
                found += 1
                print(f"{relative.as_posix()}:{number}: {line.strip()}")
    print(f"{found} lines carry {PLACEHOLDER}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Change the distribution and repository name everywhere; the "
                    "import name is left alone.")
    parser.add_argument("--name", help="the new distribution and repository name")
    parser.add_argument("--repository-owner",
                        help="the account or organisation that holds the repository")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="the repository root (default: the parent of tools/)")
    parser.add_argument("--apply", action="store_true",
                        help="write the changes; without it nothing is written")
    parser.add_argument("--list-placeholders", action="store_true",
                        help="list every line that still carries an owner placeholder")
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    if arguments.list_placeholders:
        return list_placeholders(root)
    if not arguments.name and not arguments.repository_owner:
        parser.error("give --name, --repository-owner, or --list-placeholders")

    now = current_names(root)
    old_name = now["name"]
    new_name = arguments.name or old_name
    new_owner = arguments.repository_owner or now["owner"]

    if not VALID_NAME.match(new_name):
        raise SystemExit(f"{new_name!r} is not a valid distribution name")
    if IMPORT_NAME in new_name:
        raise SystemExit(
            f"{new_name!r} contains the import name {IMPORT_NAME!r}. The two must stay "
            "distinguishable in text, or the next rename could not tell them apart. "
            "Package indexes treat '-', '_' and '.' alike, so a hyphen loses nothing.")
    if arguments.repository_owner and not VALID_OWNER.match(arguments.repository_owner):
        raise SystemExit(f"{arguments.repository_owner!r} is not a valid owner name")
    if arguments.repository_owner and now["owner"] is None:
        raise SystemExit("pyproject.toml carries no [project.urls] Repository of the "
                         "form https://HOST/OWNER/NAME to change the owner of")

    replacements = []
    if now["owner"] is not None and new_owner != now["owner"]:
        # The owner first, in the position a repository URL gives it, so that a
        # later name change still finds the name as a word.
        replacements.append((
            re.compile(re.escape(f"{now['host']}/{now['owner']}/")
                       + r"(?=" + re.escape(old_name) + WORD_AFTER + r")"),
            f"{now['host']}/{new_owner}/"))
    if new_name != old_name:
        replacements.append((word(old_name), new_name))
    if not replacements:
        print("nothing to change: the name and the owner are already as requested")
        return 0

    files = walk(root)
    import_name_before = count_import_name(root, files)
    changed, untouched_protected = [], 0
    for relative, protected in files:
        text = read_text(root / relative)
        if text is None:
            continue
        if protected:
            untouched_protected += len(word(old_name).findall(text))
            continue
        new_text, hits = text, 0
        for pattern, replacement in replacements:
            new_text, count = pattern.subn(replacement, new_text)
            hits += count
        if hits:
            changed.append((relative, hits))
            if arguments.apply:
                write_text(root / relative, new_text)

    renamed = []
    if new_name != old_name:
        renamed = rename_paths(root, word(old_name), new_name, arguments.apply)

    verb = "changed" if arguments.apply else "would change"
    print(f"name:  {old_name} -> {new_name}")
    print(f"owner: {now['owner']} -> {new_owner}")
    for relative, hits in changed:
        print(f"  {verb} {hits:3d} in {relative.as_posix()}")
    for relative, target in renamed:
        print(f"  {'renamed' if arguments.apply else 'would rename'} "
              f"{relative.as_posix()} -> {target.as_posix()}")
    print(f"{verb} {sum(hits for _, hits in changed)} occurrences in "
          f"{len(changed)} files")
    print(f"left untouched: {untouched_protected} occurrences of {old_name!r} in the "
          f"protected trees ({', '.join(PROTECTED_TREES)}), whose bytes are pinned")
    if not arguments.apply:
        print("dry run: nothing was written. Add --apply to write.")
        return 0

    problems = []
    files = walk(root)
    if new_name != old_name:
        for relative, protected in files:
            text = None if protected else read_text(root / relative)
            if text is not None and word(old_name).search(text):
                problems.append(f"{old_name!r} still occurs in {relative.as_posix()}")
            if not protected and word(old_name).search(relative.as_posix()):
                problems.append(f"{old_name!r} still occurs in the path "
                                f"{relative.as_posix()}")
    import_name_after = count_import_name(root, files)
    if import_name_after != import_name_before:
        problems.append(f"the import name occurred {import_name_before} times before "
                        f"and {import_name_after} times after")
    after = current_names(root)
    if after["name"] != new_name:
        problems.append(f"pyproject.toml names {after['name']!r}, not {new_name!r}")
    if after["extras"] != now["extras"]:
        problems.append(f"the declared extras changed: {now['extras']} -> "
                        f"{after['extras']}")
    if new_owner != now["owner"] and after["owner"] != new_owner:
        problems.append(f"the repository owner reads {after['owner']!r}, not "
                        f"{new_owner!r}")
    for problem in problems:
        print(f"CHECK FAILED: {problem}")
    if problems:
        return 1
    print(f"checked: 0 occurrences of the old name left, the import name occurs "
          f"{import_name_after} times as before, pyproject.toml parses and names "
          f"{new_name!r}, {len(after['extras'])} of {len(now['extras'])} extras kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())

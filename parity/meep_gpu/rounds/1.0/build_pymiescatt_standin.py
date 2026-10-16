#!/usr/bin/env python3
"""Build the PyMieScatt stand-in from its pinned sdist and install it into an environment.

    python build_pymiescatt_standin.py --python <prefix>/bin/python --install
    python build_pymiescatt_standin.py --python <prefix>/bin/python --sdist PyMieScatt-1.8.1.1.tar.gz

Two rows of the MEEP identity corpus (``ex.mie_scattering`` and
``ex.differential_cross_section``) import PyMieScatt. PyMieScatt 1.8.1.1, its last
release, imports ``trapz`` from ``scipy.integrate``, which SciPy 1.14 removed, so it does
not import in either certification reference. The stand-in is that release with the two
lines of ``pymiescatt_standin.patch`` changed, so ``trapz`` comes from NumPy. Its source
is not kept in this repository: everything that defines it is pinned in
``environments/locks/numerics-relevant.json`` (``corpus_only.standin``): the sdist's URL
and SHA-256, the patch, and the SHA-256 of each of the five package files the patched
sdist must hold.

Steps, each a flushed line on stdout, each refusing (exit 1) on any mismatch:

1. fetch the sdist from PyPI (or read ``--sdist``) and check its SHA-256;
2. unpack it into a new working directory (members confined to one top directory, no
   links);
3. apply the patch: each removed line must be at its stated line number, with no context
   and no fuzz, and each line written takes the line ending of the line it replaces (the
   sdist's sources end their lines in CRLF);
4. check the five package files against their pinned SHA-256, and that the package holds
   no other file;
5. build a wheel with the environment's own pip and setuptools (``pip wheel --no-deps
   --no-build-isolation --no-index``), so nothing is fetched and no unpinned build tool
   is used; the wheel's metadata reflects that setuptools, so its own SHA-256 is printed
   but not pinned, and the package files inside it are checked instead;
6. with ``--install``, install it (``pip install --no-deps --no-index``) and check the
   installed files, read back through the environment's own interpreter (``-s -I``).

Standard library only. It installs nothing without ``--install``, and never anything but
the wheel it built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
LOCK_DIR = os.path.join(REPO_ROOT, "environments", "locks")
NUMERICS_FILE = "numerics-relevant.json"
#: Zip timestamps of the wheel (the sdist's own date), so a rebuild with the same
#: setuptools gives the same bytes.
SOURCE_DATE_EPOCH = "1609286400"


class Refused(Exception):
    """A step found something other than what the lock pins."""


def say(text: str) -> None:
    print(f"{time.strftime('%F %T')} stand-in: {text}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(lock_dir: str = LOCK_DIR) -> dict:
    """``corpus_only.standin`` of the lock directory's numerics file."""
    with open(os.path.join(lock_dir, NUMERICS_FILE), encoding="utf-8") as handle:
        numerics = json.load(handle)
    spec = (numerics.get("corpus_only") or {}).get("standin")
    if not isinstance(spec, dict):
        raise Refused(f"{NUMERICS_FILE} has no corpus_only.standin entry")
    for key in ("name", "version", "sdist", "patch", "files"):
        if key not in spec:
            raise Refused(f"corpus_only.standin has no {key!r}")
    return spec


# --------------------------------------------------------------------------
# The patch
# --------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(text: str) -> dict:
    """``{relative path: [(old start, [removed lines], [added lines])]}`` of a unified diff.

    Prose before the first ``---`` line is ignored. A hunk may carry no context line:
    every line of it is a removal or an addition, and their counts match its header.
    """
    files: dict = {}
    current = None
    hunk = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise Refused(f"patch line {index + 1}: '---' without '+++'")
            old = line[4:].split("\t")[0].strip()
            new = lines[index + 1][4:].split("\t")[0].strip()
            if not (old.startswith("a/") and new.startswith("b/") and old[2:] == new[2:]):
                raise Refused(f"patch line {index + 1}: expected a/<path> and b/<path>, got {old} {new}")
            current = files.setdefault(old[2:], [])
            hunk = None
            index += 2
            continue
        match = _HUNK.match(line)
        if match:
            if current is None:
                raise Refused(f"patch line {index + 1}: a hunk before any file header")
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            hunk = (old_start, [], [], old_count, new_count)
            current.append(hunk)
            index += 1
            continue
        if hunk is not None and line[:1] in ("-", "+"):
            (hunk[1] if line[0] == "-" else hunk[2]).append(line[1:])
        elif hunk is not None and line[:1] == " ":
            raise Refused(f"patch line {index + 1}: a context line; the stand-in's patch carries none")
        elif current is not None and line.strip():
            raise Refused(f"patch line {index + 1}: not part of a hunk: {line[:80]!r}")
        index += 1
    result = {}
    for path, hunks in files.items():
        result[path] = []
        for old_start, removed, added, old_count, new_count in hunks:
            if len(removed) != old_count or len(added) != new_count:
                raise Refused(f"{path}: a hunk at line {old_start} has {len(removed)} removed and "
                              f"{len(added)} added lines, its header says {old_count} and {new_count}")
            result[path].append((old_start, removed, added))
    if not result:
        raise Refused("the patch changes no file")
    return result


def apply_hunks(data: bytes, hunks, path: str = "<file>") -> bytes:
    """``data`` with ``hunks`` applied: each removed line exactly at its stated line number.

    Lines are compared without their endings; each added line takes the ending of the
    line it replaces (or, for a pure insertion, of the line before it).
    """
    lines = data.decode("utf-8").splitlines(keepends=True)
    offset = 0
    previous_end = 0
    for old_start, removed, added in sorted(hunks, key=lambda hunk: hunk[0]):
        if old_start < previous_end:
            raise Refused(f"{path}: hunks overlap at line {old_start}")
        at = old_start - 1 + offset if removed else old_start + offset
        stretch = lines[at:at + len(removed)]
        if [line.rstrip("\r\n") for line in stretch] != removed:
            found = [line.rstrip("\r\n") for line in stretch]
            raise Refused(f"{path}: line {old_start} is {found!r}, the patch removes {removed!r}")
        template = stretch[0] if stretch else (lines[at - 1] if at > 0 else "\n")
        ending = template[len(template.rstrip("\r\n")):] or "\n"
        lines[at:at + len(removed)] = [line + ending for line in added]
        offset += len(added) - len(removed)
        previous_end = old_start + len(removed)
    return "".join(lines).encode("utf-8")


# --------------------------------------------------------------------------
# The sdist
# --------------------------------------------------------------------------

def fetch_sdist(spec: dict, work: str, sdist: str | None) -> str:
    expected = spec["sdist"]["sha256"]
    if sdist is None:
        url = spec["sdist"]["url"]
        sdist = os.path.join(work, os.path.basename(url))
        say(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as response, open(sdist, "wb") as handle:
            shutil.copyfileobj(response, handle)
    actual = sha256_file(sdist)
    say(f"sdist {os.path.basename(sdist)} sha256 {actual}")
    if actual != expected:
        raise Refused(f"the sdist's SHA-256 is {actual}, the lock pins {expected}")
    return sdist


def unpack(sdist: str, destination: str) -> str:
    """Unpack ``sdist`` under ``destination``; returns its single top directory."""
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        tops = set()
        for member in members:
            name = member.name
            if name.startswith("/") or ".." in name.split("/"):
                raise Refused(f"the sdist holds a member outside its directory: {name}")
            if not (member.isfile() or member.isdir()):
                raise Refused(f"the sdist holds a member that is not a file or directory: {name}")
            tops.add(name.split("/")[0])
        if len(tops) != 1:
            raise Refused(f"the sdist has {len(tops)} top directories: {sorted(tops)}")
        if hasattr(tarfile, "data_filter"):
            archive.extractall(destination, filter="data")
        else:  # the checks above are what the data filter enforces for these members
            archive.extractall(destination)
    return os.path.join(destination, tops.pop())


def check_package_files(root: str, spec: dict, where: str) -> None:
    """The package directory under ``root`` holds exactly the pinned files, byte for byte."""
    package = spec["name"]
    present = sorted(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
                     for dirpath, _dirs, names in os.walk(os.path.join(root, package))
                     for name in names)
    expected = sorted(spec["files"])
    if present != expected:
        raise Refused(f"{where}: the package holds {present}, the lock pins {expected}")
    for relative in expected:
        actual = sha256_file(os.path.join(root, relative))
        if actual != spec["files"][relative]:
            raise Refused(f"{where}: {relative} has SHA-256 {actual}, the lock pins "
                          f"{spec['files'][relative]}")
    say(f"{where}: {len(expected)} of {len(expected)} package files are the pinned bytes")


# --------------------------------------------------------------------------
# Build, check and install
# --------------------------------------------------------------------------

def child_environment(tmp: str | None = None) -> dict:
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
               PIP_NO_INPUT="1", PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH)
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
                 "PIP_FIND_LINKS", "PIP_CONSTRAINT", "PIP_REQUIRE_VIRTUALENV"):
        env.pop(name, None)
    if tmp:
        env["TMPDIR"] = tmp
    return env


def run(argv, what: str, tmp: str | None = None) -> None:
    say(f"{what}: {' '.join(argv)}")
    done = subprocess.run(argv, env=child_environment(tmp), capture_output=True, text=True,
                          check=False)
    for line in (done.stdout + done.stderr).splitlines():
        print(f"    {line}", flush=True)
    if done.returncode != 0:
        raise Refused(f"{what} exited {done.returncode}")


def wheel_stem(name: str) -> str:
    """A distribution name as wheel file names spell it (normalized, lower case)."""
    return re.sub(r"[-_.]+", "_", name).lower()


def build_wheel(python: str, source: str, out: str, spec: dict, tmp: str | None = None) -> str:
    os.makedirs(out)
    run([python, "-s", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
         "--no-cache-dir", "--wheel-dir", out, source], "build", tmp)
    wheels = [name for name in os.listdir(out) if name.endswith(".whl")]
    stem = f"{wheel_stem(spec['name'])}-{spec['version']}"
    # setuptools spells the file name in the case of the project name or normalized,
    # depending on its version; both name the same distribution.
    if len(wheels) != 1 or wheels[0].lower() != f"{stem}-py3-none-any.whl":
        raise Refused(f"the build made {wheels}, expected one {stem}-py3-none-any.whl")
    path = os.path.join(out, wheels[0])
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        infos = {name.split("/")[0] for name in names if name.split("/")[0].endswith(".dist-info")}
        if len(infos) != 1 or next(iter(infos)).lower() != f"{stem}.dist-info":
            raise Refused(f"the wheel's metadata directories are {sorted(infos)}, expected one "
                          f"{stem}.dist-info")
        dist_info = next(iter(infos)) + "/"
        package = sorted(name for name in names if not name.startswith(dist_info))
        if package != sorted(spec["files"]):
            raise Refused(f"the wheel holds {package} outside {dist_info}, the lock pins "
                          f"{sorted(spec['files'])}")
        for relative, pinned in spec["files"].items():
            actual = sha256_bytes(wheel.read(relative))
            if actual != pinned:
                raise Refused(f"the wheel's {relative} has SHA-256 {actual}, the lock pins {pinned}")
    say(f"wheel {wheels[0]} sha256 {sha256_file(path)} (not pinned: its metadata names the "
        f"setuptools that built it); {len(spec['files'])} of {len(spec['files'])} package "
        "files are the pinned bytes")
    return path


_READ_BACK = r"""
import hashlib, json, sys
from importlib import metadata
name, files = sys.argv[1], json.loads(sys.argv[2])
try:
    dist = metadata.distribution(name)
except metadata.PackageNotFoundError:
    print(json.dumps(None)); raise SystemExit(0)
out = {"version": dist.version, "installer": (dist.read_text("INSTALLER") or "").strip(),
       "files": {}}
for relative in files:
    try:
        with open(dist.locate_file(relative), "rb") as handle:
            out["files"][relative] = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        out["files"][relative] = None
print(json.dumps(out))
"""


def read_installed(python: str, spec: dict):
    done = subprocess.run([python, "-s", "-I", "-c", _READ_BACK, spec["name"],
                           json.dumps(sorted(spec["files"]))],
                          env=child_environment(), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise Refused(f"reading the installed stand-in back failed: {done.stderr[-500:]}")
    return json.loads(done.stdout)


def check_installed(python: str, spec: dict) -> None:
    installed = read_installed(python, spec)
    if installed is None:
        raise Refused(f"{python} does not see {spec['name']} after the install")
    if installed["version"] != spec["version"]:
        raise Refused(f"installed {spec['name']} is {installed['version']}, not {spec['version']}")
    differing = [relative for relative, pinned in spec["files"].items()
                 if installed["files"].get(relative) != pinned]
    if differing:
        raise Refused(f"installed {spec['name']} {spec['version']} is not the stand-in: "
                      f"{', '.join(differing)} differ from the pinned bytes (uninstall it first)")
    say(f"installed: {spec['name']} {spec['version']} (installer {installed['installer'] or 'unknown'}), "
        f"{len(spec['files'])} of {len(spec['files'])} package files are the pinned bytes")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True,
                        help="the interpreter of the environment that builds (and receives) it")
    parser.add_argument("--sdist", help="read this copy of the sdist instead of fetching it")
    parser.add_argument("--install", action="store_true",
                        help="install the wheel into --python's environment and check it")
    parser.add_argument("--work", help="working directory (must not exist); default a new "
                        "temporary directory, removed at the end")
    parser.add_argument("--wheel-dir", help="keep the built wheel in this directory")
    parser.add_argument("--lock-dir", default=LOCK_DIR, help="read the pins from this directory")
    arguments = parser.parse_args(argv)
    if arguments.work:
        os.makedirs(arguments.work)
        work, owned = arguments.work, False
    else:
        work, owned = tempfile.mkdtemp(prefix="pymiescatt-standin-"), True
    try:
        spec = load_spec(arguments.lock_dir)
        say(f"{spec['name']} {spec['version']} for {arguments.python}")
        sdist = fetch_sdist(spec, work, arguments.sdist)
        source = unpack(sdist, os.path.join(work, "src"))
        patch_path = os.path.join(REPO_ROOT, spec["patch"])
        with open(patch_path, encoding="utf-8") as handle:
            patch = parse_patch(handle.read())
        for relative, hunks in sorted(patch.items()):
            target = os.path.join(source, relative)
            with open(target, "rb") as handle:
                data = handle.read()
            with open(target, "wb") as handle:
                handle.write(apply_hunks(data, hunks, relative))
            say(f"patched {relative} ({len(hunks)} hunk{'s' if len(hunks) != 1 else ''})")
        check_package_files(source, spec, "patched sdist")
        tmp = os.path.join(work, "tmp")
        os.makedirs(tmp)
        wheel = build_wheel(arguments.python, source, os.path.join(work, "dist"), spec, tmp)
        if arguments.wheel_dir:
            os.makedirs(arguments.wheel_dir, exist_ok=True)
            shutil.copy2(wheel, arguments.wheel_dir)
            say(f"kept {os.path.join(arguments.wheel_dir, os.path.basename(wheel))}")
        if arguments.install:
            run([arguments.python, "-s", "-m", "pip", "install", "--no-deps", "--no-index",
                 "--no-cache-dir", wheel], "install", tmp)
            check_installed(arguments.python, spec)
        say("done")
        return 0
    except Refused as refusal:
        say(f"REFUSING: {refusal}")
        return 1
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        say(f"REFUSING: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

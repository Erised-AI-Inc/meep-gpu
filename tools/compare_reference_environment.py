#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare this Python environment with the certification reference of its platform.

    python tools/compare_reference_environment.py [--require-reference]

The certification reference of each platform is a pair of lock files under
``environments/locks/``: an explicit conda lock (every package by URL and MD5) and
a pip requirements file (every PyPI package by version and wheel hash). The
build scripts ``parity/meep_gpu/build_meep_133_macos.sh`` and
``build_meep_133_linux.sh`` install exactly those files.
``environments/locks/numerics-relevant.json`` names, per platform, the packages
whose builds can change what MEEP or the GPU route computes. For each of them
this prints whether the environment holds the reference's build (same),
another build (different, with both), none (missing), a difference the
reference documents (differs by design), or something it cannot read (not
read), then one verdict line:

    matches the certification reference ...
    differs from the certification reference in N of D numerics-relevant packages: ...
    cannot be confirmed to match the certification reference: K of D ... could not be read

Conda packages are read from the environment's ``conda-meta`` directory and
judged by version and build; PyPI packages are read from the installed
distributions and judged by version, by origin (a conda build is not the PyPI
wheel) and, where ``numerics-relevant.json`` names one, by the wheel's build
tag. Outside a conda environment (a pip-only install) the compiled libraries
MEEP links cannot be read and are reported as not read, while its NumPy, SciPy
and mpi4py are PyPI builds and are reported as different from the reference's
conda builds. The host's macOS or glibc version is checked against the lock's
floor, which an explicit conda install never checks.

This report judges packages only. MEEP itself, built from source on top of the
lock, is not a package of it: ``tools/check_install.py --require-reference``
imports it and also requires the version, precision and MPI build that
``numerics-relevant.json`` names (``judge_meep`` below).

The report is informational and the exit status is 0, unless
``--require-reference`` is given: then it is 0 only when the verdict is
"matches", and 1 otherwise. It never raises: a failure to read anything becomes
a line of the report. ``tools/check_install.py`` prints the same report as its
section 7.

Options:

    --require-reference  exit 1 unless the environment matches the reference
    --prefix PATH        compare the conda environment at PATH instead of the
                         running interpreter's
    --platform NAME      compare against this platform's lock (osx-arm64,
                         linux-64) instead of this host's
    --lock-dir PATH      read the locks from PATH instead of environments/locks
    --json               print the comparison as JSON

Standard library only; this file is not part of the installed package.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import sys

LOCK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "environments", "locks")
NUMERICS_FILE = "numerics-relevant.json"

#: The distribution names under which a conda-judged Python package can be
#: installed by pip. A conda package absent from this table is a compiled
#: library (or conda metadata) that only ``conda-meta`` can describe.
DISTRIBUTION_NAMES = {
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "mpi4py": ("mpi4py",),
    "h5py": ("h5py",),
    "cupy": ("cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"),
}

SAME = "same"
DIFFERENT = "different"
MISSING = "missing"
BY_DESIGN = "by design"
UNREAD = "not read"

#: Verdict kinds.
MATCHES = "matches"
DIFFERS = "differs"
UNCONFIRMED = "unconfirmed"
NOT_COMPARED = "not compared"

_EXPLICIT_LINE = re.compile(r"^(https://\S+/([^/#\s]+))#([0-9a-f]{32})$")
_PIP_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)(.*)$")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


# --------------------------------------------------------------------------
# Lock files
# --------------------------------------------------------------------------

def normalize(name: str) -> str:
    """A distribution name as PyPI compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def split_conda_filename(filename: str):
    """``(name, version, build)`` of a conda package file name."""
    for suffix in (".conda", ".tar.bz2"):
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    else:
        raise ValueError(f"not a conda package file name: {filename}")
    parts = filename.rsplit("-", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"not a conda package file name: {filename}")
    return parts[0], parts[1], parts[2]


def parse_conda_lock(text: str) -> dict:
    """``{name: {name, version, build, md5, url, fn}}`` from an explicit conda lock.

    Accepts comments, blank lines and the ``@EXPLICIT`` marker; refuses any other
    line, a line without an MD5, a file without the marker, and a name listed twice.
    """
    packages = {}
    explicit = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "@EXPLICIT":
            explicit = True
            continue
        match = _EXPLICIT_LINE.match(line)
        if match is None:
            raise ValueError(f"line {number} is neither a comment, @EXPLICIT nor an "
                             f"https URL with an MD5: {line[:120]}")
        url, filename, md5 = match.groups()
        name, version, build = split_conda_filename(filename)
        if name in packages:
            raise ValueError(f"line {number}: {name} is listed twice")
        packages[name] = {"name": name, "version": version, "build": build, "md5": md5,
                          "url": url, "fn": filename}
    if not explicit:
        raise ValueError("no @EXPLICIT line: not an explicit conda lock")
    return packages


def parse_pip_lock(text: str) -> dict:
    """``{normalized name: {name, version, sha256}}`` from a hashed requirements file."""
    packages = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = _PIP_LINE.match(line)
        if match is None:
            raise ValueError(f"line {number} is not name==version: {line[:120]}")
        name, version, rest = match.groups()
        packages[normalize(name)] = {"name": name, "version": version,
                                     "sha256": _HASH.findall(rest)}
    return packages


def judged_names(spec: dict):
    """``[(name, group)]`` of the conda-judged packages, each once, in file order."""
    seen = {}
    for group, names in (spec.get("conda") or {}).items():
        for name in names:
            seen.setdefault(name, group)
    return list(seen.items())


def load_reference(platform_tag: str, lock_dir: str = LOCK_DIR) -> dict:
    """The lock files and numerics set of one platform. Raises when they cannot be read."""
    with open(os.path.join(lock_dir, NUMERICS_FILE), encoding="utf-8") as handle:
        numerics = json.load(handle)
    platforms = numerics.get("platforms") or {}
    if platform_tag not in platforms:
        raise LookupError(f"no certification reference lock for {platform_tag}; there "
                          f"are locks for {', '.join(sorted(platforms)) or 'no platform'}")
    spec = platforms[platform_tag]
    conda_path = os.path.join(lock_dir, spec["conda_lock"])
    pip_path = os.path.join(lock_dir, spec["pip_lock"])
    with open(conda_path, encoding="utf-8") as handle:
        conda = parse_conda_lock(handle.read())
    with open(pip_path, encoding="utf-8") as handle:
        pip = parse_pip_lock(handle.read())
    return {"platform": platform_tag, "spec": spec, "conda": conda, "pip": pip,
            "conda_lock": spec["conda_lock"], "pip_lock": spec["pip_lock"]}


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------

def host_platform(system: str | None = None, machine: str | None = None):
    """The conda platform name of this host, or None when no lock can exist for it."""
    system = system or platform.system()
    machine = machine or platform.machine()
    return {("Darwin", "arm64"): "osx-arm64",
            ("Linux", "x86_64"): "linux-64"}.get((system, machine))


def read_conda_meta(prefix: str):
    """``{name: record}`` from ``<prefix>/conda-meta``, or None when it has none."""
    directory = os.path.join(prefix, "conda-meta")
    if not os.path.isdir(directory):
        return None
    records = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            name = record["name"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        records[name] = {key: record.get(key) for key in
                         ("name", "version", "build", "build_number", "md5", "url")}
    return records


def wheel_build_tag(text):
    """The ``Build:`` field of a distribution's ``WHEEL`` file.

    ``""`` when the file has no such field (a wheel without a build tag), None when
    there is no file to read.
    """
    if text is None:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "build":
            return value.strip()
    return ""


def read_distributions(paths=None) -> dict:
    """``{normalized name: {name, version, installer, build_tag}}`` of the installed
    distributions.

    ``paths`` is a list of directories; None means this interpreter's ``sys.path``.
    The first distribution of a name wins, as the first one on the path is imported.
    """
    from importlib import metadata

    found = {}
    try:
        distributions = list(metadata.distributions(path=paths) if paths is not None
                             else metadata.distributions())
    except Exception:  # noqa: BLE001 - an unreadable path is reported as nothing found
        return found
    for distribution in distributions:
        try:
            name = distribution.metadata["Name"]
            if not name or normalize(name) in found:
                continue
            installer = (distribution.read_text("INSTALLER") or "").strip() or None
            found[normalize(name)] = {"name": name, "version": distribution.version,
                                      "installer": installer,
                                      "build_tag": wheel_build_tag(distribution.read_text("WHEEL"))}
        except Exception:  # noqa: BLE001 - one broken distribution is skipped
            continue
    return found


def read_host() -> dict:
    """The host facts a lock's floor is judged against: macOS and glibc versions."""
    host = {"macos": None, "glibc": None}
    try:
        if platform.system() == "Darwin":
            host["macos"] = platform.mac_ver()[0] or None
        elif platform.system() == "Linux":
            text = os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else ""
            if text and text.startswith("glibc "):
                host["glibc"] = text.split()[1]
    except (OSError, ValueError, IndexError):
        pass
    return host


def read_user_site():
    """The user site directory when it is on the import path and holds distributions."""
    try:
        import site

        if not site.ENABLE_USER_SITE:
            return None
        directory = site.getusersitepackages()
        if not directory or not os.path.isdir(directory):
            return None
        held = sorted(entry for entry in os.listdir(directory)
                      if entry.endswith((".dist-info", ".egg-info")))
        return {"path": directory, "distributions": len(held)} if held else None
    except Exception:  # noqa: BLE001
        return None


#: What a build script writes into a prefix it solved (MEEP_REFERENCE_SOLVE=1)
#: instead of installing the lock.
SOLVED_MARKER = os.path.join("share", "meep-gpu-build", "NOT_THE_CERTIFICATION_REFERENCE.txt")


def read_solved_marker(prefix):
    """The path of the solved-build marker in ``prefix``, or None when it has none."""
    try:
        path = os.path.join(prefix, SOLVED_MARKER) if prefix else None
        return path if path and os.path.isfile(path) else None
    except (OSError, TypeError, ValueError):
        return None


def read_environment(prefix: str | None = None) -> dict:
    """What the comparison reads: conda records, distributions, Python, host.

    With ``prefix``, the conda environment at that path and the distributions in
    its ``lib/python*/site-packages``. Without, the running interpreter: the first
    of ``sys.prefix`` and ``sys.base_prefix`` that has ``conda-meta`` (a virtual
    environment on top of a conda one reads the conda one), and the distributions
    on ``sys.path``.
    """
    environment = {"host": read_host(), "user_site": None}
    if prefix is not None:
        environment["where"] = os.path.abspath(prefix)
        environment["conda"] = read_conda_meta(prefix)
        environment["conda_prefix"] = os.path.abspath(prefix) if environment["conda"] is not None else None
        sites = sorted(glob.glob(os.path.join(prefix, "lib", "python*", "site-packages")))
        environment["distributions"] = read_distributions(sites)
        environment["python"] = None
        environment["solved_marker"] = read_solved_marker(environment["conda_prefix"])
        return environment
    environment["where"] = sys.prefix
    environment["conda"] = None
    environment["conda_prefix"] = None
    for candidate in dict.fromkeys((sys.prefix, sys.base_prefix)):
        records = read_conda_meta(candidate)
        if records is not None:
            environment["conda"], environment["conda_prefix"] = records, candidate
            break
    environment["distributions"] = read_distributions()
    environment["python"] = platform.python_version()
    environment["user_site"] = read_user_site()
    environment["solved_marker"] = read_solved_marker(environment["conda_prefix"])
    return environment


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

def _version_key(text: str):
    return tuple(int(part) if part.isdigit() else 0 for part in re.split(r"[.]", text))


def _conda_text(entry) -> str:
    return f"{entry.get('version')}={entry.get('build')}"


def _by_design(rule: dict, record: dict) -> bool:
    """True when ``record`` is the one build the rule documents.

    ``certified_build``: the certified environment's own build of the package, by
    version and build, and by MD5 when both the rule and the record carry one (a
    ``conda-meta`` record without an MD5 is judged by version and build, as a
    package the lock holds is).
    """
    if (rule or {}).get("rule") != "certified_build":
        return False
    if record.get("version") != rule.get("version") or record.get("build") != rule.get("build"):
        return False
    return not (rule.get("md5") and record.get("md5") and record["md5"] != rule["md5"])


def _distribution(environment: dict, names):
    for name in names:
        found = (environment.get("distributions") or {}).get(normalize(name))
        if found:
            return found
    return None


def judge_conda(name: str, locked, environment: dict, spec: dict) -> dict:
    """One row for a conda-judged package."""
    row = {"name": name, "kind": "conda",
           "reference": _conda_text(locked) if locked else "not in the lock"}
    if locked is None:
        row.update(status=UNREAD, found="-",
                   note="the numerics set names a package the lock does not hold")
        return row
    conda = environment.get("conda")
    record = (conda or {}).get(name)
    distribution = _distribution(environment, DISTRIBUTION_NAMES.get(name, ()))
    # A positive signal only: pip and uv always write INSTALLER, so a distribution whose
    # INSTALLER is absent or unreadable is not taken for a pip install over conda's.
    pip_installed = (distribution is not None and distribution.get("installer") is not None
                     and distribution.get("installer") != "conda")
    if record is not None:
        row["found"] = _conda_text(record)
        if pip_installed:
            row.update(status=DIFFERENT,
                       found=f"{distribution['name']} {distribution['version']} "
                             f"(installed by {distribution.get('installer') or 'an unknown installer'} "
                             f"over the conda build {_conda_text(record)})")
            return row
        if record.get("version") == locked["version"] and record.get("build") == locked["build"]:
            if record.get("md5") and record["md5"] != locked["md5"]:
                row.update(status=DIFFERENT,
                           note=f"same version and build, another file (MD5 "
                                f"{record['md5']} against {locked['md5']})")
            else:
                row["status"] = SAME
            return row
        rule = (spec.get("by_design") or {}).get(name)
        if _by_design(rule, record):
            row.update(status=BY_DESIGN, note=rule.get("why", ""))
        else:
            row["status"] = DIFFERENT
        return row
    if distribution is not None:
        row["found"] = (f"{distribution['name']} {distribution['version']} (installed by "
                        f"{distribution.get('installer') or 'an unknown installer'})")
        if distribution.get("installer") == "conda" and conda is None:
            if distribution["version"] == locked["version"]:
                row.update(status=UNREAD, note="same version; the build cannot be read "
                           "without conda-meta")
            else:
                row["status"] = DIFFERENT
        else:
            row.update(status=DIFFERENT,
                       note="not the conda build the reference holds")
        return row
    if conda is not None:
        row.update(status=MISSING, found="absent")
        return row
    if name == "python" and environment.get("python"):
        row["found"] = environment["python"]
        if environment["python"] == locked["version"]:
            row.update(status=UNREAD, note="same version; the build cannot be read outside "
                       "a conda environment")
        else:
            row["status"] = DIFFERENT
        return row
    if DISTRIBUTION_NAMES.get(name):
        row.update(status=MISSING, found="absent")
        return row
    row.update(status=UNREAD, found="not read",
               note="not a conda environment; this library's build cannot be read")
    return row


def _tag_text(tag) -> str:
    return f", build {tag}" if tag else ""


def judge_pypi(name: str, locked, environment: dict, spec: dict | None = None) -> dict:
    """One row for a PyPI-judged package.

    By version and origin, and by the wheel's build tag when the spec's
    ``pypi_build_tags`` names one for this package (PyPI can carry two wheels of
    one version for one platform, told apart only by that tag).
    """
    tag = ((spec or {}).get("pypi_build_tags") or {}).get(name)
    row = {"name": name, "kind": "pypi",
           "reference": (f"{locked['version']}{_tag_text(tag)} (PyPI wheel)" if locked
                         else "not in the lock")}
    if locked is None:
        row.update(status=UNREAD, found="-",
                   note="the numerics set names a package the lock does not hold")
        return row
    distribution = _distribution(environment, (name,))
    if distribution is None:
        row.update(status=MISSING, found="absent")
        return row
    installer = distribution.get("installer")
    found_tag = distribution.get("build_tag")
    row["found"] = (f"{distribution['version']}{_tag_text(found_tag if tag else None)} "
                    f"(installed by {installer or 'an unknown installer'})")
    if installer == "conda":
        row.update(status=DIFFERENT, note="a conda build; the reference holds the PyPI wheel")
    elif distribution["version"] != locked["version"]:
        row["status"] = DIFFERENT
    elif tag is None:
        row["status"] = SAME
    elif found_tag is None:
        row.update(status=UNREAD, note="the wheel's build tag could not be read (no WHEEL file)")
    elif found_tag == tag:
        row["status"] = SAME
    else:
        row.update(status=DIFFERENT,
                   note=f"another wheel of the same version: build tag "
                        f"{found_tag or 'none'}, where the reference's is {tag}")
    return row


def _meep_text(facts: dict) -> str:
    precision = {True: "single precision", False: "double precision"}.get(
        facts.get("single_precision"), "precision not read")
    mpi = {True: "MPI build", False: "serial build"}.get(facts.get("mpi"), "MPI not read")
    return f"{facts.get('version') or 'version not read'}, {precision}, {mpi}"


def judge_meep(expected, facts) -> dict:
    """``{ok, reference, found, line}``: MEEP against the build the reference names.

    ``expected`` is the platform's ``meep`` entry of ``numerics-relevant.json``
    (``result["meep_reference"]``); ``facts`` holds ``version``,
    ``single_precision`` and ``mpi`` as an imported MEEP reports them. Anything
    unread, or no expectation, is not a match.
    """
    facts = facts if isinstance(facts, dict) else {}
    found = _meep_text(facts) if facts else "not read"
    if not isinstance(expected, dict):
        return {"ok": False, "reference": None, "found": found,
                "line": f"MEEP       {found}: not compared (no reference names a MEEP "
                        "build for this platform)"}
    reference = _meep_text(expected)
    ok = bool(facts) and all(facts.get(key) == expected.get(key)
                             for key in ("version", "single_precision", "mpi"))
    if ok:
        line = f"MEEP       {found}: the reference's build"
    else:
        line = (f"MEEP       {found}: NOT the reference's build ({reference}), so this "
                "environment is not the certification reference, whatever its packages")
    return {"ok": ok, "reference": reference, "found": found, "line": line}


def judge_host(spec: dict, host: dict):
    """``(findings, notes)``: the host facts below the lock's floor, and the unread ones."""
    below, notes = [], []
    for fact, floor in (spec.get("host_floor") or {}).items():
        value = (host or {}).get(fact)
        label = {"macos": "macOS", "glibc": "glibc"}.get(fact, fact)
        if not value:
            notes.append(f"{label} version not read; the lock needs {floor} or later")
        elif _version_key(value) < _version_key(floor):
            below.append(f"{label} {value} is below the lock's floor {floor}")
    return below, notes


def compare(reference: dict, environment: dict) -> dict:
    """The comparison of one environment with one platform's reference."""
    spec = reference["spec"]
    rows = []
    for name, group in judged_names(spec):
        row = judge_conda(name, reference["conda"].get(name), environment, spec)
        row["group"] = group
        rows.append(row)
    for name in spec.get("pypi") or ():
        row = judge_pypi(name, reference["pip"].get(normalize(name)), environment, spec)
        row["group"] = "pypi"
        rows.append(row)
    below, host_notes = judge_host(spec, environment.get("host"))
    counts = {status: sum(1 for row in rows if row["status"] == status)
              for status in (SAME, DIFFERENT, MISSING, BY_DESIGN, UNREAD)}
    total = len(rows)
    off = [row["name"] for row in rows if row["status"] in (DIFFERENT, MISSING)]
    unread = [row["name"] for row in rows if row["status"] == UNREAD]
    designed = [row["name"] for row in rows if row["status"] == BY_DESIGN]
    if off or below:
        kind = DIFFERS
        parts = []
        if off:
            parts.append(f"differs from the certification reference in {len(off)} of "
                         f"{total} numerics-relevant packages: {', '.join(off)}")
        if below:
            parts.append(("and " if off else "differs from the certification reference: ")
                         + "; ".join(below))
        verdict = "; ".join(parts)
        if unread:
            verdict += f"; {len(unread)} more could not be read"
    elif unread:
        kind = UNCONFIRMED
        why = ("this is not a conda environment, so the compiled libraries cannot be read"
               if environment.get("conda") is None else "their builds cannot be read")
        verdict = (f"cannot be confirmed to match the certification reference: {len(unread)} "
                   f"of {total} numerics-relevant packages could not be read ({why}): "
                   f"{', '.join(unread)}")
    else:
        kind = MATCHES
        verdict = (f"matches the certification reference: {counts[SAME]} of {total} "
                   "numerics-relevant packages are the reference's builds")
        if designed:
            verdict += (f" and {len(designed)} {'differs' if len(designed) == 1 else 'differ'}"
                        f" from it by design ({', '.join(designed)})")
    notes = list(host_notes)
    if environment.get("solved_marker"):
        notes.append(f"{environment['solved_marker']} is present: the build script made "
                     "this environment with MEEP_REFERENCE_SOLVE=1, by a solve instead of "
                     "from the lock, and recorded that it is not the certification reference")
    user_site = environment.get("user_site")
    if user_site:
        notes.append(f"the user site directory ({user_site['path']}) is on this "
                     f"interpreter's import path and holds {user_site['distributions']} "
                     "distributions; the reference runs with it off (python -s, or "
                     "PYTHONNOUSERSITE=1)")
    return {"platform": reference["platform"], "kind": kind, "verdict": verdict,
            "rows": rows, "counts": counts, "total": total, "host_below_floor": below,
            "notes": notes, "conda_lock": reference["conda_lock"],
            "pip_lock": reference["pip_lock"], "meep_reference": spec.get("meep"),
            "locked_packages": {"conda": len(reference["conda"]), "pypi": len(reference["pip"])},
            "where": environment.get("where"), "conda_prefix": environment.get("conda_prefix"),
            "conda_packages": None if environment.get("conda") is None
            else len(environment["conda"]),
            "distributions": len(environment.get("distributions") or {})}


def not_compared(reason: str, platform_tag=None) -> dict:
    return {"platform": platform_tag, "kind": NOT_COMPARED,
            "verdict": f"not compared with the certification reference: {reason}",
            "rows": [], "counts": {}, "total": 0, "host_below_floor": [], "notes": []}


def report(platform_tag: str | None = None, prefix: str | None = None,
           lock_dir: str = LOCK_DIR, environment: dict | None = None) -> dict:
    """The comparison of this environment (or ``prefix``, or ``environment``). Never raises."""
    try:
        tag = platform_tag or host_platform()
        if tag is None:
            return not_compared(f"no certification reference lock for "
                                f"{platform.system()} {platform.machine()}; there are "
                                "locks for osx-arm64 and linux-64")
        try:
            reference = load_reference(tag, lock_dir)
        except (OSError, ValueError, LookupError, KeyError, TypeError) as exc:
            return not_compared(f"the lock files could not be read ({exc})", tag)
        if environment is None:
            environment = read_environment(prefix)
        return compare(reference, environment)
    except Exception as exc:  # noqa: BLE001 - the report never raises
        return not_compared(f"the comparison stopped ({type(exc).__name__}: {exc})",
                            platform_tag)


def passes(result: dict) -> bool:
    """True only for the verdict "matches"."""
    return result.get("kind") == MATCHES


def format_report(result: dict, indent: str = "   ") -> list:
    """The report as lines, without a trailing verdict prefix."""
    lines = []
    if result.get("kind") == NOT_COMPARED:
        return [f"{indent}{result['verdict']}"]
    locked = result.get("locked_packages") or {}
    lines.append(f"{indent}reference  {result['platform']}: environments/locks/"
                 f"{result['conda_lock']} ({locked.get('conda')} conda packages) and "
                 f"{result['pip_lock']} ({locked.get('pypi')} PyPI packages)")
    if result.get("conda_prefix"):
        lines.append(f"{indent}read       conda environment {result['conda_prefix']} "
                     f"({result.get('conda_packages')} conda packages), "
                     f"{result.get('distributions')} Python distributions")
    else:
        lines.append(f"{indent}read       {result.get('distributions')} Python distributions; "
                     "no conda-meta directory, so not a conda environment")
    width_name = max([len("package")] + [len(row["name"]) for row in result["rows"]])
    width_ref = max([len("reference")] + [len(row["reference"]) for row in result["rows"]])
    width_found = max([len("this environment")]
                      + [len(str(row.get("found", ""))) for row in result["rows"]])
    width_found = min(width_found, 44)
    lines.append(f"{indent}{'package':<{width_name}}  {'reference':<{width_ref}}  "
                 f"{'this environment':<{width_found}}  verdict")
    printed = set()
    for row in result["rows"]:
        status = {BY_DESIGN: "differs by design"}.get(row["status"], row["status"])
        lines.append(f"{indent}{row['name']:<{width_name}}  {row['reference']:<{width_ref}}  "
                     f"{str(row.get('found', '')):<{width_found}}  {status}")
        # A note is printed under the first row it explains, not repeated under each.
        if row.get("note") and row["status"] != SAME and row["note"] not in printed:
            printed.add(row["note"])
            lines.append(f"{indent}{'':<{width_name}}  ({row['note']})")
    counts = result.get("counts") or {}
    lines.append(f"{indent}of {result['total']}: {counts.get(SAME, 0)} same, "
                 f"{counts.get(DIFFERENT, 0)} different, {counts.get(MISSING, 0)} missing, "
                 f"{counts.get(BY_DESIGN, 0)} differ by design, {counts.get(UNREAD, 0)} not read")
    for finding in result.get("host_below_floor") or ():
        lines.append(f"{indent}host: {finding}")
    for note in result.get("notes") or ():
        lines.append(f"{indent}note: {note}")
    lines.append(f"{indent}Verdict: this environment {result['verdict']}.")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare this environment with the certification reference lock.")
    parser.add_argument("--require-reference", action="store_true",
                        help="exit 1 unless the environment matches the reference")
    parser.add_argument("--prefix", help="compare the conda environment at this path")
    parser.add_argument("--platform", choices=("osx-arm64", "linux-64"),
                        help="compare against this platform's lock")
    parser.add_argument("--lock-dir", default=LOCK_DIR,
                        help="read the locks from this directory")
    parser.add_argument("--json", action="store_true", help="print the comparison as JSON")
    arguments = parser.parse_args(argv)
    result = report(arguments.platform, arguments.prefix, arguments.lock_dir)
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    else:
        print("Certification reference environment", flush=True)
        for line in format_report(result):
            print(line, flush=True)
    if arguments.require_reference and not passes(result):
        print("FAILED: --require-reference was given and this environment does not match "
              "the certification reference.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

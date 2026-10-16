#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""The steps of the reference build scripts that are easier to get right in Python.

``build_meep_133_macos.sh`` and ``build_meep_133_linux.sh`` run each subcommand with the
new environment's own interpreter (``<prefix>/bin/python -s``), so nothing here needs a
package beyond the standard library and nothing reads the user's site directory.

    fetch URL SHA256 OUT [--from F]   download a file (or copy the local file F) and refuse it
                                      unless its SHA-256 is SHA256
    extract TARBALL DEST              unpack a tarball; refuse a member outside DEST or a link
    keep-corpus SOURCE DEST LIST SIBLINGS
                                      copy SOURCE/examples and SOURCE/tests (the MEEP identity
                                      corpus, the tarball's python/ tree) to DEST and write the
                                      SHA-256 of every file to LIST, one "<sha256>  <path>" line;
                                      copy the CORPUS_SIBLINGS beside SOURCE to the same place
                                      beside DEST and list them in SIBLINGS the same way
    verify-corpus DEST LIST SIBLINGS  refuse unless the kept corpus and its siblings are the
                                      listed bytes and no other file is in the corpus
    refresh-corpus SOURCE RECORD_DIR STAGING --tarball-sha256 S --corpus-root R
                                      bring an existing prefix's kept corpus to what a build
                                      keeps now, from SOURCE (the pinned tarball's python/,
                                      freshly extracted): add what is absent, refuse what
                                      differs, rewrite the two lists and only the record's
                                      ``corpus`` entry, and verify
    pip-lock-parts LOCK DEST          split a pip lock into DEST/<n>-<name>.txt, one pin each
                                      with the lock's option lines, in the lock's order
    load-commands OUT IMAGE...        the libraries each image links directly, as JSON
    same-load-commands BEFORE AFTER   refuse unless two load-command records are equal
    link-ffmpeg PREFIX OUT            link the imageio-ffmpeg executable into PREFIX/bin/ffmpeg
    link-libcuda PREFIX OUT           (Linux) PREFIX/lib/meep-gpu-libcuda/libcuda.so{,.1} ->
                                      the driver's libcuda.so.1, for Triton's launcher link
    tools PREFIX PLATFORM OUT         where each program the round runs is found on the
                                      reference PATH; refuse one of the environment's own
                                      programs found outside the prefix
    compiler-facts CXX FLAGS OUT      (a timing control) the compiler, what its target flags
                                      resolve to on this CPU, and the CPU
    identity CHECK_INSTALL MODE OUT   tools/check_install.py's identity verdict in MODE (keep
                                      or flush), and nothing else it checks: its
                                      identity_check, which runs MEEP in a process of its
                                      own and steps nothing; exit 0 pass, 1 fail, 3 when the
                                      only reason is the named refusal of a flush reference
    manifest RECORD_DIR OUT ...       the reference build record (``reference_build.json``)

Every subcommand prints a flushed line per step and exits 1 with ``REFUSING: <why>`` when
the build must stop. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

#: The reference PATH after the environment's own ``bin``: what a certification child runs
#: with (``<prefix>/bin:/usr/bin:/bin``).
REFERENCE_PATH_TAIL = ("/usr/bin", "/bin")

#: Programs the package, its tests, the round, the timing harness and the identity corpus
#: run, found by an inventory of the tree (argv heads of every subprocess call, every
#: ``shutil.which``, every shell script). ``prefix``: the environment carries it, and the
#: build refuses one found elsewhere. ``host``: the operating system's own (the GPU driver,
#: the macOS Command Line Tools, the base system the C library comes with), recorded where
#: it is found and never required here.
TOOLS = {
    "osx-arm64": {
        "prefix": ("python", "mpirun", "git", "ffmpeg", "otool", "c++filt", "clang", "make"),
        "host": ("xcrun", "zsh", "caffeinate", "sysctl", "sw_vers", "pmset", "ioreg", "vm_stat",
                 "ps", "df", "hostname", "uname", "which", "tar"),
    },
    "linux-64": {
        "prefix": ("python", "mpirun", "git", "ffmpeg", "objdump", "c++filt", "readelf",
                   "x86_64-conda-linux-gnu-cc", "make"),
        "host": ("nvidia-smi", "ldconfig", "lscpu", "getconf", "ps", "df", "hostname", "uname",
                 "which", "tar"),
    },
}

#: Where the driver library is looked for when ``ldconfig`` is not on the PATH.
LDCONFIG_CANDIDATES = ("/sbin/ldconfig", "/usr/sbin/ldconfig")


class Refused(Exception):
    """The build must stop here; the message says why."""


def say(text: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, document) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------- fetch, extract -----
def fetch(url: str, expected: str, out: str, local: str = "") -> None:
    """Download ``url`` (or copy the local file ``local``, a tarball fetched before) to
    ``out``; refused and removed unless its SHA-256 is ``expected`` either way."""
    if local:
        say(f"copying {local} (in place of {url})")
        shutil.copyfile(local, out)
    else:
        say(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=600) as response, open(out, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
    actual = sha256_of(out)
    say(f"expected {expected}")
    say(f"actual   {actual}")
    if actual != expected:
        os.remove(out)
        raise Refused(f"the SHA-256 of {os.path.basename(out)} is {actual}, not {expected}")


def extract(tarball: str, dest: str) -> None:
    """Unpack ``tarball`` into ``dest``: files and directories only, all inside ``dest``."""
    root = os.path.realpath(dest)
    os.makedirs(root, exist_ok=True)
    with tarfile.open(tarball, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = os.path.realpath(os.path.join(root, member.name))
            parts = member.name.replace("\\", "/").split("/")
            if member.name.startswith("/") or ".." in parts or not (
                    target == root or target.startswith(root + os.sep)):
                raise Refused(f"{member.name} lies outside the extraction directory")
            if not (member.isfile() or member.isdir()):
                raise Refused(f"{member.name} is not a file or a directory")
        if hasattr(tarfile, "data_filter"):
            archive.extractall(root, members=members, filter="data")
        else:  # every member was checked above
            archive.extractall(root, members=members)
    say(f"extracted {len(members)} members into {dest}")


# ------------------------------------------------------------------------ corpus -----
CORPUS_DIRECTORIES = ("examples", "tests")

#: The files OUTSIDE the tarball's python/ tree that corpus rows read, relative to the
#: tarball's top directory (the parent of python/), where the build keeps them: beside the
#: kept python/, so a row's own relative path reaches them. FOUND BY TOOL over the 194 rows
#: (2026-10-15): a grep of python/examples and python/tests for paths that leave python/,
#: and every row's capture, structure initialisation and first steps under a file-access
#: trace of the process and its children. Both find this one file, which
#: test_simulation.py reads as ``dirname(__file__)/../../tests/cyl-ellipsoid-eps-ref.h5``:
#: test_epsilon_input_file hands it to MEEP as ``epsilon_input_file`` (MEEP opens it when
#: the structure is built, and aborts the process when it is absent) and test_numpy_epsilon
#: reads it with h5py (absent, the capture keeps the simulation built before the read: the
#: row measures vacuum, silently).
CORPUS_SIBLINGS = ("tests/cyl-ellipsoid-eps-ref.h5",)


def read_listing(listing: str) -> dict:
    """``{path: sha256}`` from a list of ``<sha256>  <path>`` lines; refuse another line."""
    entries = {}
    with open(listing, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            digest, sep, relative = line.rstrip("\n").partition("  ")
            if not sep or len(digest) != 64 or not relative:
                raise Refused(f"{listing}:{number} is not a \"<sha256>  <path>\" line")
            entries[relative] = digest
    return entries


def write_listing(listing: str, entries: dict) -> str:
    """Write ``{path: sha256}`` sorted by path; return the list's own SHA-256."""
    text = "".join(f"{entries[relative]}  {relative}\n" for relative in sorted(entries))
    with open(listing, "w", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_files(source: str) -> list:
    """Every corpus file under ``source`` (a python/ tree), relative and sorted."""
    found = []
    for directory in CORPUS_DIRECTORIES:
        top = os.path.join(source, directory)
        if not os.path.isdir(top):
            raise Refused(f"the tarball has no python/{directory} directory")
        for base, dirs, files in os.walk(top):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(base, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    raise Refused(f"{path} is not a regular file")
                found.append(os.path.relpath(path, source).replace(os.sep, "/"))
    return sorted(found)


def sibling_source(source: str, relative: str) -> str:
    """Where a sibling lies beside the python/ tree ``source``; refuse one that is absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(source)), relative)
    if os.path.islink(path) or not os.path.isfile(path):
        raise Refused(f"the tarball has no {relative} beside python/ (a corpus sibling)")
    return path


def keep_corpus(source: str, dest: str, listing: str, siblings: str) -> dict:
    """Copy the corpus and its siblings and list every file's SHA-256;
    ``{files, list_sha256, siblings}``."""
    if os.path.exists(dest):
        raise Refused(f"{dest} already exists; the corpus is kept once, from the tarball")
    for relative in CORPUS_SIBLINGS:
        target = os.path.join(os.path.dirname(os.path.abspath(dest)), relative)
        if os.path.lexists(target):
            raise Refused(f"{target} already exists; the corpus is kept once, from the tarball")
    kept = {}
    for relative in corpus_files(source):
        target = os.path.join(dest, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(source, relative), target)
        kept[relative] = sha256_of(target)
    beside = {}
    for relative in CORPUS_SIBLINGS:
        target = os.path.join(os.path.dirname(os.path.abspath(dest)), relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(sibling_source(source, relative), target)
        beside[relative] = sha256_of(target)
    digest = write_listing(listing, kept)
    write_listing(siblings, beside)
    say(f"kept {len(kept)} corpus files in {dest}; their list {os.path.basename(listing)} "
        f"has SHA-256 {digest}")
    say(f"kept {len(beside)} corpus sibling(s) beside it, listed in "
        f"{os.path.basename(siblings)}: {', '.join(sorted(beside))}")
    return {"files": len(kept), "list_sha256": digest, "siblings": beside}


def verify_corpus(dest: str, listing: str, siblings: str) -> list:
    """The corpus files and siblings whose bytes or presence differ from the lists, and a
    sibling list that is not :data:`CORPUS_SIBLINGS` (empty when none)."""
    problems = []
    listed = read_listing(listing)
    for relative, digest in sorted(listed.items()):
        path = os.path.join(dest, relative)
        if not os.path.isfile(path):
            problems.append(f"{relative} is missing")
        elif sha256_of(path) != digest:
            problems.append(f"{relative} differs")
    for directory in CORPUS_DIRECTORIES:
        for base, _dirs, files in os.walk(os.path.join(dest, directory)):
            for name in files:
                relative = os.path.relpath(os.path.join(base, name), dest).replace(os.sep, "/")
                if relative not in listed:
                    problems.append(f"{relative} is not in the list")
    beside = read_listing(siblings)
    if sorted(beside) != sorted(CORPUS_SIBLINGS):
        problems.append(f"the sibling list names {sorted(beside)}, not the corpus siblings "
                        f"{sorted(CORPUS_SIBLINGS)}")
    for relative, digest in sorted(beside.items()):
        path = os.path.join(os.path.dirname(os.path.abspath(dest)), relative)
        if not os.path.isfile(path):
            problems.append(f"sibling {relative} is missing")
        elif sha256_of(path) != digest:
            problems.append(f"sibling {relative} differs")
    return problems


def corpus_entry(record_dir: str, corpus_root: str, corpus_list: str,
                 corpus_siblings: str) -> dict:
    """The record's ``corpus`` entry: the kept python/ tree (``root``, relative to the
    prefix), its list (``list``, relative to ``record_dir``, with its SHA-256 and line
    count) and the siblings beside it (``siblings``: each file's path relative to the
    parent of ``root``, and its SHA-256)."""
    with open(corpus_list, encoding="utf-8") as handle:
        files = sum(1 for _ in handle)
    return {"root": corpus_root,
            "list": os.path.relpath(corpus_list, record_dir),
            "list_sha256": sha256_of(corpus_list),
            "files": files,
            "siblings": [{"path": relative, "sha256": digest}
                         for relative, digest in sorted(read_listing(corpus_siblings).items())]}


def render_json(document) -> str:
    """The bytes :func:`write_json` writes for ``document``."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def files_under(top: str) -> list:
    """Every file and link under ``top``, relative to it, sorted (empty when it is absent)."""
    found = []
    for base, dirs, files in os.walk(top):
        for name in files + [name for name in dirs if os.path.islink(os.path.join(base, name))]:
            found.append(os.path.relpath(os.path.join(base, name), top).replace(os.sep, "/"))
    return sorted(found)


#: What a refresh writes beside the record: when, from which tarball, the record's SHA-256
#: before and after, and the files it added.
REFRESH_LOG = "corpus_refresh.json"


def refresh_corpus(source: str, record_dir: str, staging: str, tarball_sha256: str,
                   corpus_root: str, corpus_list: str, corpus_siblings: str) -> dict:
    """Bring the kept corpus of an EXISTING prefix to what a build keeps now.

    ``source`` is the pinned tarball's python/ tree, freshly extracted. The corpus a build
    would keep is staged under ``staging`` (:func:`keep_corpus`) and compared with the
    prefix's: a file of it that the prefix lacks is copied in, one the prefix holds with
    other bytes, and a file in the prefix's corpus that it does not hold, are refused by
    name before anything is written (a refresh adds; it replaces nothing). Then both lists
    are rewritten, the record's ``corpus`` entry and nothing else is replaced, the result is
    verified (:func:`verify_corpus`) and :data:`REFRESH_LOG` is written. MEEP, the locks and
    every package of the prefix are untouched. Refused, before anything is written: no
    record, a record that is not as the build wrote it, a record of another tarball or
    another corpus root.
    """
    record_path = os.path.join(record_dir, "reference_build.json")
    if not os.path.isfile(record_path):
        raise Refused(f"no reference build record at {record_path}: a refresh works on a "
                      "reference environment this script built")
    with open(record_path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise Refused(f"{record_path} does not read: {exc}") from exc
    if not isinstance(document, dict) or render_json(document) != text:
        raise Refused(f"{record_path} is not as the build wrote it, so rewriting its corpus "
                      "entry alone cannot be shown; rebuild the environment")
    built_from = (document.get("meep") or {}).get("tarball_sha256")
    if built_from != tarball_sha256:
        raise Refused(f"the record's MEEP was built from the tarball {built_from}, not "
                      f"{tarball_sha256}: the corpus is kept from the tarball MEEP was built "
                      "from")
    previous = document.get("corpus")
    named = previous.get("root") if isinstance(previous, dict) else None
    if named != corpus_root:
        raise Refused(f"the record names the corpus root {named!r}, not {corpus_root!r}")
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(record_dir)))
    root = os.path.join(prefix, corpus_root)
    staged = os.path.join(staging, "kept", "python")
    staged_list = os.path.join(staging, os.path.basename(corpus_list))
    staged_siblings = os.path.join(staging, os.path.basename(corpus_siblings))
    keep_corpus(source, staged, staged_list, staged_siblings)
    wanted = {relative: (os.path.join(staged, relative), os.path.join(root, relative), digest)
              for relative, digest in read_listing(staged_list).items()}
    for relative, digest in read_listing(staged_siblings).items():
        wanted["../" + relative] = (os.path.join(os.path.dirname(staged), relative),
                                    os.path.join(os.path.dirname(root), relative), digest)
    differ, add = [], []
    for relative, (origin, target, digest) in sorted(wanted.items()):
        if not os.path.lexists(target):
            add.append((relative, origin, target, digest))
        elif os.path.islink(target) or not os.path.isfile(target) or sha256_of(target) != digest:
            differ.append(relative)
    extra = [relative for directory in CORPUS_DIRECTORIES
             for relative in (f"{directory}/{name}"
                              for name in files_under(os.path.join(root, directory)))
             if relative not in wanted]
    if differ or extra:
        raise Refused(f"the kept corpus at {root} is not the pinned tarball's: "
                      f"{len(differ)} file(s) with other bytes ({', '.join(differ[:5])}), "
                      f"{len(extra)} file(s) the tarball does not hold ({', '.join(extra[:5])}); "
                      "a refresh adds what a build keeps and replaces nothing: rebuild the "
                      "environment")
    for relative, origin, target, digest in add:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(origin, target)
        if sha256_of(target) != digest:
            raise Refused(f"{target} was copied with other bytes than {origin}")
        say(f"added {relative}")
    for kept, staged_copy in ((corpus_list, staged_list), (corpus_siblings, staged_siblings)):
        shutil.copyfile(staged_copy, kept + ".refresh")
        os.replace(kept + ".refresh", kept)
    document["corpus"] = corpus_entry(record_dir, corpus_root, corpus_list, corpus_siblings)
    rendered = render_json(document)
    with open(record_path + ".refresh", "w", encoding="utf-8") as handle:
        handle.write(rendered)
    os.replace(record_path + ".refresh", record_path)
    problems = verify_corpus(root, corpus_list, corpus_siblings)
    if problems:
        raise Refused(f"after the refresh {len(problems)} corpus file(s) are not the listed "
                      "bytes: " + "; ".join(problems[:10]))
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "tarball_sha256": tarball_sha256,
               "record_sha256_before": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "record_sha256_after": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
               "corpus_before": previous, "corpus_after": document["corpus"],
               "added": [relative for relative, *_ in add]}
    write_json(os.path.join(record_dir, REFRESH_LOG), summary)
    say(f"refreshed the corpus of {prefix}: added {len(add)} file(s) "
        f"({', '.join(summary['added']) or 'none'}); record {summary['record_sha256_before']} "
        f"-> {summary['record_sha256_after']}; only its corpus entry changed")
    return summary


# ---------------------------------------------------------------- pip lock parts -----
#: A pin of a pip lock: one line, the version and the SHA-256 of the one wheel it names.
PIP_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9._+!-]+ --hash=sha256:[0-9a-f]{64}")


def pip_lock_parts(lock: str, dest: str) -> list:
    """Split a pip lock into one requirement file per pin, in the lock's order.

    pip keeps every wheel it downloads until the command exits, so one install of the
    whole lock holds all of its wheels at once beside what they unpack to; one install per
    part holds one wheel at a time. Each part is the lock's option lines and one of its
    pins, unchanged, so the set installed is the lock's either way. Refuses, before it
    writes anything, a lock that does not require hashes, holds no pin, carries a pin over
    more than one line or a line that is neither an option nor a pin by version and
    SHA-256, or names an option after its first pin; and a ``dest`` that exists.
    """
    name = os.path.basename(lock)
    options, pins = [], []
    with open(lock, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.rstrip("\n")
            if not text.strip() or text.startswith("#"):
                continue
            if text.endswith("\\") or text[0].isspace():
                raise Refused(f"{name}:{number}: a pin carried over more than one line; "
                              "the parts take one line per pin")
            if text.startswith("--"):
                if pins:
                    raise Refused(f"{name}:{number}: an option after the first pin")
                options.append(text)
            elif PIP_PIN.fullmatch(text):
                pins.append(text)
            else:
                raise Refused(f"{name}:{number}: neither an option nor a pin by version and "
                              f"SHA-256: {text!r}")
    if "--require-hashes" not in options:
        raise Refused(f"{name} does not require hashes")
    if not pins:
        raise Refused(f"{name} holds no pin")
    if os.path.exists(dest):
        raise Refused(f"{dest} already exists")
    os.makedirs(dest)
    width = len(str(len(pins)))
    parts = []
    for index, pin in enumerate(pins, 1):
        path = os.path.join(dest, f"{index:0{width}d}-{pin.split('==', 1)[0]}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(options + [pin]) + "\n")
        parts.append(path)
    say(f"split {name} into {len(parts)} parts of one pin each, each with the lock's "
        f"{len(options)} option lines")
    return parts


# ----------------------------------------------------------------- load commands -----
def elf_needed(path: str):
    """``DT_NEEDED`` of a 64-bit little-endian ELF file, or None when it is not one."""
    import struct

    with open(path, "rb") as handle:
        data = handle.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    phoff, = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    loads, dynamic = [], None
    for index in range(phnum):
        p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz = struct.unpack_from(
            "<IIQQQQ", data, phoff + index * phentsize)
        if p_type == 1:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == 2:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        return []
    entries, strtab = [], None
    for offset in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        if tag == 0:
            break
        entries.append((tag, value))
        if tag == 5:
            strtab = value
    base = next((offset + (strtab - vaddr) for vaddr, offset, size in loads
                 if strtab is not None and vaddr <= strtab < vaddr + size), None)
    needed = []
    for tag, value in entries:
        if tag == 1 and base is not None:
            end = data.index(b"\0", base + value)
            needed.append(data[base + value:end].decode("utf-8", "replace"))
    return needed


def macho_dylibs(path: str):
    """The dylibs a 64-bit Mach-O file loads (LC_LOAD_DYLIB and kin), or None."""
    import struct

    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != 0xFEEDFACF:
        return None
    ncmds, = struct.unpack_from("<I", data, 16)
    offset, dylibs = 32, []
    for _ in range(ncmds):
        command, size = struct.unpack_from("<II", data, offset)
        if command in (0xC, 0x80000018, 0x8000001F, 0x80000023):
            name_offset, = struct.unpack_from("<I", data, offset + 8)
            dylibs.append(data[offset + name_offset:offset + size].split(b"\0", 1)[0]
                          .decode("utf-8", "replace"))
        offset += size
    return dylibs


def load_commands(images) -> dict:
    """``{image: [libraries it links directly]}``, read from the file headers."""
    out = {}
    for image in images:
        found = macho_dylibs(image)
        if found is None:
            found = elf_needed(image)
        if found is None:
            raise Refused(f"{image} is neither a 64-bit Mach-O nor a 64-bit ELF file")
        out[os.path.basename(image)] = found
        say(f"{os.path.basename(image)}: {len(found)} libraries linked directly")
    return out


def same_load_commands(before: str, after: str) -> None:
    with open(before, encoding="utf-8") as handle:
        first = json.load(handle)
    with open(after, encoding="utf-8") as handle:
        second = json.load(handle)
    if first != second:
        changed = sorted(name for name in set(first) | set(second)
                         if first.get(name) != second.get(name))
        raise Refused(f"the load commands of {', '.join(changed)} changed after MEEP was "
                      "installed")
    say(f"{len(first)} image(s): the same load commands as when MEEP was installed")


# ------------------------------------------------------------------------ ffmpeg -----
def site_packages(prefix: str) -> list:
    """The ``site-packages`` directories of ``prefix``'s ``lib/python*``, each once.

    The environment's Python ships ``lib/python3.1`` as a link to ``lib/python3.12`` (for
    build systems that read the version as ``sys.version[:3]``); a ``python*`` entry that
    is a link is skipped, so no directory is read under two names.
    """
    lib = os.path.join(prefix, "lib")
    return sorted(os.path.join(lib, name, "site-packages") for name in os.listdir(lib)
                  if name.startswith("python") and not os.path.islink(os.path.join(lib, name))
                  and os.path.isdir(os.path.join(lib, name, "site-packages")))


def link_ffmpeg(prefix: str) -> dict:
    """``PREFIX/bin/ffmpeg`` -> the one executable the imageio-ffmpeg wheel installed."""
    found = []
    for site in site_packages(prefix):
        binaries = os.path.join(site, "imageio_ffmpeg", "binaries")
        if os.path.isdir(binaries):
            found += [os.path.join(binaries, name) for name in sorted(os.listdir(binaries))
                      if name.startswith("ffmpeg-")]
    if len(found) != 1:
        raise Refused(f"the environment holds {len(found)} imageio-ffmpeg executables, not 1 "
                      "(is imageio-ffmpeg in the pip lock?)")
    target = found[0]
    if not os.access(target, os.X_OK):
        raise Refused(f"{target} is not executable")
    link = os.path.join(prefix, "bin", "ffmpeg")
    if os.path.lexists(link):
        raise Refused(f"{link} already exists; the environment's ffmpeg is the wheel's only")
    linked = macho_dylibs(target)
    if linked is None:
        linked = elf_needed(target)
    outside = [name for name in (linked or [])
               if not name.startswith(("/usr/lib/", "/System/Library/"))]
    if outside:
        raise Refused(f"{target} loads libraries outside the operating system: {outside}")
    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
    done = subprocess.run([link, "-version"], capture_output=True, text=True, timeout=120)
    first = (done.stdout.splitlines() or [""])[0]
    if done.returncode != 0 or not first.startswith("ffmpeg version"):
        raise Refused(f"{link} -version exited {done.returncode}: {first or done.stderr[-300:]}")
    say(f"linked {link} -> {target}")
    say(first)
    return {"link": "bin/ffmpeg", "target": os.path.relpath(target, prefix),
            "sha256": sha256_of(target), "version": first, "links": linked or []}


# ----------------------------------------------------------------------- libcuda -----
def driver_libcuda() -> str | None:
    """The driver's ``libcuda.so.1`` as the dynamic linker's cache lists it, or None."""
    tool = shutil.which("ldconfig") or next(
        (path for path in LDCONFIG_CANDIDATES if os.access(path, os.X_OK)), None)
    if tool is None:
        return None
    done = subprocess.run([tool, "-p"], capture_output=True, text=True, timeout=120)
    for line in done.stdout.splitlines():
        if line.strip().startswith("libcuda.so.1 ") and "x86-64" in line:
            return line.split("=>")[-1].strip()
    return None


def link_libcuda(prefix: str) -> dict:
    """Triton links its launcher with ``-lcuda``; a driver installs ``libcuda.so.1`` only."""
    target = driver_libcuda()
    if target is None:
        say("no libcuda.so.1 in the dynamic linker's cache: no NVIDIA driver on this host; "
            "the Triton launcher link needs TRITON_LIBCUDA_PATH set by hand on the GPU host")
        return {"directory": None, "target": None}
    directory = os.path.join(prefix, "lib", "meep-gpu-libcuda")
    os.makedirs(directory)
    for name in ("libcuda.so", "libcuda.so.1"):
        os.symlink(target, os.path.join(directory, name))
    say(f"{directory}/libcuda.so{{,.1}} -> {target} (the GPU driver's)")
    return {"directory": os.path.relpath(directory, prefix), "target": target}


# ------------------------------------------------------------------------- tools -----
def find_tools(prefix: str, platform: str) -> dict:
    """Each program on the reference PATH; refuse an environment program found elsewhere."""
    spec = TOOLS[platform]
    bin_dir = os.path.join(prefix, "bin")
    path = os.pathsep.join([bin_dir, *REFERENCE_PATH_TAIL])
    found = {"prefix": {}, "host": {}}
    outside = []
    for name in spec["prefix"]:
        where = shutil.which(name, path=path)
        found["prefix"][name] = where
        if where is None or os.path.dirname(where) != bin_dir:
            outside.append(f"{name} ({where or 'not found'})")
    host_path = os.pathsep.join([*REFERENCE_PATH_TAIL, "/usr/sbin", "/sbin"])
    for name in spec["host"]:
        found["host"][name] = shutil.which(name, path=host_path)
    say(f"{len(spec['prefix']) - len(outside)} of {len(spec['prefix'])} of the environment's "
        "programs are in its bin; host programs: "
        + ", ".join(f"{name} {'found' if where else 'absent'}"
                    for name, where in found["host"].items()))
    if outside:
        raise Refused("the environment's own programs must come from <prefix>/bin on the "
                      f"reference PATH: {', '.join(outside)}")
    return found


# ----------------------------------------------------------------- compiler facts -----
def compiler_facts(cxx: str, flags: str) -> dict:
    """What a timing control's flags made of this CPU; recorded, never judged."""
    def run(command):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=120)
            return (done.stdout + done.stderr).strip()
        except (OSError, subprocess.SubprocessError) as error:
            return f"unavailable: {type(error).__name__}: {error}"

    system = platform_module.system()
    record = {"flags": flags, "compiler": cxx, "platform": platform_module.platform(),
              "portable_binary_note": ("--enable-portable-binary changes only what "
                                       "AX_CC_MAXOPT picks when CFLAGS is unset; CFLAGS and "
                                       "CXXFLAGS were set, so it had no effect"),
              "compiler_version": run([cxx, "-v"])[-2000:]}
    if system == "Linux":
        record["march_resolution"] = run([cxx, *flags.split(), "-Q", "--help=target"])[-6000:]
        record["cpu"] = next((line.split(":", 1)[1].strip() for line in
                              run(["lscpu"]).splitlines() if line.startswith("Model name")), None)
    else:
        record["mcpu_resolution"] = run([cxx, *flags.split(), "-###", "-x", "c++", "-c",
                                         os.devnull])[-4000:]
        record["cpu"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    say(f"compiler {cxx}; cpu {record.get('cpu')}")
    return record


# ---------------------------------------------------------------------- identity -----
#: The one refusal a Mac's flush verdict may carry and still be recorded as pending: MEEP's
#: switch has no effect on Apple silicon, and a check that does not drive the processor's
#: FZ bit itself says so by name.
NO_FLUSH_REFERENCE = "no flush reference"
PASSED, PENDING, FAILED = 0, 3, 1


def identity_verdict(check_install: str, mode: str) -> tuple:
    """``(status, lines)``: check_install's identity verdict in ``mode``, alone.

    ``status`` is PASSED, FAILED, or PENDING when the verdict's only reason is the named
    refusal of a flush reference.

    ``check_install.py --require-identity-reference MODE`` goes on, when the verdict is a
    pass, to the rest of the installation check, which steps a simulation on the GPU; a
    build records the identity verdict only, so it calls the check's own
    ``identity_check`` and nothing after it.
    """
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("check_install_for_the_build", check_install)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = types.SimpleNamespace(require_identity_reference=mode, identity_build_record=None,
                                      gpu_id=0, verbose=False, require_reference=False,
                                      require_gpu=False)
    lines, failure = module.identity_check(arguments)
    lines = [f"MEEP identity reference (--require-identity-reference {mode})", *lines]
    if failure is None:
        lines.append(f"PASSED: the MEEP this environment loads is an identity reference in {mode}.")
        return PASSED, lines
    lines.append(f"FAILED: {failure.reason}")
    lead = f"--require-identity-reference {mode} was given and "
    reasons = failure.reason[len(lead):].rstrip(".").split("; and ") \
        if failure.reason.startswith(lead) else [failure.reason]
    if mode == "flush" and len(reasons) == 1 and NO_FLUSH_REFERENCE in reasons[0]:
        return PENDING, lines
    return FAILED, lines


# ---------------------------------------------------------------------- manifest -----
def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_text(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def host_facts() -> dict:
    facts = {"system": platform_module.system(), "machine": platform_module.machine(),
             "release": platform_module.release()}
    if facts["system"] == "Darwin":
        facts["macos"] = platform_module.mac_ver()[0]
    else:
        text = os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else ""
        facts["glibc"] = text.split()[1] if text and text.startswith("glibc ") else None
    return facts


def manifest(record_dir: str, arguments) -> dict:
    """``reference_build.json``: what makes this prefix the reference, in one record."""
    verdicts = {}
    for mode in ("keep", "flush"):
        text = read_text(os.path.join(record_dir, f"identity_{mode}.txt"))
        status = read_text(os.path.join(record_dir, f"identity_{mode}.status"))
        verdicts[mode] = {"exit_status": int(status) if status and status.strip().isdigit()
                          else None,
                          "verdict": (text or "").strip().splitlines()[-1:] or None,
                          "outcome": (read_text(os.path.join(record_dir,
                                                             f"identity_{mode}.outcome"))
                                      or "not run").strip()}
    environment = read_json(os.path.join(record_dir, "reference_environment.json")) or {}
    census = read_json(os.path.join(record_dir, "fma_census.json")) or {}
    build = read_json(os.path.join(record_dir, "build_record.json")) or {}
    locks = {}
    for name in sorted(os.listdir(record_dir)):
        if name.endswith((".conda.txt", ".pip.txt")) or name in ("numerics-relevant.json",
                                                                  "additions.json"):
            locks[name] = sha256_of(os.path.join(record_dir, name))
    document = {
        "schema": "meep-gpu/reference-build/1",
        "platform": arguments.platform,
        "role": arguments.role,
        "meep": {"version": "1.33.0", "tarball": arguments.tarball_url,
                 "tarball_sha256": arguments.tarball_sha256,
                 "cflags": arguments.cflags, "cxxflags": arguments.cxxflags,
                 "configure": arguments.configure},
        "images": build.get("sha256"),
        "fma_census": census.get("verdict"),
        "corpus": corpus_entry(record_dir, arguments.corpus_root, arguments.corpus_list,
                               arguments.corpus_siblings),
        "locks_sha256": locks,
        "reference_environment": {"kind": environment.get("kind"),
                                  "verdict": environment.get("verdict")},
        "identity": verdicts,
        "load_commands": read_json(os.path.join(record_dir, "load_commands.json")),
        "ffmpeg": read_json(os.path.join(record_dir, "ffmpeg.json")),
        "libcuda": read_json(os.path.join(record_dir, "libcuda.json")),
        "tools": read_json(os.path.join(record_dir, "tools.json")),
        "macos_sdk": (read_text(os.path.join(record_dir, "macos_sdk.txt")) or "").strip() or None,
        "host": host_facts(),
        "build_record_sha256": (sha256_of(os.path.join(record_dir, "build_record.json"))
                                if os.path.isfile(os.path.join(record_dir, "build_record.json"))
                                else None),
    }
    return document


# -------------------------------------------------------------------------- main -----
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("url")
    p.add_argument("sha256")
    p.add_argument("out")
    p.add_argument("--from", dest="local", default="")
    p = sub.add_parser("extract")
    p.add_argument("tarball")
    p.add_argument("dest")
    p = sub.add_parser("keep-corpus")
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("listing")
    p.add_argument("siblings")
    p = sub.add_parser("verify-corpus")
    p.add_argument("dest")
    p.add_argument("listing")
    p.add_argument("siblings")
    p = sub.add_parser("refresh-corpus")
    p.add_argument("source")
    p.add_argument("record_dir")
    p.add_argument("staging")
    for name in ("tarball-sha256", "corpus-root", "corpus-list", "corpus-siblings"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("pip-lock-parts")
    p.add_argument("lock")
    p.add_argument("dest")
    p = sub.add_parser("load-commands")
    p.add_argument("out")
    p.add_argument("images", nargs="+")
    p = sub.add_parser("same-load-commands")
    p.add_argument("before")
    p.add_argument("after")
    p = sub.add_parser("link-ffmpeg")
    p.add_argument("prefix")
    p.add_argument("out")
    p = sub.add_parser("link-libcuda")
    p.add_argument("prefix")
    p.add_argument("out")
    p = sub.add_parser("tools")
    p.add_argument("prefix")
    p.add_argument("platform", choices=sorted(TOOLS))
    p.add_argument("out")
    p = sub.add_parser("compiler-facts")
    p.add_argument("cxx")
    p.add_argument("flags")
    p.add_argument("out")
    p = sub.add_parser("identity")
    p.add_argument("check_install")
    p.add_argument("mode", choices=("keep", "flush"))
    p.add_argument("out")
    p = sub.add_parser("manifest")
    p.add_argument("record_dir")
    p.add_argument("out")
    for name in ("platform", "role", "tarball-url", "tarball-sha256", "cflags", "cxxflags",
                 "configure", "corpus-root", "corpus-list", "corpus-siblings"):
        p.add_argument(f"--{name}", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            fetch(arguments.url, arguments.sha256, arguments.out, arguments.local)
        elif arguments.command == "extract":
            extract(arguments.tarball, arguments.dest)
        elif arguments.command == "keep-corpus":
            keep_corpus(arguments.source, arguments.dest, arguments.listing, arguments.siblings)
        elif arguments.command == "verify-corpus":
            problems = verify_corpus(arguments.dest, arguments.listing, arguments.siblings)
            if problems:
                raise Refused(f"{len(problems)} corpus file(s) are not the listed bytes: "
                              + "; ".join(problems[:10]))
            say("every corpus file and sibling is the listed bytes, and no other file is in "
                "the corpus")
        elif arguments.command == "refresh-corpus":
            refresh_corpus(arguments.source, arguments.record_dir, arguments.staging,
                           arguments.tarball_sha256, arguments.corpus_root,
                           arguments.corpus_list, arguments.corpus_siblings)
        elif arguments.command == "pip-lock-parts":
            pip_lock_parts(arguments.lock, arguments.dest)
        elif arguments.command == "load-commands":
            write_json(arguments.out, load_commands(arguments.images))
        elif arguments.command == "same-load-commands":
            same_load_commands(arguments.before, arguments.after)
        elif arguments.command == "link-ffmpeg":
            write_json(arguments.out, link_ffmpeg(arguments.prefix))
        elif arguments.command == "link-libcuda":
            write_json(arguments.out, link_libcuda(arguments.prefix))
        elif arguments.command == "tools":
            write_json(arguments.out, find_tools(arguments.prefix, arguments.platform))
        elif arguments.command == "compiler-facts":
            write_json(arguments.out, compiler_facts(arguments.cxx, arguments.flags))
        elif arguments.command == "identity":
            status, lines = identity_verdict(arguments.check_install, arguments.mode)
            with open(arguments.out, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            for line in lines:
                print(f"    {line}", flush=True)
            return status
        elif arguments.command == "manifest":
            write_json(arguments.out, manifest(arguments.record_dir, arguments))
            say(f"wrote {arguments.out}")
    except Refused as exc:
        print(f"  REFUSING: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

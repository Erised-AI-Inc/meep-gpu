"""Pack, verify and unpack the evidence-archive inputs of a Metal certification round.

A Metal round on a Mac that does not hold the evidence archive reads a few measured
records from ``parity/meep_gpu/results/`` that the repository does not carry. A round
directory under ``parity/meep_gpu/rounds/`` names them:

``INPUT_FILES.txt``
    one path per line, relative to ``results/``: exactly the files the bundle carries.
``INPUT_DIRS.txt``
    the directories those files sit in. Only the listed files ship from each.
``INPUTS_SHA256SUMS``
    ``<sha256>  <path>`` per file, the format ``shasum -a 256 -c`` reads.
``metal_round_inputs.tgz.sha256``
    the digest of the bundle that was cut from the evidence archive for that manifest.

The bundle itself is not committed: its census rows record the interpreter paths of
the host that cut them.

    python parity/meep_gpu/build_metal_round_inputs.py check  --round <round-dir>
    python parity/meep_gpu/build_metal_round_inputs.py pack   --round <round-dir> --out <file.tgz>
    python parity/meep_gpu/build_metal_round_inputs.py verify --round <round-dir> --bundle <file.tgz>
    python parity/meep_gpu/build_metal_round_inputs.py unpack --round <round-dir> --bundle <file.tgz>

``check`` verifies the files present under ``results/`` against the manifest, with no
bundle (the inputs step on the host that holds the archive). ``verify`` checks a
bundle's digest against the committed one, its member set and every member's digest,
and writes nothing. ``pack`` writes a
deterministic tarball (sorted members, mtime 0, no owner names, gzip mtime 0) and its
``.sha256``, then verifies what it wrote. ``unpack`` checks the bundle's digest against
the committed one, extracts only files that are absent, never writes through a symbolic
link (on the archive host ``results/`` entries link into the archive), refuses a present
file whose digest differs, and ends with ``check``.

Exit status: 0 verified; 1 a file or the bundle is missing, differs or cannot be placed
safely; 2 a usage or manifest error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]

FILES = "INPUT_FILES.txt"
DIRS = "INPUT_DIRS.txt"
SUMS = "INPUTS_SHA256SUMS"
BUNDLE = "metal_round_inputs.tgz"
BUNDLE_SUM = BUNDLE + ".sha256"
#: Members are copied in chunks of this size, so a large census file is never held
#: in memory twice.
CHUNK = 1 << 20


class ManifestError(Exception):
    """The round directory's manifest is malformed or inconsistent (exit 2)."""


class InputsRefused(Exception):
    """A file or bundle does not match the manifest, or cannot be placed safely (exit 1)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _lines(path: Path) -> List[str]:
    if not path.is_file():
        raise ManifestError(f"{path} is missing")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _safe_relative(text: str) -> str:
    """A manifest path: relative, POSIX, no ``..``, no empty or ``.`` component."""
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts:
        raise ManifestError(f"{text!r} is not a relative path")
    if any(part in ("", ".", "..") for part in path.parts) or "\\" in text:
        raise ManifestError(f"{text!r} has a component that leaves results/")
    return str(path)


def read_manifest(round_dir: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
    """``(files, dirs, sums)``, after checking the three files agree with one another."""
    files = [_safe_relative(line) for line in _lines(round_dir / FILES)]
    dirs = [_safe_relative(line) for line in _lines(round_dir / DIRS)]
    if len(set(files)) != len(files):
        raise ManifestError(f"{FILES} lists a file twice")
    if len(set(dirs)) != len(dirs):
        raise ManifestError(f"{DIRS} lists a directory twice")
    sums: Dict[str, str] = {}
    for line in _lines(round_dir / SUMS):
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestError(f"{SUMS}: {line!r} is not '<sha256>  <path>'")
        sums[_safe_relative(name)] = digest
    if set(sums) != set(files):
        raise ManifestError(
            f"{SUMS} and {FILES} name different files: "
            f"only in {SUMS} {sorted(set(sums) - set(files))}, "
            f"only in {FILES} {sorted(set(files) - set(sums))}")
    outside = [name for name in files
               if not any(name.startswith(directory + "/") for directory in dirs)]
    if outside:
        raise ManifestError(f"{outside} lie under no directory of {DIRS}")
    unused = [directory for directory in dirs
              if not any(name.startswith(directory + "/") for name in files)]
    if unused:
        raise ManifestError(f"{DIRS} names {unused}, which hold no listed file")
    return files, dirs, sums


def sums_text(files: Sequence[str], sums: Dict[str, str]) -> str:
    return "".join(f"{sums[name]}  {name}\n" for name in files)


def committed_bundle_digest(round_dir: Path) -> str:
    lines = _lines(round_dir / BUNDLE_SUM)
    digest, _, name = lines[0].partition("  ")
    if len(lines) != 1 or name != BUNDLE or len(digest) != 64:
        raise ManifestError(f"{BUNDLE_SUM} must be one line '<sha256>  {BUNDLE}'")
    return digest


def check_present(round_dir: Path, results: Path) -> List[str]:
    """Every manifest file under ``results`` with its digest; return the failures."""
    files, _dirs, sums = read_manifest(round_dir)
    failures = []
    for name in files:
        path = results / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        found = sha256_file(path)
        if found != sums[name]:
            failures.append(f"differs: {name} (sha256 {found}, manifest {sums[name]})")
    return failures


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def pack(round_dir: Path, results: Path, out: Path) -> str:
    """Write the bundle; return its sha256. Refuses a source file that disagrees."""
    out = out.resolve()
    if str(out).startswith(str(REPO.resolve()) + os.sep):
        raise InputsRefused(
            f"{out} is inside the repository; the bundle carries host paths and must "
            f"not be written where it could be committed")
    failures = check_present(round_dir, results)
    if failures:
        raise InputsRefused("the archive does not match the manifest: "
                            + "; ".join(failures))
    files, _dirs, sums = read_manifest(round_dir)
    manifest = sums_text(files, sums).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    with open(temporary, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w",
                              format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(_tarinfo(SUMS, len(manifest)), io.BytesIO(manifest))
                for name in files:
                    source = results / name
                    with open(source, "rb") as handle:
                        archive.addfile(_tarinfo(name, source.stat().st_size), handle)
    os.replace(temporary, out)
    digest = sha256_file(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {BUNDLE}\n",
                                                   encoding="utf-8")
    problems = verify_bundle(round_dir, out)
    if problems:
        raise InputsRefused("the bundle just written does not verify: "
                            + "; ".join(problems))
    return digest


def _members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        try:
            name = _safe_relative(member.name)
        except ManifestError as error:
            raise InputsRefused(f"bundle member: {error}") from None
        if not member.isreg():
            raise InputsRefused(f"bundle member {member.name!r} is not a regular file")
        if name in members:
            raise InputsRefused(f"bundle member {name!r} appears twice")
        members[name] = member
    return members


def verify_bundle(round_dir: Path, bundle: Path) -> List[str]:
    """Member set == manifest + ``INPUTS_SHA256SUMS``, and every digest matches."""
    files, _dirs, sums = read_manifest(round_dir)
    problems: List[str] = []
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _members(archive)
        expected = set(files) | {SUMS}
        if set(members) != expected:
            problems.append(f"members differ from the manifest: extra "
                            f"{sorted(set(members) - expected)}, missing "
                            f"{sorted(expected - set(members))}")
        if SUMS in members:
            carried = archive.extractfile(members[SUMS]).read().decode("utf-8")
            if carried != sums_text(files, sums):
                problems.append(f"the bundle's {SUMS} is not the committed manifest")
        for name in files:
            if name not in members:
                continue
            digest = hashlib.sha256()
            handle = archive.extractfile(members[name])
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
            if digest.hexdigest() != sums[name]:
                problems.append(f"differs: {name}")
    return problems


def _symlinked_component(results: Path, name: str) -> str:
    """The first existing component of ``results/name``'s parents that is a link."""
    current = results
    for part in PurePosixPath(name).parts[:-1]:
        current = current / part
        if current.is_symlink():
            return str(current)
        if not current.exists():
            break
    return ""


def unpack(round_dir: Path, bundle: Path, results: Path) -> Tuple[int, int]:
    """Extract absent files; return ``(extracted, already_present)``."""
    if results.is_symlink() or not results.is_dir():
        raise InputsRefused(f"{results} must be a real directory")
    expected = committed_bundle_digest(round_dir)
    found = sha256_file(bundle)
    if found != expected:
        raise InputsRefused(
            f"{bundle} has sha256 {found}; {BUNDLE_SUM} names {expected}. This is "
            f"not the bundle cut for this manifest")
    problems = verify_bundle(round_dir, bundle)
    if problems:
        raise InputsRefused("the bundle does not verify: " + "; ".join(problems))
    files, _dirs, sums = read_manifest(round_dir)
    # EVERY REFUSAL BEFORE THE FIRST WRITE, so a refused unpack leaves results/ as it
    # found it rather than half-filled.
    absent: List[str] = []
    for name in files:
        target = results / name
        link = _symlinked_component(results, name)
        if link or target.is_symlink():
            raise InputsRefused(
                f"{link or target} is a symbolic link; unpacking would write "
                f"through it, into whatever it points at. Run 'check' instead")
        if target.exists():
            if not target.is_file() or sha256_file(target) != sums[name]:
                raise InputsRefused(
                    f"{target} is present and is not the manifest's file; move "
                    f"it out of results/ rather than overwrite it")
            continue
        absent.append(name)
    extracted, present = 0, len(files) - len(absent)
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _members(archive)
        for name in absent:
            target = results / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".unpack.tmp")
            source = archive.extractfile(members[name])
            with open(temporary, "wb") as handle:
                for block in iter(lambda: source.read(CHUNK), b""):
                    handle.write(block)
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
            extracted += 1
    failures = check_present(round_dir, results)
    if failures:
        raise InputsRefused("after unpacking: " + "; ".join(failures))
    return extracted, present


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "pack", "verify", "unpack"))
    parser.add_argument("--round", required=True, type=Path,
                        help="the round directory holding the manifest")
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="the results tree (default: this directory's results/)")
    parser.add_argument("--out", type=Path, help="pack: the bundle to write")
    parser.add_argument("--bundle", type=Path, help="verify, unpack: the bundle")
    arguments = parser.parse_args(list(argv) or None)
    round_dir = arguments.round.resolve()
    try:
        files, dirs, _sums = read_manifest(round_dir)
        if arguments.action == "check":
            failures = check_present(round_dir, arguments.results)
            for failure in failures:
                print(failure, flush=True)
            print(f"{len(files) - len(failures)} of {len(files)} input files verified "
                  f"under {arguments.results.name}/", flush=True)
            return 1 if failures else 0
        if arguments.action == "pack":
            if arguments.out is None:
                parser.error("pack needs --out")
            digest = pack(round_dir, arguments.results, arguments.out)
            print(f"packed {len(files)} files from {len(dirs)} directories; "
                  f"sha256 {digest}; {arguments.out.stat().st_size} bytes", flush=True)
            committed = round_dir / BUNDLE_SUM
            if committed.is_file() and committed_bundle_digest(round_dir) != digest:
                print(f"NOTE: {BUNDLE_SUM} in the round directory names another "
                      f"digest; this bundle is not the one the manifest was cut with",
                      flush=True)
            return 0
        if arguments.bundle is None:
            parser.error(f"{arguments.action} needs --bundle")
        if arguments.action == "verify":
            problems = verify_bundle(round_dir, arguments.bundle)
            # THE SAME DIGEST CHECK ``unpack`` MAKES: a bundle whose members verify but
            # whose digest is not the committed one was not cut for this manifest.
            found = sha256_file(arguments.bundle)
            expected = committed_bundle_digest(round_dir)
            if found != expected:
                problems.append(f"{arguments.bundle} has sha256 {found}; {BUNDLE_SUM} "
                                f"names {expected}")
            for problem in problems:
                print(problem, flush=True)
            print(f"bundle {'does not verify' if problems else 'verified'}: "
                  f"{len(files)} files", flush=True)
            return 1 if problems else 0
        extracted, present = unpack(round_dir, arguments.bundle, arguments.results)
        print(f"unpacked {extracted} files, {present} already present and identical; "
              f"{len(files)} of {len(files)} verified", flush=True)
        return 0
    except ManifestError as error:
        print(f"REFUSED (manifest): {error}", flush=True)
        return 2
    except InputsRefused as error:
        print(f"REFUSED: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

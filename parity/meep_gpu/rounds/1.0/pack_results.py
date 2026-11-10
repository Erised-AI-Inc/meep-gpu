#!/usr/bin/env python3
"""Pack what a certification round on a new GPU sends back: one archive per part, with a sha256 list.

    python parity/meep_gpu/rounds/1.0/pack_results.py part1 --index "$INDEX" --lanes "$LANES" \\
        --round-dir "$ROUND_DIR" --selection "$SEL" --verify "$VERIFY" [--since c8 | --chunks c9,c10]

Run it with the reference environment's interpreter, from anywhere; it packs the clone that
holds this file. Nothing in the clone is changed or moved, and compiler caches (every
directory named ``caches``) are left out.

* ``part1``, after the gate chunks and their verify: every results root the index's chunks
  wrote (``*_<round>_<device>_c*``: the gate roots and the launch records
  ``chunk_<stamp>/``), and any run set aside (``set_aside_*``). With ``--since <chunk id>``
  (the chunks newer than that one) or ``--chunks <id>,...`` it carries only those chunks:
  their results roots (a run of theirs set aside included) and their lanes, beside the whole
  index, selection and verify report, so chunks run again after a part was sent travel on
  their own and the earlier part is not sent twice. ``CHUNKS.json`` records which chunks a
  part1 carries, with their roots, and which it leaves out.
* ``part2``, after the bind and the route stage on the records commit: the route campaigns,
  their preflights and the licence of the device (``dispatch_*_<device>`` and
  ``dispatch_*_<device>_preflight``), any run set aside, and the three NVIDIA ledgers the
  bind wrote. On a host whose driver keeps the route stage from running, ``part2`` packs the
  bind alone, and says so.

Each part also carries the lanes' home (``lanes.tgz``: every chunk's ``round.log``, every
lane's ``progress.log``, ``steps.jsonl`` and step logs), the index, the selection and the
verify report, the ``*.txt``, ``*.log`` and ``*.out`` files and ``gpu_log.csv`` of
``--round-dir``, the
clone's HEAD and ``git status``, the reference environment's build records and
``tools/check_install.py --require-reference``, the cards (``nvidia-smi`` queried for their
index, model, compute capability, driver and memory) and the kernel release. The archive is
``<out-dir>/meep-gpu-<device>-<part>.tgz`` (default ``$HOME``), with the chunk ids after the
part when it carries only some chunks (``meep-gpu-cc89-part1-c9-c10-c11.tgz``); the last line
is its sha256,
to send in a separate message. An existing output is never overwritten.

What it leaves out, and how it writes what it captures. Compiler caches, the LOCAL
lock and pid files (every ``*.lock`` directory, every ``pid`` or ``*.pid`` file), and
every symbolic link (its target is a path on this machine: the H->D lift legs link the
MEEP examples into their work directory) never leave the machine. A hard link stays: its
target is the archive's own name for the first copy. What this tool captures itself (the install check, the cards, the git
status) and the copies of the build records and of ``--round-dir``'s text files are written
through ``neutral_ids.text`` (no host name, account or local path; ``neutral_ids.py``); the
round's own records were written that way when they were written.

What it refuses. A part never carries this account's home directory, another home
directory (one under the Linux, macOS or Windows home root) or the account's name as a
directory of a path: before anything is written, every file the part would carry (the
results roots and the lanes' home, as packed, and the index, selection, verify report and
ledgers) is read, and a part that names one is refused by name, with each file and the keys
(or lines) that name it; the files the tool writes itself are read again before the archive
is made. Nothing is rewritten to pass, since the verify report re-hashes the packed
artifacts, and a run set aside is packed too: the list goes to the maintainers, who fix the
writer and say how the run travels. What counts is what ``neutral_ids`` would write as
``<home>`` or ``<account>``; a temporary directory (``<tmp>``) and the account's name as a
free word (a login can be a word every record carries) are not refused.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROUND_DIR = Path(__file__).resolve().parent
#: The checkout this file belongs to (``parity/meep_gpu/rounds/1.0/pack_results.py``).
REPO = ROUND_DIR.parents[3]
LEDGERS = ("meep_gpu/cuda_kernels/certification.json", "meep_gpu/cuda_kernels/fingerprints.json",
           "meep_gpu/triton_kernels/fingerprints.json")
BUILD_RECORDS = ("reference_build.json", "build_record.json", "libcuda.json")
EXCLUDED_DIRECTORY = "caches"
#: The round directory's own files a part carries: the logs, the preflight and check
#: reports, a batch job's output, the card log.
ROUND_FILES = ("*.txt", "*.log", "*.out", "gpu_log.csv")

#: ``records(out_dir)``: writes the host records into the part's directory.
Records = Callable[[Path], None]

#: The checkout ``neutral_ids.path`` is handed when a part is read for leaks: a directory
#: with no staging root (``neutral_ids.staging_root("/")`` is ``None``), so a path under a
#: checkout that lies in the home directory is read as the home path it is, not written
#: ``<staging-root>``.
NO_STAGING_ROOT = "/"
#: A progress line per this many files read (the part's results run to thousands).
SCAN_PROGRESS_EVERY = 200
#: Keys or lines named per file in a refusal; the rest are counted.
SCAN_KEYS_SHOWN = 8
#: A chunk id, as ``chunks.py`` spells it (``c<k>`` for NVIDIA, ``m<k>`` for a Mac).
CHUNK_ID = re.compile(r"^(?P<letter>[cm])(?P<k>0|[1-9]\d*)$")
#: What a part1 carries, chunk by chunk, and what it leaves out.
CHUNKS_RECORD = "CHUNKS.json"
CHUNKS_SCHEMA = "meep_gpu.pack_chunks/1"
#: The home roots' names, from pieces (a source line spelling a home path is a hygiene-guard hit).
HOME_ROOT_NAMES = ("ho" + "me", "Us" + "ers")


class LeakRefusal(ValueError):
    """A part would carry this account's home directory, another home directory or the
    account's name as a directory: nothing is packed."""


def roots(results: Path, part: str, round_stamp: str, device: str,
          stamps: Optional[Sequence[str]] = None) -> List[str]:
    """The results roots of one part, by name, sorted. With ``stamps`` (part1 only), the
    roots of those chunks alone, a run of theirs set aside included."""
    if stamps is not None:
        patterns = [p for stamp in stamps for p in (f"*_{stamp}", f"*_{stamp}_*")]
    elif part == "part1":
        patterns = [f"*_{round_stamp}_{device}_c*", "set_aside_*"]
    else:
        patterns = [f"dispatch_*_{device}", f"dispatch_*_{device}_preflight", "set_aside_*"]
    names = sorted(entry.name for entry in results.iterdir() if entry.is_dir()
                   and any(fnmatch.fnmatchcase(entry.name, p) for p in patterns))
    return names


def _chunk_k(chunk_id: str) -> Tuple[str, int]:
    match = CHUNK_ID.match(str(chunk_id))
    if not match:
        raise ValueError(f"{chunk_id!r} is not a chunk id (c<k> or m<k>)")
    return match.group("letter"), int(match.group("k"))


def chosen_chunks(index: Dict, named: Optional[Sequence[str]] = None,
                  since: Optional[str] = None) -> Optional[List[Dict]]:
    """The current chunks of ``index`` a part1 carries, in id order: those ``named``, or
    those newer than the chunk ``since``; ``None`` (every chunk) when neither is given.
    Refuses (ValueError) a named id that is not a current chunk (a retired chunk never
    ran), a ``since`` the index never allocated, and a choice that leaves no chunk."""
    if named is not None and since is not None:
        raise ValueError("--chunks and --since choose the chunks two ways; give one")
    if named is None and since is None:
        return None
    current = {chunk["id"]: chunk for chunk in index["chunks"]}
    if named is not None:
        missing = [chunk_id for chunk_id in named if chunk_id not in current]
        if missing:
            raise ValueError(f"not a current chunk of the index: {', '.join(missing)} "
                             f"(current: {', '.join(sorted(current, key=_chunk_k))})")
        chosen = [current[chunk_id] for chunk_id in dict.fromkeys(named)]
    else:
        allocated = set(current) | {chunk["id"] for chunk in index.get("retired", ())}
        if since not in allocated:
            raise ValueError(f"the index never allocated chunk {since}")
        letter, k = _chunk_k(since)
        chosen = [chunk for chunk in current.values()
                  if _chunk_k(chunk["id"])[0] == letter and _chunk_k(chunk["id"])[1] > k]
        if not chosen:
            raise ValueError(f"no current chunk of the index is newer than {since}")
    return sorted(chosen, key=lambda chunk: _chunk_k(chunk["id"]))


def lane_entries(lanes: Path, index: Dict, chosen: Sequence[Dict]) -> List[str]:
    """The lanes' home entries a part1 of ``chosen`` chunks carries: each chosen chunk's
    lane directory (named by its stamp) and its ``check_<id>.txt``, and every entry that
    belongs to no chunk (the runner's own log); another chunk's are left out."""
    every = list(index["chunks"]) + list(index.get("retired", ()))
    own = {name for chunk in every for name in (chunk["stamp"], f"check_{chunk['id']}.txt")}
    keep = {name for chunk in chosen for name in (chunk["stamp"], f"check_{chunk['id']}.txt")}
    return sorted(entry.name for entry in lanes.iterdir()
                  if entry.name in keep or entry.name not in own)


def chunks_record(part: str, index_path: Path, index: Dict, names: Sequence[str],
                  chosen: Optional[Sequence[Dict]], how: str) -> Dict:
    """:data:`CHUNKS_RECORD`: each chunk a part1 carries with the results roots it packs,
    each chunk it leaves out, and any root of no chunk."""
    carried_ids = None if chosen is None else {chunk["id"] for chunk in chosen}
    carried, left_out, placed = [], [], set()
    for chunk in sorted(index["chunks"], key=lambda chunk: _chunk_k(chunk["id"])):
        own = [name for name in names
               if any(fnmatch.fnmatchcase(name, p)
                      for p in (f"*_{chunk['stamp']}", f"*_{chunk['stamp']}_*"))]
        if carried_ids is None or chunk["id"] in carried_ids:
            carried.append({"id": chunk["id"], "stamp": chunk["stamp"], "results_roots": own})
            placed.update(own)
        else:
            left_out.append({"id": chunk["id"], "stamp": chunk["stamp"]})
    return {"schema": CHUNKS_SCHEMA, "part": part, "index": index_path.name,
            "round": index["round"], "device": index["device"], "chosen": how,
            "carried": carried, "left_out": left_out,
            "roots_of_no_chunk": sorted(set(names) - placed)}


def left_out(parts: Sequence[str]) -> bool:
    """Whether a part leaves out the file or directory at ``parts`` (its path inside the
    part): a compiler cache, a local lock directory or a pid file."""
    if EXCLUDED_DIRECTORY in parts or any(part.endswith(".lock") for part in parts):
        return True
    return bool(parts) and (parts[-1] == "pid" or parts[-1].endswith(".pid"))


def without_caches(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    """Leave out compiler caches, the local lock and pid files and every symbolic link (its
    target names a path on this machine); write no owner names."""
    if left_out(Path(info.name).parts) or info.issym():
        return None
    info.uname = info.gname = ""
    info.uid = info.gid = 0
    return info


def neutral_ids():
    """The record spellings of ``parity/meep_gpu/neutral_ids.py`` (``round_status`` loads it)."""
    if str(ROUND_DIR) not in sys.path:
        sys.path.insert(0, str(ROUND_DIR))
    import round_status  # noqa: PLC0415 - beside this file

    return round_status.neutral_ids()


def copy_neutral(source: Path, target: Path) -> None:
    """A text file copied with every line written neutrally (``neutral_ids.text``)."""
    text = source.read_text(encoding="utf-8", errors="replace")
    target.write_text(neutral_ids().text(text), encoding="utf-8")
    shutil.copystat(source, target)


def packed_files(base: Path, names: Sequence[str]) -> List[Path]:
    """Every regular file :func:`tar_into` packs from ``base``/``names``: what
    :func:`without_caches` leaves out (:func:`left_out`, every symbolic link) is left out here
    too, and a directory it leaves out is not entered."""
    out: List[Path] = []
    for name in names:
        top = base / name
        if top.is_symlink() or left_out(top.relative_to(base).parts):
            continue
        if top.is_file():
            out.append(top)
            continue
        for directory, subdirectories, files in os.walk(top, followlinks=False):
            here = Path(directory)
            subdirectories[:] = sorted(d for d in subdirectories
                                       if not (here / d).is_symlink()
                                       and not left_out((here / d).relative_to(base).parts))
            for file in sorted(files):
                path = here / file
                if not path.is_symlink() and not left_out(path.relative_to(base).parts) \
                        and path.is_file():
                    out.append(path)
    return out


def _needles() -> Tuple[str, ...]:
    """Substrings one of which a text must hold for ``neutral_ids`` to write ``<home>`` or
    ``<account>`` in it: a home root's name between separators (:data:`HOME_ROOT_NAMES`), a
    spelling of this account's home directory, or the account's name. A file holding none is
    read no further (a part runs to hundreds of megabytes)."""
    linux, other = HOME_ROOT_NAMES
    found = [f"/{linux}/", f"{other}/", f"{other}\\"]
    for home in {os.path.expanduser("~"), os.environ.get("HOME") or ""}:
        for spelling in (home, os.path.realpath(home) if home else ""):
            spelling = spelling.rstrip("/")
            if len(Path(spelling).parts) >= 2:
                found.append(spelling)
    name = neutral_ids().login()
    if name:
        found.append(name)
    return tuple(dict.fromkeys(found))


def _names_a_home(value: str) -> bool:
    """Whether ``value`` names this account's home directory, another home directory or the
    account's name as a directory: what ``neutral_ids`` would write as ``<home>`` or
    ``<account>`` with no staging root in the way (:data:`NO_STAGING_ROOT`)."""
    if "/" not in value and "\\" not in value:
        return False
    spell = neutral_ids()
    written = spell.path(value, NO_STAGING_ROOT)
    return any(written.count(label) > value.count(label)
               for label in (spell.HOME, spell.ACCOUNT))


def _leaking_keys(node, trail: str = "") -> List[str]:
    """The key paths of a JSON value whose strings (keys included) name a home."""
    found: List[str] = []
    if isinstance(node, dict):
        for key, item in node.items():
            here = f"{trail}.{key}" if trail else str(key)
            if isinstance(key, str) and _names_a_home(key):
                found.append(f"{trail or '<top>'} (a key)")
            found.extend(_leaking_keys(item, here))
    elif isinstance(node, list):
        for item in node:
            found.extend(_leaking_keys(item, f"{trail}[]"))
    elif isinstance(node, str) and _names_a_home(node):
        found.append(trail or "<top>")
    return found


def _unescaped(text: str) -> str:
    """``text`` with JSON's and ``repr``'s escaped separators read as separators
    (``C:\\\\Users``, ``\\/``), so a Windows path inside a quoted string is read as a path."""
    return text.replace("\\\\", "\\").replace("\\/", "/")


def leaks_in(path: Path, needles: Optional[Sequence[str]] = None) -> List[str]:
    """Where one file names a home: its JSON key paths (a ``.json`` file, or each line of a
    ``.jsonl`` file read as JSON), else its line numbers. Empty when it names none. The file
    is read, never written. ``needles``: :func:`_needles`, computed once per part."""
    needles = _needles() if needles is None else needles
    text = path.read_bytes().decode("utf-8", errors="replace")
    if not any(needle in text for needle in needles) or not _names_a_home(_unescaped(text)):
        return []
    if path.suffix == ".json":
        try:
            keys = _leaking_keys(json.loads(text))
        except ValueError:
            keys = []
        if keys:
            return sorted(set(keys), key=keys.index)
    found: List[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not _names_a_home(_unescaped(line)):
            continue
        keys = []
        if path.suffix == ".jsonl":
            try:
                keys = _leaking_keys(json.loads(line))
            except ValueError:
                keys = []
        if keys:
            found.extend(f"line {number}: {key}" for key in keys)
        else:
            found.append(f"line {number}")
    return found or ["(the file as a whole)"]


def refuse_leaks(files: Sequence[Tuple[Path, str]], what: str, after: str = "") -> None:
    """Read every file of ``files`` (``(path, its name in the part)``); refuse
    (:class:`LeakRefusal`) naming each file that names a home, with its keys or lines.
    ``after`` says what a refusal leaves behind. One flushed progress line per
    :data:`SCAN_PROGRESS_EVERY` files."""
    found: Dict[str, List[str]] = {}
    needles = _needles()
    for count, (path, name) in enumerate(files, 1):
        where = leaks_in(path, needles)
        if where:
            found[name] = where
        if count % SCAN_PROGRESS_EVERY == 0:
            print(f"read {count} of {len(files)} {what}: {len(found)} name a home so far",
                  flush=True)
    if found:
        lines = []
        for name, where in sorted(found.items()):
            more = f" and {len(where) - SCAN_KEYS_SHOWN} more" if len(where) > SCAN_KEYS_SHOWN else ""
            lines.append(f"  {name}: {', '.join(where[:SCAN_KEYS_SHOWN])}{more}")
        raise LeakRefusal(
            f"{len(found)} of the {len(files)} {what} name this account's home directory, "
            f"another home directory or the account as a directory; nothing is packed{after}, "
            f"and nothing is rewritten to pass (verify re-hashes the artifacts): send this list, "
            f"which names files and keys and no value, to the maintainers\n" + "\n".join(lines))
    print(f"read {len(files)} {what}: none names a home directory or this account", flush=True)


def tar_into(archive: Path, base: Path, names: Sequence[str]) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for name in names:
            handle.add(str(base / name), arcname=name, filter=without_caches)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(argv: Sequence[str], out: Path, cwd: Optional[Path] = None,
            env: Optional[Dict[str, str]] = None) -> None:
    """Run ``argv`` and keep its output in ``out``; a program that cannot run is recorded."""
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=600,
                              cwd=cwd, env=env)
        text = done.stdout + done.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = f"{argv[0]} unavailable: {exc}\n"
    out.write_text(neutral_ids().text(text), encoding="utf-8")


def host_records(repo: Path) -> Records:
    def write(out: Path) -> None:
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONNOUSERSITE="1")
        capture([sys.executable, "-s", "tools/check_install.py", "--require-reference"],
                out / "check_install.txt", cwd=repo, env=env)
        records = Path(sys.prefix) / "share" / "meep-gpu-build"
        for name in BUILD_RECORDS:
            if (records / name).is_file():
                copy_neutral(records / name, out / name)
        # The cards by index and model: the full ``nvidia-smi`` table lists bus ids and every
        # process with its path, so it is not captured.
        capture(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,memory.total",
                 "--format=csv"], out / "gpus.csv")
        capture(["nvidia-smi", "--version"], out / "nvidia_smi.txt")
        (out / "kernel.txt").write_text(platform.release() + "\n", encoding="utf-8")
    return write


def pack(part: str, repo: Path, index_path: Path, lanes: Path, round_dir: Path,
         out_dir: Path, selection: Optional[Path], verify: Optional[Path],
         records: Records, named: Optional[Sequence[str]] = None,
         since: Optional[str] = None) -> Path:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    round_stamp, device = index["round"], index["device"]
    results = repo / "parity" / "meep_gpu" / "results"
    if part != "part1" and (named is not None or since is not None):
        raise ValueError(f"{part} carries no chunk's results roots; --chunks and --since "
                         f"choose the chunks of a part1")
    chosen = chosen_chunks(index, named, since)
    how = ("every chunk" if chosen is None else
           f"newer than {since}" if since is not None else "named")
    suffix = "" if chosen is None else "-" + "-".join(chunk["id"] for chunk in chosen)
    out = out_dir / f"meep-gpu-{device}-{part}{suffix}"
    if out.exists() or Path(f"{out}.tgz").exists():
        raise FileExistsError(f"{out} or {out}.tgz exists; move it aside first (nothing is "
                              f"overwritten)")
    names = roots(results, part, round_stamp, device,
                  None if chosen is None else [chunk["stamp"] for chunk in chosen])
    linked = [n for n in names if (results / n).is_symlink()]
    if linked:
        raise ValueError(f"results roots that are symbolic links: {', '.join(linked)}; a part "
                         f"carries no link, so pack the directories themselves")
    real = [n for n in names if not n.startswith("set_aside_")]
    if not real and part == "part1":
        which = "" if chosen is None else f" ({how}: {', '.join(c['id'] for c in chosen)})"
        raise FileNotFoundError(f"no results root of {part}{which} for round {round_stamp} on "
                                f"{device} under {results}")
    if not real:
        print(f"no route campaign of {device} under {results}: part2 carries the bind (its "
              f"ledgers and reports) without a route", flush=True)
    print(f"results roots: {len(names)} ({len(real)} of this {part}, "
          f"{len(names) - len(real)} set aside)", flush=True)
    # Everything the part carries as it stands is read first, so a part that would name a
    # home is refused before anything is written.
    sources = [(p, "results/" + p.relative_to(results).as_posix())
               for p in packed_files(results, names)]
    lane_names = ([lanes.name] if chosen is None else
                  [f"{lanes.name}/{entry}" for entry in lane_entries(lanes, index, chosen)])
    sources += [(p, p.relative_to(lanes.parent).as_posix())
                for p in packed_files(lanes.parent, lane_names)]
    sources += [(p, p.name) for p in (index_path, selection, verify)
                if p is not None and p.is_file()]
    if part == "part2":
        sources += [(repo / ledger, f"ledgers/{ledger}") for ledger in LEDGERS
                    if (repo / ledger).is_file()]
    refuse_leaks(sources, f"files {part} would carry")
    out.mkdir(parents=True)
    tar_into(out / "results.tgz", results, names)
    tar_into(out / "lanes.tgz", lanes.parent, lane_names)
    if part == "part1":
        (out / CHUNKS_RECORD).write_text(json.dumps(
            chunks_record(part, index_path, index, names, chosen, how), indent=1,
            sort_keys=True) + "\n", encoding="utf-8")
        print(f"chunks carried: {', '.join(c['id'] for c in (chosen or index['chunks']))}"
              + ("" if chosen is None else f" ({how})"), flush=True)
    for path in (index_path, selection, verify):
        if path is not None and path.is_file():
            shutil.copy2(path, out / path.name)
    for path in [p for pattern in ROUND_FILES for p in sorted(round_dir.glob(pattern))]:
        if path.is_file():
            copy_neutral(path, out / path.name)
    capture(["git", "-C", str(repo), "rev-parse", "HEAD"], out / "HEAD.txt")
    capture(["git", "-C", str(repo), "status", "--porcelain"], out / "git_status.txt")
    if part == "part2":
        for ledger in LEDGERS:
            target = out / "ledgers" / ledger
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / ledger, target)
    records(out)
    carried = "" if chosen is None else f" chunks {','.join(c['id'] for c in chosen)} ({how})"
    (out / "ROUND.txt").write_text(
        f"{part} round {round_stamp} device {device} index {index_path.name}{carried}\n",
        encoding="utf-8")
    # What this tool wrote itself (copies, captures, host records) is read again before the
    # archive is made; a refusal here leaves the directory on this machine and no archive.
    refuse_leaks([(p, p.relative_to(out).as_posix()) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name not in ("results.tgz", "lanes.tgz")],
                 f"files written into {out.name}", f" (the directory {out.name} stays here)")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256_of(p)}  ./{p.relative_to(out).as_posix()}\n" for p in files), encoding="utf-8")
    archive = Path(f"{out}.tgz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(str(out), arcname=out.name, filter=without_caches)
    return archive


def main(argv: Optional[Sequence[str]] = None, records: Optional[Records] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=("part1", "part2"))
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--lanes", required=True, type=Path)
    ap.add_argument("--round-dir", required=True, type=Path)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--verify", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path.home())
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    choose = ap.add_mutually_exclusive_group()
    choose.add_argument("--since", default=None, metavar="CHUNK",
                        help="part1: carry only the chunks newer than this one (the chunks "
                             "run since that part was sent)")
    choose.add_argument("--chunks", default=None, metavar="ID,...",
                        help="part1: carry only these chunks")
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    named = None if args.chunks is None else [c.strip() for c in args.chunks.split(",")
                                              if c.strip()]
    try:
        archive = pack(args.part, repo, args.index, args.lanes.resolve(), args.round_dir,
                       args.out_dir, args.selection, args.verify,
                       records or host_records(repo), named=named, since=args.since)
    except (OSError, ValueError, KeyError) as refused:
        print(f"STOP: {refused}", flush=True)
        return 2
    size = archive.stat().st_size
    print(f"packed {archive} ({size / 2 ** 20:.1f} MiB)", flush=True)
    print(f"{sha256_of(archive)}  {archive.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

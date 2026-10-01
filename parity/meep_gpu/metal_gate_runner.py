#!/usr/bin/env python3
"""Run a Metal byte gate and weld its artifact to the bytes it imported.

Metal gates are deliberately separate processes: an MPS device gate must not
share a process with a CPU-MEEP oracle.  That separation used to leave a
provenance hole: gates could report a hand-curated source hash, or no decisive
machine-readable release verdict at all.  This runner is the required public
entry point for every ``gate_metal_*.py`` script and the standalone Metal
expansion probes that license complex families.  It imports one measurement, runs its
``main`` function, then records the actual imported Python files by reading each
module's ``__file__`` *after* the gate has run.

The output artifact has two normalized fields:

``release``
    ``released`` is true only for a zero exit and a successful automatic weld;
    ``reasons`` explains any refusal, failed gate, missing artifact, or tree
    mismatch.

``source_sha256``
    Absolute source paths mapped to SHA-256 values read from the files imported
    by this process.  It is intentionally not a list inferred from the current
    working directory or copied from a staging tree.

The artifact directory also contains ``source_sha256.txt``.  It is an
append-only-in-scope campaign manifest: a later gate run in the same directory
refuses release if a path already has a different digest, preventing a single
campaign from silently mixing trees.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
EXIT_CANNOT_CERTIFY = 75
SOURCE_MANIFEST_ENV = "MEEP_GPU_GATE_SOURCE_MANIFEST"

# Every Metal gate except the complete-step arbiter writes directly to the path
# supplied as ``--out``.  The arbiter predates the common gate convention and uses
# that argument as its result directory, with a stable file inside it.  Keep the
# exception declarative here rather than asking the runner to guess whether a path
# is intended to become a directory after the gate has executed.
#
# The Metal driver-route gate is the second entry, for a different reason: its
# ``--out`` is a CAMPAIGN LEG directory holding ``gate.json`` beside
# ``cases.jsonl``, ``controls.jsonl``, ``summary.json`` and ``progress.log``, because
# progress reporting requires each row on disk as it lands and a single artifact file cannot
# carry that.
DIRECTORY_OUTPUT_ARTIFACTS = {"gate_metal_whole_step.py": "whole_step.json",
                              "gate_dispatch_metal_route.py": "gate.json"}


def discover_gates(directory: Path = HERE) -> tuple[Path, ...]:
    """Return every Metal gate that direct execution must route through here."""
    return tuple(sorted(directory.glob("gate_metal_*.py")))


def discover_dispatch_gates(directory: Path = HERE) -> tuple[Path, ...]:
    """Metal gates deliberately named OUTSIDE the ``gate_metal_*`` glob.

    NAMED ONE BY ONE, and the explicitness is the point rather than a shortcoming.
    ``gate_dispatch_metal_route.py`` measures the DISPATCH LADDER rather than a
    kernel family, and the ``gate_metal_*`` stem is read by three other mechanisms
    that would each misread it:

    * ``recut_metal_gates.sh``'s fleet loop globs that stem and would launch this
      multi-leg campaign with a single gate's arguments, counting the result as a
      failed gate in a routine re-cut;
    * ``test_metal_weld_contract``'s family derivation takes families FROM those
      stems and would invent a family ``dispatch_metal_route``, then demand a
      per-family weld for something that is not a family;
    * ``test_gate_provenance``'s pattern sweeps them too.

    Renaming to fit the glob would trip all three; staying outside costs exactly
    this function and one line in :func:`main`'s allowlist. The runner's contract is
    unchanged either way — every gate below still runs through
    :func:`run_current_measurement` and gets the same source-welded release.
    """
    names = ("gate_dispatch_metal_route.py",)
    return tuple(directory / name for name in names if (directory / name).is_file())


def discover_probes(directory: Path = HERE) -> tuple[Path, ...]:
    """Return the standalone MPS expansion measurements used by Metal arms."""
    names = (
        "probe_metal_beta_expansion.py",
        "probe_metal_folded_complex_expansion.py",
        "probe_metal_cylindrical_complex.py",
    )
    return tuple(directory / name for name in names if (directory / name).is_file())


def sha256_of(path: Path) -> str:
    """Return the content digest for one actual source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_python_source(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path.suffix == ".py" and path.is_file()


def runtime_source_sha256(*, gate_path: Path,
                          modules: Iterable[ModuleType] | None = None,
                          root: Path | None = None) -> dict[str, str]:
    """Hash the gate and every package source actually imported by the process.

    ``module.__file__`` is consulted here, after import.  This is deliberately
    stricter than following imports from the working directory: a staged tree or
    a path shadowing mistake must show up in the artifact's path-and-digest pair.
    """
    root = (API_ROOT if root is None else root).resolve()
    paths = {gate_path.resolve()}
    for module in tuple(sys.modules.values()) if modules is None else modules:
        location = getattr(module, "__file__", None)
        if not location:
            continue
        try:
            path = Path(location).resolve()
        except OSError:
            continue
        if _is_python_source(path, root):
            paths.add(path)
    return {str(path): sha256_of(path) for path in sorted(paths)}


def _atomic_json_write(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_artifact(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"gate wrote no artifact at {path}"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"artifact is unreadable ({exc!r})"]
    if not isinstance(value, dict):
        return {}, ["artifact root is not a JSON object"]
    return value, []


def _manifest_path(artifact_path: Path) -> Path:
    explicit = os.environ.get(SOURCE_MANIFEST_ENV)
    if explicit:
        return Path(explicit).resolve()
    return artifact_path.parent / "source_sha256.txt"


def _read_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.exists():
        return {}, []
    rows: dict[str, str] = {}
    reasons: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"source manifest is unreadable ({exc!r})"]
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            digest, source = line.split("\t", 1)
        except ValueError:
            reasons.append(f"source manifest line {line_number} is malformed")
            continue
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            reasons.append(f"source manifest line {line_number} has an invalid digest")
        elif source in rows and rows[source] != digest:
            reasons.append(f"source manifest has conflicting records for {source}")
        else:
            rows[source] = digest
    return rows, reasons


def _write_manifest(path: Path, sources: Mapping[str, str]) -> None:
    content = "".join(f"{digest}\t{source}\n" for source, digest in sorted(sources.items()))
    _atomic_text_write(content, path)


def _existing_manifest_reasons(manifest: Mapping[str, str],
                               sources: Mapping[str, str]) -> list[str]:
    return [
        f"campaign source manifest disagrees for {source}"
        for source, digest in sources.items()
        if source in manifest and manifest[source] != digest
    ]


def check_artifact_weld(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Rehash an artifact's runtime sources against the live tree.

    This is the verifier a release check uses; it deliberately has no access to
    an old staged digest.  A changed source, a path outside the package tree, or
    a malformed value is a failed weld rather than a condition to hand-wave.
    """
    recorded = payload.get("source_sha256")
    if not isinstance(recorded, dict) or not recorded:
        return ["artifact has no normalized source_sha256 record"]
    root = (API_ROOT if root is None else root).resolve()
    reasons: list[str] = []
    for name, digest in recorded.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            reasons.append("source_sha256 contains a non-string path or digest")
            continue
        path = Path(name)
        if not path.is_absolute() or not _is_python_source(path, root):
            reasons.append(f"recorded source is not a Python file below API root: {name}")
            continue
        if sha256_of(path) != digest:
            reasons.append(f"recorded digest disagrees with live source: {name}")
    return reasons


def finalize_artifact(*, artifact_path: Path, gate_path: Path, exit_code: int,
                      runner_reasons: Sequence[str] = (),
                      modules: Iterable[ModuleType] | None = None,
                      root: Path | None = None) -> dict[str, Any]:
    """Add release and runtime-source evidence, then verify it without copying.

    The gate's legacy/hand-curated ``source_sha256`` field is preserved under
    ``gate_reported_source_sha256`` for diagnosis.  It cannot serve as the
    release weld because it need not name the source path the process imported.
    """
    root = API_ROOT if root is None else root
    payload, reasons = _read_artifact(artifact_path)
    reasons.extend(runner_reasons)
    if "source_sha256" in payload:
        payload["gate_reported_source_sha256"] = payload["source_sha256"]

    sources = runtime_source_sha256(gate_path=gate_path, modules=modules, root=root)
    payload["source_sha256"] = sources
    manifest_path = _manifest_path(artifact_path)
    manifest, manifest_reasons = _read_manifest(manifest_path)
    reasons.extend(manifest_reasons)
    reasons.extend(_existing_manifest_reasons(manifest, sources))

    weld_reasons = check_artifact_weld(payload, root=root)
    reasons.extend(weld_reasons)
    if not _existing_manifest_reasons(manifest, sources) and not manifest_reasons:
        _write_manifest(manifest_path, {**manifest, **sources})

    if exit_code == EXIT_CANNOT_CERTIFY:
        reasons.append("gate could not certify on this host (exit 75)")
    elif exit_code != 0:
        reasons.append(f"gate returned exit status {exit_code}")
    payload["weld"] = {
        "passed": not weld_reasons and not manifest_reasons
                  and not _existing_manifest_reasons(manifest, sources),
        "source_count": len(sources),
        "manifest": str(manifest_path),
    }
    payload["release"] = {
        "released": exit_code == 0 and not reasons,
        "reasons": reasons,
    }
    _atomic_json_write(payload, artifact_path)
    print(
        f"[release] released={payload['release']['released']} "
        f"sources={len(sources)} artifact={artifact_path}",
        flush=True,
    )
    return payload


def finalize_related_artifact(*, artifact_path: Path,
                              parent_artifact: Path) -> dict[str, Any]:
    """Weld a companion measurement emitted by an already-welded gate.

    The complex gate writes its expansion probe in the same process as its primary
    JSON result.  The parent artifact is therefore the only honest runtime-source
    record for the companion: it is verified here before being inherited, rather
    than copied by a caller or reconstructed from the working directory.
    """
    parent, reasons = _read_artifact(parent_artifact)
    if not parent.get("release", {}).get("released"):
        reasons.append(f"parent artifact is not released: {parent_artifact}")
    parent_sources = parent.get("source_sha256")
    if not isinstance(parent_sources, dict) or not parent_sources:
        reasons.append(f"parent artifact has no normalized source map: {parent_artifact}")
        parent_sources = {}
    else:
        reasons.extend(check_artifact_weld(parent))

    payload, artifact_reasons = _read_artifact(artifact_path)
    reasons.extend(artifact_reasons)
    if "source_sha256" in payload:
        payload["measurement_reported_source_sha256"] = payload["source_sha256"]
    payload["source_sha256"] = parent_sources
    payload["parent_artifact"] = str(parent_artifact.resolve())
    weld_reasons = check_artifact_weld(payload)
    reasons.extend(weld_reasons)
    payload["weld"] = {
        "passed": not reasons,
        "source_count": len(parent_sources),
        "inherited_from": str(parent_artifact.resolve()),
    }
    payload["release"] = {"released": not reasons, "reasons": reasons}
    _atomic_json_write(payload, artifact_path)
    print(f"[release] related released={payload['release']['released']} "
          f"sources={len(parent_sources)} artifact={artifact_path}", flush=True)
    return payload


def _load_gate(gate_path: Path) -> ModuleType:
    name = f"_metal_gate_runtime_{gate_path.stem}_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(name, gate_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import gate at {gate_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _call_main(module: ModuleType, gate_args: Sequence[str]) -> int:
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError(f"{module.__file__} does not define callable main()")
    parameters = tuple(inspect.signature(main).parameters.values())
    if parameters:
        result = main(list(gate_args))
    else:
        previous_argv = sys.argv
        try:
            sys.argv = [str(module.__file__), *gate_args]
            result = main()
        finally:
            sys.argv = previous_argv
    return 0 if result is None else int(result)


def artifact_path_from_args(gate_path: Path, gate_args: Sequence[str]) -> Path:
    """Read the one standardized ``--out`` argument accepted by every gate."""
    output: Path | None = None
    for index, value in enumerate(gate_args):
        if value == "--out" and index + 1 < len(gate_args):
            output = Path(gate_args[index + 1]).resolve()
            break
        if value.startswith("--out="):
            output = Path(value.partition("=")[2]).resolve()
            break
    if output is not None:
        child = DIRECTORY_OUTPUT_ARTIFACTS.get(gate_path.name)
        return output / child if child else output
    raise ValueError("every Metal gate run must provide --out ARTIFACT.json")


def run_current_measurement(gate_path: str | Path, gate_args: Sequence[str]) -> int:
    """Execute and finalize one gate or standalone expansion measurement.

    Every direct gate calls the gate-named wrapper below under its ``__main__``
    guards.  The campaign shell also invokes this generic entry for the three
    standalone probes, so the JSON that licenses an expansion arm receives the same
    source-welded release contract instead of becoming an ambient input.
    """
    path = Path(gate_path).resolve()
    artifact = artifact_path_from_args(path, gate_args)
    for import_path in (str(API_ROOT), str(HERE)):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    reasons: list[str] = []
    exit_code = 1
    try:
        module = _load_gate(path)
        exit_code = _call_main(module, gate_args)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        if exc.code not in (None, 0):
            reasons.append(f"gate raised SystemExit({exc.code!r})")
    except Exception as exc:  # noqa: BLE001 - must preserve a final artifact
        reasons.append(f"gate raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
    payload = finalize_artifact(
        artifact_path=artifact,
        gate_path=path,
        exit_code=exit_code,
        runner_reasons=reasons,
    )
    if exit_code == EXIT_CANNOT_CERTIFY:
        return EXIT_CANNOT_CERTIFY
    return 0 if payload["release"]["released"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", nargs="?", type=Path,
                        help="Metal gate script to execute")
    parser.add_argument("gate_args", nargs=argparse.REMAINDER,
                        help="arguments forwarded to the gate; begin with --")
    parser.add_argument("--verify", type=Path, metavar="ARTIFACT",
                        help="verify an existing artifact's source weld")
    parser.add_argument("--weld-related", type=Path, metavar="ARTIFACT",
                        help="weld a gate-produced companion measurement")
    parser.add_argument("--parent-artifact", type=Path, metavar="GATE.json",
                        help="released gate artifact that produced --weld-related")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        payload, reasons = _read_artifact(arguments.verify)
        reasons.extend(check_artifact_weld(payload))
        if reasons:
            for reason in reasons:
                print(f"[weld] FAILED: {reason}", flush=True)
            return 1
        print(f"[weld] PASS: {arguments.verify}", flush=True)
        return 0
    if arguments.weld_related:
        if not arguments.parent_artifact:
            parser.error("--weld-related requires --parent-artifact")
        payload = finalize_related_artifact(
            artifact_path=arguments.weld_related.resolve(),
            parent_artifact=arguments.parent_artifact.resolve(),
        )
        return 0 if payload["release"]["released"] else 1
    if arguments.gate is None:
        parser.error("GATE is required unless --verify is used")
    allowed = (set(discover_gates()) | set(discover_probes())
               | set(discover_dispatch_gates()))
    if arguments.gate.resolve() not in allowed:
        parser.error("GATE must be a checked-in gate_metal_*.py, a Metal expansion "
                     "probe, or one of the named dispatch gates "
                     f"({', '.join(p.name for p in discover_dispatch_gates())})")
    forwarded = tuple(arguments.gate_args)
    if forwarded[:1] == ("--",):
        forwarded = forwarded[1:]
    return run_current_measurement(arguments.gate, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

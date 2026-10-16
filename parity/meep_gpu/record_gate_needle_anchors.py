"""Record, and check, the bytes every gate-harness needle hands the compiler.

A source-mutation leg's verdict is a function of the mutated device text it compiles,
so two runs that rewrite the same kernels into the same bytes measure the same thing.
This tool derives those bytes HOST-SIDE, with no device, for every needle table the
harness builds with ``gate_needles.gate_table`` (and for the two harness tables kept
in another shape: the ``beta_hd_gate_common`` device-mutation table of
``gate_special_kz_fused_hd_pair.py``, and ``no_pml_ref.CURL_GROUPINGS``), and pins
them in ``gate_needle_anchors.json``:

* per table, the device strings its legs rewrite (the gate's own ``originals``), by
  digest;
* per needle, whether it is borrowed from the probe's shared default or owned by the
  gate, its identity (a ``Needle``'s rewrite, or a function's qualified name), and,
  for every target it matches, the site count and the digest of the mutated text.

``parity/meep_gpu/test_mutation_anchors.py`` runs the check on every test run. A
difference names the table, the needle and the target, and means one of three
things, each a real change: a kernel the gate mutates was edited (its mutation legs
now compile different bytes and are owed a re-run), a needle was re-spelled (a lane's
inversion, or a drift), or a needle stopped matching (anchor rot). Re-mint with
``--write`` in the same change that made it, so the record's diff shows which.

The CUDA kernel modules are imported under a stub ``cupy`` module that exists only in
this process: their device strings are module-level constants that read nothing from
CuPy, so the bytes are the ones a device host compiles. The check therefore runs in a
child process when called from the test suite, and never installs the stub there.

Progress: one flushed line per table (the progress-reporting rule).

Usage::

    python record_gate_needle_anchors.py            # check against the record
    python record_gate_needle_anchors.py --write    # re-mint the record
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import types
from typing import Any, Callable, Dict, List, Mapping, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
RECORD = os.path.join(HERE, "gate_needle_anchors.json")
STARTED = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - STARTED:6.2f} s] {message}", flush=True)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def install_cupy_stub() -> None:
    """A ``cupy`` whose every attribute is inert, so the kernel modules import."""
    class _Inert:
        def __getattr__(self, name):
            return _Inert()

        def __call__(self, *args, **kwargs):
            return _Inert()

    for name in ("cupy", "cupy.cuda", "cupy.cuda.compiler", "cupy.cuda.runtime",
                 "cupyx"):
        module = types.ModuleType(name)
        module.__getattr__ = lambda attribute: _Inert()  # type: ignore[attr-defined]
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module


def kernels(stem: str):
    return importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")


def attributes(stem: str, *names: str) -> Dict[str, str]:
    module = kernels(stem)
    return {f"{stem}.{name}": getattr(module, name) for name in names}


def accessor(stem: str, function: str, collection: str) -> Dict[str, str]:
    module = kernels(stem)
    return {f"{stem}.{function}({name})": getattr(module, function)(name)
            for name in getattr(module, collection)}


def _curl_pair() -> Dict[str, str]:
    return attributes("step_curl_kernels", "_step_B_pml_real_kernel_code",
                      "_step_D_pml_real_kernel_code")


def _constitutive_pair() -> Dict[str, str]:
    return attributes("constitutive_kernels", "_update_H_pml_real_kernel_code",
                      "_update_E_pml_real_kernel_code")


#: EVERY TABLE BUILT WITH ``gate_needles.gate_table``: (gate id, module, attribute,
#: the device strings its source-mutation legs rewrite). The targets transcribe each
#: gate's own ``originals``; a gate id built anywhere under ``parity/meep_gpu`` and
#: missing here fails ``test_mutation_anchors.py`` by name.
TABLES: Tuple[Tuple[str, str, str, Callable[[], Dict[str, str]]], ...] = (
    ("probe_fused_kernel_bit_identity:hand", "probe_fused_kernel_bit_identity",
     "HAND_NEEDLES", lambda: {**_curl_pair(), **_constitutive_pair()}),
    ("gate_cuda_bfast", "gate_cuda_bfast", "SHARED_NEEDLES",
     lambda: accessor("bfast_curl", "kernel_source", "BFAST_KERNELS")),
    ("gate_cuda_conductive", "gate_cuda_conductive", "SOURCE_MUTATIONS_ALL",
     lambda: accessor("conductive_kernels", "kernel_template", "CONDUCTIVE_KERNELS")),
    ("gate_cuda_cylindrical_constitutive", "gate_cuda_cylindrical_constitutive",
     "SHARED_NEEDLES", _constitutive_pair),
    ("gate_cuda_cylindrical_real", "gate_cuda_cylindrical_real", "SHARED_NEEDLES",
     lambda: attributes("cylindrical_kernels", "_cyl_step_B_pml_real_kernel_code",
                        "_cyl_step_D_pml_real_kernel_code")),
    ("gate_cuda_folded_constitutive", "gate_cuda_folded_constitutive",
     "SHARED_NEEDLES", _constitutive_pair),
    ("gate_cuda_folded_curl", "gate_cuda_folded_curl", "SHARED_NEEDLES", _curl_pair),
    ("gate_cuda_no_pml_curl", "gate_cuda_no_pml_curl", "SOURCE_MUTATIONS_ALL",
     lambda: accessor("no_pml_curl", "kernel_source", "NO_PML_KERNELS")),
    ("gate_cuda_nonlinear", "gate_cuda_nonlinear", "SHARED_NEEDLES",
     lambda: {**attributes("nonlinear_constitutive",
                           "_update_E_pml_real_nonlinear_kernel_code"),
              **attributes("constitutive_kernels", "_update_H_pml_real_kernel_code")}),
    ("gate_cuda_nonlinear_curl_null", "gate_cuda_nonlinear_curl_null",
     "SOURCE_MUTATIONS", _curl_pair),
    ("gate_cuda_special_kz", "gate_cuda_special_kz", "SHARED_NEEDLES",
     lambda: accessor("special_kz_curl", "kernel_source", "SPECIAL_KZ_KERNELS")),
)


def derive_table(gate: str, module_name: str, attribute: str,
                 targets: Callable[[], Dict[str, str]]) -> Dict[str, Any]:
    gate_needles = importlib.import_module("gate_needles")
    table = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(table, gate_needles.NeedleTable) or table.gate != gate:
        raise SystemExit(f"{module_name}.{attribute} is not the needle table of "
                         f"{gate!r}: {table!r}")
    try:
        sources = targets()
    except Exception as exc:  # noqa: BLE001 - a target the tree cannot derive is recorded
        # A TARGET THE TREE CANNOT DERIVE IS RECORDED BY NAME, NOT RAISED. A gate whose
        # kernels do not derive on this tree (a lifter not yet re-cut onto a moved base
        # text, which holds its derivation by name rather than failing at import) has
        # no bytes for its needles to rewrite. The table is kept with every needle at
        # zero sites and the reason beside it, so the record still lists the gate and
        # test_mutation_anchors.py names each of its needles as matching nothing.
        refusal = f"{type(exc).__name__}: {exc}"
        return {"module": module_name, "attribute": attribute, "refused": refusal,
                "targets": {},
                "needles": {name: {"borrowed": name in table.borrowed,
                                   "identity": gate_needles.transform_spec(table[name]),
                                   "sites": {}, "mutated_sha256": {}}
                            for name in table}}
    needles: Dict[str, Any] = {}
    for name in table:
        transform = table[name]
        sites: Dict[str, int] = {}
        mutated: Dict[str, str] = {}
        for label, text in sorted(sources.items()):
            new_text, count = transform(text)
            if count:
                sites[label] = int(count)
                mutated[label] = sha(new_text)
            elif new_text != text:
                raise SystemExit(f"{gate}: {name} rewrote {label} while reporting "
                                 f"0 sites")
        needles[name] = {"borrowed": name in table.borrowed,
                         "identity": gate_needles.transform_spec(transform),
                         "sites": sites, "mutated_sha256": mutated}
    return {"module": module_name, "attribute": attribute,
            "targets": {label: sha(text) for label, text in sorted(sources.items())},
            "needles": needles}


def derive_beta_hd() -> Dict[str, Any]:
    """The real beta H->D weld gate's device-mutation table, through its own kit."""
    kit = importlib.import_module("beta_hd_gate_common")
    gate = importlib.import_module("gate_special_kz_fused_hd_pair")
    adapter = gate.SpecialKzAdapter()
    try:
        target = adapter.kernel_source()
    except Exception as exc:  # noqa: BLE001 - recorded by name, as derive_table does
        refusal = f"{type(exc).__name__}: {exc}"
        return {"module": "gate_special_kz_fused_hd_pair",
                "attribute": "DEVICE_MUTATIONS", "refused": refusal, "target": None,
                "sites": {}, "unresolved": [f"the weld does not derive: {refusal}"],
                "mutated_sha256": {}}
    resolved = kit.mutation_needles_resolve(adapter)
    mutated = {}
    for name in adapter.device_mutations:
        # A needle that does not land is RECORDED BY NAME, not raised: the kit refuses
        # it with SystemExit (the gate's own rule), and the record names it beside the
        # resolver's unresolved list rather than stopping every other table's check.
        try:
            text = kit.mutated_source(adapter, name)
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - recorded by name
            mutated[name] = f"not landed: {exc}"
            continue
        mutated[name] = None if text is None else sha(text)
    return {"module": "gate_special_kz_fused_hd_pair",
            "attribute": "DEVICE_MUTATIONS",
            "target": sha(target),
            "sites": {row["needle"]: row["sites"] for row in resolved["needles"]},
            "unresolved": resolved["unresolved"],
            "mutated_sha256": mutated}


def derive_no_pml_groupings() -> Dict[str, Any]:
    """Every row of ``no_pml_ref.CURL_GROUPINGS`` on one seeded NumPy fixture."""
    import numpy as np  # noqa: PLC0415

    ref = importlib.import_module("no_pml_ref")
    rng = np.random.default_rng(ref.SEED + 41)
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "By", "Bz", "Dx", "Dy", "Dz")
    base = {n: rng.uniform(-1.0, 1.0, (7, 6, 5)).astype(np.float32) for n in names}
    out: Dict[str, Any] = {}
    for grouping in ref.CURL_GROUPINGS:
        rows = {}
        for sub_step, source_names, target_names in (
                ("step_B", ("Ex", "Ey", "Ez"), ("Bx", "By", "Bz")),
                ("step_D", ("Hx", "Hy", "Hz"), ("Dx", "Dy", "Dz"))):
            for boundaries in (("periodic",) * 3, ("metallic",) * 3,
                               ("metallic", "periodic", "metallic")):
                for dtdx in (np.float32(0.5), np.float32(0.35)):
                    sources = {n: base[n].copy() for n in source_names}
                    state = {n: base[n].copy() for n in target_names}
                    ref.reference_plain_step(np, sources, state, dtdx, sub_step,
                                             boundaries, grouping)
                    key = f"{sub_step}|{','.join(boundaries)}|{float(dtdx)!r}"
                    rows[key] = hashlib.sha256(b"".join(
                        np.ascontiguousarray(state[n]).tobytes()
                        for n in target_names)).hexdigest()
        out[grouping] = rows
    return out


def derive() -> Dict[str, Any]:
    for path in (HERE, REPO_API):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("MEEP_GPU_DISPATCH", "0")
    install_cupy_stub()
    record: Dict[str, Any] = {
        "_what_this_is": (
            "The device text every gate-harness needle hands the compiler, derived "
            "host-side by parity/meep_gpu/record_gate_needle_anchors.py: per table, "
            "the strings its legs rewrite and, per needle, its identity, whether it "
            "is borrowed from the probe's shared default, and per matched target the "
            "site count and the mutated text's sha256. Checked on every test run by "
            "test_mutation_anchors.py; re-mint with --write in the change that moves "
            "it."),
        "tables": {}}
    for index, (gate, module_name, attribute, targets) in enumerate(TABLES, 1):
        entry = derive_table(gate, module_name, attribute, targets)
        record["tables"][gate] = entry
        armed = sum(1 for needle in entry["needles"].values() if needle["sites"])
        log(f"table {index}/{len(TABLES)} {gate}: {armed} of {len(entry['needles'])} "
            f"needles match its {len(entry['targets'])} target(s)"
            + (f"; REFUSED: {entry['refused'][:200]}" if "refused" in entry else ""))
    record["beta_hd_device_mutations"] = derive_beta_hd()
    log(f"beta H->D device mutations: "
        f"{len(record['beta_hd_device_mutations']['sites'])} needles, unresolved "
        f"{record['beta_hd_device_mutations']['unresolved']}")
    record["no_pml_ref_curl_groupings"] = derive_no_pml_groupings()
    log(f"no_pml_ref groupings: {sorted(record['no_pml_ref_curl_groupings'])}")
    return record


def differences(recorded: Any, live: Any, path: str = "") -> List[str]:
    if isinstance(recorded, Mapping) and isinstance(live, Mapping):
        out: List[str] = []
        for key in sorted(set(recorded) | set(live)):
            where = f"{path}/{key}"
            if key not in live:
                out.append(f"{where}: in the record, not derived")
            elif key not in recorded:
                out.append(f"{where}: derived, not in the record")
            else:
                out.extend(differences(recorded[key], live[key], where))
        return out
    if recorded != live:
        return [f"{path}: recorded {recorded!r}, derived {live!r}"]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="re-mint gate_needle_anchors.json from this tree")
    parser.add_argument("--record", default=RECORD)
    args = parser.parse_args(argv)
    live = derive()
    if args.write:
        with open(args.record, "w", encoding="utf-8") as handle:
            json.dump(live, handle, indent=1, sort_keys=True)
            handle.write("\n")
        log(f"wrote {args.record}")
        return 0
    if not os.path.exists(args.record):
        log(f"no record at {args.record}; mint it with --write")
        return 1
    with open(args.record, encoding="utf-8") as handle:
        recorded = json.load(handle)
    found = differences(recorded, live)
    for line in found:
        print(f"DIFFERS {line}", flush=True)
    log(f"{len(found)} difference(s) against {os.path.basename(args.record)}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

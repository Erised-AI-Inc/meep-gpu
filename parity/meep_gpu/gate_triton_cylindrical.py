"""Bit-identity gate for the Triton cylindrical (Dcyl) kernels, against the array path.

THE REFERENCE IS THE ARRAY PATH ITSELF. Every leg builds a twin pair of real
``Grid``/``Fields``/``PML`` triples on one array module, seeds both with the same
full-volume state, steps one with :mod:`meep_gpu.stepping` (the aligned oracle: MEEP
1.33.0's forms, single precision, no contraction) and the other with the Triton plan, and
compares every stored word of every field and auxiliary volume as ``uint32``. Nothing is
transcribed here, so the gate cannot drift from the oracle it measures.

THE FAMILIES. This file carries the shared machinery and the REAL m = 0 family
(:mod:`meep_gpu.triton_kernels.cylindrical_triton`: the column-serial radial scan and the
curl kernel; the root constitutive plan on a Dcyl grid). The complex family
(:mod:`.cylindrical_complex`) is :mod:`gate_triton_cylindrical_complex`, which imports
this machinery and adds its cases and its mutations.

THE SEEDINGS. ``random`` (every word nonzero), ``planted`` (random with a sparse
lattice of -0, +0 and subnormal words) and ``lattice`` (random with every fourth z
column replaced by an alternating -0/+0 pattern that differs per volume). A random state
exercises every chunk-boundary plane at the first launch; the signed-zero classes a
random state never reaches -- a radial sum's first row, a dropped zero addend, a plain
constitutive select -- are reached by the lattice, which is why several must-catch
mutations name it.

Legs (``--legs``, comma-separated; every leg needs a CUDA device with CuPy and Triton
unless run through a host stand-in by a caller that passes ``xp``):

* ``synthetic``    -- every case x sub-step x seeding: ONE launch of the plan against
  ``stepping.step_B`` / ``step_D``, bytewise;
* ``guard``        -- the same rows with contraction ENABLED; it must diverge on at least
  one row (the control bites), and is recorded, not required identical;
* ``multi_step``   -- the curl plans in the driver's order with the array path's
  constitutive sub-steps between them, compared after every sub-step, for ``--steps``;
* ``constitutive`` -- :func:`cylindrical_triton.plan_cylindrical_constitutive` against
  ``stepping.update_H`` / ``update_E``, through MEEP's start of H;
* ``mutations``    -- each mutant kernel text must diverge (must-catch) or stay identical
  (declared null), on the seedings it names.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=N python -u gate_triton_cylindrical.py --out <dir>/gate.json

Every row prints one flushed line as it lands and the JSON artifact is rewritten after
every row (the progress-reporting rule). The verdict is ``verdict.pass``.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import platform
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

FIELD_NAMES: Tuple[str, ...] = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                                "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
AUX_PREFIXES: Tuple[str, ...] = ("fu_", "f_w_")
SEEDINGS: Tuple[str, ...] = ("random", "planted", "lattice")
SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
SEED = 20261008
DEFAULT_STEPS = 64

#: The real m = 0 family's cases: an r-high + z layer (two r chunks, three z chunks),
#: an r layer alone, a z layer alone (one r chunk), a thin layer, at Courant numbers of
#: which three are not float32 words (the axis add's double product is then visible).
REAL_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "m0_rz_c05", "m": 0, "complex": False, "cell": (2.0, 0.0, 4.0),
     "thickness": {"x": (0, 5), "z": 5}, "courant": 0.5},
    {"name": "m0_rz_c037", "m": 0, "complex": False, "cell": (2.0, 0.0, 4.0),
     "thickness": {"x": (0, 5), "z": 5}, "courant": 0.37},
    {"name": "m0_r_c0314", "m": 0, "complex": False, "cell": (2.4, 0.0, 3.0),
     "thickness": {"x": (0, 4)}, "courant": 0.3141592653589793},
    {"name": "m0_z_c045", "m": 0, "complex": False, "cell": (2.0, 0.0, 4.0),
     "thickness": {"z": 3}, "courant": 0.4472135954999579},
    {"name": "m0_thin_c03", "m": 0, "complex": False, "cell": (1.3, 0.0, 2.1),
     "thickness": {"x": (0, 2), "z": 2}, "courant": 0.3},
)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# State: twin Fields, seeded, compared word for word
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any], xp: Any) -> Tuple[Any, Any]:
    """A real Dcyl ``Fields``/``PML`` pair for one case on array module ``xp``."""
    kwargs = {}
    if case.get("accurate") is not None:
        kwargs["accurate_fields_near_cylorigin"] = bool(case["accurate"])
    grid = Grid(resolution=case.get("resolution", 10), cell_size=tuple(case["cell"]),
                cylindrical=True, m=case["m"], courant=case["courant"],
                boundaries={"z": case.get("z", "metallic")}, xp=xp, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=bool(case.get("complex", case["m"] != 0)))
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=case["thickness"])


def state_names(fields: Any) -> List[str]:
    """Every allocated field and auxiliary volume, in a fixed order."""
    names = []
    for name in FIELD_NAMES:
        if getattr(fields, name, None) is not None:
            names.append(name)
        for prefix in AUX_PREFIXES:
            if getattr(fields, prefix + name, None) is not None:
                names.append(prefix + name)
    return names


def _words(array: Any) -> Any:
    return array.view(array.real.dtype) if array.dtype.kind == "c" else array


def seed_state(fields: Any, seed: int, seeding: str) -> None:
    """Seed every volume on the host, then copy it to the volume's array module."""
    if seeding not in SEEDINGS:
        raise ValueError(f"seeding must be one of {SEEDINGS}, got {seeding!r}")
    rng = np.random.default_rng(seed)
    xp = fields.grid.xp
    for name in state_names(fields):
        array = getattr(fields, name)
        shape = _words(array).shape
        values = rng.standard_normal(shape).astype(np.float32)
        values[values == 0] = np.float32(0.5)
        if seeding == "planted":
            flat = values.reshape(-1)
            picks = rng.choice(flat.size, size=max(3, flat.size // 25), replace=False)
            third = len(picks) // 3
            flat[picks[:third]] = np.float32(-0.0)
            flat[picks[third:2 * third]] = np.float32(0.0)
            flat[picks[2 * third:]] = np.float32(1e-40) * np.sign(flat[picks[2 * third:]])
        elif seeding == "lattice":
            # Pairs of adjacent z CELLS share a pattern value (both parts of a complex
            # cell alike), so a stencil reads -0 against -0 as well as against +0.
            r = np.arange(shape[0]).reshape(-1, 1, 1)
            cell = np.arange(shape[2]).reshape(1, 1, -1) // (2 if array.dtype.kind == "c" else 1)
            phase = sum(ord(ch) for ch in name)
            pattern = np.where((r * 7 + (cell // 2) * 3 + phase) % 4 < 2,
                               np.float32(-0.0), np.float32(0.0))
            columns = np.broadcast_to((cell // 2) % 4 == 1, shape)
            values[columns] = np.broadcast_to(pattern, shape)[columns]
        _words(array)[...] = xp.asarray(values)


def copy_state(source: Any, target: Any) -> None:
    for name in state_names(source):
        getattr(target, name)[...] = getattr(source, name)
    target.h_separated = source.h_separated


def host_words(array: Any) -> np.ndarray:
    words = _words(array)
    get = getattr(words, "get", None)
    return np.ascontiguousarray(get() if callable(get) else np.asarray(words)).view(np.uint32)


def compare(left: Any, right: Any) -> Dict[str, Any]:
    """``{"differing", "words", "volumes", "first"}`` over every state volume."""
    differing = words = 0
    volumes: Dict[str, int] = {}
    first = None
    for name in state_names(left):
        a, b = host_words(getattr(left, name)), host_words(getattr(right, name))
        mismatch = a != b
        n = int(np.count_nonzero(mismatch))
        words += a.size
        if n:
            volumes[name] = n
            differing += n
            if first is None:
                index = tuple(int(v) for v in np.argwhere(mismatch)[0])
                first = {"volume": name, "index": index,
                         "reference": f"0x{int(a[index]):08x}",
                         "kernel": f"0x{int(b[index]):08x}"}
    return {"differing": differing, "words": words, "volumes": volumes, "first": first}


def flip_read(reference: Any, candidate: Any) -> int:
    """The planted-flip control: one low bit of one candidate word flipped, the two states
    compared, the bit restored. Returns how many more words the compare reports than it did
    before the flip; anything but exactly 1 means the compare is not reading the state."""
    before = compare(reference, candidate)["differing"]
    name = state_names(candidate)[0]
    flat = _words(getattr(candidate, name)).reshape(-1).view("uint32")
    middle = flat.size // 2
    flat[middle] ^= 1
    after = compare(reference, candidate)["differing"]
    flat[middle] ^= 1
    return after - before


def moved_words(before: Any, after: Any) -> int:
    return sum(int(np.count_nonzero(host_words(getattr(before, n)) != host_words(getattr(after, n))))
               for n in state_names(before))


def signed_zero_words(fields: Any) -> int:
    """How many stored words are -0: the class a lattice must reach to be non-vacuous."""
    return sum(int(np.count_nonzero(host_words(getattr(fields, n)) == np.uint32(0x80000000)))
               for n in state_names(fields))


# ---------------------------------------------------------------------------
# Plans, built on the harness route from the twin's own arrays
# ---------------------------------------------------------------------------

def real_curl_plan(fields: Any, pml: Any, sub_step: str, kernel: Any = None,
                   scan_kernel: Any = None) -> Any:
    """:func:`cylindrical_triton.plan_cylindrical_curl_from_arrays` on ``fields``."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl
    from meep_gpu.triton_kernels.launch import chunk_bounds_for

    spec = cyl.SUB_STEPS[sub_step]
    arrays = {n: getattr(fields, n) for n in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: getattr(fields, "fu_" + n) for n in spec["targets"]})
    flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{spec['suffix']}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}
    grid = fields.grid
    return cyl.plan_cylindrical_curl_from_arrays(
        sub_step, arrays, flat, grid.dt / grid.dx, float(grid.courant), grid.xp,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=cyl.radial_scan_starts(fields, pml, sub_step, None),
        kernel=kernel, scan_kernel=scan_kernel)


def real_constitutive_plan(fields: Any, pml: Any, side: str, kernel: Any = None) -> Any:
    """The root ``ConstitutivePlan`` bound as :func:`plan_cylindrical_constitutive` binds it."""
    from meep_gpu.triton_kernels import coverage
    from meep_gpu.triton_kernels.launch import ConstitutivePlan, chunk_bounds_for

    spec = coverage.CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, 256,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["aux"]],
        [getattr(fields, n) for n in spec["sources"]],
        ([fields.inverse_epsilon_for(n) for n in spec["targets"]] if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, chunk_bounds=chunk_bounds_for(fields, pml, None), h_state=fields)


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def one_launch(case: Dict[str, Any], sub_step: str, seeding: str, xp: Any,
               planner: Callable[..., Any], seed: int, guard: Optional[bool] = None,
               kernel: Any = None, scan_kernel: Any = None) -> Dict[str, Any]:
    """One sub-step from a seeded state: the plan against the array path, bytewise."""
    reference, pml = build(case, xp)
    candidate, pml_k = build(case, xp)
    seed_state(reference, seed, seeding)
    copy_state(reference, candidate)
    before, _ = build(case, xp)
    copy_state(reference, before)
    getattr(stepping, sub_step)(reference, pml)
    planner(candidate, pml_k, sub_step, kernel=kernel, scan_kernel=scan_kernel).run(guard)
    row = compare(reference, candidate)
    row.update({"case": case["name"], "sub_step": sub_step, "seeding": seeding,
                "moved": moved_words(before, reference),
                "signed_zero_words": signed_zero_words(before),
                "flip_read": flip_read(reference, candidate)})
    return row


def run_synthetic(cases: Sequence[Dict[str, Any]], xp: Any, planner: Callable[..., Any],
                  results: Dict[str, Any], out_path: Optional[str],
                  key: str = "synthetic", guard: Optional[bool] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total = len(cases) * len(SUB_STEPS) * len(SEEDINGS)
    for index, case in enumerate(cases):
        for sub_step in SUB_STEPS:
            for seeding in SEEDINGS:
                row = one_launch(case, sub_step, seeding, xp, planner,
                                 SEED + 101 * index, guard=guard)
                rows.append(row)
                log(f"  {key} {len(rows)}/{total} {case['name']} {sub_step} {seeding}: "
                    f"{row['differing']}/{row['words']} words differ, moved {row['moved']}")
                results[key] = summarize(rows)
                save(results, out_path)
    return results[key]


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"rows": list(rows), "total": len(rows),
            "identical": sum(1 for r in rows if r["differing"] == 0),
            "differing_words": sum(r["differing"] for r in rows),
            "words": sum(r["words"] for r in rows),
            "all_moved": all(r.get("moved", 1) > 0 for r in rows),
            "flip_read_one": all(r.get("flip_read", 1) == 1 for r in rows)}


def run_multi_step(cases: Sequence[Dict[str, Any]], xp: Any, planner: Callable[..., Any],
                   steps: int, results: Dict[str, Any], out_path: Optional[str]
                   ) -> Dict[str, Any]:
    """The curl plans in the driver's order, the array path's constitutive between them."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        reference, pml = build(case, xp)
        candidate, pml_k = build(case, xp)
        seed_state(reference, SEED + 7 + index, "planted")
        copy_state(reference, candidate)
        plans = {s: planner(candidate, pml_k, s) for s in SUB_STEPS}
        differing = words = 0
        first = None
        t0 = time.time()
        for step in range(steps):
            for sub_step, update in (("step_B", "update_H"), ("step_D", "update_E")):
                getattr(stepping, sub_step)(reference, pml)
                plans[sub_step].run()
                row = compare(reference, candidate)
                differing += row["differing"]
                words += row["words"]
                if row["differing"] and first is None:
                    first = {"step": step, "sub_step": sub_step, **row}
                getattr(stepping, update)(reference, pml)
                getattr(stepping, update)(candidate, pml_k)
            if (step + 1) % 16 == 0 or step + 1 == steps:
                log(f"  multi_step {case['name']} step {step + 1}/{steps}: "
                    f"{differing}/{words} words differ ({time.time() - t0:.1f} s)")
        rows.append({"case": case["name"], "steps": steps, "differing": differing,
                     "words": words, "first": first,
                     "flip_read": flip_read(reference, candidate)})
        results["multi_step"] = {"rows": rows, "total": len(rows),
                                 "identical": sum(1 for r in rows if r["differing"] == 0)}
        save(results, out_path)
    return results["multi_step"]


def run_engine(cases: Sequence[Dict[str, Any]], xp: Any,
               planner: Callable[..., Any], refusal: Callable[..., Sequence[str]],
               results: Dict[str, Any], out_path: Optional[str]) -> Dict[str, Any]:
    """The ENGINE route: each curl plan built from the engine's own objects through its
    coverage predicate (the route ``launch.plan_step`` takes), then compared with
    ``stepping`` from a random full-volume state. A predicate that refuses an admitted
    case is a failed row, named by its reasons, not a skipped one."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        for sub_step in SUB_STEPS:
            reference, pml = build(case, xp)
            candidate, pml_k = build(case, xp)
            seed_state(reference, SEED + 211 * index, "random")
            copy_state(reference, candidate)
            before, _ = build(case, xp)
            copy_state(reference, before)
            plan = planner(candidate, pml_k, sub_step)
            if plan is None:
                row = {"case": case["name"], "sub_step": sub_step, "differing": None,
                       "words": 0, "refused": list(refusal(candidate, pml_k, sub_step))}
            else:
                getattr(stepping, sub_step)(reference, pml)
                plan.run()
                row = compare(reference, candidate)
                row.update({"case": case["name"], "sub_step": sub_step,
                            "plan": repr(plan), "moved": moved_words(before, reference),
                            "flip_read": flip_read(reference, candidate)})
            rows.append(row)
            log(f"  engine {len(rows)}/{len(cases) * len(SUB_STEPS)} {case['name']} "
                f"{sub_step}: {row.get('differing')}/{row['words']} words differ"
                f"{' REFUSED ' + str(row['refused'])[:200] if 'refused' in row else ''}")
            refused = [r for r in rows if "refused" in r]
            results["engine"] = {
                "rows": rows, "total": len(rows), "refused": len(refused),
                "identical": sum(1 for r in rows if r.get("differing") == 0),
                "all_moved": all(r.get("moved", 1) > 0 for r in rows if "refused" not in r),
                "flip_read_one": all(r.get("flip_read", 1) == 1 for r in rows)}
            save(results, out_path)
    return results["engine"]


def run_constitutive(cases: Sequence[Dict[str, Any]], xp: Any,
                     planner: Callable[..., Any], results: Dict[str, Any],
                     out_path: Optional[str]) -> Dict[str, Any]:
    """The constitutive plan against ``update_H`` / ``update_E``, start of H included."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        for side, separated in (("H", False), ("H", True), ("E", True)):
            for seeding in SEEDINGS:
                reference, pml = build(case, xp)
                candidate, pml_k = build(case, xp)
                seed_state(reference, SEED + 31 + index, seeding)
                reference.h_separated = separated
                copy_state(reference, candidate)
                getattr(stepping, "update_" + side)(reference, pml)
                planner(candidate, pml_k, side).run()
                row = compare(reference, candidate)
                row.update({"case": case["name"], "side": side,
                            "h_separated_before": separated, "seeding": seeding,
                            "h_flag_agrees": reference.h_separated == candidate.h_separated,
                            "flip_read": flip_read(reference, candidate)})
                rows.append(row)
                log(f"  constitutive {case['name']} {side} separated={separated} "
                    f"{seeding}: {row['differing']}/{row['words']}")
        results["constitutive"] = summarize(rows)
        save(results, out_path)
    return results["constitutive"]


# ---------------------------------------------------------------------------
# Mutations: kernel text edited one way, compiled, launched
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def shipped_source(function: Any) -> str:
    """One shipped ``@triton.jit`` function's source, dedented."""
    return textwrap.dedent(inspect.getsource(getattr(function, "fn", function)))


def apply_edits(source: str, edits: Sequence[Tuple[str, str]]) -> str:
    """Apply text edits; every needle must be present, or the mutation is DISARMED."""
    for old, new in edits:
        if old not in source:
            raise AssertionError(f"mutation needle not found in the shipped text: {old!r}")
        source = source.replace(old, new)
    return source


def compile_mutant(source: str, name: str, module: Any) -> Any:
    """Compile a mutated copy of one shipped kernel from a REAL file in ``module``'s scope.

    ``triton.jit`` reads its function's source from the file it was defined in, so the
    mutant is written to a temporary module whose globals are ``module``'s.
    """
    header = ("import importlib\n"
              f"globals().update({{k: v for k, v in vars(importlib.import_module("
              f"{module.__name__!r})).items() if not k.startswith('__')}})\n"
              "import triton\nimport triton.language as tl\n\n")
    if not source.lstrip().startswith("@"):
        raise AssertionError(f"{name}'s shipped source carries no decorator")
    body = source
    handle = tempfile.NamedTemporaryFile("w", suffix=f"_{name}.py", delete=False)
    handle.write(header + body)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(f"cyl_mutant_{len(_TEMPORARY)}", handle.name)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return getattr(loaded, name)


#: The real family's mutations: (name, kernel, edits, expected, seedings). ``kernel`` is
#: "scan" or "curl"; ``expected`` is "caught" or "null". The Appendix A inversion (the
#: previous pairwise curl grouping), the section 7.2 battery (whole-grid recurrence,
#: membership a row off, lattices swapped, fu stored on every cell, the predicate on
#: coefficient identity), the scan's own forms, the axis rules, and the near ghost put
#: back (a declared null: unobservable by construction).
#:
#: A planted division is spelled ``tl.math.div_rn``, never a plain ``/``. Triton lowers a
#: float ``/`` to ``div.full.f32`` as inline asm, which neither subnormal policy reaches:
#: under ``flush`` the policy's PTX audit refuses that mutant at compile, so the row
#: would never measure. ``div_rn`` is round-to-nearest under both policies
#: (``div.rn.f32`` under ``keep``, ``div.rn.ftz.f32`` under ``flush``), the spelling any
#: flush-uniform kernel must use. ``s2`` is that mutant: the shipped scan multiplies by
#: ``rinv``, as MEEP does.
REAL_MUTATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...], str, Tuple[str, ...]], ...] = (
    ("a1_pairwise_curl_grouping", "curl",
     (("curl1 = curl_stencil(a_z, a, c, c_r, dtdx, pz, pr)",
       "curl1 = dtdx * ((a_z - a) + (c - c_r))"),), "caught", SEEDINGS),
    ("a1_phi_term_kept", "curl",
     (("curl0 = curl_stencil(c_p, c, b, b_z, dtdx, pp, pz)",
       "curl0 = dtdx * (((c_p - c) + b) - b_z)"),), "caught", ("lattice",)),
    ("b1_whole_grid_recurrence", "curl",
     (("du0 = record_in(cb, lattice + 8, k)", "du0 = k >= 0"),), "caught", SEEDINGS),
    ("b2_membership_one_row_off", "curl",
     (("du1 = record_in(cb, lattice + 0, i)", "du1 = record_in(cb, lattice + 0, i + 1)"),),
     "caught", SEEDINGS),
    ("b3_lattices_swapped", "curl",
     (("        lattice = RECORD_HALF_INTEGER\n", "        lattice = 0\n"),), "caught",
     SEEDINGS),
    ("b4_fu_stored_on_every_cell", "curl",
     (("tl.store(u1 + idx, n1, mask=live & du1)", "tl.store(u1 + idx, n1, mask=live)"),),
     "caught", SEEDINGS),
    ("b5_predicate_on_coefficient_identity", "curl",
     (("du0 = record_in(cb, lattice + 8, k)", "du0 = km_z != 1.0"),), "caught", SEEDINGS),
    ("n1_fu_load_unconditional", "curl",
     (("p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)",
       "p1 = tl.load(u1 + idx, mask=live, other=0.0)"),), "null", SEEDINGS),
    ("c1_axis_add_in_float32", "curl",
     (("tl.cast(tl.cast(b, tl.float64) * tl.load(courant + COURANT_FOUR),",
       "tl.cast(b * tl.cast(tl.load(courant + COURANT_FOUR), tl.float32),"),),
     "caught", SEEDINGS),
    ("c2_br_axis_zero_dropped", "curl",
     (("v0 = tl.where(at_r, 0.0, v0)", "v0 = v0"),), "caught", SEEDINGS),
    ("c3_dp_axis_zero_dropped", "curl",
     (("v1 = tl.where(at_r, 0.0, v1)", "v1 = v1"),), "caught", SEEDINGS),
    ("c4_ownership_mask_dropped", "curl",
     (("curl0 = tl.where(at_r, 0.0, curl0)", "curl0 = curl0"),), "caught", SEEDINGS),
    ("g1_near_ghost_restored", "curl",
     (("vr = live & (si >= 0) & (si < nx)", "vr = live & (si < nx)"),
      ("o_r = si * nyz + j * nz + k", "o_r = tl.where(si < 0, 0, si) * nyz + j * nz + k")),
     "null", SEEDINGS),
    ("s1_one_segment", "scan",
     (("for t in range(0, n_segs):", "for t in range(0, 1):"),), "caught", SEEDINGS),
    ("s2_divide_not_multiply", "scan",
     (("tl.load(rinv + (g - 1)) * (here - prev)",
       "tl.math.div_rn(here - prev, tl.load(w + g))"),),
     "caught", SEEDINGS),
    ("s3_first_row_not_from_zero", "scan",
     (("after = total + tl.load(rinv + (g - 1)) * (here - prev)",
       "after = tl.where(g == s + 1, tl.load(rinv + (g - 1)) * (here - prev), "
       "total + tl.load(rinv + (g - 1)) * (here - prev))"),), "caught", ("lattice",)),
    ("s4_no_wall_row", "scan",
     (("mask=live & (g < rows_stored), other=0.0)",
       "mask=live & (g < rows_stored), other=1.0)"),), "caught", SEEDINGS),
    ("s5_increment_distributed", "scan",
     (("after = total + tl.load(rinv + (g - 1)) * (here - prev)",
       "after = total + (tl.load(rinv + (g - 1)) * here - tl.load(rinv + (g - 1)) * prev)"),),
     "caught", SEEDINGS),
)


def run_mutations(cases: Sequence[Dict[str, Any]], xp: Any, planner: Callable[..., Any],
                  table: Sequence[Tuple[str, str, Tuple[Tuple[str, str], ...], str,
                                        Tuple[str, ...]]],
                  kernels: Dict[str, Tuple[Any, Any]], compiler: Callable[..., Any],
                  results: Dict[str, Any], out_path: Optional[str],
                  key: str = "mutations",
                  case_for: Optional[Callable[[str], Dict[str, Any]]] = None
                  ) -> Dict[str, Any]:
    """Each mutant over the named seedings, both sub-steps; caught means any word differs.

    ``kernels`` maps "scan"/"curl" to ``(shipped function, module)``. ``compiler`` turns a
    mutated source into a launchable kernel (:func:`compile_mutant` on a device).
    """
    rows: List[Dict[str, Any]] = []
    for name, which, edits, expected, seedings in table:
        function, module = kernels[which]
        mutant = compiler(apply_edits(shipped_source(function), edits),
                          getattr(function, "fn", function).__name__, module)
        case = case_for(name) if case_for is not None else cases[1 % len(cases)]
        differing = words = 0
        for seeding in seedings:
            for sub_step in SUB_STEPS:
                row = one_launch(case, sub_step, seeding, xp, planner, SEED + 3,
                                 kernel=mutant if which != "scan" else None,
                                 scan_kernel=mutant if which == "scan" else None)
                differing += row["differing"]
                words += row["words"]
        verdict = "caught" if differing else "null"
        rows.append({"mutation": name, "kernel": which, "case": case["name"],
                     "seedings": list(seedings), "differing": differing, "words": words,
                     "expected": expected, "observed": verdict,
                     "as_expected": verdict == expected})
        log(f"  {key} {len(rows)}/{len(table)} {name}: {differing}/{words} -> {verdict} "
            f"(expected {expected})")
        results[key] = {"rows": rows, "total": len(rows),
                        "as_expected": sum(1 for r in rows if r["as_expected"])}
        save(results, out_path)
    return results[key]


# ---------------------------------------------------------------------------
# Artifact and verdict
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: Optional[str]) -> None:
    if not out_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)
    os.replace(tmp, out_path)


def environment() -> Dict[str, Any]:
    """Every version that decides this gate's answer."""
    info: Dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__,
                            "host": platform.node()}
    for name in ("cupy", "triton"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            info[name] = f"unavailable: {exc!r}"
    info["subnormal_policy"] = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY", "")
    try:
        from meep_gpu import subnormal_policy  # noqa: PLC0415
        info["subnormal_policy_stamp"] = subnormal_policy.policy_stamp()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        info["subnormal_policy_stamp"] = f"unavailable: {exc!r}"
    info["dispatch"] = os.environ.get("MEEP_GPU_DISPATCH", "")
    return info


def policy_end_stamp() -> Dict[str, Any]:
    """The subnormal-policy stamp read at the END of the run.

    Its ``triton`` block carries the PTX audit's counters over every kernel this run
    compiled (``ptx_compiles``, ``ptx_audited``, ``ptx_with_ftz``, ``ptx_violations``),
    which the start-of-run stamp in :func:`environment` cannot: nothing has compiled
    yet when that one is taken. Recorded, not judged; a violation already raised at
    compile.
    """
    try:
        from meep_gpu import subnormal_policy  # noqa: PLC0415
        return subnormal_policy.policy_stamp()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {"unavailable": repr(exc)}


def verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """The gate's reading of its own rows. Every clause is stated, none inferred."""
    out: Dict[str, Any] = {"clauses": {}}
    for key in ("synthetic", "constitutive"):
        leg = results.get(key)
        if leg:
            out["clauses"][f"{key}_all_identical"] = (
                leg["identical"] == leg["total"] > 0 and leg["all_moved"])
            out["clauses"][f"{key}_flip_control_reads_one"] = leg.get("flip_read_one", True)
            out[key] = f"{leg['identical']}/{leg['total']} rows identical"
    guard = results.get("guard")
    if guard:
        out["clauses"]["guard_control_bites"] = guard["identical"] < guard["total"]
        out["guard"] = f"{guard['identical']}/{guard['total']} rows identical with contraction on"
    multi = results.get("multi_step")
    if multi:
        out["clauses"]["multi_step_all_identical"] = multi["identical"] == multi["total"] > 0
        out["clauses"]["multi_step_flip_control_reads_one"] = all(
            r.get("flip_read", 1) == 1 for r in multi["rows"])
        out["multi_step"] = f"{multi['identical']}/{multi['total']} cases identical"
    engine = results.get("engine")
    if engine:
        out["clauses"]["engine_all_identical"] = (
            engine["identical"] == engine["total"] > 0 and engine["all_moved"]
            and engine["flip_read_one"])
        out["engine"] = (f"{engine['identical']}/{engine['total']} rows identical, "
                         f"{engine['refused']} refused")
    for key in ("mutations",):
        leg = results.get(key)
        if leg:
            out["clauses"][f"{key}_all_as_expected"] = leg["as_expected"] == leg["total"] > 0
            out[key] = f"{leg['as_expected']}/{leg['total']} as expected"
    out["pass"] = bool(out["clauses"]) and all(out["clauses"].values())
    return out


def device_xp() -> Any:
    """CuPy on a CUDA device, or a named refusal: this gate measures device bytes."""
    try:
        import cupy  # noqa: PLC0415
        cupy.cuda.runtime.getDeviceCount()
        import triton  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001 - reported by name
        raise SystemExit(f"this gate needs a CUDA device with CuPy and Triton: {exc!r}")
    # Both sides of every compare run under one subnormal policy, installed on the host,
    # CuPy and Triton before anything compiles: the one the environment names, else the
    # Triton table's certified ``keep`` (planted states carry subnormal words).
    from meep_gpu import subnormal_policy  # noqa: PLC0415
    subnormal_policy.install_subnormal_policy(
        os.environ.get(subnormal_policy.POLICY_ENV) or "keep")
    return cupy


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--legs",
                        default="synthetic,guard,multi_step,engine,constitutive,mutations")
    args = parser.parse_args(argv)
    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    xp = device_xp()
    from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: PLC0415

    results: Dict[str, Any] = {
        "what": "bit-identity gate for the Triton cylindrical m = 0 kernels against the "
                "array path",
        "environment": environment(), "legs": legs, "steps": args.steps,
        "cases": [case["name"] for case in REAL_CASES],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(results, args.out)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)
    if "synthetic" in legs:
        log("leg: synthetic")
        run_synthetic(REAL_CASES, xp, real_curl_plan, results, args.out)
    if "guard" in legs:
        log("leg: guard (contraction enabled; must diverge somewhere)")
        run_synthetic(REAL_CASES, xp, real_curl_plan, results, args.out, key="guard",
                      guard=True)
    if "multi_step" in legs:
        log("leg: multi_step")
        run_multi_step(REAL_CASES, xp, real_curl_plan, args.steps, results, args.out)
    if "engine" in legs:
        log("leg: engine")
        run_engine(REAL_CASES, xp,
                   lambda f, p, s: cyl.plan_cylindrical_curl(f, p, s),
                   lambda f, p, s: cyl.cylindrical_curl_coverage(f, p).reasons,
                   results, args.out)
    if "constitutive" in legs:
        log("leg: constitutive")
        run_constitutive(REAL_CASES[:3], xp, real_constitutive_plan, results, args.out)
    if "mutations" in legs:
        log("leg: mutations")
        run_mutations(REAL_CASES, xp, real_curl_plan, REAL_MUTATIONS,
                      {"scan": (cyl.cyl_radial_difference, cyl),
                       "curl": (cyl.cyl_pml_curl_step, cyl)},
                      compile_mutant, results, args.out)
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["subnormal_policy_end"] = policy_end_stamp()
    results["verdict"] = verdict(results)
    save(results, args.out)
    _stamp_provenance(results)
    log(f"VERDICT: {json.dumps(results['verdict'], default=str)}")
    return 0 if results["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

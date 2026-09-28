"""Device gate for the NONLINEAR fused magnetic B/H pair on Triton.

``step_B`` welded into ``update_H`` on a chi2/chi3 run — the arm
:mod:`meep_gpu.triton_kernels.nonlinear_fused_magnetic_pair` adds, and the Triton
twin of the Metal family
``metal_kernels/nonlinear_fused_magnetic_pair.py`` certified by
``gate_metal_below_the_cut_fused_pairs.py``.

===========================================================================
WHAT IS NEW HERE, AND WHAT IS NOT
===========================================================================

**NO ARITHMETIC IS NEW.** The kernel a covered configuration launches is
``kernels.fused_curl_constitutive_B`` — the SHIPPED ordinary fused pair, already
device-certified (``fingerprints.json``'s ``bit_identity_gate.legs.
fused_curl_constitutive_B``: ``single_launch_guarded_vs_array_path 60/60``) — through
the SHIPPED ``launch.FusedPairPlan`` built by the SHIPPED ``launch.plan_fused_pair``.
The product adds one predicate, one plan builder and one checkable identity.

**WHAT IS THEREFORE ACTUALLY AT RISK, and what every leg below is aimed at:**

1. **The admission may be WIDER than what it claims to inherit.** The weld is
   supposed to be the conjunction of the two shipped nonlinear SPINE arms plus the
   seam clauses. If it admits a configuration either spine arm refuses, it has
   silently widened a certified half. ``equivalence`` (no device) sweeps that, and
   every device leg re-asserts it on the real CuPy objects it stepped.
2. **The scope view may leak.** :class:`~meep_gpu.triton_kernels.nonlinear_fused_magnetic_pair._LinearScopeView`
   inverts exactly one attribute read. If it masked anything else, the shipped
   predicate would be evaluating a fiction. ``scope_view`` (no device) enumerates
   every attribute the view answers differently from the real ``Fields``, and the
   set must be ``{'has_nonlinearity'}``.
3. **The chi2/chi3 run may not in fact step this seam identically.** The whole
   licence is "``step_B`` and ``update_H`` never read chi2 or chi3". That is the
   claim the device legs measure, as uint32 words, against BOTH the array path and
   the two separately certified spine products, at one launch and at ~60.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs, in order (driver.py:3282-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

On the configurations this arm admits the two fill passes are DEAD — both return at
their first line without ``grid.has_symmetry()`` (stepping.py:1481-1482, :1565-1566)
and every clause set in the predicate refuses a fold — so the launch spans
``step_B``, ``zero_metal_B`` and ``update_H``. The magnetic source slot is REAL and
is refused by name; an ELECTRIC source is injected in the D/E half and does not
disqualify this pair. Both corpus rows source electrically.

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-20_closed/`` at
the B->H cell (``nonlinear run PML``, ``nonlinear run``): TWO seam-instances,
``examples:3rd-harm-1d.py`` and ``tests:Test3rdHarm1d.test_3rd_harm_1d``, both with
``in_seam_source == false`` and ``live_in_seam_passes == ['zero_metal_B']``. The
``corpus_admission`` leg re-derives that from the census rather than citing it.

===========================================================================
WHAT THIS GATE REFUSES TO INFER
===========================================================================

* **A mutation that does not parse is not a mutation**, and a mutation whose lines
  the scored case never reaches is not a mutation either. Every rewrite is compiled
  with ``ast.parse`` before any device is involved, and every entry declares the
  CASE whose boundary/wall configuration enters the branch it rewrites — the
  dead-branch failure this project has already paid for. A rewrite that DELETES an
  ``if`` is scored on a case where that ``if`` was TRUE, so the deletion is not
  itself the change being measured.
* **Bytes alone cannot prove the fused path ran.** Every launch is counted through
  a proxy that owns the kernel object, and a leg reporting identity with zero fused
  launches is a silent fallback, not a pass.
* **A no-op agreeing with a no-op is trivially identical.** Every leg carries a
  NON-VACUITY floor: the compared state must MOVE.
* **The reference must really be the array path.** ``FdtdDriver.step`` consults
  ``fastpath`` BEFORE the module-level sub-step functions this harness substitutes,
  and the two nonlinear SPINE arms are WIRED — so on this very family a reference
  driver left alone would step on Triton. The fast path is disabled explicitly on
  every driver, and the per-route call counter must show the reference reaching the
  array path once per step for all five passes or the leg fails.
* **A false ``released``.** Only a device run writes ``release.released``; the
  ``--no-device`` path writes ``False`` with its reason.

POLICY, stamped: ``num_warps=launch.FUSED_DEFAULT_NUM_WARPS`` (1),
``BLOCK=kernels.DEFAULT_BLOCK`` (256), ``enable_fp_fusion=kernels.ENABLE_FP_FUSION``
(False). BOTH float32 subnormal policies are gated, one process each
(``--subnormal-policy keep|flush``); the policy is installed with ``strict=True``
before the first device compile, because a process that asked to keep and quietly
did not is a process whose bytes mean nothing.

Usage::

    # laptop: every leg that needs neither CUDA nor Triton
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_nonlinear_fused_magnetic_pair.py --no-device

    # device, one process per policy
    python -u parity/meep_gpu/probe_triton_nonlinear_fused_magnetic_pair.py \\
        --subnormal-policy keep --out results/<fresh>/gate_keep.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PRODUCT_MODULE = "meep_gpu.triton_kernels.nonlinear_fused_magnetic_pair"

#: 3rd-harm-1d.py's own chi3 decade. The corpus rows sweep around it.
CHI3_CORPUS = 0.06

#: The device cases. ``(name, cell, boundaries, steps)``.
#:
#: EVERY CELL IS 3-D. A 2-D cell has an INVARIANT axis, and declaring an invariant
#: axis metallic is refused by ``Grid`` by name (grid.py:842-877) — a gate that
#: tripped over that would die in the constructor instead of measuring a wall.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int], ...] = (
    # THE CORPUS GEOMETRY: periodic in x and y, METALLIC in z, which is exactly
    # what both nonlinear rows declare (`boundary_kinds ['periodic','periodic',
    # 'metallic']`, `metallic [false,false,true]`). ZM_Z fires; ZM_X/ZM_Y do not.
    ("corpus_wall_z", (1.0, 1.0, 1.2),
     {"x": "periodic", "y": "periodic", "z": "metallic"}, 60),
    # EVERY WALL LIVE: all three ZM planes and all three METALLIC ghost arms are
    # entered. This is the mutation case, for that reason.
    ("wall_xyz", (1.0, 1.1, 1.2), "metallic", 60),
    # NO WALL AT ALL: ZM_* all compile to False and every ghost arm is the
    # periodic wrap. The complement of the case above.
    ("periodic_xyz", (1.0, 1.1, 1.2), "periodic", 60),
    # ONE LAUNCH. The ~60-step legs above measure accumulation; this one measures
    # the single step, where a defect has nowhere to hide behind a later pass.
    ("corpus_wall_z_single_launch", (1.0, 1.0, 1.2),
     {"x": "periodic", "y": "periodic", "z": "metallic"}, 1),
)

#: The case every kernel mutation is scored on unless it names another. ``wall_xyz``
#: is the only case that enters BOTH the metallic ghost arms and all three ``ZM_*``
#: clears, so a rewrite of either is live there.
MUTATION_CASE_INDEX = 1

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported with its launch evidence rather than as a pass.
MUTATION_STEPS = 4

#: The value classes, and what each is for.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero_lattice", "subnormal_band")

#: The driver call sites one launch of this plan performs, in driver order.
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: Names that MUST appear in the scanned state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: Read-only material inputs under the names the scan really finds. A leg that
#: changes one of these measured something other than the step; a NAME THAT MATCHES
#: NOTHING is two absent checks, not a weaker one, which is what
#: :func:`_assert_material_names_are_real` exists to stop.
MATERIAL = ("eps", "inv_eps")

SEED = 0x5A17

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    """A per-case seed from a DIGEST, never ``hash()``.

    ``PYTHONHASHSEED`` salts ``hash()`` of a tuple of strings, so a hash-seeded
    case cannot be replayed from its own record. A blake2b of the joined parts can.
    """
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2 ** 31 - 1)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not "
            f"loosen the floor.")


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records every counter at zero, because nothing had compiled yet.
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception as exc:  # noqa: BLE001
            payload["subnormal_policy_reread_error"] = repr(exc)
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001 - a missing stamper must not lose the run
        payload["provenance_stamp_error"] = repr(exc)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The exact bytes this gate binds itself to."""
    names = (
        "meep_gpu/triton_kernels/nonlinear_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/nonlinear_update_e.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_nonlinear_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the module adds no arithmetic, and says so checkably
# ---------------------------------------------------------------------------

def no_arithmetic_leg() -> Dict[str, Any]:
    """The premise, read off the shipped source rather than off the docstring."""
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        "nonlinear_fused_magnetic_pair.py")
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    jitted = [node.name for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              for decorator in node.decorator_list if "jit" in ast.dump(decorator)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    # The ONE call that must build the plan. A second spelling of the binding list
    # would be the crossed-sub-lattice hazard waiting to happen.
    builders = sorted({node.func.id for node in ast.walk(tree)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name)
                       and node.func.id.startswith("plan_")})
    findings: List[str] = []
    if jitted:
        findings.append(f"the module decorates {jitted} with a JIT")
    if classes != ["_LinearScopeView"]:
        findings.append(f"the module defines classes {classes}")
    if "import triton" in text:
        findings.append("the module imports the optional dependency")
    if builders != ["plan_fused_pair"]:
        findings.append(f"the plan is built through {builders}, not plan_fused_pair")
    if "FusedPairPlan(" in text:
        findings.append("the module constructs a FusedPairPlan directly, which "
                        "would re-spell the binding list")
    return {"leg": "no_arithmetic", "device": False, "jit_functions": jitted,
            "classes": classes, "plan_builders_called": builders,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the scope view masks exactly one attribute
# ---------------------------------------------------------------------------

def scope_view_leg() -> Dict[str, Any]:
    """Every attribute the view answers differently from the real ``Fields``.

    A view that masked a second fact would put the shipped predicate in front of a
    fiction, and no byte comparison downstream could see it: the arrays would still
    be the real ones. This enumerates instead of asserting.
    """
    import importlib  # noqa: PLC0415

    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), dimensions=3,
                courant=0.35)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.set_background_eps(2.25)
    fields.set_nonlinear_volumes({"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
                                 {"Ex": CHI3_CORPUS, "Ey": CHI3_CORPUS,
                                  "Ez": CHI3_CORPUS})
    fields.enable_pml_storage()
    view = product._LinearScopeView(fields)  # noqa: SLF001 - the object under test

    names = sorted({name for name in dir(fields) if not name.startswith("__")})
    differing: List[str] = []
    unreadable: List[str] = []
    for name in names:
        try:
            real = getattr(fields, name)
        except Exception:  # noqa: BLE001 - an unreadable attribute is not a mask
            unreadable.append(name)
            continue
        try:
            through = getattr(view, name)
        except Exception as exc:  # noqa: BLE001
            differing.append(f"{name}: the view RAISED {exc!r}")
            continue
        same = through is real
        if not same:
            try:
                same = bool(through == real)
            except Exception:  # noqa: BLE001 - arrays compare elementwise
                same = False
                try:
                    same = bool(np.array_equal(np.asarray(through),
                                               np.asarray(real)))
                except Exception:  # noqa: BLE001
                    same = False
        if not same:
            differing.append(name)
    findings: List[str] = []
    if differing != ["has_nonlinearity"]:
        findings.append(f"the view answers {differing} differently, not exactly "
                        f"['has_nonlinearity']")
    if bool(getattr(view, "has_nonlinearity")) is not False:
        findings.append("the view does not invert has_nonlinearity")
    if bool(getattr(fields, "has_nonlinearity")) is not True:
        findings.append("the fixture is not nonlinear; the leg measured nothing")
    return {"leg": "scope_view", "device": False, "attributes_scanned": len(names),
            "attributes_the_view_changes": differing,
            "attributes_unreadable_on_the_real_fields": unreadable,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the weld is never wider than what it conjoins
# ---------------------------------------------------------------------------

def _laptop_fixture(**keywords):
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    chi3 = keywords.pop("chi3", CHI3_CORPUS)
    complex_storage = keywords.pop("complex_storage", False)
    pml_thickness = keywords.pop("pml_thickness", 2)
    uniform_pml = keywords.pop("uniform_pml", False)
    cell_size = keywords.pop("cell_size", (0.8, 0.8, 0.8))
    dimensions = keywords.pop("dimensions", 3)
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                courant=0.35, **keywords)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if chi3 is not None:
        fields.set_nonlinear_volumes({"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
                                     {"Ex": chi3, "Ey": chi3, "Ez": chi3})
    fields.enable_pml_storage()
    thickness = pml_thickness if uniform_pml else tuple(
        (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
        for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _MagneticSource:
    field_type = "B"


class _ElectricSource:
    field_type = "D"


#: The configurations the equivalence sweep asks over. Each perturbs ONE clause.
EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus_family", {}, ()),
    ("electric_source", {}, (_ElectricSource(),)),
    ("magnetic_source", {}, (_MagneticSource(),)),
    ("undeclared_sources", {}, None),
    ("linear_run", {"chi3": None}, ()),
    ("complex_storage", {"complex_storage": True}, ()),
    ("no_absorber", {"pml_thickness": 0}, ()),
    ("metallic_walls", {"boundaries": "metallic"}, ()),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, ()),
    ("beta_2d", {"cell_size": (0.8, 0.8, 0.0), "dimensions": 2, "beta": 0.31}, ()),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld == spine curl AND spine H AND the seam clauses``, measured."""
    import importlib  # noqa: PLC0415

    from meep_gpu.grid import Mirror  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    cases = list(EQUIVALENCE_CASES) + [
        ("folded", {"symmetry": (Mirror("Y", 1),), "uniform_pml": True}, ()),
    ]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in cases:
        fields, pml = _laptop_fixture(**dict(keywords))
        report = product.nonlinear_fused_magnetic_pair_equivalence(
            fields, pml, sources)
        # The merge-bar host is NumPy, so every predicate carries the backend
        # clause. Discounting it is what lets this leg ask about the others.
        covered = not [r for r in report["weld_reasons"] if "array module" not in r]
        rows.append({"case": name, "weld_covered_modulo_backend": covered,
                     "weld_never_wider": report["weld_never_wider_than_the_halves"],
                     "weld_reasons": report["weld_reasons"]})
        if not report["weld_never_wider_than_the_halves"]:
            findings.append(f"{name}: the weld admitted where a half refused")
        admitted += int(covered)
    if admitted == 0:
        findings.append("no case was admitted modulo the backend clause: this leg "
                        "measured only refusals and could not see a widening")
    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted, "findings": findings,
            "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — what the corpus says, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results", "predicate_coverage_2026-08-16_wired_convention")
MATRIX = os.path.join(HERE, "results", "fusion_matrix_triton_2026-08-20_closed",
                      "fusion_matrix.json")


def corpus_admission_leg() -> Dict[str, Any]:
    """The two rows this arm can serve, counted rather than cited.

    Read from the census's own configuration block: a row is in this cell when it
    declares a nonlinearity, an active absorber, real storage, no fold, no
    cylindrical axis, k = 0, no beta and no BFAST; it is REACHABLE when it declares
    no MAGNETIC source. The matrix's own count is loaded beside it and the two must
    agree — a disagreement means one of them is measuring something else.
    """
    findings: List[str] = []
    record: List[dict] = []
    for name in ("examples.jsonl", "tests.jsonl"):
        path = os.path.join(CENSUS, name)
        if not os.path.exists(path):
            return {"leg": "corpus_admission", "device": False,
                    "findings": [f"the census is absent: {path}"], "passed": False}
        record.extend(json.loads(line) for line in
                      open(path, encoding="utf-8").read().splitlines() if line.strip())
    matched = {(r.get("leg"), r.get("row")): r for r in (
        json.loads(line) for line in open(
            os.path.join(CENSUS, "tests_param_matched.jsonl"),
            encoding="utf-8").read().splitlines() if line.strip())}
    record = [matched.get((r.get("leg"), r.get("row")), r) for r in record]
    record = [r for r in record if r.get("measured")]

    in_cell: List[str] = []
    reachable: List[str] = []
    for row in record:
        configuration = row.get("configuration") or {}
        if not configuration.get("has_nonlinearity"):
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if configuration.get("has_symmetry") or any(configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical") or configuration.get("bfast_active"):
            continue
        if float(configuration.get("beta") or 0.0) != 0.0:
            continue
        if configuration.get("has_bloch"):
            continue
        name = f"{row['leg']}:{row['row']}"
        in_cell.append(name)
        if "B" not in tuple(configuration.get("source_field_types") or ()):
            reachable.append(name)

    matrix_rows: List[str] = []
    if os.path.exists(MATRIX):
        matrix = json.load(open(MATRIX, encoding="utf-8"))
        matrix_rows = sorted(
            entry["row"] for entry in matrix["seam_instances"]
            if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"])
            == ("B->H", "nonlinear run PML", "nonlinear run")
            and not entry["in_seam_source"])
        if matrix_rows != sorted(reachable):
            findings.append(
                f"this leg derived {sorted(reachable)} from the census while the "
                f"fusion matrix scores {matrix_rows} at the same cell")
    else:
        findings.append(f"the fusion matrix is absent: {MATRIX}")
    if not reachable:
        findings.append("no reachable row: this arm would be worth nothing")
    return {"leg": "corpus_admission", "device": False, "rows_scanned": len(record),
            "rows_in_the_cell": sorted(in_cell),
            "rows_clearing_the_source_seam": sorted(reachable),
            "matrix_rows_at_the_same_cell": matrix_rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# The device harness
# ---------------------------------------------------------------------------

def seed_values(cp, driver, seed: int, value_class: str) -> None:
    """Seed every settable volume in ONE value class, identically on every route."""
    rng = np.random.default_rng(seed)
    shape = tuple(driver.shape)

    def draw() -> np.ndarray:
        if value_class == "uniform":
            values = rng.uniform(-0.25, 0.25, size=shape)
        elif value_class == "signed_zero_lattice":
            # +0.0 and -0.0 on a random lattice. The two words differ (0x00000000
            # vs 0x80000000), so a byte gate SEES the difference — this is what
            # exercises every `tl.where(flag, 0.0, v)` and every subtraction's
            # sign convention.
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            # Below 2**-126: the words the two float32 subnormal policies disagree
            # about. Identical under either policy is the claim; the policy stamp
            # says which one this process ran.
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover - VALUE_CLASSES is the whole domain
            raise ValueError(f"unknown value class {value_class!r}")
        return np.ascontiguousarray(values.astype(np.float32))

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(draw()))
    # E and H are DERIVED and `set_field` refuses them by name (driver.py:3909).
    # Written in place here, which is legal for a harness and is what makes the
    # single-launch leg non-vacuous: without a nonzero E the B curl is exactly zero
    # on step 1 and the leg would compare two ramps that never differ.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str,
                 electric: bool, chi3: float = CHI3_CORPUS):
    """One nonlinear PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=3,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (2.25 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.set_chi3(float(chi3))
    driver.setup_pml(2)
    if electric:
        # ADMITTED: the driver injects an electric source in the D/E seam, not this
        # one. Carrying one is what stops the source clause from being exercised
        # only in its refusing direction. Both corpus rows source electrically.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY. `FdtdDriver.step` consults `fastpath`
    # BEFORE the module-level sub-step functions this harness substitutes
    # (driver.py:3281-3282), and the two nonlinear SPINE arms are WIRED — so on this
    # very family a driver left alone would step `step_B` and `update_H` on Triton
    # and the "array path" reference would be no such thing. The per-route call
    # counter re-checks that this held.
    driver.invalidate_fast_path()
    driver._fast_path = None          # noqa: SLF001 - the harness owns the route
    driver._fast_path_stale = False   # noqa: SLF001
    return driver


def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("__"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not complete")
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape",
                    "left": list(a.shape), "right": list(b.shape)}
        if not np.array_equal(a, b):
            where = int(np.flatnonzero(a != b)[0])
            return {"array": name, "index": where,
                    "left_word": int(a[where]), "right_word": int(b[where]),
                    "differing_words": int(np.count_nonzero(a != b))}
    only = sorted(set(left) ^ set(right))
    return {"array": "<inventory>", "reason": "asymmetric", "names": only} if only else None


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


class CountingKernel:
    """Owns the JIT kernel and counts every launch through the plan's ``run``."""

    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*args, **kwargs):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*args, **kwargs)

        return run


def kernel_ptx(jit: Any) -> List[str]:
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                out.append(asm["ptx"])
    return out


class Route:
    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


class _Absorbed:
    """The sentinel left where a driver pass was absorbed by the fused launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``.

    Counting is PER ROUTE. A single global counter would mix the reference driver's
    legitimate array-path calls with a fallback in the fused route, which is
    precisely the event this instrument exists to see.
    """
    originals = {name: getattr(driver_module, name) for name in SEAM_PASSES}

    def bump(role: str, kind: str, name: str) -> None:
        key = f"{role}/{kind}:{name}"
        counter[key] = counter.get(key, 0) + 1

    def replacement(name):
        def wrapper(fields, *args):
            role = roles.get(id(fields), "unknown")
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                bump(role, "array_path", name)
                return originals[name](fields, *args)
            bump(role, "substituted", name)
            plan.plans[name].run()
            return None
        return wrapper

    for name in SEAM_PASSES:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def separate_route(driver):
    """The TWO separately certified Triton spine products this launch replaces.

    They are the arms ``launch.plan_step`` really selects on a nonlinear row today
    — ``nonlinear run PML`` on ``step_B`` and ``nonlinear run`` on ``update_H`` —
    so this route is not a second model of the seam, it is the shipped one.

    The three in-seam passes stay on the ARRAY PATH here: no Triton product owns
    the wall clear, and both symmetry fills are no-ops without a mirror plane. All
    three are COUNTED, so a difference in which of them ran is a counter event
    rather than an invisible correction.
    """
    from meep_gpu.triton_kernels import nonlinear_update_e as spine  # noqa: PLC0415

    curl = spine.plan_nonlinear_run_pml_curl(driver.fields, driver.pml, "step_B")
    constitutive = spine.plan_nonlinear_run_constitutive(driver.fields, driver.pml, "H")
    missing = [name for name, plan in
               (("nonlinear run PML curl", curl),
                ("nonlinear run constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the certified "
            f"products it replaces")
    return Route({"step_B": curl, "update_H": constitutive})


def fused_route(plan):
    return Route({
        "step_B": plan,
        "fill_symmetry_bc_B": _Absorbed("fill_symmetry_bc_B", plan),
        "zero_metal_B": _Absorbed("zero_metal_B", plan),
        "fill_folded_far_ghosts_B": _Absorbed("fill_folded_far_ghosts_B", plan),
        "update_H": _Absorbed("update_H", plan),
    })


def _swapped_coefficients(cp, driver, plan, group: str):
    """The plan's coefficient bindings with ONE group moved to the WRONG lattice.

    ``SUB_STEPS['step_B']`` carries ``suffix == '_h'`` so the curl half reads the
    HALF-INTEGER split-field pair, while ``CONSTITUTIVE_SIDES['H']['half_integer']
    is False`` so the constitutive half reads the INTEGER one. The kernel takes
    twelve pointers and never asks which lattice they came from, so a swap is a
    half-cell error in the absorber profile: converged, smooth and wrong.
    """
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    pml = driver.pml
    if group == "curl":
        arrays = [getattr(pml, f"{stem}_{axis}")            # INTEGER, wrongly
                  for axis in "xyz" for stem in ("kms", "sinv")]
        plan._curl_coefficients = tuple(  # noqa: SLF001 - the armed seam
            CupyPointer(_flat(a)) for a in arrays)
    elif group == "constitutive":
        arrays = [getattr(pml, f"{stem}_{axis}_h")          # HALF-INTEGER, wrongly
                  for axis in "xyz" for stem in ("kps", "kms")]
        plan._constitutive_coefficients = tuple(  # noqa: SLF001
            CupyPointer(_flat(a)) for a in arrays)
    else:  # pragma: no cover
        raise ValueError(group)
    return plan


#: The HOST mutations: plan-level choices, not kernel text. ``(id, why, expectation,
#: apply)``. Each is applied to a freshly built plan just before the leg runs.
def host_mutation_table() -> Tuple[Tuple[str, str, str, Callable[[Any, Any, Any], Any]], ...]:
    def h1_integer_coefficients_on_the_curl(cp, driver, plan):
        return _swapped_coefficients(cp, driver, plan, "curl")

    def h2_half_integer_coefficients_on_the_constitutive(cp, driver, plan):
        return _swapped_coefficients(cp, driver, plan, "constitutive")

    def h3_zero_metal_all_false(cp, driver, plan):
        """The wall clear switched off at the HOST while the walls are real.

        The counterpart of the kernel-side ``m2``: same defect, other side of the
        seam. Scored on ``wall_xyz``, where all three flags are really True — on a
        periodic case this rewrite would change nothing and measure nothing.
        """
        plan.zero_metal = (0, 0, 0)
        return plan

    def h4_wall_table_is_the_d_side(cp, driver, plan):
        """``zero_metal`` rotated: still three planes, the WRONG three.

        ``stepping._zero_metal`` clears component ``m`` on walled axis ``a`` exactly
        when ``IYEE_SHIFTS[m][a] == 0``; for B that table is the DIAGONAL
        (x->Bx, y->By, z->Bz). Rotating the flags keeps the same COUNT of cleared
        planes, so a leg that only counted would pass. Scored on ``corpus_wall_z``,
        where exactly ONE axis is walled and the rotation moves the clear onto a
        component the array path leaves alone.
        """
        flags = tuple(plan.zero_metal)
        plan.zero_metal = (flags[2], flags[0], flags[1])
        return plan

    return (
        ("h1_integer_coefficients_on_the_curl", "the crossed sub-lattices",
         "caught", h1_integer_coefficients_on_the_curl),
        ("h2_half_integer_coefficients_on_the_constitutive",
         "the crossed sub-lattices", "caught",
         h2_half_integer_coefficients_on_the_constitutive),
        ("h3_zero_metal_all_false", "the wall clear, switched off at the host",
         "caught", h3_zero_metal_all_false),
        ("h4_wall_table_is_the_d_side", "the DIAGONAL vs the OFF-DIAGONAL table",
         "caught", h4_wall_table_is_the_d_side),
    )


#: Host mutation id -> the CASES index it is scored on.
HOST_MUTATION_CASE: Dict[str, int] = {
    "h4_wall_table_is_the_d_side": 0,   # corpus_wall_z: exactly ONE walled axis
}


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_magnetic: bool = False, electric: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    seed = case_seed(name, str(cell), str(boundaries), value_class)
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric)
            for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(mutant if mutant is not None
                            else _shipped_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "value_class": value_class, "seed": seed,
        "electric_source": bool(electric),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "host_mutation": None, "first_divergence": None,
        "control_divergence": None,
    }
    try:
        plan = product.plan_nonlinear_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        if plan is None:
            verdict = product.nonlinear_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](cp, fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(product.REPLACES)
        row["zero_metal_flags"] = list(plan.zero_metal)
        # THE IDENTITY, RE-ASSERTED ON THE REAL DEVICE OBJECTS. The laptop leg
        # sweeps it on NumPy fixtures; this is the same question asked of the very
        # configuration whose bytes are about to be compared.
        row["equivalence"] = product.nonlinear_fused_magnetic_pair_equivalence(
            fused.fields, fused.pml, tuple(fused._sources))
        route = fused_route(plan) if install_fused else Route({})
        separate_plans = separate_route(separate)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_magnetic:
            frozen = {pass_name: _Absorbed(pass_name, None)
                      for pass_name in SEAM_PASSES}
            route.plans.update(frozen)
            separate_plans.plans.update(frozen)
            routes.append((reference.fields, Route(dict(frozen))))
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
        row["per_step"] = []
        started = time.time()
        for step in range(1, steps + 1):
            before = snapshot(cp, fused)
            reference.step()
            separate.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_separate = snapshot(cp, separate)
            after_fused = snapshot(cp, fused)
            versus_array = first_divergence(after_fused, after_reference)
            versus_separate = first_divergence(after_fused, after_separate)
            control = first_divergence(after_separate, after_reference)
            step_moved = moved(before, after_fused)
            row["per_step"].append({
                "step": step,
                "fused_vs_array": versus_array,
                "fused_vs_separate_certified_products": versus_separate,
                "oracle_control_vs_array": control,
                "arrays_moved": len(step_moved),
                "fused_launches": kernel.calls,
            })
            divergence = versus_array or versus_separate
            if step == 1 or step % 10 == 0 or divergence is not None:
                log(f"  {name} [{value_class}] step {step}/{steps} "
                    f"identical={divergence is None} control={control is None} "
                    f"moved={len(step_moved)} launches={kernel.calls} "
                    f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        return row
    finally:
        undo()
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True,
               require_array_reference: bool = True) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(f"VACUOUS: these arrays never moved: {row['arrays_never_moved']}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    counts = row.get("launches") or {}
    fell_back = {key: value for key, value in counts.items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    if require_array_reference:
        # THE REFERENCE MUST REALLY BE THE ARRAY PATH. `fastpath` is consulted
        # before these functions and the nonlinear spine arms are WIRED, so a
        # missing count here means the reference stepped on Triton and the
        # comparison compared two Triton routes.
        steps = int(row.get("steps_budget") or 0)
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} "
                    f"{seen} times, expected {steps}: the fast path was not off")
        equivalence = row.get("equivalence") or {}
        if not equivalence.get("weld_never_wider_than_the_halves", False):
            failures.append(
                "on the very configuration this leg stepped, the weld admitted "
                "where one of its halves refused")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations, on the SHIPPED fused kernel
# ---------------------------------------------------------------------------

def _shipped_kernel() -> Any:
    """The shipped JIT object. DEVICE ONLY — importing ``kernels`` needs Triton."""
    from meep_gpu.triton_kernels.kernels import fused_curl_constitutive_B  # noqa: PLC0415

    return fused_curl_constitutive_B


#: The one spelling of the kernel's text, used on BOTH hosts.
KERNEL_NAME = "fused_curl_constitutive_B"
KERNEL_FILE = os.path.join(API_ROOT, "meep_gpu", "triton_kernels", "kernels.py")


def shipped_source() -> str:
    """The shipped kernel's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported, and read that way on EVERY host —
    not only the laptop. Two spellings of the same text (``inspect.getsource`` on
    a CUDA box, a file read on the merge bar) would let a rewrite match in one
    place and miss in the other, which is exactly how a mutation silently disarms.
    One spelling, so ``mutation_arming`` on the laptop measures the same bytes the
    device legs mutate.

    The text is EXACT — never ``ast.unparse``d — because what several rewrites are
    about is the PARENTHESISATION, and unparsing re-derives minimal parentheses.

    The docstring IS stripped, and that is not cosmetic: the body's docstring names
    the codes and branches it does NOT carry, so a text match over the raw source
    would find those names in prose.
    """
    text = open(KERNEL_FILE, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef)
                 and found.name == KERNEL_NAME), None)
    if node is None:
        raise AssertionError(f"{KERNEL_NAME} is not defined in {KERNEL_FILE}")
    segment = ast.get_source_segment(text, node)
    if segment is None:  # pragma: no cover - only on a source-less module
        raise AssertionError(f"{KERNEL_NAME}'s source segment is unavailable")
    lines = segment.splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    margin = node.col_offset
    lines = [lines[0].lstrip()] + [
        line[margin:] if line[:margin].strip() == "" else line.lstrip()
        for line in lines[1:]]
    # THE DECORATOR IS PREPENDED, and it is not cosmetic. `ast.get_source_segment`
    # on a FunctionDef starts at the ``def`` line and EXCLUDES the decorator list,
    # so a mutant compiled from the bare segment is a PLAIN PYTHON FUNCTION: it has
    # no ``[grid]`` launcher, and the first mutation leg dies with
    # "'function' object is not subscriptable" — measured 2026-08-20, which aborted
    # the whole mutation leg after every byte leg had already passed. Read off the
    # node rather than hard-coded, so a change of decorator moves with it.
    decorators = [ast.get_source_segment(text, d) for d in node.decorator_list]
    if any(segment is None for segment in decorators):  # pragma: no cover
        raise AssertionError(f"{KERNEL_NAME}'s decorator source is unavailable")
    return "\n".join([f"@{segment}" for segment in decorators] + lines)


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n")
    text = header + source.replace("fused_curl_constitutive_B", kernel_name)
    # A MUTATION THAT DOES NOT PARSE IS NOT A MUTATION. Checked before any device
    # is involved, because a SyntaxError inside the loader aborts the whole
    # mutation leg on the first device run.
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_nonlinear_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_nonlinear_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a contiguous run of statements, preserving the block's indent.

    Every replacement line takes the indent of the block's FIRST line, so a
    compound statement in position 0 would produce an empty suite — which is why
    the entries below that drop an ``if`` drop its whole body with it.
    """
    lines = source.splitlines()
    # MATCHED ON COMMENT-STRIPPED TEXT, so a rewrite stays armed across a trailing
    # comment or a reflow. A mutation that silently stops matching reports a real
    # defect as uncaught, which this tree has already paid for once.
    stripped = [line.split("#", 1)[0].strip() for line in lines]
    needles = [text.split("#", 1)[0].strip() for text in before]
    for index in range(len(lines) - len(needles) + 1):
        if stripped[index:index + len(needles)] != needles:
            continue
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        replaced = lines[:index] + [indent + text for text in after] \
            + lines[index + len(needles):]
        return "\n".join(replaced) + "\n", 1
    return source, 0


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite)."""

    def m1_wall_table_is_the_d_side(source: str) -> Tuple[str, int]:
        """The DIAGONAL wall table rotated onto the OFF-DIAGONAL pairing.

        ``stepping._zero_metal`` clears component ``m`` on walled axis ``a`` exactly
        when ``IYEE_SHIFTS[m][a] == 0`` (stepping.py:2233-2247); ``fields.IYEE_SHIFTS``
        (fields.py:214-219) gives ``Bx (0,1,1)``, ``By (1,0,1)``, ``Bz (1,1,0)``, so
        the B table is x->Bx, y->By, z->Bz — the DIAGONAL. This rotation still
        clears one plane per walled axis and still clears it before both consumers.
        It clears the WRONG one, which no counting check can see.

        SCORED ON ``wall_xyz``, where every ``ZM_*`` is True and every ``at_*``
        plane is real.
        """
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["if ZM_X:", "    v1 = tl.where(at_x, 0.0, v1)",
             "if ZM_Y:", "    v2 = tl.where(at_y, 0.0, v2)",
             "if ZM_Z:", "    v0 = tl.where(at_z, 0.0, v0)"])

    def m2_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's ``step_boundaries(B_stuff)`` undone.

        The ``if`` goes WITH its body: ``_rewrite_block`` re-indents to the block's
        first line, so a compound statement in position 0 would produce an empty
        suite and the module would not import — which this harness records as an
        unarmed mutation rather than a caught defect. Dropping the guard is also
        the truer defect ("the clear was not carried at all"), and it is NOT the
        change being measured: on ``wall_xyz`` all three flags are True, so the
        guarded lines executed on the pristine kernel too. The assignments left
        behind are pure identity, so a signed zero survives them.
        """
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["v0 = v0", "v1 = v1", "v2 = v2"])

    def m3_zero_metal_only_on_the_store(source: str) -> Tuple[str, int]:
        """Cleared into ``B``, but the constitutive half reads the UNCLEARED value.

        This is the whole point of applying the clear to the REGISTER: the array
        path leaves ONE value in ``B`` and both consumers must see it. Here the
        store gets the cleared word and ``update_H`` gets the raw one — a defect
        that leaves ``Bx``/``By``/``Bz`` byte-identical and moves only ``H`` and
        ``f_w_H*``. Same guard note as ``m2``; scored on ``wall_xyz``.
        """
        source, hits = _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["z0 = tl.where(at_x, 0.0, v0)",
             "z1 = tl.where(at_y, 0.0, v1)",
             "z2 = tl.where(at_z, 0.0, v2)"])
        if not hits:
            return source, 0
        for register in ("0", "1", "2"):
            source = source.replace(
                f"tl.store(f{register} + idx, v{register}, mask=live)",
                f"tl.store(f{register} + idx, z{register}, mask=live)")
        return source, hits

    def m4_history_read_after_write(source: str) -> Tuple[str, int]:
        """``f_w`` read AFTER it is written: wrong wherever ``kms != 0``, i.e. in
        the PML. Unguarded — these three lines run on every case."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
             "src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m5_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source,
            ["a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"],
            ["a0 = a0 + (kp_0 * src0 - km_0 * prev0)"])

    def m6_curl_parentheses_flattened(source: str) -> Tuple[str, int]:
        """``stepping._curl_from_operands``' grouping flattened.

        ``((c_y - c) + (b - b_z))`` and ``(((c_y - c) + b) - b_z)`` are the same
        algebra and different float32 roundings. The comment three lines above it
        says DO NOT flatten these parens; this measures what happens when someone
        does. Unguarded."""
        needle = "curl0 = dtdx * ((c_y - c) + (b - b_z))"
        return source.replace(needle, "curl0 = dtdx * (c_y - c + b - b_z)"), \
            source.count(needle)

    def m7_ownership_mask_dropped(source: str) -> Tuple[str, int]:
        """``stepping._mask_non_owned_cells`` undone on the forward arm.

        Guarded on ``BCX/BCY/BCZ == METALLIC`` and on the ``else`` (forward) arm,
        both of which ``wall_xyz`` enters — every axis is METALLIC there and the
        B curl is forward (``BACKWARD == 0``)."""
        return _rewrite_block(
            source,
            ["if BCX == METALLIC:", "curl0 = tl.where(at_x, 0.0, curl0)",
             "if BCY == METALLIC:", "curl1 = tl.where(at_y, 0.0, curl1)",
             "if BCZ == METALLIC:", "curl2 = tl.where(at_z, 0.0, curl2)"],
            ["curl0 = curl0", "curl1 = curl1", "curl2 = curl2"])

    def m8_metallic_ghost_becomes_a_wrap(source: str) -> Tuple[str, int]:
        """The x METALLIC ghost read as a periodic wrap.

        The metallic ghost is an exact 0.0 served by ``other=`` past the wall; a
        wrap reads the far face instead. Guarded on ``BCX == METALLIC``, which
        ``wall_xyz`` enters."""
        return _rewrite_block(
            source,
            ["if BCX == METALLIC:", "vx = live & (si >= 0) & (si < nx)",
             "else:", "si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))"],
            ["si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))"])

    def m9_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_wall_table_is_the_d_side", "the DIAGONAL vs the OFF-DIAGONAL table",
         "caught", m1_wall_table_is_the_d_side),
        ("m2_zero_metal_dropped", "the wall clear's slot", "caught",
         m2_zero_metal_dropped),
        ("m3_zero_metal_only_on_the_store", "the register-vs-memory carry",
         "caught", m3_zero_metal_only_on_the_store),
        ("m4_history_read_after_write", "the f_w ordering", "caught",
         m4_history_read_after_write),
        ("m5_constitutive_association", "float32 association", "caught",
         m5_constitutive_association),
        ("m6_curl_parentheses_flattened", "the curl grouping", "caught",
         m6_curl_parentheses_flattened),
        ("m7_ownership_mask_dropped", "the ownership mask", "caught",
         m7_ownership_mask_dropped),
        ("m8_metallic_ghost_becomes_a_wrap", "the metallic ghost", "caught",
         m8_metallic_ghost_becomes_a_wrap),
        ("m9_commuted_multiply", "commuted multiply", "null", m9_commuted_multiply),
    )


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#: Anything not named here runs on :data:`MUTATION_CASE_INDEX` (``wall_xyz``),
#: which is the only case entering BOTH the metallic ghost arms and all three
#: ``ZM_*`` clears.
MUTATION_CASE: Dict[str, int] = {}


def mutation_arming_leg() -> Dict[str, Any]:
    """Every rewrite ARMS and PARSES, checked on the laptop before any device run."""
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, why, expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        parses = True
        error = None
        try:
            ast.parse("import triton\nimport triton.language as tl\n"
                      "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n"
                      + mutated.replace("fused_curl_constitutive_B", "mutant"))
        except SyntaxError as exc:  # noqa: PERF203
            parses, error = False, repr(exc)
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "rewrite_hits": hits, "parses": parses, "error": error,
                     "changed": mutated != source,
                     "case": CASES[MUTATION_CASE.get(name, MUTATION_CASE_INDEX)][0]})
        if not hits or mutated == source:
            findings.append(f"{name}: the rewrite matched nothing")
        if not parses:
            findings.append(f"{name}: the mutant does not parse ({error})")
    return {"leg": "mutation_arming", "device": False, "mutations": rows,
            "findings": findings, "passed": not findings}


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "kind": "source", "device": True}
        if hits == 0 or mutated == source:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_nonlinear_fused_B"
        mutant = compile_mutant(mutated, kernel_name)
        case_index = MUTATION_CASE.get(name, MUTATION_CASE_INDEX)
        case = CASES[case_index]
        row["case"], row["case_index"] = case[0], case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], MUTATION_STEPS,
                      product, mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["arrays_never_moved"] = leg.get("arrays_never_moved")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


def run_host_mutations(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in host_mutation_table():
        name, why, expectation, _apply = entry
        case_index = HOST_MUTATION_CASE.get(name, MUTATION_CASE_INDEX)
        case = CASES[case_index]
        leg = run_leg(cp, f"host_mutation:{name}", case[1], case[2],
                      MUTATION_STEPS, product, host_mutation=entry)
        row = {"mutation": name, "why": why, "expectation": expectation,
               "kind": "host", "device": True, "case": case[0],
               "case_index": case_index,
               "zero_metal_flags": leg.get("zero_metal_flags"),
               "caught": leg.get("first_divergence") is not None,
               "first_divergence": leg.get("first_divergence"),
               "launches": leg.get("fused_kernel_launches")}
        rows.append(row)
        log(f"  host mutation {name}: caught={row['caught']} on {row['case']} "
            f"expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, asked of a real CuPy driver."""

    class _Magnetic:
        field_type = "B"

    walls = {"x": "periodic", "y": "periodic", "z": "metallic"}
    cell = CASES[0][1]
    rows: List[Dict[str, Any]] = []
    for name, chi3, sources, needle in (
        ("magnetic_source", CHI3_CORPUS, (_Magnetic(),), "is magnetic"),
        ("undeclared_sources", CHI3_CORPUS, None, "was not declared"),
        ("linear_run", 0.0, (), "no chi2/chi3 is installed"),
    ):
        driver = build_driver(cp, cell, walls, SEED, "uniform", electric=False,
                              chi3=chi3)
        try:
            verdict = product.nonlinear_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            plan = product.plan_nonlinear_fused_magnetic_pair(
                driver.fields, driver.pml, sources)
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "reasons": list(verdict.reasons), "plan_is_none": plan is None,
                "passed": (not verdict.covered) and plan is None
                          and any(needle in reason for reason in verdict.reasons),
            })
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()
        log(f"  refusal {name}: covered={rows[-1]['covered']} "
            f"passed={rows[-1]['passed']}")
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def environment(cp: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import triton  # noqa: PLC0415

        payload["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        payload["triton"] = f"absent: {exc!r}"
    if cp is not None:
        payload["cupy"] = cp.__version__
        payload["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_nonlinear_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. BOTH are gated, one process each: a run that "
             "installs NOTHING is a MIXED configuration (CuPy appends -ftz=true "
             "unconditionally; Triton natively keeps) attributable to no policy.")
    parser.add_argument(
        "--plant-defect", default=None,
        help="install a named kernel mutation as THE PRODUCT'S kernel for every "
             "device leg. The release verdict must FLIP to False; a gate whose "
             "verdict does not move against a planted defect is not measuring.")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_nonlinear_fused_magnetic_pair",
        "product": PRODUCT_MODULE,
        "kernel": "kernels.fused_curl_constitutive_B (the SHIPPED ordinary pair)",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "kernels.DEFAULT_BLOCK"},
        "planted_defect": args.plant_defect,
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (no_arithmetic_leg, scope_view_leg, equivalence_leg,
                corpus_admission_leg, mutation_arming_leg):
        started = time.time()
        row = leg()
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, args.out)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal "
            "in this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST,
        # and it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device.
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation or "
                        "refusal has run: this artifact releases nothing"],
        }
        save(payload, args.out)
        log(f"\nno-device verdict: {payload['passed']}  ->  {args.out}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    if args.subnormal_policy == "flush":
        # MEASURED REFUSAL, not a precaution. `install_subnormal_policy('flush')`
        # raises SubnormalPolicyUnattainable in a process that has not imported
        # MEEP and says why in its own words (subnormal_policy.py:1920, measured
        # here 2026-08-20): `mp.set_zero_subnormals` is the only exposure of this
        # process's FTZ/DAZ bits the package may use, and `subnormal_policy` will
        # not import MEEP itself because that would break meep_gpu's
        # no-MEEP-import boundary and initialize MPI as a side effect. Its
        # instruction to a caller that needs flush is exactly this.
        #
        # A PARITY PROBE MAY DO WHAT THE PACKAGE MAY NOT: the boundary protects
        # `meep_gpu/**`, which ships; this file is a comparator. The same helper
        # and the same reasoning as
        # `probe_fused_kernel_bit_identity.import_meep_for_host_policy`.
        #
        # NOT UNCONDITIONAL: importing MEEP initializes MPI and moves the host
        # FPU, so a `keep` leg that imported it would differ from one that did not
        # for reasons unrelated to the kernel.
        record: Dict[str, Any] = {"requested": True}
        try:
            import meep  # noqa: PLC0415 - see above; parity probe, not package

            record.update(imported=True,
                          meep_version=getattr(meep, "__version__", "<unknown>"),
                          has_set_zero_subnormals=hasattr(meep,
                                                          "set_zero_subnormals"))
        except Exception as exc:  # noqa: BLE001 - a refusal is the result
            record.update(imported=False,
                          why=f"{type(exc).__name__}: {exc}"[:400])
        payload["meep_import_for_host_policy"] = record
        log(f"[policy] MEEP import for the host half of 'flush': {record}")
        save(payload, args.out)

    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    import importlib  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)

    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    save(payload, args.out)

    planted = None
    if args.plant_defect:
        table = {name: rewrite for name, _why, _expect, rewrite in mutation_table()}
        if args.plant_defect not in table:
            raise SystemExit(f"unknown mutation {args.plant_defect!r}; "
                             f"choose from {sorted(table)}")
        mutated, hits = table[args.plant_defect](shipped_source())
        if not hits:
            raise SystemExit(f"{args.plant_defect} armed nothing")
        planted = compile_mutant(mutated, "planted_defect_nonlinear_fused_B")
        log(f"PLANTED DEFECT: {args.plant_defect} ({hits} rewrite hits)")

    log("\n=== device legs ===")
    for name, cell, boundaries, steps in CASES:
        for value_class in (VALUE_CLASSES if steps > 1 else ("uniform",)):
            row = run_leg(cp, name, cell, boundaries, steps, product,
                          value_class=value_class, mutant=planted)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            log(f"  {name} [{value_class}]: passed={passed} {failures}")
            save(payload, args.out)

    log("\n=== armed harness mutations ===")
    row = run_leg(cp, "armed:no_substitution", CASES[0][1], CASES[0][2], 3,
                  product, install_fused=False, mutant=planted)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", CASES[0][1], CASES[0][2], 3,
                  product, freeze_magnetic=True, mutant=planted)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(_shipped_kernel())
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== armed host mutations ===")
    payload["mutations"].extend(run_host_mutations(cp, product))
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        # A DECLARED NULL MUST BE CONFIRMED, not merely permitted, and a mutation
        # that was never ARMED measured nothing whatever it then reported.
        if row.get("error") is not None:
            return False
        if row.get("kind") == "source" and not row.get("rewrite_hits"):
            return False
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"] and not args.plant_defect,
        "reasons": ([] if (payload["passed"] and not args.plant_defect) else
                    [f"device legs ok: {device_ok}",
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}",
                     "mutations whose measured outcome did not match their "
                     "declared expectation, or that were never armed: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row)))]
                    + ([f"A DEFECT WAS PLANTED ({args.plant_defect}): this run is a "
                        f"control and releases nothing"] if args.plant_defect else [])),
    }
    save(payload, args.out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: {payload['passed']}  ->  {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Device byte gate: the no-absorber ``step_D`` welded into stored-E ``update_E``.

THE PRODUCT is ``meep_gpu/triton_kernels/no_pml_fused_electric_pair.py`` — one launch
for ``step_D``, ``zero_metal_D`` and ``update_E`` on a run with NO absorber and a STORED
E, with the in-seam electric deposit carried across the launch by
``deposit_repair.PLAIN_PATH``.

WHAT IS NEW HERE, AND IT IS THE REASON THIS GATE EXISTS RATHER THAN A TWELFTH COPY OF
ITS SIBLINGS. Every fused pair released before today inverts the SPLIT-FIELD
constitutive recurrence when its seam carries a deposit. This one cannot: with no
absorber ``stepping.update_E`` writes ``E[...] = (D - sum P) * inv_eps`` straight to
storage (``stepping.py:1019-1022``), there is no ``f_w`` to save and no ``kps``/``kms`` to
re-apply. ``deposit_repair.PLAIN_PATH`` is the second repair, and THIS is the first
device run of any product that declares it.

THE FOUR THINGS EVERY DEVICE LEG MEASURES, per COMPLETE driver step:

  1. the fused route against the ARRAY PATH, as uint32 words over the whole inventory
     (the primary volumes, the auxiliaries and every polarization buffer);
  2. the fused route against the SEPARATELY CERTIFIED PRODUCTS it replaces — the
     ``conductive no-PML``/``no-PML`` curl arm on ``step_D`` and the ``no-PML stored E``
     arm on ``update_E``, with the driver's own wall clear and injection between them.
     That route is the shipped composition, not a second model of it;
  3. the LAUNCH COUNT, from a counter wrapped around the JIT handle. Agreement with a
     kernel that never ran is agreement about the array path;
  4. an ORACLE CONTROL: the separate route against the array path. If that itself
     diverges the leg measured nothing about the weld.

THE NULL CONTROL. The same carry cases with the deposit bracket REMOVED. It MUST
diverge: the launch then consumes a pre-injection D and the repair is the whole
difference. A bracket that changes nothing is not load-bearing, and a carry leg that
passes beside a null control that also passes has measured the reference against itself.

THE CLAUSE THAT USED TO PRICE THIS CELL IS MEASURED HERE TOO — in BOTH of its lives.
Until 2026-09-01 a NON-INTEGRATED electric source on a conductive run sent the driver
through ``_inject_electric_through_conductivity``'s whole-volume rescale, which
canonicalised every ``-0.0`` in the target D component to ``+0.0`` at cells no deposit
closure can name; the product refused that BY NAME and this gate's precursor leg ran it
anyway and required the divergence. THE DRIVER NOW REPLAYS THE RESCALE SPARSELY at the
deposit cells the sources publish (identity everywhere else), the clause was lifted on
that change, and ``lifted_refusal`` below is the lift's measurement: the formerly
refused configuration, WITH the bracket, must now be BYTE-IDENTICAL to the array path
at EVERY step — while the RETIRED whole-volume composition, replayed verbatim as an
armed control on a third driver, MUST still diverge from today's array path on the
same signed-zero seed. A lift whose control cannot diverge measured nothing; a lift
whose carried route diverges anywhere is no lift. The integrated-source and lossless
companions of the original three-leg protocol ride in the same leg and must stay
byte-identical. What the clause still refuses — a scaled source publishing NO deposit
table, the driver's dense fallback — is asserted in ``predicate`` and ``refusals``.

WHAT A PASS DOES NOT LICENSE. Nothing about a folded grid (refused by name, and the
fold's own divergence with NO SOURCE AT ALL is measured in
``probe_plain_deposit_repair.fold_control``), nothing about an active absorber (a
different repair), nothing about the magnetic seam (``update_H`` writes nothing on this
branch), nothing about dispatch, and nothing about throughput.

Progress (the progress-reporting rule): one flushed line per leg and per step, and the artifact is
re-written after every leg so an interrupted run keeps everything up to the failure.

Usage (from the staged tree's root)::

    python -u parity/meep_gpu/probe_triton_no_pml_fused_electric_pair.py \\
        --out results/triton_no_pml_fused_electric_pair_<tag> \\
        --subnormal-policy keep
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.dirname(os.path.dirname(HERE))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PRODUCT_MODULE = "meep_gpu.triton_kernels.no_pml_fused_electric_pair"
KERNEL_NAME = "no_pml_fused_curl_constitutive_D"

#: Every file whose bytes decide this gate's answer. The two certified halves are here
#: because the transcription leg diffs against them; ``deposit_repair`` is here because
#: the carry legs execute its second repair and the whole product rests on it.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/no_pml_fused_electric_pair.py",
    "meep_gpu/triton_kernels/no_pml_conductive.py",
    "meep_gpu/triton_kernels/no_pml.py",
    "meep_gpu/triton_kernels/no_pml_stored_e.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/deposit_repair.py",
    "meep_gpu/stepping.py",
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/sources.py",
    "meep_gpu/dispersion.py",
    "parity/meep_gpu/probe_triton_no_pml_fused_electric_pair.py",
)

SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

#: What this seam WRITES. The movement floor is stated against these rather than
#: against the whole inventory, because a B-side auxiliary this launch never touches is
#: evidence about the grid, not about the weld.
SEAM_OUTPUTS: Tuple[str, ...] = ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez")

#: Material inputs. A weld that moved one has corrupted the run whatever the fields say.
MATERIAL: Tuple[str, ...] = ("eps_Ex", "eps_Ey", "eps_Ez",
                             "inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez")

#: Scratch the array path allocates and the fused route may not need. Compared by NAME
#: rather than by value, so an inventory asymmetry is reported instead of crashing.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero_lattice", "subnormal_band")

SIGMA = 0.4
MUTATION_STEPS = 4
CARRY_STEPS = 6

#: ``(name, cell, boundaries, steps, sigma, components, poles)``.
#:
#: THE WALL CASES ARE NOT DECORATION. Two of this cell's three corpus rows are METALLIC
#: on one axis, and the wall clear is the pass this product carries INLINE — so a case
#: list without a metallic axis would leave the three ``ZM_*`` constexprs compiled to
#: nothing and every mutation of them a null.
#:
#: THE MIXED-CONDUCTIVITY CASE is the only one that puts a live ``COND == 1`` arm beside
#: a ``COND == 0`` arm in the SAME launch, which is what the three per-target constexprs
#: exist for.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int, Any, Any, int], ...] = (
    ("lossless_periodic", (1.2, 1.2, 0.0), "periodic", 6, None, None, 2),
    ("lossless_wall_xy", (1.2, 1.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 6, None, None, 2),
    ("conductive_periodic", (1.2, 1.2, 0.0), "periodic", 6, SIGMA, None, 2),
    ("conductive_wall_xy", (1.2, 1.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 6, SIGMA, None, 2),
    ("conductive_mixed_components", (1.2, 1.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 6, SIGMA, ("Dz",), 2),
    # FIVE POLES: the pole count `absorber-1d.py` and `TestAbsorber.test_absorber`
    # carry. It is the deepest unrolled chain any corpus row of this cell drives.
    ("conductive_five_poles", (1.2, 1.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 6, SIGMA, None, 5),
    ("lossless_3d", (0.8, 0.8, 0.8), "periodic", 4, None, None, 2),
)

CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: The cases that run WITH an in-seam electric deposit. Every one of them must be one
#: the product ADMITS, so a carry leg measures the product rather than a refusal. The
#: carry family uses an INTEGRATED source; the formerly refused NON-integrated
#: conductive case is measured separately in `lifted_refusal_leg`, beside the retired
#: whole-volume composition it was lifted against.
CARRY_CASES: Tuple[str, ...] = ("lossless_periodic", "lossless_wall_xy",
                                "conductive_wall_xy", "conductive_five_poles")

DEFAULT_MUTATION_CASE = "conductive_wall_xy"

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    """A per-case seed from a DIGEST, never ``hash()``.

    ``PYTHONHASHSEED`` salts ``hash()`` of a string, so a hash-seeded case is a
    different case in every process and a divergence found once cannot be re-run.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in SOURCES:
        path = os.path.join(API_ROOT, name)
        if os.path.exists(path):
            out[name] = sha256(path)
    return out


# ---------------------------------------------------------------------------
# The transcription leg — no device
# ---------------------------------------------------------------------------

def _statements(text: str) -> List[str]:
    """Significant lines only: blanks and comments dropped, indentation kept."""
    return [line.rstrip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _function_source(path: str, name: str) -> str:
    text = open(os.path.join(API_ROOT, path), encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{path} declares no function {name}")


def _dedent(lines: Sequence[str]) -> List[str]:
    return [line.strip() for line in lines]


def _contains_block(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Is ``needle`` a contiguous run of ``haystack``, compared on stripped text?"""
    hay, pin = _dedent(haystack), _dedent(needle)
    if not pin:
        return False
    return any(hay[index:index + len(pin)] == pin
               for index in range(0, len(hay) - len(pin) + 1))


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic block in the weld is a CONTIGUOUS run of a certified body.

    Checked as WHOLE BLOCKS rather than line by line, which is the difference that
    catches a reordering: a line-set comparison passes on a body whose statements were
    permuted, and permuting the pole chain or the conductive tail is a different float32
    number in every cell.

    Three blocks, three sources:

      * the curl, from the ghost rule to the per-component conductive tail —
        ``no_pml_conductive.conductive_plain_curl_step``;
      * the wall clear — ``kernels.fused_curl_constitutive_D``'s D-side ``ZM_*`` block,
        which clears the two TANGENTIAL components of each metallic axis;
      * each of the three pole chains and its ``s * inv_eps`` store —
        ``no_pml_stored_e.stored_e_constitutive_step``, with its opening
        ``tl.load(D + idx)`` replaced by the register the curl left, and NOTHING else.
    """
    findings: List[str] = []
    product = _statements(_function_source(
        "meep_gpu/triton_kernels/no_pml_fused_electric_pair.py", KERNEL_NAME))
    curl = _statements(_function_source(
        "meep_gpu/triton_kernels/no_pml_conductive.py", "conductive_plain_curl_step"))
    stored = _statements(_function_source(
        "meep_gpu/triton_kernels/no_pml_stored_e.py", "stored_e_constitutive_step"))
    wall = _statements(_function_source(
        "meep_gpu/triton_kernels/kernels.py", "fused_curl_constitutive_D"))

    def span(lines: Sequence[str], first: str, last: str) -> List[str]:
        stripped = _dedent(lines)
        try:
            start = next(index for index, line in enumerate(stripped)
                         if line.startswith(first))
            end = next(index for index, line in enumerate(stripped)
                       if index > start and line.startswith(last))
        except StopIteration:
            raise AssertionError(f"no span {first!r}..{last!r}")
        return list(lines[start:end + 1])

    # 1. THE CURL, one contiguous block from the ghost rule to the last conductive tail.
    curl_block = span(curl, "idx = tl.program_id(0)", "v2 = v2 - curl2")
    if not _contains_block(product, curl_block):
        findings.append(
            f"the curl block ({len(curl_block)} statements) is not a contiguous run of "
            f"the product; conductive_plain_curl_step was edited on the way in")

    # 2. THE WALL, from the certified fused kernel's own D-side block.
    wall_block = span(wall, "if ZM_X:", "v1 = tl.where(at_z, 0.0, v1)")
    if not _contains_block(product, wall_block):
        findings.append(
            "the zero_metal_D block is not a contiguous run of "
            "kernels.fused_curl_constitutive_D's own")

    # 3. THE THREE POLE CHAINS. Each is the certified chain with ONE line changed --
    #    the opening load -- and the change is asserted to BE that one line.
    for index, (register, letter) in enumerate((("v0", "a"), ("v1", "b"), ("v2", "c"))):
        chain = span(stored, f"s{index} = tl.load(g{index} + idx",
                     f"tl.store(f{index} + idx, s{index} * tl.load(e{index} + idx")
        welded = list(chain)
        opening = welded[0].strip()
        expected = f"s{index} = tl.load(g{index} + idx, mask=live, other=0.0)"
        if opening != expected:
            findings.append(f"the certified chain {index} does not open with {expected!r}")
            continue
        welded[0] = welded[0].replace(opening, f"s{index} = {register}")
        welded[-1] = welded[-1].replace(f"tl.store(f{index} + idx",
                                        f"tl.store(h{index} + idx")
        if not _contains_block(product, welded):
            findings.append(
                f"pole chain {index} is not the certified chain with only its opening "
                f"load replaced by the register {register}")
        # ...and the SUBSTITUTION IS THE ONLY EDIT: every interior statement must be
        # the certified one, unchanged, in order.
        if _dedent(welded[1:-1]) != _dedent(chain[1:-1]):
            findings.append(f"pole chain {index}'s interior was edited, not welded")

    # 4. THE DECLARED CONSTANTS agree with the table they are read from -- AND WITH
    #    THE OTHER ONE WHERE THE TWO AGREE.
    #
    #    THE TWO TABLES DISAGREE ON PURPOSE, and this leg is where that is pinned
    #    rather than remembered. `launch.SUB_STEPS['step_D']['sources']` is
    #    ('Hx','Hy','Hz'), which is right under an absorber where H is stored;
    #    `no_pml_conductive.SUB_STEPS` says ('Bx','By','Bz') and says why
    #    (no_pml.py:178-181): `Fields.enable_field_storage` deliberately does NOT
    #    allocate H on this branch and `get_H` returns the B array itself. This
    #    product's curl half IS that arm, so its table is the one that decides -- and
    #    a leg that checked against `launch`'s would fail a correct binding, which is
    #    what it did on this gate's first run.
    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import no_pml_conductive as arm  # noqa: PLC0415
    from meep_gpu.triton_kernels import no_pml_fused_electric_pair as module  # noqa: PLC0415
    spec = arm.SUB_STEPS[module.CURL_SUB_STEP]
    if module.BACKWARD != int(spec["backward"]):
        findings.append("BACKWARD disagrees with the curl arm's own table")
    if module.CURL_TARGETS != tuple(spec["targets"]):
        findings.append("CURL_TARGETS disagrees with the curl arm's own table")
    if module.CURL_SOURCES != tuple(spec["sources"]):
        findings.append("CURL_SOURCES disagrees with the curl arm's own table")
    if module.CURL_SOURCES != ("Bx", "By", "Bz"):
        findings.append(
            "the curl arm's step_D sources are no longer the B volumes; without an "
            "absorber Fields does not allocate H at all, so a binding that followed "
            "launch.SUB_STEPS would bind three Nones")
    if tuple(launch_module.SUB_STEPS["step_D"]["targets"]) != module.CURL_TARGETS:
        findings.append("the two SUB_STEPS tables disagree about step_D's TARGETS, "
                        "which they never have and which no clause here allows for")
    if module.DERIVE != 0:
        findings.append("DERIVE must be 0: step_D differences B directly")

    # 5. THE REPAIR DECLARATION, both halves of it.
    from meep_gpu import deposit_repair  # noqa: PLC0415
    if module.CARRIES_DEPOSIT_REPAIR is not True:
        findings.append("CARRIES_DEPOSIT_REPAIR is not True")
    if module.REPAIR_PATHS != (deposit_repair.PLAIN_PATH,):
        findings.append(f"REPAIR_PATHS is {module.REPAIR_PATHS}, not the plain path")
    text = open(os.path.join(
        API_ROOT, "meep_gpu/triton_kernels/no_pml_fused_electric_pair.py"),
        encoding="utf-8").read()
    if "carries_repair=CARRIES_DEPOSIT_REPAIR" not in text:
        findings.append("the seam clause is not passed the module's own flag")
    if "repair_paths=REPAIR_PATHS" not in text:
        findings.append("the seam clause is not passed the module's own repair paths")
    return {"leg": "transcription", "device": False, "passed": not findings,
            "findings": findings,
            "curl_block_statements": len(curl_block),
            "wall_block_statements": len(wall_block)}


# ---------------------------------------------------------------------------
# The predicate leg — no device
# ---------------------------------------------------------------------------

def _laptop_fixture(poles: int = 2, sigma: Optional[float] = SIGMA,
                    complex_storage: bool = False, offdiag: bool = False,
                    symmetry: Sequence[str] = (), thickness: int = 0,
                    stored: bool = True):
    """A ``(fields, pml)`` pair on NumPy, for the clauses that need no device."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    cell = (1.2, 1.2, 0.0)
    grid = Grid(resolution=12.0, cell_size=cell, dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
                symmetry=tuple(symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if offdiag:
        diagonal = {name: np.full(grid.shape, 2.25, dtype=np.float32)
                    for name in ("Ex", "Ey", "Ez")}
        inverse = {name: np.full(grid.shape, 1.0 / 2.25, dtype=np.float32)
                   for name in ("Ex", "Ey", "Ez")}
        rows = {"Ex": {"Ey": np.full(grid.shape, 0.05, dtype=np.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, rows)
    else:
        fields.set_background_eps(2.25)
    for index in range(poles):
        sigma_map = {"Ex": 0.30 + 0.05 * index, "Ey": 0.0 if index else 0.20,
                     "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma_map, grid,
            np.complex64 if complex_storage else np.float32))
    if sigma is not None:
        fields.set_d_conductivity(np.full(grid.shape, float(sigma), dtype=np.float32))
    if thickness:
        fields.enable_pml_storage()
    elif stored:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=({"x": thickness, "y": thickness}
                                    if thickness else 0))
    return fields, pml


class _Electric:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = np.array([1])
    _point_iy = np.array([1])
    _point_iz = np.array([0])


class _ScaledElectric(_Electric):
    is_integrated = False


class _ScaledElectricNoTable:
    """A NON-integrated electric source publishing NO deposit table AT ALL.

    ``hasattr(source, "_point_ix")`` is False — the driver's dense-fallback
    partition (``_inject_electric_through_conductivity``) — where
    :class:`_ElectricWithoutIndex` carries the attribute as ``None`` and is
    refused by the repair's own index clause instead. No in-tree source class is
    either; both stay because the clauses they exercise fail CLOSED on a
    duck-typed stranger.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = False


class _ElectricWithoutIndex:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = None


class _Magnetic:
    field_type = "B"
    component = "Hz"
    is_integrated = True
    _point_ix = np.array([1])
    _point_iy = np.array([1])
    _point_iz = np.array([0])


def predicate_leg() -> Dict[str, Any]:
    """Both directions, on real ``Fields``/``PML`` objects and a NumPy host.

    The array-module clause refuses everything here, so what this leg measures is the
    REASONS rather than the verdict: each row asserts that the named clause is present
    (or absent) among them. That is the discriminating question for a predicate, and it
    is answerable without a device.
    """
    rows: List[Dict[str, Any]] = []
    from meep_gpu.triton_kernels import no_pml_fused_electric_pair as module  # noqa: PLC0415

    def ask(label: str, needle: Optional[str], present: bool, **fixture):
        sources = fixture.pop("sources", (_Electric(),))
        fields, pml = _laptop_fixture(**fixture)
        verdict = module.no_pml_fused_electric_pair_coverage(fields, pml, sources)
        found = (needle is None or
                 any(needle in reason for reason in verdict.reasons))
        rows.append({"case": label, "needle": needle, "expected_present": present,
                     "found": bool(found), "passed": bool(found) is bool(present),
                     "covered": bool(verdict.covered),
                     "reasons": [reason for reason in verdict.reasons
                                 if "not cupy" not in reason][:6]})

    ask("an active absorber is refused", "an active PML layer is installed", True,
        thickness=2)
    ask("an active absorber also trips the plain repair's own clause", "SPLIT-FIELD",
        True, thickness=2)
    ask("stores_E False is refused", "stores_E", True, poles=0, sigma=None,
        stored=False)
    ask("an off-diagonal row is refused", "STENCIL", True, offdiag=True)
    ask("a mirror plane is refused", "mirror plane", True, symmetry=("Y",))
    ask("an undeclared source list is refused", "was not declared", True,
        sources=None)
    ask("an electric source with no deposit index is refused",
        "does not publish the index", True, sources=(_ElectricWithoutIndex(),))
    # THE 2026-09-01 LIFT, pinned in both directions: a NON-integrated electric
    # source that PUBLISHES its deposit table is no longer refused (the driver
    # replays the condinv rescale sparsely at those cells, identity everywhere
    # else), while the driver's dense fallback — a scaled source with NO deposit
    # table at all — is still refused by name.
    ask("a NON-INTEGRATED electric source publishing its deposit table is no "
        "longer refused (the 2026-09-01 lift)",
        "NOT integrated", False, sources=(_ScaledElectric(),))
    ask("a scaled electric source publishing NO deposit table is refused (the "
        "driver's dense whole-volume fallback)",
        "publishes NO deposit table", True, sources=(_ScaledElectricNoTable(),))
    # ...and the same source with NO conductivity never had the clause: the driver
    # takes the plain per-source injection loop, which writes only the deposit cells.
    ask("a NON-INTEGRATED electric source on a LOSSLESS run keeps the clause quiet",
        "NO deposit table", False, sigma=None, sources=(_ScaledElectricNoTable(),))
    ask("an integrated electric source keeps the rescale clause quiet",
        "NOT integrated", False, sources=(_Electric(),))
    ask("a magnetic source does not trip the electric seam clause",
        "is electric", False, sources=(_Magnetic(),))
    return {"leg": "predicate", "device": False,
            "passed": all(row["passed"] for row in rows),
            "findings": [row["case"] for row in rows if not row["passed"]],
            "rows": rows}


def repair_declaration_leg() -> Dict[str, Any]:
    """The second repair really admits this configuration, and refuses its twin.

    On real engine objects and with no device: what the carry legs below depend on is
    that ``deposit_repair.repairable(..., paths=REPAIR_PATHS)`` says yes to a plain
    layer and no to an active one, and that the DEFAULT declaration still says no to
    both — which is what makes ``REPAIR_PATHS`` a load-bearing declaration rather than
    a label.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.triton_kernels import no_pml_fused_electric_pair as module  # noqa: PLC0415

    plain_fields, plain_pml = _laptop_fixture()
    active_fields, active_pml = _laptop_fixture(thickness=2)
    rows = [
        {"case": "plain layer, the product's declared paths",
         "verdict": deposit_repair.repairable(plain_fields, "D", plain_pml,
                                              paths=module.REPAIR_PATHS),
         "expected": True},
        {"case": "plain layer, the DEFAULT declaration",
         "verdict": deposit_repair.repairable(plain_fields, "D", plain_pml),
         "expected": False},
        {"case": "active layer, the product's declared paths",
         "verdict": deposit_repair.repairable(active_fields, "D", active_pml,
                                              paths=module.REPAIR_PATHS),
         "expected": False},
        {"case": "the MAGNETIC seam, the product's declared paths",
         "verdict": deposit_repair.repairable(plain_fields, "B", plain_pml,
                                              paths=module.REPAIR_PATHS),
         "expected": False},
    ]
    for row in rows:
        covered, reasons = row.pop("verdict")
        row["covered"] = bool(covered)
        row["reasons"] = list(reasons)
        row["passed"] = bool(covered) is bool(row["expected"])
    return {"leg": "repair_declaration", "device": False,
            "passed": all(row["passed"] for row in rows),
            "findings": [row["case"] for row in rows if not row["passed"]],
            "rows": rows}


NO_DEVICE_LEGS = (transcription_leg, predicate_leg, repair_declaration_leg)


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------

def seed_values(cp, driver, seed: int, value_class: str) -> None:
    """Seed every settable volume in ONE value class, identically on every route."""
    rng = np.random.default_rng(seed)
    shape = tuple(driver.shape)

    def draw() -> np.ndarray:
        if value_class == "uniform":
            values = rng.uniform(-0.25, 0.25, size=shape)
        elif value_class == "signed_zero_lattice":
            # +0.0 and -0.0 on a random lattice. The two words differ (0x00000000 vs
            # 0x80000000), so a byte gate SEES the difference -- which is what
            # exercises every `tl.where(flag, 0.0, v)` in the wall clear and every
            # subtraction's sign convention in the pole chain.
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            # Below 2**-126: the words the two float32 subnormal policies disagree
            # about. The policy stamp says which one this process ran.
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover - VALUE_CLASSES is the whole domain
            raise ValueError(f"unknown value class {value_class!r}")
        return np.ascontiguousarray(values.astype(np.float32))

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(draw()))
    # E and H are DERIVED and `set_field` refuses them by name. Written in place here,
    # which is legal for a harness and is what makes the first step non-vacuous: E is
    # what `step_B` differences, and this product's own output.
    #
    # THE POLARIZATION HISTORY IS SEEDED TOO. Left at zero, `D - sum P` is `D` and the
    # whole unrolled chain becomes an identity every mutation of it reproduces.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
                 "f_cond_Bx", "f_cond_By", "f_cond_Bz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())
    for state in driver.fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is not None:
                    array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str, *,
                 electric: bool, sigma: Optional[float], components: Any,
                 poles: int, integrated: bool = True):
    """One NO-ABSORBER driver with a STORED E, seeded identically for every route.

    NO ``setup_pml`` CALL AT ALL, and that is the family's own precondition rather than
    a fixture choice: both certified halves refuse an active layer by name, and a
    fixture that set one up would report a family failure that was really a fixture
    failure.
    """
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    dimensions = sum(1 for extent in cell if extent > 0.0)
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, and a DIFFERENT one per component. Both halves of that are
    # measurements: a uniform material makes a mis-indexed coefficient a scaling many
    # wrong transcriptions reproduce, and a material that varies in SPACE but not in
    # COMPONENT makes `e0` and `e1` the same volume, so `s0 * e1` is a bitwise no-op.
    driver.set_epsilon_components({
        name: cp.asarray(np.ascontiguousarray(
            (2.25 + 0.30 * np.sin(index * np.float32(0.037 + 0.011 * offset))
             + 0.17 * offset).astype(np.float32)))
        for offset, name in enumerate(("Ex", "Ey", "Ez"))})
    # THE POLES ARE WHAT MAKE E STORED. Without at least one, `stores_E` is False on a
    # no-absorber run and `update_E` returns without writing -- the E half refuses that
    # by name and this product would have nothing to weld to.
    for pole in range(poles):
        driver.add_susceptibility(
            Susceptibility(frequency=1.0 + 0.3 * pole, gamma=0.1), 0.30 / (pole + 1))
    if sigma is not None:
        # A VARYING sigma, for the reason the epsilon above varies: a constant makes
        # `condfac`/`condinv` uniform and every coefficient read becomes a scaling a
        # wrong index reproduces.
        values = np.ascontiguousarray(
            (float(sigma) * (0.6 + 0.4 * np.sin(index * np.float32(0.023))))
            .astype(np.float32))
        driver.set_conductivity({name: cp.asarray(values) for name in components}
                                if components is not None else cp.asarray(values))
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR, injected BETWEEN the two halves
        # (driver.py:3294-3299). INTEGRATED by default; the formerly refused
        # NON-integrated conductive case is measured deliberately in
        # `lifted_refusal_leg` (with the retired whole-volume passes as its armed
        # control) rather than smuggled into a carry leg.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4,
                           "is_integrated": bool(integrated)})
    else:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it in
        # the B/H seam, not this one. Carrying one is what stops the source clause from
        # being tested only in its refusing direction.
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY. `FdtdDriver.step` consults `fastpath`
    # BEFORE the module-level sub-step functions this harness substitutes, and BOTH
    # no-absorber arms are WIRED -- so a driver left alone would step `step_D` and
    # `update_E` on Triton and the "array path" reference would be no such thing. The
    # per-route call counter re-checks that this held.
    driver.invalidate_fast_path()
    driver._fast_path = None          # noqa: SLF001 - the harness owns the route
    driver._fast_path_stale = False   # noqa: SLF001
    return driver


def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("_"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    if not fields.stores_E:
        raise AssertionError(
            "this driver does not store E, so update_E returns without writing and "
            "the constitutive half this gate exists for never runs")
    # EVERY POLARIZATION BUFFER IS PART OF THE COMPARISON. `update_P` is driven by the
    # array this repair rewrites (fields.drive_field returns the STORED E on this
    # branch), so a repair that got E right and P wrong would read as identical without
    # them.
    for index, state in enumerate(fields.polarizations):
        for attribute in ("P", "P_prev"):
            for component, array in sorted(
                    (getattr(state, attribute, None) or {}).items()):
                if array is not None:
                    found[f"pol{index}.{attribute}.{component}"] = array
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
    return ({"array": "<inventory>", "reason": "asymmetric", "names": only}
            if only else None)


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
    legitimate array-path calls with a fallback in the fused route, which is precisely
    the event this instrument exists to see.
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
    """The TWO separately certified Triton products this launch replaces.

    They are the arms ``launch.plan_step`` really selects on a no-absorber stored-E row
    today -- ``conductive no-PML`` (or ``no-PML``) on ``step_D`` and ``no-PML stored E``
    on ``update_E`` -- so this route is not a second model of the seam, it is the
    shipped one.

    THE THREE IN-SEAM PASSES STAY ON THE ARRAY PATH here, and one of them is the whole
    point: no certified Triton product owns ``zero_metal_D``, and this fused kernel
    carries it INLINE. Comparing against a route where the driver runs its own is what
    makes the inline wall a measured claim rather than a described one. All three are
    counted. THE INJECTION ALSO STAYS ON THE ARRAY PATH, which is what makes this the
    right oracle for the carry family: the certified kernels with the driver's own
    deposit between them.
    """
    from meep_gpu.triton_kernels import no_pml, no_pml_conductive  # noqa: PLC0415
    from meep_gpu.triton_kernels import no_pml_stored_e  # noqa: PLC0415

    curl = no_pml_conductive.plan_conductive_plain_curl(
        driver.fields, driver.pml, "step_D")
    which = "conductive no-PML"
    if curl is None:
        curl = no_pml.plan_plain_curl(driver.fields, driver.pml, "step_D")
        which = "no-PML"
    constitutive = no_pml_stored_e.plan_stored_e_constitutive(driver.fields, driver.pml)
    missing = [name for name, plan in
               ((f"{which} curl", curl),
                ("no-PML stored E constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so this "
            f"leg could not compare the fused launch against the certified products it "
            f"replaces")
    return Route({"step_D": curl, "update_E": constitutive}), which


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit.

    THE BRACKET IS THE SHIPPED ONE, and so is the path it declares.
    ``deposit_repair.LeadingRepairPlan`` / ``TrailingRepairPlan`` are the exact pair
    ``launch._install_fused_pair`` puts in these two slots, and ``REPAIR_PATHS`` is the
    product's own declaration -- a harness that assembled its own, or that passed the
    default split-field declaration, could not license the one that ships.

    ALL FIVE PASSES ARE REPLACED on this route. ``zero_metal_D`` is carried INLINE by
    the kernel; the two fills are refused by the fold clause and are no-ops without a
    mirror. Leaving any of them on the array path would double a pass the launch
    already performed.

    ``bracket=False`` is the NULL CONTROL and it must diverge: it is the same launch
    with the repair removed, which is the configuration the whole flag exists to forbid.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.triton_kernels import no_pml_fused_electric_pair as module  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "D")
    if not seam or not bracket:
        return Route({name: (plan if name == "step_D" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml, seam,
                                               "D", module.REPAIR_PATHS)
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, driver.fields,
                                                 driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_D"] = leading
    plans["update_E"] = trailing
    return Route(plans), leading


# ---------------------------------------------------------------------------
# One device leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, case, steps: int, product, *,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            electric: bool = False, bracket: bool = True,
            integrated: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    _label, cell, boundaries, _steps, sigma, components, poles = case
    seed = case_seed(name, str(cell), str(boundaries), value_class, str(components),
                     str(sigma), str(poles), str(integrated))
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric=electric,
                         sigma=sigma, components=components, poles=poles,
                         integrated=integrated)
            for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.no_pml_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "value_class": value_class, "seed": seed, "poles": poles,
        "sigma": sigma, "conductive_components": (list(components) if components
                                                  else ("all" if sigma else None)),
        "electric_source": bool(electric), "is_integrated": bool(integrated),
        "bracketed": bool(bracket), "fused_substituted": bool(install_fused),
        "host_mutation": None, "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_no_pml_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.no_pml_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["cond_flags"] = list(plan.cond)
        row["pole_counts"] = list(plan.counts)
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources),
                                         bracket=bracket)
        else:
            route = Route({})
        separate_plans, which_arm = separate_route(separate)
        row["separate_curl_arm"] = which_arm
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
        reference_opening = snapshot(cp, reference)
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
                "deposit_repairs": (leading.repairs if leading is not None else None),
            })
            divergence = versus_array or versus_separate
            if step == 1 or step % 5 == 0 or divergence is not None:
                log(f"  {name} [{value_class}] step {step}/{steps} "
                    f"identical={divergence is None} control={control is None} "
                    f"moved={len(step_moved)} launches={kernel.calls} "
                    f"repairs={leading.repairs if leading is not None else '-'} "
                    f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        reference_final = snapshot(cp, reference)
        reference_moved = moved(reference_opening, reference_final)
        row["arrays_total"] = len(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        # THE FLOOR IS RELATIVE TO THE ARRAY PATH. An absolute "every array moved"
        # floor fails a CORRECT kernel for properties of the CONFIGURATION; what is
        # evidence is an array the ARRAY PATH moves and this route does not, which is a
        # pass this weld swallowed.
        row["reference_never_moved"] = sorted(
            set(reference_final) - set(reference_moved) - set(material))
        row["inert_here_but_moving_on_the_array_path"] = sorted(
            (set(final) - set(ever_moved) - set(material)) & set(reference_moved))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(reference_final))
        row["private_scratch_on_the_array_route"] = sorted(
            key for key in vars(reference.fields) if key in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            key for key in vars(fused.fields) if key in PRIVATE_SCRATCH)
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
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
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False,
               require_array_reference: bool = True) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it exists to be "
            "is inert, and the thing it was meant to prove load-bearing is not")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg could "
            f"not have measured the fused launch: {row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if (require_launches_at_least is not None
            and (row.get("fused_kernel_launches") or 0) < require_launches_at_least):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"fewer than the {require_launches_at_least} this leg needs")
    if require_moved:
        swallowed = row.get("inert_here_but_moving_on_the_array_path") or []
        if swallowed:
            failures.append(
                f"the fused route left {swallowed} INERT while the array path moves "
                f"them: a pass this weld was supposed to carry did not run")
        reference_movers = set(row.get("reference_seam_outputs_moved") or ())
        if not reference_movers:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes on this "
                "configuration, so agreement with it is agreement about nothing")
        movers = set(row.get("seam_outputs_moved") or ())
        if movers != reference_movers:
            failures.append(
                f"the fused route's seam-output movement {sorted(movers)} is not the "
                f"array path's {sorted(reference_movers)}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if row.get("inventory_asymmetry_vs_array"):
        failures.append(
            f"the compared inventories differ by NAME between the fused route and the "
            f"array path: {row['inventory_asymmetry_vs_array']}")
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the seam "
            "carried nothing, so this leg measured the quiet case under a carry name")
    counts = row.get("launches") or {}
    fell_back = {key: value for key, value in counts.items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    if require_array_reference:
        # THE REFERENCE MUST REALLY BE THE ARRAY PATH. `fastpath` is consulted before
        # these functions and both no-absorber arms are WIRED, so a missing count here
        # means the reference stepped on Triton and the comparison compared two Triton
        # routes.
        steps = len(row.get("per_step") or ())
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} {seen} "
                    f"times, expected {steps}: the fast path was not off")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Kernel mutations
# ---------------------------------------------------------------------------

def shipped_source() -> str:
    return _function_source(
        "meep_gpu/triton_kernels/no_pml_fused_electric_pair.py", KERNEL_NAME)


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    THE DECORATOR IS PUT BACK, and its absence is not cosmetic.
    ``ast.get_source_segment`` on a ``FunctionDef`` returns the segment from ``def``
    onwards -- decorator lines are separate nodes and are NOT in it -- so the shipped
    source this harness extracts carries no ``@triton.jit``. A mutant compiled without
    it is a PLAIN PYTHON FUNCTION: ``CountingKernel.__getitem__`` then fails with
    ``TypeError: 'function' object is not subscriptable``, which is what the first
    device run of this gate reported after every product leg had already passed. It is
    asserted rather than assumed, because the failure mode if it silently stopped being
    needed is a mutant that runs as host Python and reports every defect as uncaught.
    """
    assert not source.lstrip().startswith("@"), (
        "the extracted body already carries a decorator; adding a second would "
        "compile a different object from the one this gate certifies")
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n@triton.jit\n")
    text = header + source.replace(KERNEL_NAME, kernel_name)
    # A MUTATION THAT DOES NOT PARSE IS NOT A MUTATION. Checked before any device work,
    # so a broken rewrite is reported as unarmed rather than as caught.
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_no_pml_electric_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_no_pml_electric_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _replace_all(source: str, before: str, after: str) -> Tuple[str, int]:
    hits = source.count(before)
    return source.replace(before, after), hits


def _chain(source: str, *pairs: Tuple[str, str]) -> Tuple[str, int]:
    """Several rewrites as ONE mutation, reporting the MINIMUM hit count.

    THE MINIMUM, not the sum: a mutation whose second edit matched nothing is only
    half planted, and a sum would report it as armed. Zero from any member is zero
    for the whole.
    """
    hits: List[int] = []
    for before, after in pairs:
        source, found = _replace_all(source, before, after)
        hits.append(found)
    return source, min(hits) if hits else 0


#: A mutation may only be declared ``null`` or ``unreached`` with a reason, and the
#: reason must be a MEASUREMENT rather than an expectation. Both entries below were
#: declared ``caught`` on this gate's third device run and came back UNCAUGHT; each
#: was then established to be arithmetically dead rather than merely unlucky, and the
#: establishment is what is written here.
MUTATION_EVIDENCE: Dict[str, str] = {
    "weld_reads_D_from_memory_instead_of_the_register": (
        "NULL, and the reason is a property of THIS weld rather than of welds in "
        "general. The kernel stores v0 to `f0 + idx` immediately before the "
        "constitutive half reads it, and the read is the SAME LANE'S OWN INDEX -- no "
        "cross-lane traffic, no barrier needed -- so memory and register carry the "
        "same word. The register substitution here is therefore a BANDWIDTH property, "
        "not an arithmetic one, which is exactly the opposite of the PML fused pair "
        "where the constitutive half reads `f_w` and `f` state the curl half has "
        "already overwritten. Measured uncaught over 4 complete steps on "
        "conductive_wall_xy, and the mutation is kept because the day the store moves "
        "below the constitutive read it becomes caught."),
    "ownership_mask_dropped_on_curl0": (
        "NULL, and it is a structural fact worth having found. On the D side "
        "`stepping._mask_non_owned_cells` and `stepping.zero_metal_D` touch EXACTLY "
        "the same (component, row) set: under BACKWARD a metallic axis masks the two "
        "curls whose targets are TANGENTIAL to it, and zero_metal_D clears those same "
        "two D components on that axis's row 0. So on any grid where the mask fires, "
        "the wall this product carries INLINE overwrites its result with zero before "
        "anything reads it -- and on a grid where the wall does not fire, the mask "
        "does not either (both are gated on the same METALLIC declaration). The "
        "mutation cannot be armed by choosing a different case, which is why it is "
        "declared here instead of re-scored. It is kept because it is live on the "
        "ARRAY path, where the injection sits between the two passes."),
}


def mutation_table() -> Tuple[Tuple[str, str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:
    """``(id, why it is armed, expectation, rewrite)``.

    THE WELD ITSELF IS THE FIRST TARGET. ``s0 = v0`` is the whole substitution this
    product makes, so a mutation that reads D from memory instead of from the register
    is the exact defect a careless transcription produces -- and on a run where the
    launch stores D before the constitutive half it is INVISIBLE except at the wall,
    where the register carries the cleared value and the memory does not yet.

    THE WALL is the second: this family carries ``zero_metal_D`` inline, and clearing
    the wrong component of an axis is a plane of wrong values with no crash.

    THE POLE CHAIN is the third, and the two rewrites there are the ones the certified
    arm's own gate arms: pre-summing the poles, and reversing their order.
    """
    return (
        ("weld_reads_D_from_memory_instead_of_the_register",
         "the whole substitution: `s0 = v0` is the weld, and a memory read is what an "
         "un-welded transcription writes",
         "null",
         lambda s: _replace_all(s, "        s0 = v0\n",
                                "        s0 = tl.load(f0 + idx, mask=live, other=0.0)\n")),
        ("weld_reads_the_wrong_components_register",
         "component 1's chain fed by component 0's register",
         "caught",
         lambda s: _replace_all(s, "        s1 = v1\n", "        s1 = v0\n")),
        ("wall_clears_the_normal_component_instead_of_the_tangential_pair",
         "zero_metal_D clears the two TANGENTIAL components of a metallic axis "
         "(stepping.py:2206-2247); clearing the normal one is a plane of wrong values",
         "caught",
         lambda s: _replace_all(
             s,
             "            v1 = tl.where(at_x, 0.0, v1)\n"
             "            v2 = tl.where(at_x, 0.0, v2)\n",
             "            v0 = tl.where(at_x, 0.0, v0)\n"
             "            v2 = tl.where(at_x, 0.0, v2)\n")),
        ("wall_runs_after_the_store_so_the_constitutive_reads_the_uncleared_value",
         "the array path clears D BEFORE update_E reads it (driver.py:3310 then :3313); "
         "a wall that reached only the STORE would leave the constitutive half reading "
         "the uncleared register, which is a plane of wrong E with a correct D beside it",
         "caught",
         lambda s: _chain(
             s,
             ("        if ZM_X:\n",
              "        pre1 = v1\n        pre2 = v2\n        if ZM_X:\n"),
             ("        s1 = v1\n", "        s1 = pre1\n"),
             ("        s2 = v2\n", "        s2 = pre2\n"))),
        ("poles_are_pre_summed",
         "`D - (P0 + P1)` is a different float32 number from `(D - P0) - P1`; the "
         "certified arm's gate caught this 20/20 at two poles",
         "caught",
         lambda s: _replace_all(
             s,
             "        if NP0 > 0:\n"
             "            s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)\n"
             "        if NP0 > 1:\n"
             "            s0 = s0 - tl.load(a1 + idx, mask=live, other=0.0)\n",
             "        if NP0 > 1:\n"
             "            s0 = s0 - (tl.load(a0 + idx, mask=live, other=0.0)\n"
             "                       + tl.load(a1 + idx, mask=live, other=0.0))\n"
             "        elif NP0 > 0:\n"
             "            s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)\n")),
        ("inv_eps_component_swapped",
         "component 0 scaled by component 1's inverse permittivity: a bitwise no-op "
         "unless each component carries its OWN epsilon volume, which build_driver "
         "gives them for exactly this reason",
         "caught",
         lambda s: _replace_all(
             s,
             "        tl.store(h0 + idx, s0 * tl.load(e0 + idx, mask=live, other=0.0), mask=live)\n",
             "        tl.store(h0 + idx, s0 * tl.load(e1 + idx, mask=live, other=0.0), mask=live)\n")),
        ("conductive_tail_drops_the_condinv_multiply",
         "`((v*cf) - curl) * ci` is three sequential in-place passes on the array path "
         "(stepping.py:1996-1998); dropping the last one is the classic conductive "
         "transcription error",
         "caught",
         lambda s: _replace_all(
             s,
             "            v0 = ((v0 * tl.load(cf0 + idx, mask=live, other=0.0)) - curl0) \\\n"
             "                * tl.load(ci0 + idx, mask=live, other=0.0)\n",
             "            v0 = (v0 * tl.load(cf0 + idx, mask=live, other=0.0)) - curl0\n")),
        ("curl_parentheses_flattened",
         "`dtdx * ((c_y - c) + (b - b_z))` is the array path's grouping "
         "(stepping._curl_from_operands); flattening it re-associates the sum",
         "caught",
         lambda s: _replace_all(
             s, "        curl0 = dtdx * ((c_y - c) + (b - b_z))\n",
             "        curl0 = dtdx * (c_y - c + b - b_z)\n")),
        ("ownership_mask_dropped_on_curl0",
         "stepping._mask_non_owned_cells zeroes the row MEEP's loop never touches",
         "null",
         lambda s: _replace_all(
             s,
             "            if BCY == METALLIC:\n"
             "                curl0 = tl.where(at_y, 0.0, curl0)\n"
             "            if BCZ == METALLIC:\n"
             "                curl0 = tl.where(at_z, 0.0, curl0)\n",
             "            if BCZ == METALLIC:\n"
             "                curl0 = tl.where(at_z, 0.0, curl0)\n")),
        ("D_store_dropped",
         "the array path leaves the stepped displacement in D and the next step's curl "
         "and update_P read it; a weld that fused the store away loses it",
         "caught",
         lambda s: _replace_all(s, "        tl.store(f0 + idx, v0, mask=live)\n", "")),
    )


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """Which case arms this mutation. The wall ones need a metallic axis."""
    if name.startswith("wall_"):
        return "conductive_wall_xy", CASES_BY_NAME["conductive_wall_xy"]
    if name.startswith("conductive_"):
        return "conductive_wall_xy", CASES_BY_NAME["conductive_wall_xy"]
    if name.startswith("poles_"):
        return "conductive_five_poles", CASES_BY_NAME["conductive_five_poles"]
    return DEFAULT_MUTATION_CASE, CASES_BY_NAME[DEFAULT_MUTATION_CASE]


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation,
                               "rewrite_hits": hits, "device": True}
        if hits == 0:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_no_pml_electric_D"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001 - an uncompilable mutant is a failure
            row["error"] = f"the mutant did not import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: DID NOT COMPILE")
            continue
        # A MUTANT THAT IS NOT A JIT HANDLE IS A MUTANT THAT MEASURES NOTHING: it would
        # be launched as host Python and every defect would report as uncaught.
        if not hasattr(mutant, "cache") and not hasattr(mutant, "run"):
            row["error"] = ("the mutant is not a triton.jit handle "
                            f"({type(mutant).__name__}); it would not launch")
            rows.append(row)
            log(f"  mutation {name}: NOT A JIT HANDLE")
            continue
        case_name, case = mutation_case_for(name)
        row["case"] = case_name
        # THE CARRY IS ON. A mutation scored without the deposit would leave the
        # repair's interaction with the mutated register unmeasured.
        leg = run_leg(cp, f"mutation:{name}", case, MUTATION_STEPS, product,
                      value_class="uniform", mutant=mutant, electric=True)
        row["leg"] = leg
        row["caught"] = leg.get("first_divergence") is not None
        row["fused_kernel_launches"] = leg.get("fused_kernel_launches")
        row["ptx_specializations"] = leg.get("ptx_specializations")
        row["pristine_ptx_specializations"] = len(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} (expected {expectation}) "
            f"on {case_name}")
    return rows


# ---------------------------------------------------------------------------
# Host mutations — the bindings, not the arithmetic
# ---------------------------------------------------------------------------

def _clear_wall_flags(_driver, plan):
    plan.zero_metal = (0, 0, 0)
    return plan


def _clear_cond_flags(_driver, plan):
    plan.cond = (0, 0, 0)
    return plan


def _swap_inv_eps(_driver, plan):
    slots = list(plan._inv_eps)  # noqa: SLF001 - the mutation is the point
    slots[0], slots[1] = slots[1], slots[0]
    plan._inv_eps = tuple(slots)  # noqa: SLF001
    return plan


class _OnePoleShort:
    """``plan._poles`` with component 0's LAST pole withheld from every launch.

    THE COUNT AND THE BINDING MOVE TOGETHER, and that is not tidiness. The plan
    asserts per launch that the resolved group length equals the compiled count
    (``no_pml_fused_electric_pair.py:906-910``) -- a real guard, and a correct one --
    so a mutation that lowered ``counts`` alone would trip it and RAISE instead of
    launching a chain one subtraction short. That is the plan refusing an inconsistent
    state, not the gate catching an arithmetic defect, and it was measured doing
    exactly that on this gate's second device run. Moving both compiles ``NP0`` one
    short AND resolves one fewer pointer, which is the defect a truncated unrolled
    chain actually is.
    """

    __slots__ = ("inner", "counts")

    def __init__(self, inner) -> None:
        self.inner = inner
        counts = list(inner.counts)
        counts[0] = max(0, counts[0] - 1)
        self.counts = tuple(counts)

    def arrays(self):
        groups = [list(group) for group in self.inner.arrays()]
        groups[0] = groups[0][:self.counts[0]]
        return groups


def _drop_a_pole(_driver, plan):
    shim = _OnePoleShort(plan._poles)  # noqa: SLF001 - the mutation is the point
    if shim.counts == plan.counts:
        raise AssertionError(
            "component 0 carries no pole to drop, so this mutation is unarmed on "
            "this case")
    plan._poles = shim  # noqa: SLF001
    plan.counts = shim.counts
    return plan


def host_mutation_table() -> Tuple[Tuple[str, str, str, Callable[..., Any]], ...]:
    return (
        ("wall_flags_cleared",
         "ZM_X/ZM_Y/ZM_Z compiled to zero on a metallic grid: the launch stops "
         "carrying zero_metal_D and the constitutive half reads uncleared wall cells",
         "caught", _clear_wall_flags),
        ("cond_flags_cleared",
         "COND0/1/2 compiled to zero on a conductive grid: the launch takes the "
         "lossless tail on a component that carries a sigma",
         "caught", _clear_cond_flags),
        ("inv_eps_bindings_swapped",
         "components 0 and 1 scaled by each other's inverse permittivity",
         "caught", _swap_inv_eps),
        ("one_pole_dropped_from_the_count",
         "NP0 one short: the unrolled chain stops one subtraction early",
         "caught", _drop_a_pole),
    )


def run_host_mutations(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    for entry in host_mutation_table():
        name, why, expectation, _apply = entry
        leg = run_leg(cp, f"host_mutation:{name}", case, MUTATION_STEPS, product,
                      value_class="uniform", host_mutation=entry, electric=True)
        caught = leg.get("first_divergence") is not None
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "caught": caught, "device": True, "leg": leg})
        log(f"  host mutation {name}: caught={caught} (expected {expectation})")
    return rows


# ---------------------------------------------------------------------------
# The lifted refusal — measured in both directions, not asserted
# ---------------------------------------------------------------------------

def _retired_whole_volume_inject(self, electric, source_time):
    """The RETIRED composition of ``_inject_electric_through_conductivity``, verbatim.

    The whole-volume difference passes the driver replaced on 2026-09-01: snapshot the
    target component, inject, then ``array -= before; array *= condinv;
    array += before`` over the WHOLE volume. Kept HERE, in the gate that measures the
    lift, as the armed control's body — it is the composition whose ``-0.0``
    canonicalisation (and, on device, whose subnormal flushing) the refusal priced,
    and the control MUST reproduce that divergence or the lift measured nothing.
    Integrated sources are injected point-wise first, exactly as the retired driver
    did and today's does.
    """
    integrated = [source for source in electric if source.is_integrated]
    scaled = [source for source in electric if not source.is_integrated]
    for source in integrated:
        source.inject(self.fields, source_time)
    if not scaled:
        return
    by_target: Dict[str, list] = {}
    for source in scaled:
        by_target.setdefault("D" + source.component[1], []).append(source)
    before = {}
    for name in sorted(by_target):
        if self.fields.condinv_for(name) is None:
            continue
        before[name] = getattr(self.fields, name).copy()
    for source in scaled:
        source.inject(self.fields, source_time)
    for name, prev in before.items():
        array = getattr(self.fields, name)
        array -= prev
        array *= self.fields.condinv_for(name)
        array += prev


def lifted_refusal_leg(cp, product) -> Dict[str, Any]:
    """The FORMERLY refused configuration, carried — with the retired passes armed.

    THE THREE-LEG SIGNED-ZERO PROTOCOL, re-run against the lifted clause. Until
    2026-09-01 this leg's precursor (``priced_refusal``) ran the NON-integrated
    conductive case around the predicate and REQUIRED the divergence the refusal
    priced. The driver now replays the condinv rescale sparsely at the published
    deposit cells, the clause is lifted, and the burden of proof inverts:

    1. the lifted case (non-integrated, conductive) must be ADMITTED by the shipped
       predicate and BYTE-IDENTICAL to the array path at every step, through the
       full carry protocol (bracket on, launches counted, certified-singles route
       beside it);
    2. its two companions from the recorded protocol — the INTEGRATED source on the
       same conductive case and the same scaled source on the LOSSLESS case — must
       stay byte-identical exactly as they always were;
    3. the RETIRED whole-volume composition, replayed verbatim on a third driver
       against today's array path, MUST still diverge on the same signed-zero seed.
       That control is what separates "the driver change made the routes agree"
       from "this seed stopped discriminating": a lift whose control cannot diverge
       measured nothing, and every leg above it would pass vacuously.
    """
    row: Dict[str, Any] = {"leg": "lifted_refusal:non_integrated_conductive",
                           "device": True, "armed": True, "legs": {}}
    case = CASES_BY_NAME["conductive_wall_xy"]
    try:
        # -- 1. the lifted case, through the FULL carry protocol -----------------
        carried = run_leg(cp, "lifted:non_integrated_conductive", case, CARRY_STEPS,
                          product, value_class="signed_zero_lattice", electric=True,
                          integrated=False)
        passed, failures = verdict_of(carried, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        carried["passed"], carried["failures"] = passed, failures
        row["legs"]["lifted_non_integrated"] = carried

        # -- 2. the two companions of the recorded three-leg protocol ------------
        integrated = run_leg(cp, "lifted:integrated_companion", case, CARRY_STEPS,
                             product, value_class="signed_zero_lattice",
                             electric=True, integrated=True)
        passed, failures = verdict_of(integrated, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        integrated["passed"], integrated["failures"] = passed, failures
        row["legs"]["integrated_companion"] = integrated

        lossless = run_leg(cp, "lifted:lossless_companion",
                           CASES_BY_NAME["lossless_wall_xy"], CARRY_STEPS,
                           product, value_class="signed_zero_lattice",
                           electric=True, integrated=False)
        passed, failures = verdict_of(lossless, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        lossless["passed"], lossless["failures"] = passed, failures
        row["legs"]["lossless_companion"] = lossless

        # -- 3. the ARMED CONTROL: the retired passes must still diverge ---------
        row["legs"]["retired_control"] = _retired_passes_control(cp, case)

        failures = []
        for name in ("lifted_non_integrated", "integrated_companion",
                     "lossless_companion"):
            if not row["legs"][name].get("passed"):
                failures.append(f"{name}: {row['legs'][name].get('failures')}")
        control = row["legs"]["retired_control"]
        if not control.get("diverges_at_some_step"):
            failures.append(
                "the RETIRED whole-volume composition did NOT diverge from the "
                "sparse array path on any step: the control is disarmed and the "
                "lift's byte-identity requirement was vacuous on this seed")
        if control.get("error"):
            failures.append(f"retired_control: {control['error']}")
        row["failures"] = failures
        row["passed"] = not failures
        log(f"  lifted refusal: carried={row['legs']['lifted_non_integrated'].get('passed')} "
            f"integrated={row['legs']['integrated_companion'].get('passed')} "
            f"lossless={row['legs']['lossless_companion'].get('passed')} "
            f"retired_control_diverges={control.get('diverges_at_some_step')} "
            f"at step {control.get('diverged_at_step')}")
        return row
    except Exception as exc:  # noqa: BLE001 - a raised leg is a recorded leg
        row["error"] = repr(exc)
        row["passed"] = False
        return row


def _retired_passes_control(cp, case) -> Dict[str, Any]:
    """Replay the retired whole-volume composition beside today's array path.

    TWO ARRAY-PATH DRIVERS, identical in every way except the injection
    composition: no fused plan is installed on either, so what diverges is the
    driver change ALONE — the exact delta the clause was lifted on. The
    divergence's word classes are recorded (signed-zero pairs, and on device the
    subnormals the whole-volume passes flush) so the artifact says WHAT the
    retired composition destroyed, not merely that something moved.
    """
    import types  # noqa: PLC0415

    _label, cell, boundaries, _steps, sigma, components, poles = case
    seed = case_seed("lifted_refusal_control", str(cell))
    made = [build_driver(cp, cell, boundaries, seed, "signed_zero_lattice",
                         electric=True, sigma=sigma, components=components,
                         poles=poles, integrated=False)
            for _ in range(2)]
    reference, retired = made
    row: Dict[str, Any] = {"leg": "retired_control", "device": True, "armed": True}
    try:
        retired._inject_electric_through_conductivity = types.MethodType(
            _retired_whole_volume_inject, retired)
        row["per_step"] = []
        for step in range(1, CARRY_STEPS + 1):
            reference.step()
            retired.step()
            cp.cuda.runtime.deviceSynchronize()
            divergence = first_divergence(snapshot(cp, retired),
                                          snapshot(cp, reference))
            row["per_step"].append({"step": step,
                                    "retired_vs_array": divergence})
            if divergence is not None and "diverged_at_step" not in row:
                row["diverged_at_step"] = step
                row["divergence"] = divergence
                row["word_classes"] = _divergence_word_classes(
                    cp, retired, reference)
        row["diverges_at_some_step"] = any(
            entry["retired_vs_array"] is not None for entry in row["per_step"])
        return row
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
        return row
    finally:
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


def _divergence_word_classes(cp, retired, reference) -> Dict[str, Any]:
    """Classify every differing word pair across the two drivers' stored volumes.

    The retired passes' two known casualties are the SIGNED ZERO (``x - x`` is
    ``+0.0``) and, on device arithmetic that flushes, the SUBNORMAL. Any word pair
    outside those classes is recorded as ``other`` — a finding, not a pass/fail
    input, because the control's job is to diverge, and HOW it diverged is
    evidence about the retired composition rather than about the product.
    """
    classes = {"signed_zero_pair": 0, "subnormal_flush": 0, "other": 0}
    left_state = snapshot(cp, retired)
    right_state = snapshot(cp, reference)
    for name, left in left_state.items():
        right = right_state.get(name)
        if right is None or left.shape != right.shape:
            continue
        differ = np.flatnonzero(left != right)
        for index in differ[:4096]:
            lw = int(left.reshape(-1)[index])
            rw = int(right.reshape(-1)[index])
            if {lw, rw} <= {0x00000000, 0x80000000}:
                classes["signed_zero_pair"] += 1
            elif (lw & 0x7F800000) == 0 or (rw & 0x7F800000) == 0:
                classes["subnormal_flush"] += 1
            else:
                classes["other"] += 1
    return classes


# ---------------------------------------------------------------------------
# Refusals, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and three it must ADMIT."""
    rows: List[Dict[str, Any]] = []
    for name, case_name, electric, integrated, sources_of, needle, admit in (
        ("integrated_electric_source_is_admitted", "conductive_wall_xy", True, True,
         None, None, True),
        ("magnetic_source_only_is_admitted", "conductive_wall_xy", False, True,
         None, None, True),
        ("lossless_grid_is_admitted", "lossless_periodic", True, True,
         None, None, True),
        # The 2026-09-01 lift, on the device's own objects: the formerly refused
        # scaled source (which publishes its deposit table) is ADMITTED, and the
        # driver's dense fallback — a scaled source with NO table — stays refused.
        ("non_integrated_electric_source_on_a_conductive_run_is_admitted",
         "conductive_wall_xy", True, False, None, None, True),
        ("scaled_source_publishing_no_deposit_table", "conductive_wall_xy",
         True, True, lambda d: (_ScaledElectricNoTable(),),
         "publishes NO deposit table", False),
        ("undeclared_source_list", "conductive_wall_xy", True, True,
         lambda d: None, "was not declared", False),
    ):
        case = CASES_BY_NAME[case_name]
        _label, cell, boundaries, _steps, sigma, components, poles = case
        driver = build_driver(cp, cell, boundaries, 11, "uniform", electric=electric,
                              sigma=sigma, components=components, poles=poles,
                              integrated=integrated)
        try:
            sources = (tuple(driver._sources) if sources_of is None
                       else sources_of(driver))
            verdict = product.no_pml_fused_electric_pair_coverage(
                driver.fields, driver.pml, sources)
            built = product.plan_no_pml_fused_electric_pair(
                driver.fields, driver.pml, sources)
            row = {"refusal": name, "covered": bool(verdict.covered),
                   "reasons": list(verdict.reasons),
                   "plan_is_None": built is None, "admits": admit}
            if admit:
                row["passed"] = bool(verdict.covered) and built is not None
            else:
                row["passed"] = (not verdict.covered) and built is None and any(
                    needle in reason for reason in verdict.reasons)
            rows.append(row)
            log(f"  refusal {name}: passed={row['passed']}")
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()
    return rows


def environment(cp: Any = None) -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    out: Dict[str, Any] = {
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "env": {key: os.environ.get(key) for key in
                ("CUDA_VISIBLE_DEVICES", "CUPY_CACHE_DIR", "TRITON_CACHE_DIR",
                 "MEEP_GPU_SUBNORMAL_POLICY")},
    }
    if cp is not None:
        try:
            out["cupy"] = cp.__version__
            out["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception as exc:  # noqa: BLE001
            out["cupy_error"] = repr(exc)
        try:
            import triton  # noqa: PLC0415
            out["triton"] = triton.__version__
        except Exception as exc:  # noqa: BLE001
            out["triton_error"] = repr(exc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_no_pml_fused_electric_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument("--subnormal-policy", default="keep",
                        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO "
                             "before the first device compile")
    args = parser.parse_args(argv)

    out = args.out
    artifact = out if out.endswith(".json") else os.path.join(out, "gate.json")
    if not out.endswith(".json"):
        os.makedirs(out, exist_ok=True)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_no_pml_fused_electric_pair",
        "product": PRODUCT_MODULE,
        "kernel": KERNEL_NAME,
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "block": "kernels.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "value_classes": list(VALUE_CLASSES),
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "host_mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in NO_DEVICE_LEGS:
        started = time.time()
        try:
            row = leg()
        except Exception as exc:  # noqa: BLE001 - a leg that cannot run is a failure
            row = {"leg": getattr(leg, "__name__", str(leg)), "device": False,
                   "error": repr(exc), "passed": False, "findings": [repr(exc)]}
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, artifact)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal in "
            "this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation or "
                        "refusal has run: this artifact releases nothing"],
        }
        save(payload, artifact)
        log(f"\nno-device verdict: {payload['passed']}  ->  {artifact}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    product = importlib.import_module(PRODUCT_MODULE)
    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "cases": {case[0]: {"cell": list(case[1]), "steps": case[3],
                            "sigma": case[4],
                            "components": (list(case[5]) if case[5] else "all"),
                            "poles": case[6]} for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_carry": CARRY_STEPS,
        "steps_per_mutation": MUTATION_STEPS,
        "value_classes": list(VALUE_CLASSES),
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for case in CASES:
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"quiet:{case[0]}", case, case[3], product,
                          value_class=value_class)
            passed, failures = verdict_of(row, require_launches=case[3])
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{case_name}", case, CARRY_STEPS, product,
                          value_class=value_class, electric=True)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{case_name}", case, CARRY_STEPS, product,
                      electric=True, bracket=False)
        # REQUIRES DIVERGENCE. A bracket that changes nothing is not load-bearing. The
        # launch floor is "at least one" and the vacuity census is off: a leg that must
        # diverge STOPS at the first divergent step, so an exact launch count and a
        # whole-run moved-state census are statements about steps that never ran.
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False,
                                      require_array_reference=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it consumes "
                      "a pre-injection D and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== armed harness mutation: the plan is built but never installed ===")
    case = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    row = run_leg(cp, "armed:no_substitution", case, 3, product,
                  install_fused=False, electric=True)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== the LIFTED REFUSAL: three legs + the retired-passes control ===")
    payload["lifted_refusal"] = lifted_refusal_leg(cp, product)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.no_pml_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    for _row in payload["mutations"]:
        if _row["expectation"] != "caught":
            _row["evidence"] = MUTATION_EVIDENCE.get(_row["mutation"])
    save(payload, artifact)

    log("\n=== armed host mutations ===")
    payload["host_mutations"] = run_host_mutations(cp, product)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, artifact)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])
    lifted_ok = bool(payload["lifted_refusal"].get("passed"))

    #: The only expectations a mutation may declare. `caught` needs no justification;
    #: `null` and `unreached` each need an entry in MUTATION_EVIDENCE, so a rewrite
    #: that failed to catch cannot be excused by relabelling it.
    VOCABULARY = ("caught", "null", "unreached")

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits", 1):
            return False
        expectation = row["expectation"]
        if expectation not in VOCABULARY:
            return False
        if expectation != "caught" and not MUTATION_EVIDENCE.get(row["mutation"]):
            return False
        return bool(row.get("caught")) is (expectation == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    host_ok = all(bool(row.get("caught")) is (row["expectation"] == "caught")
                  for row in payload["host_mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and lifted_ok and mutation_ok and host_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"],
        "reasons": ([] if payload["passed"] else
                    [f"device legs ok: {device_ok}",
                     f"refusals ok: {refusal_ok}",
                     f"the lifted refusal ok: {lifted_ok}",
                     "kernel mutations whose measured outcome did not match their "
                     "declared expectation: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row))),
                     f"host mutations ok: {host_ok}"]),
        "host": "the measurement machine; every device row above ran there",
    }
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)
    except Exception as error:  # noqa: BLE001 - a stamp that cannot run is recorded
        payload["provenance_error"] = repr(error)
    save(payload, artifact)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: {payload['passed']}  ->  {artifact}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

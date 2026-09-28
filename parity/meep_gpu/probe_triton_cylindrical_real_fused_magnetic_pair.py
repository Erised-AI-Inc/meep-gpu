"""Byte gate for the Dcyl m = 0 fused magnetic pair: ``step_B`` welded into ``update_H``.

DEVICE STATUS: **RELEASED 2026-08-20** on the GPU host, RTX A6000 GPU 6, Triton 3.1.0 /
    CuPy 13.5.1, ``keep`` subnormal policy —
    ``parity/meep_gpu/results/triton_cylindrical_real_fused_magnetic_pair_2026-08-20/``.
    4/4 device cases bit-identical over 10 complete steps against both oracles, 2/2
    armed harness mutations refused, 9/10 kernel mutations caught with the tenth a
    confirmed null, 3/3 refusals.

    THE FIRST RUN FOUND A DEFECT IN THIS HARNESS, not in the kernel.
    ``m5_ownership_mask_dropped`` came back UNCAUGHT because the fixture left the
    ELECTRIC constitutive history at zero, which makes ``Ep`` exactly ``+0.0`` on both
    planes the ownership mask touches — so the mask wrote a value already present and
    the mutation could not fail. :func:`build_driver` now seeds ``f_w_E*`` and
    :func:`mask_observability_leg` measures both arms.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair.cylindrical_real_fused_magnetic_pair_coverage`
admits, the array-path radial prefix followed by ONE launch of
``cyl_real_fused_curl_constitutive_B`` leaves the engine in a state that is
BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_B`` / ``fill_symmetry_bc_B`` /
    ``zero_metal_B`` / ``fill_folded_far_ghosts_B`` / ``update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the cylindrical curl
    (``cylindrical_triton.CylindricalCurlPlan`` on ``step_B``) and the cylindrical
    constitutive (``cylindrical_triton.plan_cylindrical_constitutive`` on side
    ``H``), with ``zero_metal_B`` on the array path in that route because no Triton
    product owns the wall clear,

over every allocated volume: the primaries, the split-field PML auxiliaries and the
constitutive ``f_w`` history. Comparison is on the uint32 view. ``allclose`` appears
nowhere.

WHAT THE GATE REFUSES TO INFER.

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array must
  MOVE during the leg; a leg whose magnetic trio is frozen in all three routes is
  armed and must be caught by the moved-state census.
* **"The two fill passes are inert" is a measurement, not a reading of another
  module's guard.** :data:`~...INERT_PASSES` names them, :func:`inert_passes_leg`
  executes both on a randomised state of an ADMITTED Dcyl grid and requires that not
  one word moves, and the device legs additionally ABSORB them into the fused
  launch — so a fill that ever did work on such a grid would show up as a byte
  divergence and not as a quiet difference between two routes that both ran it.
* **A branch whose constexpr is structurally false is not a branch a mutation may be
  scored on.** ``ZM_X`` and ``ZM_Y`` cannot be true on any admitted Dcyl grid
  (:func:`structural_walls_leg` measures that over a sweep of shapes rather than
  asserting it), so no mutation rewrites them; ``ZM_Z`` is true on every one, and
  that is the row the wall-clear mutation targets.

===========================================================================
THE PREFIX IS UPSTREAM OF THE SEAM, and this gate compares like with like
===========================================================================

``cyl_pml_curl_step`` consumes a radial prefix scan the HOST computes; that decision
belongs to :mod:`~meep_gpu.triton_kernels.cylindrical_triton` and is unchanged here.
What matters for fusion is that the prefix runs BEFORE the curl, so no launch
straddles it. Both the separate route and the fused route compute the same prefix
through the same shipped function; what the fusion removes is the SECOND device
launch — the constitutive one — and the ``B`` round trip between them. The launch
counter in this gate counts the FUSED kernel only, so the count it reports is
one per step, not one per device call in the step.

Usage::

    # laptop, no CUDA, no Triton — the legs that do not need a device
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_cylindrical_real_fused_magnetic_pair.py \\
        --no-device --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_cylindrical_real_fused_magnetic_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SEED = 20260820

#: (name, cell (r, -, z), resolution, courant, steps). Every case is Dcyl at m = 0
#: with a METALLIC z declaration — which is not a choice: ``_boundary_kinds`` must
#: resolve to ``('axis', 'periodic', 'metallic')`` or the curl predicate refuses the
#: grid outright, so every admitted configuration is walled on z and ``ZM_Z`` fires
#: on every one. :func:`structural_walls_leg` measures that rather than asserting it.
#:
#: THE COURANT NUMBERS ARE SWEPT and three of the four are not powers of two: the
#: PML coefficients are the family of non-representable multiplicands the curl's
#: association mutations need in order to be visible at all.
CASES: Tuple[Tuple[str, Tuple[float, float, float], float, float, int], ...] = (
    ("square_res10", (2.0, 0.0, 2.0), 10.0, 0.5, 10),
    ("tall_thin", (3.2, 0.0, 1.1), 10.0, 0.3141592653589793, 10),
    ("short_wide", (1.1, 0.0, 3.2), 10.0, 0.37, 10),
    ("coarse", (2.6, 0.0, 2.6), 6.0, 0.4472135954999579, 10),
)

#: Steps per armed mutation.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3281-3289).
#: FIVE names, of which the product CARRIES three (:data:`REPLACES`) and the other
#: two are inert on every admitted configuration. The fused route absorbs all five,
#: so a fill that ever did work on such a grid becomes a byte divergence here.
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: The names that MUST appear in the dynamic state inventory.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds. A set
#: of names that matches nothing disables BOTH the vacuity floor and the material
#: check, silently; ``_assert_material_names_are_real`` is what stops a rename doing
#: it again.
MATERIAL = ("eps", "inv_eps")

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input and "
            f"material_changed would never fire. Fix the names, do not loosen the "
            f"floor.")


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
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
    names = (
        "meep_gpu/triton_kernels/cylindrical_real_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/cylindrical_triton.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_cylindrical_real_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_SOURCE_FILE = {
    "cyl_pml_curl_step": "cylindrical_triton.py",
    "constitutive_step": "kernels.py",
    "fused_curl_constitutive_B": "kernels.py",
    "cyl_real_fused_curl_constitutive_B": "cylindrical_real_fused_magnetic_pair.py",
}


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where importing ``kernels.py`` raises, so the transcription check bites
    at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is the
    PARENTHESISATION, and unparsing re-derives minimal parentheses. The docstring IS
    stripped: these bodies document the branches they do NOT carry, so a text search
    over the raw source finds those names in prose and reports them as live code.
    """
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels", _SOURCE_FILE[name])
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    segment = ast.get_source_segment(text, node)
    if segment is None:  # pragma: no cover - only on a source-less module
        raise AssertionError(f"{name}'s source segment is unavailable")
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
    return "\n".join(lines)


def _statements(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text."""
    lines = source.splitlines(keepends=True)
    stripped = [line.split("#", 1)[0].strip() for line in lines]
    target = [item.strip() for item in before]
    for start in range(len(lines) - len(target) + 1):
        if stripped[start:start + len(target)] != target:
            continue
        raw = lines[start]
        indent = raw[:len(raw) - len(raw.lstrip())]
        block = "".join(indent + item.strip() + "\n" for item in after)
        return ("".join(lines[:start]) + block
                + "".join(lines[start + len(target):]), 1)
    return source, 0


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

#: The cylindrical curl's arithmetic, whose grouping decides float32 bits.
CURL_LINES = (
    "curl0 = dtdx * ((c_p - c) + (b - b_z))",
    "curl1 = dtdx * ((a_z - a) + (c - c_r))",
    "curl2 = dtdx * ((b_r - b) + (a - a_p))",
    "n0 = ((p0 * km_y) - curl0) * si_y",
    "n1 = ((p1 * km_z) - curl1) * si_z",
    "n2 = ((p2 * km_x) - curl2) * si_x",
    "v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z",
    "v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x",
    "v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y",
)

#: The m = 0 substitutions the cylindrical curl makes and this body must keep.
CYLINDRICAL_LINES = (
    "curl2 = dtdx * (tl.load(pfx + idx + nyz, mask=live, other=0.0)",
    "- tl.load(pfx + idx, mask=live, other=0.0))",
    "at_r, at_z = i == 0, k == 0",
    "v0 = tl.where(at_r, 0.0, v0)",
)

#: The wall clear, byte-copied from ``kernels.fused_curl_constitutive_B``'s ZM block
#: — the same three rows ``stepping._zero_metal`` writes for the B family.
ZERO_METAL_LINES = (
    "v0 = tl.where(at_x, 0.0, v0)",
    "v1 = tl.where(at_y, 0.0, v1)",
    "v2 = tl.where(at_z, 0.0, v2)",
)

#: The constitutive half, byte-copied from the same function.
CONSTITUTIVE_LINES = (
    "prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
    "src0 = v0",
    "a0 = tl.load(h0 + idx, mask=live, other=0.0)",
    "a0 = a0 + kp_0 * src0",
    "a0 = a0 - km_0 * prev0",
    "a1 = a1 + kp_1 * src1",
    "a1 = a1 - km_1 * prev1",
    "a2 = a2 + kp_2 * src2",
    "a2 = a2 - km_2 * prev2",
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    fused = _statements(_shipped_text("cyl_real_fused_curl_constitutive_B"))
    curl = _statements(_shipped_text("cyl_pml_curl_step"))
    plain_pair = _statements(_shipped_text("fused_curl_constitutive_B"))
    constitutive = _statements(_shipped_text("constitutive_step"))

    findings: List[str] = []
    for line in CURL_LINES + CYLINDRICAL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in curl:
            findings.append(
                f"cylindrical_triton.cyl_pml_curl_step no longer contains {line!r}; "
                f"the fused body's transcription source has moved")
    for line in ZERO_METAL_LINES + CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
        if line not in plain_pair:
            findings.append(
                f"kernels.fused_curl_constitutive_B no longer contains {line!r}; the "
                f"wall clear / constitutive transcription source has moved")
    # And the constitutive grouping traces one step further back, to the sub-step
    # kernel the plain pair itself copied.
    for line in ("a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"):
        if line not in constitutive:
            findings.append(
                f"kernels.constitutive_step no longer contains {line!r}")

    # NO BOUNDARY CONSTEXPRS. The Dcyl declaration is compiled in; a BC branch here
    # would mean this body serves a declaration the predicate refuses.
    for needle in ("BCX", "BCY", "BCZ", "PERIODIC", "METALLIC", "MIRROR"):
        if any(needle in line for line in fused):
            findings.append(
                f"the fused body mentions {needle!r}: this kernel serves exactly the "
                f"Dcyl triple ('axis', 'periodic', 'metallic') and compiles it in")
    # The wall clear must sit AFTER the m = 0 axis rule and BEFORE the stores.
    try:
        axis_rule = fused.index("v0 = tl.where(at_r, 0.0, v0)")
        clear = fused.index("v0 = tl.where(at_x, 0.0, v0)")
        store = fused.index("tl.store(f0 + idx, v0, mask=live)")
        constitutive_read = fused.index("src0 = v0")
    except ValueError as exc:  # pragma: no cover - the line checks above fire first
        findings.append(f"could not locate the ordering landmarks: {exc}")
    else:
        if not axis_rule < clear < store < constitutive_read:
            findings.append(
                f"the order is axis rule {axis_rule}, wall clear {clear}, store "
                f"{store}, constitutive read {constitutive_read}; the driver runs "
                f"step_B (which ends with the axis rules), then zero_metal_B, then "
                f"update_H, and both consumers must see the cleared register")
    return {
        "leg": "transcription", "device": False,
        "fused_statements": len(fused),
        "curl_lines_checked": len(CURL_LINES) + len(CYLINDRICAL_LINES),
        "zero_metal_lines_checked": len(ZERO_METAL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "findings": findings, "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — are the two omitted passes really inert?
# ---------------------------------------------------------------------------

def _dcyl(cell, resolution, courant, xp=None):
    """A real Dcyl m = 0 Grid/Fields/PML triple.

    ``z`` is declared METALLIC explicitly: ``Grid``'s own default is periodic and the
    curl predicate requires the Dcyl triple ``('axis', 'periodic', 'metallic')``. The
    r entry is left UNSET because ``Grid`` refuses one — the r axis carries its own
    boundary pair (the axis at r = 0, a PEC at r_max).
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    kwargs = {"xp": xp} if xp is not None else {}
    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=0,
                boundaries={"z": "metallic"}, courant=courant, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    thickness = {"x": (0, 0.3), "z": 0.3}
    return grid, fields, PML(grid=grid, thickness=thickness)


def _seed(fields, label: str, names: Sequence[str]) -> None:
    """Fill the named volumes from a DIGEST-derived seed.

    NOT ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``, so a
    hash-seeded fixture is a different fixture every process and a failing case
    cannot be replayed.
    """
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big"))
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-0.5, 0.5, size=array.shape).astype(array.dtype)


def _words(fields, names: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = np.ascontiguousarray(
                np.asarray(array, dtype=np.float32)).view(np.uint32).ravel().copy()
    return out


ALL_VOLUMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
               "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy",
               "fu_Dz", "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def inert_passes_leg() -> Dict[str, Any]:
    """``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B``, EXECUTED and measured.

    The product's :data:`REPLACES` names three driver passes and the seam has five.
    The other two are omitted on the claim that they cannot execute on a grid this
    predicate admits. That claim is a reading of another module's guard until
    somebody runs them, so this leg runs them: a randomised state on an admitted Dcyl
    grid, both passes applied, every allocated volume compared as uint32.

    NON-VACUITY: a leg where nothing was seeded would report "no word moved" about an
    array of zeros. Every compared volume is required to be non-constant before the
    passes run, and the CONTROL — ``zero_metal_B``, which is NOT inert on these grids
    — is required to move exactly the plane it should.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_magnetic_pair as product)

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, cell, resolution, courant, _steps in CASES:
        grid, fields, pml = _dcyl(cell, resolution, courant)
        verdict = product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, ())
        residual = [reason for reason in verdict.reasons if "cupy" not in reason]
        if residual:
            findings.append(f"{name}: the predicate refuses this grid for a reason "
                            f"other than the NumPy host: {residual}")
            continue
        _seed(fields, f"inert|{name}", ALL_VOLUMES)
        before = _words(fields, ALL_VOLUMES)
        constant = sorted(key for key, value in before.items()
                          if len(set(value.tolist())) <= 1)
        if constant:
            findings.append(f"{name}: {constant} are constant before the passes run; "
                            f"this leg would report inertness about nothing")
        for pass_name in product.INERT_PASSES:
            getattr(stepping, pass_name)(fields)
        after = _words(fields, ALL_VOLUMES)
        moved = sorted(key for key in before if not np.array_equal(before[key],
                                                                   after[key]))
        # THE CONTROL: the pass this product DOES carry must move something on the
        # same grid, or the instrument is measuring a frozen state rather than two
        # inert passes.
        stepping.zero_metal_B(fields)
        control = _words(fields, ALL_VOLUMES)
        control_moved = sorted(key for key in after
                               if not np.array_equal(after[key], control[key]))
        rows.append({"case": name, "shape": list(grid.shape),
                     "inert_passes": sorted(product.INERT_PASSES),
                     "moved_by_the_inert_passes": moved,
                     "moved_by_zero_metal_B": control_moved,
                     "walls": list(product.zero_metal_axes(grid))})
        if moved:
            findings.append(
                f"{name}: {sorted(product.INERT_PASSES)} moved {moved} — they are NOT "
                f"inert on this grid and REPLACES may not omit them")
        if not control_moved:
            findings.append(
                f"{name}: zero_metal_B moved nothing either, so this leg cannot tell "
                f"an inert pass from a frozen state")
    return {"leg": "inert_passes", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — which ZM constexprs can ever be true?
# ---------------------------------------------------------------------------

def structural_walls_leg() -> Dict[str, Any]:
    """``zero_metal_axes`` over every admitted Dcyl shape this gate can build.

    The kernel carries all three ZM rows because the block is a byte-copy of
    ``kernels.fused_curl_constitutive_B``'s. Two of them can never be entered on a
    Dcyl grid — the r axis carries its own boundary pair rather than a declared PEC,
    and phi is the one-cell invariant axis, which ``Grid`` refuses a PEC on by name
    (grid.py:842-877). THAT IS MEASURED HERE, over a sweep, and it is what licenses
    the mutation table to arm ``ZM_Z`` alone: a mutation rewriting a branch whose
    constexpr is false on every scored case would report "uncaught" while measuring
    nothing.
    """
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_magnetic_pair as product)

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for r_extent in (1.1, 2.0, 2.6, 3.2):
        for z_extent in (1.1, 2.0, 3.2):
            for resolution in (6.0, 10.0):
                grid, fields, pml = _dcyl((r_extent, 0.0, z_extent), resolution, 0.5)
                verdict = product.cylindrical_real_fused_magnetic_pair_coverage(
                    fields, pml, ())
                residual = [reason for reason in verdict.reasons
                            if "cupy" not in reason]
                walls = tuple(bool(value) for value in zero_metal_axes(grid))
                rows.append({"cell": [r_extent, 0.0, z_extent],
                             "resolution": resolution, "shape": list(grid.shape),
                             "admitted": not residual, "walls": list(walls)})
                if residual:
                    continue
                if walls != (False, False, True):
                    findings.append(
                        f"an ADMITTED Dcyl grid {tuple(grid.shape)} reports walls "
                        f"{walls}, not (False, False, True): the mutation table's "
                        f"choice to arm ZM_Z alone is no longer licensed")
    admitted = [row for row in rows if row["admitted"]]
    if not admitted:
        findings.append("no configuration in the sweep was admitted; this leg "
                        "measured nothing")
    # A PEC on the INVARIANT phi axis is refused by Grid BY NAME, not by this
    # predicate. Measured rather than cited: the refusal is what makes ZM_Y dead.
    phi_pec: Dict[str, Any]
    try:
        _dcyl((2.0, 0.0, 2.0), 10.0, 0.5)
        from meep_gpu.grid import Grid  # noqa: PLC0415

        Grid(resolution=10.0, cell_size=(2.0, 0.0, 2.0), cylindrical=True, m=0,
             boundaries={"y": "metallic", "z": "metallic"}, courant=0.5)
    except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
        phi_pec = {"refused": True, "error": f"{type(exc).__name__}: {exc}"}
    else:
        phi_pec = {"refused": False, "error": None}
        findings.append(
            "Grid ACCEPTED a metallic declaration on the invariant phi axis; ZM_Y is "
            "no longer structurally dead and the kernel's dead branch is now live "
            "and unmeasured")
    return {"leg": "structural_walls", "device": False, "rows": rows,
            "admitted": len(admitted), "phi_pec_refused_by_grid": phi_pec,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — is the curl's ownership mask observable at all?
# ---------------------------------------------------------------------------

def mask_observability_leg() -> Dict[str, Any]:
    """Does the fixture put a NON-ZERO value where the ownership mask acts?

    THE MASK THIS PRODUCT CARRIES ZEROES TWO PLANES: ``Bx``'s curl at r = 0 and
    ``Bz``'s curl at z = 0 (``_mask_non_owned_cells``:1865, on the ``is_axis`` and
    ``is_metallic`` clauses). Both curls read ``Ep`` (= ``Ey``) as their only
    non-invariant operand there, so if ``Ep`` is exactly ``+0.0`` on those planes the
    mask is REDUNDANT — the value it writes is the value already present — and a
    mutation that drops it measures nothing while reporting "uncaught".

    THAT IS NOT HYPOTHETICAL. Measured on the first device run of this gate: with the
    ELECTRIC constitutive history ``f_w_E*`` left at zero, ``Ep`` never leaves ``+0.0``
    on either plane — the m = 0 axis rule sets ``Dp[0] = 0`` (stepping.py:589) and
    ``zero_metal_D`` clears ``Dy`` at z = 0, and ``update_E`` accumulates
    ``E += kps*(D*inv_eps) - kms*f_w_prev``, which stays at zero when both terms are.
    ``m5_ownership_mask_dropped`` came back UNCAUGHT for exactly that reason.

    So this leg measures BOTH arms: the seeded fixture (which :func:`build_driver`
    now uses) must leave ``Ep`` non-zero on both planes after one complete step, and
    the un-seeded control must leave it exactly zero. A leg that reported only the
    first would not show that the seeding is what does the work.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    SEEDED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
              "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    CONTROL = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, cell, resolution, courant, _steps in CASES:
        for arm, names in (("electric_history_seeded", SEEDED),
                           ("electric_history_zero", CONTROL)):
            _grid, fields, pml = _dcyl(cell, resolution, courant)
            _seed(fields, f"mask|{name}", names)
            for _ in range(2):
                driver_module.step_B(fields, pml)
                driver_module.zero_metal_B(fields)
                driver_module.update_H(fields, pml)
                driver_module.step_D(fields, pml)
                driver_module.zero_metal_D(fields)
                driver_module.update_E(fields, pml)
            ep = np.asarray(fields.Ey, dtype=np.float32)
            axis_row = np.ascontiguousarray(ep[0, :, :]).view(np.uint32)
            wall_plane = np.ascontiguousarray(ep[:, :, 0]).view(np.uint32)
            row = {"case": name, "arm": arm,
                   "nonzero_words_on_the_r0_row": int(np.count_nonzero(axis_row)),
                   "nonzero_words_on_the_z0_plane": int(np.count_nonzero(wall_plane))}
            rows.append(row)
            if arm == "electric_history_seeded" and not (
                    row["nonzero_words_on_the_r0_row"]
                    and row["nonzero_words_on_the_z0_plane"]):
                findings.append(
                    f"{name}: Ep is zero on a masked plane even with the electric "
                    f"history seeded, so the ownership mask is unobservable and "
                    f"m5_ownership_mask_dropped would be a FALSE null: {row}")
            if arm == "electric_history_zero" and (
                    row["nonzero_words_on_the_r0_row"]
                    or row["nonzero_words_on_the_z0_plane"]):
                findings.append(
                    f"{name}: the un-seeded CONTROL is non-zero on a masked plane, so "
                    f"this leg cannot show that seeding the electric history is what "
                    f"makes the mask observable: {row}")
    return {"leg": "mask_observability", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the design sweep, on NumPy
# ---------------------------------------------------------------------------

def _array_path_seam(fields, pml, driver_module) -> None:
    driver_module.step_B(fields, pml)
    driver_module.fill_symmetry_bc_B(fields)
    driver_module.zero_metal_B(fields)
    driver_module.fill_folded_far_ghosts_B(fields)
    driver_module.update_H(fields, pml)


def _emulated_seam(fields, pml, driver_module, *, carry_wall: bool = True,
                   order: str = "clear_then_constitutive") -> None:
    """The FUSED semantics, spelled with array ops so a laptop can execute them.

    THIS IS NOT THE KERNEL AND DOES NOT PRETEND TO BE. It cannot see a register
    reuse or a rounding order inside one launch — those are the device gate's. What
    it CAN decide is the ORDERING design: whether ``update_H`` consumes the
    wall-cleared ``B`` or the un-cleared one, which is the only thing the seam's
    third pass changes.
    """
    driver_module.step_B(fields, pml)
    if carry_wall and order == "clear_then_constitutive":
        driver_module.zero_metal_B(fields)
        driver_module.update_H(fields, pml)
    elif carry_wall:
        # The defect: update_H consumes a pre-clear B and the wall lands afterwards.
        driver_module.update_H(fields, pml)
        driver_module.zero_metal_B(fields)
    else:
        # The defect: the wall clear is not carried at all.
        driver_module.update_H(fields, pml)


def design_sweep_leg() -> Dict[str, Any]:
    """Each ORDERING choice flipped, on NumPy, against the array-path composition."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    knobs = (("faithful", {}, "null"),
             ("drop_the_wall_clear", {"carry_wall": False}, "caught"),
             ("constitutive_before_the_wall",
              {"order": "constitutive_then_clear"}, "caught"))
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, cell, resolution, courant, _steps in CASES:
        _grid, reference, reference_pml = _dcyl(cell, resolution, courant)
        _seed(reference, f"design|{name}", ALL_VOLUMES)
        before = _words(reference, ALL_VOLUMES)
        _array_path_seam(reference, reference_pml, driver_module)
        oracle = _words(reference, ALL_VOLUMES)
        moved_names = sorted(key for key in oracle
                             if not np.array_equal(oracle[key], before[key]))
        for knob, kwargs, expectation in knobs:
            _grid2, fields, pml = _dcyl(cell, resolution, courant)
            _seed(fields, f"design|{name}", ALL_VOLUMES)
            _emulated_seam(fields, pml, driver_module, **kwargs)
            candidate = _words(fields, ALL_VOLUMES)
            differing = sorted(key for key in oracle
                               if not np.array_equal(candidate[key], oracle[key]))
            rows.append({"case": name, "knob": knob, "expectation": expectation,
                         "identical": not differing, "differing_arrays": differing,
                         "arrays_moved_by_the_array_path": moved_names})
            if expectation == "caught" and not differing:
                findings.append(f"{name}/{knob}: expected to be caught and was not")
            if expectation == "null" and differing:
                findings.append(
                    f"{name}: the FAITHFUL emulation diverges from the array path on "
                    f"{differing} — the design, not the kernel, is wrong")
        if not moved_names:
            findings.append(f"{name}: VACUOUS — the array path moved nothing")
    return {"leg": "design_sweep", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real grids."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_magnetic_pair as product)

    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = [reason for reason in verdict.reasons if "cupy" not in reason]
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left, "needle": needle, "passed": ok})

    _grid, fields, pml = _dcyl((2.0, 0.0, 2.0), 10.0, 0.5)
    record("dcyl_m0_no_sources",
           product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, ()), True)
    # THE ADMITTING DIRECTION. An ELECTRIC source is injected in the D/E half and
    # must NOT disqualify this pair — and it is what all three corpus rows carry.
    record("dcyl_m0_electric_source",
           product.cylindrical_real_fused_magnetic_pair_coverage(
               fields, pml, (_Source("D"),)), True)
    record("dcyl_m0_magnetic_source",
           product.cylindrical_real_fused_magnetic_pair_coverage(
               fields, pml, (_Source("B"),)), False, "is magnetic")
    record("undeclared_sources",
           product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, None),
           False, "was not declared")
    record("no_pml",
           product.cylindrical_real_fused_magnetic_pair_coverage(fields, None, ()),
           False, "no active PML")

    # m != 0 forces complex storage; refused by NAME rather than by the absence of an
    # i*m/r term.
    m_grid = Grid(resolution=10.0, cell_size=(2.0, 0.0, 2.0), cylindrical=True, m=1,
                  boundaries={"z": "metallic"}, courant=0.5)
    m_fields = Fields(grid=m_grid, force_complex_fields=True)
    m_fields.enable_pml_storage()
    record("m_is_one",
           product.cylindrical_real_fused_magnetic_pair_coverage(
               m_fields, PML(grid=m_grid, thickness={"x": (0, 0.3), "z": 0.3}), ()),
           False, "carries m = 0")

    # A Cartesian grid belongs to the ordinary pair.
    # THE INVARIANT-AXIS PEC, avoided by name: a 2-D cell is translationally
    # invariant along z, and a metallic declaration there is not a wall but a
    # polarization filter — Grid refuses it (grid.py:842-877). This leg is about the
    # product's own refusals; it must not die inside the Grid constructor first.
    flat = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    flat_fields = Fields(grid=flat, force_complex_fields=False)
    flat_fields.enable_pml_storage()
    record("cartesian_grid",
           product.cylindrical_real_fused_magnetic_pair_coverage(
               flat_fields, PML(grid=flat, thickness=0.2), ()),
           False, "not cylindrical")

    findings = [row["case"] for row in rows if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 6 — what the corpus says this is worth
# ---------------------------------------------------------------------------

CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                      "predicate_coverage_2026-08-16_wired_convention")
METAL_CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                            "metal_coverage_tranche6_2026-08-19")


def _census_rows(root: str) -> List[dict]:
    def load(name: str) -> List[dict]:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            return []
        return [json.loads(line) for line in open(path, encoding="utf-8")
                if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load("tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def corpus_admission_leg() -> Dict[str, Any]:
    """The funnel, on BOTH backends' censuses, and whether the two agree row for row.

    THE CROSS-CHECK IS THE POINT. A disagreement would mean one of the two backends'
    predicate ladders is wrong, which matters more than the product; reported either
    way. This cell is the one with NO attrition — every row that drives it clears the
    magnetic seam — so a funnel that narrows anywhere is itself the finding.
    """
    def funnel(root: str, curl_key: str, constitutive_key: str) -> Dict[str, Any]:
        rows = _census_rows(root)

        def covered(row: dict, key: str) -> bool:
            entry = row["predicates"].get(key, {})
            return bool(entry.get("covered_modulo_backend", entry.get("covered")))

        def label(row: dict) -> str:
            return f"{row['leg']}:{row['row']}"

        curl = [r for r in rows if covered(r, curl_key)]
        both = [r for r in curl if covered(r, constitutive_key)]
        admitted = [r for r in both
                    if not any(str(kind) == "B" for kind in
                               (r["configuration"].get("source_field_types") or []))]
        return {"rows": len(rows), "curl": len(curl), "both": len(both),
                "admitted": len(admitted),
                "admitted_rows": sorted(label(r) for r in admitted)}

    triton = funnel(CENSUS, "cylindrical_curl@step_B",
                    "cylindrical_constitutive@update_H")
    metal = funnel(METAL_CENSUS, "cylindrical_real_curl@step_B",
                   "cylindrical_real_constitutive@update_H")
    findings: List[str] = []
    # AN ABSENT CENSUS IS NOT AN EMPTY ONE. A staged tree that did not carry the
    # record would funnel 0 -> 0 -> 0 and this leg would report "worth zero" — a
    # measurement about the staging dressed as one about the corpus.
    for label, root, block in (("Triton", CENSUS, triton),
                               ("Metal", METAL_CENSUS, metal)):
        if block["rows"] != 186:
            findings.append(
                f"the {label} census at {root} yielded {block['rows']} measured rows, "
                f"not the 186 this derivation prices against: the record is absent or "
                f"truncated on this host and NOTHING below is a measurement about the "
                f"corpus")
    if findings:
        return {"leg": "corpus_admission", "device": False, "triton": triton,
                "metal": metal, "seam_instances_gained": None,
                "findings": findings, "passed": False}
    if triton["admitted_rows"] != metal["admitted_rows"]:
        findings.append(
            f"the two backends' predicates do not agree on this cell's rows: Triton "
            f"{triton['admitted_rows']} vs Metal {metal['admitted_rows']} — one of "
            f"the two predicate stacks is wrong and that matters more than this "
            f"product")
    if not triton["admitted"]:
        findings.append("the Triton predicate admits NO corpus row: this product is "
                        "worth zero seam-instances and should not have been built")
    if not (triton["curl"] == triton["both"] == triton["admitted"]):
        findings.append(
            f"the funnel narrows: {triton['curl']} -> {triton['both']} -> "
            f"{triton['admitted']}. This cell was priced as having NO attrition; "
            f"the docstring's claim needs correcting, not the product")
    return {"leg": "corpus_admission", "device": False, "triton": triton,
            "metal": metal, "seam_instances_gained": triton["admitted"],
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------

def build_driver(cp, cell, resolution, courant, seed: int, electric: bool):
    """One Dcyl m = 0 PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=resolution, dimensions=2, cylindrical=True, m=0,
        force_complex_fields=False, courant=courant, boundaries={"z": "metallic"},
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml({"x": {"high": 4}, "z": 4})
    if electric:
        # An ELECTRIC source is admitted: the driver injects it in the D/E seam, not
        # this one. All three corpus rows that reach this cell carry exactly one.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.25 * cell[0], 0.0, 0.0), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    # BOTH constitutive histories are seeded, and the ELECTRIC one is what makes the
    # curl's ownership mask observable at all. MEASURED 2026-08-20, on the first
    # device run of this gate: with ``f_w_E*`` left at zero, ``Ey`` (= Ep) is EXACTLY
    # +0.0 on the r = 0 row and on the z = 0 wall plane forever — the m = 0 axis rule
    # sets ``Dp[0] = 0`` and ``zero_metal_D`` clears ``Dy`` at z = 0, and with a zero
    # history the accumulation never leaves zero. Both masked curls are then already
    # an exact +0.0 before the mask runs, so ``m5_ownership_mask_dropped`` came back
    # UNCAUGHT while measuring nothing. That was a defect in this fixture, not a
    # property of the kernel. :func:`mask_observability_leg` now measures the
    # difference rather than leaving it to be rediscovered.
    for name in ("f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(np.ascontiguousarray(
                rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)))
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
    """The two separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: PLC0415

    curl = cyl.plan_cylindrical_curl(driver.fields, driver.pml, "step_B")
    constitutive = cyl.plan_cylindrical_constitutive(driver.fields, driver.pml, "H")
    missing = [name for name, plan in (("cylindrical curl", curl),
                                       ("cylindrical constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so this "
            f"leg could not compare the fused launch against the products it replaces")
    # `zero_metal_B` and both fills stay on the ARRAY PATH here: no Triton product
    # owns the wall clear, and the two fills are inert. All three are counted, so a
    # difference in which of them ran is a counter event rather than an invisible
    # correction.
    return Route({"step_B": curl, "update_H": constitutive})


def fused_route(plan):
    """All FIVE passes absorbed.

    Three are CARRIED by the launch; the other two are claimed inert. Absorbing them
    rather than leaving them on the array path is what turns that claim into a byte
    measurement: a fill that ever did work on an admitted Dcyl grid would show up
    here as a divergence against both oracles.
    """
    return Route({name: (plan if name == "step_B" else _Absorbed(name, plan))
                  for name in SEAM_PASSES})


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, resolution, courant, steps: int, product,
            mutant: Any = None, install_fused: bool = True,
            freeze_magnetic: bool = False, electric: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, cell, resolution, courant, SEED, electric)
    separate = build_driver(cp, cell, resolution, courant, SEED, electric)
    fused = build_driver(cp, cell, resolution, courant, SEED, electric)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.cyl_real_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "cell": list(cell),
        "resolution": resolution, "courant": repr(courant),
        "electric_source": bool(electric),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_cylindrical_real_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1, kernel=kernel)
        if plan is None:
            verdict = product.cylindrical_real_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
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
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} ({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
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
        for target in (reference, separate, fused):
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True) -> Tuple[bool, List[str]]:
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
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(product) -> str:
    return textwrap.dedent(
        inspect.getsource(product.cyl_real_fused_curl_constitutive_B.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = "import triton\nimport triton.language as tl\n\n"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_cyl_real_pair.py", delete=False, encoding="utf-8")
    handle.write(header + source.replace("cyl_real_fused_curl_constitutive_B",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cyl_real_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite)."""

    def m1_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(B_stuff), undone.

        ``ZM_Z`` is TRUE on every admitted Dcyl grid (:func:`structural_walls_leg`
        measures that), so this rewrite is live on every case. ``ZM_X`` and ``ZM_Y``
        are not touched: their constexprs are structurally false, and a rewrite of a
        dead branch reports "uncaught" while measuring nothing.

        NO ``if`` IN THE REPLACEMENT. ``_rewrite_block`` indents every replacement
        line to the indent of the block's FIRST line, so a compound statement in
        position 0 produces an empty suite and the mutated module fails to IMPORT —
        which the harness records as an unarmed mutation, not as a caught defect.
        The assignment is a pure identity with no arithmetic, so a signed zero
        survives it."""
        return _rewrite_block(
            source, ["if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"], ["v2 = v2"])

    def m2_zero_metal_after_the_constitutive(source: str) -> Tuple[str, int]:
        """The wall clear moved AFTER ``update_H`` consumes B — the seam's ORDER.

        TWO EDITS, and they have to move together. The clear is taken off the
        register (so the constitutive half reads the UN-cleared value) and put onto
        the store instead (so ``Bz`` still ends up exactly as the array path leaves
        it). A comparison that only looked at ``B`` would therefore see nothing;
        ``Hz`` and ``f_w_Hz`` carry the un-cleared plane, which is what makes this a
        measurement about the seam's order rather than about the clear's existence —
        that one is m1's.

        NO ``if`` IN THE FIRST REPLACEMENT: ``_rewrite_block`` indents every
        replacement line to the block's first line, so a compound statement in
        position 0 produces an empty suite and the mutant fails to IMPORT."""
        hits = 0
        source, moved = _rewrite_block(
            source, ["if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"], ["v2 = v2"])
        hits += moved
        source, moved = _rewrite_block(
            source, ["tl.store(f2 + idx, v2, mask=live)"],
            ["tl.store(f2 + idx, tl.where(at_z, 0.0, v2), mask=live)"])
        hits += moved
        return source, hits if hits == 2 else 0

    def m3_axis_rule_dropped(source: str) -> Tuple[str, int]:
        """``Br[0] = 0`` — the m = 0 on-axis rule (stepping._cylindrical_axis_zero_B:648).

        Without it ``Bx`` keeps its stepped value on r = 0, which ``update_H`` then
        turns into a wrong ``Hx`` row."""
        return _rewrite_block(
            source, ["v0 = tl.where(at_r, 0.0, v0)"], ["v0 = v0 + 0.0"])

    def m4_bz_curl_is_the_generic_one(source: str) -> Tuple[str, int]:
        """``Bz``'s whole curl replaced by the generic four-operand stencil.

        ``step_B`` :343-346 does NOT compute Bz's curl from the raw operands: it
        extends Ep by one zero wall row, prefixes it at ``ir0 = 0.0`` and takes ONE
        forward difference of the prefix. The generic form below is the substitution
        undone — the sibling of the ``bz_flat_grouping`` mutation the cylindrical
        curl's own gate measured moving 211 ``Bz`` floats on step 0."""
        return _rewrite_block(
            source,
            ["curl2 = dtdx * (tl.load(pfx + idx + nyz, mask=live, other=0.0)",
             "- tl.load(pfx + idx, mask=live, other=0.0))"],
            ["curl2 = dtdx * ((b_r - b) + (a - a_p))"])

    def m5_ownership_mask_dropped(source: str) -> Tuple[str, int]:
        """The r = 0 / z = 0 ownership mask (stepping._mask_non_owned_cells:1865).

        OBSERVABLE ONLY BECAUSE THE FIXTURE SEEDS THE ELECTRIC CONSTITUTIVE HISTORY,
        and that is measured rather than assumed: :func:`mask_observability_leg`
        reports ``Ep`` non-zero on both masked planes with ``f_w_E*`` seeded and
        EXACTLY zero without it. With the un-seeded fixture this mutation came back
        UNCAUGHT on 2026-08-20 while measuring nothing, which is a defect in the
        harness and not a property of the kernel."""
        return _rewrite_block(
            source,
            ["curl0 = tl.where(at_r, 0.0, curl0)",
             "curl2 = tl.where(at_z, 0.0, curl2)"],
            ["curl0 = curl0", "curl2 = curl2"])

    def m6_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source, ["a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"],
            ["a0 = a0 + (kp_0 * src0 - km_0 * prev0)"])

    def m7_history_read_after_write(source: str) -> Tuple[str, int]:
        """f_w read AFTER it is written: wrong only where kms != 0, i.e. in the PML."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)", "src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0", "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m8_phi_term_sign_flipped(source: str) -> Tuple[str, int]:
        """The invariant-axis term COMPUTED, and computed wrong.

        phi has n = 1, so its rolled operand equals the original and ``(c_p - c)`` is
        exactly ``+0.0``; ``(c_p + c)`` is ``2c``, which is loud.

        WHY NOT THE ELISION ITSELF. Dropping ``(c_p - c)`` outright — the signed-zero
        trap ``cylindrical_triton``'s docstring names — differs from the shipped body
        ONLY where ``(b - b_z)`` is ``-0.0``, and whether that pattern arises is a
        property of the FIXTURE, not of the kernel. A mutation whose visibility this
        gate cannot predict would be scored on a guess. The elision is refused by the
        TRANSCRIPTION leg instead, which requires the four-operand line verbatim and
        is deterministic."""
        return _rewrite_block(
            source, ["curl0 = dtdx * ((c_p - c) + (b - b_z))"],
            ["curl0 = dtdx * ((c_p + c) + (b - b_z))"])

    def m9_coefficient_index_moved(source: str) -> Tuple[str, int]:
        """The constitutive coefficient read on the WRONG axis for component 0.

        ``H_CONSTITUTIVE_TERMS`` indexes each component on its OWN axis; reading
        component 0's on z is MEEP's ``dsigw`` for the wrong direction."""
        return _rewrite_block(
            source, ["kp_0 = tl.load(kp0 + i, mask=live, other=0.0)"],
            ["kp_0 = tl.load(kp0 + k, mask=live, other=0.0)"])

    def m10_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_zero_metal_dropped", "the wall clear's slot", "caught",
         m1_zero_metal_dropped),
        ("m2_zero_metal_after_the_constitutive", "the seam's ORDER", "caught",
         m2_zero_metal_after_the_constitutive),
        ("m3_axis_rule_dropped", "the m = 0 on-axis rule", "caught",
         m3_axis_rule_dropped),
        ("m4_bz_curl_is_the_generic_one", "Bz's prefix substitution", "caught",
         m4_bz_curl_is_the_generic_one),
        ("m5_ownership_mask_dropped", "the cell-0 ownership mask", "caught",
         m5_ownership_mask_dropped),
        ("m6_constitutive_association", "float32 association", "caught",
         m6_constitutive_association),
        ("m7_history_read_after_write", "the f_w ordering", "caught",
         m7_history_read_after_write),
        ("m8_phi_term_sign_flipped", "the invariant-axis term", "caught",
         m8_phi_term_sign_flipped),
        ("m9_coefficient_index_moved", "dsigw's own axis", "caught",
         m9_coefficient_index_moved),
        ("m10_commuted_multiply", "commuted multiply", "null", m10_commuted_multiply),
    )


#: Mutation id -> the CASES index it is scored on. Every case here is the same
#: Dcyl m = 0 declaration with the same compiled branches, so no mutation can land on
#: a grid that fails to enter the lines it rewrites — the dead-branch hazard this
#: family carries is the ZM_X / ZM_Y rows, and no mutation touches them.
#: :func:`mutation_case_for` asserts that rather than leaving it to a reader.
MUTATION_CASE: Dict[str, int] = {}

#: CASES[0] is SQUARE (nr == nz), and that is load-bearing for one mutation rather
#: than a default. ``m9_coefficient_index_moved`` reads ``kp0`` — a length-nr column —
#: at the z index; on ``short_wide`` (nr = 11, nz = 32) that is an OUT-OF-BOUNDS read,
#: which is a memory fault reported as an inert defect rather than a wrong answer.
#: On a square case the read is in bounds and the leg measures the wrong VALUE, which
#: is the defect being armed.
DEFAULT_MUTATION_CASE = 0

#: Constexpr names no mutation may rewrite, because they are false on every admitted
#: configuration (measured by :func:`structural_walls_leg`).
DEAD_CONSTEXPRS: Tuple[str, ...] = ("ZM_X", "ZM_Y")


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on, with the dead-branch check."""
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    return index, CASES[index]


def _assert_no_dead_branch_rewrites(source: str) -> List[str]:
    """No rewrite may change a line whose enclosing constexpr is structurally false."""
    findings: List[str] = []
    for name, _why, _expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        if hits == 0:
            findings.append(f"{name}: the rewrite matched nothing")
            continue
        original = source.splitlines()
        changed = mutated.splitlines()
        for index, line in enumerate(original):
            if index < len(changed) and changed[index] == line:
                continue
            # Walk back to the nearest enclosing `if`, by indentation.
            indent = len(line) - len(line.lstrip())
            for back in range(index - 1, -1, -1):
                candidate = original[back]
                if not candidate.strip():
                    continue
                if len(candidate) - len(candidate.lstrip()) < indent:
                    if any(dead in candidate for dead in DEAD_CONSTEXPRS):
                        findings.append(
                            f"{name} rewrites a line guarded by {candidate.strip()!r}, "
                            f"whose constexpr is false on every admitted "
                            f"configuration: the mutation would measure nothing")
                    break
            break
    return findings


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(product)
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "device": True}
        if hits == 0 or mutated == source:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_cyl_real_fused_B"
        mutant = compile_mutant(mutated, kernel_name)
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3],
                      MUTATION_STEPS, product, mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, asked of a real CuPy driver."""
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    class _Magnetic:
        field_type = FIELD_TYPE_B

    rows: List[Dict[str, Any]] = []
    for name, needle, sources in (
        ("magnetic_source", "is magnetic", (_Magnetic(),)),
        ("undeclared_sources", "was not declared", None),
    ):
        driver = build_driver(cp, CASES[0][1], CASES[0][2], CASES[0][3], SEED,
                              electric=False)
        try:
            verdict = product.cylindrical_real_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            plan = product.plan_cylindrical_real_fused_magnetic_pair(
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

    # A CARTESIAN driver: the ordinary fused pair's territory, refused by name.
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(cell_size=(2.0, 2.0, 0.0), resolution=10.0, dimensions=2,
                        force_complex_fields=False, courant=0.5,
                        boundaries={"x": "metallic", "y": "metallic",
                                    "z": "periodic"},
                        prefer_gpu=True, gpu_id=0)
    try:
        driver.setup_pml({"x": 4, "y": 4})
        verdict = product.cylindrical_real_fused_magnetic_pair_coverage(
            driver.fields, driver.pml, ())
        plan = product.plan_cylindrical_real_fused_magnetic_pair(
            driver.fields, driver.pml, ())
        rows.append({
            "case": "cartesian_grid", "device": True,
            "covered": bool(verdict.covered), "reasons": list(verdict.reasons),
            "plan_is_none": plan is None,
            "passed": (not verdict.covered) and plan is None
                      and any("not cylindrical" in reason
                              for reason in verdict.reasons),
        })
    finally:
        driver.close()
        cp.get_default_memory_pool().free_all_blocks()
    log(f"  refusal cartesian_grid: covered={rows[-1]['covered']} "
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
        HERE, "results", "triton_cylindrical_real_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep'.")
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {
        "gate": "triton_cylindrical_real_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair",
        "kernel": "cyl_real_fused_curl_constitutive_B",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "cylindrical_triton.DEFAULT_BLOCK"},
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, inert_passes_leg, structural_walls_leg,
                mask_observability_leg, design_sweep_leg, predicate_leg,
                corpus_admission_leg):
        started = time.time()
        row = leg()
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} findings={row.get('findings')} "
            f"({row['seconds']} s)")
        save(payload, args.out)

    # The dead-branch audit runs on the SOURCE TEXT, so it needs no device.
    dead = _assert_no_dead_branch_rewrites(
        _shipped_text("cyl_real_fused_curl_constitutive_B"))
    payload["no_device_legs"].append({
        "leg": "dead_branch_audit", "device": False, "findings": dead,
        "dead_constexprs": list(DEAD_CONSTEXPRS), "passed": not dead})
    log(f"  dead_branch_audit: passed={not dead} findings={dead}")
    save(payload, args.out)

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
        save(payload, args.out)
        log(f"\nno-device verdict: {payload['passed']}  ->  {args.out}")
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

    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_magnetic_pair as product)

    backends.guard_kernel_compilation(cp)
    # THE MACHINE AND THE DEVICE, so this artifact can be REBOUND from what it
    # measured. This family already holds a ledger entry, and its host line is not
    # rewritten on a rebind — but the entry's pins move in this round anyway (its
    # pinned test module and this probe are both edited), so the next artifact is
    # made seed-shaped here rather than left as the one that is not.
    import triton_device_identity  # noqa: PLC0415

    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "byte_cases": {case[0]: case[4] for case in CASES},
        "steps_per_mutation": MUTATION_STEPS,
        "total_steps": (sum(case[4] for case in CASES)
                        + MUTATION_STEPS * (len(mutation_table()) + 2)),
    }
    save(payload, args.out)
    log("\n=== device legs ===")
    for name, cell, resolution, courant, steps in CASES:
        row = run_leg(cp, name, cell, resolution, courant, steps, product)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== armed harness mutations ===")
    row = run_leg(cp, "armed:no_substitution", CASES[0][1], CASES[0][2], CASES[0][3],
                  3, product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", CASES[0][1], CASES[0][2],
                  CASES[0][3], 3, product, freeze_magnetic=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.cyl_real_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits"):
            return False
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"],
        "reasons": ([] if payload["passed"] else
                    [f"device legs ok: {device_ok}",
                     "mutations whose measured outcome did not match their declared "
                     "expectation, or that were never armed: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row))),
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}"]),
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

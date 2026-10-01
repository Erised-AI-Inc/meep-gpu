"""Byte gate for the Dcyl m = 0 fused ELECTRIC pair: ``step_D`` into ``update_E``.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.cylindrical_real_fused_electric_pair.cylindrical_real_fused_electric_pair_coverage`
admits, ONE launch of ``cyl_real_fused_curl_constitutive_D`` — bracketed by the
SHIPPED deposit repair wherever the seam carries an electric deposit — leaves the
engine in a state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_D`` / the electric injection /
    ``zero_metal_D`` / ``update_E``, with both symmetry fills inert), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the cylindrical curl
    (``cylindrical_triton.plan_cylindrical_curl`` on ``step_D``) and the cylindrical
    constitutive (``plan_cylindrical_constitutive`` on side ``E``), with
    ``zero_metal_D`` on the array path in both routes because no Triton product owns
    it,

over every allocated volume: the primaries, the split-field PML auxiliaries and the
constitutive ``f_w`` history. Comparison is on the uint32 view. ``allclose`` appears
nowhere.

WHAT IS NEW HERE AGAINST THE MAGNETIC TWIN
==========================================

:mod:`meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair` released
2026-08-20 over the SAME two certified halves on the other seam, and its kernel
already CARRIES both ``BACKWARD`` arms. What this gate is measuring for the first
time is everything that is compile-time absent at ``BACKWARD = 0``:

* ``Dz``'s curl as the BACKWARD difference of the radial prefix (stepping.py:454-455)
  — the four-operand grouping, not the two-operand forward one;
* the D-family ownership mask — ``Dy`` at r = 0 AND at z = 0, ``Dz`` at r = 0;
* the m = 0 axis rules ``_cylindrical_axis_zero_D`` (stepping.py:589):
  ``Dz[0] += (4*Courant) * Hp[0]``, a POST-add on the updated field, then
  ``Dp[0] = 0``;
* ``zero_metal_D``'s SIX rows (each axis clears the two TANGENTIAL components)
  against the magnetic twin's three;
* the constitutive half's inverse-permittivity multiply — the ``SCALE = 1`` arm.

...and **THE DEPOSIT REPAIR**, which the magnetic twin does not carry at all. All
three corpus rows of this cell declare an ELECTRIC source, injected between the two
halves, so the CARRY family below is the product rather than an extra.

WHAT THE GATE REFUSES TO INFER
==============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array must
  MOVE during the leg; a leg whose electric trio is frozen in all three routes is
  armed and must be caught by the moved-state census.
* **A bracket that changes nothing is not load-bearing.** Every CARRY case is run
  TWICE — once with the shipped :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
  :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair and once WITHOUT it — and
  the unbracketed run MUST DIVERGE.
* **Two omitted driver passes are a claim until somebody runs them.**
  :func:`inert_passes_leg` executes ``fill_symmetry_bc_D`` and
  ``fill_folded_far_ghosts_D`` on a randomised state on an admitted grid and requires
  that no word moves, with ``zero_metal_D`` as the control that must move one.
* **A mutation that rewrites a dead branch measures nothing.**
  :func:`structural_walls_leg` measures which ``ZM`` constexprs can ever be true on
  an admitted Dcyl grid, and :func:`_assert_no_dead_branch_rewrites` refuses a
  rewrite that touches the others.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_cylindrical_real_fused_electric_pair.py \\
        --no-device --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_cylindrical_real_fused_electric_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>
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

SEED = 20260831

#: Temporary mutant modules, deleted at the end of the run.
_TEMPORARY: List[str] = []

#: (name, cell (r, -, z), resolution, courant, steps). Every case is Dcyl at m = 0
#: with a METALLIC z declaration — which is not a choice: ``_boundary_kinds`` must
#: resolve to ``('axis', 'periodic', 'metallic')`` or the curl predicate refuses the
#: grid outright, so every admitted configuration is walled on z and ``ZM_Z`` fires
#: on every one. :func:`structural_walls_leg` measures that rather than asserting it.
#:
#: THE COURANT NUMBERS ARE SWEPT and three of the four are not powers of two: the PML
#: coefficients are the family of non-representable multiplicands the curl's
#: association mutations need in order to be visible at all.
CASES: Tuple[Tuple[str, Tuple[float, float, float], float, float, int], ...] = (
    ("square_res10", (2.0, 0.0, 2.0), 10.0, 0.5, 10),
    ("tall_thin", (3.2, 0.0, 1.1), 10.0, 0.3141592653589793, 10),
    ("short_wide", (1.1, 0.0, 3.2), 10.0, 0.37, 10),
    ("coarse", (2.6, 0.0, 2.6), 6.0, 0.4472135954999579, 10),
)

#: The CARRY family: the same grids run with a real ELECTRIC deposit IN THIS SEAM.
#: Named so the two families cannot drift apart. THIS IS THE PRODUCT, not an extra —
#: all three corpus rows of this cell declare an electric source, so a gate that only
#: ran the quiet family would certify a kernel the corpus never reaches.
CARRY_CASES: Tuple[str, ...] = ("square_res10", "short_wide")

#: Steps per armed mutation, and per carry / null-control leg.
MUTATION_STEPS = 3
CARRY_STEPS = 6

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
#: FIVE names, of which the product CARRIES three (``REPLACES``) and the other two
#: are inert on every admitted configuration. The fused route absorbs all five, so a
#: fill that ever did work on such a grid becomes a byte divergence here.
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
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
#: it.
MATERIAL = ("eps", "inv_eps")

#: Private ``Fields`` scratch that is NOT physical state, and is out of the
#: comparison for that reason. ``_fmp_scratch`` is the buffer
#: ``Fields.displacement_minus_polarization`` allocates on first use; on a CARRY leg
#: the fused route allocates it through ``deposit_repair.apply`` and the array path
#: through ``stepping.update_E``, so its presence differs by route and by leg while
#: carrying no state either route reads across a step.
#:
#: THE RULE IS THE LEADING UNDERSCORE, not this list: a private attribute is not
#: physical state, and every PUBLIC volume stays in the comparison.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)


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
    """Every file this run binds itself to, KEYED BY ITS REPO-RELATIVE PATH."""
    names = (
        "meep_gpu/triton_kernels/cylindrical_real_fused_electric_pair.py",
        "meep_gpu/triton_kernels/cylindrical_real_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/cylindrical_triton.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_cylindrical_real_fused_electric_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_SOURCE_FILE = {
    "cyl_pml_curl_step": "cylindrical_triton.py",
    "constitutive_step": "kernels.py",
    "fused_curl_constitutive_D": "kernels.py",
    "cyl_real_fused_curl_constitutive_B": "cylindrical_real_fused_magnetic_pair.py",
    "cyl_real_fused_curl_constitutive_D": "cylindrical_real_fused_electric_pair.py",
}


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, so the transcription check bites at the merge bar rather than only on a
    device run. The text is exact — never ``ast.unparse``d — because what is being
    checked is the PARENTHESISATION.
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
    """Executable lines, comments and blanks removed, indentation normalised."""
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

#: The m = 0 substitutions the BACKWARD arm makes and this body must keep. Every one
#: of these is compile-time ABSENT in the magnetic twin, which is why this gate is
#: the first to execute any of them.
CYLINDRICAL_LINES = (
    "p_here = tl.load(pfx + idx, mask=live, other=0.0)",
    "p_down = tl.load(pfx + o_r, mask=vr, other=0.0)",
    "curl2 = dtdx * ((p_down - p_here) + (a - a_p))",
    "at_r, at_z = i == 0, k == 0",
    "curl0 = tl.where(at_z, 0.0, curl0)",
    "curl1 = tl.where(at_r, 0.0, curl1)",
    "curl1 = tl.where(at_z, 0.0, curl1)",
    "curl2 = tl.where(at_r, 0.0, curl2)",
    "v1 = tl.where(at_r, 0.0, v1)",
    "v2 = tl.where(at_r, v2 + four_dtdx * tl.load(hp + idx, mask=live, other=0.0),",
    "v2)",
)

#: The wall clear, byte-copied from ``kernels.fused_curl_constitutive_D``'s ZM block
#: — the SIX rows ``stepping._zero_metal`` writes for the D family, against the
#: magnetic twin's three.
ZERO_METAL_LINES = (
    "v1 = tl.where(at_x, 0.0, v1)",
    "v2 = tl.where(at_x, 0.0, v2)",
    "v0 = tl.where(at_y, 0.0, v0)",
    "v2 = tl.where(at_y, 0.0, v2)",
    "v0 = tl.where(at_z, 0.0, v0)",
    "v1 = tl.where(at_z, 0.0, v1)",
)

#: The constitutive half, byte-copied from the same function. ``src`` is the SCALE=1
#: arm's product, which the magnetic twin does not have at all.
CONSTITUTIVE_LINES = (
    "prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
    "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
    "src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)",
    "src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)",
    "a0 = a0 + kp_0 * src0",
    "a0 = a0 - km_0 * prev0",
    "a1 = a1 + kp_1 * src1",
    "a1 = a1 - km_1 * prev1",
    "a2 = a2 + kp_2 * src2",
    "a2 = a2 - km_2 * prev2",
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    fused = _statements(_shipped_text("cyl_real_fused_curl_constitutive_D"))
    curl = _statements(_shipped_text("cyl_pml_curl_step"))
    twin = _statements(_shipped_text("cyl_real_fused_curl_constitutive_B"))
    plain_pair = _statements(_shipped_text("fused_curl_constitutive_D"))
    constitutive = _statements(_shipped_text("constitutive_step"))

    findings: List[str] = []
    for line in CURL_LINES + CYLINDRICAL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in curl:
            findings.append(
                f"cylindrical_triton.cyl_pml_curl_step no longer contains {line!r}; "
                f"the fused body's transcription source has moved")
        # ...and the RELEASED magnetic twin carries the same line, because the two
        # bodies differ only in which BACKWARD arm compiles.
        if line not in twin:
            findings.append(
                f"cylindrical_real_fused_magnetic_pair's kernel no longer contains "
                f"{line!r}; the two twins have drifted and only one of them is gated")
    for line in ZERO_METAL_LINES + CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
        if line not in plain_pair:
            findings.append(
                f"kernels.fused_curl_constitutive_D no longer contains {line!r}; the "
                f"wall clear / constitutive transcription source has moved")
    # And the constitutive grouping traces one step further back.
    for line in ("a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"):
        if line not in constitutive:
            findings.append(
                f"kernels.constitutive_step no longer contains {line!r}")

    # THE WALL CLEAR IS THE D FAMILY'S SIX ROWS, not the B family's three. A body
    # carrying three would be clearing the NORMAL component on each axis instead of
    # the two tangential ones.
    clears = [line for line in fused
              if line.startswith("v") and "tl.where(at_" in line
              and "0.0, v" in line and "four_dtdx" not in line]
    if len(clears) < len(ZERO_METAL_LINES):
        findings.append(
            f"the fused body carries {len(clears)} wall-clear rows, fewer than the "
            f"{len(ZERO_METAL_LINES)} zero_metal_D writes for the D family")

    # NO BOUNDARY CONSTEXPRS. The Dcyl declaration is compiled in; a BC branch here
    # would mean this body serves a declaration the predicate refuses.
    for needle in ("BCX", "BCY", "BCZ", "PERIODIC", "METALLIC", "MIRROR"):
        if any(needle in line for line in fused):
            findings.append(
                f"the fused body mentions {needle!r}: this kernel serves exactly the "
                f"Dcyl triple ('axis', 'periodic', 'metallic') and compiles it in")

    # THE ORDER: the m = 0 axis rules, then the wall clear, then the store, then the
    # constitutive read. The driver runs step_D (which ends with the axis rules),
    # then zero_metal_D, then update_E, and BOTH consumers must see the cleared
    # register.
    try:
        axis_rule = fused.index("v1 = tl.where(at_r, 0.0, v1)")
        clear = fused.index("v0 = tl.where(at_z, 0.0, v0)")
        store = fused.index("tl.store(f0 + idx, v0, mask=live)")
        read = fused.index("src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)")
    except ValueError as exc:  # pragma: no cover - the line checks above fire first
        findings.append(f"could not locate the ordering landmarks: {exc}")
    else:
        if not axis_rule < clear < store < read:
            findings.append(
                f"the order is axis rule {axis_rule}, wall clear {clear}, store "
                f"{store}, constitutive read {read}; the driver runs step_D (which "
                f"ends with the axis rules), then zero_metal_D, then update_E")
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
    """``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D``, EXECUTED and measured.

    The product's ``REPLACES`` names three driver passes and the seam has five. The
    other two are omitted on the claim that they cannot execute on a grid this
    predicate admits. That claim is a reading of another module's guard until
    somebody runs them, so this leg runs them: a randomised state on an admitted Dcyl
    grid, both passes applied, every allocated volume compared as uint32.

    NON-VACUITY: a leg where nothing was seeded would report "no word moved" about an
    array of zeros. Every compared volume is required to be non-constant before the
    passes run, and the CONTROL — ``zero_metal_D``, which is NOT inert on these grids
    — is required to move something.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_electric_pair as product)
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, cell, resolution, courant, _steps in CASES:
        grid, fields, pml = _dcyl(cell, resolution, courant)
        verdict = product.cylindrical_real_fused_electric_pair_coverage(
            fields, pml, ())
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
        moved = sorted(key for key in before
                       if not np.array_equal(before[key], after[key]))
        # THE CONTROL: the pass this product DOES carry must move something on the
        # same grid, or the instrument is measuring a frozen state.
        stepping.zero_metal_D(fields)
        control = _words(fields, ALL_VOLUMES)
        control_moved = sorted(key for key in after
                               if not np.array_equal(after[key], control[key]))
        rows.append({"case": name, "shape": list(grid.shape),
                     "inert_passes": sorted(product.INERT_PASSES),
                     "moved_by_the_inert_passes": moved,
                     "moved_by_zero_metal_D": control_moved,
                     "walls": list(zero_metal_axes(grid))})
        if moved:
            findings.append(
                f"{name}: {sorted(product.INERT_PASSES)} moved {moved} — they are NOT "
                f"inert on this grid and REPLACES may not omit them")
        if not control_moved:
            findings.append(
                f"{name}: zero_metal_D moved nothing either, so this leg cannot tell "
                f"an inert pass from a frozen state")
    return {"leg": "inert_passes", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — which ZM constexprs can ever be true?
# ---------------------------------------------------------------------------

def structural_walls_leg() -> Dict[str, Any]:
    """``zero_metal_axes`` over every admitted Dcyl shape this gate can build.

    The kernel carries all three ZM blocks because the block is a byte-copy of
    ``kernels.fused_curl_constitutive_D``'s. Two of them can never be entered on a
    Dcyl grid — the r axis carries its own boundary pair rather than a declared PEC,
    and phi is the one-cell invariant axis, which ``Grid`` refuses a PEC on by name
    (grid.py:842-877). THAT IS MEASURED HERE, over a sweep, and it is what licenses
    the mutation table to arm ``ZM_Z`` alone.
    """
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, cell, resolution, courant, _steps in CASES:
        grid, _fields, _pml = _dcyl(cell, resolution, courant)
        walls = tuple(bool(value) for value in zero_metal_axes(grid))
        rows.append({"case": name, "shape": list(grid.shape),
                     "zero_metal_axes": list(walls)})
        if walls[0] or walls[1]:
            findings.append(
                f"{name}: zero_metal_axes reports {walls}; ZM_X/ZM_Y were priced as "
                f"structurally false on every Dcyl grid and the mutation table arms "
                f"ZM_Z alone on that basis")
        if not walls[2]:
            findings.append(
                f"{name}: ZM_Z is FALSE, so the wall clear this product carries does "
                f"nothing on this case and every ZM mutation is a null")
    return {"leg": "structural_walls", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the predicate, on real grids
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real grids."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_electric_pair as product)

    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    def _deposit(fields):
        """A REAL source that publishes the index the injection writes."""
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        centre = (0.25 * float(fields.grid.cell_size[0]), 0.0, 0.0)
        source = VolumeSource(grid=fields.grid, component="Ez", center=centre,
                              size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = [reason for reason in verdict.reasons if "cupy" not in reason]
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left, "needle": needle, "passed": ok})

    _grid, fields, pml = _dcyl((2.0, 0.0, 2.0), 10.0, 0.5)
    coverage = product.cylindrical_real_fused_electric_pair_coverage
    record("dcyl_m0_no_sources", coverage(fields, pml, ()), True)
    # THE ADMITTING DIRECTION, TWICE. A MAGNETIC source is injected in the B/H half
    # and must not disqualify this pair; and the ELECTRIC deposit — which all three
    # corpus rows carry — is ADMITTED because the repair carries it.
    record("dcyl_m0_magnetic_source", coverage(fields, pml, (_Source("B"),)), True)
    record("dcyl_m0_electric_deposit_is_carried",
           coverage(fields, pml, (_deposit(fields),)), True)
    # FAIL CLOSED: a source the repair cannot save is refused BY NAME.
    record("electric_source_without_a_deposit_index",
           coverage(fields, pml, (_Source("D"),)), False,
           "does not publish the index it writes")
    record("undeclared_sources", coverage(fields, pml, None), False,
           "was not declared")
    record("no_pml", coverage(fields, None, ()), False, "no active PML")

    # m != 0 forces complex storage; refused by NAME rather than by the absence of an
    # i*m/r term.
    m_grid = Grid(resolution=10.0, cell_size=(2.0, 0.0, 2.0), cylindrical=True, m=1,
                  boundaries={"z": "metallic"}, courant=0.5)
    m_fields = Fields(grid=m_grid, force_complex_fields=True)
    m_fields.enable_pml_storage()
    record("m_is_one",
           coverage(m_fields,
                    PML(grid=m_grid, thickness={"x": (0, 0.3), "z": 0.3}), ()),
           False, "carries m = 0")

    # A Cartesian grid belongs to the ordinary pair. THE INVARIANT-AXIS PEC is
    # avoided by name: a 2-D cell is translationally invariant along z and Grid
    # refuses a metallic declaration there (grid.py:842-877).
    flat = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    flat_fields = Fields(grid=flat, force_complex_fields=False)
    flat_fields.enable_pml_storage()
    record("cartesian_grid",
           coverage(flat_fields, PML(grid=flat, thickness=0.2), ()),
           False, "not cylindrical")

    findings = [row["case"] for row in rows if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — what the corpus says this is worth
# ---------------------------------------------------------------------------

CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                      "predicate_coverage_triton_2026-08-31_cells")
BOARD = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                     "fusion_matrix_triton_2026-08-31_cells", "fusion_matrix.json")
CELL = ("D->E", "cylindrical PML", "cylindrical")


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
    """The funnel, and whether it lands on the cell the board scored.

    AN ABSENT CENSUS IS NOT AN EMPTY ONE: a staged tree that did not carry the record
    would funnel 0 -> 0 and report "worth zero seam-instances" — a measurement about
    the staging dressed as one about the corpus.
    """
    rows = _census_rows(CENSUS)
    findings: List[str] = []
    if len(rows) != 186:
        return {"leg": "corpus_admission", "device": False, "census": CENSUS,
                "rows": len(rows),
                "findings": [
                    f"the census at {CENSUS} yielded {len(rows)} measured rows, not "
                    f"the 186 this derivation prices against: the record is absent "
                    f"or truncated on this host and NOTHING below is a measurement "
                    f"about the corpus"],
                "passed": False}

    def covered(row: dict, key: str) -> bool:
        entry = row["predicates"].get(key, {})
        return bool(entry.get("covered_modulo_backend", entry.get("covered")))

    def label(row: dict) -> str:
        return f"{row['leg']}:{row['row']}"

    curl = [r for r in rows if covered(r, "cylindrical_curl@step_D")]
    both = [r for r in curl if covered(r, "cylindrical_constitutive@update_E")]
    electric = [r for r in both
                if any(str(kind) != "B" for kind in
                       (r["configuration"].get("source_field_types") or []))]
    funnel = {"rows": len(rows), "curl": len(curl), "both": len(both),
              "carrying_an_electric_deposit": len(electric),
              "cell_rows": sorted(label(r) for r in both)}
    if not both:
        findings.append(
            "the two halves admit NO corpus row together: this product is worth zero "
            "seam-instances and should not have been built")

    board_rows: Optional[List[str]] = None
    if os.path.exists(BOARD):
        board = json.loads(open(BOARD, encoding="utf-8").read())
        board_rows = sorted(
            entry["row"] for entry in board["seam_instances"]
            if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"]) == CELL)
        if board_rows != funnel["cell_rows"]:
            findings.append(
                f"this funnel lands on {funnel['cell_rows']} and the board scored "
                f"{board_rows} into cell {CELL}: two derivations over one census "
                f"disagree, and one of them is wrong")
    else:
        findings.append(
            f"the board at {BOARD} is absent, so the cross-check that this funnel "
            f"lands on the cell the matrix scored could not run")
    # EVERY ROW OF THIS CELL CARRIES AN ELECTRIC DEPOSIT, which is why
    # CARRIES_DEPOSIT_REPAIR is the product rather than one clause of it.
    if both and len(electric) != len(both):
        findings.append(
            f"only {len(electric)} of {len(both)} rows in this cell carry an electric "
            f"deposit; the module's docstring says all of them do")
    return {"leg": "corpus_admission", "device": False, "census": CENSUS,
            "board_rows": board_rows, "funnel": funnel,
            "seam_instances_gained": len(both),
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------

def build_driver(cp, cell, resolution, courant, seed: int, electric: bool,
                 magnetic: bool = True):
    """One Dcyl m = 0 PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=resolution, dimensions=2, cylindrical=True, m=0,
        force_complex_fields=False, courant=courant, boundaries={"z": "metallic"},
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately: the constitutive half multiplies by inverse
    # epsilon per cell, and a uniform material would make an index defect invisible.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml({"x": {"high": 4}, "z": 4})
    if magnetic:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it
        # in the B/H seam, not this one. Carrying one is what stops the source clause
        # from being tested only in its refusing direction.
        driver.add_source({"component": "Hy", "frequency": 0.31,
                           "center": (0.25 * cell[0], 0.0, 0.0), "width": 0.4})
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR. Injected BETWEEN the two halves
        # (driver.py:3294-3299), so the fused launch consumes a pre-injection D and
        # the shipped repair is what puts the difference back. All three corpus rows
        # of this cell carry exactly one.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.25 * cell[0], 0.0, 0.0), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    # BOTH constitutive histories are seeded, and the ELECTRIC one is what makes the
    # curl's ownership mask observable at all. MEASURED on the magnetic twin's first
    # device run (2026-08-20): with ``f_w_E*`` left at zero, ``Ep`` is EXACTLY +0.0 on
    # the r = 0 row and on the z = 0 wall plane forever — the m = 0 axis rule sets
    # ``Dp[0] = 0`` and ``zero_metal_D`` clears ``Dy`` at z = 0 — so a mask mutation
    # came back UNCAUGHT while measuring nothing. That was a defect in the fixture.
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
        # THE LEADING UNDERSCORE IS THE RULE — see PRIVATE_SCRATCH.
        if name.startswith("_"):
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
    """The two separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: PLC0415

    curl = cyl.plan_cylindrical_curl(driver.fields, driver.pml, "step_D")
    constitutive = cyl.plan_cylindrical_constitutive(driver.fields, driver.pml, "E")
    missing = [name for name, plan in (("cylindrical curl", curl),
                                       ("cylindrical constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so this "
            f"leg could not compare the fused launch against the products it replaces")
    # `zero_metal_D` and both fills stay on the ARRAY PATH here: no Triton product
    # owns the wall clear, and the two fills are inert. All three are counted, so a
    # difference in which of them ran is a counter event rather than an invisible
    # correction. THE INJECTION ALSO STAYS ON THE ARRAY PATH, which is what makes
    # this route the right oracle for the carry family: the two certified kernels
    # with the driver's own deposit between them.
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """All FIVE passes absorbed, BRACKETED when the seam carries a deposit.

    Three are CARRIED by the launch; the other two are claimed inert. Absorbing them
    rather than leaving them on the array path is what turns that claim into a byte
    measurement.

    THE BRACKET IS THE SHIPPED ONE. ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan`` are the exact pair ``launch._install_fused_pair`` puts in
    these two slots; a harness that assembled its own could not license the one that
    ships. ``bracket=False`` is the NULL CONTROL and it must diverge.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "D")
    if not seam or not bracket:
        return Route({name: (plan if name == "step_D" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, driver.fields,
                                                 driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_D"] = leading
    plans["update_E"] = trailing
    return Route(plans), leading


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, resolution, courant, steps: int, product,
            mutant: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, electric: bool = False,
            magnetic: bool = True, bracket: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST
    byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    built = [build_driver(cp, cell, resolution, courant, SEED, electric, magnetic)
             for _ in range(3)]
    reference, separate, fused = built
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.cyl_real_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "cell": list(cell),
        "resolution": resolution, "courant": repr(courant),
        "electric_source": bool(electric), "magnetic_source": bool(magnetic),
        "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_cylindrical_real_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        if plan is None:
            verdict = product.cylindrical_real_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources),
                                         bracket=bracket)
        else:
            route = Route({})
        separate_plans = separate_route(separate)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_electric:
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
                "deposit_repairs": (leading.repairs if leading is not None
                                    else None),
            })
            divergence = versus_array or versus_separate
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} "
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
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["private_scratch_on_the_array_route"] = sorted(
            name for name in vars(reference.fields) if name in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            name for name in vars(fused.fields) if name in PRIVATE_SCRATCH)
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        row["elapsed_seconds"] = time.time() - started
        return row
    finally:
        undo()
        for target in built:
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied.

    ``require_launches_at_least`` is for a leg that is EXPECTED to stop early: a leg
    that must diverge breaks at the first divergent step, so an exact launch count and
    the whole-run vacuity census are both statements about a run that did not happen.
    """
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
            f"fewer than the {require_launches_at_least} this leg needs to have "
            f"measured the fused path at all")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(f"VACUOUS: these arrays never moved: {row['arrays_never_moved']}")
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
        inspect.getsource(product.cyl_real_fused_curl_constitutive_D.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = "import triton\nimport triton.language as tl\n\n"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_cyl_real_electric_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + source.replace("cyl_real_fused_curl_constitutive_D",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cyl_real_electric_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite)."""

    def m1_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(D_stuff), undone.

        ``ZM_Z`` is TRUE on every admitted Dcyl grid (:func:`structural_walls_leg`
        measures that), so this rewrite is live on every case. ``ZM_X`` and ``ZM_Y``
        are not touched: their constexprs are structurally false, and a rewrite of a
        dead branch reports "uncaught" while measuring nothing.

        NO ``if`` IN THE REPLACEMENT: ``_rewrite_block`` indents every replacement
        line to the block's first line, so a compound statement in position 0
        produces an empty suite and the mutant fails to IMPORT. The assignments are
        pure identities with no arithmetic, so a signed zero survives them."""
        return _rewrite_block(
            source,
            ["if ZM_Z:", "v0 = tl.where(at_z, 0.0, v0)",
             "v1 = tl.where(at_z, 0.0, v1)"],
            ["v0 = v0", "v1 = v1"])

    def m2_zero_metal_after_the_constitutive(source: str) -> Tuple[str, int]:
        """The wall clear moved AFTER ``update_E`` consumes D — the seam's ORDER.

        TWO EDITS, and they have to move together. The clear is taken off the
        registers (so the constitutive half reads the UN-cleared values) and put onto
        the stores instead (so ``Dr`` and ``Dp`` still end up exactly as the array
        path leaves them). A comparison that only looked at ``D`` would therefore see
        nothing; ``Ex``/``Ey`` and their ``f_w`` carry the un-cleared plane, which is
        what makes this a measurement about the seam's ORDER rather than about the
        clear's existence — that one is m1's."""
        hits = 0
        source, moved_a = _rewrite_block(
            source,
            ["if ZM_Z:", "v0 = tl.where(at_z, 0.0, v0)",
             "v1 = tl.where(at_z, 0.0, v1)"],
            ["v0 = v0", "v1 = v1"])
        hits += moved_a
        source, moved_b = _rewrite_block(
            source,
            ["tl.store(f0 + idx, v0, mask=live)",
             "tl.store(f1 + idx, v1, mask=live)"],
            ["tl.store(f0 + idx, tl.where(at_z, 0.0, v0), mask=live)",
             "tl.store(f1 + idx, tl.where(at_z, 0.0, v1), mask=live)"])
        hits += moved_b
        return source, hits if hits == 2 else 0

    def m3_axis_zero_dropped(source: str) -> Tuple[str, int]:
        """``Dp[0] = 0`` — half of the m = 0 on-axis rule
        (stepping._cylindrical_axis_zero_D:560).

        Without it ``Dy`` keeps its stepped value on r = 0, which ``update_E`` then
        turns into a wrong ``Ey`` row. COMPILE-TIME ABSENT in the magnetic twin: this
        line is inside the ``BACKWARD`` arm its builder binds to 0."""
        return _rewrite_block(
            source, ["v1 = tl.where(at_r, 0.0, v1)"], ["v1 = v1 + 0.0"])

    def m4_axis_hp_add_dropped(source: str) -> Tuple[str, int]:
        """``Dz[0] += (4*Courant) * Hp[0]`` — the OTHER half of the axis rule.

        A POST-add on the updated field, deliberately not folded into the curl
        (measured: the fold breaks m = 0 under PML). Dropping it leaves the r = 0 row
        of ``Dz`` short by exactly that term. COMPILE-TIME ABSENT in the twin, and it
        is the only consumer of the ``hp`` and ``four_dtdx`` arguments — which the
        twin binds and never dereferences."""
        return _rewrite_block(
            source,
            ["v2 = tl.where(at_r, v2 + four_dtdx * tl.load(hp + idx, mask=live, other=0.0),",
             "v2)"],
            ["v2 = v2"])

    def m5_dz_curl_is_the_generic_one(source: str) -> Tuple[str, int]:
        """``Dz``'s curl replaced by the generic four-operand stencil.

        ``step_D`` :425-426 does NOT compute Dz's curl from the raw operands: its
        ``first`` source is the radial PREFIX, backward-differenced. The generic form
        below is the substitution undone. COMPILE-TIME ABSENT in the magnetic twin,
        whose arm takes the FORWARD difference of the extended prefix instead."""
        return _rewrite_block(
            source,
            ["p_here = tl.load(pfx + idx, mask=live, other=0.0)",
             "p_down = tl.load(pfx + o_r, mask=vr, other=0.0)",
             "curl2 = dtdx * ((p_down - p_here) + (a - a_p))"],
            ["curl2 = dtdx * ((b_r - b) + (a - a_p))"])

    def m6_ownership_mask_dropped(source: str) -> Tuple[str, int]:
        """The D-family r = 0 / z = 0 ownership mask
        (stepping._mask_non_owned_cells:1865).

        FOUR rows here against the magnetic twin's two, and all four are compile-time
        absent there. OBSERVABLE ONLY BECAUSE THE FIXTURE SEEDS THE ELECTRIC
        CONSTITUTIVE HISTORY — measured on the twin's first device run, where an
        unseeded ``f_w_E*`` made the same mutation come back UNCAUGHT while measuring
        nothing."""
        return _rewrite_block(
            source,
            ["curl0 = tl.where(at_z, 0.0, curl0)",
             "curl1 = tl.where(at_r, 0.0, curl1)",
             "curl1 = tl.where(at_z, 0.0, curl1)",
             "curl2 = tl.where(at_r, 0.0, curl2)"],
            ["curl0 = curl0", "curl1 = curl1", "curl2 = curl2"])

    def m7_inverse_epsilon_dropped(source: str) -> Tuple[str, int]:
        """The constitutive product reduced to the SCALE = 0 arm — the twin's.

        ``update_E`` writes ``source * inverse_epsilon`` (stepping.py:1011-1013);
        ``update_H`` writes ``source``. A body that dropped the multiply would be the
        MAGNETIC twin's constitutive half welded onto the electric seam, which is the
        single most plausible copy defect this product could carry."""
        return _rewrite_block(
            source, ["src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)"],
            ["src0 = v0"])

    def m8_inverse_epsilon_at_a_shifted_cell(source: str) -> Tuple[str, int]:
        """Component 0's inverse permittivity read one cell away from its own.

        NOT A COMPONENT SWAP: ``FdtdDriver.set_epsilon`` is documented isotropic and
        ``inverse_epsilon_for`` returns the SAME array for Ex, Ey and Ez, so swapping
        one component's pointer for another's is a bitwise no-op — the measured
        reason an earlier product's equivalent mutation came back UNCAUGHT. This
        shifts the INDEX instead, where the varying epsilon
        :func:`build_driver` installs makes it visible, and
        :func:`_assert_the_material_actually_varies` measures that variation on the
        grid the case builds rather than assuming it.

        ``idx - 1`` IS IN BOUNDS EVERYWHERE THE MASK ADMITS, and lane 0 is masked
        off rather than dereferenced. A FORWARD shift would read one float past the
        allocation on the last lane, which is a fault reported as an inert defect.

        WHY NOT MASKING LANE 0 ALONE, which is what the first cut of this mutation
        did: it came back UNCAUGHT on the device, and the reason was measured rather
        than guessed. Stored cell 0 is r = 0 AND z = 0, where component 0's curl is
        already zeroed by the ownership mask and its ``v0`` cleared again by
        ``zero_metal_D``'s ZM_Z row — so ``src0`` is ``0.0 * anything`` there and the
        material it reads cannot matter. The rewrite changed the one cell whose value
        is a fixed point of the defect."""
        return _rewrite_block(
            source, ["src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)"],
            ["src0 = v0 * tl.load(ie0 + idx - 1, mask=live & (idx > 0), other=0.0)"])

    def m9_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source, ["a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"],
            ["a0 = a0 + (kp_0 * src0 - km_0 * prev0)"])

    def m10_history_read_after_write(source: str) -> Tuple[str, int]:
        """f_w read AFTER it is written: wrong only where kms != 0, i.e. in the PML."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
             "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
             "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m11_phi_term_sign_flipped(source: str) -> Tuple[str, int]:
        """The invariant-axis term COMPUTED, and computed wrong.

        phi has n = 1, so its rolled operand equals the original and ``(c_p - c)`` is
        exactly ``+0.0``; ``(c_p + c)`` is ``2c``, which is loud. The ELISION itself
        is refused by the TRANSCRIPTION leg instead, which requires the four-operand
        line verbatim and is deterministic."""
        return _rewrite_block(
            source, ["curl0 = dtdx * ((c_p - c) + (b - b_z))"],
            ["curl0 = dtdx * ((c_p + c) + (b - b_z))"])

    def m12_coefficient_index_moved(source: str) -> Tuple[str, int]:
        """The constitutive coefficient read on the WRONG axis for component 0.

        ``E_CONSTITUTIVE_TERMS`` indexes each component on its OWN axis; reading
        component 0's on z is MEEP's ``dsigw`` for the wrong direction. Scored on a
        SQUARE case, where the read stays in bounds and the leg measures a wrong
        VALUE rather than a fault."""
        return _rewrite_block(
            source, ["kp_0 = tl.load(kp0 + i, mask=live, other=0.0)"],
            ["kp_0 = tl.load(kp0 + k, mask=live, other=0.0)"])

    def m13_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_zero_metal_dropped", "the wall clear's slot", "caught",
         m1_zero_metal_dropped),
        ("m2_zero_metal_after_the_constitutive", "the seam's ORDER", "caught",
         m2_zero_metal_after_the_constitutive),
        ("m3_axis_zero_dropped", "Dp[0] = 0", "caught", m3_axis_zero_dropped),
        ("m4_axis_hp_add_dropped", "the on-axis Dz add", "caught",
         m4_axis_hp_add_dropped),
        ("m5_dz_curl_is_the_generic_one", "Dz's prefix substitution", "caught",
         m5_dz_curl_is_the_generic_one),
        ("m6_ownership_mask_dropped", "the D-family ownership mask", "caught",
         m6_ownership_mask_dropped),
        ("m7_inverse_epsilon_dropped", "the SCALE=1 arm", "caught",
         m7_inverse_epsilon_dropped),
        ("m8_inverse_epsilon_at_a_shifted_cell", "the material index", "caught",
         m8_inverse_epsilon_at_a_shifted_cell),
        ("m9_constitutive_association", "float32 association", "caught",
         m9_constitutive_association),
        ("m10_history_read_after_write", "the f_w ordering", "caught",
         m10_history_read_after_write),
        ("m11_phi_term_sign_flipped", "the invariant-axis term", "caught",
         m11_phi_term_sign_flipped),
        ("m12_coefficient_index_moved", "dsigw's own axis", "caught",
         m12_coefficient_index_moved),
        ("m13_commuted_multiply", "commuted multiply", "null", m13_commuted_multiply),
    )


#: Mutation id -> the CASES index it is scored on. Every case here is the same Dcyl
#: m = 0 declaration with the same compiled branches, so no mutation can land on a
#: grid that fails to enter the lines it rewrites — the dead-branch hazard this family
#: carries is the ZM_X / ZM_Y rows, and no mutation touches them.
MUTATION_CASE: Dict[str, int] = {}

#: CASES[0] is SQUARE (nr == nz), and that is load-bearing for one mutation rather
#: than a default. ``m12_coefficient_index_moved`` reads ``kp0`` — a length-nr column
#: — at the z index; on ``short_wide`` (nr = 11, nz = 32) that is an OUT-OF-BOUNDS
#: read, which is a memory fault reported as an inert defect rather than a wrong
#: answer. On a square case the read is in bounds and the leg measures the wrong
#: VALUE, which is the defect being armed.
DEFAULT_MUTATION_CASE = 0

#: Constexpr names no mutation may rewrite, because they are false on every admitted
#: configuration (measured by :func:`structural_walls_leg`).
DEAD_CONSTEXPRS: Tuple[str, ...] = ("ZM_X", "ZM_Y")


#: Mutations whose visibility rests on the material VARYING between adjacent cells.
#: Checked by BUILDING the grid the case builds, not declared: a uniform epsilon
#: would make the rewrite a bitwise no-op and the leg would report a real defect as
#: uncaught, which is exactly what an isotropic fixture did to an earlier product's
#: component-swap mutation.
MUTATION_REQUIRES_A_VARYING_MATERIAL: Tuple[str, ...] = (
    "m8_inverse_epsilon_at_a_shifted_cell",)


def _assert_the_material_actually_varies(case_name: str, cell, resolution,
                                         courant) -> None:
    """``inv_eps`` must differ between adjacent stored cells on the scored grid."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415

    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=0,
                boundaries={"z": "metallic"}, courant=courant)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = grid.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(np.float32))
    flat = np.asarray(fields.inverse_epsilon_for("Ex")).reshape(-1)
    same = int(np.count_nonzero(flat[1:] == flat[:-1]))
    if same == flat.size - 1:
        raise AssertionError(
            f"mutation on case {case_name!r}: inv_eps is CONSTANT across every "
            f"adjacent stored cell, so reading it one cell away is a bitwise no-op "
            f"and the leg would report a real defect as uncaught")


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on, with the dead-branch check."""
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    if not 0 <= index < len(CASES):
        raise AssertionError(f"mutation {name} names case index {index}, which is "
                             f"outside the {len(CASES)} this gate builds")
    case = CASES[index]
    if name in MUTATION_REQUIRES_A_VARYING_MATERIAL:
        _assert_the_material_actually_varies(case[0], case[1], case[2], case[3])
    return index, case


def _dead_blocks(source: str) -> Dict[str, List[str]]:
    """Each structurally-false constexpr's guarded block, as STRIPPED lines.

    The block runs from its ``if`` to the next line at or below its own indent.
    """
    lines = source.splitlines()
    blocks: Dict[str, List[str]] = {}
    for constexpr in DEAD_CONSTEXPRS:
        for index, line in enumerate(lines):
            if line.strip() != f"if {constexpr}:":
                continue
            indent = len(line) - len(line.lstrip())
            end = index + 1
            while (end < len(lines)
                   and (not lines[end].strip()
                        or (len(lines[end]) - len(lines[end].lstrip())) > indent)):
                end += 1
            blocks[constexpr] = [item.strip() for item in lines[index:end]
                                 if item.strip()]
            break
    return blocks


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Is ``needle`` a consecutive run inside ``haystack``?"""
    n = len(needle)
    return any(list(haystack[start:start + n]) == list(needle)
               for start in range(len(haystack) - n + 1))


def _assert_no_dead_branch_rewrites(source: str) -> List[str]:
    """No rewrite may change a line whose enclosing constexpr is structurally false.

    COMPARED AS TEXT, NOT BY LINE INDEX, and that is a correction rather than a
    style: several of these rewrites replace a two-line block with one line, so every
    later line SHIFTS. An index-based comparison then reports a rewrite located
    BEFORE a dead block as having touched it — measured here on
    ``m4_axis_hp_add_dropped``, which rewrites the axis rule and shifts both ZM
    blocks by one. What has to hold is that the dead block's own statements survive
    the rewrite intact, wherever they end up.
    """
    findings: List[str] = []
    blocks = _dead_blocks(source)
    missing = [name for name in DEAD_CONSTEXPRS if name not in blocks]
    if missing:
        findings.append(
            f"the kernel carries no {missing} block at all, so this check cannot "
            f"tell a rewrite that avoided one from a body that dropped it")
    for name, _why, _expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        if not hits:
            findings.append(f"{name}: the rewrite matched nothing")
            continue
        changed = [line.strip() for line in mutated.splitlines() if line.strip()]
        for constexpr, block in blocks.items():
            if not _contains_run(changed, block):
                findings.append(
                    f"{name} rewrites a line inside the {constexpr} block, whose "
                    f"constexpr is FALSE on every admitted Dcyl grid: the mutation "
                    f"would report 'uncaught' while measuring nothing")
    return findings


def dead_branch_leg() -> Dict[str, Any]:
    """Every mutation is armed, and none of them rewrites a structurally dead branch."""
    findings = _assert_no_dead_branch_rewrites(
        _shipped_text("cyl_real_fused_curl_constitutive_D"))
    return {"leg": "dead_branch", "device": False,
            "mutations": [name for name, _w, _e, _r in mutation_table()],
            "dead_constexprs": list(DEAD_CONSTEXPRS),
            "findings": findings, "passed": not findings}


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
        kernel_name = f"mutant_{index}_cyl_real_D"
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
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and two it must ADMIT."""

    class _Electric:
        field_type = "D"

    def _deposit(fields):
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        centre = (0.25 * float(fields.grid.cell_size[0]), 0.0, 0.0)
        source = VolumeSource(grid=fields.grid, component="Ez", center=centre,
                              size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    cell, resolution, courant = CASES[0][1], CASES[0][2], CASES[0][3]
    rows: List[Dict[str, Any]] = []
    for name, needle, sources in (
        ("electric_source_without_a_deposit_index",
         "does not publish the index it writes", (_Electric(),)),
        ("electric_deposit_is_carried", None, lambda f: (_deposit(f),)),
        ("undeclared_sources", "was not declared", None),
        ("no_sources", None, ()),
    ):
        driver = build_driver(cp, cell, resolution, courant, SEED, electric=False,
                              magnetic=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.cylindrical_real_fused_electric_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_cylindrical_real_fused_electric_pair(
                driver.fields, driver.pml, bound)
            if needle is None:
                passed = bool(verdict.covered) and plan is not None
            else:
                passed = ((not verdict.covered) and plan is None
                          and any(needle in reason for reason in verdict.reasons))
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "expected": "admitted" if needle is None else f"refused: {needle}",
                "reasons": list(verdict.reasons), "plan_is_none": plan is None,
                "passed": passed,
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
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
    }
    try:
        import triton  # noqa: PLC0415

        payload["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        payload["triton"] = f"absent: {exc!r}"
    if cp is not None:
        # THE CUPY VERSION IS PART OF THE CLAIM, not decoration: everything
        # downstream of the radial prefix scan is bit-identical only for a given
        # `cupy.cumsum` summation order.
        payload["cupy"] = cp.__version__
        payload["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_cylindrical_real_fused_electric_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep'.")
    args = parser.parse_args(argv)

    out = args.out
    if out.endswith(".json"):
        artifact = out
    else:
        os.makedirs(out, exist_ok=True)
        artifact = os.path.join(out, "gate.json")

    payload: Dict[str, Any] = {
        "gate": "triton_cylindrical_real_fused_electric_pair",
        "product": "meep_gpu.triton_kernels.cylindrical_real_fused_electric_pair",
        "kernel": "cyl_real_fused_curl_constitutive_D",
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
                predicate_leg, corpus_admission_leg, dead_branch_leg):
        started = time.time()
        row = leg()
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

    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_electric_pair as product)

    backends.guard_kernel_compilation(cp)
    # THE MACHINE AND THE DEVICE, so this artifact can be SEEDED into a weld. The
    # weld contract binds a host string to the record's validated Triton and
    # capability lists; `environment()` below records the device but no hostname, and
    # a seeding tool that supplied the missing half would be typing a fact instead of
    # reading one. Resolved by name off the parity directory, as gate_provenance is.
    import triton_device_identity  # noqa: PLC0415

    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[4] for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for name, cell, resolution, courant, steps in CASES:
        row = run_leg(cp, f"quiet:{name}", cell, resolution, courant, steps, product)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    index_by_name = {case[0]: case for case in CASES}
    for case_name in CARRY_CASES:
        name, cell, resolution, courant, _steps = index_by_name[case_name]
        row = run_leg(cp, f"carry:{name}", cell, resolution, courant, CARRY_STEPS,
                      product, electric=True, magnetic=False)
        passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, resolution, courant, _steps = index_by_name[case_name]
        row = run_leg(cp, f"null_control:{name}", cell, resolution, courant,
                      CARRY_STEPS, product, electric=True, magnetic=False,
                      bracket=False)
        # REQUIRES DIVERGENCE, and the launch floor is "at least one" rather than the
        # budget: a leg that must diverge STOPS at the first divergent step, so an
        # exact count and a whole-run moved-state census are statements about steps
        # that never ran. The ORACLE control is still checked unconditionally, which
        # is what says the divergence is the missing bracket rather than a broken
        # harness.
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it "
                      "consumes a pre-injection D and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== armed harness mutations ===")
    name, cell, resolution, courant, _steps = CASES[0]
    row = run_leg(cp, "armed:no_substitution", cell, resolution, courant, 3, product,
                  install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", cell, resolution, courant, 3,
                  product, freeze_electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.cyl_real_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, artifact)

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
        "host": "the measurement machine; every device row above ran there",
    }
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

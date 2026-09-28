"""THE D->E OFF-DIAGONAL FUSED PAIR: a MEASURED REFUSAL, not a product.

WHAT WAS ASKED. Weld the D curl into the off-diagonal ``update_E`` — the two
cells that rank 3rd and 4th on the Triton fusion matrix by REACHABLE demand
(``results/fusion_matrix_triton_2026-08-20/fusion_matrix.json``)::

    ceiling 9  (rows 19)  D->E  folded PML -> folded off-diagonal
    ceiling 8  (rows 16)  D->E  PML        -> off-diagonal

17 seam-instances of 387, and they would have been the FIRST at D->E on Triton
(which serves 0 of 186 there today).

WHAT THIS GATE MEASURES INSTEAD. That no such product can exist, on Triton or on
any other backend, for a reason that is structural rather than a matter of
effort, bindings or scope:

    **THE OFF-DIAGONAL CONSTITUTIVE ARM IS A STENCIL OVER THE CURL ARM'S OWN
    IN-PLACE OUTPUT.**

``stepping._offdiagonal_terms`` (stepping.py:1235-1253) reads each PARTNER's D
volume at FOUR indices — its own, one down the partner axis, one up the
component's own axis, and the corner — and those D volumes are exactly what
``step_D`` writes, in place, in the first half of the same launch. A fused
launch therefore has program A reading a cell that program B is concurrently
writing, with no grid-wide barrier available to order them. The diagonal
constitutive (:func:`meep_gpu.triton_kernels.kernels.constitutive_step`, and the
shipped ``fused_curl_constitutive_D`` built on it) reads D at ``idx`` and
NOWHERE ELSE, which is why the pointwise pair fuses and this one does not.

Recomputing the neighbour curl inside the program instead of loading it does not
escape: the split-field recurrence (stepping.py:1952, transcribed at
kernels.py:497-509) reads ``D[m]`` and ``fu_D[m]`` at the neighbour index ``m``
— the PRE-step values of the two arrays the curl half overwrites — so the halo
recompute is a read of the same clobbered state by another route. THAT SECOND
CLAUSE IS A DERIVATION FROM A PARSED FACT (leg 0 parses the recurrence's own
index expressions), NOT A DEVICE MEASUREMENT; only the plain-load form is
executed below.

===========================================================================
THE DISCRIMINATING EXPERIMENT
===========================================================================

An assembled fused kernel that disagrees with its reference proves nothing on
its own — a transcription error looks exactly the same. Four legs separate the
two, and the release verdict is their CONJUNCTION:

  C1  POINTWISE CONTROL. The SHIPPED ``fused_curl_constitutive_D`` (diagonal
      constitutive) on the subject's own fixture, same block sweep, same
      repeats. MUST be bit-identical: same fusion technique, same seam, same
      schedule — the only difference is that its constitutive half reads no
      neighbour.
  C2  ZERO-COEFFICIENT CONTROL. The SUBJECT kernel with every surviving row
      coefficient exactly +0.0. The stencil still executes and its loads are
      still racy; the products are 0.0 whatever they read. MUST be
      bit-identical — which places the disagreement inside the coupling term
      and nowhere else in the weld.
  C3  CURL-IS-IDENTITY CONTROL. The SUBJECT kernel on a fixture where the curl
      half provably leaves D and fu_D unchanged (all axes periodic, uniform H
      so every curl operand cancels to an exact 0.0, fu_D seeded +0.0, kms and
      sinv exactly 1.0). The racy read then returns the same bits whether it
      lands before or after the neighbour's store. MUST be bit-identical — this
      is what certifies the weld's ARITHMETIC, deterministically, without any
      assumption about intra-CTA visibility. The precondition is MEASURED
      (``curl_is_identity_holds``), never assumed.
  S1  THE SUBJECT. The same kernel, same shape, real random H and real
      coefficients. MUST NOT be bit-identical, and its differing-word count MUST
      vary across repeats at a fixed block — nondeterminism is the signature of
      a race, and a transcription error would be perfectly repeatable.

C1 + C2 + C3 say the weld is right. S1 says it is wrong anyway. The only
difference between C3 and S1 is whether the neighbour's stored value CHANGED,
which is the claim.

===========================================================================
WHAT THIS GATE DOES NOT CLAIM
===========================================================================

* It does not claim a seam-instance. The delta is ZERO and the fusion matrix is
  re-derived with the new predicate rather than edited by hand.
* It does not claim anything about Metal. Metal's own matrix already refuses
  both cells on the binding ceiling ([34, 36] pointers against 30); that is a
  SECOND and INDEPENDENT refusal and the count is not the operative one.
* The SUBJECT KERNEL IS NOT A PRODUCT AND IS NOT SHIPPED. It lives in this file,
  is assembled by splicing the two shipped bodies (with a declared rename map
  that is reversed and byte-checked), and exists only to be refuted.

Usage::

    # laptop: the structural legs, no device
    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_triton_fused_offdiag_electric.py --no-device \\
      --out parity/meep_gpu/results/<fresh>/no_device.json

    # CUDA host, verified-empty device
    CUDA_VISIBLE_DEVICES=<n> python -u \\
      parity/meep_gpu/gate_triton_fused_offdiag_electric.py \\
      --subnormal-policy keep \\
      --out parity/meep_gpu/results/<fresh>/keep.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(HERE), str(API_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # laptop
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

import probe_fused_kernel_bit_identity as probe  # noqa: E402

SEED = 20260820
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
CODE_OF = {PERIODIC: 0, METALLIC: 1}

D_NAMES = ("Dx", "Dy", "Dz")
FU_NAMES = ("fu_Dx", "fu_Dy", "fu_Dz")
H_NAMES = ("Hx", "Hy", "Hz")
E_NAMES = ("Ex", "Ey", "Ez")
FW_NAMES = ("f_w_Ex", "f_w_Ey", "f_w_Ez")
#: Everything the D->E seam writes. The comparison basis, every leg.
STATE_NAMES = D_NAMES + FU_NAMES + E_NAMES + FW_NAMES


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], out: Path) -> None:
    """Atomic rewrite after every leg (the progress-reporting rule), provenance-stamped."""
    from gate_provenance import stamp as _stamp  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    _stamp(payload)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, out)


def case_rng(*labels: Any) -> np.random.Generator:
    """PER-CASE SEED FROM A DIGEST, never ``hash()``.

    ``hash()`` of a tuple of strings is salted with PYTHONHASHSEED, so a
    hash-seeded gate draws a different fixture in every process and a failing
    case cannot be replayed. This is reproducible across processes and machines.
    """
    key = "|".join(str(part) for part in labels).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return np.random.default_rng(SEED + offset)


# ===========================================================================
# LEG 0 — THE STRUCTURAL CENSUS, parsed from the shipped sources
# ===========================================================================
#
# THE MIRRORED-EVALUATOR TRAP (measured 2026-08-20: three planted assembly
# defects each left 86 of 87 tests passing) says a test that re-implements a
# kernel's assembly mirrors a defect instead of executing it. So nothing here
# restates what a kernel does: every classification below is PARSED out of the
# emitted source with ``ast``, and the marker names it filters on are asserted
# to exist in the thing they filter (the name-drift trap).

#: Every shipped Triton ``update_E`` kernel, and the parameters that carry the D
#: volumes the curl arm writes. EVERY ONE of them spells those ``g0/g1/g2``,
#: which the census asserts rather than assumes: a name that is not a parameter
#: of the kernel is reported as name drift, never silently skipped.
#: ``(module, kernel, D-volume parameter names)``.
CONSTITUTIVE_ARMS: Tuple[Tuple[str, str, Optional[Tuple[str, ...]]], ...] = (
    ("kernels", "constitutive_step", ("g0", "g1", "g2")),
    ("complex_fields", "bloch_constitutive_step", ("g0", "g1", "g2")),
    ("dispersive_update_e", "constitutive_step_dispersive", ("g0", "g1", "g2")),
    ("no_pml_stored_e", "stored_e_constitutive_step", ("g0", "g1", "g2")),
    ("complex_no_pml_stored_e", "complex_stored_e_step", ("g0", "g1", "g2")),
    ("offdiag_update_e", "offdiag_constitutive_step", ("g0", "g1", "g2")),
    ("folded_offdiag_update_e", "folded_offdiag_constitutive_step",
     ("g0", "g1", "g2")),
    ("complex_offdiag_update_e", "complex_offdiag_update_e_step",
     ("g0", "g1", "g2")),
    ("folded_offdiag_dispersive_update_e",
     "folded_offdiag_dispersive_constitutive_step", ("g0", "g1", "g2")),
    ("nonlinear_update_e", "nonlinear_constitutive_step", ("g0", "g1", "g2")),
)

#: The curl kernels whose recurrence is checked for reading its own target.
#: ``symmetry`` spells its folded curl ``pml_curl_step_folded``; the folded
#: CONSTITUTIVE has no kernel of its own — ``plan_folded_constitutive`` launches
#: ``kernels.constitutive_step``, which the census classifies above.
CURL_ARMS: Tuple[Tuple[str, str], ...] = (
    ("kernels", "pml_curl_step"),
    ("symmetry", "pml_curl_step_folded"),
    ("complex_fields", "bloch_pml_curl_step"),
)


def kernel_dir() -> Path:
    import meep_gpu  # noqa: PLC0415

    return Path(meep_gpu.__file__).resolve().parent / "triton_kernels"


def _find_function(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _add_leaves(node: ast.AST) -> List[ast.AST]:
    """Flatten a top-level ``a + b + c`` chain into its leaves."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_leaves(node.left) + _add_leaves(node.right)
    return [node]


def _offset_of(arg: ast.AST, pointers: Sequence[str]) -> Optional[str]:
    """The offset in ``P + <offset>``, however the Add chain associates.

    ``tl.load(g0 + 2 * idx + 1)`` parses as ``((g0 + 2*idx) + 1)``, so matching
    only ``BinOp(left=Name)`` would miss the second word of every complex
    interleaved load and classify a pointwise arm as a stencil. Measured: it
    did exactly that to ``complex_fields.bloch_constitutive_step``.
    """
    leaves = _add_leaves(arg)
    named = [leaf for leaf in leaves
             if isinstance(leaf, ast.Name) and leaf.id in pointers]
    if len(named) != 1:
        return None
    rest = [leaf for leaf in leaves if leaf is not named[0]]
    return " + ".join(ast.unparse(leaf) for leaf in rest) if rest else "0"


def _store_offsets(node: ast.FunctionDef) -> List[str]:
    """Every ``tl.store(P + EXPR, ...)`` offset in the kernel.

    THIS IS THE KERNEL'S OWN DEFINITION OF "THE CELL I AM RESPONSIBLE FOR", read
    out of the kernel rather than assumed. A pointwise constitutive arm loads
    its source at exactly these; a stencil loads somewhere else as well. Using
    the kernel's own store set as the yardstick means the census needs no
    vocabulary of index-variable names to keep in step with the tree, and it
    absorbs the complex interleave (``word``, ``word + 1``) for free.
    """
    offsets: List[str] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (isinstance(func, ast.Attribute) and func.attr == "store"
                and isinstance(func.value, ast.Name) and func.value.id == "tl"
                and call.args):
            leaves = _add_leaves(call.args[0])
            rest = [leaf for leaf in leaves[1:]]
            offsets.append(" + ".join(ast.unparse(leaf) for leaf in rest)
                           if rest else "0")
    return offsets


def _load_offsets(node: ast.FunctionDef, pointers: Sequence[str],
                  own_cell: Sequence[str]) -> List[str]:
    """Every ``tl.load(P + EXPR, ...)`` offset expression, for P in ``pointers``.

    Also follows ONE level of helper call: a body that hands a pointer to
    ``_offdiag_term(g1, ...)`` loads it inside the helper, so the call's INDEX
    ARGUMENTS are recorded too and classified by the same rule. The helper is
    recognised by being handed a pointer, NOT by a name suffix — a suffix rule
    that matched nothing (``_four_point_sum`` does not end in ``_term``) would
    disable the classification silently.
    """
    found: List[str] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        is_load = (isinstance(func, ast.Attribute) and func.attr == "load"
                   and isinstance(func.value, ast.Name) and func.value.id == "tl")
        if is_load and call.args:
            offset = _offset_of(call.args[0], pointers)
            if offset is not None:
                found.append(offset)
            continue
        if isinstance(func, ast.Name):
            names = [a.id for a in call.args if isinstance(a, ast.Name)]
            if any(name in pointers for name in names):
                for arg in call.args:
                    text = ast.unparse(arg)
                    # An argument counts as an index if the kernel itself
                    # stores at it, or if it is built from the flat strides.
                    if text in own_cell or "nyz" in text or "* nz" in text:
                        found.append(text)
    return found


def stencil_census() -> Dict[str, Any]:
    """Which constitutive arms read the curl arm's output at a NEIGHBOUR index.

    A cell classified ``pointwise`` reads its D volumes only at ``idx``; that is
    the property a cross-sub-step fusion needs, because ``idx`` is the one index
    the fusing program computed itself. Anything else is a read of another
    program's cell.
    """
    root = kernel_dir()
    rows: List[Dict[str, Any]] = []
    for module, kernel, pointers in CONSTITUTIVE_ARMS:
        path = root / f"{module}.py"
        row: Dict[str, Any] = {"module": module, "kernel": kernel}
        if not path.exists():
            row["status"] = "module absent"
            rows.append(row)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = _find_function(tree, kernel)
        if node is None:
            # NAME DRIFT: a marker that matches nothing must SAY SO, never
            # silently disable the check it feeds.
            row["status"] = "KERNEL NAME NOT FOUND — census cannot classify"
            rows.append(row)
            continue
        params = [a.arg for a in node.args.args]
        row["parameters"] = len(params)
        if pointers is None:
            # Not hand-declared: infer the D-volume parameters by position from
            # the shipped comment convention is NOT safe, so this arm is
            # recorded as unclassified rather than guessed at.
            row["status"] = "not classified by this census"
            rows.append(row)
            continue
        missing = [name for name in pointers if name not in params]
        if missing:
            row["status"] = (f"declared D-volume parameters {missing} are not "
                             f"parameters of {kernel} — name drift")
            rows.append(row)
            continue
        own_cell = sorted(set(_store_offsets(node)))
        row["own_cell_offsets"] = own_cell
        offsets = sorted(set(_load_offsets(node, pointers, own_cell)))
        row["d_volume_load_offsets"] = offsets
        row["distinct_offsets"] = len(offsets)
        if not offsets:
            # NOT a stencil verdict: finding no load at all means the pattern
            # missed, and a pattern that matches nothing must say so rather than
            # answer the question it never asked.
            row["status"] = ("no load of a declared D volume was found — the "
                             "census pattern did not match; NOT classified")
            rows.append(row)
            continue
        # POINTWISE iff every load offset is one the kernel itself STORES at.
        foreign = [text for text in offsets if text not in own_cell]
        row["offsets_outside_the_cell_it_writes"] = foreign
        row["pointwise"] = not foreign
        row["status"] = "pointwise" if row["pointwise"] else "STENCIL"
        rows.append(row)

    curls: List[Dict[str, Any]] = []
    for module, kernel in CURL_ARMS:
        path = root / f"{module}.py"
        entry: Dict[str, Any] = {"module": module, "kernel": kernel}
        if not path.exists():
            entry["status"] = "module absent"
            curls.append(entry)
            continue
        node = _find_function(ast.parse(path.read_text(encoding="utf-8")), kernel)
        if node is None:
            entry["status"] = "KERNEL NAME NOT FOUND"
            curls.append(entry)
            continue
        targets = ("f0", "f1", "f2")
        aux = ("u0", "u1", "u2")
        params = [a.arg for a in node.args.args]
        missing = [n for n in targets + aux if n not in params]
        if missing:
            entry["status"] = f"name drift: {missing}"
            curls.append(entry)
            continue
        entry["reads_own_target"] = bool(_load_offsets(node, targets, _store_offsets(node)))
        entry["reads_own_auxiliary"] = bool(_load_offsets(node, aux, _store_offsets(node)))
        entry["status"] = (
            "the recurrence reads the PRE-step value of both arrays it writes"
            if entry["reads_own_target"] and entry["reads_own_auxiliary"]
            else "does NOT read its own outputs")
        curls.append(entry)

    classified = [r for r in rows if "pointwise" in r]
    return {
        "constitutive_arms": rows,
        "curl_arms": curls,
        "classified": len(classified),
        "pointwise": sum(1 for r in classified if r["pointwise"]),
        "stencil": sum(1 for r in classified if not r["pointwise"]),
        "what_it_licenses": (
            "a STENCIL constitutive arm cannot be fused with the curl that "
            "writes its source volumes in place: the neighbour cells belong to "
            "other programs and no grid-wide barrier exists inside one launch"),
    }


# ===========================================================================
# THE SUBJECT — assembled by SPLICING the two shipped bodies, and checked back
# ===========================================================================
#
# ASSEMBLE FROM CERTIFIED ARITHMETIC, VERBATIM. Not a re-derivation and not a
# transcription: the two halves are cut out of their own source files and the
# only edits are (i) a declared token rename, because both kernels spell their
# parameters f0/g0/e0 and the fused signature needs one name apiece, and (ii)
# the ONE fusion substitution the shipped pairs document — the constitutive
# half's own-index load of D becomes the register the curl half just computed.
# ``build_subject_source`` REVERSES both and asserts the result is byte-equal
# to the shipped slices, so "verbatim" is machine-checked rather than claimed.

#: Applied SIMULTANEOUSLY (one pass, word boundaries) to the off-diagonal
#: constitutive body. Left: its own parameter name. Right: the fused signature's.
OFFDIAG_RENAME: Dict[str, str] = {
    "f0": "e0", "f1": "e1", "f2": "e2",        # E targets
    "g0": "f0", "g1": "f1", "g2": "f2",        # D volumes == the curl's targets
    "e0": "ie0", "e1": "ie1", "e2": "ie2",     # inverse epsilon
    "u01": "r01", "u02": "r02",                # row coefficients (u* is curl aux)
    "u11": "r11", "u12": "r12",
    "u21": "r21", "u22": "r22",
}

CURL_START = "idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"
CURL_END = "==================== the constitutive half"
OFFDIAG_START = "--- per-axis neighbour indices, both directions, with the ghost"


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    node = _find_function(ast.parse(text), name)
    if node is None:
        raise RuntimeError(f"{name} not found in {path}")
    lines = text.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def _slice_between(source: str, start_marker: str,
                   end_marker: Optional[str]) -> str:
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if start_marker in line)
    if end_marker is None:
        stop = len(lines)
    else:
        stop = next(i for i, line in enumerate(lines) if end_marker in line)
    return textwrap.dedent("\n".join(lines[start:stop]).rstrip())


def _rename(text: str, mapping: Dict[str, str]) -> str:
    pattern = re.compile(r"\b(" + "|".join(
        sorted(map(re.escape, mapping), key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(0)], text)


#: The one fusion edit, and its exact inverse. Keyed by component index.
FUSION_EDIT: Tuple[Tuple[str, str], ...] = tuple(
    (f"gs{n} = tl.load(f{n} + idx, mask=live, other=0.0)", f"gs{n} = v{n}")
    for n in range(3))

SUBJECT_HEADER = '''
"""ASSEMBLED REFUTATION SUBJECT. Never shipped, never dispatched, never an arm.

Built by ``gate_triton_fused_offdiag_electric.build_subject_source`` from the
shipped ``kernels.fused_curl_constitutive_D`` curl half and the shipped
``offdiag_update_e.offdiag_constitutive_step`` constitutive body. It is WRONG on
purpose and the gate exists to measure how.
"""
import triton
import triton.language as tl

from meep_gpu.triton_kernels.kernels import METALLIC, PERIODIC
from meep_gpu.triton_kernels.offdiag_update_e import (
    _masked_row_sum, _offdiag_term)


@triton.jit
def fused_curl_offdiag_constitutive_D(
    f0, f1, f2,                     # curl targets:      Dx, Dy, Dz
    u0, u1, u2,                     # curl auxiliaries:  fu_Dx, fu_Dy, fu_Dz
    g0, g1, g2,                     # curl sources:      Hx, Hy, Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER sub-lattice
    e0, e1, e2,                     # constitutive targets:     Ex, Ey, Ez
    w0, w1, w2,                     # constitutive auxiliaries: f_w_E*
    ie0, ie1, ie2,                  # inverse-epsilon volumes
    r01, r02, r11, r12, r21, r22,   # the six off-diagonal row coefficients
    kp0, km0, kp1, km1, kp2, km2,   # constitutive kps/kms, HALF-INTEGER
    nx, ny, nz, n_elem, dtdx,
    BACKWARD: tl.constexpr,
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
    R01: tl.constexpr, R02: tl.constexpr,
    R11: tl.constexpr, R12: tl.constexpr,
    R21: tl.constexpr, R22: tl.constexpr,
    WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
    BLOCK: tl.constexpr,
):
'''


def build_subject_source(pointwise_variant: bool = False,
                         mutation: Optional[str] = None
                         ) -> Tuple[str, Dict[str, Any]]:
    """The spliced fused kernel, plus the record that the splice is verbatim.

    ``pointwise_variant`` is the PLANTED DEFECT that must flip the release
    verdict: every neighbour index in the coupling's loads is rewritten to
    ``idx``, which turns the stencil into a pointwise arm. The kernel is then
    numerically wrong against the array path — and bit-identical between the
    fused and two-launch routes, because there is no longer a cross-program
    read. If the gate's refusal survived that rewrite, the refusal would be
    measuring something other than the stencil.
    """
    root = kernel_dir()
    curl_full = _function_source(root / "kernels.py", "fused_curl_constitutive_D")
    curl_half = _slice_between(curl_full, CURL_START, CURL_END)
    offdiag_full = _function_source(root / "offdiag_update_e.py",
                                    "offdiag_constitutive_step")
    offdiag_body = _slice_between(offdiag_full, OFFDIAG_START, None)

    renamed = _rename(offdiag_body, OFFDIAG_RENAME)
    fused_body = renamed
    for original, replacement in FUSION_EDIT:
        if original not in fused_body:
            raise RuntimeError(
                f"the fusion substitution target {original!r} is absent from "
                f"the renamed off-diagonal body — the splice markers or the "
                f"shipped kernel have moved")
        fused_body = fused_body.replace(original, replacement)

    # --- machine-checked verbatim: reverse both edits, compare bytes ---------
    reversed_body = fused_body
    for original, replacement in FUSION_EDIT:
        reversed_body = reversed_body.replace(replacement, original)
    inverse = {v: k for k, v in OFFDIAG_RENAME.items()}
    reversed_body = _rename(reversed_body, inverse)
    verbatim = reversed_body == offdiag_body

    planted: Optional[Dict[str, Any]] = None
    if pointwise_variant:
        # Every helper-call index argument that is not already ``idx`` becomes
        # ``idx``. Those are exactly the o_d / o_u / o_ud slots.
        before = fused_body
        fused_body = re.sub(r"^(\s+)\w+ \* nyz \+ \w+ \* nz \+ \w+,$",
                            r"\1idx,", fused_body, flags=re.M)
        planted = {"kind": "pointwise_variant",
                   "lines_rewritten": sum(
                       1 for a, b in zip(before.splitlines(),
                                         fused_body.splitlines()) if a != b)}
        if not planted["lines_rewritten"]:
            raise RuntimeError("the pointwise plant rewrote NOTHING: it would "
                               "report a null while measuring nothing")

    if mutation == "flatten_curl_parens":
        # ``((c_y - c) + (b - b_z))`` -> ``(((c_y - c) + b) - b_z)``, the
        # association C and Triton alike would choose (kernels.py's note 2). A
        # DIFFERENT float32 number. Spelling it ``c_y - c + (b - b_z)`` would
        # re-parse to the original and mutate nothing — measured on the first
        # device run, where it scored a false NULL.
        before = curl_half
        curl_half = curl_half.replace(
            "curl0 = dtdx * ((c_y - c) + (b - b_z))",
            "curl0 = dtdx * ((c_y - c) + b - b_z)")
        if curl_half == before:
            raise RuntimeError("flatten_curl_parens rewrote nothing")
    elif mutation == "mispair_row_coefficients":
        before = fused_body
        fused_body = fused_body.replace("f1, r01, idx", "f1, r02, idx")
        fused_body = fused_body.replace("f2, r02, idx", "f2, r01, idx")
        if fused_body == before:
            raise RuntimeError("mispair_row_coefficients rewrote nothing")
    elif mutation == "reload_own_D_from_memory":
        # The NULL control: put the own-index load back, keep the fusion. Within
        # one program the value at ``idx`` is the one this program just stored.
        before = fused_body
        for original, replacement in FUSION_EDIT:
            fused_body = fused_body.replace(replacement, original)
        if fused_body == before:
            raise RuntimeError("reload_own_D_from_memory rewrote nothing")
    elif mutation is not None:
        raise ValueError(f"unknown mutation {mutation!r}")

    source = (SUBJECT_HEADER
              + textwrap.indent(curl_half, "    ") + "\n\n"
              + textwrap.indent(fused_body, "    ") + "\n")
    record = {
        "curl_half_lines": len(curl_half.splitlines()),
        "constitutive_body_lines": len(fused_body.splitlines()),
        "rename_map": OFFDIAG_RENAME,
        "fusion_edit": [list(pair) for pair in FUSION_EDIT],
        "splice_is_verbatim_after_reversing_both_edits": verbatim,
        "planted": planted,
        "mutation": mutation,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    if not verbatim and mutation is None and not pointwise_variant:
        raise RuntimeError(
            "the splice is NOT verbatim: reversing the rename and the fusion "
            "edit did not reproduce the shipped off-diagonal body")
    return source, record


STANDALONE_HEADER = '''
"""The SEPARATE off-diagonal update_E, rebuilt from its own source with the
neighbour indices rewritten to ``idx``. The counterpart of the fused plant: the
two-launch route must run the SAME arithmetic as the planted fused kernel, or
the comparison measures the plant instead of the fusion."""
import triton
import triton.language as tl

from meep_gpu.triton_kernels.kernels import METALLIC, PERIODIC
from meep_gpu.triton_kernels.offdiag_update_e import (
    _masked_row_sum, _offdiag_term)


@triton.jit
'''


def pointwise_standalone_kernel() -> Any:
    """``offdiag_constitutive_step`` with every neighbour index made ``idx``.

    Compiled from the SHIPPED function's own source by the SAME rewrite the
    fused plant applies, so the planted pair really is the planted pair.
    """
    root = kernel_dir()
    body = textwrap.dedent(
        _function_source(root / "offdiag_update_e.py",
                         "offdiag_constitutive_step"))
    planted = re.sub(r"^(\s+)\w+ \* nyz \+ \w+ \* nz \+ \w+,$", r"\1idx,",
                     body, flags=re.M)
    rewritten = sum(1 for a, b in zip(body.splitlines(), planted.splitlines())
                    if a != b)
    if not rewritten:
        raise RuntimeError("the standalone pointwise plant rewrote NOTHING")
    source = STANDALONE_HEADER + planted + "\n"
    import tempfile  # noqa: PLC0415

    path = Path(tempfile.mkdtemp(prefix="offdiag_pointwise_")) / "standalone.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("offdiag_pointwise", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.offdiag_constitutive_step


_SUBJECT_CACHE: Dict[Any, Any] = {}


def subject_kernel(pointwise_variant: bool = False,
                   mutation: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    key = (pointwise_variant, mutation)
    if key in _SUBJECT_CACHE:
        return _SUBJECT_CACHE[key]
    source, record = build_subject_source(pointwise_variant, mutation)
    ast.parse(source)  # syntax is checked on the laptop too
    import tempfile  # noqa: PLC0415

    directory = tempfile.mkdtemp(prefix="fused_offdiag_subject_")
    path = Path(directory) / "subject.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"fused_offdiag_subject_{abs(hash(key)) % 10 ** 8}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the assembled subject")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record["path"] = str(path)
    value = (module.fused_curl_offdiag_constitutive_D, record)
    _SUBJECT_CACHE[key] = value
    return value


# ===========================================================================
# THE LAUNCHER for the subject (the shipped plans launch the reference routes)
# ===========================================================================

#: ROW_SLOTS order, restated from ``offdiag_update_e.ROW_SLOTS`` and asserted
#: against it — a private copy that drifted would mispair every coefficient.
ROW_SLOTS = (("Ex", "Ey"), ("Ex", "Ez"), ("Ey", "Ez"),
             ("Ey", "Ex"), ("Ez", "Ex"), ("Ez", "Ey"))


def run_subject(kernel: Any, arrays: Dict[str, Any],
                curl_flat: Dict[str, Any], constitutive_flat: Dict[str, Any],
                rows: Dict[str, Dict[str, Any]], codes: Sequence[int],
                zero_metal: Sequence[bool], wall_axes: Sequence[int],
                dtdx: float, block: int, num_warps: int = 1) -> Tuple[int, ...]:
    from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

    shape = tuple(int(n) for n in arrays["Dx"].shape)
    n_elem = shape[0] * shape[1] * shape[2]
    row_arrays = [(rows.get(row) or {}).get(partner) for row, partner in ROW_SLOTS]
    row_mask = tuple(int(v is not None) for v in row_arrays)
    if not any(row_mask):
        raise ValueError("no row slot survives; the subject would be the "
                         "diagonal pair")
    component_of_slot = tuple("ExEyEz".find(row) // 2 for row, _ in ROW_SLOTS)
    bound_rows = [row_arrays[i] if row_arrays[i] is not None
                  else arrays[D_NAMES[component_of_slot[i]]]
                  for i in range(len(ROW_SLOTS))]
    grid = ((n_elem + block - 1) // block,)
    p = CupyPointer  # Triton resolves a pointer through ``data_ptr()``.
    kernel[grid](
        *[p(arrays[n]) for n in D_NAMES],
        *[p(arrays[n]) for n in FU_NAMES],
        *[p(arrays[n]) for n in H_NAMES],
        *[p(curl_flat[f"{stem}_{axis}"]) for axis in "xyz"
          for stem in ("kms", "sinv")],
        *[p(arrays[n]) for n in E_NAMES],
        *[p(arrays[n]) for n in FW_NAMES],
        *[p(arrays["inv_eps_" + n]) for n in E_NAMES],
        *[p(a) for a in bound_rows],
        *[p(constitutive_flat[f"{stem}_{axis}"]) for axis in "xyz"
          for stem in ("kps", "kms")],
        shape[0], shape[1], shape[2], n_elem, dtdx,
        BACKWARD=1,
        BCX=codes[0], BCY=codes[1], BCZ=codes[2],
        ZM_X=int(zero_metal[0]), ZM_Y=int(zero_metal[1]), ZM_Z=int(zero_metal[2]),
        R01=row_mask[0], R02=row_mask[1], R11=row_mask[2],
        R12=row_mask[3], R21=row_mask[4], R22=row_mask[5],
        WM_X=int(wall_axes[0]), WM_Y=int(wall_axes[1]), WM_Z=int(wall_axes[2]),
        BLOCK=block,
        num_warps=num_warps,
        enable_fp_fusion=ENABLE_FP_FUSION,
    )
    return row_mask


def reference_zero_metal_D(state: Dict[str, Any], flags: Sequence[bool]) -> None:
    """``stepping.zero_metal_D``: each wall clears its two tangential D."""
    for axis, flag in enumerate(flags):
        if not flag:
            continue
        for component_axis, component in enumerate(D_NAMES):
            if component_axis != axis:
                state[component][probe._face(axis, 0)] = 0


def run_two_launch_offdiag(arrays: Dict[str, Any], curl_flat, constitutive_flat,
                           rows, codes, zero_metal, wall_axes, dtdx: float,
                           block: int, kernel: Any = None) -> None:
    """The certified sub-step products, in the driver's order, separately.

    ``kernel=`` IS LOAD-BEARING: the planted-pointwise leg routes its rewritten
    constitutive through it. A route that dropped it would launch the shipped
    kernel and compare a planted fused kernel against an unplanted reference,
    which is a different question and would report a false non-identity.
    """
    from meep_gpu.triton_kernels import offdiag_update_e as odmod  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import plan_from_arrays  # noqa: PLC0415

    plan_from_arrays("step_D", arrays, curl_flat, list(codes), dtdx,
                     block=block).run(guard=False)
    reference_zero_metal_D(arrays, zero_metal)
    odmod.plan_offdiagonal_constitutive_from_arrays(
        arrays, constitutive_flat, rows, list(codes), list(wall_axes),
        block=block, kernel=kernel).run(guard=False)


def run_two_launch_diagonal(arrays, curl_flat, constitutive_flat, codes,
                            zero_metal, dtdx: float, block: int) -> None:
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_constitutive_from_arrays, plan_from_arrays)

    plan_from_arrays("step_D", arrays, curl_flat, list(codes), dtdx,
                     block=block).run(guard=False)
    reference_zero_metal_D(arrays, zero_metal)
    plan_constitutive_from_arrays("E", arrays, constitutive_flat,
                                  block=block).run(guard=False)


def run_shipped_fused_diagonal(arrays, curl_flat, constitutive_flat, codes,
                               zero_metal, dtdx: float, block: int) -> None:
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_fused_pair_from_arrays)

    plan_fused_pair_from_arrays(
        "D", arrays, curl_flat, constitutive_flat, list(codes),
        list(zero_metal), dtdx, block=block, num_warps=1).run(guard=False)


# ===========================================================================
# FIXTURES
# ===========================================================================

INV_TENSOR = np.linalg.inv(np.array([[2.0, 0.35, 0.20],
                                     [0.35, 2.5, 0.15],
                                     [0.20, 0.15, 3.0]], dtype=np.float64))


def row_volumes(shape, rng, value_class: str,
                zero: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """All six slots live. ``zero`` makes every coefficient exactly +0.0."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, row in enumerate(E_NAMES):
        out[row] = {}
        for j, partner in enumerate(E_NAMES):
            if i == j:
                continue
            if zero:
                volume = np.zeros(shape, dtype=np.float32)
            else:
                volume = (np.float32(INV_TENSOR[i, j])
                          * rng.uniform(0.5, 1.5, size=shape)
                          .astype(np.float32)).astype(np.float32)
            out[row][partner] = np.ascontiguousarray(volume)
    return out


def host_state(shape, rng, value_class: str) -> Dict[str, np.ndarray]:
    """``uniform``, ``subnormal_band`` and ``signed_zero_lattice`` field draws."""
    state: Dict[str, np.ndarray] = {}
    for name in STATE_NAMES + H_NAMES:
        if value_class == "uniform":
            values = rng.uniform(-1.0, 1.0, size=shape)
        elif value_class == "subnormal_band":
            values = rng.uniform(1.0, 2.0, size=shape) * (2.0 ** -140)
            values *= np.where(rng.random(size=shape) < 0.5, -1.0, 1.0)
        elif value_class == "signed_zero_lattice":
            # A LATTICE, not an all-zero field. Measured on the first full
            # device run: an array of nothing but +-0.0 leaves E and f_w
            # unmoved in every route, so the leg is VACUOUS and the gate's own
            # non-vacuity floor refused all 90 of those cases. The class is
            # what it was meant to be — normal values with a +-0 lattice laid
            # over roughly a third of the cells, so the arithmetic moves AND
            # the signed zeros are carried through it.
            values = rng.uniform(-1.0, 1.0, size=shape)
            draw = rng.random(size=shape)
            values = np.where(draw < 0.18, 0.0,
                              np.where(draw < 0.36, -0.0, values))
        else:
            raise ValueError(value_class)
        state[name] = np.ascontiguousarray(values.astype(np.float32))
    for name in E_NAMES:
        state["inv_eps_" + name] = np.ascontiguousarray(
            rng.uniform(0.2, 0.9, size=shape).astype(np.float32))
    return state


def flat_ones(shape, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {f"{stem}_{axis}": np.ones(shape["xyz".index(axis)], dtype=np.float32)
            for axis in "xyz" for stem in keys}


def synthetic_flat(shape, rng, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {f"{stem}_{axis}": rng.uniform(0.5, 1.0, size=shape["xyz".index(axis)])
            .astype(np.float32) for axis in "xyz" for stem in keys}


def broadcast(flat: Dict[str, np.ndarray], shape) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in flat.items():
        axis = "xyz".index(key[-1])
        out[key] = value.reshape(probe.broadcast_shape(axis, shape[axis]))
    return out


# ===========================================================================
# THE LEGS
# ===========================================================================

def _to_device(host: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {name: cp.asarray(value) for name, value in host.items()}


def _moved(before: Dict[str, np.ndarray], after: Dict[str, Any],
           names: Sequence[str]) -> Dict[str, bool]:
    """Movement on the UINT32 WORDS, not on ``!=``.

    ``-0.0 != 0.0`` is False, so a float comparison calls a signed-zero flip
    "unmoved" — the one place the +-0 lattice most needs the floor to look. The
    whole gate compares bytes; the floor does too."""
    return {name: bool(np.any(probe.to_host(after[name]).view(np.uint32)
                              != before[name].view(np.uint32)))
            for name in names}


def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return probe.combine({name: probe.bit_compare(a[name], b[name])
                          for name in STATE_NAMES})


def one_leg(label: str, shape: Tuple[int, int, int], boundaries, block: int,
            value_class: str, *, subject: bool, curl_identity: bool = False,
            zero_rows: bool = False, repeat: int = 0,
            pointwise_variant: bool = False,
            source_mutation: Optional[str] = None,
            host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One (route, fixture, block) measurement, fused vs the separate products."""
    case: Dict[str, Any] = {
        "label": label, "shape": list(shape), "boundaries": list(boundaries),
        "block": block, "value_class": value_class, "subject": subject,
        "curl_identity": curl_identity, "zero_rows": zero_rows,
        "repeat": repeat, "pointwise_variant": pointwise_variant,
        "source_mutation": source_mutation, "host_mutation": host_mutation,
    }
    for axis in range(3):
        if boundaries[axis] == METALLIC and shape[axis] == 1:
            case["skipped"] = ("metallic invariant axis: Grid refuses it by "
                               "name (grid.py:842-877)")
            return case

    rng = case_rng(label, shape, boundaries, block, value_class, curl_identity,
                   zero_rows, repeat)
    state = host_state(shape, rng, value_class)
    if curl_identity and value_class == "signed_zero_lattice":
        # MEASURED, first device run: the recurrence normalises a -0.0 in D to
        # +0.0 (``(-0.0 + 0.0) - 0.0`` is ``+0.0``), so a signed-zero lattice on
        # D makes the curl-identity precondition unsatisfiable and the leg
        # reports nothing. The lattice is therefore carried on the arrays the
        # curl half does not touch — E, f_w and the row coefficients — and D
        # and fu_D take the identity-preserving draw. C1 and C2 have no such
        # precondition and carry the lattice on D as well.
        case["signed_zero_lattice_restricted_to"] = list(E_NAMES + FW_NAMES)
        plain = host_state(shape, case_rng(label, shape, "plainD", repeat),
                           "uniform")
        for name in D_NAMES:
            state[name] = plain[name]
    if curl_identity:
        # The curl half is made the identity on D and fu_D: uniform H so every
        # operand difference is an exact 0.0 under the periodic wrap, fu_D = +0,
        # kms = sinv = 1. MEASURED below, not assumed.
        for name in H_NAMES:
            state[name] = np.full(shape, np.float32(0.375), dtype=np.float32)
        for name in FU_NAMES:
            state[name] = np.zeros(shape, dtype=np.float32)
        curl_flat = flat_ones(shape, ("kms", "sinv"))
    else:
        curl_flat = synthetic_flat(shape, rng, ("kms", "sinv"))
    constitutive_flat = synthetic_flat(shape, rng, ("kps", "kms"))
    rows = row_volumes(shape, rng, value_class, zero=zero_rows)
    codes = tuple(CODE_OF[b] for b in boundaries)
    zero_metal = tuple(b == METALLIC for b in boundaries)
    wall_axes = tuple(1 if b == METALLIC else 0 for b in boundaries)
    dtdx = 0.5

    kernel_curl_flat = dict(curl_flat)
    kernel_constitutive_flat = dict(constitutive_flat)
    if host_mutation == "swap_kps_kms":
        kernel_constitutive_flat = {
            key.replace("kps", "TMP").replace("kms", "kps").replace("TMP", "kms"):
            value for key, value in constitutive_flat.items()}
    elif host_mutation == "metallic_as_periodic":
        codes_kernel = (0, 0, 0)
    if host_mutation != "metallic_as_periodic":
        codes_kernel = codes

    fused_arrays = _to_device(state)
    plain_arrays = _to_device(state)
    before = {name: state[name].copy() for name in STATE_NAMES}

    error = None
    try:
        if subject:
            kernel, record = subject_kernel(pointwise_variant, source_mutation)
            case["subject_record"] = {
                k: v for k, v in record.items() if k != "rename_map"}
            device_rows = {row: {p: cp.asarray(v) for p, v in partners.items()}
                           for row, partners in rows.items()}
            case["row_mask"] = list(run_subject(kernel, fused_arrays,
                        {k: cp.asarray(v) for k, v in kernel_curl_flat.items()},
                        {k: cp.asarray(v)
                         for k, v in kernel_constitutive_flat.items()},
                        device_rows, codes_kernel, zero_metal, wall_axes,
                        dtdx, block))
            run_two_launch_offdiag(
                plain_arrays,
                {k: cp.asarray(v) for k, v in curl_flat.items()},
                {k: cp.asarray(v) for k, v in constitutive_flat.items()},
                {row: {p: cp.asarray(v) for p, v in partners.items()}
                 for row, partners in rows.items()},
                codes, zero_metal, wall_axes, dtdx, block,
                kernel=(pointwise_standalone_kernel() if pointwise_variant
                        else None))
        else:
            run_shipped_fused_diagonal(
                fused_arrays,
                {k: cp.asarray(v) for k, v in kernel_curl_flat.items()},
                {k: cp.asarray(v)
                 for k, v in kernel_constitutive_flat.items()},
                codes_kernel, zero_metal, dtdx, block)
            run_two_launch_diagonal(
                plain_arrays,
                {k: cp.asarray(v) for k, v in curl_flat.items()},
                {k: cp.asarray(v) for k, v in constitutive_flat.items()},
                codes, zero_metal, dtdx, block)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001 - the artifact owns the failure
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        return case

    # --- NON-VACUITY FLOOR ---------------------------------------------------
    # A zero-initialised constitutive leaves every word +0.0 forever, so a
    # deliberately wrong reference would still report IDENTICAL. Every leg must
    # MOVE state, in the route that is being compared.
    moved = _moved(before, plain_arrays, E_NAMES + FW_NAMES)
    case["moved_E_and_fw"] = moved
    case["non_vacuous"] = all(moved.values())

    if curl_identity:
        # MEASURED precondition, not assumed: D and fu_D must come back bitwise
        # unchanged, or the control is not the control it claims to be.
        held = all(
            probe.to_host(plain_arrays[name]).tobytes() == before[name].tobytes()
            for name in D_NAMES + FU_NAMES)
        case["curl_is_identity_holds"] = held
        if not held:
            case["error"] = ("the curl-identity precondition FAILED: the curl "
                             "half moved D or fu_D, so this leg proves nothing")
            return case

    case["fused_vs_two_launch"] = _compare(fused_arrays, plain_arrays)
    case["bit_identical"] = bool(case["fused_vs_two_launch"]["bit_identical"])
    case["differing_floats"] = int(
        case["fused_vs_two_launch"].get("differing_floats", -1))
    if not case["bit_identical"]:
        # WHAT THE DISAGREEMENT IS MADE OF. A zero row coefficient does not
        # fully mask the racy read: ``near * (+0.0)`` carries the SIGN of
        # ``near``, so a stale operand of the opposite sign yields ``-0.0``
        # where the reference has ``+0.0`` and that survives ``diag + total``
        # wherever ``diag`` is itself a zero. Measured on the first full run:
        # exactly one word of 24576, in one C2 case of 90. Recording the
        # magnitude separates that from a real numerical divergence.
        worst = 0.0
        zeros_only = True
        for name in STATE_NAMES:
            a = probe.to_host(fused_arrays[name])
            b = probe.to_host(plain_arrays[name])
            mask = a.view(np.uint32) != b.view(np.uint32)
            if not mask.any():
                continue
            worst = max(worst, float(np.max(np.abs(
                a[mask].astype(np.float64) - b[mask].astype(np.float64)))))
            if not (np.all(a[mask] == 0.0) and np.all(b[mask] == 0.0)):
                zeros_only = False
        case["max_abs_difference"] = worst
        case["every_difference_is_a_signed_zero"] = zeros_only
    return case


def run_device_legs(payload: Dict[str, Any], out: Path,
                    repeats: int) -> Dict[str, Any]:
    shape_big = (16, 16, 16)
    shape_small = (8, 8, 8)          # 512 elements: one BLOCK=512 program
    periodic = (PERIODIC, PERIODIC, PERIODIC)
    walled = (METALLIC, PERIODIC, METALLIC)
    blocks = (64, 128, 256, 512, 1024)

    legs: Dict[str, List[Dict[str, Any]]] = {}
    payload["device_legs"] = legs

    def record(name: str, cases: List[Dict[str, Any]]) -> None:
        legs[name] = cases
        save(payload, out)

    # --- C1: the POINTWISE control, the shipped diagonal fused pair ----------
    cases = []
    classes = ("uniform", "subnormal_band", "signed_zero_lattice")
    total = len(blocks) * repeats * 2 * len(classes)
    for value_class in classes:
      for boundaries in (periodic, walled):
        for block in blocks:
            for repeat in range(repeats):
                case = one_leg("C1_pointwise_control", shape_big, boundaries,
                               block, value_class, subject=False, repeat=repeat)
                cases.append(case)
                log(f"  C1 {len(cases)}/{total} {value_class} "
                    f"bc={boundaries[0][:4]}/{boundaries[1][:4]}"
                    f"/{boundaries[2][:4]} block={block} r={repeat} "
                    f"identical={case.get('bit_identical')} "
                    f"diff={case.get('differing_floats')}")
                record("C1_pointwise_control", cases)

    # --- C2: the ZERO-COEFFICIENT control ------------------------------------
    cases = []
    for value_class in classes:
      for block in blocks:
        for repeat in range(repeats):
            case = one_leg("C2_zero_coefficient", shape_big, periodic, block,
                           value_class, subject=True, zero_rows=True,
                           repeat=repeat)
            cases.append(case)
            log(f"  C2 {len(cases)}/{len(blocks)*repeats*len(classes)} "
                f"{value_class} block={block} r={repeat} "
                f"identical={case.get('bit_identical')} "
                f"diff={case.get('differing_floats')}")
            record("C2_zero_coefficient", cases)

    # --- C3: the CURL-IS-IDENTITY control, all three value classes -----------
    cases = []
    for value_class in classes:
        for block in blocks:
            case = one_leg("C3_curl_identity", shape_big, periodic, block,
                           value_class, subject=True, curl_identity=True)
            cases.append(case)
            log(f"  C3 {len(cases)}/{len(classes)*len(blocks)} "
                f"class={value_class} block={block} "
                f"identical={case.get('bit_identical')} "
                f"precondition={case.get('curl_is_identity_holds')} "
                f"diff={case.get('differing_floats')}")
            record("C3_curl_identity", cases)

    # --- S1: THE SUBJECT ------------------------------------------------------
    cases = []
    for boundaries in (periodic, walled):
        for block in blocks:
            for repeat in range(repeats):
                case = one_leg("S1_subject", shape_big, boundaries, block,
                               "uniform", subject=True, repeat=repeat)
                cases.append(case)
                log(f"  S1 {len(cases)}/{len(blocks)*repeats*2} "
                    f"bc={boundaries[0][:4]}"
                    f"/{boundaries[1][:4]}/{boundaries[2][:4]} block={block} "
                    f"r={repeat} identical={case.get('bit_identical')} "
                    f"diff={case.get('differing_floats')}")
                record("S1_subject", cases)

    # --- S2: the SINGLE-PROGRAM observation ----------------------------------
    # 512 elements under BLOCK=512 is one program. Recorded as an observation
    # rather than a control: Triton gives no cross-LANE ordering guarantee
    # inside a CTA either, so an identical result here is consistent with the
    # claim but does not carry it. C3 is the control that does.
    cases = []
    for repeat in range(repeats):
        for block in (512, 64):
            case = one_leg("S2_single_program", shape_small, periodic, block,
                           "uniform", subject=True, repeat=repeat)
            case["programs"] = (512 + block - 1) // block
            cases.append(case)
            log(f"  S2 block={block} programs={case['programs']} r={repeat} "
                f"identical={case.get('bit_identical')} "
                f"diff={case.get('differing_floats')}")
            record("S2_single_program", cases)

    # --- M: mutations, source and host ---------------------------------------
    # EVERY MUTATION IS SCORED ON A LEG THAT ENTERS THE GUARD IT REWRITES.
    # A mutation scored on a case that never reaches the rewritten line reports
    # UNCAUGHT while measuring nothing (the dead-branch trap this project has
    # already paid for), so the guard is named beside each one and the case is
    # chosen to enter it. In particular NOTHING is scored on S1: S1's own
    # verdict is non-identity, so every mutation would be trivially "caught"
    # there. All source mutations are scored on legs that claim IDENTITY.
    cases = []
    mutations = (
        # SOURCE. Guard: none — the curl expression is unconditional. Scored on
        # C2, whose H is random (so the curl is nonzero) and whose verdict is
        # IDENTICAL, so a change is visible. A curl-identity fixture would make
        # this mutation invisible: both forms evaluate to an exact 0.0.
        ("flatten_curl_parens", "unconditional (kernels.py curl expression)",
         dict(label="M_flatten_curl", subject=True, zero_rows=True,
              value_class="uniform", boundaries=periodic,
              source_mutation="flatten_curl_parens"), "expect_CAUGHT"),
        # SOURCE. Guard: `if R01:` / `if R02:` on component 0 — C3 has all six
        # slots live, so both arms are entered (asserted through row_mask).
        ("mispair_row_coefficients", "if R01: / if R02: on component 0",
         dict(label="M_mispair_rows", subject=True, curl_identity=True,
              value_class="uniform", boundaries=periodic,
              source_mutation="mispair_row_coefficients"), "expect_CAUGHT"),
        # SOURCE NULL. Guard: none. Within one program the value at ``idx`` is
        # the one this program stored, so putting the load back must be a null.
        ("reload_own_D_from_memory", "unconditional (the fusion substitution)",
         dict(label="M_reload_own_D", subject=True, curl_identity=True,
              value_class="uniform", boundaries=periodic,
              source_mutation="reload_own_D_from_memory"), "expect_NULL"),
        # HOST. Guard: none — the constitutive tail runs on every component.
        ("swap_kps_kms", "unconditional (the constitutive tail)",
         dict(label="M_swap_kps_kms", subject=True, curl_identity=True,
              value_class="uniform", boundaries=periodic,
              host_mutation="swap_kps_kms"), "expect_CAUGHT"),
        # HOST, on the SHIPPED control. Guard: `if BC* == METALLIC` — scored on
        # the WALLED fixture, the only one that enters it.
        ("metallic_as_periodic", "if BC* == METALLIC (needs a walled case)",
         dict(label="M_metallic_as_periodic", subject=False,
              value_class="uniform", boundaries=walled,
              host_mutation="metallic_as_periodic"), "expect_CAUGHT"),
    )
    for name, guard, kwargs, expectation in mutations:
        boundaries = kwargs.pop("boundaries")
        label = kwargs.pop("label")
        subject = kwargs.pop("subject")
        case = one_leg(label, shape_big, boundaries, 256, subject=subject,
                       **kwargs)
        case["mutation"] = name
        case["enclosing_guard"] = guard
        case["expectation"] = expectation
        if case.get("launch_error") or case.get("error"):
            case["outcome"] = "NOT SCORED — the leg did not run"
        else:
            caught = not case.get("bit_identical", True)
            case["outcome"] = "CAUGHT" if caught else "NULL CONFIRMED"
        cases.append(case)
        log(f"  M {name}: {case['outcome']} (expected {expectation}) "
            f"diff={case.get('differing_floats')}")
        record("M_mutations", cases)

    # --- P: THE PLANTED DEFECT that must FLIP the release verdict ------------
    cases = []
    for block in (64, 256):
        for repeat in range(repeats):
            case = one_leg("P_pointwise_plant", shape_big, periodic, block,
                           "uniform", subject=True, pointwise_variant=True,
                           repeat=repeat)
            cases.append(case)
            log(f"  P pointwise-plant block={block} r={repeat} "
                f"identical={case.get('bit_identical')} "
                f"diff={case.get('differing_floats')}")
            record("P_pointwise_plant", cases)
    return legs


# ===========================================================================
# THE VERDICT
# ===========================================================================

def _all(cases: Sequence[Dict[str, Any]], key: str, want: Any) -> bool:
    ran = [c for c in cases if not c.get("skipped") and not c.get("error")
           and c.get("launch_error") is None and "bit_identical" in c]
    return bool(ran) and all(c.get(key) == want for c in ran)


def verdict(legs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    c1 = legs.get("C1_pointwise_control", [])
    c2 = legs.get("C2_zero_coefficient", [])
    c3 = legs.get("C3_curl_identity", [])
    s1 = legs.get("S1_subject", [])
    plant = legs.get("P_pointwise_plant", [])
    muts = legs.get("M_mutations", [])
    scoreboard = {c["mutation"]: {"outcome": c.get("outcome"),
                                  "expected": c.get("expectation"),
                                  "guard": c.get("enclosing_guard"),
                                  "row_mask": c.get("row_mask")}
                  for c in muts if "mutation" in c}
    mutations_as_expected = all(
        (row["outcome"] == "CAUGHT") == (row["expected"] == "expect_CAUGHT")
        for row in scoreboard.values())

    s1_ran = [c for c in s1 if "bit_identical" in c]
    s1_counts = sorted({c["differing_floats"] for c in s1_ran})
    per_key: Dict[str, List[int]] = {}
    for c in s1_ran:
        key = f"{c['boundaries'][0][:4]}|block={c['block']}"
        per_key.setdefault(key, []).append(c["differing_floats"])
    # A RACE is a POSSIBILITY of disagreement, not a guarantee: whether a stale
    # read is returned depends on the schedule, and at BLOCK=1024 on a 4096-cell
    # grid there are four programs and the window can close entirely. So the
    # measured clause is SCHEDULE DEPENDENCE — the same fixture and the same
    # kernel giving different answers — not universal disagreement. A
    # transcription error would give the same wrong answer every time.
    schedule_dependent = (
        any(len(set(v)) > 1 for v in per_key.values())          # across repeats
        or len({tuple(sorted(set(v))) for v in per_key.values()}) > 1)  # across blocks
    disagrees_somewhere = any(not c["bit_identical"] for c in s1_ran)

    clauses = {
        "C1 the pointwise control is bit-identical everywhere":
            _all(c1, "bit_identical", True),
        # C2 IS NOT AN EXACT CONTROL AND THE GATE SAYS SO. A +0.0 coefficient
        # zeroes the MAGNITUDE of the racy operand but not its SIGN, and a
        # -0.0 total survives `diag + total` wherever diag is itself a zero —
        # which the subnormal band produces by underflow. So the clause is not
        # "identical" but "identical except for signed zeros", which is
        # measured word by word rather than argued.
        "C2 the zero-coefficient subject differs at most in signed zeros":
            all(c.get("bit_identical")
                or c.get("every_difference_is_a_signed_zero", False)
                for c in c2 if "bit_identical" in c),
        "C3 the curl-identity subject is bit-identical":
            _all(c3, "bit_identical", True),
        "C3 the curl-identity precondition held":
            _all(c3, "curl_is_identity_holds", True),
        # A CLAUSE THAT PASSES BECAUSE CASES WERE EXCLUDED IS THE VACUITY
        # HAZARD IN ANOTHER COAT. Under the FLUSH policy ``D * 1.0`` flushes a
        # subnormal to zero, so the curl half is not the identity and the whole
        # subnormal_band cut of C3 refuses itself by name. That is the right
        # answer, and it means C3 must be shown to have run on more than one
        # value class before its verdict counts for anything.
        "C3 ran on at least two value classes":
            len({c["value_class"] for c in c3 if "bit_identical" in c}) >= 2,
        "S1 the subject DISAGREES with the two-launch route somewhere":
            disagrees_somewhere,
        "S1 the disagreement is SCHEDULE-DEPENDENT (a race, not a defect)":
            schedule_dependent,
        "every leg moved state (non-vacuity)":
            all(c.get("non_vacuous", False)
                for c in c1 + c2 + c3 + s1 if "bit_identical" in c),
        "the splice is verbatim after reversing both declared edits":
            all(c.get("subject_record", {}).get(
                "splice_is_verbatim_after_reversing_both_edits", False)
                for c in s1 if "subject_record" in c),
        "every mutation scored as expected (CAUGHT / NULL CONFIRMED)":
            bool(scoreboard) and mutations_as_expected,
    }
    released = all(clauses.values())
    plant_ran = [c for c in plant if "bit_identical" in c]
    plant_flips = bool(plant_ran) and all(c["bit_identical"] for c in plant_ran)
    s1_worst = max([c.get("max_abs_difference", 0.0) for c in s1_ran] or [0.0])
    return {
        "clauses": clauses,
        "mutation_scoreboard": scoreboard,
        "s1_max_abs_difference": s1_worst,
        "c3_value_classes_that_ran":
            sorted({c["value_class"] for c in c3 if "bit_identical" in c}),
        "c3_cases_refused_by_their_own_precondition": [
            {"value_class": c["value_class"], "block": c["block"],
             "why": c.get("error")}
            for c in c3 if "bit_identical" not in c and c.get("error")],
        "c2_cases_differing": [
            {"value_class": c["value_class"], "block": c["block"],
             "repeat": c["repeat"], "differing_floats": c["differing_floats"],
             "max_abs_difference": c.get("max_abs_difference"),
             "every_difference_is_a_signed_zero":
                 c.get("every_difference_is_a_signed_zero")}
            for c in c2 if c.get("bit_identical") is False],
        "REFUSAL ESTABLISHED": released,
        "s1_differing_float_counts": s1_counts,
        "s1_counts_per_configuration": {k: sorted(v) for k, v in per_key.items()},
        "planted_pointwise_variant_becomes_identical": plant_flips,
        "what_the_plant_shows": (
            "rewriting every neighbour index in the coupling to idx removes "
            "the cross-program read and the disagreement with it; the refusal "
            "is therefore measuring the stencil and not the weld"),
        "seam_instance_delta": 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"))
    args = parser.parse_args(argv)
    out = Path(args.out)

    payload: Dict[str, Any] = {
        "gate": "gate_triton_fused_offdiag_electric",
        "claim": ("the D->E off-diagonal fused pair CANNOT be built: the "
                  "constitutive arm is a stencil over the curl arm's in-place "
                  "output"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "seed": SEED,
        "device_mode": not args.no_device,
        "requested_subnormal_policy": args.subnormal_policy,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    save(payload, out)

    log("=== LEG 0: the structural census (parsed, no device) ===")
    payload["stencil_census"] = stencil_census()
    for row in payload["stencil_census"]["constitutive_arms"]:
        log(f"  {row['module']}.{row['kernel']}: {row['status']}"
            + (f"  offsets={row['d_volume_load_offsets']}"
               if "d_volume_load_offsets" in row else ""))
    for row in payload["stencil_census"]["curl_arms"]:
        log(f"  {row['module']}.{row['kernel']}: {row['status']}")
    save(payload, out)

    log("=== the assembled subject ===")
    _source, record = build_subject_source()
    payload["subject_assembly"] = record
    log(f"  curl half {record['curl_half_lines']} lines, constitutive body "
        f"{record['constitutive_body_lines']} lines, verbatim="
        f"{record['splice_is_verbatim_after_reversing_both_edits']}")
    # ROW_SLOTS may not drift from the shipped table (name-drift trap).
    try:
        from meep_gpu.triton_kernels.offdiag_update_e import (  # noqa: PLC0415
            ROW_SLOTS as SHIPPED_SLOTS)
        payload["row_slots_match_shipped"] = (
            tuple(ROW_SLOTS) == tuple(SHIPPED_SLOTS))
    except Exception as exc:  # noqa: BLE001
        payload["row_slots_match_shipped"] = f"unreadable: {exc!r}"
    log(f"  ROW_SLOTS match shipped: {payload['row_slots_match_shipped']}")
    save(payload, out)

    if args.no_device:
        payload["verdict"] = {
            "REFUSAL ESTABLISHED": False,
            "why": "--no-device: the structural legs ran, nothing executed"}
        save(payload, out)
        log("no-device run complete")
        return 0

    if cp is None or not _TRITON_AVAILABLE:
        payload["error"] = "cupy and triton are both required for the device legs"
        save(payload, out)
        return 2

    from meep_gpu import subnormal_policy  # noqa: PLC0415

    if args.subnormal_policy == "flush":
        # ``mp.set_zero_subnormals`` is the only exposure of this process's
        # FTZ/DAZ bits the package may use, so the FLUSH cut needs MEEP in the
        # process. The shared helper owns the import (and its MPI side effect);
        # this gate does not reach for it under ``keep``.
        payload["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
        save(payload, out)
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    payload["cupy"] = cp.__version__
    payload["triton"] = triton.__version__
    payload["device"] = probe.device_info()
    # SHARED BOX. Contention cannot change a bit-identity verdict (nothing here
    # is timed) but it CAN change how often a race window is hit, so the
    # occupancy at start is recorded rather than assumed.
    try:
        import subprocess  # noqa: PLC0415
        payload["nvidia_smi_compute_apps_at_start"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        payload["nvidia_smi_compute_apps_at_start"] = f"unreadable: {exc!r}"
    log(f"subnormal policy: {payload['subnormal_policy'].get('policy')!r}")
    save(payload, out)

    log("=== device legs ===")
    legs = run_device_legs(payload, out, args.repeats)
    payload["verdict"] = verdict(legs)
    save(payload, out)
    log("=== VERDICT ===")
    for clause, value in payload["verdict"]["clauses"].items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"  REFUSAL ESTABLISHED: {payload['verdict']['REFUSAL ESTABLISHED']}")
    log(f"  planted pointwise variant becomes identical: "
        f"{payload['verdict']['planted_pointwise_variant_becomes_identical']}")
    return 0 if payload["verdict"]["REFUSAL ESTABLISHED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

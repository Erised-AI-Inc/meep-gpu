#!/usr/bin/env python
"""HOST probe for the SCRATCH-OUTPUT off-diagonal D/E welds — no device needed.

WHAT THIS ESTABLISHES, AND WHAT IT DELIBERATELY DOES NOT.

It establishes two things, both on this backend's own arms and both as uint32
word comparisons:

  TRANSCRIPTION — every arithmetic block in each weld kernel is a certified
  body. The curl half is ``kernels.pml_curl_step``'s own text between its decode
  and its stores, compared BYTE FOR BYTE against the weld's ``_step_cell``; the
  wall block is ``kernels.fused_curl_constitutive_D``'s own D-side ``ZM_*``
  block; the term helper is ``offdiag_update_e._offdiag_term`` under exactly
  four declared needles; the row-sum helper is that module's verbatim; and the
  three per-component tails and the index prologue are contiguous runs of
  ``offdiag_constitutive_step``. It also asserts that NO displacement load
  survives below the seam — in the fused signature ``g0``/``g1``/``g2`` are the
  MAGNETIC field, so one missed read is a smooth, plausible, entirely wrong
  answer rather than a compile error.

  THE SEAM ARITHMETIC — the scratch-output choreography (step out of place,
  resolve the in-seam passes per cell by closed form, run the constitutive half
  against the resolved volume, rotate) is byte-identical to the driver's real
  pass order over two complete steps, on a fixture taxonomy, with every null
  diverging. The curl and constitutive arithmetic in this leg is the ARRAY
  PATH's own — that is the point: what is under test here is the CHOREOGRAPHY,
  not the kernel.

It establishes NOTHING about the kernel's execution. Triton cannot run on the
laptop, and the claim that matters — that the weld's one launch is bit-identical
to the two-launch reference AT EVERY BLOCK SIZE, which is precisely where the
2026-08-20 in-place subject failed — is a device claim and belongs to
``gate_triton_offdiag_stencil_welds.py``. This probe is what makes that gate's
failure modes distinguishable: a device disagreement with these legs green is a
kernel defect, and with these legs red is a design defect.

Progress reporting: one flushed line per leg, and the JSON is written and fsynced at the end
of every leg group rather than once at exit.

    PYTHONPATH=. python -u \
      parity/meep_gpu/probe_triton_offdiag_scratch_weld.py --out <dir>/probe.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.triton_kernels.offdiag_scratch_weld import (  # noqa: E402
    certified_tail, lifted_tail,
)

#: Every file whose bytes this probe's verdict depends on. Hashed into the
#: artifact so a later reader can tell whether the tree still ships them.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/offdiag_fused_electric_pair.py",
    "meep_gpu/triton_kernels/offdiag_update_e.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/stepping.py",
    "parity/meep_gpu/probe_triton_offdiag_scratch_weld.py",
)

#: The families this probe knows how to check, and the anchors each one's lift
#: is cut at. Anchors are PER FAMILY because the certified curls do not decode
#: or store alike; cutting at the wrong one silently compares a shorter body.
FAMILIES: Dict[str, Dict[str, Any]] = {
    "offdiag_fused_electric_pair": {
        "module": "meep_gpu/triton_kernels/offdiag_fused_electric_pair.py",
        "kernel": "offdiag_fused_curl_constitutive_D",
        "curl_module": "meep_gpu/triton_kernels/kernels.py",
        "curl_kernel": "pml_curl_step",
        "curl_decode_end": "    i = plane // ny\n",
        "curl_store_start": "    tl.store(u0 + idx, n0, mask=live)",
        "lift_decode_end": "        idx = i * nyz + j * nz + k\n",
        "lift_return_start": "        return v0, v1, v2, n0, n1, n2",
        "constitutive_module": "meep_gpu/triton_kernels/offdiag_update_e.py",
        "constitutive_kernel": "offdiag_constitutive_step",
        "term": "_offdiag_term",
        "welded_term": "_welded_offdiag_term",
        "tap_function": "_tap",
        "tap_component_argument": 0,
        "taps": 15,
        "tap_cells": 12,
        # The wall block is lifted whole from the certified fused kernel and
        # lives in its own function; the folded family has no such block, because
        # its closed form interleaves two passes per component.
        "seam_function": "_seam_resolve",
        "seam_block": ("if ZM_X:", "v1 = tl.where(at_z, 0.0, v1)"),
        "seam_block_source": ("meep_gpu/triton_kernels/kernels.py",
                              "fused_curl_constitutive_D"),
        "ghost_block": ("di, dj, dk = i - 1", "at_x, at_y, at_z = i == 0"),
        "carried_passes": ("zero_metal_D",),
    },
    "folded_offdiag_fused_electric_pair": {
        "module": "meep_gpu/triton_kernels/folded_offdiag_fused_electric_pair.py",
        "kernel": "folded_offdiag_fused_curl_constitutive_D",
        "curl_module": "meep_gpu/triton_kernels/symmetry.py",
        "curl_kernel": "pml_curl_step_folded",
        "curl_decode_end": "    i = plane // ny\n",
        "curl_store_start": "    tl.store(u0 + idx, n0, mask=live)",
        "lift_decode_end": "        idx = i * nyz + j * nz + k\n",
        "lift_return_start": "        return v0, v1, v2, n0, n1, n2",
        "constitutive_module":
            "meep_gpu/triton_kernels/folded_offdiag_update_e.py",
        "constitutive_kernel": "folded_offdiag_constitutive_step",
        "term": "_folded_offdiag_term",
        "welded_term": "_welded_folded_offdiag_term",
        "tap_function": "_d_final",
        "tap_component_argument": 0,
        "taps": 15,
        "tap_cells": 12,
        "seam_function": None,
        "seam_block": None,
        "seam_block_source": None,
        "ghost_block": ("di, dj, dk = i - 1", "uvz = live & (uk < nz)"),
        "carried_passes": ("fill_symmetry_bc_D", "zero_metal_D"),
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    return {name: sha256(os.path.join(API_ROOT, name))
            for name in SOURCES if os.path.exists(os.path.join(API_ROOT, name))}


# ---------------------------------------------------------------------------
# Source helpers — the same four the certified Triton probes use
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


def _span(lines: Sequence[str], first: str, last: str) -> List[str]:
    stripped = _dedent(lines)
    try:
        start = next(index for index, line in enumerate(stripped)
                     if line.startswith(first))
        end = next(index for index, line in enumerate(stripped)
                   if index > start and line.startswith(last))
    except StopIteration:
        raise AssertionError(f"no span {first!r}..{last!r}")
    return list(lines[start:end + 1])


# ---------------------------------------------------------------------------
# LEG GROUP 1 — transcription
# ---------------------------------------------------------------------------

def transcription_leg(family: str) -> Dict[str, Any]:
    """Every arithmetic block in the weld is a certified body.

    Whole BLOCKS rather than line sets, which is the difference that catches a
    reordering: a line-set comparison passes on a body whose statements were
    permuted, and permuting the split-field recurrence or the row accumulation is
    a different float32 number in every cell.
    """
    spec = FAMILIES[family]
    findings: List[str] = []
    measures: Dict[str, Any] = {}

    curl_source = _function_source(spec["curl_module"], spec["curl_kernel"])
    lift_source = _function_source(spec["module"], "_step_cell")
    certified = textwrap.dedent(certified_tail(
        curl_source, decode_end=spec["curl_decode_end"],
        store_start=spec["curl_store_start"]))
    lifted = textwrap.dedent(lifted_tail(
        lift_source, decode_end=spec["lift_decode_end"],
        return_start=spec["lift_return_start"]))
    measures["curl_lift_chars"] = len(certified)
    measures["curl_lift_statements"] = len(_statements(certified))
    if certified != lifted:
        findings.append(
            f"the weld's _step_cell is NOT the certified {spec['curl_kernel']} "
            f"body between its anchors: {len(certified)} chars against "
            f"{len(lifted)}")
    if measures["curl_lift_statements"] < 40:
        findings.append(
            f"the lifted curl body is only {measures['curl_lift_statements']} "
            f"statements; the anchors have crossed and this leg is comparing "
            f"almost nothing")

    product = _statements(_function_source(spec["module"], spec["kernel"]))
    constitutive = _statements(
        _function_source(spec["constitutive_module"], spec["constitutive_kernel"]))
    wall = _statements(_function_source(
        "meep_gpu/triton_kernels/kernels.py", "fused_curl_constitutive_D"))

    # 1. The index prologue and the ghost-index block, from the certified
    #    constitutive body, contiguous and in order.
    prologue = _span(constitutive, "idx = tl.program_id(0)", "i = plane // ny")
    if not _contains_block(product, prologue):
        findings.append("the index prologue is not a contiguous run of the "
                        "certified off-diagonal constitutive body")
    ghosts = _span(constitutive, *spec["ghost_block"])
    measures["ghost_block_statements"] = len(ghosts)
    if not _contains_block(product, ghosts):
        findings.append("the neighbour-index block is not a contiguous run of "
                        "the certified constitutive body")

    # 2. The wall block, where the family lifts one. The unfolded family's
    #    closed form IS `fused_curl_constitutive_D`'s own D-side ZM block; the
    #    folded family's interleaves the fill and the clear per component, is NEW
    #    text, and is checked below by its constants instead.
    if spec["seam_function"] is not None:
        wall = _statements(_function_source(*spec["seam_block_source"]))
        wall_block = _span(wall, *spec["seam_block"])
        seam = _statements(_function_source(spec["module"],
                                            spec["seam_function"]))
        measures["wall_block_statements"] = len(wall_block)
        if not _contains_block(seam, wall_block):
            findings.append(
                f"the closed form's wall block is not a contiguous run of "
                f"{spec['seam_block_source'][1]}'s own D-side ZM block")
    else:
        # The NEW text this family can be blamed for. Its two constants are the
        # certified ones: the mirror source row and the fill's parity spelling.
        closed = _statements(_function_source(spec["module"], "_d_final"))
        fill = _statements(_function_source(
            "meep_gpu/triton_kernels/symmetry.py", "mirror_ghost_fill"))
        measures["closed_form_statements"] = len(closed)
        if not any("MIRROR_ROW" in line for line in closed):
            findings.append("the closed form does not redirect to MIRROR_ROW")
        if not any("PHASE * tl.load" in line for line in fill):
            findings.append(
                "symmetry.mirror_ghost_fill no longer spells its parity "
                "`PHASE * value`, so the closed form's spelling is no longer "
                "the certified one")
        if not any("PH_X * value" in line for line in closed):
            findings.append(
                "the closed form does not carry the fill's `PH * value` parity")

    # 3. The coefficient loads and the three per-component tails.
    coefficients = _span(constitutive, "kp_0 = tl.load(kp0 + i",
                         "km_2 = tl.load(km2 + k")
    if not _contains_block(product, coefficients):
        findings.append("the half-integer coefficient block is not a contiguous "
                        "run of the certified constitutive body")
    for index in range(3):
        tail = _span(constitutive, f"tl.store(w{index} + idx",
                     f"tl.store(f{index} + idx")
        measures[f"tail_{index}_statements"] = len(tail)
        if not _contains_block(product, tail):
            findings.append(
                f"component {index}'s constitutive tail is not a contiguous run "
                f"of the certified body")

    # 4. The term helper: FOUR needles and no other edit. The needles are the
    #    four displacement loads; every other token — including the folded
    #    family's runtime ghost weight and its MG arm — must survive untouched.
    term = _function_source(spec["constitutive_module"], spec["term"])
    welded = _function_source(spec["module"], spec["welded_term"])
    needles = (("tl.load(g + o_c, mask=v_c, other=0.0)", "near_c"),
               ("tl.load(g + o_d, mask=v_d, other=0.0)", "near_d"),
               ("tl.load(g + o_u, mask=v_u, other=0.0)", "far_u"),
               ("tl.load(g + o_ud, mask=v_ud, other=0.0)", "far_ud"))
    body = _statements(term.split('"""')[2])
    reversed_body = _statements(welded.split('"""')[2])
    applied = 0
    for old_text, new_text in needles:
        pattern = re.compile(r"\b" + re.escape(new_text) + r"\b")
        hits = sum(len(pattern.findall(line)) for line in reversed_body)
        if hits:
            applied += 1
        reversed_body = [pattern.sub(old_text.replace("\\", "\\\\"), line)
                         for line in reversed_body]
    measures["term_needles_applied"] = applied
    if applied != len(needles):
        findings.append(
            f"only {applied} of {len(needles)} displacement needles reverse in "
            f"{spec['welded_term']}; a load the weld cannot serve may be left "
            f"standing")
    if _dedent(reversed_body) != _dedent(body):
        findings.append(
            f"{spec['welded_term']} is not {spec['term']} under the four "
            f"declared needles alone")

    # 5. The row-sum helper is the certified one, verbatim.
    if _dedent(_statements(_function_source(spec["module"], "_masked_row_sum"))) \
            != _dedent(_statements(_function_source(
                spec["constitutive_module"], "_masked_row_sum"))):
        findings.append("_masked_row_sum was edited on the way in")

    # 6. NO DISPLACEMENT LOAD SURVIVES BELOW THE SEAM. In the fused signature the
    #    three `g` pointers are the MAGNETIC field; a missed read is a smooth
    #    wrong answer, not a compile error. The step helpers legitimately take
    #    them AS ARGUMENTS, which is a pass-through and not a load.
    kernel_text = _function_source(spec["module"], spec["kernel"])
    stray = [line.strip() for line in kernel_text.splitlines()
             if any(f"tl.load(g{index}" in line for index in range(3))]
    measures["stray_displacement_loads"] = len(stray)
    if stray:
        findings.append(
            f"the welded kernel still loads a displacement through g*: {stray}")

    return {"leg": "transcription", "family": family, "passed": not findings,
            "findings": findings, "measures": measures}


# ---------------------------------------------------------------------------
# LEG GROUP 1b — the tap binding: every redirected load reads the cell the
#                certified load addressed, under the mask it was served with
# ---------------------------------------------------------------------------
#
# THIS IS WHERE A TRANSCRIPTION ERROR WOULD LIVE. The substitution replaces four
# addressed loads per term with four values; if one value is re-derived at the
# wrong cell, or under the wrong mask, or for the wrong partner component, the
# kernel compiles, runs, and returns a smooth wrong field. Nothing else in this
# probe would see it, and on the device it would look exactly like the race the
# 2026-08-20 round measured — which is the confusion this leg exists to prevent.

def _calls(tree: ast.AST, name: str) -> List[ast.Call]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


def _index_expression(coordinates: Sequence[ast.AST]) -> str:
    """``(cx, cy, cz)`` -> the flat index the certified body would have written."""
    cx, cy, cz = (ast.unparse(node) for node in coordinates)
    return ast.unparse(ast.parse(f"{cx} * nyz + {cy} * nz + {cz}",
                                 mode="eval").body)


def _parameter_names(path: str, name: str) -> List[str]:
    """The positional parameter names of one ``@triton.jit`` function.

    READ, never spelled: the four certified term helpers and the four welded ones
    do not agree on arity (the folded pair carries a runtime ghost weight and a
    constexpr arm the plain one has no parameter for), so the correspondence
    below is by NAME and the names come from the definitions themselves.
    """
    tree = ast.parse(textwrap.dedent(_function_source(path, name)))
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef) and child.name == name)
    return [argument.arg for argument in node.args.args]


def tap_binding_leg(family: str) -> Dict[str, Any]:
    """Each redirected load is the certified load's cell, mask and component."""
    spec = FAMILIES[family]
    findings: List[str] = []
    measures: Dict[str, Any] = {}

    weld_tree = ast.parse(textwrap.dedent(
        _function_source(spec["module"], spec["kernel"])))
    certified_tree = ast.parse(textwrap.dedent(_function_source(
        spec["constitutive_module"], spec["constitutive_kernel"])))
    certified_names = _parameter_names(spec["constitutive_module"], spec["term"])
    welded_names = _parameter_names(spec["module"], spec["welded_term"])
    tap_call = spec["tap_function"]

    # 1. The tap table: every `tap_* = <tap_call>(COMP, cx, cy, cz, mask, ...)`.
    taps: Dict[str, Tuple[int, str, str]] = {}
    for node in ast.walk(weld_tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Name) and call.func.id == tap_call):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue          # the own-cell call binds a tuple; handled below
        if not target.id.startswith("tap_"):
            continue          # dfin* are the own-cell registers, not foreign taps
        component = call.args[spec["tap_component_argument"]]
        if not (isinstance(component, ast.Constant)
                and component.value in (0, 1, 2)):
            findings.append(f"{target.id} passes a non-constant component")
            continue
        first = spec["tap_component_argument"] + 1
        taps[target.id] = (int(component.value),
                           _index_expression(call.args[first:first + 3]),
                           ast.unparse(call.args[first + 3]))
    measures["taps_declared"] = len(taps)
    measures["distinct_tap_cells"] = len({cell for _c, cell, _m in taps.values()})
    # FIFTEEN (component, cell) taps over TWELVE distinct cells: each partner is
    # read one step DOWN its own axis (3), one step UP the row component's axis
    # (6 — the two rows that share an up-face read different components there),
    # and at the corner of the two (6). The two counts are asserted separately
    # because they fail differently: a wrong tap count is a missing or duplicated
    # redirect, a wrong cell count is a redirect at the wrong neighbour.
    if len(taps) != spec["taps"]:
        findings.append(
            f"{len(taps)} foreign taps are declared; this family's certified "
            f"body makes {spec['taps']} (component, cell) neighbour reads")
    if measures["distinct_tap_cells"] != spec["tap_cells"]:
        findings.append(
            f"the taps address {measures['distinct_tap_cells']} distinct cells; "
            f"the certified body addresses {spec['tap_cells']}")

    # 2. The own-cell registers, which stand in for the `idx` loads.
    own_registers: List[str] = []
    for node in ast.walk(weld_tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Tuple)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("_seam_resolve", tap_call)):
            own_registers = [ast.unparse(element) for element in target.elts]
        elif (isinstance(target, ast.Name) and target.id.startswith("dfin")
              and isinstance(node.value, ast.Call)):
            own_registers.append(target.id)
    if sorted(set(own_registers)) != ["dfin0", "dfin1", "dfin2"]:
        findings.append(
            f"the own-cell displacement registers are {sorted(set(own_registers))}, "
            f"not the three the seam resolves")

    # 3. Pair the call sites, in source order, and check every argument.
    certified_calls = _calls(certified_tree, spec["term"])
    welded_calls = _calls(weld_tree, spec["welded_term"])
    measures["term_call_sites"] = len(welded_calls)
    if len(certified_calls) != len(welded_calls):
        findings.append(
            f"the certified body has {len(certified_calls)} coupling call sites "
            f"and the weld has {len(welded_calls)}; the row-mask arms do not "
            f"correspond")
        return {"leg": "tap_binding", "family": family, "passed": False,
                "findings": findings, "measures": measures}

    #: What must be carried through the substitution UNCHANGED — the coefficient
    #: slot, the two indices it is read at, their masks, and (where the family
    #: has them) the runtime ghost weight and its constexpr arm.
    carried = [name for name in welded_names if name in certified_names]
    measures["carried_arguments"] = carried
    for position, (before, after) in enumerate(zip(certified_calls, welded_calls)):
        certified_args = {name: ast.unparse(value) for name, value
                          in zip(certified_names, before.args)}
        welded_args = {name: ast.unparse(value) for name, value
                       in zip(welded_names, after.args)}
        partner_name = certified_args.get("g", "")
        if not (partner_name.startswith("g") and partner_name[1:].isdigit()):
            findings.append(f"call {position}: certified partner {partner_name!r} "
                            f"is not one of the three displacement pointers")
            continue
        partner = int(partner_name[1:])

        # 3a. The COEFFICIENT slot and the two indices it is read at are carried
        #     through untouched — the coefficient multiply sits BETWEEN the
        #     shifts and mispairing it is the certified gate's own m3 mutation.
        for key in carried:
            if certified_args[key] != welded_args[key]:
                findings.append(
                    f"call {position}: {key} is {welded_args[key]!r} in the weld "
                    f"and {certified_args[key]!r} in the certified body")

        # 3b. The OWN-cell load becomes the register for the same component, and
        #     only because the certified mask there was the dispatch's own.
        if certified_args["v_c"] != "live":
            findings.append(
                f"call {position}: the certified own-cell load is masked by "
                f"{certified_args['v_c']!r} and not by the dispatch mask, so a "
                f"register substitution would lose that mask")
        if welded_args["near_c"] != f"dfin{partner}":
            findings.append(
                f"call {position}: the own-cell displacement is "
                f"{welded_args['near_c']!r}, not the resolved register "
                f"dfin{partner} the certified g{partner}[idx] load names")

        # 3c. The three foreign loads become taps at the certified cells.
        for slot, index_key, mask_key in (("near_d", "o_d", "v_d"),
                                          ("far_u", "o_u", "v_u"),
                                          ("far_ud", "o_ud", "v_ud")):
            name = welded_args[slot]
            if name not in taps:
                findings.append(
                    f"call {position}: {slot} is {name!r}, which is not a "
                    f"declared foreign tap")
                continue
            component, index_expression, mask = taps[name]
            want_index = ast.unparse(
                ast.parse(certified_args[index_key], mode="eval").body)
            want_mask = ast.unparse(
                ast.parse(certified_args[mask_key], mode="eval").body)
            if component != partner:
                findings.append(
                    f"call {position}: {slot} re-derives component {component} "
                    f"where the certified load read g{partner}")
            if index_expression != want_index:
                findings.append(
                    f"call {position}: {slot} is re-derived at "
                    f"{index_expression!r}, not at the certified "
                    f"{want_index!r}")
            if mask != want_mask:
                findings.append(
                    f"call {position}: {slot} is served under mask {mask!r}, not "
                    f"the certified {want_mask!r}")

    # 4. The tap serves +0.0 where the certified load served `other=0.0`.
    tap_body = _statements(_function_source(spec["module"], tap_call))
    if "return tl.where(valid, value, 0.0)" not in _dedent(tap_body):
        findings.append(
            f"{tap_call} does not serve +0.0 outside its mask, so a masked-off "
            f"neighbour would carry a stepped value the certified load never saw")

    return {"leg": "tap_binding", "family": family, "passed": not findings,
            "findings": findings, "measures": measures}


# ---------------------------------------------------------------------------
# LEG GROUP 1c — is the binding leg ARMED?
# ---------------------------------------------------------------------------
#
# A source-comparison leg that matched nothing would pass on any kernel at all,
# which is the same vacuity hazard a coverage predicate has. This leg rewrites
# the weld's own source SIX ways — one per class of defect the substitution can
# have — and requires the binding leg to refuse each. It is scored on the leg
# that claims identity, never on a leg that already fails.

ARMING_MUTATIONS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "offdiag_fused_electric_pair": (
        ("tap_reads_the_wrong_component",
         "tap_yd_1 = _tap(1, i, dj, k, dvy,",
         "tap_yd_1 = _tap(2, i, dj, k, dvy,"),
        ("tap_reads_the_wrong_cell",
         "tap_yd_1 = _tap(1, i, dj, k, dvy,",
         "tap_yd_1 = _tap(1, i, uj, k, dvy,"),
        ("tap_serves_the_wrong_mask",
         "tap_yd_1 = _tap(1, i, dj, k, dvy,",
         "tap_yd_1 = _tap(1, i, dj, k, live,"),
        ("own_cell_register_swapped",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,",
         "dfin2, tap_yd_1, tap_xu_1, tap_xu_yd_1,"),
        ("coefficient_mispaired",
         "u01, idx,\n                ui * nyz + j * nz + k,\n                live, uvx)",
         "u02, idx,\n                ui * nyz + j * nz + k,\n                live, uvx)"),
        ("corner_tap_collapsed_to_the_face",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_1,"),
    ),
    "folded_offdiag_fused_electric_pair": (
        ("tap_reads_the_wrong_component",
         "tap_yd_1 = _d_final(1, i, dj, k, dvy,",
         "tap_yd_1 = _d_final(2, i, dj, k, dvy,"),
        ("tap_reads_the_wrong_cell",
         "tap_yd_1 = _d_final(1, i, dj, k, dvy,",
         "tap_yd_1 = _d_final(1, i, uj, k, dvy,"),
        ("tap_serves_the_wrong_mask",
         "tap_yd_1 = _d_final(1, i, dj, k, dvy,",
         "tap_yd_1 = _d_final(1, i, dj, k, live,"),
        ("own_cell_register_swapped",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,",
         "dfin2, tap_yd_1, tap_xu_1, tap_xu_yd_1,"),
        ("coefficient_mispaired",
         "u01, idx,\n                ui * nyz + j * nz + k,\n                live, uvx, wy, MG_Y)",
         "u02, idx,\n                ui * nyz + j * nz + k,\n                live, uvx, wy, MG_Y)"),
        ("corner_tap_collapsed_to_the_face",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_yd_1,",
         "dfin1, tap_yd_1, tap_xu_1, tap_xu_1,"),
        # FOLD-SPECIFIC. The runtime ghost weight and its constexpr arm belong to
        # the PARTNER axis, not to the row component's own; crossing them applies
        # one axis's parity to another axis's ghost lane, which is a plane of
        # wrong values and not a crash.
        ("ghost_weight_taken_from_the_wrong_axis",
         "live, uvx, wy, MG_Y)", "live, uvx, wx, MG_Y)"),
        ("ghost_arm_taken_from_the_wrong_axis",
         "live, uvx, wy, MG_Y)", "live, uvx, wy, MG_X)"),
    ),
}

#: Why each mutation is a defect worth arming against. Kept beside the table
#: rather than inside it so the two families share the prose for the six they
#: share.
ARMING_REASONS: Dict[str, str] = {
    "tap_reads_the_wrong_component":
        "the redirect re-derives the wrong displacement component",
    "tap_reads_the_wrong_cell":
        "the redirect steps UP the partner axis where the certified load "
        "stepped DOWN — the half-cell registration error",
    "tap_serves_the_wrong_mask":
        "the redirect drops the ghost mask, so a boundary neighbour carries a "
        "stepped value instead of the zero ghost",
    "own_cell_register_swapped":
        "the own-cell displacement comes from the wrong component's register",
    "coefficient_mispaired":
        "the row coefficient is bound to the other partner — the certified "
        "off-diagonal gate's own m3",
    "corner_tap_collapsed_to_the_face":
        "the corner tap is replaced by the up-face tap, which is the composed "
        "ghost rule dropped",
    "ghost_weight_taken_from_the_wrong_axis":
        "the mirror ghost weight of one axis is applied to another axis's "
        "ghost lane",
    "ghost_arm_taken_from_the_wrong_axis":
        "the weighted/unweighted arm is selected by the wrong axis's fold flag",
}


def arming_leg(family: str) -> Dict[str, Any]:
    """Every class of substitution defect is REFUSED by the binding leg."""
    spec = FAMILIES[family]
    original = _function_source
    outcomes: List[Dict[str, Any]] = []
    findings: List[str] = []
    try:
        for name, old, new in ARMING_MUTATIONS[family]:
            reason = ARMING_REASONS[name]
            def patched(path: str, function: str, _old: str = old,
                        _new: str = new) -> str:
                text = original(path, function)
                if path == spec["module"] and _old in text:
                    return text.replace(_old, _new, 1)
                return text

            globals()["_function_source"] = patched
            try:
                row = arming_target(family)
            finally:
                globals()["_function_source"] = original
            caught = not row["passed"]
            outcomes.append({"mutation": name, "caught": caught,
                             "reason": reason,
                             "first_finding": (row["findings"] or [None])[0]})
            if not caught:
                findings.append(
                    f"{name} was NOT caught: {reason} — the binding leg is "
                    f"disarmed against it")
    finally:
        globals()["_function_source"] = original
    return {"leg": "arming", "family": family, "passed": not findings,
            "findings": findings, "mutations": outcomes,
            "measures": {"mutations": len(outcomes),
                         "caught": sum(1 for row in outcomes if row["caught"])}}


def arming_target(family: str) -> Dict[str, Any]:
    """The leg the arming battery is scored on. One name, so it cannot drift."""
    return tap_binding_leg(family)


# ---------------------------------------------------------------------------
# LEG GROUP 2 — the seam choreography, on NumPy
# ---------------------------------------------------------------------------

VALUE_CLASSES: Tuple[str, ...] = ("normal", "signed_zero_lattice", "subnormal_band")

#: The fixture taxonomy, PER FAMILY. Boundary terminations decide which in-seam
#: passes are live and which ghost arms the coupling takes, and the walled/folded
#: cases are the only ones on which the corresponding null can diverge at all — a
#: taxonomy without them would score that null "predicted null" and measure
#: nothing. Each entry names the fixture and whether the family's predicate is
#: expected to ADMIT it, so a fixture built to evidence a refusal is scored as a
#: refusal rather than silently skipped.
FIXTURES: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "offdiag_fused_electric_pair": (
        {"label": "periodic", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "periodic", "y": "periodic", "z": "periodic"}},
        {"label": "walled_xy", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}},
        {"label": "walled_x", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "periodic", "z": "periodic"}},
        {"label": "walled_3d", "admitted": True, "cell": (0.6, 0.6, 0.6),
         "dimensions": 3,
         "boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"}},
    ),
    "folded_offdiag_fused_electric_pair": (
        {"label": "fold_y_even_walled", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
         "symmetry": (("Y", 1),)},
        {"label": "fold_y_odd_walled", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
         "symmetry": (("Y", -1),)},
        {"label": "fold_xy_even_walled", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
         "symmetry": (("X", 1), ("Y", 1))},
        {"label": "fold_xy_mixed_walled", "admitted": True, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
         "symmetry": (("X", -1), ("Y", 1))},
        {"label": "fold_x_walled_3d", "admitted": True, "cell": (0.6, 0.6, 0.6),
         "dimensions": 3,
         "boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"},
         "symmetry": (("X", 1),)},
        # THE REFUSAL'S OWN EVIDENCE. A folded PERIODIC axis stores MEEP's far
        # ghost slot, so `fill_folded_far_ghosts_D` writes inside the seam; the
        # predicate refuses it by name and the weld model — which does not carry
        # that pass — must DISAGREE with the driver here. A refusal nobody
        # measured is a guess.
        {"label": "fold_y_periodic_far", "admitted": False, "cell": (1.2, 1.2, 0.0),
         "boundaries": {"x": "metallic", "y": "periodic", "z": "periodic"},
         "symmetry": (("Y", 1),)},
    ),
}

#: Every stored volume the comparison walks. A weld that agreed on E while
#: leaving fu_D or f_w_E wrong is a weld that fails on step 2, so the census is
#: over the whole state rather than over the sub-step's nominal outputs.
STATE_VOLUMES: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "Bx", "By", "Bz", "Hx", "Hy", "Hz",
)


def _seed_class(array: np.ndarray, rng: np.random.Generator,
                value_class: str) -> np.ndarray:
    """Fill one volume from the named value class, in float32."""
    shape = array.shape
    values = rng.standard_normal(shape).astype(np.float32)
    if value_class == "normal":
        return values
    if value_class == "signed_zero_lattice":
        # Normal values with a +/-0 lattice over about a third of the cells. An
        # array of nothing but +/-0 leaves every route unmoved and makes the leg
        # vacuous, which the 2026-08-20 round measured and refused; the lattice
        # is what makes a sign observable without that.
        picks = rng.random(shape)
        values = np.where(picks < 0.18, np.float32(0.0), values)
        values = np.where((picks >= 0.18) & (picks < 0.36),
                          np.float32(-0.0), values)
        return values.astype(np.float32)
    if value_class == "subnormal_band":
        return (values * np.float32(1e-40)).astype(np.float32)
    raise ValueError(value_class)


def build_fixture(fixture: Dict[str, Any], value_class: str,
                  seed: int) -> Tuple[Any, Any]:
    """A real ``Grid``/``Fields``/``PML`` on NumPy, with off-diagonal rows live."""
    rng = np.random.default_rng(seed)
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=fixture["cell"],
                dimensions=fixture.get("dimensions", 2),
                boundaries=fixture["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in fixture.get("symmetry", ())))
    fields = Fields(grid=grid)
    shape = grid.shape
    diagonal = {name: np.full(shape, 2.25, dtype=np.float32)
                for name in ("Ex", "Ey", "Ez")}
    inverse = {name: np.full(shape, 1.0 / 2.25, dtype=np.float32)
               for name in ("Ex", "Ey", "Ez")}
    # SPATIALLY VARYING rows. A uniform coefficient makes the registration
    # (which node the coefficient is read at) invisible — the certified
    # off-diagonal gate records that as its own measured refinement — so the
    # rows vary cell to cell here for the same reason.
    rows = {
        "Ex": {"Ey": (0.05 + 0.01 * rng.random(shape)).astype(np.float32),
               "Ez": (0.03 + 0.01 * rng.random(shape)).astype(np.float32)},
        "Ey": {"Ez": (0.02 + 0.01 * rng.random(shape)).astype(np.float32),
               "Ex": (0.04 + 0.01 * rng.random(shape)).astype(np.float32)},
        "Ez": {"Ex": (0.01 + 0.01 * rng.random(shape)).astype(np.float32),
               "Ey": (0.06 + 0.01 * rng.random(shape)).astype(np.float32)},
    }
    fields.set_epsilon_volumes(diagonal, inverse, rows)
    fields.enable_pml_storage()
    # A MIRRORED axis takes the HIGH face only: its cell 0 lies ON the mirror
    # plane, which is a boundary condition and not an absorber, and `PML` refuses
    # the low face there by name (pml.py:408). The absorber has to be active
    # SOMEWHERE — every arm here is the split-field path — so an all-periodic
    # fixture still gets one face.
    folded = {axis for axis, _phase in fixture.get("symmetry", ())}
    thickness: Dict[str, Any] = {}
    for axis, kind in fixture["boundaries"].items():
        if axis.upper() in folded:
            thickness[axis] = {"high": 0.2}
        elif kind == "metallic":
            thickness[axis] = 0.2
    pml = PML(grid=grid, thickness=thickness or {"x": 0.2})
    for name in STATE_VOLUMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = _seed_class(array, rng, value_class)
    return fields, pml


def _words(array: Any) -> np.ndarray:
    """The uint32 words of a float32 volume.

    UINT32 AND NOT FLOAT: ``-0.0 != 0.0`` is False, so a float comparison is
    blind exactly where the signed-zero class matters, and NaN would compare
    unequal to itself where a bit-identity claim wants equality.
    """
    return np.ascontiguousarray(array).view(np.uint32).reshape(-1)


def snapshot(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(getattr(fields, name), copy=True)
            for name in STATE_VOLUMES if getattr(fields, name, None) is not None}


def compare(left: Dict[str, np.ndarray],
            right: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Differing uint32 words per volume, and the total."""
    out: Dict[str, int] = {}
    total = 0
    if set(left) != set(right):
        raise AssertionError("the two states hold different volumes")
    for name in sorted(left):
        differ = int(np.count_nonzero(_words(left[name]) != _words(right[name])))
        if differ:
            out[name] = differ
        total += differ
    out["_total"] = total
    return out


def reference_step(fields: Any, pml: Any) -> None:
    """The driver's own D-seam order for a row with no in-seam source.

    ``driver.py:3302-3313``: ``step_D``; (no electric source on the rows these
    cells serve); ``fill_symmetry_bc_D``; ``zero_metal_D``;
    ``fill_folded_far_ghosts_D``; ``update_E``. ALL THREE PASSES ARE CALLED,
    including the ones a family refuses — a reference that dropped a pass would
    agree with a weld that dropped it too, which is the whole failure mode the
    refusal-evidence fixture exists to expose.
    """
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


#: Which in-seam passes each family's closed form CARRIES, in driver order. The
#: complement of this list against ``reference_step``'s three calls is exactly
#: what that family refuses, and the refusal-evidence fixture is what measures
#: the difference.
CARRIED_PASSES: Dict[str, Tuple[str, ...]] = {
    "offdiag_fused_electric_pair": ("zero_metal_D",),
    "folded_offdiag_fused_electric_pair": ("fill_symmetry_bc_D", "zero_metal_D"),
}


def weld_step(family: str, fields: Any, pml: Any, *, skip_seam: bool = False,
              seam_after_update: bool = False,
              skip_rotation: bool = False) -> None:
    """The SCRATCH-OUTPUT choreography, with the nulls as keyword arms.

    Step out of place into a twin, resolve the family's carried in-seam passes
    over the twin, make the twin live (the rotation), then run the certified
    constitutive half — which now reads a volume nothing is writing.

    The curl and constitutive arithmetic here is the ARRAY PATH's own. That is
    deliberate: this leg tests the CHOREOGRAPHY, and substituting a hand model of
    either half would test the model. Applying the seam passes through
    ``stepping``'s own functions rather than re-deriving them is the same choice
    for the same reason — what is under test is WHERE they run (over the scratch,
    before the constitutive read), not a second transcription of WHAT they do.
    """
    twins = {name: np.array(getattr(fields, name), copy=True)
             for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")}
    live = {name: getattr(fields, name) for name in twins}
    for name, twin in twins.items():
        setattr(fields, name, twin)
    stepping.step_D(fields, pml)          # writes ONLY the twins
    if not (skip_seam or seam_after_update):
        _carried_seam(family, fields)
    if skip_rotation:
        # THE NULL: the engine keeps its pre-launch references, so the
        # constitutive half reads the volume the launch did not write.
        for name, array in live.items():
            setattr(fields, name, array)
    stepping.update_E(fields, pml)
    if seam_after_update:
        _carried_seam(family, fields)


def _carried_seam(family: str, fields: Any) -> None:
    """Run this family's carried in-seam passes over the stepped volume."""
    for name in CARRIED_PASSES[family]:
        getattr(stepping, name)(fields)


def choreography_leg(family: str, fixture: Dict[str, Any], value_class: str,
                     steps: int, seed: int) -> Dict[str, Any]:
    """Reference and weld, two complete steps, every stored volume, uint32.

    ON AN ADMITTED FIXTURE the two must agree word for word. On a fixture the
    predicate REFUSES they must DISAGREE — that is the refusal's discriminating
    experiment, and a refusal that no fixture separates is a guess.
    """
    reference_fields, reference_pml = build_fixture(fixture, value_class, seed)
    weld_fields, weld_pml = build_fixture(fixture, value_class, seed)
    start = compare(snapshot(reference_fields), snapshot(weld_fields))
    if start["_total"]:
        raise AssertionError("the two fixtures did not start identical")
    per_step: List[Dict[str, int]] = []
    for _ in range(steps):
        reference_step(reference_fields, reference_pml)
        weld_step(family, weld_fields, weld_pml)
        per_step.append(compare(snapshot(reference_fields), snapshot(weld_fields)))
    admitted = bool(fixture.get("admitted", True))
    identical = all(row["_total"] == 0 for row in per_step)
    return {"leg": "choreography", "family": family, "fixture": fixture["label"],
            "value_class": value_class, "steps": steps, "admitted": admitted,
            "differing_words_per_step": [row["_total"] for row in per_step],
            "detail": per_step[-1],
            "passed": identical if admitted else (not identical)}


def predicate_leg(family: str, fixture: Dict[str, Any],
                  seed: int) -> Dict[str, Any]:
    """The PREDICATE's verdict on each fixture, beside the arithmetic.

    Asked on a NumPy host, so the array-module clause is subtracted: what is
    under test is whether this family admits or refuses for the reason the
    fixture was built to carry, not whether Triton is installed. A fixture the
    arithmetic separates but the predicate admits is the one silent failure this
    package has (rule 2), so the two verdicts are recorded side by side.
    """
    module = __import__(
        "meep_gpu.triton_kernels." + family, fromlist=["*"])
    predicate = getattr(module, family + "_coverage")
    fields, pml = build_fixture(fixture, "normal", seed)
    verdict_here = predicate(fields, pml, ())
    reasons = [reason for reason in verdict_here.reasons
               if "not cupy" not in reason]
    admits = not reasons
    expected = bool(fixture.get("admitted", True))
    return {"leg": "predicate", "family": family, "fixture": fixture["label"],
            "admits": admits, "expected": expected, "reasons": reasons,
            "passed": admits is expected}


NULLS: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("seam_passes_skipped", {"skip_seam": True},
     "the carried in-seam passes never run, so a wall or fold plane keeps its "
     "raw stepped value and the coupling reads it"),
    ("seam_after_update", {"seam_after_update": True},
     "the carried passes run AFTER the constitutive half, which is the driver's "
     "order reversed: the coupling reads an unresolved plane"),
    ("rotation_skipped", {"skip_rotation": True},
     "the launch writes the twin and the engine keeps its old references, so "
     "the constitutive half reads a displacement one step stale"),
)


def null_leg(family: str, fixture: Dict[str, Any], value_class: str, name: str,
             arms: Dict[str, Any], reason: str, seed: int) -> Dict[str, Any]:
    """A null that MUST diverge, or is recorded predicted_null with its reason."""
    reference_fields, reference_pml = build_fixture(fixture, value_class, seed)
    weld_fields, weld_pml = build_fixture(fixture, value_class, seed)
    reference_step(reference_fields, reference_pml)
    weld_step(family, weld_fields, weld_pml, **arms)
    difference = compare(snapshot(reference_fields), snapshot(weld_fields))
    seam_is_live = (any(kind == "metallic"
                        for kind in fixture["boundaries"].values())
                    or bool(fixture.get("symmetry")))
    armable = seam_is_live or name == "rotation_skipped"
    return {"leg": "null", "family": family, "null": name,
            "fixture": fixture["label"], "value_class": value_class,
            "differing_words": difference["_total"], "detail": difference,
            "armable": armable, "reason": reason,
            "predicted_null": (not armable),
            "passed": (difference["_total"] > 0) if armable
                      else (difference["_total"] == 0)}


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def verdict(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    clauses: Dict[str, Any] = {}
    transcription = [row for row in rows if row["leg"] == "transcription"]
    bindings = [row for row in rows if row["leg"] == "tap_binding"]
    arming = [row for row in rows if row["leg"] == "arming"]
    choreography = [row for row in rows if row["leg"] == "choreography"]
    nulls = [row for row in rows if row["leg"] == "null"]
    predicates = [row for row in rows if row["leg"] == "predicate"]
    clauses["every_arithmetic_block_is_a_certified_body"] = bool(
        transcription) and all(row["passed"] for row in transcription)
    clauses["every_redirected_load_reads_the_cell_the_certified_load_addressed"] = (
        bool(bindings) and all(row["passed"] for row in bindings))
    clauses["the_binding_leg_is_armed_against_every_substitution_defect"] = (
        bool(arming) and all(row["passed"] for row in arming))
    clauses["the_choreography_is_byte_identical_to_the_driver_pass_order"] = bool(
        choreography) and all(row["passed"] for row in choreography)
    clauses["every_armable_null_diverges"] = all(
        row["passed"] for row in nulls) and any(
            row["armable"] for row in nulls)
    # A leg group that ran on ONE value class proves nothing about the other two,
    # and the signed-zero and subnormal classes are exactly where a word-level
    # claim can fail while a float one passes.
    clauses["the_choreography_ran_on_all_three_value_classes"] = (
        {row["value_class"] for row in choreography} == set(VALUE_CLASSES))
    # EVERY fixture the taxonomy declares must have been walked, admitted or
    # refused, and the predicate must have agreed with the arithmetic on each.
    # A refusal fixture that the arithmetic separates but the predicate ADMITS is
    # rule 2's one silent failure, so the two verdicts are one clause.
    clauses["the_predicate_and_the_arithmetic_agree_on_every_fixture"] = (
        bool(predicates) and all(row["passed"] for row in predicates))
    walked = {row["family"] for row in choreography}
    clauses["a_refusal_fixture_measured_the_pass_the_weld_does_not_carry"] = all(
        any(row["fixture"] == fixture["label"] and not row["admitted"]
            and any(count > 0 for count in row["differing_words_per_step"])
            for row in choreography)
        for family, family_fixtures in FIXTURES.items() if family in walked
        for fixture in family_fixtures if not fixture.get("admitted", True))
    clauses["the_choreography_ran_on_every_declared_fixture"] = all(
        {fixture["label"] for fixture in family_fixtures}
        <= {row["fixture"] for row in choreography
            if row["family"] == family}
        for family, family_fixtures in FIXTURES.items()
        if any(row["family"] == family for row in choreography))
    return {"clauses": clauses, "released": all(clauses.values()),
            "counts": {"transcription": len(transcription),
                       "tap_binding": len(bindings), "arming": len(arming),
                       "predicate": len(predicates),
                       "choreography": len(choreography), "nulls": len(nulls)}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="where to write probe.json (default: no artifact)")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--family", default=None,
                        help="restrict to one family (default: all known)")
    args = parser.parse_args(argv)

    families = ([args.family] if args.family else sorted(FAMILIES))
    started = time.time()
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "probe": "triton_offdiag_scratch_weld",
        "numpy_version": np.__version__,
        "seed": args.seed,
        "steps": args.steps,
        "families": families,
        "source_sha256": source_hashes(),
        "supersedes": "results/triton_fused_offdiag_electric_2026-08-20",
        "rows": rows,
    }

    for family in families:
        row = transcription_leg(family)
        rows.append(row)
        log(f"transcription {family}: "
            f"{'PASS' if row['passed'] else 'FAIL'} "
            f"curl_lift={row['measures'].get('curl_lift_statements')} stmts "
            f"needles={row['measures'].get('term_needles_applied')} "
            f"stray_loads={row['measures'].get('stray_displacement_loads')} "
            f"({time.time() - started:.1f} s)")
        for finding in row["findings"]:
            log(f"    ! {finding}")
        if args.out:
            save(payload, args.out)

        row = tap_binding_leg(family)
        rows.append(row)
        log(f"tap_binding {family}: {'PASS' if row['passed'] else 'FAIL'} "
            f"taps={row['measures'].get('taps_declared')} "
            f"cells={row['measures'].get('distinct_tap_cells')} "
            f"call_sites={row['measures'].get('term_call_sites')} "
            f"({time.time() - started:.1f} s)")
        for finding in row["findings"]:
            log(f"    ! {finding}")
        if args.out:
            save(payload, args.out)

        row = arming_leg(family)
        rows.append(row)
        log(f"arming {family}: {'PASS' if row['passed'] else 'FAIL'} "
            f"caught={row['measures']['caught']}/{row['measures']['mutations']} "
            f"({time.time() - started:.1f} s)")
        for mutation in row["mutations"]:
            log(f"    {mutation['mutation']:34s} "
                f"{'CAUGHT' if mutation['caught'] else 'NOT CAUGHT'}")
        if args.out:
            save(payload, args.out)

        for fixture in FIXTURES[family]:
            row = predicate_leg(family, fixture, args.seed)
            rows.append(row)
            log(f"predicate {family} {fixture['label']:22s} "
                f"admits={row['admits']!s:5s} expected={row['expected']!s:5s} "
                f"{'PASS' if row['passed'] else 'FAIL'} "
                f"({time.time() - started:.1f} s)")
            for reason in row["reasons"][:3]:
                log(f"      refused: {reason}")
            if args.out:
                save(payload, args.out)

        for fixture in FIXTURES[family]:
            for value_class in VALUE_CLASSES:
                row = choreography_leg(family, fixture, value_class, args.steps,
                                       args.seed)
                rows.append(row)
                log(f"choreography {family} {fixture['label']:22s} "
                    f"{value_class:20s} "
                    f"differing={row['differing_words_per_step']} "
                    f"admitted={row['admitted']!s:5s} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
                if args.out:
                    save(payload, args.out)

        for fixture in [f for f in FIXTURES[family] if f.get("admitted", True)][:2]:
            for name, arms, reason in NULLS:
                row = null_leg(family, fixture, "normal", name, arms, reason,
                               args.seed)
                rows.append(row)
                log(f"null {family} {name:22s} {fixture['label']:22s} "
                    f"moved={row['differing_words']} armable={row['armable']} "
                    f"{'PASS' if row['passed'] else 'FAIL'} "
                    f"({time.time() - started:.1f} s)")
                if args.out:
                    save(payload, args.out)

    payload["verdict"] = verdict(rows)
    payload["elapsed_s"] = round(time.time() - started, 2)
    log(f"VERDICT released={payload['verdict']['released']} "
        f"clauses={payload['verdict']['clauses']}")
    if args.out:
        save(payload, args.out)
        log(f"wrote {args.out}")
    return 0 if payload["verdict"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Byte gate for the COMPLEX fused magnetic pair: complex ``step_B`` into ``update_H``.

DEVICE STATUS: **RELEASED 2026-08-20.** This ran on the GPU host and released. Its
    first device run found FOUR defects and every one was in THIS HARNESS rather
    than in the kernel: 2-D cases declaring a PEC on the invariant z axis, which
    ``Grid`` refuses by name; a mutation whose replacement opened a suite on the
    block's first line, so the mutant failed to IMPORT and scored as unarmed;
    ``MATERIAL`` naming files the inventory scan does not produce, which matched
    nothing and thereby disabled both the vacuity floor and ``material_changed``;
    and mutations scored on a grid that never enters the branch they rewrite. The
    last is now a laptop guard in this product's test file. This stream was built and
laptop-tested under a standing instruction not to run a device gate — the CUDA host
was owned by a separate four-stream campaign — so no leg below marked ``device`` has
ever executed and this gate has produced no device artifact. The ``--no-device``
legs HAVE run, on a laptop, and are labelled as such in the payload. Nothing in this
repository claims byte identity for this product.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT, once it runs: for every configuration
:func:`~meep_gpu.triton_kernels.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage`
admits, ONE launch of ``complex_fused_curl_constitutive_B`` leaves the engine in a
state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_B`` / ``fill_symmetry_bc_B`` /
    ``zero_metal_B`` / ``fill_folded_far_ghosts_B`` / ``update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the complex split-field
    curl (``complex_fields.plan_complex_pml_curl`` on ``step_B``) and the complex
    constitutive (``complex_fields.plan_complex_constitutive`` on side H), with
    ``zero_metal_B`` and the two symmetry fills on the array path in that route
    because no Triton product owns them,

over every allocated volume: the complex primaries, the split-field PML auxiliaries
and the constitutive ``f_w`` history. Comparison is on the uint32 word view.
``allclose`` appears nowhere.

WHAT THE GATE REFUSES TO INFER.

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array must
  MOVE during the leg; a leg whose magnetic trio is frozen in all three routes is
  armed and must be caught by the moved-state census.
* **"The two symmetry fills are no-ops here" is a claim, not a construction.** It is
  MEASURED on the laptop (``NO-DEVICE LEG 3``), with a folded control that separates,
  so the leg is about the fold rather than about the probe.
* **An unlicensed EXPANSION arm is not a small error.** The arm is a measured
  platform fact, the two licensable arms differ in roughly a quarter of words, and a
  record cut under one subnormal policy licenses nothing under the other.
  ``NO-DEVICE LEG 2`` runs the SHIPPED
  :func:`~meep_gpu.triton_kernels.complex_fields.expansion_license` over this
  product's pattern set and records the verdict, and it carries FALSIFICATION rows —
  a keep record read as flush, a NEITHER pattern, a missing pattern, an
  evidence-free ``AMBIGUOUS_BOTH`` — that must all refuse. A licence leg that only
  ever says yes is not a check.

===========================================================================
THE SEAM, AND WHY IT IS THE MAGNETIC ONE
===========================================================================

The driver runs five passes (driver.py:3281-3288)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

The magnetic sources are refused by the predicate; the two symmetry fills return at
their first line unless ``grid.has_symmetry()`` and both halves refuse a symmetry,
so on every admitted configuration they cannot write a word; ``zero_metal_B`` is
carried inline, on BOTH word planes. From the 186-row Triton census
(``results/predicate_coverage_2026-08-16_wired_convention`` — the licensed one; the
plain ``_wired`` record's probe carries no resolved policy and reports every complex
family as covering 0 rows for that reason alone) this admits **12 rows** against 1
for the same construction on the electric half. The funnel and the four
magnetic-source rows that cost this arm its other four are reproduced by
``results/triton_complex_fused_magnetic_pair_census_2026-08-19T2355/count_corpus_admission.py``.

POLICY, stamped and unchanged: ``num_warps=1`` (the settled cross-sub-step policy),
``BLOCK=complex_fields.DEFAULT_BLOCK``,
``enable_fp_fusion=kernels.ENABLE_FP_FUSION`` (False), subnormal policy ``keep``
installed with ``strict=True`` before the first device compile — which is
``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`` and the policy the probe
artifact this gate consumes was cut under. This gate does not tune and does not
time.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_complex_fused_magnetic_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> \\
    MEEP_GPU_COMPLEX_EXPANSION_PROBE=<keep-cut probe.json> python -u \\
        parity/meep_gpu/probe_triton_complex_fused_magnetic_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import copy
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

SEED = 20260819

#: The keep-cut expansion probe artifacts in this tree, newest first. The licence
#: leg scores EVERY one of them through the shipped rule; the device legs consume
#: whichever the environment names (``MEEP_GPU_COMPLEX_EXPANSION_PROBE``), and the
#: leg records which that was so the artifact says what licensed its own bytes.
KEEP_PROBES: Tuple[str, ...] = (
    "parity/meep_gpu/results/expansion_probe_2026-08-17/expansion_probe_keep.json",
    "parity/meep_gpu/results/complex_expansion_convention_2026-08-16/results/"
    "probe_keep/probe.json",
)

#: (name, dimensions, cell, boundaries, k_point, pml, steps). Every case stores
#: complex fields. THE SET IS SWEPT AND NEITHER AXIS IS DECORATION: a k = 0 case is
#: what makes the phase-skip reduction to the plain complex path measurable, a
#: phased case is the only place the rotation mutations have anywhere to be caught,
#: and a case with real METALLIC walls is the only place the wall-clear mutations
#: fire. A run with none of the three would report those mutations as uncaught.
CASES: Tuple[Tuple[str, int, Tuple[float, float, float], Any,
                   Tuple[float, float, float], Dict[str, Any], int], ...] = (
    ("bloch_x_periodic", 2, (3.0, 3.0, 0.0), "periodic", (0.23, 0.0, 0.0),
     {"y": 5}, 10),
    ("bloch_edge_minus_one", 2, (3.0, 3.0, 0.0), "periodic", (0.5, 0.0, 0.0),
     {"y": 5}, 10),
    ("k0_complex_storage", 2, (3.0, 3.0, 0.0), "periodic", (0.0, 0.0, 0.0),
     {"x": 5, "y": 5}, 10),
    ("metallic_walls_k0", 2, (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.0), {"x": 5, "y": 5}, 10),
    ("bloch_x_metallic_y", 2, (3.0, 3.0, 0.0),
     ("periodic", "metallic", "periodic"), (0.19, 0.0, 0.0), {"y": 5}, 10),
)

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3281-3288).
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: Read-only material inputs: a leg that changes one of these measured something
#: other than the step.
#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds.
#:
#: MEASURED 2026-08-20: this tuple read ``("inverse_epsilon", "epsilon")`` while
#: the scan takes its keys from ``vars(fields)``, which names them ``eps`` and
#: ``inv_eps``. Nothing matched, and that silently broke BOTH directions of the
#: check it feeds: the vacuity floor demanded that two read-only INPUTS move,
#: which no correct kernel does, so every device leg failed as VACUOUS; and
#: ``material_changed`` -- a kernel writing its own material -- could never fire,
#: because the set it scans was always empty. A name that matches nothing is not a
#: weaker check, it is two absent ones. ``_assert_material_names_are_real`` below
#: is what stops a rename from doing this again.
MATERIAL = ("eps", "inv_eps")


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    """Every MATERIAL name must appear in the inventory the scan actually built."""
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not "
            f"loosen the floor.")

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records every counter at zero, because nothing had compiled yet — a record
    # that cannot tell an installed policy apart from an inert one.
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
        "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_complex_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where a ``@triton.jit`` object is ``None``, so the transcription check
    bites at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is
    the PARENTHESISATION, and unparsing re-derives minimal parentheses and would
    silently rewrite ``((c_y_re - c_re) + (b_re - b_z_re))`` into a different
    grouping than the one whose float32 bits the gate is about.

    The docstring IS stripped, and that is not cosmetic: these bodies document the
    codes and branches they do NOT carry, so a text search over the raw source
    finds those names in prose and reports them as live code.
    """
    import ast  # noqa: PLC0415

    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"bloch_pml_curl_step": "complex_fields.py",
                         "bloch_constitutive_step": "complex_fields.py",
                         "complex_fused_curl_constitutive_B":
                             "complex_fused_magnetic_pair.py"}[name])
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
    # Normalise the indentation to what ``textwrap.dedent(inspect.getsource(fn))``
    # produces, which is what :func:`shipped_source` hands the mutation table on a
    # CUDA host. Two spellings of the same text would let a rewrite match here and
    # miss there — the exact way a mutation silently disarms.
    margin = node.col_offset
    lines = [lines[0].lstrip()] + [
        line[margin:] if line[:margin].strip() == "" else line.lstrip()
        for line in lines[1:]]
    return "\n".join(lines)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text.

    Matching on stripped lines rather than on an exact substring is what keeps a
    rewrite armed across a reindentation or a trailing comment. A mutation that
    silently stops matching reports a real defect as uncaught, which this tree has
    already paid for once.
    """
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


def _statements(text: str) -> List[str]:
    """Executable lines, comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


#: The arithmetic lines the fused body must reproduce VERBATIM from the certified
#: complex curl. Each is a line whose grouping decides float32 bits; a reformat is
#: a different number, not a style change.
CURL_LINES = (
    "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))",
    "t0_im = ((c_y_im - c_im) + (b_im - b_z_im))",
    "t1_re = ((a_z_re - a_re) + (c_re - c_x_re))",
    "t1_im = ((a_z_im - a_im) + (c_im - c_x_im))",
    "t2_re = ((b_x_re - b_re) + (a_re - a_y_re))",
    "t2_im = ((b_x_im - b_im) + (a_im - a_y_im))",
    "curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)",
    "r_re = (r_re + n0_re) - p0_re",
    "r_im = (r_im + n0_im) - p0_im",
    "v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)",
)

#: The constitutive accumulation, from ``complex_fields.bloch_constitutive_step``'s
#: SCALE=0 arm. Two separate accumulations, left to right, on BOTH planes;
#: flattening them is a different float32 number.
CONSTITUTIVE_LINES = (
    "t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
    "a_re = a_re + t_re",
    "a_im = a_im + t_im",
    "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
    "a_re = a_re - t_re",
    "a_im = a_im - t_im",
)

#: The weld itself: the constitutive source is the register the curl produced, on
#: both planes, for all three components.
WELD_LINES = (
    "src_re = v0_re", "src_im = v0_im",
    "src_re = v1_re", "src_im = v1_im",
    "src_re = v2_re", "src_im = v2_im",
)

#: The wall clear, carried inline on BOTH planes. ``stepping._zero_metal`` writes
#: complex zero; a transcription that cleared only the real plane is wrong exactly
#: where the imaginary part is nonzero at a wall.
WALL_LINES = (
    "v0_re = tl.where(at_x, 0.0, v0_re)", "v0_im = tl.where(at_x, 0.0, v0_im)",
    "v1_re = tl.where(at_y, 0.0, v1_re)", "v1_im = tl.where(at_y, 0.0, v1_im)",
    "v2_re = tl.where(at_z, 0.0, v2_re)", "v2_im = tl.where(at_z, 0.0, v2_im)",
)


#: Words per complex cell — the real and imaginary planes at ``2*idx`` / ``2*idx+1``.
#: Named because the weld check below counts loads and "1" would be wrong for a
#: reason that has nothing to do with the weld.
WORDS_PER_COMPLEX_CELL = 2


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415

    fused = _statements(_shipped_text("complex_fused_curl_constitutive_B"))
    curl = _statements(_shipped_text("bloch_pml_curl_step"))
    constitutive = _statements(_shipped_text("bloch_constitutive_step"))

    findings: List[str] = []
    for line in CURL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in curl:
            findings.append(
                f"complex_fields.bloch_pml_curl_step no longer contains {line!r}; "
                f"the fused body's transcription source has moved")
    for line in CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
        if line not in constitutive:
            findings.append(
                f"complex_fields.bloch_constitutive_step no longer contains "
                f"{line!r}; the constitutive grouping's source has moved")
    for line in WELD_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the weld line {line!r}")
    for line in WALL_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the wall-clear line {line!r}: "
                f"stepping._zero_metal writes COMPLEX zero and both planes must be "
                f"cleared")

    # THE WELD REPLACED A LOAD, AND THE LOAD MUST BE GONE. If the constitutive half
    # still read B from memory, the "fusion" would be a launch-count optimisation
    # with a live re-read — a different kernel with a different hazard. TWO is the
    # right count and not one: the curl half loads each source component's real and
    # imaginary word at ``2 * idx`` and ``2 * idx + 1``. An unwelded body would
    # carry FOUR, because ``bloch_constitutive_step`` loads the same pair again.
    for needle in ("tl.load(g0 + 2 * idx", "tl.load(g1 + 2 * idx",
                   "tl.load(g2 + 2 * idx"):
        occurrences = sum(1 for line in fused if needle in line)
        if occurrences != WORDS_PER_COMPLEX_CELL:
            findings.append(
                f"{needle!r} appears {occurrences} times, not "
                f"{WORDS_PER_COMPLEX_CELL}; the curl half reads each source "
                f"component's two words once and the constitutive half must read B "
                f"from the register, not from memory")

    # NO MULTIPLY OUTSIDE THE THREE LICENSED HELPERS. A fourth would be a fourth
    # operand orientation, needing its own probe pattern before it could be
    # licensed — and this product's whole pattern claim is that it adds none.
    helpers = sorted({name for name in product.LICENSED_MULTIPLY_HELPERS
                      if any(name in line for line in fused)})
    stray = sorted({line for line in fused
                    if "_mul_" in line or "_rotate_" in line
                    if not any(name in line for name in product.LICENSED_MULTIPLY_HELPERS)})
    if stray:
        findings.append(
            f"the fused body calls a complex-multiply helper outside "
            f"{product.LICENSED_MULTIPLY_HELPERS}: {stray}")
    if set(helpers) != set(product.LICENSED_MULTIPLY_HELPERS):
        findings.append(
            f"the fused body calls {helpers}, not all of "
            f"{list(product.LICENSED_MULTIPLY_HELPERS)}; the pattern-set claim "
            f"names orientations the kernel no longer launches")

    # THE inv_eps ARM MUST NOT BE HERE. `SCALE` belongs to the E side; carrying it
    # would add the D-LEFT orientation this product's pattern set does not cover.
    if any("SCALE" in line for line in fused):
        findings.append(
            "the fused body mentions SCALE; that is the E-side inv_eps arm, which "
            "adds an operand orientation outside this product's probe pattern set")
    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "curl_lines_checked": len(CURL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "weld_lines_checked": len(WELD_LINES),
        "wall_lines_checked": len(WALL_LINES),
        "multiply_helpers_called": helpers,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the EXPANSION licence, and what refuses it
# ---------------------------------------------------------------------------

def _neither(record: Dict[str, Any]) -> Dict[str, Any]:
    broken = copy.deepcopy(record)
    broken["patterns"]["c8_mul_c8"] = "NEITHER"
    return broken


def _missing_pattern(record: Dict[str, Any]) -> Dict[str, Any]:
    broken = copy.deepcopy(record)
    broken["patterns"].pop("f4_mul_c8_coefficient_left", None)
    return broken


def _cross_policy(record: Dict[str, Any]) -> Dict[str, Any]:
    """The candidates cut under one policy, the platform's bytes under the other.

    This is the 2026-08-11 defect in its licensable-looking disguise: every pattern
    names an arm, and the comparison that produced them was broken.
    """
    broken = copy.deepcopy(record)
    broken.setdefault("candidates", {})["policy"] = "flush"
    return broken


def _evidence_free_ambiguity(record: Dict[str, Any]) -> Dict[str, Any]:
    """``AMBIGUOUS_BOTH`` asserted with no vectors behind it.

    ``count_nonzero`` over an empty array is zero, so an empty probe makes every
    arm "match"; the rule must refuse the claim rather than exclude the pattern.
    """
    broken = copy.deepcopy(record)
    for name in broken["patterns"]:
        broken["patterns"][name] = "AMBIGUOUS_BOTH"
    broken.pop("vectors", None)
    broken.pop("detail", None)
    return broken


def _disagreeing_arms(record: Dict[str, Any]) -> Dict[str, Any]:
    broken = copy.deepcopy(record)
    broken["patterns"]["python_float_left"] = "NAIVE"
    broken["patterns"]["c8_mul_c8"] = "FMA_V1"
    return broken


def expansion_licence_leg() -> Dict[str, Any]:
    """Which arm this product binds, on what evidence, and what refuses it.

    The rule is NOT re-implemented: this calls the shipped
    :func:`~meep_gpu.triton_kernels.complex_fields.expansion_license` over
    ``PRODUCT_PROBE_PATTERNS``, which for this product is the BASE four because it
    launches no operand orientation its two halves do not (leg 1 checks that by
    scanning the body).

    THE FALSIFICATION ROWS ARE THE POINT. A licence leg that reads one good
    artifact and reports 'FMA_V1, measured' measures the artifact, not the rule;
    each broken record below must refuse, and a broken record that LICENSES is the
    finding.
    """
    from meep_gpu.triton_kernels import complex_fields as complex_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415

    patterns = list(product.PRODUCT_PROBE_PATTERNS)
    findings: List[str] = []
    if tuple(patterns) != tuple(complex_module.PROBE_PATTERNS):
        findings.append(
            f"this product declares pattern set {patterns}, which is not the base "
            f"set {list(complex_module.PROBE_PATTERNS)}; leg 1's body scan says it "
            f"launches no orientation outside the base four, so the two disagree")

    licensed: List[Dict[str, Any]] = []
    for relative in KEEP_PROBES:
        path = os.path.join(API_ROOT, relative)
        if not os.path.exists(path):
            findings.append(f"the declared keep probe {relative} is not in the tree")
            continue
        record = json.load(open(path, encoding="utf-8"))
        verdict = complex_module.expansion_license(record, patterns)
        row = {
            "artifact": relative,
            "arm": verdict["arm"],
            "basis": verdict["basis"],
            "expansion": verdict["expansion"],
            "refusals": list(verdict["refusals"]),
            "discriminating": dict(verdict["discriminating"]),
            "non_discriminating": list(verdict["non_discriminating"]),
            "policy_resolved": verdict["policy_resolved"],
            "candidate_policy": verdict["candidate_policy"],
            "environment": verdict["environment"],
        }
        licensed.append(row)
        if verdict["arm"] is None:
            findings.append(f"{relative} licenses nothing: {verdict['refusals']}")
            continue
        if verdict["basis"] != "measured":
            findings.append(
                f"{relative} licenses {verdict['arm']} on basis "
                f"{verdict['basis']!r}; this product's arm must be MEASURED on the "
                f"run's own probe, not taken from ENVIRONMENT_DEFAULTS")
        if verdict["policy_resolved"] != complex_module.CERTIFIED_UNDER_SUBNORMAL_POLICY:
            findings.append(
                f"{relative} was cut under {verdict['policy_resolved']!r}, not the "
                f"{complex_module.CERTIFIED_UNDER_SUBNORMAL_POLICY!r} every complex "
                f"arm in this package was certified under; a licence does not "
                f"transfer across the policy boundary")
    arms = sorted({row["arm"] for row in licensed if row["arm"]})
    if len(arms) > 1:
        findings.append(
            f"the keep-cut artifacts in this tree license DIFFERENT arms {arms}; "
            f"one constexpr cannot represent a platform whose artifacts disagree")

    # The falsification rows, against the first licensing artifact.
    base = next((json.load(open(os.path.join(API_ROOT, row["artifact"]),
                                encoding="utf-8"))
                 for row in licensed if row["arm"]), None)
    refusals: List[Dict[str, Any]] = []
    if base is None:
        findings.append(
            "no artifact licensed an arm, so the falsification rows had nothing to "
            "break and this leg measured nothing")
    else:
        for name, broken, needle in (
            ("neither_pattern", _neither(base), "NO licensable arm"),
            ("missing_pattern", _missing_pattern(base), "not classified"),
            ("candidates_cut_under_another_policy", _cross_policy(base),
             "broken comparison"),
            ("evidence_free_ambiguity", _evidence_free_ambiguity(base), ""),
            ("discriminating_patterns_disagree", _disagreeing_arms(base),
             "disagree"),
            ("not_a_record", None, "no probe record"),
            ("host_backend", {**copy.deepcopy(base), "backend": "numpy"}, "cupy"),
        ):
            verdict = complex_module.expansion_license(broken, patterns)
            refused = verdict["arm"] is None
            named = (not needle) or any(needle in reason
                                        for reason in verdict["refusals"])
            refusals.append({
                "case": name, "arm": verdict["arm"],
                "refusals": list(verdict["refusals"])[:3],
                "refused": refused, "reason_names_the_defect": named,
                "passed": refused and named,
            })
            if not (refused and named):
                findings.append(
                    f"falsification row {name!r} was NOT refused by name "
                    f"(arm={verdict['arm']!r}); the licence rule does not catch it")

    return {
        "leg": "expansion_licence",
        "device": False,
        "probe_patterns": patterns,
        "certified_under_subnormal_policy":
            complex_module.CERTIFIED_UNDER_SUBNORMAL_POLICY,
        "licensed": licensed,
        "falsification": refusals,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — are the two symmetry fills really inert in this seam?
# ---------------------------------------------------------------------------

def _complex_state(fields) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in ("Bx", "By", "Bz", "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = np.ascontiguousarray(
                np.asarray(array)).view(np.uint32).ravel().copy()
    return out


def _build_complex(cell, boundaries, k_point, mirrors=(), thickness=0.4,
                   dimensions=2, seed=SEED):
    """A real complex Grid/Fields/PML triple on NumPy, seeded identically."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = (rng.uniform(-0.25, 0.25, size=array.shape)
                      + 1j * rng.uniform(-0.25, 0.25, size=array.shape)
                      ).astype(array.dtype)
    return fields, pml


def _array_path_seam(fields, pml, driver_module) -> None:
    """The five driver passes this product replaces, in driver order."""
    driver_module.step_B(fields, pml)
    driver_module.fill_symmetry_bc_B(fields)
    driver_module.zero_metal_B(fields)
    driver_module.fill_folded_far_ghosts_B(fields)
    driver_module.update_H(fields, pml)


def _welded_seam(fields, pml, driver_module, *, wall: str = "before",
                 history: str = "read_then_write") -> None:
    """The FUSED semantics, spelled with array ops so a laptop can execute them.

    THIS IS NOT THE KERNEL AND DOES NOT PRETEND TO BE. It cannot see a memory
    hazard, a register reuse or a rounding order inside one launch — those are the
    device gate's to measure. What it CAN decide is the DESIGN question the weld
    answers: which value reaches ``update_H``, and in which order relative to the
    wall clear and to the ``f_w`` history read. Each is a knob and the leg reports
    which flips the array path can see.

    THE TWO SYMMETRY FILLS ARE SIMPLY ABSENT HERE. That is the module's central
    claim and this leg is where it is measured: on an unfolded complex run the
    five-pass array path and this three-pass emulation must be BIT-IDENTICAL, and
    the folded control must separate.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415

    driver_module.step_B(fields, pml)
    if wall == "before":
        driver_module.zero_metal_B(fields)

    for component, source_name, axis_name in stepping.H_CONSTITUTIVE_TERMS:
        target = getattr(fields, component)
        source = getattr(fields, source_name)
        history_array = getattr(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(pml, axis_name,
                                                       half_integer=False)
        kps, kms = np.asarray(kps), np.asarray(kms)
        if history == "read_then_write":
            previous = history_array.copy()
            history_array[...] = source
        else:
            # The defect: f_w is written before it is read, so `previous` is the
            # value just stored. Wrong only where kms != 0, i.e. inside the PML.
            history_array[...] = source
            previous = history_array.copy()
        target += kps * history_array
        target -= kms * previous

    if wall == "after":
        # The defect: update_H has already consumed the pre-clear B, so H and
        # f_w_H at the wall carry the stepped displacement rather than zero.
        driver_module.zero_metal_B(fields)
    elif wall == "dropped":
        pass


def symmetry_inertness_leg() -> Dict[str, Any]:
    """Do ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` write a word here?

    The module refuses a symmetry and then treats both passes as no-ops. That is
    read off ``stepping._fill_symmetry_ghost_cells``'s and
    ``_fill_folded_far_ghosts``' first lines, which is an argument; this is the
    measurement, with the folded control that makes it a statement about the FOLD
    rather than about the probe.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []

    unfolded = (
        ("bloch_x", (3.0, 3.0, 0.0), "periodic", (0.23, 0.0, 0.0)),
        ("k0", (3.0, 3.0, 0.0), "periodic", (0.0, 0.0, 0.0)),
        ("metallic_x", (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
         (0.0, 0.0, 0.0)),
    )
    for name, cell, boundaries, k_point in unfolded:
        five, five_pml = _build_complex(cell, boundaries, k_point)
        before = _complex_state(five)
        _array_path_seam(five, five_pml, driver_module)
        after_five = _complex_state(five)

        three, three_pml = _build_complex(cell, boundaries, k_point)
        _welded_seam(three, three_pml, driver_module)
        after_three = _complex_state(three)

        differing = sorted(k for k in after_five
                           if not np.array_equal(after_five[k], after_three[k]))
        moved = sorted(k for k in after_five
                       if not np.array_equal(after_five[k], before[k]))
        rows.append({"case": name, "folded": False, "identical": not differing,
                     "differing_arrays": differing, "arrays_moved": moved})
        if differing:
            findings.append(
                f"{name}: the two symmetry fills are NOT inert on an unfolded "
                f"complex run — they moved {differing}")
        if not moved:
            findings.append(f"{name}: VACUOUS, the array path moved nothing")

    # THE FOLDED CONTROL. Real fields, because a fold plus complex storage is
    # refused by the halves anyway and this leg is only asking whether the two
    # passes CAN write. If the control does not separate, this leg is measuring a
    # dead call rather than an inert one.
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def folded(seed=SEED):
        # z stays PERIODIC: an invariant axis has no outer face, and Grid refuses
        # a wall there (grid.py:863). The fold is on y, which is the axis the two
        # fills act on.
        grid = Grid(resolution=10.0, cell_size=(3.0, 3.0, 0.0), dimensions=2,
                    boundaries=("metallic", "metallic", "periodic"),
                    symmetry=(Mirror("Y", 1),))
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        rng = np.random.default_rng(seed)
        for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "fu_Bx", "fu_By", "fu_Bz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            array = getattr(fields, name, None)
            if array is None:
                continue
            array[...] = rng.uniform(-0.25, 0.25,
                                     size=array.shape).astype(array.dtype)
        return fields, PML(grid=grid, thickness=0.4)

    five, five_pml = folded()
    _array_path_seam(five, five_pml, driver_module)
    after_five = _complex_state(five)
    three, three_pml = folded()
    _welded_seam(three, three_pml, driver_module)
    after_three = _complex_state(three)
    control_differing = sorted(k for k in after_five
                               if not np.array_equal(after_five[k], after_three[k]))
    rows.append({"case": "folded_control", "folded": True,
                 "identical": not control_differing,
                 "differing_arrays": control_differing})
    if not control_differing:
        findings.append(
            "the FOLDED control did not separate: dropping the two symmetry fills "
            "changed nothing even on a folded grid, so this leg cannot distinguish "
            "an inert pass from a dead one")

    return {"leg": "symmetry_inertness", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the design sweep, on NumPy
# ---------------------------------------------------------------------------

def design_sweep_leg() -> Dict[str, Any]:
    """Each design choice flipped, on NumPy, against the array-path composition.

    A choice whose flip is byte-identical is reported as a NULL WITH ITS REASON —
    a measurement about the configuration, not a licence to drop the choice from
    the kernel.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    knobs = (
        ("faithful", {}, "null"),
        ("wall_clear_dropped", {"wall": "dropped"}, "caught"),
        ("wall_clear_after_the_constitutive_read", {"wall": "after"}, "caught"),
        ("history_written_before_it_is_read",
         {"history": "write_then_read"}, "caught"),
    )
    cases = (
        ("bloch_x", (3.0, 3.0, 0.0), "periodic", (0.23, 0.0, 0.0)),
        ("k0", (3.0, 3.0, 0.0), "periodic", (0.0, 0.0, 0.0)),
        ("metallic_walls", (3.0, 3.0, 0.0),
         ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0)),
    )

    rows: List[Dict[str, Any]] = []
    for name, cell, boundaries, k_point in cases:
        reference, reference_pml = _build_complex(cell, boundaries, k_point)
        before = _complex_state(reference)
        _array_path_seam(reference, reference_pml, driver_module)
        oracle = _complex_state(reference)
        moved_names = sorted(key for key in oracle
                             if not np.array_equal(oracle[key], before[key]))
        for knob, kwargs, expectation in knobs:
            fields, pml = _build_complex(cell, boundaries, k_point)
            _welded_seam(fields, pml, driver_module, **kwargs)
            candidate = _complex_state(fields)
            differing = sorted(key for key in oracle
                               if not np.array_equal(candidate[key], oracle[key]))
            rows.append({
                "case": name, "knob": knob, "expectation": expectation,
                "identical": not differing, "differing_arrays": differing,
                "arrays_moved_by_the_array_path": moved_names,
            })
    findings: List[str] = []
    for row in rows:
        if row["knob"] == "faithful" and not row["identical"]:
            findings.append(
                f"{row['case']}: the FAITHFUL emulation diverges from the array "
                f"path on {row['differing_arrays']} — the design, not the kernel, "
                f"is wrong")
        if not row["arrays_moved_by_the_array_path"]:
            findings.append(f"VACUOUS case (nothing moved): {row['case']}")
    # A 'caught' knob may legitimately be invisible on a case with no wall; the
    # requirement is that it is caught SOMEWHERE, which is what the case sweep is
    # for. Reported per knob rather than per row.
    for knob, _kwargs, expectation in knobs:
        if expectation != "caught":
            continue
        if all(row["identical"] for row in rows if row["knob"] == knob):
            findings.append(
                f"{knob}: expected to be caught on at least one case and was "
                f"caught on none")
    return {"leg": "design_sweep", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real complex grids."""
    from meep_gpu import expansion_refusal  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415

    probe_path = os.path.join(API_ROOT, KEEP_PROBES[0])
    probe = json.load(open(probe_path, encoding="utf-8"))

    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    def _deposit(fields, component):
        """A REAL source that publishes the index the injection writes.

        ``_Source`` is enough to PLACE a source in the seam and deliberately not
        enough to CARRY one. A row about the carry needs the engine's own source, so
        this builds one and refuses to return a source that deposits nothing --
        otherwise the admitting row would pass by measuring an empty scatter.
        """
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component=component,
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    def _with_flag_false(module, call):
        """``call()`` with this family's carry declaration held down, then restored.

        The OTHER DIRECTION of the 2026-08-30 flip, measured through the shipped
        predicate rather than argued. Restored in a finally so a raising predicate
        cannot leave the module lying to every row after it.
        """
        saved = module.CARRIES_DEPOSIT_REPAIR
        module.CARRIES_DEPOSIT_REPAIR = False
        try:
            return call()
        finally:
            module.CARRIES_DEPOSIT_REPAIR = saved

    def residual(verdict) -> List[str]:
        """The reasons that are not the NumPy-host backend clause.

        MATCHED ON THE CLAUSE, NOT ON THE WORD 'cupy'. The sibling gates filter
        every reason mentioning cupy, which on this family silently swallows the
        EXPANSION clause too — it names the backend the probe must be for — and a
        refusal that vanishes from the residual is a refusal the leg reports as an
        admission. Measured here: with ``probe={}`` the word-filter left an empty
        residual list and the row passed as ADMITTED.
        """
        return [reason for reason in verdict.reasons
                if not ("array module is" in reason and "not cupy" in reason)]

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = residual(verdict)
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left[:6], "needle": needle, "passed": ok})

    # ``policy_is_installed()`` is False on this host — nothing compiles here — so
    # without the declaration ``_policy_in_force`` answers UNREADABLE_POLICY and
    # every complex arm refuses for that alone. Declaring is what makes the OTHER
    # clauses the thing being measured.
    with expansion_refusal.declaring_run_policy("keep"):
        bloch, bloch_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                          (0.23, 0.0, 0.0))
        record("bloch_no_sources",
               product.complex_fused_magnetic_pair_coverage(bloch, bloch_pml, (),
                                                            probe=probe), True)
        record("bloch_electric_source",
               product.complex_fused_magnetic_pair_coverage(
                   bloch, bloch_pml, (_Source("D"),), probe=probe), True)
        # THE IN-SEAM MAGNETIC DEPOSIT IS CARRIED AS OF 2026-08-30. Same three rows
        # as the two folded families: carried, refused by name without a publishable
        # index, and refused for BEING in the seam again once the flag is held down.
        record("bloch_magnetic_deposit_is_carried",
               product.complex_fused_magnetic_pair_coverage(
                   bloch, bloch_pml, (_deposit(bloch, "Hy"),), probe=probe), True)
        record("bloch_magnetic_source_without_a_deposit_index",
               product.complex_fused_magnetic_pair_coverage(
                   bloch, bloch_pml, (_Source("B"),), probe=probe),
               False, "does not publish the index it writes")
        record("bloch_magnetic_deposit_refused_when_the_flag_is_held_False",
               _with_flag_false(
                   product,
                   lambda: product.complex_fused_magnetic_pair_coverage(
                       bloch, bloch_pml, (_deposit(bloch, "Hy"),), probe=probe)),
               False, "is magnetic")
        record("undeclared_sources",
               product.complex_fused_magnetic_pair_coverage(bloch, bloch_pml, None,
                                                            probe=probe),
               False, "was not declared")
        record("no_probe_artifact",
               product.complex_fused_magnetic_pair_coverage(bloch, bloch_pml, (),
                                                            probe={}),
               False, "expansion probe artifact")
        record("no_pml",
               product.complex_fused_magnetic_pair_coverage(bloch, None, (),
                                                            probe=probe),
               False, "PML")

        walls, walls_pml = _build_complex(
            (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0))
        record("metallic_walls_k0",
               product.complex_fused_magnetic_pair_coverage(walls, walls_pml, (),
                                                            probe=probe), True)

        folded, folded_pml = _build_complex(
            (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
            (0.0, 0.0, 0.0), mirrors=(("Y", 1),))
        record("folded_grid",
               product.complex_fused_magnetic_pair_coverage(folded, folded_pml, (),
                                                            probe=probe),
               False, "mirror plane")

        # The plan builder must refuse exactly where the predicate does — a licence
        # refused at the coverage seam but still bound at the plan seam is how an
        # unlicensed arm reaches a kernel.
        builder_rows = []
        for label, args, kwargs in (
            ("magnetic_source_without_a_deposit_index",
             (bloch, bloch_pml, (_Source("B"),)), {"probe": probe}),
            ("undeclared_sources", (bloch, bloch_pml, None), {"probe": probe}),
            ("no_probe_artifact", (bloch, bloch_pml, ()), {"probe": {}}),
            ("folded_grid", (folded, folded_pml, ()), {"probe": probe}),
        ):
            plan = product.plan_complex_fused_magnetic_pair(*args, **kwargs)
            builder_rows.append({"case": label, "plan_is_none": plan is None,
                                 "passed": plan is None})

    findings = [row["case"] for row in rows if not row["passed"]]
    findings += [f"builder:{row['case']}" for row in builder_rows
                 if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "builder_rows": builder_rows, "findings": findings,
            "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 6 — the seam passes are the driver's own
# ---------------------------------------------------------------------------

def seam_binding_leg() -> Dict[str, Any]:
    """``REPLACES`` must name the call sites ``driver.step`` actually runs there.

    The seam's extent is DERIVED from :data:`fastpath.DRIVER_SLOTS`, not counted
    in lines. It was a literal 12-line window until 2026-09-02, when the
    dispatch surface grew the two fill consults — ``fill_B``/``fill_D`` joined
    ``DRIVER_SLOTS`` and the two folded-far passes gained consults of their own —
    which moved ``update_H`` out of a fixed count of lines below ``step_B`` and
    read as "the seam is not where it was". The seam is now walked consult by
    consult: it runs from the ``step_B`` consult to the ``update_H`` consult
    that closes it, and reaching a consult this seam does NOT own before that
    closer is itself the finding. That asserts strictly more than the line count
    did — a foreign slot dispatched inside the seam is now caught, where a window
    could only notice the two endpoints drifting apart.
    """
    """``REPLACES`` must name the call sites ``driver.step`` actually runs there.

    Read off the driver source rather than remembered: the whole product is a
    claim about a specific run of consecutive passes, and a driver that grew a
    sixth pass in this seam would leave this kernel silently swallowing it.
    """
    from meep_gpu import fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415

    opens, fill, closes = "step_B", "fill_B", "update_H"
    # The consults this seam owns: its curl slot, its fill slot, the far-ghost
    # pass that fill slot's second half stands in for, and the constitutive slot
    # that closes it. Every other name in DRIVER_SLOTS is foreign to this seam.
    owned = {opens, fill, closes, fastpath.FAR_FILL_PASSES[fill]}
    consults = tuple(fastpath.DRIVER_SLOTS) + tuple(fastpath.FAR_FILL_PASSES.values())
    foreign = [name for name in consults if name not in owned]

    # A CONSULT MAY BE SPELLED AS A NAME RATHER THAN A LITERAL, and one is.
    # ``synchronize_magnetic_fields`` closes this seam through the by-name channel
    # ``fastpath.SYNC_PASS_OWNERS`` opens — ``dispatch(SYNC_UPDATE_H_PASS, ...)`` —
    # so a scan that matched only ``dispatch("update_H"`` reads the half-step as a
    # ``step_B`` with no closer and reports the seam as moved. It has not moved: that
    # site consults the SAME slot under a name a product spanning into the electric
    # half can decline. The alias is DERIVED from ``fastpath`` (the identifier whose
    # value is the pass, and the slot the channel says owns it), never spelled here,
    # so a renamed constant is a failure rather than a silent miss.
    aliases: Dict[str, str] = {}
    for pass_name, owner in fastpath.SYNC_PASS_OWNERS.items():
        aliases[f'dispatch("{pass_name}"'] = owner
        for attribute in dir(fastpath):
            if getattr(fastpath, attribute, None) == pass_name:
                aliases[f"dispatch({attribute}"] = owner

    def consulted(item: str) -> Optional[str]:
        for spelling, owner in aliases.items():
            if spelling in item:
                return owner
        return next((name for name in consults if f'dispatch("{name}"' in item), None)

    text = open(os.path.join(API_ROOT, "meep_gpu", "driver.py"),
                encoding="utf-8").read().splitlines()
    findings: List[str] = []
    windows: List[Dict[str, Any]] = []
    for index, line in enumerate(text):
        if f'dispatch("{opens}"' not in line:
            continue
        end: Optional[int] = None
        trespass: Optional[Tuple[int, str]] = None
        for offset, item in enumerate(text[index + 1:], start=1):
            name = consulted(item)
            if name is None:
                continue
            if name == closes:
                end = offset
                break
            if name not in owned:
                trespass = (index + 1 + offset, name)
                break
        if end is None:
            if trespass is not None:
                findings.append(
                    f"driver.py:{index + 1} dispatches {opens} and reaches the "
                    f"{trespass[1]} consult at driver.py:{trespass[0]} before any "
                    f"{closes}; the seam this product spans is not where it was")
            else:
                findings.append(
                    f"driver.py:{index + 1} dispatches {opens} with no {closes} "
                    f"consult after it; the seam this product spans is not where "
                    f"it was")
            continue
        body = text[index:index + end + 1]
        calls = [name for name in
                 ("fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B")
                 if any(name + "(" in item for item in body)]
        injects = any("source.inject" in item for item in body)
        windows.append({"driver_line": index + 1, "passes_between": calls,
                        "closes_at_line": index + 1 + end,
                        "injects_a_source": injects})
        expected = [name for name in product.REPLACES
                    if name not in ("step_B", "update_H")]
        if calls != expected:
            findings.append(
                f"driver.py:{index + 1} runs {calls} between step_B and update_H, "
                f"but the product declares it replaces {expected}")
        if not injects:
            findings.append(
                f"driver.py:{index + 1} shows no source injection in this seam; the "
                f"predicate's magnetic-source refusal would be refusing nothing")
    if not windows:
        findings.append("no step_B/update_H seam was found in driver.py at all")
    return {"leg": "seam_binding", "device": False, "windows": windows,
            "declared_replaces": list(product.REPLACES),
            "seam_owns": sorted(owned), "foreign_consults": foreign,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec, seed: int,
                 electric: bool):
    """One complex-storage PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, deliberate
        boundaries=boundaries, k_point=k_point, prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    if electric:
        # An ELECTRIC source is admitted: the driver injects it in the D/E seam,
        # not this one. Carrying one is what stops the source clause from being
        # tested only in its refusing direction.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        host = (rng.uniform(-0.25, 0.25, size=shape)
                + 1j * rng.uniform(-0.25, 0.25, size=shape)).astype(np.complex64)
        driver.set_field(name, cp.asarray(np.ascontiguousarray(host)))
    for name in ("f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            host = (rng.uniform(-0.05, 0.05, size=shape)
                    + 1j * rng.uniform(-0.05, 0.05, size=shape)).astype(np.complex64)
            array[...] = cp.asarray(np.ascontiguousarray(host))
    return driver


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing."""
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
    """The plan bundle installed for one driver, keyed by driver call site."""

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


def separate_route(driver, probe):
    """The two separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    curl = complex_fields.plan_complex_pml_curl(driver.fields, driver.pml, "step_B",
                                                probe=probe)
    constitutive = complex_fields.plan_complex_constitutive(
        driver.fields, driver.pml, "H", probe=probe)
    missing = [name for name, plan in
               (("complex curl", curl), ("complex constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # The three seam passes stay on the ARRAY PATH here: no Triton product owns the
    # wall clear, and both symmetry fills are no-ops on every admitted case. All
    # three are counted, so a difference in which of them ran is a counter event
    # rather than an invisible correction.
    return Route({"step_B": curl, "update_H": constitutive})


def fused_route(plan):
    return Route({name: (plan if name == "step_B" else _Absorbed(name, plan))
                  for name in SEAM_PASSES})


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, dimensions, cell, boundaries, k_point, pml_spec,
            steps: int, product, probe, mutant: Any = None,
            install_fused: bool = True, freeze_magnetic: bool = False,
            electric: bool = True, expansion: Optional[int] = None
            ) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             SEED, electric)
    separate = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                            SEED, electric)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         SEED, electric)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.complex_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": str(boundaries),
        "k_point": list(k_point), "pml": dict(pml_spec),
        "electric_source": bool(electric),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "expansion_override": expansion,
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_complex_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            verdict = product.complex_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            # THE ARM, OVERRIDDEN. Not a source rewrite: the constexpr IS the
            # licence, so flipping it here measures whether the licence is
            # load-bearing rather than decorative.
            plan.expansion = int(expansion)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        route = fused_route(plan) if install_fused else Route({})
        separate_plans = separate_route(separate, probe)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_magnetic:
            # ARMED: every route's magnetic seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
            frozen = {key: _Absorbed(key, None) for key in SEAM_PASSES}
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
        inspect.getsource(product.complex_fused_curl_constitutive_B.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    The three complex-multiply helpers are IMPORTED into the mutant's namespace
    rather than copied, for the same reason the shipped module imports them: a
    second spelling of the arm is a second place for it to be wrong, and a mutant
    carrying its own copy would no longer be testing the shipped multiply.
    """
    header = ("import triton\n"
              "import triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import (\n"
              "    _rotate_field_left, _mul_field_left, _mul_coefficient_left)\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_complex_pair.py", delete=False, encoding="utf-8")
    handle.write(header + source.replace("complex_fused_curl_constitutive_B",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_complex_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was MEASURED."""

    def m1_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(B_stuff), undone.

        Visible only where the walls are real, which is what the metallic cases in
        CASES exist for."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"],
            # NO ``if`` IN THE REPLACEMENT. ``_rewrite_block`` indents every
            # replacement line to the indent of the block's FIRST line, so a
            # compound statement in position 0 produces an empty suite and the
            # mutated module fails to IMPORT -- which this harness records as an
            # unarmed mutation, not as a caught defect. Dropping the guard along
            # with the clear is also the truer defect: what is being armed is "the
            # wall clear was not carried at all". The assignments are pure
            # identity, with no arithmetic, so a signed zero survives them.
            ["v0_re = v0_re", "v0_im = v0_im"])

    def m2_wall_clear_only_on_the_real_plane(source: str) -> Tuple[str, int]:
        """THE COMPLEX-SPECIFIC HAZARD. ``array[face] = 0`` on a complex64 volume
        writes complex zero; a transcription that cleared only the real plane is
        wrong exactly where the imaginary part is nonzero at a wall, and nowhere
        else — which is invisible to any real-field test."""
        hits = 0
        for register, coordinate in (("v0_im", "at_x"), ("v1_im", "at_y"),
                                     ("v2_im", "at_z")):
            needle = f"{register} = tl.where({coordinate}, 0.0, {register})"
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = {register}")
        return source, hits

    def m3_phase_applied_to_every_lane(source: str) -> Tuple[str, int]:
        """The Bloch rotation applied everywhere, not only on the wrapped plane.

        ``_apply_bloch_phase`` multiplies ONE plane (S:1767-1771); applying it to
        every lane is the whole-volume version of the same multiply and is caught
        on any phased case."""
        hits = 0
        for register, predicate in (("b_x_re", "wx"), ("b_x_im", "wx"),
                                    ("c_x_re", "wx"), ("c_x_im", "wx")):
            needle = f"{register} = tl.where({predicate}, rot_{register[-2:]}, {register})"
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = rot_{register[-2:]}")
        return source, hits

    def m4_phase_dropped_on_x(source: str) -> Tuple[str, int]:
        """The wrap phase thrown away on x: the Bloch run becomes a plain periodic
        one. Invisible at k = 0 by construction, which is why the case list carries
        two phased rows."""
        return _rewrite_block(
            source,
            ["rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)",
             "b_x_re = tl.where(wx, rot_re, b_x_re)",
             "b_x_im = tl.where(wx, rot_im, b_x_im)"],
            ["b_x_re = b_x_re", "b_x_im = b_x_im"])

    def m5_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation on the real plane: same algebra, different
        float32 rounding."""
        return _rewrite_block(
            source,
            ["t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
             "a_re = a_re + t_re",
             "a_im = a_im + t_im",
             "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
             "a_re = a_re - t_re",
             "a_im = a_im - t_im"],
            ["kp_re, kp_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
             "km_re, km_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
             "a_re = a_re + (kp_re - km_re)",
             "a_im = a_im + (kp_im - km_im)"])

    def m6_history_read_after_write(source: str) -> Tuple[str, int]:
        """f_w read AFTER it is written: wrong only where kms != 0, i.e. in the PML."""
        return _rewrite_block(
            source,
            ["prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)",
             "prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)",
             "src_re = v0_re",
             "src_im = v0_im",
             "tl.store(w0 + 2 * idx, src_re, mask=live)",
             "tl.store(w0 + 2 * idx + 1, src_im, mask=live)"],
            ["src_re = v0_re",
             "src_im = v0_im",
             "tl.store(w0 + 2 * idx, src_re, mask=live)",
             "tl.store(w0 + 2 * idx + 1, src_im, mask=live)",
             "prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)",
             "prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)"])

    def m7_word_index_not_doubled(source: str) -> Tuple[str, int]:
        """The complex word addressing undone on the H history: ``w0 + idx`` reads
        the wrong plane of the wrong cell. The load and the store move TOGETHER, so
        the mutant is self-consistent and only the array-path comparison refuses
        it."""
        hits = 0
        for needle, replacement in (
                ("tl.load(w0 + 2 * idx, mask=live, other=0.0)",
                 "tl.load(w0 + idx, mask=live, other=0.0)"),
                ("tl.store(w0 + 2 * idx, src_re, mask=live)",
                 "tl.store(w0 + idx, src_re, mask=live)")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m8_imaginary_plane_of_the_weld_dropped(source: str) -> Tuple[str, int]:
        """The weld carried on the real plane only: the imaginary source is zero.

        THE OLD REWRITE COULD NOT EXPRESS THIS, and was measured armed-and-uncaught
        on the GPU host 2026-08-20 for a reason that is structural rather than about
        the kernel. It replaced ``src_im = v{c}_im`` with a load of
        ``f{c} + 2*idx + 1`` -- meaning to read the PRE-STEP imaginary B -- but the
        fused kernel STORES the new value at :481-482 and welds at :506-507, in
        that order. The load therefore returned exactly ``v{c}_im`` and the mutant
        was bit-identical to the original BY CONSTRUCTION: a guaranteed no-op
        reported as an inert defect, which is the most expensive kind of green.

        Zeroing the imaginary source is what the mutation's own name says and is
        visible wherever the imaginary plane is live -- which the Bloch cases make
        it, and the k = 0 seeded-complex cases are the control for.
        """
        hits = 0
        for component in ("0", "1", "2"):
            needle = f"src_im = v{component}_im"
            hits += source.count(needle)
            source = source.replace(needle, "src_im = 0.0")
        return source, hits

    def m9_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes on the plane-wise adds; only
        the PTX may move. It is the control that says a caught mutation means
        something."""
        needle = "r_re = (r_re + n0_re) - p0_re"
        return source.replace(needle, "r_re = (n0_re + r_re) - p0_re"), \
            source.count(needle)

    return (
        ("m1_wall_clear_dropped", "the wall clear's slot", "caught",
         m1_wall_clear_dropped),
        ("m2_wall_clear_only_on_the_real_plane",
         "complex zero is TWO words", "caught",
         m2_wall_clear_only_on_the_real_plane),
        ("m3_phase_applied_to_every_lane", "the single-plane multiply", "caught",
         m3_phase_applied_to_every_lane),
        ("m4_phase_dropped_on_x", "the Bloch wrap", "caught", m4_phase_dropped_on_x),
        ("m5_constitutive_association", "float32 association", "caught",
         m5_constitutive_association),
        ("m6_history_read_after_write", "the f_w ordering", "caught",
         m6_history_read_after_write),
        ("m7_word_index_not_doubled", "complex word addressing", "caught",
         m7_word_index_not_doubled),
        ("m8_imaginary_plane_of_the_weld_dropped", "the weld is two registers",
         "caught", m8_imaginary_plane_of_the_weld_dropped),
        ("m9_commuted_multiply", "commuted add", "null", m9_commuted_multiply),
    )


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#:
#: THERE IS NO SINGLE CASE THAT SERVES THEM ALL, and the code used to claim there
#: was: every mutation ran on CASES[4] (bloch_x_metallic_y) under the comment "the
#: only one in CASES where every mutation above has somewhere to be caught". That
#: case declares x PERIODIC, so `if ZM_X:` is never entered and
#: m1_wall_clear_dropped rewrote lines that never executed — measured UNCAUGHT on
#: 2026-08-20 while measuring nothing. The two requirements genuinely conflict: a
#: wall-clear mutation needs a METALLIC x, and a phase mutation needs a NONZERO
#: k, and CASES carries no grid with both. Naming the case per mutation is what
#: resolves that; asserting one case covers everything is what hid it.
MUTATION_CASE: Dict[str, int] = {
    # metallic_walls_k0 — x IS metallic there, which is what ZM_X needs.
    "m1_wall_clear_dropped": 3,
}

#: The phased, walled case: the default for every mutation that does not need a
#: metallic x. It carries a nonzero k, so the Bloch-phase legs have somewhere to
#: be caught.
DEFAULT_MUTATION_CASE = 4


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on."""
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    return index, CASES[index]


def run_mutations(cp, product, probe, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(product)
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "device": True}
        if hits == 0 or mutated == source:
            # A rewrite that matched nothing is a HARNESS defect, not a null: it
            # would report every mutation as uncaught while launching the shipped
            # kernel.
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_complex_fused_B"
        mutant = compile_mutant(mutated, kernel_name)
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3], case[4],
                      case[5], MUTATION_STEPS, product, probe, mutant=mutant)
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

def run_refusals(cp, product, probe) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, asked of a real CuPy driver."""
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    class _Magnetic:
        field_type = FIELD_TYPE_B

    def _deposit(fields, component):
        """A REAL source that publishes the index the injection writes.

        The stub classes in this leg carry a ``field_type`` and nothing else: enough
        to PLACE a source in a seam, and deliberately not enough to CARRY one. The
        carried rows below need a source the repair can actually save, and one that
        deposits nothing would make an admitting row pass by measuring an empty
        scatter, so both are checked here.
        """
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component=component,
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    base = CASES[0]
    rows: List[Dict[str, Any]] = []
    # `sources` may be a CALLABLE taking the built fields, and `needle=None` means
    # "this row must be ADMITTED" — both for the 2026-08-30 carry rows below.
    for name, case, needle, sources, use_probe in (
        ("magnetic_source_without_a_deposit_index", base,
         "does not publish the index it writes", (_Magnetic(),), probe),
        ("magnetic_deposit_is_carried", base, None,
         lambda fields: (_deposit(fields, "Hy"),), probe),
        ("undeclared_sources", base, "was not declared", None, probe),
        ("no_probe_artifact", base, "expansion probe artifact", (), {}),
    ):
        driver = build_driver(cp, case[1], case[2], case[3], case[4], case[5],
                              SEED, electric=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.complex_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, bound, probe=use_probe)
            plan = product.plan_complex_fused_magnetic_pair(
                driver.fields, driver.pml, bound, probe=use_probe)
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
        "expansion_probe_environment_variable": os.environ.get(
            "MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
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
        HERE, "results", "triton_complex_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — "
             "complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY, and the policy the "
             "expansion probe this gate consumes was cut under. A run that "
             "installs NOTHING is a MIXED configuration (CuPy appends -ftz=true "
             "unconditionally; Triton natively keeps) attributable to no policy at "
             "all, and a keep-cut licence read by such a run is a broken "
             "comparison, not a platform verdict.")
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {
        "gate": "triton_complex_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.complex_fused_magnetic_pair",
        "kernel": "complex_fused_curl_constitutive_B",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "complex_fields.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, expansion_licence_leg, symmetry_inertness_leg,
                design_sweep_leg, predicate_leg, seam_binding_leg):
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
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST, and
        # it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device. A
        # laptop leg is evidence about the design, the licence and the predicate; it
        # is not a release, and the record must not read as one.
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

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing — and on this family it
    # is worse than that, because the EXPANSION licence it consumes is
    # policy-conditional and a keep-cut record read under flush licenses nothing.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product  # noqa: PLC0415

    # THE PROBE THE DEVICE LEGS CONSUME, resolved once and RECORDED. The device
    # bytes are a function of the arm, so an artifact that does not name the record
    # that licensed it cannot be read.
    probe = complex_fields.load_expansion_probe()
    licence = complex_fields.expansion_license(probe,
                                               product.PRODUCT_PROBE_PATTERNS)
    payload["expansion"] = {
        "probe_path": os.environ.get("MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
        "arm": licence["arm"], "basis": licence["basis"],
        "expansion": licence["expansion"], "refusals": list(licence["refusals"]),
        "policy_resolved": licence["policy_resolved"],
        "candidate_policy": licence["candidate_policy"],
    }
    if licence["expansion"] is None:
        payload["device_status"] = "REFUSED BEFORE THE FIRST LEG"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "no EXPANSION arm is licensed for this run, so no device leg could "
            "measure anything: " + "; ".join(licence["refusals"])]}
        save(payload, args.out)
        log(f"\nREFUSED: {licence['refusals']}")
        return 1
    log(f"expansion arm licensed: {licence['arm']} (basis {licence['basis']})")

    backends.guard_kernel_compilation(cp)
    payload["environment"] = environment(cp)
    # Stamped BEFORE the first leg so the partial artifact an aborted run leaves
    # behind — which is the artifact a failure is read from — says what it is.
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "byte_cases": {case[0]: case[6] for case in CASES},
        "steps_per_mutation": MUTATION_STEPS,
        "total_steps": (sum(case[6] for case in CASES)
                        + MUTATION_STEPS * (len(mutation_table()) + 3)),
    }
    save(payload, args.out)
    log("\n=== device legs ===")
    for name, dimensions, cell, boundaries, k_point, pml_spec, steps in CASES:
        row = run_leg(cp, name, dimensions, cell, boundaries, k_point, pml_spec,
                      steps, product, probe)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== armed harness mutations ===")
    # The substitution removed: the bytes still agree (the array path is what ran)
    # and ONLY the launch counter can refuse it.
    base = CASES[0]
    row = run_leg(cp, "armed:no_substitution", base[1], base[2], base[3], base[4],
                  base[5], 3, product, probe, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", base[1], base[2], base[3],
                  base[4], base[5], 3, product, probe, freeze_magnetic=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    # THE LICENCE IS LOAD-BEARING, measured. The constexpr is flipped to the arm
    # the platform did NOT measure; if the bytes still agree, the probe machinery
    # is decorative on this kernel and the artifact must say so.
    other = 1 - int(licence["expansion"])
    row = run_leg(cp, "armed:unlicensed_expansion_arm", CASES[4][1], CASES[4][2],
                  CASES[4][3], CASES[4][4], CASES[4][5], 3, product, probe,
                  expansion=other)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = (f"the EXPANSION constexpr is forced to arm {other} while the "
                  f"platform measured {licence['expansion']}; the bytes must "
                  f"diverge, or the licence constrains nothing here")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.complex_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, probe, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product, probe)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])
    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted -- see the sibling
    # comment in probe_triton_folded_fused_magnetic_pair. Under the old rule any
    # expectation other than "caught" passed unconditionally, so moving a
    # stubbornly uncaught mutation to a null spelling would silence it rather
    # than explain it. Every mutation must also have been ARMED: a rewrite that
    # hit nothing measured nothing, whatever it then reported.
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

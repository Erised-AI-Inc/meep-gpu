"""Byte gate for the Metal Dcyl COMPLEX two-launch H->D pair: ``update_H`` folded with the
device radial scan, then the certified complex curl.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.cylindrical_complex_fused_hd_pair.metal_cylindrical_complex_fused_hd_pair_coverage`
admits, ``run()`` — ONE dispatch of ``cyl_complex_hd_lead_step`` (the pointwise complex H
constitutive into scratch plus the column-leader scan of the recomputed ``Hy`` into a
prefix scratch), the H/f_w_H rotation, then ONE dispatch of the CERTIFIED complex
cylindrical curl against the rotated H and the device prefix — leaves the engine in a
state that is BIT-IDENTICAL, PER COMPLETE DRIVER STEP, to

  * the NumPy array path (``stepping.update_H`` then ``stepping.step_D``, inside the
    driver's full pass list), and
  * the two SEPARATELY CERTIFIED Metal products it replaces —
    ``cylindrical_complex.plan_cylindrical_complex_constitutive`` on ``H`` and
    ``cylindrical_complex.plan_cylindrical_complex_pml_curl`` on ``step_D`` with its
    HOST prefix round trip,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere.

WHAT IS NEW AND MEASURED HERE, beyond the two pieces measured elsewhere (the scan against
``stepping.cylindrical_rderiv_prefix`` in ``gate_metal_cylindrical_complex_scan``; the
lifted constitutive body being the certified text): the SEAM between them — that the
column leader's recomputed ``Hy`` at every row is bit-identical to what the thread at
that row stored, so the prefix the curl reads IS the prefix over the post-``update_H``
field. Leg ``launch_structure`` measures it directly, and mutation
``leader_scans_pre_launch_hy`` is the defect it exists to catch.

THE LEGS

  1  binding_ceiling   the lead signature (25 bindings) COMPILES on both expansion arms;
                       lead + 6 (31) compiles and lead + 7 (32) is refused with the
                       platform's own message; the generic sweep pins the ceiling; all
                       six certified curl specialisations compile
  2  lift              the h_cell body is the certified complex H constitutive with
                       EXACTLY the declared edits and its arithmetic lines untouched; the
                       lead source carries the scan module's loop and divide helper
                       verbatim; launch 2's source IS the certified curl's, byte for byte
  3  driver_order      the only statement between the update_H and step_D consults is
                       the electric withdraw loop (h_to_d_seam.driver_seam_fact), and the
                       product's REPLACES / SEAM are the driver's span and seam name
  4  refusal           a standing integrated electric withdraw is refused BY NAME; a
                       non-integrated electric source and a magnetic integrated one are
                       admitted; an undeclared source set, a Cartesian grid, real storage
                       and a missing residency are refused
  5  product           six configurations, twelve complete driver steps each, against
                       the array path AND the certified singles, launch counts asserted
                       per cycle, Residency.verify() empty after every step
  6  value_class       the +-0 lattice, with the sign floor in place of the movement one
  7  launch_structure  per step: two launches (one lead, one curl); the prefix scratch
                       equals stepping.cylindrical_rderiv_prefix over the device's own
                       post-update Hy; the plan performs NO host sync
  8  purity            a static read/write ledger over the lead source: nothing the
                       launch writes is read by it
  9  sync              the singles' walk pays one prefix sync per step; this product's
                       walk pays none
 10  mutations         six kernel defects and three host defects through the
                       launch-counted harness, each must-catch or a measured equivalence,
                       plus a byte-neutral control that must NOT diverge
 11  disarm            the shipped bytes through the mutation path must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import h_to_d_seam  # noqa: E402
import metal_composition_matrix as matrix  # noqa: E402
import metal_gate_kit as kit  # noqa: E402
import metal_value_classes as values  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields,
    cylindrical_complex as cylc,
    cylindrical_complex_fused_hd_pair as family,
    cylindrical_complex_scan as scan,
    launch as metal_launch,
    shaders,
    subnormal,
    templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source,
)

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words

#: Twelve complete steps, the budget every fused-pair gate on this backend runs: the
#: classes this gate exists for COMPOUND (f_w_H and fu_D are state), and a wrong
#: coefficient index needs several steps to reach the low bits of the interior.
STEPS = 12

#: Every volume one complete step can touch on a complex Dcyl run.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's complete step (driver.py:3279-3330), in order. The two passes this
#: product replaces are marked; the rest run on the HOST in every engine.
PASSES: Tuple[Tuple[str, Callable[..., Any], bool], ...] = (
    ("step_B", lambda f, p: stepping.step_B(f, p), False),
    ("fill_B", lambda f, p: stepping.fill_symmetry_bc_B(f), False),
    ("zero_metal_B", lambda f, p: stepping.zero_metal_B(f), False),
    ("fill_folded_far_ghosts_B", lambda f, p: stepping.fill_folded_far_ghosts_B(f), False),
    ("update_H", lambda f, p: stepping.update_H(f, p), True),
    ("step_D", lambda f, p: stepping.step_D(f, p), True),
    ("fill_D", lambda f, p: stepping.fill_symmetry_bc_D(f), False),
    ("zero_metal_D", lambda f, p: stepping.zero_metal_D(f), False),
    ("fill_folded_far_ghosts_D", lambda f, p: stepping.fill_folded_far_ghosts_D(f), False),
    ("update_E", lambda f, p: stepping.update_E(f, p), False),
)

#: The case matrix: (label, m, z kind, accurate near-axis rows). Both curl arms (m = 0,
#: |m| = 1, |m| >= 2), both z kinds, both signs of m, and the accurate branch at a
#: Courant number Grid accepts for it.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m1_z_metallic", {"m": 1, "z_kind": "metallic"}),
    ("m0_z_metallic", {"m": 0, "z_kind": "metallic"}),
    ("m2_z_periodic", {"m": 2, "z_kind": "periodic"}),
    ("m_minus1_z_periodic", {"m": -1, "z_kind": "periodic"}),
    ("m1_z_periodic", {"m": 1, "z_kind": "periodic"}),
    ("m3_z_metallic_accurate", {"m": 3, "z_kind": "metallic", "accurate": True,
                                "courant": 0.25}),
)
MUTATION_CASE = "m1_z_metallic"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build(label: str, seed: int, value_class: str = values.UNIFORM) -> Tuple[Any, Any]:
    """The composition matrix's Dcyl complex fixture, with the auxiliaries seeded too.

    ``matrix.cylindrical`` fills the twelve field volumes; the split-field and
    auxiliary volumes are STATE and are seeded here, because a zero auxiliary is a
    fixed point of the recurrence for one step and would hide a wrong coefficient.
    """
    keywords = dict(dict(CASES)[label])
    fields, pml = matrix.cylindrical(m=keywords["m"], complex_storage=True,
                                     z_kind=keywords["z_kind"],
                                     accurate=bool(keywords.get("accurate", False)),
                                     courant=float(keywords.get("courant", 0.37)))
    rng = np.random.default_rng(seed)
    if value_class == values.PM_ZERO_LATTICE:
        for name in STATE:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = values.pm_zero_lattice_complex(array.shape)
        return fields, pml
    for name in ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.uniform(-0.3, 0.3, array.shape)
                          + 1j * rng.uniform(-0.3, 0.3, array.shape)).astype(array.dtype)
    return fields, pml


def state_of(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in left:
        count = differing(left[name], right[name])
        if count:
            out[name] = count
    return out


def state_census(fields: Any) -> int:
    return sum(int(subnormal.census(array)) for array in state_of(fields).values())


def volume_source(fields: Any, component: str, integrated: bool) -> Any:
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.15, 0.0, 0.05), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


# ---------------------------------------------------------------------------
# The three engines
# ---------------------------------------------------------------------------

class ArrayEngine:
    """The NumPy array path, the driver's pass list, no device."""

    def __init__(self, fields: Any, pml: Any) -> None:
        self.fields, self.pml = fields, pml

    def step(self) -> None:
        for _name, function, _replaced in PASSES:
            function(self.fields, self.pml)


class ProductEngine:
    """The device engine: host passes everywhere except the seam, which the plan owns."""

    def __init__(self, fields: Any, pml: Any, plan_functions: Optional[Dict[str, Any]] = None,
                 curl_functions: Optional[Dict[str, Any]] = None,
                 order: str = "lead_then_curl", rotate: bool = True) -> None:
        self.fields, self.pml = fields, pml
        self.residency = Residency()
        self.plan = family.plan_metal_cylindrical_complex_fused_hd_pair(
            fields, pml, sources=(), residency=self.residency,
            lead_functions=plan_functions, curl_functions=curl_functions)
        if self.plan is None:
            raise AssertionError("the product refused an admitted configuration")
        self.order = order
        self.rotate = rotate
        self.residency.sync_in()

    def run_seam(self) -> None:
        if self.order == "lead_then_curl":
            self.plan.run()
        elif self.order == "curl_then_lead":
            self.plan.run_curl()
            self.plan.run_lead()
        elif self.order == "curl_only":
            self.plan.run_curl()
        else:
            raise ValueError(self.order)
        if not self.rotate:
            # THE NO-ROTATION DEFECT: put the engine's references back on the
            # pre-launch buffers, so the next curl and the host read stale H.
            for name in family.ROTATED_NAMES:
                current = getattr(self.fields, name)
                setattr(self.fields, name, self.plan.rotated[name])
                self.plan.rotated[name] = current

    def step(self) -> Tuple[Tuple[int, int], Dict[str, int]]:
        """One complete step; returns the sync counters the SEAM consumed and the
        residency invariant measured IMMEDIATELY after the seam's sync_out — before
        the host wall clear rewrites D on the host, which is the array path's own
        write and not a stale mirror."""
        seam_done = False
        inside = (0, 0)
        stale: Dict[str, int] = {}
        for _name, function, replaced in PASSES:
            if replaced:
                if not seam_done:
                    self.residency.sync_in()
                    before = (self.residency.syncs_out, self.residency.syncs_in)
                    self.run_seam()
                    inside = (self.residency.syncs_out - before[0],
                              self.residency.syncs_in - before[1])
                    self.residency.sync_out()
                    stale = self.residency.verify()
                    seam_done = True
                continue
            function(self.fields, self.pml)
        return inside, stale


class SinglesEngine:
    """The two separately certified Metal products, with the curl's host prefix scan."""

    def __init__(self, fields: Any, pml: Any) -> None:
        self.fields, self.pml = fields, pml
        self.residency = Residency()
        self.h_plan = cylc.plan_cylindrical_complex_constitutive(
            fields, pml, "H", self.residency)
        self.d_plan = cylc.plan_cylindrical_complex_pml_curl(
            fields, pml, "step_D", self.residency)
        if self.h_plan is None or self.d_plan is None:
            raise AssertionError("a certified single refused an admitted configuration")
        self.residency.sync_in()

    def step(self) -> Tuple[Tuple[int, int], Dict[str, int]]:
        seam_done = False
        inside = (0, 0)
        stale: Dict[str, int] = {}
        for _name, function, replaced in PASSES:
            if replaced:
                if not seam_done:
                    self.residency.sync_in()
                    before = (self.residency.syncs_out, self.residency.syncs_in)
                    self.h_plan.run()
                    self.d_plan.run()
                    inside = (self.residency.syncs_out - before[0],
                              self.residency.syncs_in - before[1])
                    self.residency.sync_out()
                    stale = self.residency.verify()
                    seam_done = True
                continue
            function(self.fields, self.pml)
        return inside, stale


# ---------------------------------------------------------------------------
# LEG binding_ceiling
# ---------------------------------------------------------------------------

def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, (message.splitlines()[0] if message else "")


def _sweep_source(pointers: int) -> str:
    lines = [f"    device float* p{n} [[buffer({n})]]," for n in range(pointers)]
    touch = " + ".join(f"p{n}[0]" for n in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n_elem; float dtdx; };", "",
        "kernel void ceiling_sweep(", *lines,
        f"    constant Params& prm [[buffer({pointers})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float touch = {touch};",
        *(f"    p{n}[idx] = p{n}[idx] + touch * prm.dtdx;" for n in range(pointers)),
        "}", ""))


def bound_expansion() -> Tuple[Optional[str], str]:
    probe = cylc.load_expansion_probe()
    arm = cylc.expansion_from_probe(probe) if probe else None
    return arm, (arm or "FMA_V1")


def leg_binding_ceiling(payload: Dict[str, Any], out: str) -> None:
    probe_arm, expansion = bound_expansion()
    headroom = MAX_BUFFER_BINDINGS - family.LEAD_BINDINGS
    row: Dict[str, Any] = {
        "declared_max_buffer_bindings": MAX_BUFFER_BINDINGS,
        "lead_bindings_declared": family.LEAD_BINDINGS,
        "lead_pointers_declared": family.LEAD_POINTERS,
        "curl_bindings": family.CURL_BINDINGS,
        "probe_bound_expansion": probe_arm,
        "lead": {}, "lead_plus": {}, "generic_sweep": {}, "curl": {},
    }
    for arm in templates.EXPANSION_ARMS:
        ok, error = _compiles(family.lead_source(arm))
        row["lead"][arm] = {"text_bindings": family.lead_signature_bindings(arm),
                            "compiled": ok, "error": error}
        log(f"[binding_ceiling] lead {arm} bindings={row['lead'][arm]['text_bindings']} "
            f"compiled={ok}")
    for extra in range(0, headroom + 2):
        ok, error = _compiles(family.refuted_lead_plus_source(extra, expansion))
        row["lead_plus"][str(family.LEAD_BINDINGS + extra)] = {"compiled": ok,
                                                                "error": error}
        log(f"[binding_ceiling] lead+{extra} bindings={family.LEAD_BINDINGS + extra} "
            f"compiled={ok} {error[:80]}")
    for pointers in range(MAX_BUFFER_BINDINGS - 3, MAX_BUFFER_BINDINGS + 2):
        ok, _error = _compiles(_sweep_source(pointers))
        row["generic_sweep"][f"{pointers}_pointers_{pointers + 1}_bindings"] = ok
    for bcz in (templates.PERIODIC, templates.METALLIC):
        for m_arm in cylc.M_ARMS:
            ok, error = _compiles(family.curl_source(bcz, m_arm, expansion))
            row["curl"][f"bcz{bcz}/{cylc.M_ARM_LABELS[m_arm]}"] = {"compiled": ok,
                                                                    "error": error}
    largest = max((int(k.split("_")[0]) for k, ok in row["generic_sweep"].items() if ok),
                  default=-1)
    smallest = min((int(k.split("_")[0]) for k, ok in row["generic_sweep"].items()
                    if not ok), default=-1)
    row["ceiling_measured"] = largest + 1
    row["ceiling_measured_equals_declared"] = (largest + 1 == MAX_BUFFER_BINDINGS
                                               and smallest == largest + 1)
    at_ceiling = row["lead_plus"][str(MAX_BUFFER_BINDINGS)]
    over = row["lead_plus"][str(MAX_BUFFER_BINDINGS + 1)]
    row["lead_headroom_measured"] = headroom
    row["passed"] = bool(
        all(v["compiled"] for v in row["lead"].values())
        and all(v["text_bindings"] == family.LEAD_BINDINGS for v in row["lead"].values())
        and all(v["compiled"] for k, v in row["lead_plus"].items()
                if int(k) <= MAX_BUFFER_BINDINGS)
        and at_ceiling["compiled"] and not over["compiled"]
        and "out of bounds" in over["error"] and "buffer" in over["error"]
        and row["ceiling_measured_equals_declared"]
        and all(v["compiled"] for v in row["curl"].values()))
    payload["legs"]["binding_ceiling"] = row
    save(payload, out)
    log(f"[binding_ceiling] ceiling={row['ceiling_measured']} headroom={headroom} "
        f"passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG lift
# ---------------------------------------------------------------------------

def _arithmetic_lines(text: str) -> List[str]:
    """The lines that decide a bit: every ``c_mul_coefficient_left`` accumulation and
    the coefficient loads, comments stripped."""
    kept = []
    for line in text.splitlines():
        code = line.split("//", 1)[0].strip()
        if ("c_mul_coefficient_left" in code or code.startswith("float kp_")):
            kept.append(code)
    return kept


def leg_lift(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    certified = complex_fields.bloch_constitutive_source("H", expansion)
    tail = family.certified_complex_constitutive_tail(expansion)
    certified_tail = certified.split(family.DECODE_END, 1)[1][: -len("}\n")]
    # Re-derive the lift from the declared edits and require equality.
    rederived = certified_tail
    for target in range(3):
        for old, new in (
                (f"    float2 prev{target} = w{target}[ii];\n",
                 f"    float2 prev{target} = wi{target}[ii];\n"),
                (f"    float2 src{target} = g{target}[ii];\n",
                 f"    float2 src{target} = b{target}[ii];\n"),
                (f"    w{target}[ii] = src{target};\n", ""),
                (f"    float2 a{target} = f{target}[ii];\n",
                 f"    float2 a{target} = hi{target}[ii];\n"),
                (f"    f{target}[ii] = a{target};\n", "")):
            assert rederived.count(old) == 1, old
            rederived = rederived.replace(old, new)
    lead = family.lead_source(expansion)
    row = {
        "expansion": expansion,
        "lift_equals_declared_edits": rederived == tail,
        "arithmetic_lines_identical": (_arithmetic_lines(certified_tail)
                                       == _arithmetic_lines(tail)),
        "arithmetic_line_count": len(_arithmetic_lines(tail)),
        "lead_carries_certified_helpers": templates.complex_helpers(expansion) in lead,
        "lead_carries_scan_divide_helper": scan._DIVIDE_HELPER.strip() in lead,
        "lead_carries_scan_loop": all(
            line.strip() in lead for line in scan.COLUMN_SCAN_LOOP.splitlines()
            if line.strip() and "__SRC" not in line),
        "lead_leader_row_zero_is_own_cell": "c_mul_field_left(own.a1, weights[0])" in lead,
        "lead_leader_row_i_is_recompute": (
            f"c_mul_field_left(h_cell(i, j, k, {family.H_CELL_TAIL_ARGS}).a1, weights[i])"
            in lead),
        "lead_h_cell_returns": all(f"out.{reg} = {reg};" in lead for reg in
                                   ("a0", "a1", "a2", "src0", "src1", "src2")),
        "contraction_guard_once": lead.count(
            shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1,
        "curl_is_certified_byte_for_byte": {},
    }
    for bcz in (templates.PERIODIC, templates.METALLIC):
        for m_arm in cylc.M_ARMS:
            ours = family.curl_source(bcz, m_arm, expansion)
            theirs = cylc.cylindrical_curl_source(bcz, True, m_arm, expansion)
            row["curl_is_certified_byte_for_byte"][
                f"bcz{bcz}/{cylc.M_ARM_LABELS[m_arm]}"] = (
                hashlib.sha256(ours.encode()).hexdigest()
                == hashlib.sha256(theirs.encode()).hexdigest())
    row["passed"] = bool(
        row["lift_equals_declared_edits"] and row["arithmetic_lines_identical"]
        and row["arithmetic_line_count"] == 9
        and row["lead_carries_certified_helpers"] and row["lead_carries_scan_divide_helper"]
        and row["lead_carries_scan_loop"] and row["lead_leader_row_zero_is_own_cell"]
        and row["lead_leader_row_i_is_recompute"] and row["lead_h_cell_returns"]
        and row["contraction_guard_once"]
        and all(row["curl_is_certified_byte_for_byte"].values()))
    payload["legs"]["lift"] = row
    save(payload, out)
    log(f"[lift] passed={row['passed']} arithmetic_lines={row['arithmetic_line_count']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG driver_order
# ---------------------------------------------------------------------------

def leg_driver_order(payload: Dict[str, Any], out: str) -> None:
    fact = h_to_d_seam.driver_seam_fact()
    row = {
        "driver_seam_fact": fact,
        "replaces_is_the_span": tuple(family.REPLACES) == tuple(withdraw_hoist.SEAM_SPAN),
        "seam_is_the_hoist_seam": family.SEAM == withdraw_hoist.SEAM,
        "fused_pair_seams_row": list(metal_launch.FUSED_PAIR_SEAMS["update_H"]),
        "fused_pair_seams_routes_to_hoist": (
            metal_launch.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)),
        "between_is_the_withdraw_loop_only": all(
            "withdraw" in entry["statement"] or "for source in electric" in entry["statement"]
            or "update_H(self.fields, self.pml)" in entry["statement"]
            for entry in fact["between"]),
        "launches_per_run": family.MetalCylindricalComplexFusedHdPairPlan.launches_per_run,
        "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
        "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
        "installable": family.INSTALLABLE,
    }
    row["passed"] = bool(row["replaces_is_the_span"] and row["seam_is_the_hoist_seam"]
                         and row["fused_pair_seams_routes_to_hoist"]
                         and row["between_is_the_withdraw_loop_only"]
                         and row["launches_per_run"] == 2
                         and not row["carries_deposit_repair"]
                         and not row["hoists_the_withdraw"] and not row["installable"])
    payload["legs"]["driver_order"] = row
    save(payload, out)
    log(f"[driver_order] consults {fact['update_H_consult']} -> {fact['step_D_consult']} "
        f"passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG refusal
# ---------------------------------------------------------------------------

def leg_refusal(payload: Dict[str, Any], out: str) -> None:
    fields, pml = build(MUTATION_CASE, 1)
    residency = Residency()
    cover = family.metal_cylindrical_complex_fused_hd_pair_coverage

    def named(verdict: Any, *needles: str) -> bool:
        return all(any(needle in reason for reason in verdict.reasons) for needle in needles)

    admitted = cover(fields, pml, (), residency)
    integrated = cover(fields, pml, (volume_source(fields, "Ez", True),), residency)
    plain = cover(fields, pml, (volume_source(fields, "Ez", False),), residency)
    magnetic = cover(fields, pml, (volume_source(fields, "Hz", True),), residency)
    undeclared = cover(fields, pml, None, residency)
    no_residency = cover(fields, pml, (), None)
    cart_fields, cart_pml = matrix.cart(complex_storage=True)
    cartesian = cover(cart_fields, cart_pml, (), Residency())
    real_fields, real_pml = matrix.cylindrical(m=0, complex_storage=False)
    real = cover(real_fields, real_pml, (), Residency())
    row = {
        "admitted_with_empty_sources": admitted.covered,
        "integrated_electric_refused_by_name": (
            not integrated.covered
            and named(integrated, "standing", "HOISTS_THE_WITHDRAW = False")),
        "integrated_electric_reasons": list(integrated.reasons),
        "non_integrated_electric_admitted": plain.covered,
        "magnetic_integrated_admitted": magnetic.covered,
        "undeclared_sources_refused": (not undeclared.covered
                                       and named(undeclared, "not declared")),
        "no_residency_refused": not no_residency.covered
                                and named(no_residency, "residency"),
        "cartesian_refused": not cartesian.covered,
        "cartesian_reasons_head": list(cartesian.reasons)[:3],
        "real_storage_refused": not real.covered,
        "real_storage_reasons_head": list(real.reasons)[:3],
    }
    row["passed"] = bool(row["admitted_with_empty_sources"]
                         and row["integrated_electric_refused_by_name"]
                         and row["non_integrated_electric_admitted"]
                         and row["magnetic_integrated_admitted"]
                         and row["undeclared_sources_refused"]
                         and row["no_residency_refused"]
                         and row["cartesian_refused"] and row["real_storage_refused"])
    payload["legs"]["refusal"] = row
    save(payload, out)
    log(f"[refusal] passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG product / value_class / launch_structure / sync
# ---------------------------------------------------------------------------

def run_case(label: str, seed: int, steps: int,
             value_class: str = values.UNIFORM) -> Dict[str, Any]:
    reference = ArrayEngine(*build(label, seed, value_class))
    product = ProductEngine(*build(label, seed, value_class))
    singles = SinglesEngine(*build(label, seed, value_class))
    plan = product.plan
    per_step: List[Dict[str, Any]] = []
    total_compared = 0
    first_divergence = None
    sign_floor = None
    for step in range(1, steps + 1):
        reference.step()
        product_syncs, stale = product.step()
        singles_syncs, singles_stale = singles.step()
        ref_state = state_of(reference.fields)
        prod_state = state_of(product.fields)
        single_state = state_of(singles.fields)
        versus_array = compare(prod_state, ref_state)
        versus_singles = compare(prod_state, single_state)
        singles_versus_array = compare(single_state, ref_state)
        compared = sum(int(words(array).size) for array in ref_state.values())
        total_compared += compared
        band = state_census(reference.fields)
        assert not singles_stale, (label, step, singles_stale)
        moved = sum(differing(prod_state[name], np.zeros_like(prod_state[name]))
                    for name in prod_state)
        if value_class == values.PM_ZERO_LATTICE and step == 1:
            sign_floor = values.zero_sign_census(list(ref_state.values()))
        # THE SEAM, MEASURED DIRECTLY: the prefix scratch the lead wrote equals the
        # array path's own scan over the DEVICE engine's post-update Hy.
        prefix_expected = stepping.cylindrical_rderiv_prefix(
            np, np.ascontiguousarray(product.fields.Hy), 0.5)
        prefix_differing = differing(plan.prefix_host, prefix_expected)
        entry = {"step": step, "compared": compared,
                 "versus_array": versus_array, "versus_singles": versus_singles,
                 "singles_versus_array": singles_versus_array,
                 "prefix_differing": prefix_differing,
                 "launches": plan.launches, "lead_launches": plan.lead_launches,
                 "curl_launches": plan.curl_launches,
                 "product_seam_syncs": list(product_syncs),
                 "singles_seam_syncs": list(singles_syncs),
                 "singles_prefix_syncs": singles.d_plan.prefix_syncs,
                 "stale_mirrors": stale, "reference_subnormal_words": band,
                 "moved_words": moved}
        per_step.append(entry)
        if (versus_array or versus_singles) and first_divergence is None:
            first_divergence = step
        assert band == 0, (label, step, "the reference entered the subnormal band")
        assert not stale, (label, step, stale)
        assert (plan.lead_launches, plan.curl_launches, plan.launches) == (
            step, step, 2 * step), (label, step, plan.launches)
        assert product_syncs == (0, 0), (label, step, product_syncs)
        assert singles.d_plan.prefix_syncs == step, (label, step)
        log(f"[case] {label:<26} {value_class:<16} step {step:>2}/{steps} "
            f"vs_array={versus_array or 0} vs_singles={versus_singles or 0} "
            f"prefix={prefix_differing} launches={plan.launches} "
            f"seam_syncs={product_syncs} singles_syncs={singles_syncs}")
    identical = all(not e["versus_array"] and not e["versus_singles"]
                    and not e["prefix_differing"] for e in per_step)
    if value_class == values.PM_ZERO_LATTICE:
        assert sign_floor and sign_floor["positive_zero_words"] > 0 \
            and sign_floor["negative_zero_words"] > 0, sign_floor
    else:
        kit.assert_moved(per_step[-1]["moved_words"], f"{label} moved nothing", floor=64)
    return {"case": label, "value_class": value_class, "steps": steps,
            "compared_words": total_compared, "bit_identical": identical,
            "first_divergence": first_divergence, "per_step": per_step,
            "shape": list(reference.fields.grid.shape),
            "m": int(reference.fields.grid.m), "bcz": plan.bcz, "m_arm": plan.m_arm,
            "expansion": plan.expansion, "zero_sign_census": sign_floor,
            "digest": kit.state_digest(state_of(reference.fields))}


def leg_product(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for offset, (label, _keywords) in enumerate(CASES):
        rows.append(run_case(label, 61000 + offset, STEPS))
        payload["legs"]["product"] = rows
        save(payload, out)
        log(f"[product] {label:<26} bit_identical={rows[-1]['bit_identical']} "
            f"compared={rows[-1]['compared_words']}")
    assert all(row["bit_identical"] for row in rows), [
        (row["case"], row["first_divergence"]) for row in rows]


def leg_value_class(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for offset, (label, _keywords) in enumerate(CASES[:3]):
        rows.append(run_case(label, 63000 + offset, 4, values.PM_ZERO_LATTICE))
        payload["legs"]["value_class"] = rows
        save(payload, out)
        log(f"[value_class] {label:<26} pm_zero_lattice bit_identical="
            f"{rows[-1]['bit_identical']} census={rows[-1]['zero_sign_census']}")
    assert all(row["bit_identical"] for row in rows), rows


def leg_launch_structure(payload: Dict[str, Any], out: str) -> None:
    """The counters and the seam identity are asserted inside :func:`run_case`; this
    leg gathers them across the product leg and adds the standalone-scan agreement."""
    rows = payload["legs"].get("product", ())
    _probe, expansion = bound_expansion()
    # The prefix the lead wrote must also equal the STANDALONE complex scan kernel run
    # over the stored post-update Hy — the two kernels' spellings of the same loop.
    fields, pml = build(MUTATION_CASE, 2)
    engine = ProductEngine(fields, pml)
    engine.step()
    import torch  # noqa: PLC0415
    nr, ny, nz = fields.grid.shape
    src = torch.from_numpy(np.ascontiguousarray(fields.Hy).reshape(-1)).to("mps")
    weights, divisor = scan.prefix_row_vectors("step_D", nr)
    out_t = torch.from_numpy(np.full((nr, ny, nz), np.complex64(-7.5 - 7.5j)
                                     ).reshape(-1)).to("mps")
    function = scan.compile_cylindrical_complex_prefix("step_D", expansion)
    function(*scan.scan_arguments(out_t, src,
                                  torch.from_numpy(weights).to("mps"),
                                  torch.from_numpy(divisor).to("mps"), (nr, ny, nz)))
    torch.mps.synchronize()
    standalone = out_t.cpu().numpy().reshape(nr, ny, nz)
    row = {
        "per_step_counts_asserted": bool(rows),
        "launches_per_run_declared": family.MetalCylindricalComplexFusedHdPairPlan.launches_per_run,
        "lead_prefix_vs_standalone_scan_differing": differing(engine.plan.prefix_host,
                                                              standalone),
        "lead_prefix_vs_stepping_differing": differing(
            engine.plan.prefix_host,
            stepping.cylindrical_rderiv_prefix(np, np.ascontiguousarray(fields.Hy), 0.5)),
        "prefix_words": int(words(standalone).size),
        "prefix_nonzero_words": differing(standalone, np.zeros_like(standalone)),
    }
    row["passed"] = bool(row["per_step_counts_asserted"]
                         and row["lead_prefix_vs_standalone_scan_differing"] == 0
                         and row["lead_prefix_vs_stepping_differing"] == 0
                         and row["prefix_nonzero_words"] >= 64)
    payload["legs"]["launch_structure"] = row
    save(payload, out)
    log(f"[launch_structure] prefix vs standalone scan differing="
        f"{row['lead_prefix_vs_standalone_scan_differing']} passed={row['passed']}")
    assert row["passed"], row


def leg_purity(payload: Dict[str, Any], out: str) -> None:
    """A static read/write ledger over the lead source, comments stripped."""
    _probe, expansion = bound_expansion()
    source = family.lead_source(expansion)
    body = source.split("kernel void", 1)[1].split("{", 1)[1]
    code = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
    written = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2", "out")
    read = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2",
            "kp0", "km0", "kp1", "km1", "kp2", "km2", "weights", "divisor")
    ledger: Dict[str, Dict[str, int]] = {}
    for name in written + read:
        stores = len(re.findall(rf"\b{name}\[[^\]]*\]\s*=(?!=)", code))
        loads = len(re.findall(rf"\b{name}\[", code)) - stores
        ledger[name] = {"stores": stores, "loads": loads}
    # The h_cell function body (above the kernel) reads only its parameters and
    # writes nothing to device memory.
    helper = source.split("static inline h_cell_result h_cell(", 1)[1].split(
        "\nkernel void", 1)[0]
    helper_code = "\n".join(line.split("//", 1)[0] for line in helper.splitlines())
    helper_stores = len(re.findall(r"\b\w+\[[^\]]*\]\s*=(?!=)", helper_code))
    helper_loads = {name: len(re.findall(rf"\b{name}\[", helper_code)) for name in read}
    row = {
        "ledger": ledger,
        "h_cell_loads": helper_loads,
        "written_never_loaded": all(ledger[n]["loads"] == 0 for n in written),
        "read_never_stored": all(ledger[n]["stores"] == 0 for n in read),
        "every_written_is_stored": all(ledger[n]["stores"] >= 1 for n in written),
        "h_cell_stores_nothing": helper_stores == 0,
        "leader_guard": "if (i == 0) {" in code,
    }
    row["passed"] = bool(row["written_never_loaded"] and row["read_never_stored"]
                         and row["every_written_is_stored"] and row["h_cell_stores_nothing"]
                         and row["leader_guard"])
    payload["legs"]["purity"] = row
    save(payload, out)
    log(f"[purity] passed={row['passed']} ledger={ledger}")
    assert row["passed"], row


def leg_sync(payload: Dict[str, Any], out: str) -> None:
    rows = payload["legs"].get("product", ())
    product_syncs = [tuple(e["product_seam_syncs"]) for row in rows for e in row["per_step"]]
    singles_syncs = [tuple(e["singles_seam_syncs"]) for row in rows for e in row["per_step"]]
    row = {
        "steps_observed": len(product_syncs),
        "product_seam_syncs_out_total": sum(s[0] for s in product_syncs),
        "product_seam_syncs_in_total": sum(s[1] for s in product_syncs),
        "singles_seam_syncs_out_total": sum(s[0] for s in singles_syncs),
        "singles_seam_syncs_in_total": sum(s[1] for s in singles_syncs),
        "finding": ("the certified singles pay one sync_out (Hy) and one sync_in (the "
                    "prefix) per step inside the seam — the complex family's host round "
                    "trip; this product pays none: the prefix never leaves the device"),
    }
    row["passed"] = bool(row["steps_observed"] > 0
                         and row["product_seam_syncs_out_total"] == 0
                         and row["product_seam_syncs_in_total"] == 0
                         and row["singles_seam_syncs_out_total"] == row["steps_observed"]
                         and row["singles_seam_syncs_in_total"] == row["steps_observed"])
    payload["legs"]["sync"] = row
    save(payload, out)
    log(f"[sync] product seam syncs={row['product_seam_syncs_out_total']}/"
        f"{row['product_seam_syncs_in_total']} singles={row['singles_seam_syncs_out_total']}/"
        f"{row['singles_seam_syncs_in_total']} passed={row['passed']}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# LEG mutations
# ---------------------------------------------------------------------------

_LITERAL_LOOP = """static inline float2 c_div_real(float2 z, float d) {
    float rat = 0.0f / d;
    float scl = 1.0f / (d + 0.0f * rat);
    return float2((z.x + z.y * rat) * scl, (z.y - z.x * rat) * scl);
}"""
_SHIPPED_DIVIDE = """static inline float2 c_div_real(float2 z, float d) {
    return z * (1.0f / d);
}"""


def kernel_mutations(expansion: str) -> Tuple[Tuple[str, Callable[[str], str],
                                                    Optional[bool], str], ...]:
    recompute = f"c_mul_field_left(h_cell(i, j, k, {family.H_CELL_TAIL_ARGS}).a1, weights[i])"
    serial = "            acc = acc + inc;\n            out[o] = acc;\n"
    # EIGHT-row tiles, not 32: the fixture has 16 radial rows, and a tile that never
    # closes IS the serial scan (measured: a 32-row tile was inert, 0/3, on this grid).
    blocked = ("            local = local + inc;\n            out[o] = carry + local;\n"
               "            if ((i % 8) == 7) { carry = carry + local; "
               "local = float2(0.0f, 0.0f); }\n")
    return (
        ("true_divide_spelling",
         lambda s: kit.needle(s, "return z * (1.0f / d);", "return z / d;"), True,
         "the real scan's `/` on the complex increment"),
        ("leader_scans_pre_launch_hy",
         lambda s: kit.needle(s, recompute,
                              "c_mul_field_left(hi1[i * nyz + base], weights[i])"), True,
         "the leader scans the PRE-update Hy instead of recomputing the post-update one: "
         "the defect this product's whole seam exists to avoid"),
        ("leader_row_zero_pre_launch",
         lambda s: kit.needle(s, "c_mul_field_left(own.a1, weights[0])",
                              "c_mul_field_left(hi1[ii], weights[0])"), True,
         "row 0 of the scan taken from the pre-launch Hy"),
        ("blocked_scan_8",
         lambda s: kit.needle(
             kit.needle(s, serial, blocked),
             "        float2 acc = float2(0.0f, 0.0f);\n",
             "        float2 acc = float2(0.0f, 0.0f);\n"
             "        float2 carry = float2(0.0f, 0.0f);\n"
             "        float2 local = float2(0.0f, 0.0f);\n"), True,
         "an 8-row tiled scan on the 16-row fixture: a different association, a "
         "different float32 number (a 32-row tile never closes on 16 rows and measured "
         "inert, which is a fixture fact and not a kernel one)"),
        ("literal_numpy_divide_loop",
         lambda s: kit.needle(s, _SHIPPED_DIVIDE, _LITERAL_LOOP), False,
         "numpy's complex divide loop spelled literally — an equivalence on the prefix"),
        ("commuted_accumulator",
         lambda s: kit.needle(s, "acc = acc + inc;", "acc = inc + acc;"), False,
         "float addition is commutative"),
        ("byte_neutral_store_order",
         lambda s: kit.needle(s, "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
                                 "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n",
                              "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n"
                              "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"),
         False, "the two store lines swapped: no arithmetic, the control that shows the "
                "leg does not flag every edit"),
    )


def mutated_walk(label: str, seed: int, steps: int, lead_function: Any = None,
                 order: str = "lead_then_curl", rotate: bool = True) -> Tuple[int, int, int]:
    """(cases diverged, launches, steps run) for one mutant against the array path."""
    reference = ArrayEngine(*build(label, seed))
    fields, pml = build(label, seed)
    counter = kit.Counter(lead_function) if lead_function is not None else None
    engine = ProductEngine(fields, pml,
                           plan_functions=({shaders.CONTRACT_OFF: counter}
                                           if counter is not None else None),
                           order=order, rotate=rotate)
    diverged = 0
    for _step in range(steps):
        reference.step()
        engine.step()
        if compare(state_of(engine.fields), state_of(reference.fields)):
            diverged += 1
    # A KERNEL mutant is counted through its own Counter; a HOST mutant is counted by
    # every kernel launch the mutated protocol issued (`curl_only` issues no lead at
    # all, and its curls ARE the defect running).
    launches = counter.launches if counter is not None else engine.plan.launches
    return diverged, launches, steps


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    harness = kit.MutationHarness(payload, out, key="mutations")
    shipped = family.lead_source(expansion)
    steps = 3
    for label, transform, must_catch, why in kernel_mutations(expansion):
        try:
            mutant = transform(shipped)
        except LookupError:
            harness.record(label, "NEEDLE-MISSED", 0, 0, 0, must_catch, why)
            continue
        assert mutant != shipped, label
        function = getattr(compile_source(mutant), family.LEAD_KERNEL)
        diverged, launches, ran = mutated_walk(MUTATION_CASE, 71000, steps, function)
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches, diverged),
                       launches, diverged, ran, must_catch, why,
                       extra={"kind": "kernel", "steps": steps})
    # HOST DEFECTS: the plan protocol, not the shader.
    for label, order, rotate, why in (
            ("curl_before_lead", "curl_then_lead", True,
             "the curl launched before the lead reads the previous step's H and prefix"),
            ("curl_without_lead", "curl_only", True,
             "the lead hoisted out of the loop: the curl reads a prefix computed once"),
            ("rotation_undone", "lead_then_curl", False,
             "the H/f_w_H rotation undone after the lead: the curl and the host read "
             "the pre-launch H")):
        diverged, launches, ran = mutated_walk(MUTATION_CASE, 73000, steps, None, order,
                                               rotate)
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches, diverged),
                       launches, diverged, ran, True, why,
                       extra={"kind": "host", "steps": steps})


def leg_disarm(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    function = family.compile_lead(expansion)
    diverged, launches, ran = mutated_walk(MUTATION_CASE, 75000, 3, function)
    row = {"diverged": diverged, "launches": launches, "steps": ran,
           "passed": diverged == 0 and launches == ran}
    payload["legs"]["disarm"] = row
    save(payload, out)
    log(f"[disarm] diverged={diverged} launches={launches}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    started = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    payload: Dict[str, Any] = {
        "gate": "metal_cylindrical_complex_fused_hd_pair",
        "family": family.FAMILY,
        "slot": family.SLOT,
        "replaces": list(family.REPLACES),
        "environment": kit.environment_stamp(),
        "probe_environment": ENVIRONMENT,
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    probe_arm, expansion = bound_expansion()
    if probe_arm is None:
        reasons.append("the cylindrical complex expansion probe is not bound; the "
                       "product cannot be built")
    if reasons:
        return kit.cannot_certify(payload, out, reasons)
    payload["expansion"] = {"probe_bound": probe_arm, "measured": expansion}
    save(payload, out)

    legs: List[Tuple[str, Callable[[Dict[str, Any], str], None]]] = [
        ("binding_ceiling", leg_binding_ceiling),
        ("lift", leg_lift),
        ("driver_order", leg_driver_order),
        ("refusal", leg_refusal),
        ("product", leg_product),
        ("value_class", leg_value_class),
        ("launch_structure", leg_launch_structure),
        ("purity", leg_purity),
        ("sync", leg_sync),
        ("mutations", leg_mutations),
        ("disarm", leg_disarm),
    ]
    ran = kit.run_legs(legs, payload, out, kit.wanted_legs(args.legs))

    kit.provenance(
        os.path.dirname(out),
        {"meep_gpu/metal_kernels/cylindrical_complex_fused_hd_pair.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                          "cylindrical_complex_fused_hd_pair.py"),
         "meep_gpu/metal_kernels/cylindrical_complex_scan.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                          "cylindrical_complex_scan.py"),
         "meep_gpu/metal_kernels/cylindrical_complex.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "cylindrical_complex.py"),
         "meep_gpu/metal_kernels/complex_fields.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "complex_fields.py"),
         "meep_gpu/metal_kernels/offdiag_weld_common.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "offdiag_weld_common.py"),
         "meep_gpu/stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
         "meep_gpu/driver.py": os.path.join(API_ROOT, "meep_gpu", "driver.py"),
         "parity/meep_gpu/gate_metal_cylindrical_complex_fused_hd_pair.py":
             os.path.abspath(__file__)},
        kernel_sources={label: source for arm in templates.EXPANSION_ARMS
                        for label, source in family.enumerate_sources(arm).items()},
        name="provenance.json")

    compared = 0
    certified = True
    for key in ("product", "value_class"):
        for row in payload["legs"].get(key, ()):
            compared += int(row["compared_words"])
            if not row["bit_identical"]:
                certified = False
    for key in ("binding_ceiling", "lift", "driver_order", "refusal", "launch_structure",
                "purity", "sync", "disarm"):
        if key in payload["legs"] and not payload["legs"][key].get("passed"):
            certified = False
    mutations = payload["legs"].get("mutations", [])
    armed = [row for row in mutations if row["must_catch"] is True]
    nulls = [row for row in mutations if row["must_catch"] is False]
    for row in armed:
        if row["caught"] != row["ran"] or row["ran"] == 0:
            certified = False
    for row in nulls:
        if row["caught"] != 0:
            certified = False

    return kit.summarize(
        payload, out,
        claim=("two launches — the complex H constitutive folded with the column-leader "
               "device scan of the recomputed Hy, then the certified complex cylindrical "
               "curl against the rotated H and the device prefix — reproduce the array "
               "path AND the two certified singles bit for bit per complete driver step "
               "on every admitted Dcyl complex configuration here, with no host round "
               "trip in the seam"),
        scope=(f"{len(CASES)} configurations x {STEPS} steps (uniform band) + 3 x 4 "
               f"steps (+-0 lattice), all 24 stored volumes; launch counts asserted per "
               f"cycle; the binding ceiling bisected on this host"),
        stated_weakness=(
            "the synthetic fixture (16 x 1 x 20) and not the corpus rows; INSTALLABLE is "
            "False so the product executes nowhere outside this gate and its tests; no "
            "timing; the subnormal band is a refusal on this executor, checked per step; "
            "the E->B cylindrical seam is not touched"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"mutations_armed": len(armed),
               "mutations_caught_of_armed": sum(1 for row in armed
                                                if row["caught"] == row["ran"]),
               "mutations_null_controls": len(nulls),
               "mutation_verdicts": {row["mutation"]: row["verdict"] for row in mutations},
               "expansion": payload.get("expansion")})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))

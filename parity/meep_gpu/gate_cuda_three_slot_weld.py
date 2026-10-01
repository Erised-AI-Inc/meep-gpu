"""The THREE-SLOT weld's device gate: byte identity, launch structure, and the bracket.

=============================================================================
WHAT IS BEING CERTIFIED, AND WHY NEITHER HALF'S VERDICT COVERS IT
=============================================================================

``cuda_three_slot_dispersive_weld`` and ``cuda_three_slot_no_pml_dispersive_weld``
own THREE driver slots -- ``step_D``, ``update_E``, ``update_P`` -- and perform their
work in TWO device groups with their OWN host work between them::

    step_D consult    save(the deposit)            [LeadingRepairPlan]
                      LAUNCH 1: the shipped D->E kernel, whole grid, UNINJECTED D
    (the driver)      inject -> the three in-seam passes
    update_E consult  apply(the repair)            [TrailingRepairPlan]
    update_P consult  LAUNCHES 2..4: one per DRIVEN component, every driving state's
                      ADE recurrence fused into it, host rotation between

Both halves are separately RELEASED -- ``cuda_dispersive_fused_electric_pair_
2026-09-02``, ``cuda_no_pml_dispersive_fused_electric_pair_2026-09-02b``,
``ade_2026-08-19``, ``cuda_(no_pml_)fused_polarization_pair_2026-09-01`` -- and NONE
of those is a verdict about this composition, because none of them has host work
sitting between two device groups that READS the state the second group is about to
advance. ``deposit_repair.apply`` recomputes the constitutive product through
``Fields.displacement_minus_polarization`` (deposit_repair.py:781-782, :922-923),
which subtracts ``state.P[c]`` AS IT STANDS (fields.py:1096-1105), so the repair is
correct only while P is still P^n. That coupling is what this gate measures.

=============================================================================
THE COMPARISONS -- FOUR ENGINES, NOT TWO
=============================================================================

Every product leg compares per COMPLETE DRIVER STEP as uint32 WORDS over every stored
field volume AND every ``P``/``P_prev`` buffer, never ``allclose`` (``-0.0 == 0.0``
lies), against:

  1. the ARRAY PATH (``stepping``'s eleven passes);
  2. the CERTIFIED SINGLES -- ``step_D`` alone, the three in-seam passes, the
     dispersive ``update_E`` alone, the per-(state, component) ADE launches;
  3. the D->E WELD + the certified ADE singles (the composition this product's own
     first half would give);
  4. the certified ``step_D`` + in-seam passes + the E->P WELD (the composition the
     composer installs on these rows TODAY, because ``_later_seam_claimant``
     withholds ``update_E`` from the D->E pair).

All four must agree with the weld word for word. What differs is the LAUNCH COUNT,
and that difference is the entire reason this product exists -- see
:func:`leg_launch_structure`, which reports launches per step for all five
compositions INCLUDING the array path, counted by CUDA graph capture.

=============================================================================
THREE INDEPENDENT LAUNCH COUNTERS
=============================================================================

* the LAUNCHERS' own reports (``run_three_slot_polarization`` returns ``launches``);
* :class:`MemoLaunchCounter`, which wraps the shipped compile-cache memo and so counts
  every launch through every shipped route without editing a byte under ``meep_gpu/``;
* :class:`GraphNodeCounter`, new here: the span is captured into a ``cudaGraph_t``
  with the raw runtime API and its KERNEL NODES are counted. It is the only counter
  on this track that can see a CuPy elementwise launch, which is what makes the array
  path's number measurable rather than modelled. It is FAIL-SOFT by construction --
  capture forbids allocation and synchronization, so a span that cannot be captured
  records why instead of a number.

=============================================================================
THE DEPOSIT LEGS, WITH THREE CONTROLS THAT MUST NOT PASS
=============================================================================

``bracketed``          the composer's own installation. Byte-identical.
``unbracketed``        the same composition with no repair. MUST DIVERGE.
``p_before_repair``    the polarization group moved to the ``update_E`` consult, so
                       ``apply`` reads P^(n+1). MUST DIVERGE -- this is the ordering
                       the off-device probe measured diverging on 6 of 7
                       configurations, and it is the one a reader would reach for.
``wrong_repair_path``  the arm declaring the other recurrence. MUST BE REFUSED BY NAME.

Progress is one flushed line per case and the artifact is rewritten after each leg, so
an interrupted run keeps everything that landed (the progress-reporting rule).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE TWO D->E GATES, IMPORTED FOR THEIR FIXTURES AND COMPARISONS. This product IS
# each of those products with a third slot added, so their build, their pole-aware
# comparison and their movement floors answer the same questions here. Importing them
# keeps ONE copy of each: a second transcription of a seeded engine is a second place
# for a fixture to stop arming what it claims to arm.
import gate_cuda_fused_electric_pair as base  # noqa: E402
import gate_cuda_dispersive_fused_electric_pair as pml_gate  # noqa: E402

try:  # the no-absorber gate imports CuPy at module scope
    import gate_cuda_no_pml_dispersive_fused_electric_pair as no_pml_gate  # noqa: E402
except ImportError:  # a laptop: the SOURCE legs still run, the fixture legs cannot
    no_pml_gate = None

from meep_gpu import deposit_repair, stepping  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census

run_pass = base.run_pass
array_step = base.array_step
needle = base.needle
words = base.words
MemoLaunchCounter = base.MemoLaunchCounter
DRIVER_ORDER = base.DRIVER_ORDER
GUARD_OPTIONS = base.GUARD_OPTIONS
VALUE_CLASSES = base.VALUE_CLASSES

SEED = 20260902

#: Complete DRIVER STEPS every product leg runs.
STEPS = 60

ARMS: Tuple[str, ...] = ("pml", "no_pml")

#: Which D->E gate owns each arm's fixture, comparison and specs. ONE TABLE so a leg
#: cannot ask the PML gate for a no-absorber engine, and a MISSING one is a refusal
#: BY NAME rather than a silently PML fixture answering a no-absorber question.
_GATES = {"pml": pml_gate, "no_pml": no_pml_gate}


class _FixtureGateTable(dict):
    """``_GATE[arm]`` -- the arm's fixture gate, or a refusal naming the arm.

    A CLASS AND ITS INSTANCE NEED DIFFERENT NAMES. Both were ``_GATE`` here, so the
    instance shadowed the class at import and the class was unreachable by name --
    ``meep_gpu/test_package_boundary.py::test_no_module_binds_one_name_to_two_
    competing_definitions`` refuses that outright, and it is the merge bar.
    """

    def __missing__(self, arm):  # pragma: no cover - defensive
        raise KeyError(arm)

    def __getitem__(self, arm):
        gate = _GATES.get(arm)
        if gate is None:
            raise RuntimeError(
                f"the {arm!r} arm's fixture gate could not be imported on this host "
                f"(it imports CuPy at module scope); the source legs run without it "
                f"and every fixture leg needs a device")
        return gate


_GATE = _FixtureGateTable()

#: Which module each arm's product lives in, and which predicate it answers with.
_MODULE = {"pml": "three_slot_dispersive_weld",
           "no_pml": "no_pml_three_slot_dispersive_weld"}
_PREDICATE = {"pml": "covers_three_slot_dispersive_weld",
              "no_pml": "covers_no_pml_three_slot_dispersive_weld"}
#: The D->E half each arm welds, and its own launcher entry point.
_HALF_MODULE = {"pml": "dispersive_fused_electric_pair",
                "no_pml": "no_pml_dispersive_fused_electric_pair"}
#: The E->P product each arm's weld displaces at the later seam.
_TAIL_PREDICATE = {"pml": "covers_fused_polarization_pair",
                   "no_pml": "covers_no_pml_fused_polarization_pair"}


def family(arm: str):
    """The product module for one arm, imported at call time."""
    from importlib import import_module  # noqa: PLC0415

    return import_module(f"meep_gpu.cuda_kernels.{_MODULE[arm]}")


def half(arm: str):
    """The shipped D->E module this arm welds, imported at call time."""
    from importlib import import_module  # noqa: PLC0415

    return import_module(f"meep_gpu.cuda_kernels.{_HALF_MODULE[arm]}")


def owner(arm: str):
    """The module that OWNS this arm's emitted text, and with it the gate's doors.

    NOT ALWAYS :func:`family`, and that is the whole reason this exists. Both arms
    launch the SAME kernel text, so ``SOURCE_TRANSFORM``, ``_get_kernel`` and
    ``_clear_kernel_cache`` are module-level state living in ONE module -- the PML
    twin -- and the no-absorber arm declares it in ``KERNEL_OWNER_MODULE`` rather
    than re-exporting a copy that would be a second place for the memo to live.

    Assigning a transform to the no-absorber module set an attribute no compile path
    reads: the launch went through the twin's ``_get_kernel``, the unmutated body was
    emitted, and every device mutation on that arm would have been scored
    applied-and-null -- a whole leg passing while measuring nothing. Reading the
    declaration is what makes the transform reach the body that is actually compiled.
    """
    from importlib import import_module  # noqa: PLC0415

    return import_module(
        f"meep_gpu.cuda_kernels.{family(arm).KERNEL_OWNER_MODULE}")


def specs(arm: str) -> Tuple[Dict[str, Any], ...]:
    return tuple(_GATE[arm].SPECS)


def case_rng(arm: str, label: str):
    return _GATE[arm].case_rng(label)


def build(arm: str, spec: Dict[str, Any], value_class: str, rng):
    return _GATE[arm].build(spec, value_class, rng)


def compare(arm: str, left, right, arity) -> Dict[str, int]:
    return _GATE[arm].compare(left, right, arity)


def state_of(arm: str, fields, arity) -> Dict[str, Any]:
    return _GATE[arm].state_of(fields, arity)


def state_names(arm: str, arity) -> Tuple[str, ...]:
    return _GATE[arm].state_names(arity)


def pole_names(arm: str, arity) -> Tuple[str, ...]:
    return _GATE[arm].pole_names(arity)


def frozen(arm: str, fields, arity) -> Dict[str, np.ndarray]:
    return _GATE[arm].frozen(fields, arity)


def band_must_move(arm: str) -> Tuple[str, ...]:
    """The volumes the SUBNORMAL-BAND class is held to, PER ARM.

    NOT ONE LIST, and the difference is an allocation fact rather than a preference.
    The PML arm's fixture gate declares ``BAND_MUST_MOVE`` (D, fu_D and f_w_E, from
    ``gate_cuda_fused_electric_pair``): under 'flush' a state seeded entirely in the
    band leaves E the identity, while those three families are written
    unconditionally. The NO-ABSORBER arm allocates NO ``fu``/``f_w`` at all --
    ``enable_field_storage`` on the plain branch gives E no ``f_w``
    (``gate_cuda_no_pml_dispersive_fused_electric_pair``:329-334) -- and its own gate
    holds the band class to ``_TARGETS + _ELECTRIC`` because ``update_E`` there is a
    pure overwrite that writes E every step whatever the values are.

    READ FROM EACH SIBLING GATE'S OWN SYMBOLS rather than retyped here, so a floor
    this gate holds a product to is the floor that product's own gate declared.
    Asking the no-absorber module for ``BAND_MUST_MOVE`` -- which it does not define,
    and whose PML list names two families it never allocates -- was an AttributeError
    on every ``subnormal_band`` case of that arm.
    """
    gate = _GATE[arm]
    if arm == "pml":
        return tuple(gate.BAND_MUST_MOVE)
    return tuple(gate._TARGETS) + tuple(gate._ELECTRIC)  # noqa: SLF001


def _arity(spec) -> Tuple[int, int, int]:
    return tuple(int(v) for v in spec["arity"])


def fused_passes(arm: str) -> Tuple[str, ...]:
    """The driver passes this arm's product performs, READ FROM THE MODULE.

    Never transcribed: a gate carrying its own copy of ``REPLACES`` would walk a span
    the product does not claim, and the two would drift silently in either direction.
    """
    return tuple(family(arm).REPLACES)


# ---------------------------------------------------------------------------
# THE THIRD COUNTER -- kernel launches by CUDA graph capture
# ---------------------------------------------------------------------------

class GraphNodeCounter:
    """Count EVERY kernel launch in a span, including CuPy's own elementwise ones.

    THE ONE COUNTER ON THIS TRACK THAT CAN SEE THE ARRAY PATH.
    :class:`MemoLaunchCounter` wraps the shipped compile-cache memo, which is exactly
    right for our kernels and blind to a CuPy ufunc; the launch-count claim this
    product is built on is a comparison BETWEEN the array path and the weld, so a
    counter that can only see one side cannot state it.

    HOW: the span runs on a stream under ``cudaStreamBeginCapture`` /
    ``cudaStreamEndCapture`` (the RAW runtime entry points -- ``cupy.cuda.Stream``'s
    own ``end_capture`` instantiates the graph and DESTROYS the ``cudaGraph_t``,
    leaving a null handle), then ``cudaGraphGetNodes`` enumerates the nodes and
    ``cudaGraphNodeGetType`` counts the ones of type ``cudaGraphNodeTypeKernel``.

    FAIL-SOFT BY CONSTRUCTION. Capture forbids allocation through ``cudaMalloc`` and
    any synchronization, so a span that allocates a fresh block or reads a value back
    to the host cannot be captured. Every user WARMS the span first, which lets CuPy's
    pool serve from freed blocks -- and if capture still fails, ``captured`` is False
    and ``reason`` says why. A number is never invented.
    """

    #: ``cudaGraphNodeTypeKernel``.
    _KERNEL_NODE = 0
    #: ``cudaStreamCaptureModeThreadLocal`` -- narrower than Global, so a capture in
    #: this thread cannot be invalidated by unrelated work in another.
    _MODE_THREAD_LOCAL = 2

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        for name, argtypes in (
                ("cudaStreamBeginCapture", [ctypes.c_void_p, ctypes.c_int]),
                ("cudaStreamEndCapture",
                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
                ("cudaGraphGetNodes",
                 [ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.POINTER(ctypes.c_size_t)]),
                ("cudaGraphNodeGetType",
                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
                ("cudaGraphDestroy", [ctypes.c_void_p])):
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int

    def count(self, span: Callable[[], Any]) -> Dict[str, Any]:
        """Run ``span`` under capture; return the kernel-node count or the refusal."""
        cp.cuda.runtime.deviceSynchronize()
        stream = cp.cuda.Stream(non_blocking=True)
        graph = ctypes.c_void_p()
        try:
            with stream:
                rc = self.lib.cudaStreamBeginCapture(
                    ctypes.c_void_p(stream.ptr), self._MODE_THREAD_LOCAL)
                if rc != 0:
                    return {"captured": False,
                            "reason": f"cudaStreamBeginCapture returned {rc}"}
                span()
                rc = self.lib.cudaStreamEndCapture(
                    ctypes.c_void_p(stream.ptr), ctypes.byref(graph))
            if rc != 0 or not graph:
                return {"captured": False,
                        "reason": f"cudaStreamEndCapture returned {rc}; the span "
                                  f"allocated or synchronized inside the capture"}
        except Exception as error:  # noqa: BLE001 - a refusal is a measurement
            return {"captured": False,
                    "reason": f"{type(error).__name__}: {error}"}
        try:
            total = ctypes.c_size_t(0)
            if self.lib.cudaGraphGetNodes(graph, None, ctypes.byref(total)) != 0:
                return {"captured": False, "reason": "cudaGraphGetNodes refused"}
            size = int(total.value)
            nodes = (ctypes.c_void_p * max(size, 1))()
            held = ctypes.c_size_t(size)
            self.lib.cudaGraphGetNodes(
                graph, ctypes.cast(nodes, ctypes.c_void_p), ctypes.byref(held))
            kernels = 0
            for index in range(size):
                kind = ctypes.c_int(-1)
                self.lib.cudaGraphNodeGetType(
                    ctypes.c_void_p(nodes[index]), ctypes.byref(kind))
                kernels += int(kind.value == self._KERNEL_NODE)
            return {"captured": True, "kernel_launches": kernels, "nodes": size}
        finally:
            self.lib.cudaGraphDestroy(graph)


# ---------------------------------------------------------------------------
# THE COMPOSITIONS -- one function per engine the gate steps
# ---------------------------------------------------------------------------

def weld_step(arm: str, fields, pml, grid, arguments: Dict[str, Any],
              kernel: Optional[Any] = None,
              polarization: Optional[Callable[..., Any]] = None) -> Dict[str, int]:
    """ONE complete driver step with the weld installed, WITHOUT the bracket.

    The deposit legs go through the composer instead; this is the sourceless walk, so
    the two groups run back to back with the driver's own passes between them, and
    every pass the product declares in ``REPLACES`` is SKIPPED because the launches
    perform it.

    ``polarization`` is the gate's door into the SECOND group: a host mutation hands
    in a defective loop, which a walk that always called the shipped one could not be
    given.
    """
    module = family(arm)
    skipped = fused_passes(arm)
    launches = {"leading": 0, "trailing": 0}
    for name in DRIVER_ORDER:
        if name == "step_D":
            _launch_leading(arm, fields, arguments, kernel)
            launches["leading"] += 1
        elif name == "update_P":
            runner = polarization or (
                lambda f: module.run_three_slot_polarization(f, arm)
                if arm == "pml" else module.run_three_slot_polarization(f))
            report = runner(fields)
            launches["trailing"] += int(report.get("launches", 0))
        elif name not in skipped:
            run_pass(name, fields, pml)
    return launches


def _launch_leading(arm: str, fields, arguments: Dict[str, Any],
                    kernel: Optional[Any] = None) -> Dict[str, Any]:
    """GROUP 1: the SHIPPED D->E launcher, called rather than reimplemented.

    THE POLE PLAN IS RE-RESOLVED EVERY STEP, because ``PolarizationState.update``
    rotates the buffers (dispersion.py:689-691) and a plan cached across steps names
    last step's arrays -- stale in a way that still computes.
    """
    module = half(arm)
    poles, counts = module.pole_bindings(fields)
    if arm == "pml":
        return module.launch_dispersive_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            poles, counts, kernel)
    conductivity, cond = module.conductive_bindings(fields)
    return module.launch_no_pml_dispersive_fused_electric_pair(
        fields, arguments["boundary_codes"], arguments["walls"],
        arguments["dtdx"], poles, counts, conductivity, cond, kernel)


def leading_arguments(arm: str, grid, pml) -> Dict[str, Any]:
    """Everything group 1 needs from the frozen configuration, the composer's way."""
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415

    arguments = {"boundary_codes": step_curl_kernels.real_curl_boundary_codes(grid),
                 "walls": in_seam_coverage.zero_metal_axes(grid),
                 "dtdx": float(grid.dt / grid.dx)}
    if arm == "pml":
        from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: PLC0415

        arguments["tables"] = real.fused_electric_pair_tables(pml)
        arguments["fills"] = real.fused_electric_pair_fills(grid)
    return arguments


def certified_singles_step(arm: str, fields, pml, grid, tables) -> None:
    """ONE complete driver step on the CERTIFIED SINGLES, the composition with no weld.

    ``step_D`` alone, the in-seam passes as their own certified kernels, the dispersive
    ``update_E`` alone, and the per-(state, component) ADE launches. This is the one
    place the gate runs every half the fusion absorbs, and the only route by which
    "the weld computes what the certified products compute" is a MEASUREMENT rather
    than a restatement of the array-path comparison.
    """
    from meep_gpu.cuda_kernels import (ade_kernels, dispersive_kernels,  # noqa: PLC0415
                                       in_seam_coverage, in_seam_passes,
                                       step_curl_kernels)

    dtdx = float(grid.dt / grid.dx)
    walls = in_seam_coverage.zero_metal_axes(grid)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    for name in DRIVER_ORDER:
        if name == "step_D":
            if arm == "pml":
                step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
                    fields, tables["curl"], codes, dtdx)
            else:
                _no_pml_certified_curl(fields, pml, grid, codes, dtdx)
        elif name == "zero_metal_D":
            if any(walls):
                in_seam_passes.run_pass("zero_metal", fields, "D", grid=grid)
        elif name == "fill_symmetry_bc_D":
            if grid.has_symmetry():
                in_seam_passes.run_pass("fill_symmetry", fields, "D", grid=grid)
        elif name == "fill_folded_far_ghosts_D":
            if grid.has_symmetry() and any(
                    row is not None
                    for row in in_seam_coverage.folded_far_rows(grid)):
                in_seam_passes.run_pass("fill_folded_far", fields, "D", grid=grid)
        elif name == "update_E":
            if arm == "pml":
                dispersive_kernels.update_E_fused_pml_real_dispersive(fields, pml)
            else:
                dispersive_kernels.update_E_no_pml_real_dispersive(fields, pml)
        elif name == "update_P":
            ade_kernels.update_P_fused_pml_real(fields, pml)
        else:
            run_pass(name, fields, pml)


def _no_pml_certified_curl(fields, pml, grid, codes, dtdx) -> str:
    """The no-absorber arm's certified ``step_D``, through its own family's launcher.

    WHICH CERTIFIED FAMILY IS ASKED, NOT ASSUMED. The no-absorber ``step_D`` is
    served by TWO certified families that partition on conductivity:
    ``conductive_kernels`` admits a run where SOME target of this sub-step carries a
    sigma, ``no_pml_curl`` admits the rest, and the two predicates are exact inverses
    (``covers_conductive_curl``:751-760). So the branch consults
    ``covers_conductive_curl`` rather than testing a field attribute -- launching the
    conductive family on a lossless fixture would run a certified kernel OUTSIDE the
    domain its own predicate admits, and the reference this leg compares against
    would be a composition the array path would never build.

    THIS WAS THE HALF-BUILT ONE. Both names here were invented
    (``step_D_no_pml_conductive``, ``step_D_no_pml_real``); neither exists, and the
    whole no-absorber arm died on an AttributeError the moment
    ``leg_launch_structure`` reached the certified-singles span -- after its entire
    product leg had already passed. The real entry points are below, called with the
    signature ``gate_cuda_no_pml_dispersive_fused_electric_pair``:654-657 uses.
    """
    from meep_gpu.cuda_kernels import conductive_kernels, no_pml_curl  # noqa: PLC0415

    conductive, _reason = conductive_kernels.covers_conductive_curl(
        fields, pml, grid, "step_D")
    if conductive:
        return conductive_kernels.step_conductive_curl(
            fields, pml, "step_D", codes, dtdx)
    return no_pml_curl.step_no_pml_curl(fields, "step_D", codes, dtdx)


# ---------------------------------------------------------------------------
# THE PRODUCT LEG
# ---------------------------------------------------------------------------

def run_case(arm: str, spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             polarization: Optional[Callable[..., Any]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step the array path and the weld side by side; compare per COMPLETE step."""
    started = time.time()
    label = f"{arm}|{spec['label']}|{value_class}{label_suffix}"
    arity = _arity(spec)
    case: Dict[str, Any] = {"arm": arm, "label": spec["label"],
                            "value_class": value_class, "case_seed_label": label,
                            "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "symmetry": [list(p) for p in spec.get("symmetry", ())],
                            "arity": list(arity)}

    reference, ref_grid, ref_pml, host = build(arm, spec, value_class,
                                               case_rng(arm, label))
    actual, grid, pml, _ = build(arm, spec, value_class, case_rng(arm, label))
    case["shape"] = [int(n) for n in grid.shape]
    case["live_pole_words"] = _GATE[arm].live_pole_words(host, arity)
    case["driven_components"] = _driven_components(actual)
    if sum(arity) and not case["live_pole_words"]:
        case.update(passed=False,
                    why="the pole volumes are all zero, so every pole mutation is a "
                        "null and this case measures nothing about the chain")
        return case

    drift = compare(arm, reference, actual, arity)
    if drift:
        case.update(passed=False, why=f"the two builds are not identical: {drift}")
        return case

    predicate = getattr(family(arm), _PREDICATE[arm])
    covered, reason = predicate(actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": str(reason)}
    if not covered:
        case.update(passed=False, why=f"the predicate refused this fixture: {reason}")
        return case

    arguments = leading_arguments(arm, grid, pml)
    before = frozen(arm, actual, arity)
    # WARM THE MEMO ON A THROWAWAY ENGINE FIRST. MemoLaunchCounter wraps the entries
    # ALREADY IN compile_cache._compiled_kernels when it is entered, so a kernel first
    # compiled inside the counted region is launched and NEVER counted -- the counter's
    # own docstring says so, and leg_launch_structure warms for exactly this reason.
    # Before this, the first case of every run reported an EMPTY memo proxy while the
    # launches had happened, and leg_mutations clears the cache before each device
    # mutation, so every one of those cases counted zero too.
    #
    # A SEPARATE ENGINE, not `actual`: warming on the subject would advance the state
    # this case is about to compare word for word from a seeded start.
    warm, warm_grid, warm_pml, _ = build(arm, spec, value_class,
                                         case_rng(arm, label))
    weld_step(arm, warm, warm_pml, warm_grid,
              leading_arguments(arm, warm_grid, warm_pml), kernel, polarization)
    cp.cuda.runtime.deviceSynchronize()
    del warm, warm_grid, warm_pml
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    reported = {"leading": 0, "trailing": 0}
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            launched = weld_step(arm, actual, pml, grid, arguments, kernel,
                                 polarization)
            reported["leading"] += launched["leading"]
            reported["trailing"] += launched["trailing"]
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(arm, reference, actual, arity)
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items()))})
            if difference:
                break

    after = state_of(arm, actual, arity)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in state_names(arm, arity)}
    required = (state_names(arm, arity) if value_class == "uniform"
                else band_must_move(arm) + pole_names(arm, arity))
    unmoved = sorted(name for name in required if moved.get(name, 0) == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    expected_trailing = len(case["driven_components"])
    # THE SECOND COUNTER, READ BY PREFIX RATHER THAN BY NAME. MemoLaunchCounter keys
    # its tally on the COMPILE-CACHE KEY's first element, and this family's key is
    # `three_slot_polarization_real:<component>:<poles>:<sigma kinds>` (the emitter's
    # specialization axes are part of the memo key, so one kernel NAME appears under
    # one key per component). Looking up the bare KERNEL_NAME therefore found 0 on
    # every run and the two counters could never agree; summing the keys that BELONG
    # to this family is what the check meant to ask.
    memo_named = counter.named()
    memo_trailing = sum(count for key, count in memo_named.items()
                        if key == family(arm).KERNEL_NAME
                        or key.startswith(family(arm).KERNEL_NAME + ":"))
    # WHEN THE GATE SUPPLIES THE KERNEL THE MEMO CANNOT SEE IT, and that is the
    # launcher's own contract: `kernel or _get_kernel(...)` never consults the memo
    # when a kernel is handed in, so the honest expectation is ZERO through this
    # counter and the launcher's own report is the only count. Both branches of this
    # were identical before, which stated the opposite for a path no leg takes today.
    expected_memo_trailing = (expected_trailing * len(per_step)
                              if kernel is None else 0)
    launch_ok = (reported["leading"] == len(per_step)
                 and reported["trailing"] == expected_trailing * len(per_step)
                 and memo_trailing == expected_memo_trailing)
    case.update({
        "passed": bool(identical and not unmoved and launch_ok),
        "bit_identical": identical,
        "steps_run": len(per_step),
        "first_divergence": next((r["step"] for r in per_step
                                  if r["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "arrays_compared": len(state_names(arm, arity)),
        "pole_arrays_compared": len(pole_names(arm, arity)),
        "movement_floor": {"class": value_class, "unmet": unmoved},
        "launch_counts": {
            "leading_reports": reported["leading"],
            "trailing_reports": reported["trailing"],
            "expected_per_step": 1 + expected_trailing,
            "memo_proxy": memo_named,
            "memo_proxy_total": counter.total,
            # THE TWO INDEPENDENT COUNTERS, SIDE BY SIDE: the launcher's own report
            # of how many component launches it issued, and the shipped compile-cache
            # memo's tally of how many times THIS family's kernels were entered.
            "memo_trailing": memo_trailing,
            "expected_memo_trailing": expected_memo_trailing,
            "kernel_supplied_by_gate": kernel is not None,
            "agree": bool(launch_ok),
        },
        "seconds": time.time() - started,
    })
    return case


def _driven_components(fields) -> List[str]:
    """Which of Ex, Ey, Ez any registered state drives -- the launch count's own
    denominator, read from the engine rather than from the arity triple."""
    return [component for component in ("Ex", "Ey", "Ez")
            if any(state.drives(component)
                   for state in getattr(fields, "polarizations", ()) or ())]


# ---------------------------------------------------------------------------
# THE LAUNCH STRUCTURE -- the progress-reporting leg, and the reason this product exists
# ---------------------------------------------------------------------------

def leg_launch_structure(arm: str, spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """FIVE compositions from one seed: all agree word for word, none launches alike.

    THE CLAIM IS MEASURED, NEVER ASSERTED, and it is measured against every
    composition this product could be compared to rather than the flattering one:

      array_path              stepping's eleven passes
      certified_singles       step_D + in-seam passes + update_E + the ADE singles
      d_to_e_weld_plus_ade    the D->E weld + the certified ADE singles
      step_d_plus_e_to_p_weld the certified step_D + in-seam passes + the E->P weld
                              -- WHAT THE COMPOSER INSTALLS ON THESE ROWS TODAY
      three_slot_weld         this product

    IF THE WELD DOES NOT REDUCE LAUNCHES AGAINST ANY OF THEM, THAT IS THE FINDING and
    it is recorded as such rather than hidden behind the ones it beats.
    """
    from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_polarization_pair as ep  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (in_seam_coverage, in_seam_passes,  # noqa: PLC0415
                                       step_curl_kernels)

    started = time.time()
    arity = _arity(spec)
    label = f"{arm}|{spec['label']}|launches"
    engines = {name: build(arm, spec, "uniform", case_rng(arm, label))
               for name in ("array_path", "certified_singles",
                            "d_to_e_weld_plus_ade", "step_d_plus_e_to_p_weld",
                            "three_slot_weld")}
    grid = engines["three_slot_weld"][1]
    pml = engines["three_slot_weld"][2]
    dtdx = float(grid.dt / grid.dx)
    walls = in_seam_coverage.zero_metal_axes(grid)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    tables: Dict[str, Any] = {}
    if arm == "pml":
        tables["curl"] = step_curl_kernels.real_pml_curl_tables(pml, False)
    arguments = leading_arguments(arm, grid, pml)
    driven = _driven_components(engines["three_slot_weld"][0])

    def in_seam(fields):
        if grid.has_symmetry():
            in_seam_passes.run_pass("fill_symmetry", fields, "D", grid=grid)
        if any(walls):
            in_seam_passes.run_pass("zero_metal", fields, "D", grid=grid)
        if grid.has_symmetry() and any(
                row is not None for row in in_seam_coverage.folded_far_rows(grid)):
            in_seam_passes.run_pass("fill_folded_far", fields, "D", grid=grid)

    def certified_curl(fields):
        if arm == "pml":
            step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
                fields, tables["curl"], codes, dtdx)
        else:
            _no_pml_certified_curl(fields, pml, grid, codes, dtdx)

    def span_array(fields, layer):
        for name in DRIVER_ORDER:
            run_pass(name, fields, layer)

    def span_singles(fields, layer):
        certified_singles_step(arm, fields, layer, grid, tables)

    def span_d_to_e_plus_ade(fields, layer):
        for name in DRIVER_ORDER:
            if name == "step_D":
                _launch_leading(arm, fields, arguments)
            elif name == "update_P":
                ade_kernels.update_P_fused_pml_real(fields, layer)
            elif name not in tuple(half(arm).REPLACES):
                run_pass(name, fields, layer)

    def span_step_d_plus_e_to_p(fields, layer):
        for name in DRIVER_ORDER:
            if name == "step_D":
                certified_curl(fields)
            elif name in ("fill_symmetry_bc_D", "zero_metal_D",
                          "fill_folded_far_ghosts_D"):
                if name == "fill_symmetry_bc_D":
                    in_seam(fields)
            elif name == "update_E":
                ep.run_fused_polarization_pair(
                    fields, layer if arm == "pml" else None, arm)
            elif name == "update_P":
                pass  # the E->P weld advanced the polarizations in its own launches
            else:
                run_pass(name, fields, layer)

    def span_weld(fields, layer):
        weld_step(arm, fields, layer, grid, arguments)

    spans = {"array_path": span_array, "certified_singles": span_singles,
             "d_to_e_weld_plus_ade": span_d_to_e_plus_ade,
             "step_d_plus_e_to_p_weld": span_step_d_plus_e_to_p,
             "three_slot_weld": span_weld}

    # WARM EVERY SPAN ONCE OUTSIDE THE COUNTERS. A cold compile inside a
    # MemoLaunchCounter is a launch that is not counted, and a cold ALLOCATION inside
    # a graph capture is a capture that fails; one warm step fixes both, and the
    # engines are rebuilt afterwards so no span starts from a state another advanced.
    for name, span in spans.items():
        fields, _g, layer, _h = engines[name]
        span(fields, layer)
    cp.cuda.runtime.deviceSynchronize()
    engines = {name: build(arm, spec, "uniform", case_rng(arm, label))
               for name in spans}

    memo: Dict[str, Any] = {}
    graph: Dict[str, Any] = {}
    counters = {name: MemoLaunchCounter() for name in spans}
    rows: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name, span in spans.items():
            fields, _g, layer, _h = engines[name]
            with counters[name]:
                span(fields, layer)
        cp.cuda.runtime.deviceSynchronize()
        row = {"step": step}
        for name in spans:
            if name == "three_slot_weld":
                continue
            row[f"weld_vs_{name}"] = sum(compare(
                arm, engines[name][0], engines["three_slot_weld"][0],
                arity).values())
        rows.append(row)
        if any(value for key, value in row.items() if key != "step"):
            break
    for name in spans:
        memo[name] = {"total": counters[name].total,
                      "named": counters[name].named(),
                      "per_step": counters[name].total / max(len(rows), 1)}

    # THE GRAPH COUNTER, on a FRESH set of engines: capture cannot run inside the
    # comparison loop above (each span there advances a state the next step reads),
    # and one captured step is what the per-step claim is about.
    graph_engines = {name: build(arm, spec, "uniform", case_rng(arm, label))
                     for name in spans}
    graph_counter = GraphNodeCounter()
    for name, span in spans.items():
        fields, _g, layer, _h = graph_engines[name]
        span(fields, layer)  # warm the pool for THIS engine
    cp.cuda.runtime.deviceSynchronize()
    for name, span in spans.items():
        fields, _g, layer, _h = graph_engines[name]
        graph[name] = graph_counter.count(lambda f=fields, l=layer, s=span: s(f, l))

    agreed = all(all(value == 0 for key, value in row.items() if key != "step")
                 for row in rows) and len(rows) == steps
    weld_per_step = memo["three_slot_weld"]["per_step"]

    # WHICH COUNTER ANSWERS "DOES IT REDUCE LAUNCHES" DEPENDS ON THE ROW, and reading
    # the wrong one printed a false statement about the row that matters most.
    # MemoLaunchCounter wraps the shipped compile-cache memo, so it sees OUR kernels
    # and is BLIND to CuPy's own elementwise launches -- the array path therefore
    # scores 0.0 through it, and `0.0 > 4.0` reported "the weld does not reduce
    # launches against the array path" when the graph counter had just measured 235
    # against 57 on the same shape. The graph counter is the only instrument that can
    # see both sides, so it is preferred wherever the capture succeeded, and the
    # instrument used is recorded per row rather than left to be inferred.
    def _reduction(name: str) -> Dict[str, Any]:
        theirs, ours = graph.get(name, {}), graph.get("three_slot_weld", {})
        if theirs.get("captured") and ours.get("captured"):
            return {"reduces": bool(theirs["kernel_launches"]
                                    > ours["kernel_launches"]),
                    "instrument": "cuda_graph_nodes",
                    "theirs": theirs["kernel_launches"],
                    "ours": ours["kernel_launches"],
                    "spans": "the WHOLE driver step, every kernel including CuPy's"}
        if name == "array_path":
            return {"reduces": None, "instrument": "none",
                    "why": "the graph capture failed and the memo counter cannot see "
                           "the array path's CuPy launches; no number is invented"}
        return {"reduces": bool(memo[name]["per_step"] > weld_per_step),
                "instrument": "compile_cache_memo",
                "theirs": memo[name]["per_step"], "ours": weld_per_step,
                "spans": "step_D -> update_P, this package's kernels only"}

    reduction = {name: _reduction(name)
                 for name in spans if name != "three_slot_weld"}
    beats = {name: row["reduces"] for name, row in reduction.items()}
    return {
        "passed": bool(agreed),
        "arm": arm, "label": spec["label"], "steps": len(rows),
        "arity": list(arity), "driven_components": driven,
        "per_step": rows,
        "memo_counter": memo,
        "graph_counter": graph,
        "weld_launches_per_step_memo": weld_per_step,
        "launch_reduction": reduction,
        "weld_reduces_launches_against": beats,
        "reading": (
            "all five compositions agree word for word; the fusion is visible ONLY in "
            "the launch count" if agreed else
            "the five compositions do not agree"),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# THE DEPOSIT LEG -- the bracket, and the two orderings that must not pass
# ---------------------------------------------------------------------------

def leg_deposit(arm: str, spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The bracket end to end, THROUGH THE SHIPPED COMPOSER, with three controls.

    Reaching for the launchers directly would test the kernels and skip the wiring,
    and the wiring is what this leg exists for: ``arms.plan_step(..., fuse=True)`` is
    the only route by which ``fused_pairs._install_fused_triple`` brackets group 1 and
    puts group 2 in ``update_P``, and a product whose ``FUSED_PRODUCTS`` row were
    missing would fail here rather than pass on an unbracketed launch.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    started = time.time()
    arity = _arity(spec)
    rows: List[Dict[str, Any]] = []
    for mode in ("bracketed", "unbracketed", "p_before_repair"):
        label = f"{arm}|{spec['label']}|deposit|{mode}"
        row: Dict[str, Any] = {"arm": arm, "label": spec["label"], "mode": mode}
        reference, ref_grid, ref_pml, _ = build(arm, spec, "uniform",
                                                case_rng(arm, label))
        actual, grid, pml, _ = build(arm, spec, "uniform", case_rng(arm, label))
        if compare(arm, reference, actual, arity):
            row.update(passed=False, why="the two builds differ")
            rows.append(row)
            continue
        reference_sources = (base._electric_source(reference),)  # noqa: SLF001
        actual_sources = (base._electric_source(actual),)  # noqa: SLF001
        index = deposit_repair._deposit_index(actual_sources[0])  # noqa: SLF001
        if index is None or len(
                deposit_repair.in_seam_sources(actual_sources, "D")) != 1:
            row.update(passed=False,
                       why="the source built here is not an in-seam D deposit")
            rows.append(row)
            continue
        row["deposit_points"] = int(cp.asarray(index[0]).size)
        row["deposit_magnitude"] = base.deposit_magnitude(
            actual, base._electric_source(actual), steps, float(grid.dt))  # noqa: SLF001
        if row["deposit_magnitude"] <= 0.0:
            row.update(passed=False, why="this source deposits exactly zero")
            rows.append(row)
            continue

        declared = actual_sources if mode != "unbracketed" else ()
        composed = arms.plan_step(actual, pml, grid, sources=declared, fuse=True)
        leading = composed.plans.get("step_D")
        trailing = composed.plans.get("update_E")
        tail = composed.plans.get("update_P")
        installed = (type(leading).__name__, type(trailing).__name__,
                     type(tail).__name__)
        row["installed_plans"] = list(installed)
        row["selected"] = {slot: composed.selected.get(slot)
                           for slot in ("step_D", "update_E", "update_P")}
        # WHAT THE COMPOSER MUST HAVE PUT IN THE THREE SLOTS. With no source declared
        # there is nothing to bracket, so `_install_fused_triple` puts the bare halves
        # in step_D and update_P and a NoopPlan in the absorbed update_E; with one
        # declared, the leading half is wrapped by the save and update_E holds the
        # apply. A product whose FUSED_PRODUCTS row were missing fails here rather
        # than passing on an unbracketed launch.
        expected = (("_TripleHalfPlan", "NoopPlan", "_TripleHalfPlan")
                    if mode == "unbracketed" else
                    ("LeadingRepairPlan", "TrailingRepairPlan", "_TripleHalfPlan"))
        row["installed_as_expected"] = installed == expected
        if leading is None or trailing is None or tail is None:
            row.update(passed=False, why="the composer did not fuse the seam",
                       refusals=[reason for key, value in composed.reasons.items()
                                 if key.startswith("fused_pair") for reason in value])
            rows.append(row)
            continue

        dt = float(grid.dt)
        skipped = fused_passes(arm)
        per_step: List[Dict[str, Any]] = []
        for step in range(1, steps + 1):
            when = (step - 1) * dt
            base._withdraw(reference_sources, reference)  # noqa: SLF001
            for name in DRIVER_ORDER:
                run_pass(name, reference, ref_pml)
                if name == "step_D":
                    for source in reference_sources:
                        source.inject(reference, when + 0.5 * dt)
            base._withdraw(actual_sources, actual)  # noqa: SLF001
            for name in DRIVER_ORDER:
                if name == "step_D":
                    leading.run()
                    for source in actual_sources:
                        source.inject(actual, when + 0.5 * dt)
                elif name == "update_E":
                    if mode == "p_before_repair":
                        # THE ORDERING A READER WOULD REACH FOR, ARMED: the
                        # polarizations advanced at the update_E consult, so
                        # deposit_repair.apply reads P^(n+1) through
                        # displacement_minus_polarization.
                        tail.run()
                    trailing.run()
                elif name == "update_P":
                    if mode != "p_before_repair":
                        tail.run()
                elif name not in skipped:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(arm, reference, actual, arity)
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items()))})
            if difference:
                break
        identical = (len(per_step) == steps
                     and all(r["differing_words"] == 0 for r in per_step))
        repairs = int(getattr(leading, "repairs", 0))
        must_diverge = mode in ("unbracketed", "p_before_repair")
        row.update({
            "passed": bool(row["installed_as_expected"]
                           and (not identical if must_diverge
                                else identical and repairs > 0)),
            "bit_identical": identical,
            "must_diverge": must_diverge,
            "deposit_points_repaired": repairs,
            "steps_run": len(per_step),
            "first_divergence": next((r["step"] for r in per_step
                                      if r["differing_words"]), None),
            "per_step": per_step[:8],
        })
        rows.append(row)

    # THE WRONG REPAIR DECLARATION, refused BY NAME rather than mis-repairing.
    wrong = _wrong_repair_path(arm, spec)
    rows.append(wrong)
    return {
        "passed": all(row.get("passed") for row in rows) and len(rows) == 4,
        "arm": arm, "label": spec["label"], "rows": rows,
        "reading": ("the bracket AND its position are what make the deposit case "
                    "pass: unbracketed and P-before-repair both diverge"),
        "seconds": time.time() - started,
    }


def _wrong_repair_path(arm: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """This arm declaring the OTHER recurrence: a refusal by name, never a repair."""
    label = f"{arm}|{spec['label']}|wrong_path"
    fields, grid, pml, _ = build(arm, spec, "uniform", case_rng(arm, label))
    source = base._electric_source(fields)  # noqa: SLF001
    other = ((deposit_repair.PLAIN_PATH,) if arm == "pml"
             else (deposit_repair.SPLIT_FIELD_PATH,))
    row: Dict[str, Any] = {"arm": arm, "label": spec["label"],
                           "mode": "wrong_repair_path",
                           "declared": list(other),
                           "shipped": list(family(arm).REPAIR_PATHS)}
    try:
        deposit_repair.save(fields, (source,), "D", pml, paths=other)
        row.update(passed=False, refused=None,
                   why="the wrong repair declaration was accepted")
    except deposit_repair.DepositNotRepairable as error:
        row.update(passed=True, refused=f"DepositNotRepairable: {error}")
    return row


# ---------------------------------------------------------------------------
# THE REFUSALS AND THE ARBITRATION
# ---------------------------------------------------------------------------

def leg_refusal() -> Dict[str, Any]:
    """What each weld refuses BY NAME, and who takes the slots when it does not.

    THE PARTITION IS HALF OF THIS LEG: a configuration is only correctly refused if
    some other product can take it, and the ones that matter are the two-slot halves
    this product supersedes -- the composer must give the slots to exactly one.
    """
    from meep_gpu.cuda_kernels import arms, fused_pairs  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_polarization_pair as ep  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []

    def ask(arm: str, name: str, spec: Dict[str, Any], sources: Any, expect: str,
            mutate=None) -> None:
        fields, grid, pml, _ = build(arm, spec, "uniform",
                                     case_rng(arm, f"refusal|{arm}|{name}"))
        if mutate is not None:
            fields = mutate(fields)
        covered, why = getattr(family(arm), _PREDICATE[arm])(
            fields, pml, grid, sources)
        d_to_e, d_reason = getattr(
            half(arm), f"covers_{_HALF_MODULE[arm]}")(fields, pml, grid, sources)
        tail, tail_reason = getattr(ep, _TAIL_PREDICATE[arm])(
            fields, pml, grid, sources)
        rows.append({
            "arm": arm, "label": name, "covered": bool(covered),
            "reason": str(why), "expect": expect,
            "d_to_e_half_covered": bool(d_to_e), "d_to_e_reason": str(d_reason),
            "e_to_p_half_covered": bool(tail), "e_to_p_reason": str(tail_reason),
            # THE SUBSET PROPERTY THE SUPERSESSION RULE RESTS ON, per row: where the
            # weld admits, BOTH halves must admit. If that ever stopped holding,
            # refusing the shorter span would start costing rows.
            "subset_holds": (not covered) or (bool(d_to_e) and bool(tail)),
            "ok": (bool(covered) is (expect == "admit"))
                  and ((not covered) or (bool(d_to_e) and bool(tail))),
        })

    for arm in ARMS:
        spec = dict(specs(arm)[0])
        source = None
        ask(arm, "corpus_row_declared_empty_sources", spec, (), "admit")
        ask(arm, "undeclared_source_set", spec, None, "refuse")
        fields, grid, _p, _h = build(arm, spec, "uniform",
                                     case_rng(arm, f"refusal|{arm}|src"))
        source = base._electric_source(fields)  # noqa: SLF001
        ask(arm, "electric_deposit_carried_by_the_repair", spec, (source,), "admit")
        zero = dict(spec, label=f"{spec['label']}_zero_arity", arity=(0, 0, 0))
        ask(arm, "degenerate_arity_has_no_polarization_to_advance", zero, (),
            "refuse")
        ask(arm, "complex_storage", spec, (), "refuse",
            mutate=lambda f: pml_gate._WithFlag(f, "force_complex_fields", True))  # noqa: SLF001
        ask(arm, "off_diagonal_chi1inv", spec, (), "refuse",
            mutate=lambda f: pml_gate._WithFlag(f, "has_offdiagonal_epsilon", True))  # noqa: SLF001

    # THE ABSORBER PARTITION: neither weld may admit the other's arm.
    for arm in ARMS:
        spec = dict(specs(arm)[0])
        fields, grid, pml, _ = build(arm, spec, "uniform",
                                     case_rng(arm, f"partition|{arm}"))
        other = "no_pml" if arm == "pml" else "pml"
        covered, why = getattr(family(other), _PREDICATE[other])(
            fields, pml, grid, ())
        rows.append({"arm": other, "label": f"refuses_the_{arm}_arm",
                     "covered": bool(covered), "reason": str(why),
                     "expect": "refuse", "subset_holds": True,
                     "ok": not covered})

    # THE ARBITRATION, THROUGH THE SHIPPED COMPOSER.
    arbitration: List[Dict[str, Any]] = []
    for arm in ARMS:
        spec = dict(specs(arm)[0])
        fields, grid, pml, _ = build(arm, spec, "uniform",
                                     case_rng(arm, f"arbitration|{arm}"))
        source = base._electric_source(fields)  # noqa: SLF001
        plan = arms.plan_step(fields, pml, grid, sources=(source,), fuse=True)
        displaced = f"cuda_{_HALF_MODULE[arm]}"
        tail_family = ("cuda_fused_polarization_pair" if arm == "pml"
                       else "cuda_no_pml_fused_polarization_pair")
        entry = {
            "arm": arm,
            "selected": {slot: plan.selected.get(slot)
                         for slot in ("step_D", "update_E", "update_P")},
            "d_to_e_refusal": list(
                plan.reasons.get(f"fused_pair_{displaced}", ())),
            "e_to_p_refusal": list(
                plan.reasons.get(f"fused_pair_{tail_family}", ())),
            "installed": [type(plan.plans.get(slot)).__name__
                          for slot in ("step_D", "update_E", "update_P")],
        }
        label = fused_pairs.FUSED_PRODUCTS[family(arm).FAMILY]
        entry["ok"] = (
            len(set(entry["selected"].values())) == 1
            and None not in entry["selected"].values()
            and any("strictly contains" in reason
                    for reason in entry["d_to_e_refusal"])
            and any("was selected by" in reason
                    for reason in entry["e_to_p_refusal"])
            and entry["installed"] == ["LeadingRepairPlan", "TrailingRepairPlan",
                                       "_TripleHalfPlan"])
        _ = label
        arbitration.append(entry)

    return {"passed": (all(row["ok"] for row in rows)
                       and all(row["ok"] for row in arbitration)),
            "rows": rows, "arbitration": arbitration,
            "checked": len(rows) + len(arbitration)}


# ---------------------------------------------------------------------------
# THE MUTATIONS
# ---------------------------------------------------------------------------

def _drop(body: str) -> Callable[[str], Tuple[str, int]]:
    return lambda source: (source.replace(body, "", 1), source.count(body))


#: Device-text mutations of the NEW kernel, each with whether it can ever be a null.
#: ``(label, transform, predicted_null)``.
#:
#: EVERY TRANSFORM TAKES THE COMPONENT IT IS MUTATING, and that is not decoration.
#: The emitter names the per-pole parameters after the COMPONENT
#: (``P_Ex_0``/``P_Ey_0``/``P_Ez_0``, ``_signature``), and the hook below is called
#: once per component, so a transform that spelled ``P_Ex_0`` outright produced text
#: naming an identifier that does not exist in the Ey and Ez bodies -- an NVRTC
#: ``identifier "P_Ex_0" is undefined`` that took the whole gate down instead of
#: planting a defect. Two of the seven did exactly that.
SOURCE_MUTATIONS: Tuple[Tuple[str, Callable[[str, str], Tuple[str, int]],
                              Optional[str]], ...] = (
    ("drive_from_the_first_pole_slot",
     lambda s, c: needle(s, "float w = drive[idx];", f"float w = P_{c}_0[idx];"),
     None),
    ("drop_the_extent_guard",
     lambda s, c: needle(s, "    if (idx >= n_elem) return;\n", ""), None),
    ("swap_c_now_and_c_prev",
     lambda s, c: needle(s, "(p_0 * c_now_0) + (c_prev_0 * q_0)",
                         "(p_0 * c_prev_0) + (c_now_0 * q_0)"), None),
    ("regroup_the_sum",
     lambda s, c: needle(
         s, "p_out_0[idx] = ((p_0 * c_now_0) + (c_prev_0 * q_0)) + "
            "(c_drive_0 * (s_0 * w));",
         "p_out_0[idx] = (p_0 * c_now_0) + ((c_prev_0 * q_0) + "
         "(c_drive_0 * (s_0 * w)));"), None),
    ("write_the_result_into_p_now",
     lambda s, c: needle(s, "p_out_0[idx] =", f"((float*)P_{c}_0)[idx] ="), None),
    ("drive_scaled_by_the_sigma_of_pole_zero",
     lambda s, c: needle(s, "float s_1 = ", "float s_1 = s_0; //"),
     "this case drives no component with two or more poles, so there is no s_1 to "
     "collapse onto s_0"),
    # APPLIED TO THE Ez BODY ONLY, so the label is what the leg does. Left
    # component-blind it dropped the last pole of ALL THREE bodies while calling
    # itself an Ez mutation, and a fixture whose Ez carries one pole would have been
    # scored armed on the strength of the Ex and Ey edits.
    ("drop_the_last_pole_of_Ez",
     lambda s, c: (_drop_last_pole(s) if c == "Ez" else (s, 0)), None),
)


def _drop_last_pole(source: str) -> Tuple[str, int]:
    """Remove the LAST pole block of the emitted kernel. A recurrence that never ran.

    Anchored on the emitted comment the emitter writes above every pole block AND on
    the ``p_out_<i>`` store that closes it, so what is removed is a WHOLE block. The
    end anchor is the fix for a mutation that could not compile: ``[\\s\\S]*?;\\n``
    stops at the FIRST statement terminator, which is the ``float p_<i> = ...;``
    declaration one line in, so the old pattern deleted the comment and the ``p``
    load while leaving the ``q``/``sigma`` loads and the store that reads ``p_<i>``
    behind -- NVRTC refused the body with ``identifier "p_5" is undefined`` and the
    exception took the gate down instead of planting a defect.

    A body with one pole is left untouched and the mutation reports zero
    replacements, which the scorer records as an unarmed leg rather than a pass.
    """
    blocks = re.findall(
        r"\n    // pole \d+: [\s\S]*?\n    p_out_\d+\[idx\] = [^\n]*;\n", source)
    if len(blocks) < 2:
        return source, 0
    return source.replace(blocks[-1], "\n", 1), 1


#: Host-side mutations of the SECOND group's loop, each armed against a LOCAL copy of
#: the shipped loop whose no-defect twin must come back UNCAUGHT first.
HOST_MUTATIONS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("none", None),
    ("stale_component_specs",
     "no state drives more than one component, so no rotation happens between the "
     "component launches for a cached spec to go stale across"),
    ("rotate_before_launch", None),
    ("skip_the_rotation", None),
    ("component_order_reversed",
     "PREDICTED: the recurrence for one component reads only that component's P and "
     "P_prev plus the shared drive and writes only that component's slots, so the "
     "component order changes WHICH allocation carries the retired history and "
     "nothing a compared volume holds. A confirmed null is the measurement that the "
     "three component launches may be issued in any order"),
    ("state_order_reversed",
     "PREDICTED: each state owns its own P/P_prev/scratch (dispersion.py:645-651) "
     "and the recurrence reads only that state's buffers plus the shared drive, so "
     "the order of states WITHIN one component is not observable in any compared "
     "volume. A confirmed null is the measurement that the state loop may be fused "
     "into one launch"),
    # THE ONE MUTATION WHOSE VERDICT DEPENDS ON THE ARM, and the reason below is the
    # NO-PML one. `leg_mutations` supplies `None` on the PML arm instead: an ACTIVE
    # layer makes `Fields.drive_field` name `f_w_<c>` rather than the stored `E<c>`,
    # so there the substitution is a real defect -- caught by
    # `_drive_identity_problem`, which refuses it BY NAME before the launch binds
    # anything, and is recorded in `refused_by_name`.
    ("drive_from_the_stored_E",
     "the layer is inactive, so Fields.drive_field returns the stored E itself "
     "(fields.py:1160-1162) and the substitution is the identity"),
)


def mutated_polarization(arm: str, mutation: str):
    """A LOCAL copy of the shipped second group, with one defect planted.

    THE NO-DEFECT TWIN IS REQUIRED UNCAUGHT FIRST (``mutation == 'none'``): a local
    copy that diverged on its own would score every defect below as caught while
    measuring nothing about them.
    """
    module = family(arm)
    terms = module.ELECTRIC_TERMS

    def run(fields) -> Dict[str, Any]:
        order = list(terms)
        if mutation == "component_order_reversed":
            order = list(reversed(order))
        launches = 0
        cached: Dict[str, Any] = {}
        for target, _source, _axis in order:
            if mutation == "stale_component_specs":
                if not cached:
                    cached.update(module.component_specs(fields))
                spec = cached[target]
            else:
                spec = module.component_specs(fields)[target]
            states = list(spec["states"])
            if mutation == "state_order_reversed":
                states = list(reversed(states))
            if not states:
                continue
            if mutation == "rotate_before_launch":
                for state in states:
                    p = state.P[target]
                    p_prev = state.P_prev[target]
                    scratch = state._scratch
                    state.P[target] = scratch
                    state.P_prev[target] = p
                    state._scratch = p_prev
            if mutation == "drive_from_the_stored_E":
                original = fields.drive_field
                fields.drive_field = lambda component: getattr(fields, component)
                try:
                    module.launch_three_slot_polarization_component(
                        fields, arm, target, states)
                finally:
                    fields.drive_field = original
            else:
                module.launch_three_slot_polarization_component(
                    fields, arm, target, states)
            launches += 1
            if mutation == "skip_the_rotation":
                continue
            for state in states:
                p = state.P[target]
                p_prev = state.P_prev[target]
                scratch = state._scratch
                state.P[target] = scratch
                state.P_prev[target] = p
                state._scratch = p_prev
        return {"launched": True, "launches": launches}

    return run


def measure_extent_guard_overrun(arm: str) -> Dict[str, Any]:
    """DOES ``drop_the_extent_guard`` REACH THE DEVICE? Measured, not reasoned about.

    THE PROBLEM THIS SOLVES. Removing ``if (idx >= n_elem) return;`` comes back a
    NULL in the product comparison on every fixture here, and a null with no reason
    is a leg that measured nothing. Two explanations fit and they are not equally
    acceptable: either the surplus lanes never ran (the mutation is UNARMED and the
    leg is vacuous), or they ran and wrote words that no COMPARED volume holds (the
    mutation is armed and the null is a property of the allocation, not of the
    kernel). The product comparison cannot tell them apart, so this asks directly.

    HOW: the emitted kernel is compiled with the guard removed and launched over a
    DELIBERATELY OVERSIZED ``p_out`` -- ``blocks * threads`` floats where the launch
    is told ``n_elem`` -- pre-filled with a sentinel. Every word past ``n_elem`` that
    changed is a surplus lane that executed. Nothing here reads past an allocation:
    the buffer this binds is the one this function allocated, which is why the
    experiment is run on synthetic arrays rather than on a fixture's own scratch.

    A ZERO HERE WOULD BE A FAILURE, not a reassurance: it would mean the block
    geometry leaves no surplus lane and the mutation cannot be armed on this shape,
    which is exactly the vacuous fixture the charter refuses.
    """
    module = owner(arm)
    # A CELL COUNT THAT IS NOT A MULTIPLE OF THE BLOCK, on purpose: 990 = 3*256 + 222
    # is the corpus fixture's own 9*10*11, and an exact multiple would leave no
    # surplus lane to measure. gate_cuda_ade picks its 13x17x11 shape for the same
    # reason (":191-196": an exact multiple "cannot exercise it").
    n_elem = 990
    threads = module._FUSED_THREADS  # noqa: SLF001
    blocks = (n_elem + threads - 1) // threads
    span = blocks * threads
    source = module.three_slot_polarization_source(0, 1, (False,))
    mutated, hits = needle(source, "    if (idx >= n_elem) return;\n", "")
    if not hits:
        return {"armed": False,
                "why": "the guard line was not found in the emitted body"}
    kernel = cp.RawKernel(mutated, module.KERNEL_NAME,
                          options=module._COMPILE_OPTIONS)  # noqa: SLF001
    rng = np.random.default_rng(20260903)
    drive = cp.asarray(rng.uniform(-1, 1, n_elem).astype(np.float32))
    p_now = cp.asarray(rng.uniform(-1, 1, n_elem).astype(np.float32))
    p_prev = cp.asarray(rng.uniform(-1, 1, n_elem).astype(np.float32))
    sentinel = np.float32(-12345.0)
    p_out = cp.full(span, sentinel, dtype=cp.float32)
    kernel((blocks,), (threads,),
           (drive, p_now, p_out, p_prev, np.float32(0.75),
            np.float32(1.5), np.float32(-0.5), np.float32(0.25),
            np.int32(n_elem)))
    cp.cuda.runtime.deviceSynchronize()
    tail = cp.asnumpy(p_out[n_elem:])
    wrote = int(np.count_nonzero(tail != sentinel))
    return {
        "armed": bool(wrote > 0),
        "n_elem": n_elem, "blocks": blocks, "threads": threads,
        "surplus_lanes": span - n_elem,
        "surplus_lanes_that_wrote": wrote,
        "why": (
            f"the guard was removed and {wrote} of the {span - n_elem} surplus lanes "
            f"wrote past n_elem, so the mutation REACHES THE DEVICE. It is null in "
            f"the product comparison because those words fall outside every compared "
            f"volume: p_out is the state's own _scratch, shaped exactly n_elem, and "
            f"the overrun lands in the slack of the pooled block rather than in "
            f"another array the gate reads back"),
    }


def leg_mutations(arm: str, spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """Score every armed defect against its declaration.

    A mutation that does not fire is a leg that measured nothing, so the REPLACEMENT
    COUNT is recorded: a transform that matched no anchor is UNARMED and fails, and a
    mutation that applied and did not diverge is a NULL that must carry its reason.
    """
    module = family(arm)
    # THE MODULE THE TRANSFORM MUST BE PLANTED ON is the one that owns the emitted
    # text, which on the no-absorber arm is the PML twin (`owner`). Planting on
    # `module` there set an attribute nothing reads.
    text_owner = owner(arm)
    started = time.time()
    rows: List[Dict[str, Any]] = []

    # 1. THE DEVICE TEXT.
    for label, transform, predicted in SOURCE_MUTATIONS:
        applied = {"count": 0}

        def hook(component_index, count, kinds, source, _t=transform,
                 _a=applied, _m=module):
            # THE COMPONENT THIS BODY IS FOR, read from the module's own term table
            # rather than from the index, so the name the transform splices is the
            # name the emitter's signature declared for THIS launch.
            target = _m.ELECTRIC_TERMS[component_index][0]
            mutated, hits = _t(source, target)
            _a["count"] += int(hits)
            return mutated

        text_owner.SOURCE_TRANSFORM = hook
        try:
            text_owner._clear_kernel_cache()  # noqa: SLF001
            case = run_case(arm, spec, "uniform", min(steps, 12),
                            label_suffix=f"|mut:{label}")
        finally:
            text_owner.SOURCE_TRANSFORM = None
            text_owner._clear_kernel_cache()  # noqa: SLF001
        caught = not case.get("bit_identical", False)
        row: Dict[str, Any] = {
            "kind": "device", "label": label, "arm": arm,
            "replacements": applied["count"],
            "armed": applied["count"] > 0,
            "caught": bool(caught),
            "first_divergence": case.get("first_divergence"),
            "differing_words": case.get("differing_words"),
            "predicted_null": predicted,
        }
        # THE ONE NULL THIS LEG EARNS BY MEASUREMENT RATHER THAN BY DECLARATION.
        # Removing the tail guard cannot show up in a volume comparison -- p_out is
        # the state's own _scratch and the surplus lanes write past its last cell --
        # so the null is accepted ONLY if a direct experiment shows those lanes ran.
        # If they did not, the fixture's block geometry left nothing to arm and the
        # row FAILS as vacuous, which is what an unmeasured prediction would hide.
        if label == "drop_the_extent_guard" and not caught:
            overrun = measure_extent_guard_overrun(arm)
            row["overrun_experiment"] = overrun
            row["predicted_null"] = overrun["why"] if overrun["armed"] else None
        row["ok"] = bool(applied["count"] > 0
                         and (caught or row["predicted_null"] is not None))
        rows.append(row)

    # 2. THE HOST LOOP.
    for label, predicted in HOST_MUTATIONS:
        # A DEFECT THE SHIPPED LAUNCHER REFUSES IS CAUGHT, NOT A CRASH, and it has to
        # be scored rather than raised through. `drive_from_the_stored_E` substitutes
        # the stored E for `Fields.drive_field`, which on the PML arm is a DIFFERENT
        # array than the certified `f_w_<c>`: `launch_three_slot_polarization_
        # component` asks `_drive_identity_problem` before it binds anything and
        # raises ValueError by name. That is the guard doing its job -- the strongest
        # possible outcome for a mutation -- but an uncaught raise here would take the
        # whole gate down instead of recording it. Only the launcher's own refusal is
        # scored this way; the exception TYPE and TEXT go in the row so a reader can
        # see which clause fired rather than trusting that one did.
        refusal: Optional[str] = None
        try:
            case = run_case(arm, spec, "uniform", min(steps, 12),
                            polarization=mutated_polarization(arm, label),
                            label_suffix=f"|host:{label}")
        except ValueError as error:
            refusal = f"{type(error).__name__}: {error}"
            case = {"bit_identical": False, "refused_by_name": refusal}
        caught = not case.get("bit_identical", False)
        # THE PREDICTION IS ARM-DEPENDENT for exactly one mutation, and saying so is
        # the point: with an INACTIVE layer `Fields.drive_field` hands back the stored
        # E itself (fields.py:1160-1162), so the substitution is the identity and a
        # null is correct; with an ACTIVE one it names `f_w_<c>` and the same edit is
        # a real defect that must not pass. One reason for both arms would have
        # excused a PML divergence that should never be excused.
        prediction = (None if label == "drive_from_the_stored_E" and arm == "pml"
                      else predicted)
        rows.append({
            "kind": "host", "label": label, "arm": arm,
            "armed": True, "caught": bool(caught),
            "refused_by_name": refusal,
            "first_divergence": case.get("first_divergence"),
            "differing_words": case.get("differing_words"),
            "predicted_null": prediction,
            "ok": (not caught) if label == "none"
                  else bool(caught or prediction is not None),
        })

    twin = next(row for row in rows if row["label"] == "none")
    return {"passed": bool(all(row["ok"] for row in rows) and not twin["caught"]),
            "arm": arm, "label": spec["label"], "rows": rows,
            "no_defect_twin_uncaught": not twin["caught"],
            "caught": sum(1 for row in rows if row["caught"]),
            "nulls": [row["label"] for row in rows
                      if not row["caught"] and row["label"] != "none"],
            "unarmed": [row["label"] for row in rows if not row["armed"]],
            "seconds": time.time() - started}


# ---------------------------------------------------------------------------
# THE STRUCTURAL LEGS
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` must be a CONTIGUOUS run of the driver's own pass list.

    A gap would be a driver pass the launches skipped while the array path performed
    it -- which is the difference between a product that owns a span and one that owns
    a subsequence of it.
    """
    rows = []
    for arm in ARMS:
        replaces = list(fused_passes(arm))
        positions = [DRIVER_ORDER.index(name) for name in replaces
                     if name in DRIVER_ORDER]
        contiguous = (len(positions) == len(replaces)
                      and positions == list(range(positions[0],
                                                  positions[0] + len(positions))))
        # THE NO-ABSORBER ARM IS NOT CONTIGUOUS AND MUST NOT BE: its D->E half
        # REFUSES a mirror plane, so the two symmetry fills are inert on every row it
        # admits and naming them would claim work the launch never has. What IS
        # required of it is that its span be an ordered SUBSEQUENCE ending at
        # update_P.
        rows.append({"arm": arm, "replaces": replaces,
                     "contiguous": bool(contiguous),
                     "ordered_subsequence": positions == sorted(positions),
                     "ends_at_update_P": replaces[-1] == "update_P",
                     "starts_at_step_D": replaces[0] == "step_D",
                     "ok": bool(positions == sorted(positions)
                                and replaces[-1] == "update_P"
                                and replaces[0] == "step_D")})
    return {"passed": all(row["ok"] for row in rows), "rows": rows,
            "driver_order": list(DRIVER_ORDER)}


def leg_lift() -> Dict[str, Any]:
    """The emitted text against the certified strings, on the device host too.

    The host suite pins this on a laptop; it is re-asked here because a gate that
    certified a body has to have looked at the body it certified.
    """
    from meep_gpu.cuda_kernels import fused_polarization_pair as ep  # noqa: PLC0415

    module = family("pml")
    sources = module.device_sources()
    guard_present = all(ep._ADE_GUARD in source  # noqa: SLF001
                        for source in sources.values())
    drive_once = all(source.count("    float w = drive[idx];\n") == 1
                     for source in sources.values())
    arm_independent = (family("no_pml").device_sources() == sources)
    return {"passed": bool(guard_present and drive_once and arm_independent
                           and sources),
            "specializations": len(sources),
            "certified_guard_restored": guard_present,
            "certified_drive_load_once_per_component": drive_once,
            "device_text_is_arm_independent": arm_independent,
            "lift_edits": [row["line"] for row in module.LIFT_EDITS]}


# ---------------------------------------------------------------------------
# THE RUN
# ---------------------------------------------------------------------------

#: The files this gate BINDS a verdict to, curated rather than "everything imported"
#: -- which ``gate_provenance.stamp`` records separately under its own key. A
#: certification block names these four: the two product modules, the composer that
#: installs them and the E->P module whose lift machinery they import.
SUBJECT_FILES: Tuple[str, ...] = (
    "meep_gpu/cuda_kernels/three_slot_dispersive_weld.py",
    "meep_gpu/cuda_kernels/no_pml_three_slot_dispersive_weld.py",
    "meep_gpu/cuda_kernels/fused_pairs.py",
    "meep_gpu/cuda_kernels/fused_polarization_pair.py",
)


def _subject_digests() -> Dict[str, str]:
    import hashlib  # noqa: PLC0415

    out: Dict[str, str] = {}
    for relative in SUBJECT_FILES:
        path = os.path.join(_REPO_API, *relative.split("/"))
        with open(path, "rb") as handle:
            out[relative] = hashlib.sha256(handle.read()).hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="DIRECTORY for gate.json")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--legs", default="lift,driver_order,refusal,product,"
                                          "launch_structure,deposit,mutations")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the specs per arm (0 = all)")
    # THE FLOAT32 SUBNORMAL POLICY, spelled exactly as every sibling gate spells it.
    # REQUIRED, never defaulted: which policy a run compiled under is part of what a
    # certification block claims, and a gate that picked one silently would let a
    # block say "both policies" on two runs of the same one.
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    arguments = parser.parse_args()
    out = arguments.out
    os.makedirs(out, exist_ok=True)
    artifact = os.path.join(out, "gate.json")
    legs = tuple(name.strip() for name in arguments.legs.split(",") if name.strip())
    chosen = tuple(name.strip() for name in arguments.arms.split(",") if name.strip())

    started = time.time()
    record: Dict[str, Any] = {
        "gate": "cuda_three_slot_weld",
        "families": ["cuda_three_slot_dispersive_weld",
                     "cuda_three_slot_no_pml_dispersive_weld"],
        "kernel": "three_slot_polarization_real",
        "steps": arguments.steps,
        "arms": list(chosen),
        "legs_requested": list(legs),
        "subnormal_policy": arguments.subnormal_policy,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(sys.argv[1:]),
        "question": (
            "does ONE product spanning step_D -> update_E -> update_P leave every "
            "stored volume -- every P and P_prev buffer included -- byte-identical "
            "to the array path, to the certified singles, to the D->E weld plus the "
            "certified ADE launches AND to the certified step_D plus the E->P weld, "
            "per COMPLETE driver step, with an electric source deposited inside its "
            "first seam and the repair sitting between its two device groups?"),
        "subject_sha256": _subject_digests(),
    }

    def save() -> None:
        # EVERY IMPORTED REPO MODULE'S BYTES, and a mid-run rewrite RAISES rather
        # than being merged: measurements taken on either side of a rewrite describe
        # different programs. The shared stamper is the one writer.
        gate_provenance.stamp(record)
        with open(artifact, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, default=str)

    def emit(line: str) -> None:
        print(line, flush=True)

    if "lift" in legs:
        record["lift"] = leg_lift()
        emit(f"lift: passed={record['lift']['passed']} "
             f"specializations={record['lift']['specializations']}")
        save()
    if "driver_order" in legs:
        record["driver_order"] = leg_driver_order()
        emit(f"driver_order: passed={record['driver_order']['passed']}")
        save()
    if cp is None:
        record["passed"] = False
        record["why"] = "CuPy is not importable on this host; only the source legs ran"
        save()
        emit(json.dumps({"passed": False, "why": record["why"]}))
        return 1

    # THE POLICY GOES IN BEFORE ANY KERNEL IS COMPILED, and the observer goes in
    # before the policy: under 'keep' the policy's strip WRAPS the observer, so what
    # is recorded is the option tuple NVRTC was really given rather than the one this
    # file passed. Both installs are the shared probe's, never a second copy.
    if arguments.import_meep_for_host_policy:
        record["meep_host_import"] = probe.import_meep_for_host_policy()
    record["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    record["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
        arguments.subnormal_policy, _REPO_API)
    record["environment"] = probe.device_info()
    record["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save()
    emit(f"policy: {arguments.subnormal_policy} installed; "
         f"device={record['environment'].get('device_name')}")

    if "refusal" in legs:
        record["refusal"] = leg_refusal()
        emit(f"refusal: passed={record['refusal']['passed']} "
             f"checked={record['refusal']['checked']}")
        save()

    for arm in chosen:
        rows = specs(arm)
        if arguments.limit:
            rows = rows[:arguments.limit]
        if "product" in legs:
            cases = record.setdefault("product", [])
            for index, spec in enumerate(rows, start=1):
                for value_class in VALUE_CLASSES:
                    case = run_case(arm, spec, value_class, arguments.steps)
                    cases.append(case)
                    emit(f"product {arm} {index}/{len(rows)} {spec['label']}"
                         f"|{value_class}: passed={case.get('passed')} "
                         f"identical={case.get('bit_identical')} "
                         f"steps={case.get('steps_run')} "
                         f"launches={case.get('launch_counts', {}).get('expected_per_step')}"
                         f"/step ({case.get('seconds', 0):.1f} s)")
                    save()
        if "launch_structure" in legs:
            cases = record.setdefault("launch_structure", [])
            for spec in rows[:2]:
                leg = leg_launch_structure(arm, spec, min(arguments.steps, 20))
                cases.append(leg)
                per_step = {name: round(block["per_step"], 3)
                            for name, block in leg["memo_counter"].items()}
                graph = {name: block.get("kernel_launches", block.get("reason"))
                         for name, block in leg["graph_counter"].items()}
                emit(f"launches {arm} {spec['label']}: passed={leg['passed']} "
                     f"memo/step={per_step} graph/step={graph}")
                save()
        if "deposit" in legs:
            cases = record.setdefault("deposit", [])
            for spec in rows[:3]:
                leg = leg_deposit(arm, spec, min(arguments.steps, 20))
                cases.append(leg)
                emit(f"deposit {arm} {spec['label']}: passed={leg['passed']} "
                     f"modes={[(r['mode'], r.get('passed')) for r in leg['rows']]}")
                save()
        if "mutations" in legs:
            cases = record.setdefault("mutations", [])
            for spec in rows[:2]:
                leg = leg_mutations(arm, spec, arguments.steps)
                cases.append(leg)
                emit(f"mutations {arm} {spec['label']}: passed={leg['passed']} "
                     f"caught={leg['caught']} nulls={leg['nulls']} "
                     f"unarmed={leg['unarmed']}")
                save()

    verdicts = {name: value for name, value in record.items()
                if isinstance(value, dict) and "passed" in value}
    lists = {name: value for name, value in record.items()
             if isinstance(value, list) and value
             and isinstance(value[0], dict) and "passed" in value[0]}
    record["summary"] = {
        "legs": {name: bool(block["passed"]) for name, block in verdicts.items()},
        "case_groups": {name: {"cases": len(rows),
                               "passed": sum(1 for r in rows if r.get("passed"))}
                        for name, rows in lists.items()},
        "elapsed_s": round(time.time() - started, 1),
    }
    record["passed"] = bool(
        all(block["passed"] for block in verdicts.values())
        and all(r.get("passed") for rows in lists.values() for r in rows))
    save()
    print(json.dumps(record["summary"], indent=1), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

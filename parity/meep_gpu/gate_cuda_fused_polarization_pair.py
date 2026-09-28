"""Byte-identity gate for the E->P FUSED pair: ``update_E`` -> ``update_P`` welded.

WHAT IS UNDER TEST, and it is a claim about the WELD, which no verdict on either
half establishes:

* ``fused_polarization_pair.run_fused_polarization_pair`` -- THREE per-component
  launches with the host rotation between them -- reproduces the driver's own
  seam, ``stepping.update_E`` immediately followed by ``stepping.update_P``
  (driver.py:3313/:3315), byte for byte over EVERY array the seam reads or
  writes, THE POLE VOLUMES INCLUDED. Two arms, gated separately:
  ``--arm pml`` (dispersive update_E under an active layer x PML ADE) and
  ``--arm no_pml`` (the stored-E arm x layerless ADE).
* the same weld reproduces the SEPARATELY CERTIFIED kernels it replaces --
  ``dispersive_kernels.update_E_fused_pml_real_dispersive`` /
  ``update_E_no_pml_real_dispersive`` followed by
  ``ade_kernels.update_P_fused_pml_real`` -- word for word, from the same frozen
  state. Both references are scored per case; a weld identical to the engine but
  not to the certified kernels (or the reverse) is a defect in somebody's
  transcription and must be SEEN, not averaged away.
* the LAUNCH ACCOUNTING: 3 fused launches per seam step, counted TWICE and
  independently -- once by the launcher's own result and once by a counting shim
  around the compiled kernel -- against ``1 + sum over states of len(driven())``
  launches on the certified route. A gate that never proved the fused kernels
  ran would certify the certified kernels twice.

THE SEAM NULL AND ITS CONTROL, the pair this gate exists to arm. The weld's one
arithmetic claim is that the register hand-off ``float w = src_<axis>;`` IS the
certified drive load ``float w = drive[idx];``:

* ``reload_drive_from_global`` rewrites the register back into a load of the
  array the E half just stored (``f_w_<c>`` on the PML arm, the stored ``E<c>``
  without one). It MUST come back UNCAUGHT -- a float32 word stored to global and
  reloaded is the identity on the bits, and this null is what shows the register
  IS that word rather than a number that happens to be close.
* ``wrong_drive_register`` rewrites it to ``D<axis>[idx]`` -- the certified ADE
  gate's own wrong-drive control, restated against the fused text. It MUST
  DIVERGE; a leg whose control cannot fail has not shown that handing the RIGHT
  register across the seam is what produced the pass.

THE ROTATION LEGS. ``update_P``'s three buffers rotate per component on the
host, and the weld transcribes that rotation between its own launches. The
defect that "still computes" -- a launcher that cached the views and so advanced
one buffer twice while freezing another -- is armed THREE ways
(``stale_pole_pointers``, ``rotate_before_launch``,
``freeze_one_states_rotation``), each against a LOCAL copy of the shipped loop,
with ``launcher_copy_NULL`` -- the same copy WITHOUT the defect -- required
UNCAUGHT so the copy is shown equivalent before the defect is blamed.

FIXTURE TRAPS CARRIED FORWARD from the campaign log, each a floor here: the pole
volumes are IN the byte comparison and are seeded NON-ZERO (zero poles are a
fixed point of the subtraction chain); ``pole_subtraction_bites`` refuses a case
whose chain does not bite; ``oracle_moved`` refuses a fixed-point case; the
inverse-epsilon and absorber-profile floors are the certified sibling gate's
own, imported from it rather than re-derived.

USAGE (device host; this gate has no NumPy backend -- the weld's protocol is
pinned by ``test_fused_polarization_pair.py`` on the merge bar, and a NumPy
transcription of the fused kernel would be a third implementation of the seam)::

    python -u gate_cuda_fused_polarization_pair.py --arm pml \\
        --subnormal-policy keep \\
        --out results/cuda_fused_polarization_pair_<date>/keep_pml/gate.json
    python -u gate_cuda_fused_polarization_pair.py --arm no_pml \\
        --subnormal-policy flush --import-meep-for-host-policy \\
        --out results/cuda_fused_polarization_pair_<date>/flush_no_pml/gate.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
except ImportError:  # the gate refuses to run; the module is importable anyway
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_dispersive as sibling  # noqa: E402 - the fixture machinery

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import dispersive_kernels as dispersive  # noqa: E402
from meep_gpu.cuda_kernels import fused_polarization_pair as weld  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260901

#: The certified records' budget; on THIS product it is also the only leg that
#: can see every rotation defect, because the buffers move once per component
#: per step and a wrong assignment can be value-correct for one step.
MULTI_STEP_BUDGET = 60

ARMS: Tuple[str, ...] = ("pml", "no_pml")

#: The sibling gate's arm names for the same fixtures, used when borrowing its
#: seed/restore machinery: arm1 is the PML dispersive fixture shape, arm2 the
#: layerless one.
_SIBLING_ARM = {"pml": "arm1", "no_pml": "arm2"}

COURANTS: Tuple[float, ...] = sibling.COURANTS
INEXACT_COURANT = sibling.INEXACT_COURANT
VALUE_CLASSES: Tuple[str, ...] = sibling.VALUE_CLASSES

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: The sibling gate's case table, re-partitioned by the arm each spec's absorber
#: state serves. Same shapes, same folds, same walls -- the ten corpus rows this
#: product serves are the same rows the halves' gate answered for.
SPECS = sibling.SPECS
ARMS_BY_SPEC: Dict[str, Tuple[str, ...]] = {
    spec["label"]: (("pml",) if spec["pml"] else ("no_pml",)) for spec in SPECS}

ARITIES: Tuple[Tuple[int, int, int], ...] = dispersive.POLE_COUNTS_SWEPT

#: Arities a pole mutation may be scored at -- two and above, the sibling's own
#: measured bound (sum_then_subtract is EXACTLY inert at one pole).
MUTATION_ARITIES: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2), (5, 5, 5))

#: The mixed arity, which is this gate's own reduction control: one launch of
#: the sweep is then the E-half-alone kernel beside two that carry recurrences.
MIXED_ARITY: Tuple[int, int, int] = (2, 0, 3)


# ---------------------------------------------------------------------------
# SOURCE MUTATIONS, spelled against the EMITTED fused text
# ---------------------------------------------------------------------------

def _mutate_wrong_drive_register(source: str) -> Tuple[str, int]:
    """``float w = src_<a>;`` becomes ``float w = D<a>[idx];`` -- the certified
    ADE gate's wrong-drive control, restated against the fused text. MUST DIVERGE."""
    sites = 0
    out = source
    for target, source_name, axis in weld.ELECTRIC_TERMS:
        needle = f"    float w = src_{axis};\n"
        if needle in out:
            out = out.replace(needle, f"    float w = {source_name}[idx];\n", 1)
            sites += 1
    return out, sites


def _mutate_reload_drive_from_global(source: str) -> Tuple[str, int]:
    """THE SEAM NULL: the register becomes a reload of the array the E half just
    stored. MUST BE UNCAUGHT -- this is what shows the register IS the certified
    load rather than a number that happens to be close."""
    sites = 0
    out = source
    for target, _source_name, axis in weld.ELECTRIC_TERMS:
        needle = f"    float w = src_{axis};\n"
        if needle not in out:
            continue
        stored = (f"f_w_{target}" if f"f_w_{target}" in out else target)
        out = out.replace(needle, f"    float w = {stored}[idx];\n", 1)
        sites += 1
    return out, sites


def _mutate_regroup_recurrence(source: str) -> Tuple[str, int]:
    """Right-associate the two additions. float32 addition is not associative."""
    pattern = (r"\(\(p_(\d+) \* c_now_\1\) \+ \(c_prev_\1 \* q_\1\)\) "
               r"\+ \(c_drive_\1 \* \(s_\1 \* w\)\)")
    replacement = (r"(p_\1 * c_now_\1) + ((c_prev_\1 * q_\1) "
                   r"+ (c_drive_\1 * (s_\1 * w)))")
    return re.subn(pattern, replacement, source)


def _mutate_distribute_drive(source: str) -> Tuple[str, int]:
    """``c_drive * (s * w)`` becomes ``(c_drive * s) * w`` -- reassociating the
    drive product, note 2 of the certified ADE header."""
    pattern = r"\(c_drive_(\d+) \* \(s_\1 \* w\)\)"
    return re.subn(pattern, r"((c_drive_\1 * s_\1) * w)", source)


def _mutate_write_p_in_place(source: str) -> Tuple[str, int]:
    """The store lands on ``P^n`` instead of the scratch: the second-order
    recurrence silently reduced to first order, PLUS a scratch the rotation then
    promotes unwritten."""
    pattern = r"^    p_out_(\d+)\[idx\] = "
    sites = len(re.findall(pattern, source, flags=re.M))
    target = re.search(r"P_(E[xyz])_0", source)
    if not target:
        return source, 0
    component = target.group(1)
    out = re.sub(pattern, rf"    ((float*)P_{component}_\1)[idx] = ", source,
                 flags=re.M)
    return out, sites


def _mutate_load_reorder(source: str) -> Tuple[str, int]:
    """A NULL: the sigma load moves above the history load. Loads commute."""
    pattern = (r"    float q_(\d+) = p_prev_\1\[idx\];\n"
               r"(    float s_\1 = [^\n]+\n)")
    return re.subn(pattern, r"\2    float q_\1 = p_prev_\1[idx];\n", source)


SOURCE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "wrong_drive_register": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": True,
        "why": "D handed across the seam instead of the E half's register -- the "
               "single most likely silent wrong answer in dispersion, restated "
               "against the fused text"},
    "reload_drive_from_global": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": False,
        "why": "THE SEAM NULL: reloading the just-stored word must be the "
               "identity on the bits, or the register hand-off itself is not "
               "the certified arithmetic"},
    "regroup_recurrence": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": True,
        "why": "the recurrence's left association is the array path's two +="},
    "distribute_drive_product": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": True,
        "why": "sigma*w first, then c_drive: reassociating rounds differently"},
    "write_p_now_in_place": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": True,
        "why": "p_out is a third buffer; in-place destroys P^n and promotes an "
               "unwritten scratch"},
    "sum_then_subtract": {
        "arms": ("pml", "no_pml"), "pole_defect": True, "must_be_caught": True,
        "why": "the chain pre-accumulated; inert at one pole, lethal at two"},
    "drop_one_pole": {
        "arms": ("pml", "no_pml"), "pole_defect": True, "must_be_caught": True,
        "why": "the last contributor missing from the chain (its recurrence "
               "block stays, so this is the E half's defect alone)"},
    "drop_fw_store_dispersive": {
        "arms": ("pml",), "pole_defect": False, "must_be_caught": True,
        "why": "the f_w store dropped: breaks the E half AND the array the "
               "certified drive load would have read"},
    "load_reorder": {
        "arms": ("pml", "no_pml"), "pole_defect": False, "must_be_caught": False,
        "why": "a NULL: loads commute; a battery of only-must-catch legs scores "
               "the same whether the comparator works or fails everything"},
}

SOURCE_TRANSFORMS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "wrong_drive_register": _mutate_wrong_drive_register,
    "reload_drive_from_global": _mutate_reload_drive_from_global,
    "regroup_recurrence": _mutate_regroup_recurrence,
    "distribute_drive_product": _mutate_distribute_drive,
    "write_p_now_in_place": _mutate_write_p_in_place,
    "sum_then_subtract": sibling._mutate_sum_then_subtract,
    "drop_one_pole": sibling._mutate_drop_one_pole,
    "drop_fw_store_dispersive": sibling._mutate_drop_fw_store,
    "load_reorder": _mutate_load_reorder,
}

#: Host mutations: they corrupt the LAUNCH PROTOCOL rather than the device text.
#: Each rotation defect runs a LOCAL COPY of the shipped loop with the defect
#: applied; ``launcher_copy_NULL`` is the same copy without one and must be
#: UNCAUGHT, or the copy itself is what diverged.
HOST_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "stale_pole_pointers": {
        "arms": ("pml", "no_pml"), "must_be_caught": True,
        "why": "every component's (p_out, p_now, p_prev) resolved ONCE before "
               "any launch: all three components then share one p_out, and the "
               "second writes the buffer the first component's rotation already "
               "promoted into P -- stale in a way that still computes. The "
               "shipped launcher re-resolves per launch and its aliasing check "
               "refuses this binding, so the leg runs the local copy"},
    "rotate_before_launch": {
        "arms": ("pml", "no_pml"), "must_be_caught": True,
        "why": "the rotation hoisted above the launch: the kernel writes the "
               "retired history and the promoted P slot was never written"},
    "freeze_one_states_rotation": {
        "arms": ("pml", "no_pml"), "must_be_caught": True,
        "why": "one state's rotation skipped: one buffer advanced twice and "
               "another frozen -- the exact wording of the standing hazard"},
    "swap_kps_kms": {
        "arms": ("pml",), "must_be_caught": True,
        "why": "(kap+sig) and (kap-sig) exchanged in the tables this launcher "
               "binds: the absorber runs backwards"},
    "wrong_axis_tables": {
        "arms": ("pml",), "must_be_caught": True,
        "why": "every component handed axis x's coefficient vectors: component "
               "c must read AXIS c's table"},
    "launcher_copy_NULL": {
        "arms": ("pml", "no_pml"), "must_be_caught": False,
        "why": "the local copy of the shipped loop with NO defect: must be "
               "UNCAUGHT, which is what licenses blaming the defects above on "
               "the defect rather than on the copy"},
}


# ---------------------------------------------------------------------------
# THE FIXTURE (the sibling gate's, borrowed whole) AND THE SNAPSHOT
# ---------------------------------------------------------------------------

def build_case(xp, spec, courant, arity, rng):
    return sibling.build(xp, spec, courant, arity, rng)


def seed_case(fields, grid, arm: str, value_class: str, arity, rng):
    return sibling.seed_state(fields, grid, _SIBLING_ARM[arm], value_class,
                              arity, rng)


def snapshot_seam(fields, arm: str) -> Dict[str, np.ndarray]:
    """EVERY array the seam reads or writes, the pole volumes included.

    ``D`` is input-only and is still here: a weld that scribbled on its source
    would otherwise pass. ``_scratch`` is here because after a full seam step it
    holds the retired history, and the two paths must agree on the WHOLE
    seven-buffer state, not only the live slots.
    """
    names = ["Ex", "Ey", "Ez", "Dx", "Dy", "Dz"]
    if arm == "pml":
        names += ["f_w_Ex", "f_w_Ey", "f_w_Ez"]
    out = {name: to_host(getattr(fields, name)).copy() for name in names}
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            out[f"P{index}_{component}"] = to_host(state.P[component]).copy()
            out[f"Pprev{index}_{component}"] = to_host(
                state.P_prev[component]).copy()
        out[f"scratch{index}"] = to_host(state._scratch).copy()
    return out


def full_state(fields, arm: str) -> Dict[str, Any]:
    return sibling.full_state(fields, _SIBLING_ARM[arm])


def restore(fields, frozen) -> None:
    sibling.restore(fields, frozen)


def advance_sources(fields) -> None:
    """The curl's stand-in between seam steps: D moves, exactly the same exact
    float32 scale on every path."""
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = getattr(fields, name) * sibling._ADVANCE


# ---------------------------------------------------------------------------
# THE THREE ROUTES THROUGH THE SEAM
# ---------------------------------------------------------------------------

def seam_oracle(fields, layer) -> None:
    """The driver's own seam: update_E then update_P, nothing between
    (driver.py:3313/:3315)."""
    stepping.update_E(fields, layer)
    stepping.update_P(fields, layer)


def seam_certified(fields, layer, arm: str) -> int:
    """The two separately certified kernels, launched as their own launchers
    launch them. Returns the launch count (1 + one per (state, component))."""
    from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415

    if arm == "pml":
        dispersive.update_E_fused_pml_real_dispersive(fields, layer)
    else:
        dispersive.update_E_no_pml_real_dispersive(fields)
    return 1 + int(ade_kernels.update_P_fused_pml_real(fields, layer))


class _CountingKernel:
    """Wraps a compiled kernel; every invocation is one FUSED launch."""

    def __init__(self, kernel, counter: List[int]):
        self._kernel = kernel
        self._counter = counter

    def __call__(self, *args, **kwargs):
        self._counter[0] += 1
        return self._kernel(*args, **kwargs)


def seam_weld(fields, layer, arm: str, counter: Optional[List[int]] = None,
              tables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The weld, with an INDEPENDENT launch count when asked for."""
    if counter is None:
        return weld.run_fused_polarization_pair(fields, layer, arm,
                                                tables=tables)
    real_get = weld._get_kernel
    weld._get_kernel = lambda *a, **k: _CountingKernel(real_get(*a, **k), counter)
    try:
        return weld.run_fused_polarization_pair(fields, layer, arm,
                                                tables=tables)
    finally:
        weld._get_kernel = real_get


# ---------------------------------------------------------------------------
# THE HOST-MUTATION LOOPS: a LOCAL COPY of the shipped loop, defect applied
# ---------------------------------------------------------------------------

def _local_launch_component(fields, arm: str, target: str, states,
                            tables: Optional[Dict[str, Any]],
                            pointers: Optional[Dict[int, Tuple[Any, Any, Any]]]
                            ) -> None:
    """The shipped per-component launch, transcribed, with an optional pointer
    override per state -- the ONE thing the shipped launcher makes impossible.

    ``pointers[index]`` is a ``(p_out, p_now, p_prev)`` triple captured at some
    earlier moment; the shipped launcher resolves all three from the live state
    immediately before every launch AND refuses an aliased binding, which is
    exactly why the stale-pointer defect cannot be armed through it. The
    aliasing check is deliberately absent HERE, because what the leg measures is
    what that check prevents. ``launcher_copy_NULL`` runs this same transcription
    with no override and must be UNCAUGHT.
    """
    from meep_gpu.cuda_kernels.coverage import ade_sigma_is_volume  # noqa: PLC0415

    spec_index = {t: i for i, (t, _s, _a) in enumerate(weld.ELECTRIC_TERMS)}
    component_index = spec_index[target]
    axis = weld.ELECTRIC_TERMS[component_index][2]
    kinds = tuple(bool(ade_sigma_is_volume(state, target)) for state in states)
    stored = getattr(fields, target)
    n_elem = int(stored.size)
    arguments: List[Any] = [stored]
    if arm == "pml":
        arguments.append(getattr(fields, f"f_w_{target}"))
    arguments.append(getattr(fields, "D" + target[1]))
    arguments.append(fields.inverse_epsilon_for(target))
    for index, state in enumerate(states):
        p_now = pointers[index][1] if pointers else state.P[target]
        arguments.append(p_now)
    for index, state in enumerate(states):
        p_out = pointers[index][0] if pointers else state._scratch
        p_prev = pointers[index][2] if pointers else state.P_prev[target]
        arguments.append(p_out)
        arguments.append(p_prev)
        sigma = state.sigma[target]
        arguments.append(sigma if kinds[index] else np.float32(sigma))
        arguments.extend(np.float32(value) for value in state._coefficients)
    if arm == "pml":
        nx, ny, nz = stored.shape
        arguments.extend([np.int32(nx), np.int32(ny), np.int32(nz),
                          tables[f"kps_{axis}"], tables[f"kms_{axis}"]])
    else:
        arguments.append(np.int32(n_elem))
    blocks = (n_elem + weld._FUSED_THREADS - 1) // weld._FUSED_THREADS
    weld._get_kernel(arm, component_index, len(states), kinds)(
        (blocks,), (weld._FUSED_THREADS,), tuple(arguments))


def _weld_local_loop(fields, layer, arm: str, defect: Optional[str],
                     tables: Optional[Dict[str, Any]] = None) -> int:
    """The shipped ``run_fused_polarization_pair`` loop, copied, with one defect.

    THE COPY IS THE INSTRUMENT: a defect cannot be armed inside the shipped
    launcher without editing it -- the shipped path re-resolves every pointer per
    launch and refuses aliased bindings -- so the loop is restated here and
    ``launcher_copy_NULL`` (defect=None) is required UNCAUGHT to show the copy
    equivalent before any defect is blamed.

    ``stale_pole_pointers`` is the standing hazard verbatim: every component's
    ``(p_out, p_now, p_prev)`` triple is resolved ONCE, before any launch, "the
    way the certified constitutive launcher may cache its coefficient tables" --
    so all three components share one p_out and the second component writes the
    buffer the first component's rotation already promoted into P.
    """
    if arm == "pml" and tables is None:
        tables = dispersive.dispersive_tables(layer)
    launches = 0
    hoisted: Optional[Dict[str, Dict[int, Tuple[Any, Any, Any]]]] = None
    if defect == "stale_pole_pointers":
        hoisted = {}
        for target, _source, _axis in weld.ELECTRIC_TERMS:
            states = weld.component_specs(fields)[target]["states"]
            hoisted[target] = {
                index: (state._scratch, state.P[target], state.P_prev[target])
                for index, state in enumerate(states)}
    for target, _source, _axis in weld.ELECTRIC_TERMS:
        states = weld.component_specs(fields)[target]["states"]
        if defect == "rotate_before_launch":
            for state in states:
                p = state.P[target]
                p_prev = state.P_prev[target]
                scratch = state._scratch
                state.P[target] = scratch
                state.P_prev[target] = p
                state._scratch = p_prev
        _local_launch_component(
            fields, arm, target, states, tables,
            hoisted[target] if hoisted is not None else None)
        launches += 1
        if defect == "rotate_before_launch":
            continue
        for index, state in enumerate(states):
            if defect == "freeze_one_states_rotation" and index == 0:
                continue
            p = state.P[target]
            p_prev = state.P_prev[target]
            scratch = state._scratch
            state.P[target] = scratch
            state.P_prev[target] = p
            state._scratch = p_prev
    return launches


def _mutated_tables(layer, mutation: Optional[str]) -> Dict[str, Any]:
    tables = dispersive.dispersive_tables(layer)
    if mutation == "swap_kps_kms":
        return {("kms" + name[3:] if name.startswith("kps")
                 else "kps" + name[3:]): value for name, value in tables.items()}
    if mutation == "wrong_axis_tables":
        return {name: tables[name[:4] + "x"] for name in tables}
    return tables


def seam_weld_mutated(fields, layer, arm: str, host_mutation: str) -> int:
    if host_mutation in ("swap_kps_kms", "wrong_axis_tables"):
        return _weld_local_loop(fields, layer, arm, None,
                                tables=_mutated_tables(layer, host_mutation))
    defect = None if host_mutation == "launcher_copy_NULL" else host_mutation
    return _weld_local_loop(fields, layer, arm, defect)


# ---------------------------------------------------------------------------
# ONE CASE
# ---------------------------------------------------------------------------

def one_case(spec: Dict[str, Any], arm: str, arity: Tuple[int, int, int],
             courant: float, value_class: str, guard: str,
             host_mutation: Optional[str] = None,
             source_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run down three routes, every word compared as uint32."""
    started = time.time()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{arm}|{'-'.join(str(v) for v in arity)}|"
            f"{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build_case(cp, spec, courant, arity, rng)
    host = seed_case(fields, grid, arm, value_class, arity, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "arm": arm, "arity": list(arity),
        "courant": courant, "value_class": value_class, "guard": guard,
        "host_mutation": host_mutation, "source_mutation": source_mutation,
        "boundaries": list(spec["boundaries"]), "mirrors": list(spec["mirrors"]),
        "structure": sibling.structure_facts(grid),
        "operand_census": operand_census(
            {name: value for name, value in host.items()
             if isinstance(value, np.ndarray)}),
    }

    predicate = (weld.covers_fused_polarization_pair if arm == "pml"
                 else weld.covers_no_pml_fused_polarization_pair)
    covered, reason = predicate(fields, layer, grid, ())
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    plan = sibling.plan_for(fields, None)
    case["pole_subtraction_bites"] = sibling.pole_subtraction_bites(fields, plan)
    case["inverse_epsilon_bites"] = sibling.inverse_epsilon_bites(fields)
    if arm == "pml":
        case["coefficient_profile_bites"] = sibling.coefficient_profile_bites(layer)

    floors: List[str] = []
    if max(arity) > 0 and case["pole_subtraction_bites"]["moved"] == 0:
        floors.append("D - sum P does not bite; every pole and drive mutation "
                      "would be a no-op about the fixture")
    if max(arity) == 0:
        floors.append("an all-zero arity has no update_P to weld; the seam "
                      "reduces to the certified dispersive gate's own reduction "
                      "control and is not scored HERE")
    if not case["inverse_epsilon_bites"]["meets_floor"]:
        floors.append("the inverse-epsilon volumes are the identity or shared")
    if arm == "pml" and not case["coefficient_profile_bites"]["meets_floor"]:
        floors.append("the absorber profile is the interior identity everywhere")
    if floors:
        case["skipped"] = floors[0]
        case["floors_failed"] = floors
        case["seconds"] = time.time() - started
        return case

    frozen = full_state(fields, arm)
    before = snapshot_seam(fields, arm)

    # Route 1: the driver's own seam.
    seam_oracle(fields, layer)
    reference = snapshot_seam(fields, arm)
    moved = sibling.moved_fraction(before, reference)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = "the seam moved no output word; a fixed point certifies nothing"
        case["seconds"] = time.time() - started
        return case
    case["oracle_expected_certified_launches"] = 1 + sum(
        len(state.driven()) for state in fields.polarizations)

    # Route 2: the separately certified kernels, same frozen state.
    restore(fields, frozen)
    certified_launches = seam_certified(fields, layer, arm)
    certified = snapshot_seam(fields, arm)
    case["certified_launches"] = certified_launches
    case["against_certified_kernels"] = combine(
        {name: bit_compare(reference[name], certified[name])
         for name in reference})

    # Route 3: the weld, same frozen state, with the two independent counts.
    restore(fields, frozen)
    counter = [0]
    if source_mutation is not None:
        transform = SOURCE_TRANSFORMS[source_mutation]
        sites = [0]

        def door(inner_arm, component, count, kinds, source):
            mutated, n = transform(source)
            sites[0] += n
            return mutated

        weld.SOURCE_TRANSFORM = door
        weld._clear_kernel_cache()
    if host_mutation is not None:
        launches = seam_weld_mutated(fields, layer, arm, host_mutation)
        case["weld_launches"] = {"reported": launches, "counted": None}
    else:
        result = seam_weld(fields, layer, arm, counter)
        case["weld_launches"] = {"reported": result["launches"],
                                 "counted": counter[0],
                                 "recurrences": result["recurrences"]}
    if source_mutation is not None:
        weld.SOURCE_TRANSFORM = None
        weld._clear_kernel_cache()
        case["mutation_sites"] = sites[0]
    got = snapshot_seam(fields, arm)
    case["single_step"] = combine(
        {name: bit_compare(reference[name], got[name]) for name in reference})
    case["weld_vs_certified"] = combine(
        {name: bit_compare(certified[name], got[name]) for name in reference})

    # Route 3 again, over the budget, against route 1 over the budget. Run for
    # mutated legs too: the rotation defects are value-correct for one step by
    # construction and only the budget can see them.
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        seam_oracle(fields, layer)
        advance_sources(fields)
    oracle_multi = snapshot_seam(fields, arm)

    restore(fields, frozen)
    if source_mutation is not None:
        weld.SOURCE_TRANSFORM = door
        weld._clear_kernel_cache()
    multi_counter = [0]
    for _ in range(MULTI_STEP_BUDGET):
        if host_mutation is not None:
            seam_weld_mutated(fields, layer, arm, host_mutation)
        else:
            seam_weld(fields, layer, arm, multi_counter)
        advance_sources(fields)
    if source_mutation is not None:
        weld.SOURCE_TRANSFORM = None
        weld._clear_kernel_cache()
    multi = snapshot_seam(fields, arm)
    case["multi_step"] = combine(
        {name: bit_compare(oracle_multi[name], multi[name])
         for name in oracle_multi})
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET
    if host_mutation is None:
        case["multi_step"]["fused_launches_counted"] = multi_counter[0]

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# THE SWEEP AND THE MUTATION BATTERIES
# ---------------------------------------------------------------------------

def case_product(arm: str, product: str):
    out = []
    specs = [spec for spec in SPECS if arm in ARMS_BY_SPEC[spec["label"]]]
    if product == "reduced":
        specs = specs[:2]
    arities = [a for a in ARITIES if max(a) > 0]
    if product == "reduced":
        arities = [(1, 1, 1), (2, 2, 2)]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform", "cancellation")
    for spec in specs:
        for arity in arities:
            for courant in courants:
                for value_class in classes:
                    out.append((spec, arity, courant, value_class))
    return out


def run_sweep(results, out_path, arm, product, guard):
    cases: List[Dict[str, Any]] = []
    plan = case_product(arm, product)
    for index, (spec, arity, courant, value_class) in enumerate(plan, start=1):
        case = one_case(spec, arm, arity, courant, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] {index}/{len(plan)} {spec['label']} n={arity} "
                f"c={courant} {value_class} SKIPPED: {case['skipped'][:70]}")
            continue
        log(f"[{guard}] {index}/{len(plan)} {spec['label']} n={arity} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"oracle={'IDENT' if case['single_step']['bit_identical'] else 'DIVERGED'} "
            f"cert={'IDENT' if case['weld_vs_certified']['bit_identical'] else 'DIVERGED'} "
            f"multi={'IDENT' if case['multi_step']['bit_identical'] else 'DIVERGED'} "
            f"launches={case['weld_launches']['counted']}/step_expected=3 "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    ran = [leg for leg in legs if not leg.get("skipped")]
    caught = [leg for leg in ran
              if not leg["single_step"]["bit_identical"]
              or not leg["multi_step"]["bit_identical"]]
    if not ran:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = "CAUGHT" if len(caught) == len(ran) else "UNCAUGHT"
    else:
        verdict = "NULL CONFIRMED" if not caught else "NULL FIRED"
    return {"ran": len(ran), "caught": len(caught), "verdict": verdict,
            "must_be_caught": must_be_caught}


def run_source_mutations(results, out_path, arm, product):
    out: Dict[str, Any] = {}
    for name, row in SOURCE_MUTATIONS.items():
        if arm not in row["arms"]:
            continue
        arities = MUTATION_ARITIES if row["pole_defect"] else ((2, 2, 2),)
        specs = [spec for spec in SPECS if arm in ARMS_BY_SPEC[spec["label"]]]
        specs = specs[:1] if product == "reduced" else specs[:2]
        legs = []
        for spec in specs:
            for arity in arities:
                case = one_case(spec, arm, arity, INEXACT_COURANT, "uniform",
                                "fmad_false", source_mutation=name)
                if not case.get("skipped") and case.get("mutation_sites", 0) == 0:
                    case["skipped"] = "NOT ARMED: the mutation matched no site"
                legs.append(case)
                results.setdefault("source_mutations", {})[name] = {
                    "why": row["why"], "legs": legs}
                save(results, out_path)
        score = _score(legs, row["must_be_caught"])
        armed = [leg for leg in legs if leg.get("mutation_sites")]
        score["armed"] = bool(armed)
        if not armed:
            score["verdict"] = "NOT ARMED"
        score["why"] = row["why"]
        score["legs"] = legs
        results["source_mutations"][name] = score
        save(results, out_path)
        log(f"[source] {name}: {score['verdict']} ({score['caught']}/{score['ran']})")
    return out


def run_host_mutations(results, out_path, arm, product):
    for name, row in HOST_MUTATIONS.items():
        if arm not in row["arms"]:
            continue
        specs = [spec for spec in SPECS if arm in ARMS_BY_SPEC[spec["label"]]]
        specs = specs[:1] if product == "reduced" else specs[:2]
        legs = []
        for spec in specs:
            for arity in ((2, 2, 2), MIXED_ARITY):
                case = one_case(spec, arm, arity, INEXACT_COURANT, "uniform",
                                "fmad_false", host_mutation=name)
                legs.append(case)
                results.setdefault("host_mutations", {})[name] = {
                    "why": row["why"], "legs": legs}
                save(results, out_path)
        score = _score(legs, row["must_be_caught"])
        score["why"] = row["why"]
        score["legs"] = legs
        results["host_mutations"][name] = score
        save(results, out_path)
        log(f"[host] {name}: {score['verdict']} ({score['caught']}/{score['ran']})")


# ---------------------------------------------------------------------------
# THE VERDICT
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    arm = results["arm"]
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [case for case in primary if not case.get("skipped")]

    if not scored:
        reasons.append("no case was scored at all")
    oracle_ok = [c for c in scored if c["single_step"]["bit_identical"]]
    cert_ok = [c for c in scored if c["weld_vs_certified"]["bit_identical"]]
    both_refs = [c for c in scored
                 if c["against_certified_kernels"]["bit_identical"]]
    multi_ok = [c for c in scored if c["multi_step"]["bit_identical"]]
    if len(oracle_ok) != len(scored):
        reasons.append(f"weld-vs-engine divergence on "
                       f"{len(scored) - len(oracle_ok)} of {len(scored)} cases")
    if len(cert_ok) != len(scored):
        reasons.append(f"weld-vs-certified-kernels divergence on "
                       f"{len(scored) - len(cert_ok)} of {len(scored)} cases")
    if len(both_refs) != len(scored):
        reasons.append(f"the two references disagree with each other on "
                       f"{len(scored) - len(both_refs)} of {len(scored)} cases; "
                       f"the weld cannot be scored against a split oracle")
    if len(multi_ok) != len(scored):
        reasons.append(f"multi-step divergence on {len(scored) - len(multi_ok)} "
                       f"of {len(scored)} cases")

    # THE LAUNCH ACCOUNTING, on every scored case, from BOTH counts.
    for case in scored:
        counts = case["weld_launches"]
        if counts["reported"] != 3 or counts["counted"] != 3:
            reasons.append(
                f"{case['label']} n={case['arity']}: the weld reported "
                f"{counts['reported']} launches and the shim counted "
                f"{counts['counted']}; the whole claim is 3 per seam step")
            break
    for case in scored:
        expected = case["oracle_expected_certified_launches"]
        if case["certified_launches"] != expected:
            reasons.append(
                f"{case['label']}: the certified route launched "
                f"{case['certified_launches']} not {expected}; the baseline the "
                f"weld replaces was not what ran")
            break
    for case in scored:
        multi_counted = case["multi_step"].get("fused_launches_counted")
        if multi_counted != 3 * MULTI_STEP_BUDGET:
            reasons.append(
                f"{case['label']}: {multi_counted} fused launches over "
                f"{MULTI_STEP_BUDGET} steps; expected {3 * MULTI_STEP_BUDGET}")
            break

    arities = {tuple(case["arity"]) for case in scored}
    missing = [a for a in ARITIES if max(a) > 0 and a not in arities]
    if missing:
        reasons.append(f"arities {missing} were not scored; the swept list is "
                       f"the claim")
    if MIXED_ARITY not in arities:
        reasons.append("the mixed arity (2,0,3) was not scored; the "
                       "E-half-alone kernel beside driven ones is this gate's "
                       "own reduction control")
    if not [c for c in scored if max(c["arity"]) >= 2]:
        reasons.append("no case with two or more poles; every pole-order defect "
                       "is provably inert below that")
    if arm == "pml":
        folded = [c for c in scored if c["mirrors"]]
        if not folded:
            reasons.append("no folded case was scored; four of this cell's "
                           "seven corpus rows are folded, and this gate is the "
                           "first to advance the ROTATION on a folded extent")
        if not [c for c in folded if len(c["mirrors"]) == 2]:
            reasons.append("no TWO-plane fold was scored; the halves admit two "
                           "planes and the weld may not inherit that unmeasured")
    classes = {case["value_class"] for case in scored}
    for needed in ("cancellation", "subnormal_band"):
        if needed not in classes:
            reasons.append(f"the {needed} value class was never scored")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is "
                           f"not inert")
        elif not leg["must_be_caught"] and leg["verdict"] != "NULL CONFIRMED":
            reasons.append(f"host NULL {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
        elif leg["must_be_caught"] and leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for name, leg in results.get("source_mutations", {}).items():
        if leg.get("verdict") in ("NOT ARMED", "NO LEGS"):
            reasons.append(f"source mutation {name} is {leg['verdict']}")
        elif leg["must_be_caught"] and leg["verdict"] != "CAUGHT":
            reasons.append(f"source mutation {name} is {leg['verdict']}")
        elif not leg["must_be_caught"] and leg["verdict"] != "NULL CONFIRMED":
            reasons.append(f"source NULL {name} is {leg['verdict']}")

    control = [case for case in sweep.get("default_no_options", [])
               if not case.get("skipped") and case["courant"] == INEXACT_COURANT]
    control_diverged = [case for case in control
                        if not case["single_step"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run" if not control else
                    "the contraction guard is load-bearing on this weld"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "guard_control": guard_control,
        "arm": arm,
        "scored_cases": len(scored),
        "weld_vs_engine_identical": len(oracle_ok),
        "weld_vs_certified_identical": len(cert_ok),
        "references_agree": len(both_refs),
        "multi_step_identical": len(multi_ok),
        "arities_scored": sorted(arities),
        "value_classes_scored": sorted(classes),
        "claim": (
            f"the {arm} arm of fused_polarization_pair -- three per-component "
            f"launches with the host rotation between them -- is byte-identical "
            f"per complete seam step to stepping.update_E followed by "
            f"stepping.update_P AND to the separately certified kernels it "
            f"replaces, pole volumes included, over {MULTI_STEP_BUDGET} "
            f"consecutive steps, with exactly 3 fused launches per step counted "
            f"two independent ways"),
        "does_not_claim": [
            "nothing dispatches this product; plan_fast_path returns None on "
            "every branch",
            "the off-diagonal dispersive cell (absorbed_power_density.py) is "
            "REFUSED by the per-component split, not measured here",
            "the complex no-PML E->P cell is another family's and stays "
            "UNDETERMINED on the board",
            "no timing claim of any kind; 3 launches replacing 1+3S is an "
            "accounting fact, not a throughput number",
        ],
    }


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"plants": [], "all_flipped": True, "inapplicable": []}
    base = summarize(results)

    def plant(name, mutate, why_absent):
        copy_of = copy.deepcopy(results)
        applied = mutate(copy_of)
        verdict = summarize(copy_of) if applied else None
        flipped = bool(applied) and not verdict["released"]
        out["plants"].append({
            "plant": name, "applicable": bool(applied),
            "why_not_applicable": None if applied else why_absent,
            "released_after_plant": None if verdict is None else verdict["released"],
            "flipped": flipped,
            "first_reason": None if verdict is None or verdict["released"]
            else verdict["reasons"][0][:180]})
        if not applied:
            out["inapplicable"].append(name)
        elif not flipped:
            out["all_flipped"] = False

    def flip_engine_case(record):
        for case in record.get("sweep", {}).get("fmad_false", []):
            if not case.get("skipped") and case["single_step"]["bit_identical"]:
                case["single_step"]["bit_identical"] = False
                return True
        return False

    def flip_certified_case(record):
        for case in record.get("sweep", {}).get("fmad_false", []):
            if not case.get("skipped") and \
                    case["weld_vs_certified"]["bit_identical"]:
                case["weld_vs_certified"]["bit_identical"] = False
                return True
        return False

    def wrong_launch_count(record):
        for case in record.get("sweep", {}).get("fmad_false", []):
            if not case.get("skipped"):
                case["weld_launches"]["counted"] = 4
                return True
        return False

    def escape_a_mutation(record):
        for leg in record.get("source_mutations", {}).values():
            if leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "UNCAUGHT"
                leg["caught"] = 0
                return True
        for leg in record.get("host_mutations", {}).values():
            if leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "UNCAUGHT"
                leg["caught"] = 0
                return True
        return False

    def fire_the_seam_null(record):
        leg = record.get("source_mutations", {}).get("reload_drive_from_global")
        if leg and leg.get("verdict") == "NULL CONFIRMED":
            leg["verdict"] = "NULL FIRED"
            leg["caught"] = leg["ran"]
            return True
        return False

    plant("a_scored_case_diverges_from_the_engine", flip_engine_case,
          "no scored case was engine-identical to begin with")
    plant("a_scored_case_diverges_from_the_certified_kernels",
          flip_certified_case, "no scored case was certified-identical")
    plant("the_fused_launch_count_reads_four", wrong_launch_count,
          "no scored case carries a launch count")
    plant("a_must_be_caught_mutation_escapes", escape_a_mutation,
          "no armed CAUGHT mutation in the record (--skip-mutations)")
    plant("the_seam_null_fires", fire_the_seam_null,
          "the seam null was not NULL CONFIRMED in this record")
    out["released_unplanted"] = base["released"]
    return out


def save(results: Dict[str, Any], out_path: str) -> None:
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_fused_polarization_pair",
        "arm": args.arm,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "product": args.product,
        "question": ("do three per-component fused launches with the host "
                     "rotation between them reproduce update_E -> update_P byte "
                     "for byte, against the engine AND against the separately "
                     "certified kernels the weld replaces?"),
        "kernel": weld.kernel_name(args.arm),
        "lift_edits": list(weld.LIFT_EDITS),
        "arities_swept": [list(a) for a in ARITIES if max(a) > 0],
        "multi_step_budget": MULTI_STEP_BUDGET,
    }

    if cp is None:
        log("[fatal] CuPy did not import; this gate has no NumPy backend")
        results["status"] = "refused: no CuPy"
        save(results, args.out)
        return 2
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = \
            probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                  _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, args.out)

    for guard, options, _primary in GUARD_SETS:
        weld._COMPILE_OPTIONS = tuple(options)
        weld._clear_kernel_cache()
        dispersive._COMPILE_OPTIONS = tuple(options)
        dispersive._clear_kernel_cache()
        from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
        ade_kernels._COMPILE_OPTIONS = tuple(options)
        ade_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.arm, args.product, guard)

    weld._COMPILE_OPTIONS = ("--fmad=false",)
    weld._clear_kernel_cache()
    dispersive._COMPILE_OPTIONS = ("--fmad=false",)
    dispersive._clear_kernel_cache()
    from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
    ade_kernels._COMPILE_OPTIONS = ("--fmad=false",)
    ade_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.arm, args.product)
        run_source_mutations(results, args.out, args.arm, args.product)

    results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["verdict_flips_against_planted_defect"] = \
        verdict_flips_against_planted_defect(results)
    if not results["verdict_flips_against_planted_defect"]["all_flipped"]:
        results["summary"]["released"] = False
        results["summary"]["reasons"].append(
            "the release verdict did NOT flip against every applicable planted "
            "defect; a verdict that cannot go red is not a verdict")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] arm={args.arm} released={verdict['released']} "
        f"scored={verdict['scored_cases']} "
        f"engine_identical={verdict['weld_vs_engine_identical']} "
        f"certified_identical={verdict['weld_vs_certified_identical']} "
        f"multi_identical={verdict['multi_step_identical']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

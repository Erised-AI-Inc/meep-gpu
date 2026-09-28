"""Host side of the Metal real-field PML step path.

Two plan builders and one composer, mirroring the Triton package's shape:

    plan_pml_curl(fields, pml, "step_B" | "step_D", residency) -> PmlCurlPlan | None
    plan_constitutive(fields, pml, "H" | "E", residency)       -> ConstitutivePlan | None
    plan_step(fields, pml, residency, sources)                 -> MetalStepPlan

A *plan* is built once for a (fields, pml, sub-step) triple and launched per
timestep. Everything that can be resolved ahead of the loop is resolved there: the
coverage verdict, the boundary specialisation, the Yee sub-lattice pairing, the
compiled shader functions, the flattened coefficient views and the device mirrors.
NOTHING ON THE LAUNCH PATH ALLOCATES, COMPILES OR COPIES — the argument tuple is
built once at plan time and :meth:`PmlCurlPlan.run` does nothing but call it.

WHAT THE PORT DELETED RATHER THAN CARRIED, each with its reason:

* ``CupyPointer`` — gone entirely. A torch tensor is ``compile_shader``'s native
  argument type, so the whole interop adapter has no analogue to port.
* ``require_triton()`` and the ``libcuda`` development-symlink diagnostic — the
  failure mode does not exist here. Torch's absence is reported by the coverage
  predicate, by name, like every other refusal.
* ``block`` / ``num_warps`` / ``DEFAULT_BLOCK`` — DELETED, not carried as
  accepted-and-ignored keywords (the rename-outright rule). Metal exposes no block or warp
  knob: the dispatch is sized from the first tensor argument's element count, and
  the kernel carries an explicit ``n_elem`` guard with an early return so a volume
  wider than the sub-step's own extent cannot be stepped.

WHAT CHANGED CATEGORY: THE CONTRACTION GUARD. Triton's ``enable_fp_fusion`` is a
per-LAUNCH keyword, so its gate can flip contraction at launch. On Metal the
contraction directive is a file-scope property of the SOURCE, so the guard is a
SOURCE-VARIANT SELECTOR: a plan holds one compiled function per requested mode and
:meth:`PmlCurlPlan.run` picks between them. ``run(contract="fast")`` on a plan that
was not built with that variant RAISES rather than silently launching the pinned
one — a guard argument that quietly did nothing would make the gate's most
important leg vacuous. Note the polarity is the opposite of Triton's keyword:
``enable_fp_fusion=True`` (contraction on, guard removed) corresponds to
``contract="fast"`` here.

THE RESIDENCY LAYER IS NEW. See :class:`.device.Residency` and
:func:`..coverage.residency_reasons`.

WHAT MOVED OUT OF THIS FILE, and why the move is safe to make. Tranche 1 put the
whole host layer here: 812 lines of which the two plan classes duplicated each
other almost verbatim and ``plan_step`` spelled its arm table as two literal
loops. A seven-family tree needs a spine, so four pieces moved and this module
re-imports every one of them:

* :mod:`.device`       compilation + memo, ``Mirror``, ``Residency``, the frontend
                       version reader — the parts no family owns;
* :mod:`.plans`        ``KernelPlan``: the ``_args``/``_functions`` slots, the
                       contraction-variant selector with its raise-on-missing
                       contract, and ``__repr__``. The two plans below now
                       subclass it and contribute ONLY their binding layout;
* :mod:`.arms`         the arm table as DATA. Each family registers a row and
                       ``plan_step`` iterates; the fail-closed contract keeps its
                       single implementation in ``triton_kernels.launch``;
* :mod:`.preconditions` the subnormal window, promoted out of the gate.

NO SHIPPED KERNEL BYTE CHANGES. ``shaders.py`` is not touched at all, so
``enumerate_sources()`` returns byte-identical strings and every certified
kernel-source sha256 stays valid. What DOES change is this module's own sha256,
and :func:`compute_fingerprints` hashes the host modules DELIBERATELY — the host
chooses the Yee sub-lattice, binds the coefficient pointers and decides the
specialisation constants, each of those a silent wrong answer if it changes and
none of them visible in a kernel source. ``parity/meep_gpu/gate_metal_spine_byte_neutrality.py``
measures exactly that split against the certified fingerprint record: kernel
hashes EQUAL, host hashes MOVED. That comparison is the evidence the refactor was
byte-neutral, and it is better evidence than not refactoring would have produced.

DISPATCH IS WIRED FOR THIS PACKAGE AS OF PHASE 3, and the sentence this paragraph
replaces ("dispatch is deliberately not wired for *this* package") is retired
rather than softened. ``meep_gpu.fastpath.plan_fast_path`` picks a kernel TABLE at
its backend rung: a CuPy engine takes the Triton table, and a NumPy engine on a
host with an MPS device takes the METAL table, whose ladder lives in
``meep_gpu/metal_dispatch.py`` and whose release rows live there too. So a product
of this package CAN now be selected, and what decides whether it dispatches is
``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` plus the per-arm axes beside it — not
the absence of a seam. ``parity/meep_gpu/dispatch_reachability`` reads that off
``fastpath``'s parse tree AND one level into the siblings it imports, which is what
turns ``reaches_the_seam`` true for backend ``metal`` from the branch itself.

WHAT THIS FILE OWES THAT SEAM is the residency bracket, and it is installed at PLAN
TIME rather than inside the driver's consult: :func:`wrap_for_residency` puts
:class:`SyncedPlan` immediately around the LAUNCH in every dispatching slot, so the
invariant across the whole step is "the host arrays are authoritative between
launches". That is what carries the driver's inject, ``fill_symmetry_bc_*``,
``zero_metal_*``, the far ghosts, the trailing repair and
``synchronize_magnetic_fields``' backup/restore-in-place with no driver edit at
all. The gate, the composition probe and the benchmark are still callers here; the
dispatch seam is now one more.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES
# The fail-closed composition machinery is backend-free and is IMPORTED rather
# than re-implemented: `_Arm` is a label plus three closures, `_select_slot` is
# the (a)-(f) contract, and STEP_ORDER is the seam order. One definition.
from ..triton_kernels.launch import (
    STEP_ORDER,
    NoopPlan,
    _guarded_plan,
    _guarded_verdict,
    _pair_may_absorb,
    _select_slot,
)
# The deposit repair is SHARED, not ported. Both tracks reach the same two plan
# classes through the same installer shape, because a second implementation of
# "carry an in-seam deposit across a fused launch" is a second thing to get wrong
# — and the failure it would produce is a constitutive half computed against a
# pre-injection field that reports success.
from .. import deposit_repair as _deposit_repair
# The withdraw hoist is SHARED for the same reason and against the same rule: the
# H->D seam carries no injection but does carry the driver's electric `withdraw`,
# and a second implementation of "perform the seam's pass before the launch" is a
# second thing to place wrong — the campaign's own null control put it one pass
# later and diverged at step 2.
from .. import withdraw_hoist as _withdraw_hoist
from . import arms, no_pml_constitutive, shaders
from ..triton_kernels.coverage import _call
from .coverage import (
    RESIDENCY_ORDER,
    Coverage,
    constitutive_coverage,
    folded_periodic_axes,
    pml_curl_coverage,
    residency_coverage,
    sub_step_volumes,
    zero_metal_axes,
)
# Re-imported, not redefined: `device` owns compilation and residency now, and
# every caller that reached for `launch.Residency` or `launch.compile_source`
# keeps working. One definition, two spellings of the path to it.
from .device import (  # noqa: F401 - re-exported deliberately
    Mirror,
    Residency,
    compile_source,
    metal_frontend_version,
)
from .plans import KernelPlan

#: Targets, auxiliaries and sources per curl sub-step, in target order. The
#: ``suffix`` is the Yee sub-lattice the PML coefficients come from: the B curl
#: reads HALF-INTEGER positions and getting it backwards is a silent half-cell
#: error, not a crash (``stepping._curl_coefficients``, stepping.py:2473).
SUB_STEPS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "targets": ("Bx", "By", "Bz"),
        "sources": ("Ex", "Ey", "Ez"),
        "backward": False,
        "suffix": "_h",
    },
    "step_D": {
        "targets": ("Dx", "Dy", "Dz"),
        "sources": ("Hx", "Hy", "Hz"),
        "backward": True,
        "suffix": "",
    },
}


# ---------------------------------------------------------------------------
# Compilation, memoized by source string
# ---------------------------------------------------------------------------

# `_LIBRARY_CACHE` and `compile_source` moved to `device.py`; the two family
# entry-point helpers below stay here because they name SHADERS' kernels.


def compile_curl(codes: Sequence[int], backward: bool,
                 contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised curl entry point for one (boundary triple, direction, mode)."""
    return compile_source(shaders.curl_source(codes, backward, contract)).pml_curl_step


def compile_constitutive(side: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised constitutive entry point for one (side, mode)."""
    return compile_source(shaders.constitutive_source(side, contract)).constitutive_step


# ---------------------------------------------------------------------------
# The curl sub-step
# ---------------------------------------------------------------------------

class PmlCurlPlan(KernelPlan):
    """A launchable, allocation-free real-field PML curl sub-step.

    Built two ways and launched ONE way. :func:`plan_pml_curl` is the engine route
    — it reads a real ``Fields``/``PML`` and passes through the coverage predicate;
    :func:`plan_from_arrays` is the gate and benchmark route, which hands over bare
    host arrays. Both produce this object and both go through
    :meth:`.plans.KernelPlan.run`, so the bytes the gate certifies are the bytes
    the engine would launch.

    THE BINDING LAYOUT IS THE ONLY THING THIS CLASS CONTRIBUTES. ``run`` and the
    contraction-variant contract are the base's — one implementation for the whole
    package, so seven families cannot spell the guard seven ways.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "residency", "volumes")

    REPR_FIELDS = ("sub_step", "shape", "bc")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, residency: Residency,
                 targets, auxiliaries, sources, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy casts to float32 before the multiply; Metal binds a Python
        # float into `constant float&` the same way, so the two scalars are the
        # same bits. Measured, not assumed — the gate's reference leg is 12/12
        # against stepping.py at two Courant numbers including 0.35.
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.volumes = tuple(volumes)
        # Built ONCE. The launch path unpacks this tuple and does nothing else.
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2],
               self.n_elem, self.dtdx))


def _curl_functions(codes, backward: bool,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_curl(codes, backward, mode) for mode in contract_variants}


def plan_pml_curl(fields: Any, pml: Any, sub_step: str,
                  residency: Optional[Residency] = None,
                  contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                  ) -> Optional[PmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    codes = [1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml)]
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return PmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, residency,
        targets, auxiliaries, sources, coefficients,
        _curl_functions(codes, spec["backward"], contract_variants), volumes)


def plan_from_arrays(sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any],
                     codes, dtdx: float, residency: Residency,
                     functions: Optional[Dict[str, Any]] = None,
                     contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                     ) -> PmlCurlPlan:
    """Build a plan from bare host arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name, ``flat`` by ``kms_x``/``sinv_x``/... on
    the sub-lattice the caller already selected, and ``codes`` is the per-axis
    0/1 periodic/metallic triple. No coverage predicate runs here: the caller is a
    harness that has constructed the configuration deliberately, including the
    deliberately wrong ones.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING — every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return PmlCurlPlan(
        sub_step, shape, dtdx, codes, residency, targets, auxiliaries, sources,
        coefficients,
        functions if functions is not None
        else _curl_functions(codes, spec["backward"], contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# The constitutive sub-step (update_H / update_E)
# ---------------------------------------------------------------------------

class ConstitutivePlan(KernelPlan):
    """A launchable, allocation-free ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen HERE and nowhere else: ``kps_a``/``kms_a`` for the
    H side, ``kps_a_h``/``kms_a_h`` for the E side (``stepping.py:948`` vs ``:986``).
    The kernel takes six coefficient pointers and never asks which lattice they came
    from, so a swap on this line is a silent half-cell error in the absorber profile
    — converged, smooth, and wrong. The gate carries a mutation for exactly it.
    """

    __slots__ = ("side", "shape", "n_elem", "residency", "volumes")

    REPR_FIELDS = ("side", "shape")

    def __init__(self, side: str, shape, residency: Residency, targets, auxiliaries,
                 sources, inverse_epsilon, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.residency = residency
        self.volumes = tuple(volumes)
        # The H side has no inverse epsilon. Bind the SOURCES as placeholders
        # rather than a null: the substituted product never reads them, but a
        # buffer argument still has to bind, and passing None makes the failure a
        # TypeError far from its cause. Same choice the Triton plan makes.
        placeholders = tuple(sources if inverse_epsilon is None else inverse_epsilon)
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + placeholders + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem))


def _constitutive_functions(side: str,
                            contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_constitutive(side, mode) for mode in contract_variants}


def plan_constitutive(fields: Any, pml: Any, side: str,
                      residency: Optional[Residency] = None,
                      contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                      ) -> Optional[ConstitutivePlan]:
    """Build a constitutive plan from the engine's own objects, or None when refused."""
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not constitutive_coverage(fields, pml, side, residency).covered:
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n), constant=True)
         for n in spec["targets"]] if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes)


def plan_constitutive_from_arrays(side: str, arrays: Dict[str, Any],
                                  flat: Dict[str, Any], residency: Residency,
                                  functions: Optional[Dict[str, Any]] = None,
                                  contract_variants: Sequence[str] = (
                                      shaders.CONTRACT_OFF,),
                                  ) -> ConstitutivePlan:
    """Build a constitutive plan from bare host arrays — the gate's route.

    ``arrays`` is keyed by component name (plus ``inv_eps_Ex``... on the E side) and
    ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER already
    selected. ``functions`` is the same mutation seam :func:`plan_from_arrays`
    documents, and dropping it disarms every constitutive mutation leg.
    """
    spec = CONSTITUTIVE_SIDES[side]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, arrays[n]) for n in spec["aux"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, arrays["inv_eps_" + n], constant=True)
         for n in spec["targets"]] if side == "E" else None)
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{side}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, shape, residency, targets, auxiliaries, sources, inverse_epsilon,
        coefficients,
        functions if functions is not None
        else _constitutive_functions(side, contract_variants), volumes)


#: Which constitutive side each slot steps. The arm callables take the SLOT and
#: look the side up here, rather than closing over it, so one pair of functions
#: serves both slots and the two registrations cannot drift apart.
CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


# ---------------------------------------------------------------------------
# Registration — tranche 1's three products, as registry rows
# ---------------------------------------------------------------------------
#
# THE ARM TABLE IS DELIBERATELY SHORT. Three products: the real-field PML curl on
# both curl slots, the ordinary `dsigw` constitutive on both constitutive slots,
# and the no-PML NULL family, whose PREDICATE is shared verbatim with the Triton
# package because it launches nothing at all — the array path's `update_H` returns
# at stepping.py:944-945 under an inactive absorber and `update_E` at :983-984
# under an inactive absorber and `stores_E` False.
#
# TRANCHE 1 RECORDED THAT THE FAMILY "NEEDED ZERO METAL WORK". That was true of the
# predicate and FALSE of the composition, and the correction is measured: see
# `no_pml_constitutive`'s module docstring for the residency verdict that falsely
# refused the family's own configuration, and for the launch-signature mismatch
# that made `run(contract=...)` a TypeError on the Triton plan object. The
# predicate is still imported and still not copied; the plan and the composition
# are Metal's.
#
# The rows below are what tranche 1 spelled as two literal loops inside plan_step.
# Nothing about the SELECTION changed — `arms.select` calls the same imported
# `_select_slot` with the same arms — only where the table lives.

for _slot in ("step_B", "step_D"):
    arms.register(
        family="pml_curl", slot=_slot, label="PML",
        coverage=(lambda ctx, slot: pml_curl_coverage(
            ctx.fields, ctx.pml, slot, ctx.residency)),
        plan=(lambda ctx, slot: plan_pml_curl(
            ctx.fields, ctx.pml, slot, ctx.residency, ctx.contract_variants)),
        prefix="PML: ", noun="PML curl", wired=True)

for _slot, _side in (("update_H", "H"), ("update_E", "E")):
    arms.register(
        family="constitutive", slot=_slot, label="ordinary",
        coverage=(lambda ctx, slot: constitutive_coverage(
            ctx.fields, ctx.pml, CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency)),
        plan=(lambda ctx, slot: plan_constitutive(
            ctx.fields, ctx.pml, CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency,
            ctx.contract_variants)),
        prefix="ordinary: ", noun="ordinary constitutive", wired=True)

del _slot, _side

# THE NULL ARM IS REGISTERED BY ITS OWN FAMILY MODULE, and it is the one row in
# this table that is not spelled here. It used to be: tranche 1 wired the Triton
# package's `null_constitutive_coverage` / `plan_null_constitutive` directly,
# having measured that the PREDICATE carries no backend clause and is therefore
# already correct on this host. The predicate still is — `no_pml_constitutive`
# IMPORTS it rather than copying it — but the COMPOSITION was not, and the two
# defects were measured on this host rather than argued:
#
#   * the Triton plan's launch signature is `run(guard=None)`. Every Metal plan is
#     launched as `run(contract=...)` because the contraction directive is a
#     property of the SOURCE here, so a composer iterating its plans under the
#     guard leg got a TypeError from that object instead of the correct answer,
#     which for a plan that performs no floating-point operation is "nothing to
#     contract";
#   * the residency verdict FALSELY REFUSED the family's own configuration. A null
#     was declared `planned`, which means "runs on the device against the mirror";
#     it runs on neither, so the verdict demanded mirrors of `Hx` and `f_w_Hx` —
#     arrays a no-PML run does not allocate at all. Recorded as
#     `residency_covered: false` on the `no_pml_no_storage` row of the tranche-1
#     artifact, which passed because leg 8 recorded the field without asserting it.
#
# So the null arm's row moved next to the module that owns those facts. Nothing
# about the SELECTION changed: same slots, same label, same absorber gate, same
# no-preference-in-either-direction contract.
no_pml_constitutive.register_arms()


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------

class MetalStepPlan:
    """Which sub-steps of one frozen configuration Metal covers, and the plans for them.

    The shape ``fastpath.FastPathPlan`` would hold, built here because dispatch is
    not wired. ``selected`` names the ARM that won each filled slot; ``residency``
    is the verdict on holding the mirror set across ONE COMPLETE STEP, which is a
    property of the composition and not of any single plan — a curl plan is correct
    on its own, and a mirror held across an array-path wall clear is not.
    """

    __slots__ = ("plans", "reasons", "selected", "residency", "live", "synced")

    def __init__(self, plans: Dict[str, Any], reasons: Dict[str, Tuple[str, ...]],
                 selected: Optional[Dict[str, str]] = None,
                 residency: Optional[Coverage] = None,
                 live: Sequence[str] = (), synced: Sequence[str] = ()) -> None:
        self.plans = dict(plans)
        self.reasons = dict(reasons)
        self.selected = dict(selected or {})
        self.residency = residency
        self.live = tuple(live)
        self.synced = tuple(synced)

    @property
    def replaces(self) -> Tuple[str, ...]:
        return tuple(name for name in STEP_ORDER if name in self.plans)

    def describe(self) -> str:
        covered = ", ".join(self.replaces) if self.replaces else "nothing"
        held = "held" if (self.residency and self.residency.covered) else "REFUSED"
        return f"MetalStepPlan(replaces=[{covered}], residency={held})"

    def __repr__(self) -> str:
        return self.describe()


class SyncedPlan:
    """A device plan bracketed by the residency's host->device / device->host copies.

    PROMOTED OUT OF ``parity/meep_gpu/gate_metal_fused_hd_pair.synced`` — the gate
    wrote this class to drive a fused pair through the engine's own seam, the
    dispatch route needs exactly the same bracket, and two spellings of "when does
    a launch see the host state" is two chances to place it wrong.

    THE BRACKET SITS IMMEDIATELY AROUND THE LAUNCH and nowhere else, because two
    wrappers in this composition do HOST work before their launch —
    ``deposit_repair.LeadingRepairPlan`` saves the deposit points and
    ``withdraw_hoist.LeadingWithdrawPlan`` performs the seam's electric withdraw —
    and the launch must read the host state AFTER that work. A shim that
    synchronised around the whole SLOT would hand the fused launch a ``D`` still
    holding the previous step's standing dipole, which is the exact defect the
    hoist exists to prevent, and report success.

    ``run_near`` / ``run_far`` ARE ANSWERED THROUGH :meth:`__getattr__` AND THAT IS
    THE CORRECTION THE LIFT NEEDED. ``fastpath.FastPathPlan.dispatch`` reaches a
    split fill through ``hasattr(plan, "run_near")`` and then calls it, so plain
    attribute forwarding would launch the near pass UNBRACKETED — the host writes
    of the pass before it would never reach the device. Defining the two as real
    methods instead would be worse: ``hasattr`` would then answer True for
    ``MirrorGhostFillPlan``, which deliberately has no ``run_near`` because it
    fuses both passes, and the dispatch would take the split branch and raise. So
    the two names are bracketed HERE, and only when the wrapped plan really has
    them.

    ``absorbed_by`` IS DECLARED RATHER THAN FORWARDED, and it is load-bearing.
    ``fastpath._pair_identity`` identifies the two slots of a fused pair by the
    object that does the work — ``absorbed_by`` where there is one, the entry
    itself otherwise. A bare pair plan names nobody, so a wrapper that merely
    forwarded the missing attribute would answer ``id(wrapper)`` on the curl slot
    while the ``NoopPlan`` in the absorbed slot answered ``id(pair)``, and the two
    halves of one pair would stop being one pair — silently disarming
    ``_split_pairs`` and the mid-step licence clause. Naming the inner plan (or the
    inner's own ``absorbed_by``, so the answer collapses in one hop the way
    ``declaring_plan`` reads it) makes the wrapped composition identify exactly as
    the unwrapped one does.
    """

    __slots__ = ("inner", "residency", "absorbed_by", "reads", "writes")

    def __init__(self, inner: Any, residency: Any,
                 reads: Any = None, writes: Any = None) -> None:
        self.inner = inner
        self.residency = residency
        self.absorbed_by = getattr(inner, "absorbed_by", None) or inner
        #: The mirror names this launch READS and WRITES, or None for "every
        #: registered mirror" — which is the shipped behaviour and stays the
        #: default, so an unscoped composition brackets exactly as it always has.
        #:
        #: SCOPING IS WHERE THE COST IS, and the measurement is unambiguous: the
        #: unscoped bracket moves 138 copies and 2,451,852 float32 words per step
        #: on ``pml_2d`` at 0.217 ms a copy, while the two kernels bind 30 and 33
        #: of the 45 registered mirrors respectively (measured 2026-09-21 by
        #: ``device.LaunchAudit`` over that composition). The cost is per copy and
        #: linear through the origin in both directions, with no fixed drain term
        #: to pay around, so removing a copy removes its cost.
        #:
        #: THE DECLARATION MUST NOT COME FROM ``volumes``. That list is not a cover
        #: of the written set — ``folded_offdiag_dispersive_update_e`` writes a
        #: pack mirror that never appears in it — so a scope derived from
        #: ``volumes`` would omit real writes and leave the host holding
        #: pre-launch bytes, which is the silent failure this layer exists to
        #: refuse. Scopes are passed in by the composer, and
        #: :class:`device.LaunchAudit` measures the true operand set to check them.
        self.reads = None if reads is None else tuple(reads)
        self.writes = None if writes is None else tuple(writes)

    @property
    def scoped(self) -> bool:
        """Whether this bracket transports a declared subset rather than everything."""
        return self.reads is not None or self.writes is not None

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.residency.sync_in(self.reads)
        self.inner.run(*args, **kwargs)
        self.residency.sync_out(self.writes)

    def _bracketed(self, name: str) -> Any:
        pass_call = getattr(self.inner, name)  # AttributeError when the inner has none.

        def call(*args: Any, **kwargs: Any) -> None:
            self.residency.sync_in(self.reads)
            pass_call(*args, **kwargs)
            self.residency.sync_out(self.writes)

        return call

    def __getattr__(self, name: str) -> Any:
        if name in ("run_near", "run_far"):
            return self._bracketed(name)
        return getattr(self.inner, name)

    def __repr__(self) -> str:
        return f"SyncedPlan({self.inner!r})"


def synced(plan: Any, residency: Any,
           reads: Any = None, writes: Any = None) -> Any:
    """Wrap the DEVICE half of whatever sits in a slot; leave host-only work alone.

    The four shapes a slot can hold after the composer runs, and the rule is the
    gate's own, verbatim:

    * a bare device plan — WRAPPED;
    * a wrapper whose ``inner`` is its ``absorbed_by`` (``LeadingRepairPlan``,
      ``LeadingWithdrawPlan``) — its ``inner`` is wrapped in place, so the host work
      it does first stays first;
    * a plan with an ``absorbed_by`` and no ``inner`` (``NoopPlan``,
      ``TrailingRepairPlan``) — LEFT ALONE: the launch is elsewhere, and the next
      launch's ``sync_in`` carries their host writes to the device;
    * a plan that declares ``performs_device_work = False`` (the no-PML null arm) —
      LEFT ALONE: it binds no buffer and stales no mirror.
    """
    if plan is None:
        return plan
    inner = getattr(plan, "inner", None)
    if inner is not None and getattr(plan, "absorbed_by", None) is inner:
        plan.inner = SyncedPlan(inner, residency, reads, writes)
        return plan
    if getattr(plan, "absorbed_by", None) is not None:
        return plan
    if not getattr(plan, "performs_device_work", True):
        return plan
    return SyncedPlan(plan, residency, reads, writes)


def wrap_for_residency(step_plan: MetalStepPlan, residency: Any,
                       scopes: Any = None) -> MetalStepPlan:
    """Install the residency bracket on every slot of a composed step, in place.

    CALLED AT PLAN TIME BY THE DISPATCH LADDER (``meep_gpu.metal_dispatch``) and by
    nothing on the launch path, which is why ``fastpath.FastPathPlan.dispatch``
    needs no residency branch at all: by the time the driver consults a slot, the
    plan sitting in it already carries its own bracket.

    ``update_P`` HOLDS ONE PLAN ON THIS TRACK, not the Triton list, and it is
    wrapped like any other device plan. The asymmetry is the ADE plan's own —
    ``ade_update_p.MetalAdeUpdatePPlan.run`` takes a contract rather than a drive
    field — and it is answered where the driver's consult is answered, not here.

    ``scopes``, when given, is ``{slot: (reads, writes)}`` — the mirror names that
    slot's launch actually reads and writes. A slot absent from the mapping keeps the
    unscoped bracket, so a partially-declared composition is correct (and merely
    slow) rather than wrong: there is no state in which omitting a declaration
    drops a sync. Adding one is what removes copies, and getting one WRONG is what
    would corrupt a field, which is why the composer's scopes are checked against
    ``device.LaunchAudit``'s measured operand sets rather than trusted.

    Returns the same object it was handed: the mutation is the point, since the
    slots this walks are the ones ``fastpath`` will read back out of
    ``step_plan.plans``.
    """
    per_slot = dict(scopes or {})
    for slot, plan in list(step_plan.plans.items()):
        reads, writes = per_slot.get(slot, (None, None))
        step_plan.plans[slot] = synced(plan, residency, reads, writes)
    return step_plan


def live_sub_steps(fields: Any, pml: Any, sources: Any) -> Optional[Tuple[str, ...]]:
    """Which sub-steps and seam passes actually run for this configuration.

    ``None`` when the question cannot be answered, and there are now TWO ways it
    cannot be. The first is declared: ``sources`` was not passed. ``Fields`` does
    not hold the source list — the driver does — so this cannot be looked up, and
    inferring "no source seam" from not knowing is the over-covering refusal the
    residency clause exists to prevent. The caller passes ``driver._sources``
    (possibly ``()``).

    The second is UNREADABLE STATE, and it was a live defect until tranche 2. Every
    read below is on a duck-typed engine object, and ``plan_step``'s whole contract
    is that it NEVER RAISES — a raising predicate is a refusal, a raising builder
    is a refusal, and the composition returns the array path rather than exploding
    into a caller that would otherwise have stepped correctly. This function sat
    OUTSIDE the arm loop's try/except and read four attributes bare, so a
    ``fields.polarizations`` that raised on access escaped ``plan_step`` entirely.
    Caught by the composition suite's raising-proxy parametrisation, on
    ``polarizations``; the other three reads are hardened for the same reason
    rather than because each was separately observed.

    AN UNREADABLE LIST IS NOT AN EMPTY ONE. Each read that fails returns ``None``
    — "the live set could not be determined" — which makes the residency verdict
    REFUSE, which puts the composition on the array path. Defaulting to "no
    polarizations" would instead licence holding a mirror across an ``update_P``
    that does run.

    Everything else IS derivable and is derived: the wall passes from the grid's
    own metallic declaration through :func:`zero_metal_axes` (the function
    ``_zero_metal`` agrees with by construction), ``update_P`` from the registered
    polarizations, and THE FOLD'S OWN TWO SEAM PASSES from the grid's mirror
    planes.

    THE FOLD'S PASSES ARE NOT SOURCE-DRIVEN AND DERIVING THEM FROM THE SOURCE LIST
    WAS WRONG. ``fill_symmetry_bc_B``/``_D`` run whenever ``grid.has_symmetry()``
    (stepping.py:1482-1483 returns immediately otherwise) — with no source at all,
    on every folded run — and ``fill_folded_far_ghosts_B``/``_D`` run whenever some
    axis is folded PERIODIC (``_stored_past_owned``, stepping.py:1503-1518). Before
    this, a folded run with no magnetic source reported ``fill_B`` NOT LIVE while
    the driver wrote B there twice, and a mirror held across the step was stale in
    exactly the plane the fold owns. ``fill_B``/``fill_D`` are therefore the UNION
    of the source seam and the near symmetry fill; the far ghost pass is its own
    slot because ``zero_metal_*`` sits between them (driver.py:3286 / :3301) and a
    plan may not fuse across that.
    """
    if sources is None:
        return None
    try:
        grid = getattr(fields, "grid", None)
        if grid is None:
            return None
        live: List[str] = ["step_B", "update_H", "step_D", "update_E"]
        declared = tuple(sources)
        magnetic = any(str(getattr(s, "field_type", "")) == "B" for s in declared)
        electric = any(str(getattr(s, "field_type", "")) != "B" for s in declared)
        folded = bool(_call(grid, "has_symmetry", default=False)) and any(
            bool(_call(grid, "is_mirrored", axis, default=False))
            for axis in range(3))
        if magnetic or folded:
            live.append("fill_B")
        if electric or folded:
            live.append("fill_D")
        if any(zero_metal_axes(grid)):
            live.extend(("zero_metal_B", "zero_metal_D"))
        if folded and any(folded_periodic_axes(grid)):
            live.extend(("fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D"))
        if tuple(getattr(fields, "polarizations", ()) or ()):
            live.append("update_P")
    except Exception:  # noqa: BLE001 - unreadable state is not empty state
        return None
    return tuple(name for name in RESIDENCY_ORDER if name in set(live))


#: The slots NO registered family carries, each with the reason it is empty.
#: A slot recorded with a reason reads differently from one recorded silently: the
#: first says "no product carries this", the second would say "nobody asked", and
#: a composer a reader cannot tell those apart in is not auditable. Every slot in
#: ``STEP_ORDER`` must be either armed or listed here, and the byte-neutrality
#: gate's registry leg asserts exactly that partition rather than trusting it.
#:
#: ``fill_B``/``fill_D`` LEFT THIS TABLE when the folded family landed: they now
#: carry the mirror-fill arm, so an empty-table reason for them would be dead code
#: that a reader could mistake for a live claim. The arm is deliberately UNGATED so
#: an UNFOLDED run still gets a named refusal on those slots — this family's own
#: "no mirror plane is active" — rather than the composer's "nobody asked".
NO_ARM_REASONS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Cross-sub-step pairs: the two slots one launch owns, and the deposit repair
# ---------------------------------------------------------------------------
#
# THE SECOND SLOT WAS NEVER FILLED ON THIS TRACK. Every Metal fused product
# registers ``wired=False`` on the FIRST slot of the seam it spans and declares the
# rest through ``replaces_sub_steps``, so nothing here has ever written a
# ``STEP_ORDER`` slot on a pair's behalf — the composer would have left the
# constitutive slot to the SEPARATE ordinary arm, which then runs a second time
# after the fused launch already performed it. That is the composition rule the
# unwired flag has been standing in for, and it is Triton's
# ``_install_fused_pair`` (``triton_kernels/launch.py``), mirrored rather than
# reinvented.

#: Per curl slot: the constitutive slot a pair on it also owns, and the seam name
#: :mod:`..deposit_repair` knows it by. Mirrors ``triton_kernels.coverage.FUSED_PAIRS``
#: from the other side — that table is keyed by seam, this one by the slot an arm
#: is registered on, because that is what this composer iterates.
#:
#: THE ADE CHAIN IS NOT IN THIS TABLE AND MUST NOT BE. ``fused_ade_chain`` and its
#: complex sibling weld ``update_E`` to ``update_P``, which is a seam with no source
#: injection in it — the driver advances the polarizations after ``update_E``
#: (driver.py:3313) with nothing between. There is no deposit to carry there, so the
#: two-slot protocol below has nothing to say about it and it is left alone rather
#: than given a refusal that would read as a claim about a pair.
#: THE H->D ROW LANDED 2026-09-04, AHEAD OF ITS PRODUCT, and both halves of that
#: sentence are deliberate. It lands ahead because what it buys before any kernel
#: exists is the ARBITRATION (:func:`_neighbouring_seam_claimant`) and the KEY ORDER,
#: and an empty row costs nothing: :func:`_install_fused_pairs` opens each seam with
#: ``if not candidates: continue`` and records no reason, so a seam nothing bids on
#: is silent rather than reporting a claim about the run.
#:
#: THE ORDER IS LOAD-BEARING AND THE ROW IS APPENDED LAST. The loop walks
#: ``.items()`` in dict order against the LIVE ``selected``. Placed first, this seam
#: would be offered ``update_H`` before ``fused_magnetic_pair`` was, would find an ARM
#: label rather than a fused one there, and :func:`._pair_may_absorb` would let it
#: take both slots — displacing a released, gate-passing product on every row it
#: reached. ``test_the_h_to_d_seam_row_is_last`` pins the order with that reason.
#:
#: ITS SEAM NAME IS ``withdraw_hoist.SEAM``, NOT A FIELD LETTER. Nothing is injected
#: between ``update_H`` and ``step_D``, so there is no deposit list to select and
#: ``deposit_repair._in_seam_indexed`` would read any string other than ``'B'`` as
#: the ELECTRIC list — bracketing this seam with a repair for an injection that
#: happens in the NEXT one. What the driver does put between the two consults is the
#: electric ``withdraw`` loop, and :func:`_install_fused_pair` routes this name to
#: :mod:`..withdraw_hoist` instead.
FUSED_PAIR_SEAMS: Dict[str, Tuple[str, str]] = {
    "step_B": ("update_H", "B"),
    "step_D": ("update_E", "D"),
    "update_H": ("step_D", _withdraw_hoist.SEAM),
}

#: Which ARM each fused product's own kernel implements, per slot it absorbs —
#: Triton's ``FUSED_PAIR_ARMS`` clause, keyed by this track's family names.
#:
#: THE TABLE IS SHORT BECAUSE IT IS READ, NOT GUESSED. ``fused_magnetic_pair``'s
#: predicate is literally ``pml_curl_coverage(fields, pml, "step_B", ...)`` AND
#: ``constitutive_coverage(fields, pml, "H", ...)`` (fused_magnetic_pair.py:485-490),
#: which are the ``PML`` and ``ordinary`` arms' own predicates, and the
#: ``cart_pml_real`` row of ``parity/meep_gpu/metal_composition_matrix.py`` records
#: those two labels winning those two slots. Every other Metal product that welds a
#: curl slot to its constitutive slot and is NOT listed here is REFUSED BY NAME
#: below rather than absorbing a slot on a mapping nobody measured. Establishing
#: those rows is the edit owed before any of them may fuse — and it is a separate
#: measurement from the deposit repair, which is what this table is in front of.
#:
#: THE TWO FOLDED ROWS ARE READ THE SAME WAY, 2026-08-27.
#: ``metal_folded_fused_magnetic_pair_coverage`` is literally
#: ``folded_composition_curl_coverage(..., "step_B")`` AND
#: ``folded_constitutive_coverage(..., "H")`` (folded_fused_magnetic_pair.py:1234-1238),
#: and ``metal_folded_fused_pair_coverage`` is the same conjunction on ``step_D``/``E``
#: (folded_fused_pair.py:1136-1139). Those two functions ARE the ``folded`` curl arm's
#: and the ``folded`` constitutive arm's predicates — ``symmetry._curl_arm_coverage``
#: (symmetry.py:1740-1742) and ``symmetry._constitutive_arm_coverage`` (:1757-1760)
#: call nothing else, and both arms register under the label ``"folded"``
#: (symmetry.py:1807, :1813). Measured on ``metal_composition_matrix.folded()``:
#: ``step_B``/``update_H``/``step_D``/``update_E`` all come back ``'folded'``.
#:
#: NEITHER FOLDED ROW LICENSES THE DEPOSIT REPAIR, and the two questions stay
#: separate. A row here says which arm the kernel implements; the repair is declared
#: by the family's own ``CARRIES_DEPOSIT_REPAIR``. Both folded families held that at
#: ``False`` from their routing on 2026-08-27 until 2026-08-28, because a folded seam
#: runs ``fill_symmetry_bc_*``/``fill_folded_far_ghosts_*`` AFTER the injection and a
#: POINT repair never visits the MIRROR IMAGE of a deposit index — measured on a
#: corpus row, where the image came back byte-equal to a run with no source at all.
#: What retired that boundary is not a row here but ``deposit_repair.repair_cells``,
#: which extends the saved and restored set to the closure of the cells those fills
#: image each deposit point into; the two flags flipped in the same edit. See either
#: family's flag comment for what is still refused by name.
#:
#: THE TWO UNFOLDED ROWS ARE READ THE SAME WAY, 2026-08-28, and unlike the folded
#: pair they DO license the repair (see each family's flag comment).
#: ``metal_complex_fused_magnetic_pair_coverage`` is literally
#: ``complex_fields.complex_pml_curl_coverage(..., "step_B")`` AND
#: ``complex_fields.complex_constitutive_coverage(..., "H")``
#: (complex_fused_magnetic_pair.py:697-704), and those two functions ARE the
#: ``complex/Bloch`` arm's own predicates — ``complex_fields._curl_arm_coverage``
#: (complex_fields.py:1330-1333) and ``._constitutive_arm_coverage`` (:1342-1345)
#: call nothing else, and both register under the label ``"complex/Bloch"``
#: (complex_fields.py:1374, :1380). ``metal_fused_dispersive_pair_coverage`` is
#: ``coverage.pml_curl_coverage(..., "step_D")`` AND
#: ``metal_dispersive_e_coverage(...)`` (fused_dispersive_pair.py:493-499), which
#: are the ``PML`` curl arm's body (launch.py:422-427) and the
#: ``dispersive PML E`` arm's body (dispersive_update_e.py:292-294, registered at
#: :305-307). Measured through ``plan_step`` with ``fuse=False``: on
#: ``metal_composition_matrix.cart(complex_storage=True)`` both ``step_B`` and
#: ``update_H`` come back ``'complex/Bloch'``, and on
#: ``matrix.dispersive(matrix.cart())`` ``step_D`` comes back ``'PML'`` and
#: ``update_E`` ``'dispersive PML E'`` — the same two labels the
#: ``cart_pml_complex_forced`` and ``cart_pml_dispersive`` rows of
#: ``parity/meep_gpu/metal_composition_matrix.py`` record.
#: THE PLAIN D/E ROW IS READ THE SAME WAY, 2026-08-28, and it is the D-side mirror
#: of the ``fused_magnetic_pair`` row above rather than a new kind of claim.
#: ``metal_fused_electric_pair_coverage`` is literally
#: ``coverage.pml_curl_coverage(fields, pml, "step_D", ...)`` AND
#: ``coverage.constitutive_coverage(fields, pml, "E", ...)``
#: (fused_electric_pair.py, the two lines opening its predicate), which are the
#: ``PML`` curl arm's body (launch.py:422-427) and the ``ordinary`` constitutive
#: arm's -- the same two functions the magnetic row cites on the other seam. The
#: ``cart_pml_real`` row of ``parity/meep_gpu/metal_composition_matrix.py`` records
#: ``step_D="PML"`` and ``update_E="ordinary"`` winning those two slots, in the same
#: dict that records ``step_B``/``update_H`` for the magnetic pair.
#:
#: IT LICENSES THE DEPOSIT REPAIR, and on this family that is the point rather than
#: a side effect: every corpus row the pair reaches carries an electric source inside
#: the seam, so without the bracket the product would serve nothing. The declaration
#: still lives in the family's own ``CARRIES_DEPOSIT_REPAIR``; this row only says
#: which arm each slot implements.
#:
#: THE COMPLEX D/E ROW IS READ THE SAME WAY, 2026-08-30. It is the D-side mirror of
#: the ``complex_fused_magnetic_pair`` row rather than a new kind of claim:
#: ``metal_complex_fused_electric_pair_coverage`` is literally
#: ``complex_fields.complex_pml_curl_coverage(fields, pml, "step_D", ...)`` AND
#: ``complex_fields.complex_constitutive_coverage(fields, pml, "E", ...)``, which are
#: the ``complex/Bloch`` curl arm's own body (``complex_fields._curl_arm_coverage``)
#: and its constitutive arm's (``._constitutive_arm_coverage``) -- the same two
#: functions the magnetic row cites on the other seam, asked on the other two slots.
#: MEASURED through ``plan_step`` with ``fuse=False`` on
#: ``metal_composition_matrix``'s ``cart_pml_complex_forced`` row: ``step_D`` comes
#: back ``'complex/Bloch'`` and ``update_E`` comes back ``'complex/Bloch'``, which is
#: also what that row's own expectation dict records.
#:
#: IT LICENSES THE DEPOSIT REPAIR, and on this family that is the point rather than a
#: side effect, exactly as on the plain D/E row: every one of the sixteen census rows
#: admitting both halves carries an electric source inside the seam.
#:
#: THE TWO FOLDED COMPLEX MAGNETIC ROWS, 2026-08-30, are the last two families whose
#: predicates were already conjunctions of a single arm's two halves but which had no
#: row here at all -- so the seam loop refused them by name on every configuration and
#: their in-seam magnetic deposits stayed uncarryable.
#: ``metal_folded_complex_fused_magnetic_pair_coverage`` is
#: ``folded_complex_composition_curl_coverage(..., "step_B")`` AND
#: ``folded_complex_constitutive_coverage(..., "H")``, which ARE the ``folded
#: complex`` curl and constitutive arms' bodies; the folded-beta-complex family is
#: the same construction over ``folded_beta``'s two ``folded beta complex`` arms.
#: MEASURED through ``plan_step`` with ``fuse=False`` on
#: ``parity/meep_gpu/metal_composition_matrix.py``'s own rows: ``fold_complex_2d``
#: returns ``'folded complex'`` on all four arithmetic slots, and
#: ``fold_complex_2d_beta`` returns ``'folded beta complex'`` on all four -- which is
#: also what each row's expectation dict records.
#:
#: THE FOLDED DISPERSIVE D/E ROW, 2026-08-30, is the one row here whose two entries
#: name DIFFERENT arms, and that is the cell it exists for: the intersection
#: ``folded_fused_pair`` (folded curl + folded ORDINARY E) and ``fused_dispersive_pair``
#: (unfolded curl + pole-aware E) both leave uncovered.
#: ``metal_folded_fused_dispersive_pair_coverage`` is literally
#: ``symmetry.folded_composition_curl_coverage(..., "step_D")`` AND
#: ``folded_dispersive_update_e.folded_dispersive_e_coverage(...)``, which are the
#: ``folded`` curl arm's body (``symmetry._curl_arm_coverage``) and the ``folded
#: dispersive PML E`` arm's own predicate (registered under that label at
#: ``folded_dispersive_update_e.py:LABEL``). MEASURED through ``plan_step`` with
#: ``fuse=False`` on ``metal_composition_matrix``'s ``fold_real_2d_dispersive`` row:
#: ``step_D`` comes back ``'folded'`` and ``update_E`` comes back
#: ``'folded dispersive PML E'``.
#:
#: THE REAL-BETA D/E ROW, 2026-08-30. Its magnetic twin has no row here and stays
#: without one -- that cell is already served and nothing has measured its absorb
#: pair; this row is not a claim about it.
#: ``metal_beta_fused_electric_pair_coverage`` is literally
#: ``special_kz.beta_pml_curl_coverage(..., "step_D")`` AND
#: ``special_kz.beta_run_constitutive_coverage(..., "E")``, which are the
#: ``special_kz real beta`` arm's own two predicates. MEASURED through ``plan_step``
#: with ``fuse=False`` on ``metal_composition_matrix``'s ``beta_real_2d`` row: all
#: four arithmetic slots come back ``'special_kz real beta'``.
#:
#: THE Dcyl m = 0 D/E ROW, 2026-08-31, and it is the D-side mirror of a cell that had
#: no row on EITHER seam until now. ``metal_cylindrical_real_fused_electric_pair_
#: coverage`` is literally ``cylindrical_real.cylindrical_real_curl_coverage(fields,
#: pml, "step_D", ...)`` AND ``cylindrical_real.cylindrical_real_constitutive_
#: coverage(fields, pml, "E", ...)``, which ARE the ``cylindrical m=0`` curl arm's and
#: constitutive arm's own bodies — ``cylindrical_real._curl_arm_coverage`` and
#: ``._constitutive_arm_coverage`` call nothing else, and both register under that one
#: label. MEASURED through ``plan_step`` with ``fuse=False`` on
#: ``metal_composition_matrix.cylindrical(m=0, complex_storage=False,
#: z_kind='metallic')`` under ``MEEP_GPU_SUBNORMAL_POLICY=flush``: ``step_B``,
#: ``update_H``, ``step_D`` and ``update_E`` all four come back ``'cylindrical m=0'``.
#:
#: IT LICENSES THE DEPOSIT REPAIR, and on this family the flag IS the product rather
#: than one clause of it: all three corpus rows the cell reaches declare an ELECTRIC
#: source inside the seam, so the same module with ``CARRIES_DEPOSIT_REPAIR`` at False
#: would compile, gate green on every other leg and serve nothing at all.
#:
#: ITS MAGNETIC TWIN STILL HAS NO ROW HERE AND KEEPS ITS FLAG AT FALSE. The two are
#: not a pair of edits: on the B/H seam those same three rows are electric-only, so
#: the twin's source clause costs it nothing and it needs neither the row nor the
#: flag. A row here is not free — it lets the seam loop install the pair — so it is
#: added where it is read and nowhere else.
#:
#: THE Dcyl |m| >= 1 COMPLEX D/E ROW, 2026-08-31, is the same reading one |m| arm
#: over, and it is the LARGEST cell on this board — 16 seam-instances, more than any
#: other gap. ``cylindrical_fused_electric_pair_coverage`` is literally
#: ``cylindrical_complex.cylindrical_complex_pml_curl_coverage(..., "step_D")`` AND
#: ``cylindrical_complex.cylindrical_complex_constitutive_coverage(..., "E")``, which
#: ARE the ``cylindrical complex`` curl arm's and constitutive arm's own bodies.
#: MEASURED through ``plan_step`` with ``fuse=False`` on
#: ``metal_composition_matrix.cylindrical(m=1, complex_storage=True)`` and again at
#: ``m=2``, under ``MEEP_GPU_SUBNORMAL_POLICY=flush`` with the cylindrical-complex
#: expansion probe bound: all four arithmetic slots come back
#: ``'cylindrical complex'`` on both — and, since the family's ``M_ZERO`` arm of
#: 2026-09-04, at ``m=0`` with complex64 storage as well (the same row serves that
#: arm; ``gate_metal_cylindrical_complex.py`` leg ``split`` reads the selection).
#:
#: IT LICENSES THE DEPOSIT REPAIR and on this family the flag IS the product: all
#: SIXTEEN corpus rows declare an electric source inside the seam. Its magnetic twin
#: has no row here and keeps its flag at False, for the same reason the m = 0 pair
#: above gives — on the B/H seam those same sixteen rows are empty.
#:
#: BOTH Dcyl D/E ROWS EXIST BECAUSE OF ONE TECHNIQUE. Neither product's signature fits
#: the platform's 31-binding ceiling as every other pair here is built: the m = 0 one
#: is over by one pointer and the complex one by three. ``coefficient_pack`` puts the
#: curl half's six read-only PML coefficient vectors in ONE buffer for both, which is
#: what let either row be written at all.
#: THE TWO NO-ABSORBER STORED-E D/E ROWS, 2026-08-31, and the FIRST rows whose
#: bracket installs :data:`..deposit_repair.PLAIN_PATH` rather than the split-field
#: repair — both halves refuse an active absorber, so the recurrence their seam
#: inverts is ``update_E``'s plain overwrite. Each family's predicate is literally
#: its curl arm's certified body AND ``no_pml_stored_e.metal_stored_e_coverage``.
#: MEASURED through ``plan_step`` with ``fuse=False`` under
#: ``MEEP_GPU_SUBNORMAL_POLICY=flush``: on ``metal_composition_matrix.
#: real_stored_e_no_pml(with_polarization=True)`` ``step_D`` comes back
#: ``'no-PML curl'`` and ``update_E`` ``'no-PML stored E'``; wrapped in
#: ``conductive(...)`` the same fixture returns ``'conductive no-PML curl'`` and
#: ``'no-PML stored E'`` — the same labels the ``nopml_real_stored_e`` and census
#: rows record.
#:
#: BOTH LICENSE THE DEPOSIT REPAIR AND ON BOTH THE FLAG IS THE PRODUCT: every row
#: of either cell declares an ELECTRIC source inside the seam. On the CONDUCTIVE
#: family a second, measured clause refused its two corpus rows by name for as long
#: as the driver rescaled the WHOLE target volume (the -0.0 canonicalisation); the
#: driver now replays the rescale sparsely at the deposit cells the sources publish
#: (``_inject_electric_through_conductivity``), and the clause refuses only a
#: scaled source that publishes no deposit table — see the family module's lifted
#: refusal.
#:
#: THE FOUR ROWS OF THE 2026-09-01 RESIDUE ROUND ARE READ THE SAME WAY AS EVERY
#: ROW ABOVE — each family's predicate is literally the conjunction of the two
#: named arms' own bodies:
#:
#: * ``bfast_fused_electric_pair`` — ``bfast_curl.bfast_pml_curl_coverage(...,
#:   "step_D")`` AND ``bfast_curl.bfast_run_constitutive_coverage(..., "E")``,
#:   the ``BFAST`` curl and constitutive arms' own predicates; the board's
#:   ``(BFAST, BFAST)`` D->E cell names exactly those labels. Its cell was scored
#:   UNFUSABLE ON METAL at 33 pointers; the curl-side coefficient pack rebinds it
#:   at 28 (the third cell the pack has flipped).
#: * ``conductive_fused_electric_pair`` — ``conductive_pml
#:   .metal_conductive_pml_curl_coverage(..., "step_D")`` AND
#:   ``coverage.constitutive_coverage(..., "E")``, the ``conductive PML curl``
#:   and ``ordinary`` arms' own predicates (the ``cart_pml_conductive_electric``
#:   row of ``parity/meep_gpu/metal_composition_matrix.py`` records those two
#:   labels winning those two slots). Scored UNFUSABLE at 39 pointers; the
#:   both-halves twelve-vector pack rebinds it at 28.
#: * ``beta_complex_fused_electric_pair`` / ``beta_complex_fused_magnetic_pair``
#:   — ``special_kz.beta_bloch_pml_curl_coverage`` AND
#:   ``special_kz.beta_run_complex_constitutive_coverage`` on each seam's two
#:   slots: the ``special_kz complex beta`` arms' own predicates. The
#:   ``beta_complex_2d`` row of the composition matrix records that label winning
#:   all four slots, and ``results/metal_coverage_special_kz_reclose_2026-08-19``
#:   records the same on the corpus row itself.
#:
#: THREE OF THE FOUR LICENSE THE DEPOSIT REPAIR (electric seams with a declared
#: electric source on every reachable row); the magnetic beta-complex pair keeps
#: its flag False on an electric-only cell, exactly as the BFAST magnetic pair
#: does.
#:
#: THE FOUR STENCIL WELDS ARE READ THE SAME WAY, 2026-09-01, and NONE of them
#: licenses the deposit repair. Each family's predicate is literally the two arms'
#: own predicates conjoined:
#: ``offdiag_fused_electric_pair`` is ``coverage.pml_curl_coverage(..., "step_D")``
#: AND ``offdiag_update_e.offdiag_constitutive_coverage(...)``, which are the
#: ``PML`` curl arm's body (launch.py:422-427) and the ``offdiag`` constitutive
#: arm's (offdiag_update_e.py:1108-1111);
#: ``folded_offdiag_fused_electric_pair`` is
#: ``symmetry.folded_composition_curl_coverage(..., "step_D")`` AND
#: ``folded_offdiag_update_e.folded_offdiag_constitutive_coverage(...)`` — the
#: ``folded`` curl arm's and the ``folded offdiag`` constitutive arm's;
#: ``complex_no_pml_offdiag_fused_electric_pair`` is
#: ``complex_no_pml_curl.metal_complex_no_pml_curl_coverage(..., "step_D")`` AND
#: ``complex_no_pml_offdiag_update_e.metal_complex_no_pml_offdiag_coverage(...)``;
#: ``folded_complex_offdiag_fused_electric_pair`` is
#: ``folded_complex.folded_complex_composition_curl_coverage(..., "step_D")`` AND
#: ``complex_folded_offdiag_update_e.metal_complex_folded_offdiag_coverage(...)``.
#: THE REPAIR FLAG IS FORCED FALSE ON ALL FOUR, not chosen: an off-diagonal
#: chi1inv constitutive is one of the two shapes ``deposit_repair.repairable``
#: refuses BY NAME, because a point repair recomputes E at a deposit cell from
#: that cell's own displacement and this constitutive reads its neighbours'. So
#: each family's predicate refuses an in-seam electric source outright, and the
#: board's demand column for those four cells (8 of 16, 9 of 19, 1 of 2, 1 of 3)
#: is exactly that refusal counted.
FUSED_PAIR_ARMS: Dict[str, Tuple[str, str]] = {
    "fused_magnetic_pair": ("PML", "ordinary"),
    "no_pml_fused_electric_pair": ("no-PML curl", "no-PML stored E"),
    "no_pml_conductive_fused_electric_pair":
        ("conductive no-PML curl", "no-PML stored E"),
    "fused_electric_pair": ("PML", "ordinary"),
    "folded_fused_magnetic_pair": ("folded", "folded"),
    "folded_fused_pair": ("folded", "folded"),
    "complex_fused_magnetic_pair": ("complex/Bloch", "complex/Bloch"),
    "complex_fused_electric_pair": ("complex/Bloch", "complex/Bloch"),
    "fused_dispersive_pair": ("PML", "dispersive PML E"),
    "folded_fused_dispersive_pair": ("folded", "folded dispersive PML E"),
    "beta_fused_electric_pair": ("special_kz real beta", "special_kz real beta"),
    "folded_complex_fused_magnetic_pair": ("folded complex", "folded complex"),
    "folded_complex_fused_pair": ("folded complex", "folded complex"),
    "folded_beta_complex_fused_magnetic_pair": ("folded beta complex",
                                                "folded beta complex"),
    "folded_beta_complex_fused_pair": ("folded beta complex", "folded beta complex"),
    "folded_beta_real_fused_magnetic_pair": ("folded beta real",
                                             "folded beta real"),
    "folded_beta_real_fused_pair": ("folded beta real", "folded beta real"),
    "cylindrical_real_fused_electric_pair": ("cylindrical m=0", "cylindrical m=0"),
    "cylindrical_complex_fused_electric_pair": ("cylindrical complex",
                                                "cylindrical complex"),
    "bfast_fused_electric_pair": ("BFAST", "BFAST"),
    "conductive_fused_electric_pair": ("conductive PML curl", "ordinary"),
    # THE TWO H->D ROWS THAT CLOSE THE SEAM'S LAST SMALL CELLS, 2026-09-07, read off
    # each product's predicate exactly as every row above is, and written in the
    # seam's own slot order -- which on H->D is CONSTITUTIVE THEN CURL.
    #
    # `bfast_fused_hd_pair`'s predicate opens with
    # `bfast_curl.bfast_run_constitutive_coverage(fields, pml, "H", ...)` and then
    # `bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_D", ...)`, and the
    # `bfast_real_pml` row of `parity/meep_gpu/metal_composition_matrix.py` records
    # the label `BFAST` winning both slots.
    #
    # `conductive_fused_hd_pair`'s opens with
    # `coverage.constitutive_coverage(fields, pml, "H", ...)` -- the `ordinary` arm's
    # own body -- and then
    # `conductive_pml.metal_conductive_pml_curl_coverage(fields, pml, "step_D", ...)`.
    # THE TWO HALVES COME FROM DIFFERENT FAMILIES and that is the composer's own
    # answer rather than an inference: the `cart_pml_conductive_electric` row of the
    # composition matrix pins `update_H = ordinary` beside
    # `step_D = conductive PML curl`, because an electric conductivity changes the
    # CURL recurrence only (stepping._apply_curl reads condfac_for at S:479) and the
    # curl predicate is sub-step aware about which curl. The gate re-derives it from
    # `plan_step` on its own fixture (leg `arm_pairing`) rather than trusting either.
    #
    # NEITHER ROW INSTALLS ANYTHING. Both products declare `INSTALLABLE = False` on a
    # measured arbitration, and each gate's `arbitration` leg drives the composer with
    # this row applied IN-PROCESS and requires the released neighbours to keep their
    # slots anyway: on the conductive cell both neighbours install (a LOSS), and on
    # the BFAST cell only `bfast_fused_electric_pair` does, because
    # `bfast_fused_magnetic_pair` carries no row here at all (a TIE, which the
    # released precedent resolves for the incumbent).
    "bfast_fused_hd_pair": ("BFAST", "BFAST"),
    "conductive_fused_hd_pair": ("ordinary", "conductive PML curl"),
    "beta_complex_fused_electric_pair": ("special_kz complex beta",
                                         "special_kz complex beta"),
    "beta_complex_fused_magnetic_pair": ("special_kz complex beta",
                                         "special_kz complex beta"),
    "offdiag_fused_electric_pair": ("PML", "offdiag"),
    "folded_offdiag_fused_electric_pair": ("folded", "folded offdiag"),
    "complex_no_pml_offdiag_fused_electric_pair":
        ("complex no-PML curl", "complex no-PML off-diagonal"),
    "folded_complex_offdiag_fused_electric_pair":
        ("folded complex", "complex folded off-diagonal PML E"),
    # THE H->D ROW, 2026-09-05, READ OFF THE PREDICATE LIKE EVERY ROW ABOVE and
    # written in the seam's own slot order, which on this seam is CONSTITUTIVE
    # THEN CURL. ``metal_fused_hd_pair_coverage`` opens with
    # ``constitutive_coverage(fields, pml, "H", residency)`` -- the ``ordinary``
    # arm's own predicate on ``update_H`` -- and then
    # ``pml_curl_coverage(fields, pml, "step_D", residency)`` -- the ``PML`` arm's
    # own predicate on ``step_D`` (fused_hd_pair.py, the two calls at the top of the
    # predicate). The row is what lets the seam loop ASK the product at all; whether
    # it INSTALLS is decided by the product's ``INSTALLABLE`` declaration and by the
    # composition below, and today the answer is no on every corpus row: driven over
    # the standing census with this row present, ``fused_hd_pair`` installs on 0 of
    # 194 rows and the incumbents are unchanged on every row (80 -> 80 installed
    # seam-instances, ``step_B`` 49 / ``step_D`` 31 / ``update_H`` 0).
    "fused_hd_pair": ("ordinary", "PML"),
    # THE Dcyl m = 0 H->D ROW, 2026-09-06, READ OFF THE PREDICATE LIKE EVERY ROW
    # ABOVE, in the seam's own slot order (constitutive then curl).
    # ``metal_cylindrical_real_fused_hd_pair_coverage`` opens with
    # ``cylindrical_real_constitutive_coverage(fields, pml, "H", ...)`` -- the
    # ``cylindrical m=0`` constitutive arm's own predicate on ``update_H`` -- and then
    # ``cylindrical_real_curl_coverage(fields, pml, "step_D", ...)`` -- the
    # ``cylindrical m=0`` curl arm's on ``step_D``. MEASURED through ``plan_step`` with
    # ``fuse=False`` on the gate's Dcyl m = 0 fixture: all four arithmetic slots come
    # back ``'cylindrical m=0'``. The row lets the seam loop ASK the product; whether
    # it INSTALLS is decided by the product's ``INSTALLABLE`` declaration and by the
    # composition, and on this cell the answer is no: the released
    # ``cylindrical_real_fused_electric_pair`` installs first and holds ``step_D``
    # (a TIE at 5 launches either way, resolved for the incumbent -- gate
    # ``metal_cylindrical_real_fused_hd_pair_2026-09-06_cyl``, leg ``arbitration``).
    "cylindrical_real_fused_hd_pair": ("cylindrical m=0", "cylindrical m=0"),
    # THE FOLDED H->D ROW, 2026-09-07, READ OFF THE PREDICATE LIKE EVERY ROW ABOVE
    # and written in the seam's own slot order (constitutive then curl).
    # ``metal_folded_fused_hd_pair_coverage`` opens with
    # ``symmetry.folded_constitutive_coverage(fields, pml, "H", residency)`` -- the
    # ``folded`` constitutive arm's own predicate on ``update_H``
    # (``symmetry._constitutive_arm_coverage`` calls nothing else) -- and then
    # ``symmetry.folded_composition_curl_coverage(fields, pml, "step_D", residency)``
    # -- the ``folded`` curl arm's own predicate on ``step_D``
    # (``symmetry._curl_arm_coverage``). Both arms register under the label
    # ``"folded"``, which is what the two folded rows above are read from.
    # MEASURED through the composer on the folded gate's nine fixtures, leg
    # ``arbitration``: with this row present the product installs on 0 of 9 and the
    # released ``folded_fused_magnetic_pair`` still holds ``step_B``/``update_H`` on
    # every one. Over the corpus cell the same reading holds by the standing board's
    # own join: ``B_to_H`` is served on 78 of the cell's 78 rows, so this product
    # ties on 11 and loses on 67 and gains on none.
    "folded_fused_hd_pair": ("folded", "folded"),
    # THE COMPLEX Dcyl H->D ROW, 2026-09-06, read off the predicate like every
    # row above and written in the seam's own slot order (constitutive then
    # curl): `metal_cylindrical_complex_fused_hd_pair_coverage` is
    # `cylindrical_complex.cylindrical_complex_constitutive_coverage(..., "H")`
    # AND `cylindrical_complex.cylindrical_complex_pml_curl_coverage(...,
    # "step_D")` -- the `cylindrical complex` arm's own two bodies on those two
    # slots. The row lets the seam loop ASK the product; the product declares
    # INSTALLABLE = False (its module's INSTALLABLE_REASON: a launch-count tie
    # with the released D->E pair that the count does not price against the
    # host round trip this product removes), so the loop refuses it on the flag
    # on every row until a decision is taken on that arbitration.
    "cylindrical_complex_fused_hd_pair": ("cylindrical complex",
                                          "cylindrical complex"),
    # THE CARTESIAN COMPLEX/BLOCH H->D ROW, 2026-09-07, read off the predicate
    # like every row above and written in the seam's own slot order
    # (constitutive then curl): `metal_complex_fused_hd_pair_coverage` is
    # `complex_fields.complex_constitutive_coverage(..., "H")` AND
    # `complex_fields.complex_pml_curl_coverage(..., "step_D")` -- the
    # `complex/Bloch` arm's own two bodies on those two slots. The row lets the
    # seam loop ASK the product; the product declares INSTALLABLE = False, and
    # its module's INSTALLABLE_REASON carries the measurement that decides it:
    # driven through THIS composer on all 17 corpus rows of the cell, BOTH
    # neighbouring released pairs install on 17 of 17, so installing this
    # product instead is a LOSS on 17 rows, a TIE on 0 and a GAIN on 0.
    "complex_fused_hd_pair": ("complex/Bloch", "complex/Bloch"),
    # THE THREE H->D TAIL ROWS, 2026-09-07, read off each predicate like every row
    # above and written in the seam's own slot order (constitutive then curl). Each
    # family's coverage asks its variant's CONSTITUTIVE predicate on `update_H`
    # first and that variant's CURL predicate on `step_D` second, and the labels
    # here are the ones those predicates' own arms register under. THE LABELS ARE
    # NOT RETYPED FROM A BOARD: each family states them in its own `VARIANT_CELLS`
    # map, which is the same table its gate's lift joins the corpus on, so the cell
    # a gate measures and the cell this table admits cannot drift apart.
    #
    # EACH ROW IS THE PRIMARY PAIR ONLY. The two BETA families are ONE product over
    # TWO cells -- one transform emitting two device strings, with the fold read off
    # the grid by the family's own `resolve_variant` -- so their SECOND cell is a
    # `FUSED_PAIR_EXTRA_ARMS` row below rather than a widened value here, which is
    # that table's whole reason for existing.
    #
    # ALL THREE PRODUCTS DECLARE `INSTALLABLE = False`, and each module's
    # INSTALLABLE_REASON carries the measurement: driven through THIS composer on
    # each product's own eight gate fixtures, BOTH neighbouring pairs install on
    # 8 of 8, so installing an H->D product instead is a LOSS on 8, a TIE on 0 and a
    # GAIN on 0 (the `arbitration` leg of each 2026-09-07 flush artifact).
    "folded_complex_fused_hd_pair": ("folded complex", "folded complex"),
    "beta_complex_fused_hd_pair": ("special_kz complex beta",
                                   "special_kz complex beta"),
    "beta_real_fused_hd_pair": ("special_kz real beta", "special_kz real beta"),
    # --- THE SIX B->H AND D->E ROWS THE COMPOSER WAS STILL MISSING, 2026-09-17,
    # each read off its own product's predicate exactly as every row above is, and
    # each written in its seam's slot order (curl then constitutive on B->H and
    # D->E, which is the order those seams run).
    #
    # WHY THEY WERE ABSENT AND WHAT THAT COST. Every one of these six products is
    # CERTIFIED and carries a released weld; what none of them had was a row here,
    # and `dispatch_reachability.metal_certified_but_not_installed` reported each of
    # them in the same words -- "the family registers a weld and has NO
    # launch.FUSED_PAIR_ARMS row, so _install_fused_pairs has no arm pair to absorb".
    # A product the absorb table cannot name is never ASKED, so its predicate, its
    # gate and its weld all stood while the composer stepped past it on every row.
    # The board measures the cost at 29 census instances across the two seams.
    #
    # The two Dcyl rows are the bulk of it (18 complex + 3 m=0) and each mirrors the
    # ELECTRIC twin already in this table: `cylindrical_fused_magnetic_pair`'s
    # predicate is `cylindrical_complex.cylindrical_complex_pml_curl_coverage(...,
    # "step_B")` AND `...cylindrical_complex_constitutive_coverage(..., "H")`, the
    # `cylindrical complex` arm's own two bodies, exactly as
    # `cylindrical_complex_fused_electric_pair` reads on `step_D`/`update_E`.
    "cylindrical_complex_fused_magnetic_pair": ("cylindrical complex",
                                                "cylindrical complex"),
    "cylindrical_real_fused_magnetic_pair": ("cylindrical m=0", "cylindrical m=0"),
    # `complex_conductive_fused_pair_coverage` opens with
    # `metal_complex_conductive_no_pml_curl_coverage(..., "step_D")` -- the `complex
    # conductive no-PML curl` arm's own body -- and then
    # `metal_complex_stored_e_coverage(...)`, which IS `complex no-PML stored E`'s
    # `_arm_coverage` (that module exports the same function the arm binds). The two
    # halves come from different families, as they do on `conductive_fused_hd_pair`
    # above, and for the same kind of reason: a complex conductivity changes the CURL
    # recurrence while the stored-E constitutive stays the complex no-PML one.
    "complex_conductive_fused_pair": ("complex conductive no-PML curl",
                                      "complex no-PML stored E"),
    # THE BELOW-THE-CUT THREE ARE DELIBERATELY NOT HERE, and the reason is a control
    # rather than a doubt about the rows. `nonlinear_fused_magnetic_pair`,
    # `beta_fused_magnetic_pair` and `bfast_fused_magnetic_pair` each mirror an
    # electric twin already in this table, and the dry run MEASURED all three winning
    # their slots once the rows were present: `bfast_1d` selected both BFAST pairs and
    # `special_kz_2d` selected the beta magnetic pair beside its electric twin. Four
    # census instances.
    #
    # WHAT THOSE FOUR WOULD HAVE COST. `UNDECLARED_WELDS` in
    # ``test_metal_fused_pair_deposit_wiring.py`` is the population of registered
    # Metal welds that hold no row here, and the sweep over it asserts that each is
    # refused BY NAME rather than falling through to a default. Wiring these three
    # empties that population: the sweep parametrises over nothing, its own
    # non-empty guard fails, and the property survives only on the synthetic
    # `h_to_d_stand_in`. A control that can only be exercised by a fixture the test
    # file builds for itself is weaker than one exercised by three real families, and
    # four instances do not buy that trade. They are available whenever a later round
    # wants them and can name a replacement population.
}

#: ADDITIONAL (first-slot arm, second-slot arm) pairs a product may absorb BESIDE
#: its primary row -- the ``triton_kernels.launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` /
#: ``cuda_kernels.fused_pairs.FUSED_PAIR_EXTRA_ARMS`` precedent, a separate table
#: rather than a widened value shape so every pin on a primary row stays a pin.
#:
#: ONE ROW, 2026-09-05, AND IT IS A MEASURED TEXT IDENTITY RATHER THAN A WIDENING OF
#: THE KERNEL. ``nonlinear_update_e``'s spine arms -- ``nonlinear PML magnetic`` on
#: ``update_H`` and ``nonlinear PML curl`` on ``step_D`` -- are the ordinary kernels
#: under a scope view: ``plan_nonlinear_run_constitutive`` calls
#: ``launch.plan_constitutive(_LinearScopeView(fields), pml, "H", ...)`` and
#: ``plan_nonlinear_run_pml_curl`` calls ``launch.plan_pml_curl(..., "step_D", ...)``,
#: and the source each hands to ``compile_source`` is byte-identical to the ordinary
#: arm's on the same grid (measured 2026-09-05 on ``metal_composition_matrix``'s
#: ``cart_pml_real`` against ``nonlinear_real``: sha256 ``d19e6dca...`` for
#: ``update_H`` and ``33aeb5e4...`` for ``step_D`` on both). The chi2/chi3 Pade
#: factor enters ``update_E`` only, one seam later, so the H->D weld's two lifted
#: tails are the same text on a nonlinear run as on a linear one. The predicate
#: therefore admits the nonlinear cell through those two arms' own predicates
#: (``fused_hd_pair.metal_fused_hd_pair_coverage``), and this row is what lets the
#: seam loop absorb the two slots the census gives to them -- the 2 corpus rows
#: (``examples:3rd-harm-1d.py``, ``tests:Test3rdHarm1d.test_3rd_harm_1d``) on which
#: NEITHER neighbour installs (``nonlinear_fused_magnetic_pair`` has no row here and
#: there is no nonlinear D->E product), so an H->D product there is the one measured
#: GAIN: 4 launches / 0 seams becomes 3 / 1. What it does NOT do is install: the
#: product still declares ``INSTALLABLE = False``, and the device gate has not driven
#: a chi2/chi3 configuration through this weld yet.
#: TWO MORE ROWS, 2026-09-07, AND THEY ARE A DIFFERENT KIND OF EXTRA FROM THE ONE
#: ABOVE. The nonlinear row is a measured TEXT IDENTITY -- the same kernel reached
#: through a scope view. These two are a product that genuinely emits TWO DEVICE
#: STRINGS: `beta_complex_fused_hd_pair` and `beta_real_fused_hd_pair` each select
#: their curl emitter from `special_kz` or from `folded_beta` by reading the fold
#: off the grid (`resolve_variant`), and the two emissions differ. What makes ONE
#: row in the arm table correct anyway is that the predicate resolves the variant
#: FIRST and asks only that variant's two parent predicates, so no grid is admitted
#: by both -- the partition is total and exclusive, and each family's tests drive it
#: in both directions. The extra row is what lets the seam loop absorb the FOLDED
#: cell's two slots once the primary (unfolded) pair does not match them.
FUSED_PAIR_EXTRA_ARMS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "fused_hd_pair": (("nonlinear PML magnetic", "nonlinear PML curl"),),
    "beta_complex_fused_hd_pair": (("folded beta complex", "folded beta complex"),),
    "beta_real_fused_hd_pair": (("folded beta real", "folded beta real"),),
}


def _install_fused_pair(plans: Dict[str, Any], selected: Dict[str, str],
                        fields: Any, pml: Any, sources: Any, pair_name: str,
                        curl_name: str, update_name: str, label: str,
                        pair: Any,
                        repair_paths: Tuple[str, ...] = (
                            _deposit_repair.SPLIT_FIELD_PATH,)) -> None:
    """Put a built fused pair into its two slots, with the deposit repair if it needs one.

    A VERBATIM MIRROR OF ``triton_kernels.launch._install_fused_pair``, and
    deliberately so: the two tracks reach the one shared module the same way, so a
    reader who has understood the protocol on one has understood it on both.

    THE TWO SLOTS ARE THE MECHANISM, not an accounting detail. ``step_D`` and
    ``update_E`` are separate driver consults (``driver.py:3292``, ``:3303``) with
    the inject, the symmetry fill and the wall clear in between. A pair that owns
    both runs the kernel at the first consult; the second is either a ``NoopPlan``
    — the seam carried no deposit and there is nothing to do — or the repair, which
    is the only point in the step where the injected field is final and the fused
    launch's pre-injection accumulation is still known. See :mod:`..deposit_repair`.

    THE FLAG IS NOT CHECKED HERE, for the same reason it is not checked in the
    Triton copy: it is checked in the family's own coverage predicate, which passes
    ``carries_repair=CARRIES_DEPOSIT_REPAIR`` to
    ``deposit_repair.seam_source_reasons``. A product that has not declared it
    refuses every in-seam source, so a configuration with a deposit never reaches
    this function at all. Checking it twice would let the two answers disagree.

    ``repair_paths`` is the INSTALLING PRODUCT'S declaration of which recurrence its
    repair inverts, and it defaults to the split-field one — the only repair that
    existed before ``deposit_repair.PLAIN_PATH``, and the one every family routed
    here before 2026-08-31 ships. The two no-absorber stored-E welds run
    ``update_E``'s plain overwrite instead and must say so: ``deposit_repair.save``
    refuses by name any configuration whose recurrence is not among the declared
    paths, so a family routed with the wrong declaration is refused at the first
    consult rather than silently mis-repaired. The seam loop reads the value off the
    BUILT PLAN'S own ``repair_paths`` attribute — the declaration travels with the
    module that owns the flag, never a second table here.

    THE H->D SEAM IS ROUTED TO THE WITHDRAW HOIST, 2026-09-04, AND THE BRANCH IS
    HERE RATHER THAN IN A FAMILY — the same argument that keeps the deposit bracket
    out of the families, so both tracks read one protocol. Its ``pair_name`` is
    ``withdraw_hoist.SEAM``, which is not a ``deposit_repair`` field letter: nothing
    is injected in this seam, so there is no list to select and no bracket to
    install, and what a spanning product owes instead is the driver's electric
    ``withdraw``, performed immediately before its launch. There is no trailing
    plan — nothing undoes the hoist, so the second slot holds the ``NoopPlan``
    unconditionally.
    """
    if pair_name == _withdraw_hoist.SEAM:
        plans[curl_name] = _withdraw_hoist.LeadingWithdrawPlan(
            pair, fields, sources, span=(curl_name, update_name))
        plans[update_name] = NoopPlan(update_name, pair)
        selected[curl_name] = label
        selected[update_name] = label
        return
    seam = _deposit_repair.in_seam_sources(sources, pair_name)
    if not seam:
        plans[curl_name] = pair
        plans[update_name] = NoopPlan(update_name, pair)
    else:
        leading = _deposit_repair.LeadingRepairPlan(pair, fields, pml, seam,
                                                    pair_name, repair_paths)
        plans[curl_name] = leading
        plans[update_name] = _deposit_repair.TrailingRepairPlan(
            update_name, leading, fields, pml)
    selected[curl_name] = label
    selected[update_name] = label


def declaring_plan(plan: Any) -> Any:
    """The plan whose DECLARATIONS describe the device work sitting in a slot.

    THE FAIL-CLOSED DEFAULTS BELOW READ AN ATTRIBUTE OFF THE SLOT'S OCCUPANT, and
    after a pair is installed two of the four occupants are WRAPPERS — the
    ``NoopPlan``/``TrailingRepairPlan`` in the absorbed slot and the
    ``LeadingRepairPlan`` around the launch. None of the three declares
    ``replaces_sub_steps``, so read bare they would each fall back to "replaces only
    my own slot" and the composition would report the pair's OTHER passes —
    ``zero_metal_B`` for the magnetic pair — as running on the array path when the
    kernel performed them. That is the exact misreport the fail-closed default
    exists to prevent, arrived at from the other direction.

    Every wrapper in the protocol names the plan that did the work in
    ``absorbed_by``, so the declarations are read from THERE. Both slots then report
    the same ``replaces_sub_steps``, which is a union with itself and therefore
    idempotent. A plan with no ``absorbed_by`` — every ordinary arm's — answers for
    itself, unchanged.
    """
    return getattr(plan, "absorbed_by", None) or plan


def _declared_uninstallable(spec: Any) -> Optional[str]:
    """``None`` when the product may be installed, else the reason it may not.

    ``INSTALLABLE`` IS A DECLARATION ABOUT THE PRODUCT, NOT ABOUT A RUN, which is why
    it is read here and reported on every configuration rather than folded into a
    coverage predicate. A predicate answers "can this weld serve this run" — a
    question about arithmetic, and the one its gate measures — and that answer must
    not change because of where the product sits in the driver's slot path. A module
    setting ``INSTALLABLE = False`` says the other kind of thing: the arithmetic is
    certified and the COMPOSITION is refused, on every row, for a measured reason.

    IT IS BELT AND BRACES, NOT THE RULE. :func:`_neighbouring_seam_claimant` is what
    decides a seam collision from the run; this flag is what keeps a careless edit —
    a new seam row, a reordered table, a widened absorb declaration — from installing
    a product whose own module says it must not be.

    READ OFF THE MODULE THAT DECLARED THE ARM, through ``spec.coverage``'s
    ``__module__``, because this composer finds its candidates by walking the arm
    table rather than a product table and the spec is all it holds. A spec whose
    module cannot be resolved is INSTALLABLE: fail-open is right here because a
    module that is not in ``sys.modules`` did not register the arm being walked, so
    the situation is impossible rather than permissive, and fail-closed would turn a
    resolution problem into a silent permanent refusal that reads like a decision.
    """
    module = sys.modules.get(getattr(spec.coverage, "__module__", "") or "")
    if module is None or getattr(module, "INSTALLABLE", True):
        return None
    declared = getattr(module, "INSTALLABLE_REASON", "")
    return (f"{spec.family} declares INSTALLABLE = False: "
            f"{declared or 'its own module refuses the composition'}")


def _slot_degrees(seams: Dict[str, Tuple[str, Any]]) -> Dict[str, int]:
    """How many rows of ``seams`` name each slot -- its degree on the seam path.

    COMPUTED FROM THE TABLE, NEVER SPELLED. :func:`_neighbouring_seam_claimant` asks
    its question only of a span whose two END slots both have degree >= 2 -- an
    INTERIOR edge -- and the end/interior split has to follow the table so that a
    row added later moves it without anyone having to remember to.
    """
    degrees: Dict[str, int] = {}
    for curl_name, (update_name, _seam_name) in seams.items():
        for slot in (curl_name, update_name):
            degrees[slot] = degrees.get(slot, 0) + 1
    return degrees


def _is_end_edge_span(span: Sequence[str],
                      seams: Dict[str, Tuple[str, Any]]) -> bool:
    """True when either END slot of ``span`` sits in fewer than two rows of ``seams``."""
    span = tuple(span)
    degrees = _slot_degrees(seams)
    return min(degrees.get(span[0], 0), degrees.get(span[-1], 0)) < 2


def _neighbouring_seam_claimant(context: Any, family: str, curl_name: str,
                                update_name: str, span: Sequence[str] = ()
                                ) -> Optional[Tuple[str, str]]:
    """``(claimant, refusal)`` for a product that would lose a slot to this one.

    THE RULE, SINCE 2026-09-05: the question is asked ONLY of a span that is an
    INTERIOR edge of the seam path -- both of its END slots sit in two or more rows
    of ``FUSED_PAIR_SEAMS`` -- and such a product is refused where another product
    claims the seam its LAST slot opens. An end-edge span is never asked and never
    yields. The degrees are COMPUTED from the table (:func:`_slot_degrees`), never
    spelled, so a seam row added later moves the split without anyone remembering to.

    WHY THAT IS THE RIGHT SHAPE IS A MATCHING ARGUMENT, not a corpus accident. Over a
    path of four slots the maximum matching has size 2 and it is always the two END
    edges; an interior edge belongs to no maximum matching unless both of its
    neighbours are absent. So an end-edge span must never yield, and an interior span
    must yield to a neighbour that serves. The other direction -- the seam a span's
    FIRST slot closes -- is answered exactly and cheaply by :func:`._pair_may_absorb`
    reading the live ``selected``: a fact about what installed rather than a
    prediction about a predicate, and a fact cannot annihilate with another fact.
    That is also what resolves the TIE below in the released incumbent's favour --
    the B->H pair installs first and holds ``update_H`` -- so there is no tie-break
    helper and must not be.

    Degrees on this table today: ``step_B`` 1, ``update_H`` 2, ``step_D`` 2,
    ``update_E`` 1. So ``B_to_H`` (1, 2) and ``D_to_E`` (2, 1) are end edges and
    never yield, and ``H_to_D`` (2, 2) is interior and yields.

    WHAT THE GUARD CLOSED, MEASURED 2026-09-04 THROUGH THIS COMPOSER (the H->D tie
    measurement, arrangements A-G on the ``metal_composition_matrix`` rows): with the
    ``update_H`` seam row in the table, the later half ALONE refused the RELEASED
    B->H pair on every row an admitting H->D product reached -- even a stand-in with
    no absorb declaration, which can never install (arrangement F: 3 launches / 1
    seam against the shipped 2 / 2 on the 155-row ``cart_pml_real`` cell, and 4 / 0
    against 3 / 1 on the 24 off-diagonal/folded rows). The tree was safe only
    because ``fused_hd_pair`` declares ``INSTALLABLE = False`` (arrangement E equal
    to A on 12 of 12 legs). That flag protects one product; this guard protects the
    seam, and the flag is kept as belt and braces rather than relied on.

    The earlier-neighbour direction was asked here briefly on 2026-09-04 and retired
    the same day; the block below records why, and the guard above cannot reproduce
    what it did because ``E_to_P`` is an end edge by the table's own degrees.

    WHY IT EXISTS, AND IT IS ARITHMETIC RATHER THAN A CORPUS ACCIDENT. The driver's
    slot path is ``step_B — update_H — step_D — update_E``: four slots, three seams,
    and an installed pair is an edge of a matching on that path. Launches over those
    four slots are therefore

        launches = 4 - (number of installed pairs)

    so a product is worth installing only where it RAISES the matching size. A
    product spanning ``update_H``/``step_D`` takes one slot from each neighbour,
    which can never do that: where both neighbours serve it takes the matching from
    2 to 1 (one MORE launch and one FEWER seam served); where exactly one serves it
    ties at 3 launches and one seam; and where neither serves it would gain.

    GAIN, TIE OR LOSS FOR AN H->D PRODUCT, MEASURED 2026-09-04 at COMPOSER level
    (predicate AND absorb row, driven through ``plan_step(fuse=True)`` on one fixture
    per cell) over the 179 buildable Metal rows of the 194-row
    ``h_to_d_seam_2026-09-04`` basis:

    ====  =========================================================  ====================
    rows  installing an H->D product there is                         under this rule
    ====  =========================================================  ====================
     133  a LOSS: both neighbours install, 2 pairs -> 1               B->H installs first;
                                                                      H->D refused by name
                                                                      (``_pair_may_absorb``)
      22  a TIE: only B->H serves, 3 launches / 1 seam either way     the same; the released
                                                                      incumbent keeps the slot
      22  a LOSS: only D->E serves                                    the later half names
                                                                      the D->E claimant
       2  a GAIN: NEITHER serves (the nonlinear cell, 4 / 0 -> 3 / 1) it installs, where its
                                                                      arm reaches the cell
    ====  =========================================================  ====================

    0 rows on which displacing the released B->H pair is a gain. The predicate-level
    join that preceded it (155 / 24 / 0 / 0, with no "neither" row) was right about
    the predicates and wrong about the composer, which additionally requires a
    ``FUSED_PAIR_ARMS`` row; the 2 "neither" rows are why this rule must not be
    written as a blanket veto. The refusal this returns is therefore a RUN-SPECIFIC
    measured reason in the existing "who gets the slot" wording: where no
    neighbour's product admits, this returns ``None`` and the product installs.

    IT IS SLOT ARBITRATION AND NOT A VERDICT ABOUT ARITHMETIC. Every predicate that
    reaches this admitted and every gate that released one certified it; what is
    decided here is who gets the slots.

    THE ONLY SHAPE THAT ESCAPES IT is a span that REACHES the neighbouring seam
    instead of colliding with it — a four-slot ``step_B -> update_H -> step_D ->
    update_E`` weld in ONE launch, which serves three seams where today's two pairs
    serve two in two launches. ``span`` is how that is expressed: a product whose own
    span already covers the neighbouring seam's slots makes no trade.
    """
    span = tuple(span) or (curl_name, update_name)
    # R1: AN END-EDGE SPAN IS NEVER ASKED. The matching argument is in the docstring;
    # the degrees are read off the table so the answer moves with it.
    if _is_end_edge_span(span, FUSED_PAIR_SEAMS):
        return None
    neighbours = []
    # THE EARLIER-NEIGHBOUR HALF WAS REMOVED 2026-09-04, MEASURED, NOT ARGUED. It
    # asked the same question in the other direction -- "does a product claim the seam
    # my FIRST slot closes?" -- and it did two things, both wrong:
    #
    # * it was REDUNDANT where it was meant to help. An H->D span's last slot is
    #   ``step_D``, which opens the D->E seam, so the later half above already names a
    #   D->E claimant and returns before the earlier half is reached.
    # * it REVERSED A RELEASED RULING where it did fire. The only product class whose
    #   last slot opens no seam is E->P (``update_P`` is terminal), so the earlier half
    #   fired there and there alone -- refusing the polarization pair because a D->E
    #   product admits, while the later half was already refusing that D->E product
    #   because the polarization pair claims ``update_E``. Both neighbours annihilated
    #   and the seam went unserved. Measured on a dispersive PML run (poles=2): the
    #   later half names ``cuda_fused_polarization_pair`` for the D->E product, and the
    #   earlier half named ``cuda_dispersive_fused_electric_pair`` for the polarization
    #   pair. The 2026-09-02 trade ruling gives ``update_E`` to the E->P incumbent, and
    #   this function may not re-litigate it from the other side.
    #
    # What the earlier half was FOR -- an H->D product installing on a row where only
    # the B->H neighbour serves, which the launch algebra makes a TIE rather than a
    # gain -- was settled on 2026-09-04/05 with the product in hand: it is a TIE, the
    # released incumbent keeps the slot, and the mechanism is not a half of this
    # function at all but `_pair_may_absorb` reading the live `selected` after the
    # B->H pair installs first. What this function gained instead is the end-edge
    # guard above, which is what keeps the later half from refusing that incumbent.
    if update_name in FUSED_PAIR_SEAMS:
        later_update = FUSED_PAIR_SEAMS[update_name][0]
        if later_update not in span:
            neighbours.append((update_name, later_update, update_name))
    for neighbour_curl, neighbour_update, shared in neighbours:
        for spec in arms.registered(neighbour_curl):
            if (not spec.is_weld or neighbour_update not in spec.replaces
                    or spec.family == family):
                continue
            if _declared_uninstallable(spec) is not None:
                continue
            if spec.gate is not None:
                gated, gate_refusal = _guarded_plan(
                    lambda spec=spec: bool(spec.gate(context)),
                    f"the {spec.family} pair gate")
                if gate_refusal is not None or not gated:
                    continue
            verdict = _guarded_verdict(
                lambda spec=spec: spec.coverage(context, spec.slot),
                f"the {spec.family} pair predicate")
            if verdict.covered:
                return spec.family, (
                    f"{spec.family} admits this run's "
                    f"{neighbour_curl}/{neighbour_update} seam and would lose "
                    f"{shared} to this product; over the step_B..update_E slot path "
                    f"launches are 4 - (installed pairs), so a product taking a slot "
                    f"from a neighbouring seam can only tie or lose, and the slot is "
                    f"left with the product that already serves it. See "
                    f"_neighbouring_seam_claimant -- this is slot arbitration, not a "
                    f"verdict about {family}'s arithmetic, which its own gate "
                    f"certifies")
    return None


def _install_fused_pairs(plans: Dict[str, Any], reasons: Dict[str, Tuple[str, ...]],
                         selected: Dict[str, str], context: Any, fields: Any,
                         pml: Any, sources: Any,
                         fuse_labels: Optional[Sequence[str]] = None) -> None:
    """The opt-in fusion block: at most one pair per seam, refusing on every doubt.

    GUARDED EXACTLY AS AN ARM IS. ``plan_step`` never raises, so a pair predicate
    that raises is a refusal (:func:`_guarded_verdict`) and a builder that raises or
    returns ``None`` leaves the seam unfused with a named reason
    (:func:`_guarded_plan`). An opt-in optimisation that CRASHES the plan is
    strictly worse than one that refuses it.

    TWO ADMITTERS ON ONE SEAM IS AN AMBIGUITY, not a pick. The unwired weld rows
    were never run through ``_select_slot``, so the (b) clause has never been
    applied to them; it is applied here rather than resolved by table order.

    A PAIR MAY ABSORB A SLOT ONLY WHERE THE ARM TABLE ALREADY GAVE THAT SLOT TO THE
    ARM THE FUSED KERNEL IMPLEMENTS (:func:`._pair_may_absorb`, imported from the
    Triton composer). This block is the one place here that fills a ``STEP_ORDER``
    slot without consulting the arm table, and writing it unconditionally would be a
    hole straight through clauses (b) and (c).

    ``fuse_labels`` IS THE CALLER'S OFFER AND ``None`` MEANS "every product", which
    is what every caller before the dispatch seam meant by ``fuse=True``. The
    dispatch ladder passes the set its release ADMITS, and the reason is the
    2026-09-02 conductive lesson from the Triton track: a fused product occupies
    BOTH slots of its seam, so a ladder that installed it and then refused the
    label could only refuse the plan WHOLE, taking the separate certified arms
    underneath it down with it. An un-offered candidate is skipped HERE, its seam
    keeps the per-slot plans the arm table already built, and the ladder's
    refuse-by-name clause stays the backstop it was written as rather than the
    thing that stops it.
    """
    offered = None if fuse_labels is None else set(fuse_labels)
    for curl_name, (update_name, pair_name) in FUSED_PAIR_SEAMS.items():
        candidates = []
        for spec in arms.registered(curl_name):
            if not spec.is_weld or update_name not in spec.replaces:
                continue
            key = f"fused_pair_{spec.family}"
            # THE OFFER IS ASKED BEFORE THE GATE AND BEFORE THE PREDICATE, because
            # it is a fact about the CALLER rather than about the grid: a product
            # this caller will not run is not a product whose coverage the record
            # should be reporting on. The spelling carries the phrase
            # ``triton_kernels/launch.py`` uses ("offered to run") — the route
            # gate's envelope leg greps for it, and a second wording would make
            # that leg silently stop recognising the clause it watches.
            if offered is not None and spec.label not in offered:
                reasons[key] = (
                    f"{spec.label} is not among the fused labels this caller "
                    f"offered to run ({', '.join(sorted(offered)) or 'none'}); the "
                    f"{curl_name}/{update_name} seam keeps the per-slot arms",)
                continue
            # THE DECLARATION IS ASKED FIRST, BEFORE THE GATE, and the order is the
            # point: a missing absorb declaration is a fact about the TABLE, true of
            # every configuration, so it is reported on every configuration. Asking
            # the gate first would make the reason appear and disappear with the
            # grid, which reads as a claim about the run.
            pair_arms = FUSED_PAIR_ARMS.get(spec.family)
            if pair_arms is None:
                reasons[key] = (
                    f"{spec.family} has no absorb declaration: which arm each of "
                    f"its two slots implements has not been established on this "
                    f"track, and a fused product may not substitute a numerical "
                    f"product no arm admitted",)
                continue
            # THE PRODUCT'S OWN INSTALLATION DECLARATION, ASKED BESIDE THE ABSORB
            # ONE AND FOR THE SAME REASON: it is a fact about the PRODUCT, true of
            # every configuration, so it is reported on every configuration. See
            # :func:`_declared_uninstallable`.
            uninstallable = _declared_uninstallable(spec)
            if uninstallable is not None:
                reasons[key] = (uninstallable,)
                continue
            # THE GATE IS HONOURED, and a gated-out arm contributes NO reason —
            # clause (e), the same rule `ArmSpec.bind` applies for the per-slot
            # table. A gate that RAISES is a refusal rather than an exception out of
            # `plan_step`, which never raises.
            if spec.gate is not None:
                gated, gate_refusal = _guarded_plan(
                    lambda spec=spec: bool(spec.gate(context)),
                    f"the {spec.family} pair gate")
                if gate_refusal is not None:
                    reasons[key] = (gate_refusal,)
                    continue
                if not gated:
                    continue
            verdict = _guarded_verdict(
                lambda spec=spec: spec.coverage(context, spec.slot),
                f"the {spec.family} pair predicate")
            if not verdict.covered:
                reasons[key] = verdict.reasons
                continue
            candidates.append((spec, pair_arms))
        if not candidates:
            continue
        if len(candidates) > 1:
            named = ", ".join(sorted(spec.family for spec, _ in candidates))
            for spec, _ in candidates:
                # DELIBERATELY NOT SPELLED "admit this configuration": that phrase
                # is how `_select_slot`'s per-SLOT ambiguity is recognised, and a
                # probe scanning `reasons` for it would report `fused_pair_*` as an
                # ambiguous slot. This is a seam ambiguity, and it says so.
                reasons[f"fused_pair_{spec.family}"] = (
                    f"{named} all claim the {curl_name}/{update_name} seam; it is "
                    f"left unfused rather than assigned by table order",)
            continue
        spec, pair_arms = candidates[0]
        key = f"fused_pair_{spec.family}"
        # THE PRIMARY ROW FIRST, THEN EVERY DECLARED EXTRA, and the pair absorbs on
        # the FIRST arm pair the table already gave both slots to -- the certified
        # Triton loop's clause, verbatim. At most one can match, so this is a lookup
        # and not a preference order; a product with no matching declaration is
        # refused with the PRIMARY row's reason, which names the arm the slots went to.
        refusal = _pair_may_absorb(selected, curl_name, update_name, pair_arms)
        if refusal is not None:
            for extra_arms in FUSED_PAIR_EXTRA_ARMS.get(spec.family, ()):
                if _pair_may_absorb(selected, curl_name, update_name,
                                    extra_arms) is None:
                    refusal = None
                    break
        if refusal is not None:
            reasons[key] = (refusal,)
            continue
        # A NEIGHBOURING SEAM'S CLAIM, ASKED BEFORE THIS PAIR IS BUILT. See
        # :func:`_neighbouring_seam_claimant`: over ``step_B..update_E`` launches are
        # ``4 - #pairs``, so a product taking a slot from a neighbouring seam can
        # only tie or lose -- and an END-EDGE span (both released pairs) is never
        # asked, so this refuses nothing that installs today.
        neighbour = _neighbouring_seam_claimant(context, spec.family, curl_name,
                                                update_name, (curl_name, update_name))
        if neighbour is not None:
            reasons[key] = (neighbour[1],)
            continue
        pair, builder_refusal = _guarded_plan(
            lambda spec=spec: spec.plan(context, spec.slot),
            f"the {spec.family} pair builder")
        if builder_refusal is not None:
            reasons[key] = (builder_refusal,)
        elif pair is not None:
            # WHICH repair the bracket installs is the PLAN'S OWN declaration
            # (mirroring the family module's REPAIR_PATHS); a plan that declares
            # none gets the split-field default every pre-PLAIN_PATH family ships.
            _install_fused_pair(plans, selected, fields, pml, sources, pair_name,
                                curl_name, update_name, spec.label, pair,
                                tuple(getattr(pair, "repair_paths",
                                              (_deposit_repair.SPLIT_FIELD_PATH,))))
        else:
            reasons[key] = ("the fused pair was refused",)


def plan_step(fields: Any, pml: Any, residency: Optional[Residency] = None,
              sources: Any = None, synced: Sequence[str] = (),
              contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
              complex_probe: Any = None, beta_probe: Any = None,
              folded_complex_probe: Any = None,
              cylindrical_complex_probe: Any = None,
              fuse: bool = False,
              fuse_labels: Optional[Sequence[str]] = None,
              ) -> MetalStepPlan:
    """Ask every predicate independently and report the covered subset with its refusals.

    Never raises and never returns None: a configuration nothing covers is a plan
    that replaces nothing, which is the array path, which is the correct answer.

    THE FAIL-CLOSED CONTRACT is the Triton composer's, unchanged and shared: the
    arm table, :func:`_select_slot` and clauses (a)-(f) are IMPORTED from
    ``triton_kernels.launch``, so exactly one admitter fills a slot, two or more
    leave it UNSELECTED naming all admitters, a raising predicate is a REFUSAL,
    and an unselected slot is the array path.

    THE ARM TABLE CARRIES EVERY CERTIFIED FAMILY AS OF TRANCHE 2, and every row is
    REGISTERED by the module that owns it rather than spelled here: the real-field
    PML curl and the ordinary ``dsigw`` constitutive (this module), the no-PML NULL
    family (``update_H`` returns at stepping.py:944-945 under an inactive absorber
    and ``update_E`` at :983-984 with ``stores_E`` False), the complex/Bloch curl
    and constitutive, the special_kz real and complex beta curls with the real
    arm's constitutive companion, the BFAST curl, and the off-diagonal
    ``update_E``. Eighteen rows over four slots.

    THE TABLE IS COMPLETED BEFORE IT IS READ, by :func:`.arms.ensure_registered`,
    and that call is load-bearing rather than defensive. Registration is an import
    side effect: without it, ``import metal_kernels.launch`` alone produced a
    six-row table, so the SELECTION — and whether an ambiguity was detected at all
    — depended on an unrelated earlier import.

    ``complex_probe``, ``beta_probe``, ``folded_complex_probe`` and
    ``cylindrical_complex_probe`` are the measured expansion artifacts the complex,
    beta, folded-complex and cylindrical-complex arms bind their multiply arm from.
    FOUR ARGUMENTS AND NOT ONE, because the four families require DIFFERENT PATTERN
    SETS: the beta family needs a complex scalar with a signed-zero real word, the
    folded complex family needs the mirror parity as a complex coefficient on the
    LEFT, the cylindrical complex family needs a complex ROW coefficient on the left
    (the i*m/r term) and a complex SCALAR on the left whose real word is a signed
    zero (the |m| = 1 axis increment), and none of those orientations appears in the
    base five. A shared argument would let an artifact that classified five
    orientations licence a kernel that performs seven. ``None`` is not a default arm:
    each family falls back to its own environment variable and REFUSES BY NAME when
    no artifact can be read, because which arm this host's reference takes is a
    measured platform fact.

    THE NULL FAMILY'S PREDICATE IS SHARED VERBATIM WITH THE TRITON PACKAGE; ITS
    PLAN AND ITS COMPOSITION ARE NOT, and this docstring said otherwise for a
    round. The predicate carries no backend clause by design — a product that
    launches nothing is correct on NumPy — so it is imported and never copied. The
    COMPOSITION needed Metal work that tranche 1 did not do, and the two defects
    were measured rather than argued: the Triton plan's ``run(guard=None)``
    signature is a ``TypeError`` under this package's ``run(contract=...)`` launch
    protocol, and declaring a null ``planned`` made the residency verdict refuse
    the family's own configuration, demanding mirrors of arrays a no-PML run does
    not allocate. See ``no_pml_constitutive``'s module docstring for both
    measurements and the artifact rows that carry them.

    THE PLANNED/NULL SPLIT below is the second of those fixes, and it reads a
    DECLARED attribute off the plan rather than matching the arm's label.

    Every driver slot now has at least one registered arm. A configuration may
    still leave a slot unselected, but that is a named predicate refusal rather
    than a hard-coded absence.

    THE RESIDENCY VERDICT IS COMPUTED HERE and nowhere else, because it needs the
    LIVE set — which sub-steps actually run — and no single plan knows that. A plan
    whose ``residency`` verdict refuses is still individually correct; what it may
    not do is hold its mirrors across a complete step. The probe asserts the
    verdict; nothing in this package acts on it, because dispatch is not wired.

    ``fuse`` IS OPT-IN AND DEFAULTS OFF, exactly as it does on the Triton composer.
    Off, this function composes what it composed before the cross-sub-step block
    existed: one plan per slot, every weld row invisible because ``arms_for`` skips
    an unwired arm. On, a pair whose predicate admits and whose two slots were won
    by the arms its kernel implements takes BOTH slots — the launch in the curl
    slot and, in the constitutive slot, either a :class:`NoopPlan` or the deposit
    repair. ``replaces`` reports both names either way, because both sub-steps
    really are replaced.

    ``fuse_labels`` NARROWS THE OFFER TO PARTICULAR PRODUCTS. ``None`` — the
    default, and what every caller that predates the dispatch seam means — offers
    every product. A sequence offers exactly those labels and skips the rest with a
    named reason; see :func:`_install_fused_pairs`. It is read only when ``fuse``
    is true, because the fusion block is what consumes it.
    """
    plans: Dict[str, Any] = {}
    reasons: Dict[str, Tuple[str, ...]] = {}
    selected: Dict[str, str] = {}

    # THE TABLE IS COMPLETED BEFORE IT IS READ. Registration is an import side
    # effect, so without this the composition depended on which family modules the
    # caller happened to have imported — and an arm missing from the table does not
    # produce a refusal, it produces a DIFFERENT SELECTION with no ambiguity
    # detected. See `arms.ensure_registered`.
    arms.ensure_registered()

    context = arms.StepContext(fields, pml, residency, contract_variants,
                               sources=sources, synced=synced,
                               extra={"complex_probe": complex_probe,
                                      "beta_probe": beta_probe,
                                      "folded_complex_probe": folded_complex_probe,
                                      "cylindrical_complex_probe":
                                          cylindrical_complex_probe})
    for slot in STEP_ORDER:
        table = arms.arms_for(slot, context)
        if table:
            _select_slot(slot, table, arms.ambiguity(slot),
                         plans, reasons, selected)
        else:
            reasons[slot] = (NO_ARM_REASONS[slot],)

    # THE CROSS-SUB-STEP SEAM, AFTER the per-slot table and never instead of it.
    # `_pair_may_absorb` reads what the table selected, so the table has to have
    # run; and a refused fusion is not a refused step — both sub-steps keep the
    # separate plans built above and the reason is reported alongside them.
    if fuse:
        _install_fused_pairs(plans, reasons, selected, context, fields, pml, sources,
                             fuse_labels)

    # THE PLANNED/NULL SPLIT, and it is read off the PLAN rather than off the arm
    # label. `planned` means "runs on the device, against the mirror"; a null plan
    # runs on neither, binds no buffer and writes no host array, so it neither
    # requires a mirror nor stales one. Asked as a declared attribute with a
    # FAIL-CLOSED default: a plan that does not say is assumed to do device work,
    # which is the answer that demands mirrors rather than the one that waives
    # them. Matching on the arm's label instead would make a rename a silent
    # correctness change.
    #
    # READ THROUGH THE WRAPPER, not off it: after a pair is installed two slots
    # hold objects that declare nothing of their own and name the launch in
    # `absorbed_by` instead. See `declaring_plan` for why reading them bare would
    # turn the fail-closed default into an under-report.
    filled = tuple(name for name in plans if name in STEP_ORDER)
    null = tuple(name for name in filled
                 if not getattr(declaring_plan(plans[name]),
                                "performs_device_work", True))
    # A PLAN MAY PERFORM MORE DRIVER PASSES THAN ITS SLOT NAMES, and it says so
    # rather than being inferred from its label. The mirror fill fills `fill_B` and
    # ALSO performs `fill_folded_far_ghosts_B` on the device (driver.py:3287), a
    # residency fact no slot name carries. Read as a DECLARED attribute with a
    # FAIL-CLOSED default — a plan that does not say replaces only its own slot,
    # which is the answer that DEMANDS a mirror rather than waiving one. Matching on
    # the arm's label instead would make a rename a silent correctness change, the
    # same reason the planned/null split above reads an attribute.
    planned: List[str] = []
    planned_volumes: Dict[str, Tuple[str, ...]] = {}
    for name in filled:
        if name in set(null):
            continue
        replacements = tuple(
            getattr(declaring_plan(plans[name]), "replaces_sub_steps", None)
            or (name,))
        planned.extend(replacements)
        declared = getattr(declaring_plan(plans[name]), "volumes", None)
        for replacement in replacements:
            # A plan that predates per-family declarations keeps the historical
            # fail-closed table. An explicit empty tuple stays explicit; ``None``
            # alone means "not declared".
            planned_volumes[replacement] = tuple(
                sub_step_volumes(replacement) if declared is None else declared)
    planned = tuple(sorted(set(planned)))

    live = live_sub_steps(fields, pml, sources)
    residency_verdict = residency_coverage(
        None if residency is None else residency.names,
        planned, live, synced, null, planned_volumes)

    return MetalStepPlan(plans, reasons, selected, residency_verdict,
                         live or (), synced)




def explain(fields: Any, pml: Any, residency: Optional[Residency] = None) -> Coverage:
    """The aggregate curl verdict, for reports that want one line."""
    return pml_curl_coverage(fields, pml, None, residency)


# ---------------------------------------------------------------------------
# Fingerprints: what was certified, and on which toolchain
# ---------------------------------------------------------------------------

#: The host modules whose bytes are hashed beside the kernel sources. The host is
#: hashed too because it CHOOSES THE YEE SUB-LATTICE, binds the coefficient
#: pointers and decides the specialisation constants — each of those is a silent
#: wrong answer if it changes, and none of them is visible in a kernel source.
#:
#: THE SPINE MODULES ARE IN THE LIST, and leaving them out would have been the
#: quiet way to lose the guarantee. ``device.py`` owns the mirror layer, so it
#: decides which host bytes a kernel reads and whether they are float32 or float2;
#: ``plans.py`` owns the variant selector, so it decides whether a launch is the
#: guarded source or the fast one; ``arms.py`` owns which product fills a slot.
#: Each of those is a silent wrong answer if it changes and none of them is
#: visible in a kernel source — the same argument that put ``launch.py`` here.
#: ``templates.py`` and ``preconditions.py`` are listed for the same reason one
#: step removed: they emit source fragments and bound the claim.
#:
#: THE LIST IS DERIVED FROM THE PACKAGE, NOT ENUMERATED. A literal tuple was
#: correct for a one-family tranche and is a hole in a seven-family one: the
#: failure mode is a family module that binds pointers, is never added to the
#: literal, and is therefore invisible to every fingerprint check that follows.
#: Deriving it means a new module is covered the moment it lands and a re-cut is
#: FORCED rather than remembered. Test files are excluded — they emit no source
#: and bind no pointer — and the exclusion is by prefix rather than by listing, so
#: it cannot go stale either.
def _host_modules() -> Tuple[str, ...]:
    import os  # noqa: PLC0415

    here = os.path.dirname(os.path.abspath(__file__))
    return tuple(sorted(
        name for name in os.listdir(here)
        if name.endswith(".py") and not name.startswith("test_")))


#: The host modules whose bytes are hashed beside the kernel sources. The host is
#: hashed too because it CHOOSES THE YEE SUB-LATTICE, binds the coefficient
#: pointers and decides the specialisation constants — each of those is a silent
#: wrong answer if it changes, and none of them is visible in a kernel source.
#: That argument covers the spine exactly as it covers this file: ``device.py``
#: decides which host bytes a kernel reads and in which dtype, ``plans.py`` decides
#: whether a launch is the guarded source or the fast one, ``arms.py`` decides
#: which product fills a slot, and ``templates.py``/``preconditions.py`` emit
#: source fragments and bound the claim.
HOST_MODULES: Tuple[str, ...] = _host_modules()


def _module_sha256(name: str) -> str:
    import hashlib  # noqa: PLC0415
    import os  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def compute_fingerprints() -> Dict[str, Any]:
    """Everything ``fingerprints.json`` records, computed from the tree as it is.

    ``kernel_source_sha256`` covers every shipped specialisation in BOTH
    contraction modes: the guard is a property of the source here, so the guarded
    and unguarded variants are different kernels and each gets its own fingerprint.
    """
    try:
        import torch  # noqa: PLC0415

        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001 - a fingerprint may be computed without torch
        torch_version = None
    sources: Dict[str, Dict[str, str]] = {}
    for mode in shaders.CONTRACT_MODES:
        for label, source in shaders.enumerate_sources(mode).items():
            sources[f"{label}/contract-{mode}"] = {
                "sha256": shaders.source_sha256(source),
                "bytes": str(len(source)),
            }
    return {
        "metal_kernels": {
            "kernel_source_sha256": sources,
            "host_sha256": {name: _module_sha256(name) for name in HOST_MODULES},
            "validated_torch_versions": [torch_version] if torch_version else [],
            "validated_metal_frontend": metal_frontend_version(),
        },
    }


def fingerprints_path() -> str:
    import os  # noqa: PLC0415

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fingerprints.json")


def load_fingerprints() -> Dict[str, Any]:
    import json  # noqa: PLC0415

    with open(fingerprints_path(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_fingerprints() -> Dict[str, Any]:
    """Rewrite ``fingerprints.json`` from the tree. The documented regeneration path.

    Run it DELIBERATELY, after a change that is meant to move the arithmetic or
    the toolchain, and re-cut the gate. Running it to make a red test green is how
    a fingerprint stops meaning anything.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    record = compute_fingerprints()

    # MERGE, DO NOT REPLACE. compute_fingerprints() returns only the keys this
    # function owns ("metal_kernels"), so a wholesale rewrite DELETES every other
    # top-level entry in the file. Measured 2026-08-19: that is fatal to per-family
    # device welds, which live beside this key and record which source bytes a gate
    # certified — and the deletion would happen on the documented regeneration path,
    # i.e. exactly when someone edits a kernel, which is precisely when a weld must
    # survive to go red. Triton has no such rewriter, which is why the weld template
    # is safe there and was not safe here.
    path = fingerprints_path()
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}          # unreadable is rewritten, not merged into

    merged = dict(existing)
    merged.update(record)          # this function's own keys win
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record

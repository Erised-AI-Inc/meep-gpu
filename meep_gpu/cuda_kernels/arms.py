"""The hand-CUDA arm table, as DATA — which family may fill which sub-step slot.

WHY THIS EXISTS. This track ships 22 predicate families that admit 759 of 759
sub-step slots on 186 of 186 engine-accepted corpus rows
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closed/
coverage_report.txt:270-285``). NOTHING SELECTED ONE. The predicates were reachable
only from the census battery and from two stress harnesses under ``parity/``, each
of which asked a hand-picked subset in a hand-written order and resolved an
ambiguity — where it noticed one at all — by raising. "A CUDA-only run" had no
subject: there was no table saying which product claims which slot, and no composer
to fail closed when two of them claim the same one.

THE FAIL-CLOSED CONTRACT IS NOT REDEFINED HERE. ``_Arm``, ``_select_slot`` and
``STEP_ORDER`` are IMPORTED from :mod:`meep_gpu.triton_kernels.launch` — the same
backend-free import :mod:`meep_gpu.metal_kernels.arms` takes at its line 52 — so
all three tracks answer an ambiguity, a raising predicate and an unselected slot
identically. Exactly one admitter fills a slot; two or more leave the slot
UNSELECTED naming all admitters; a predicate that RAISES is a refusal; an
unselected slot is the array path, which is always correct.

WHAT IS DELIBERATELY NOT PORTED FROM THE METAL TABLE:

* **Residency.** It is a Metal concept — which private device mirrors a plan holds
  across a step — and this track has none: the hand-CUDA kernels launch against the
  CuPy arrays the array path already owns. :class:`StepContext` therefore carries no
  ``residency`` field and no arm takes one.
* **A binding ceiling.** Metal's composer must respect ``MAX_BUFFER_BINDINGS=31``.
  CUDA's parameter space is 32,764 B on sm_70+ against a widest shipped kernel of
  256 B (``fused_magnetic_pair.py:187-192``,
  ``parity/meep_gpu/build_cuda_fusion_matrix.py:33-37``), so the ceiling is nowhere
  near binding and is not carried across. Importing that constraint would refuse
  compositions this backend can perform.

THE REASONS TUPLE IS A TUPLE, and that is the single sharpest edge in this file.
``triton_kernels/launch.py:2084`` normalises a verdict with
``tuple(str(reason) for reason in reasons)``; every shipped CUDA predicate returns
``(bool, str)``, so handing the bare string through produces one "reason" PER
CHARACTER. Measured on this tree: a refusal of ``"no PML"`` came back as
``('no', 'o', ' ', 'P', 'M', 'L')`` prefixed six times. :func:`coverage_adapter` is
the ONE place the wrap happens, and it wraps in a 1-tuple.

REGISTRATION IS NOT DISPATCH, and this step does not wire one. ``meep_gpu.fastpath.
plan_fast_path`` still returns ``None`` on every branch and this package does not
touch it. :func:`plan_step` decides what a dispatch WOULD compose and says which arm
won each slot; the launch-argument resolvers below are attached to the plan and are
NOT called at plan time, because resolving them needs a CUDA host and belongs to the
step that gates a live dispatch.

THE TABLE IS POPULATED BY IMPORT, WHICH IS WHY :func:`ensure_registered` EXISTS —
the same hazard, and the same fix, the Metal table carries: a table read before
:mod:`.registry` has run answers with whatever the caller happened to import, and a
missing arm does not produce a refusal, it produces a DIFFERENT SELECTION with no
ambiguity detected.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# One definition of the fail-closed machinery, imported rather than re-implemented.
from ..triton_kernels.coverage import Coverage
from ..triton_kernels.launch import _Arm, _select_slot, STEP_ORDER  # noqa: F401


# ---------------------------------------------------------------------------
# The composition context
# ---------------------------------------------------------------------------

class StepContext:
    """Everything an arm's coverage and plan closures need, in one object.

    A registry entry cannot close over ``fields``/``pml``/``grid`` — it is built at
    import time — so the context is passed at composition time and each entry is a
    pair of functions of it. That is the whole reason the table can be data.

    ``grid`` DEFAULTS TO ``fields.grid`` BUT IS SEPARATELY SETTABLE, because every
    shipped CUDA predicate takes it as its own third positional argument and the
    census factors the CuPy-backend clause by handing in a proxy grid rather than
    filtering the clause out of a refusal string (``predicate_battery.py:372-389``).
    A composer that could not be given that proxy could not reproduce the census.

    ``licenses`` IS THREE ENTRIES AND NOT ONE. The complex families bind DIFFERENT
    expansion records, and the census establishes that this is a property of the
    arithmetic rather than a convenience: the Cartesian/cylindrical/folded/beta
    complex arms take the four-pattern record, the complex no-absorber arms need
    the seven-pattern one because ``update_P``'s first term is a complex64 array
    times a python float that the four-pattern table does not classify, and the
    complex off-diagonal arms need the PARITY licence because the mirror ghost puts
    a python int on the LEFT of a complex64 plane (``predicate_battery.py:475-495``
    and :985-1005). A shared argument would let a record that classified four
    orientations licence a kernel that performs seven.

    ``None`` IS NOT A DEFAULT ARM. A family whose licence is absent asks its
    predicate with ``license=None`` and gets that family's own named refusal, which
    is the fail-closed answer: which arm this host's reference multiply takes is a
    MEASURED platform fact and a composer may not assume it.

    ``sources`` IS ``None`` UNTIL A CALLER DECLARES IT, AND ``None`` IS NOT ``()``.
    No slot arm reads it -- the four sub-step predicates do not care what the driver
    injects between them -- but the FUSION BLOCK does, because a fused pair spans the
    injection. ``Fields`` does not hold the source list, so a product asked with
    ``None`` refuses rather than assuming an empty seam
    (``deposit_repair.seam_source_reasons``'s ``undeclared`` clause). Defaulting to
    ``()`` here would turn "nobody told me" into "there are none", which is the
    over-covering the clause exists to prevent.
    """

    __slots__ = ("fields", "pml", "grid", "licenses", "subnormal_policy", "sources")

    def __init__(self, fields: Any, pml: Any, grid: Any = None,
                 licenses: Optional[Dict[str, Any]] = None,
                 subnormal_policy: Any = None, sources: Any = None) -> None:
        self.fields = fields
        self.pml = pml
        self.grid = getattr(fields, "grid", None) if grid is None else grid
        self.licenses = dict(licenses or {})
        self.subnormal_policy = subnormal_policy
        self.sources = sources

    def license_for(self, key: Optional[str]) -> Any:
        """The expansion licence one family binds, or ``None`` when it has none."""
        return None if key is None else self.licenses.get(key)


# ---------------------------------------------------------------------------
# The slot -> argument mappings, one per positional shape
# ---------------------------------------------------------------------------
#
# MEASURED 2026-08-28 by reading every shipped ``covers_*`` signature under
# ``meep_gpu/cuda_kernels/``: the 22 census families call through exactly SIX
# positional shapes. Anything a family needs beyond the slot name — a licence, a
# policy — rides on the context, never on the slot.

#: Curl slot -> the ``sub_step`` argument. IDENTITY, and spelled anyway: the two
#: curl slots are named for the sub-steps, the two constitutive slots are not, and
#: a table where one shape is a dict and the other an implicit identity makes one of
#: them the special case. ``stress_cuda_scale.py:255-257`` spells its own for the
#: same reason.
CURL_SLOT_SUB_STEPS: Dict[str, str] = {"step_B": "step_B", "step_D": "step_D"}

#: Constitutive slot -> the ``side`` argument. The record says ``update_H`` /
#: ``update_E``; every constitutive predicate takes ``H`` / ``E``. Converted in ONE
#: place rather than at each call site, because a call site that forgot would ask
#: for a side no predicate has heard of and the ``KeyError`` would arrive three
#: frames away (``stress_cuda_scale.py:270-275``).
CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}

#: Fill slot -> the in-seam ``family`` argument. The two fill slots are named for
#: the driver's sub-steps (``fill_B`` stands for ``fill_symmetry_bc_B`` plus
#: ``fill_folded_far_ghosts_B``), and every in-seam predicate and launcher takes the
#: FAMILY letter instead (``in_seam_coverage.FAMILIES``). Converted here, once, for
#: the reason :data:`CONSTITUTIVE_SLOT_SIDES` is: a call site that forgot would ask
#: for a family no predicate has heard of.
FILL_SLOT_FAMILIES: Dict[str, str] = {"fill_B": "B", "fill_D": "D"}

#: The names of the shapes, as :func:`coverage_adapter` takes them. SIX were measured
#: off the shipped slot predicates on 2026-08-28. The SEVENTH, ``fill_by_family``, is
#: the in-seam shape ``(fields, grid, family)`` -- keyed by the FAMILY whose seam the
#: pass belongs to rather than by a sub-step -- and it carries one arm, the mirror
#: fill (:func:`covers_mirror_fill`), registered 2026-09-27.
SHAPES: Tuple[str, ...] = (
    "curl_by_sub_step",             # (fields, pml, grid, sub_step)
    "constitutive_by_side",         # (fields, pml, grid, side)
    "whole_slot",                   # (fields, pml, grid)
    "licensed_curl_by_sub_step",    # (..., sub_step, license=, subnormal_policy=)
    "licensed_constitutive_by_side",  # (..., side, license=, subnormal_policy=)
    "licensed_whole_slot",          # (..., license=, subnormal_policy=)
    "fill_by_family",               # (fields, grid, family)
)


def _bind_curl(predicate: Any, context: StepContext, slot: str,
               license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid,
                     CURL_SLOT_SUB_STEPS[slot])


def _bind_constitutive(predicate: Any, context: StepContext, slot: str,
                       license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid,
                     CONSTITUTIVE_SLOT_SIDES[slot])


def _bind_whole_slot(predicate: Any, context: StepContext, slot: str,
                     license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid)


def _bind_licensed_curl(predicate: Any, context: StepContext, slot: str,
                        license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid,
                     CURL_SLOT_SUB_STEPS[slot],
                     license=context.license_for(license_key),
                     subnormal_policy=context.subnormal_policy)


def _bind_licensed_constitutive(predicate: Any, context: StepContext, slot: str,
                                license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid,
                     CONSTITUTIVE_SLOT_SIDES[slot],
                     license=context.license_for(license_key),
                     subnormal_policy=context.subnormal_policy)


def _bind_licensed_whole_slot(predicate: Any, context: StepContext, slot: str,
                              license_key: Optional[str]) -> Any:
    return predicate(context.fields, context.pml, context.grid,
                     license=context.license_for(license_key),
                     subnormal_policy=context.subnormal_policy)


def _bind_fill(predicate: Any, context: StepContext, slot: str,
               license_key: Optional[str]) -> Any:
    # NO ``pml``: the in-seam passes read no absorber -- ``fill_symmetry_bc_*`` runs
    # whether or not a layer is installed -- and their predicates take the grid and
    # the family alone (``in_seam_coverage.covers_fill_symmetry``).
    return predicate(context.fields, context.grid, FILL_SLOT_FAMILIES[slot])


_SHAPE_BINDERS: Dict[str, Callable[..., Any]] = {
    "curl_by_sub_step": _bind_curl,
    "constitutive_by_side": _bind_constitutive,
    "whole_slot": _bind_whole_slot,
    "licensed_curl_by_sub_step": _bind_licensed_curl,
    "licensed_constitutive_by_side": _bind_licensed_constitutive,
    "licensed_whole_slot": _bind_licensed_whole_slot,
    "fill_by_family": _bind_fill,
}


def coverage_adapter(predicate: Any, shape: str,
                     license_key: Optional[str] = None) -> Callable[..., Coverage]:
    """The ONE place a ``(bool, str)`` CUDA predicate becomes a ``Coverage``.

    THE REASON IS WRAPPED IN A TUPLE, and that is not decoration. The shared
    normaliser at ``triton_kernels/launch.py:2084`` iterates whatever is in
    ``.reasons``; a bare string is iterable, so a single-string refusal arrives at
    the composer as one reason per CHARACTER, each carrying the arm's prefix.
    Measured on this tree: ``"no PML"`` came back as six prefixed one-character
    "reasons". Every family goes through here so no family can get it wrong once.

    A predicate that raises, or returns something that is not a ``(covered, reason)``
    pair, is NOT handled here on purpose: the shared ``_guarded_verdict`` already
    turns both into a named refusal, and catching them a second time would give this
    track a different answer from the other two for the same event.
    """
    if shape not in _SHAPE_BINDERS:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    binder = _SHAPE_BINDERS[shape]

    def coverage(context: StepContext, slot: str) -> Coverage:
        covered, reason = binder(predicate, context, slot, license_key)
        return Coverage(bool(covered), (str(reason),))

    return coverage


# ---------------------------------------------------------------------------
# The plan side — what a dispatch WOULD launch, and how it would resolve the args
# ---------------------------------------------------------------------------

#: Which Yee sub-lattice each curl slot's coefficient tables come from. Transcribed
#: from the shipped launchers' own docstrings — ``step_curl_kernels.py:791-804``
#: (step_B reads the HALF-INTEGER views) and :807-818 (step_D the integer ones) —
#: and it is the same split ``parity/meep_gpu/gate_cuda_folded_curl.py:219-230``
#: carries. Swapped, it is a half-cell error in the absorber profile: converged,
#: smooth and wrong, which is why it is written down beside a citation rather than
#: inferred from the slot name.
CURL_SLOT_SUB_LATTICE: Dict[str, bool] = {"step_B": True, "step_D": False}


class CudaSlotPlan:
    """What one arm would launch in one slot, and how the launch args resolve.

    THIS IS A PLAN, NOT A LAUNCH, and the split is the point of the object.
    ``resolve_launch_args`` is a closure that imports the kernel module and builds
    the device tables; it is NOT called when the plan is built, because that import
    needs CuPy and the whole census — the thing that establishes what this table
    covers — runs backend-free (``measure_predicate_coverage.py:127`` and :201 both
    lift with ``prefer_gpu=False``). A composer that resolved args at plan time
    could not be evaluated on the host that measured its own coverage.

    ``launchable`` says whether an established LAUNCH exists for this family: an
    argument resolution AND a launcher, two ends of one signature. Two families had
    both from 2026-09-17, lifted verbatim from the harnesses that already drive their
    kernels and from the certified wrappers those harnesses call. Four more have both
    since 2026-09-27 -- the three off-diagonal ``update_E`` families and the mirror
    fill -- and each of those resolutions is the SAME derivation the family's own
    wrapper performs when it is handed the layer (the ``pml=`` door), computed once
    per plan and passed through the wrapper's keyword door, so the launcher and the
    predicate still cannot disagree. The other families ship certified kernels whose
    argument resolution lives only in their own gate, and this table says so rather
    than guessing at it. A guessed resolution is a half-cell error away from
    "converged, smooth and wrong", which is exactly the class of defect this track
    spends its gates on.

    ``run()`` IS THE LAUNCH, since 2026-09-17, for the families that have one:
    it resolves the arguments for this slot and hands them to the family's launcher,
    exactly as ``fused_pairs.CudaFusedPairPlan.run`` does for a product, so the
    merge can adopt a certified single SEAM (``fastpath_cuda.SINGLE_ARM_SEAMS``) and
    a real-PML row's veto leg can be served by this table's own kernels. A plan
    without an established launch still RAISES here by name rather than guessing,
    and ``launchable`` is what keeps such a plan out of the dispatch set before any
    consult reaches it.

    ``kernel_label`` IS A LABEL AND NOT A DEVICE ENTRY POINT, and the distinction is
    written into the name because getting it wrong is silent. For 45 of the 51 arms
    it is the census record's own ``kernel=`` string for that family, quoted from
    ``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closed/
    predicate_battery.py``, so a plan can be read beside the coverage report that
    justifies it. Six diverge, each deliberately and each with its own source:

    * the certified curl pair. The closed census stores its verdict under
      ``kernel="covers_real_pml_curl"`` (that battery at :1276), i.e. the PREDICATE
      name, because one predicate serves both sub-steps. The per-sub-step labels used
      here, ``fused_step_B_pml_real`` / ``fused_step_D_pml_real``, come from the
      earlier census that named them individually
      (``results/cuda_predicate_coverage_2026-08-15_hardened/predicate_battery.py:99``)
      and from the PTX assessment's own artifact names
      (``results/cuda_ptx_assessment_2026-08-14/ptx/cuda_fused_step_B_pml_real__*.ptx``).
      Both are wrappers: the declared entry points are ``step_B_pml_real`` /
      ``step_D_pml_real`` at ``step_curl_kernels.py:552,630``.
    * the two off-diagonal complex arms, which carry the module's own
      ``KERNEL_NAMES`` (``complex_offdiag_update_e.py:331-334``) rather than the
      census's ``fused_``-prefixed wrapper name.
    * the two mirror-fill arms (2026-09-27), ``fill_symmetry_B+fill_folded_far_B``
      and ``fill_symmetry_D+fill_folded_far_D``. These are NOT census kernel
      strings: the fill slots sit outside the 759-slot census denominator, so no
      census record names them. Each label joins the two in-seam entry points the
      slot's two driver passes launch (``in_seam_passes``), and
      :class:`_MirrorFillLauncher` names the same pair as its entry points.

    Several families emit their names through ``complex_emitter``'s ``__NAME__``
    substitution, so no literal exists to quote at all. A dispatch must therefore
    resolve its entry point through the family's own launcher and never through this
    string.
    """

    __slots__ = ("family", "slot", "label", "kernel_label", "resolve_launch_args",
                 "launch_kernel", "_context", "_arguments", "launches", "last_launch")

    def __init__(self, family: str, slot: str, label: str, kernel_label: str,
                 resolve_launch_args: Optional[Callable[..., Dict[str, Any]]] = None,
                 *, launch: Optional[Any] = None,
                 context: Optional["StepContext"] = None) -> None:
        self.family = family
        self.slot = slot
        self.label = label
        self.kernel_label = kernel_label
        self.resolve_launch_args = resolve_launch_args
        self.launch_kernel = launch
        self._context = context
        self._arguments: Optional[Dict[str, Any]] = None
        self.launches = 0
        self.last_launch: Optional[Dict[str, Any]] = None

    @property
    def launchable(self) -> bool:
        """Is there an established launch -- resolution AND launcher -- for this slot?"""
        return self.resolve_launch_args is not None and self.launch_kernel is not None

    def run(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """The single launch. NEEDS A DEVICE, and nothing in a plan build calls it.

        The resolver is called as ``(context, slot)`` -- both resolvers in this
        module take the slot, because one family serves two sub-steps and the tables
        differ by sub-lattice -- where a product's resolver takes the context alone.

        RESOLVED ONCE PER PLAN, not per launch. Both flatteners document the contract
        ("called ONCE per frozen configuration, not per launch",
        ``step_curl_kernels.real_pml_curl_tables``, ``constitutive_kernels.
        real_constitutive_tables``): the tables are views of the PML's own device
        arrays and the codes and ``dtdx`` are read off the frozen grid, none of which
        moves while this plan is the frozen one, and the plan is rebuilt at every
        re-freeze. Resolving on every consult would put six view constructions and
        their guards inside the step for each of four launches -- host time a launch
        witness cannot see, on the leg whose whole point is to be the singles
        baseline a fused product is timed against.
        """
        if (self.resolve_launch_args is None or self.launch_kernel is None
                or self._context is None):
            raise RuntimeError(
                f"{self.family}/{self.slot} has no established launch (a resolver, "
                "a launcher and a composition context are all required); a guessed "
                "one is a half-cell error away from converged, smooth and wrong")
        if self._arguments is None:
            self._arguments = self.resolve_launch_args(self._context, self.slot)
        result = self.launch_kernel(self._context.fields, self.slot, self._arguments)
        self.launches += 1
        if isinstance(result, dict):
            self.last_launch = result
        return result

    @property
    def launch_grid(self) -> Optional[Tuple[int, ...]]:
        """The block count the LAST launch asked for, read off the launcher's report.

        Same contract as ``CudaFusedPairPlan.launch_grid``: ``None`` before the first
        launch means UNKNOWN, and the dispatch seam reports it as
        ``programs_per_dispatch``.
        """
        blocks = (self.last_launch or {}).get("blocks")
        try:
            return (int(blocks),)
        except (TypeError, ValueError):
            return None

    def warm(self) -> Optional[str]:
        """Compile this slot's kernel at plan time, launching nothing.

        Compiles BY ENTRY POINT through the launcher's own ``warm`` -- never through
        ``kernel_label``, which the class docstring says is a label and not a device
        entry point. Returns a REASON when there is no launcher to compile through,
        which the warm pass records as unwarmed rather than unfilling the slot.

        A LAUNCHER WHOSE SOURCE DEPENDS ON THE FROZEN CONFIGURATION says so with
        ``WARMS_FROM_CONTEXT``, and is handed this plan's context as well as the
        slot. The off-diagonal families emit one source per live row mask (and the
        dispersive one per pole arity), both read off the ``Fields`` the plan was
        built for, so "compile this slot's kernel" has no answer from the slot alone.
        The flag rather than a second positional argument for every launcher keeps
        the ``warm(slot)`` contract the two 2026-09-17 launchers were written to.
        """
        warm = getattr(self.launch_kernel, "warm", None)
        if not callable(warm):
            return (f"{self.family} declares no launcher warm; the kernel compiles "
                    "at its first launch instead")
        if getattr(self.launch_kernel, "WARMS_FROM_CONTEXT", False):
            if self._context is None:
                return (f"{self.family} compiles per frozen configuration and this "
                        "plan carries no composition context to read it from")
            return warm(self.slot, self._context) or None
        return warm(self.slot) or None

    def __repr__(self) -> str:
        return (f"CudaSlotPlan({self.family}/{self.slot}/{self.kernel_label}, "
                f"launchable={self.launchable})")


class CudaMirrorFillPlan(CudaSlotPlan):
    """The fill slot's plan: TWO driver passes behind one slot, run as two.

    ``fill_B`` stands for ``fill_symmetry_bc_B`` AND ``fill_folded_far_ghosts_B``, and
    the driver runs ``zero_metal_B`` BETWEEN them (``fastpath.FAR_FILL_PASSES``). The
    certified in-seam kernels are one pass each -- ``in_seam_passes`` states that "THE
    THREE PASSES ARE THREE LAUNCHES AND MUST NOT BE MERGED", because a near fill on a
    mirrored axis and the wall wipe on a metallic one share an edge -- so this plan
    exposes the split the dispatch seam reads off the PLAN rather than off the label:
    :meth:`run_near` is the near pass, consulted at the fill slot, and :meth:`run_far`
    is the far pass, consulted by its own driver name after the wall wipe has run on
    the array path. Nothing is reordered and nothing is fused: each half is the pass
    its gate certified, at the position the array path runs it.

    ``launch_grid`` IS THE NEAR PASS'S, and only the near pass's. The near pass
    launches at least once on every grid this arm admits (the predicate refuses an
    unfolded one), while the far pass legitimately launches NOTHING on a folded
    METALLIC axis; a block count of zero there would read as "a kernel launched over
    an empty grid" to every non-vacuity clause that reads ``programs_per_dispatch``.
    ``launches`` counts PASS CALLS, near and far alike, so it agrees with the seam's
    own ``dispatches`` bookkeeping, which books both consults of a split fill.
    """

    __slots__ = ("last_far",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_far: Optional[Dict[str, Any]] = None

    def _half(self, half: str) -> Dict[str, Any]:
        if (self.resolve_launch_args is None or self.launch_kernel is None
                or self._context is None):
            raise RuntimeError(
                f"{self.family}/{self.slot} has no established launch (a resolver, "
                "a launcher and a composition context are all required)")
        if self._arguments is None:
            self._arguments = self.resolve_launch_args(self._context, self.slot)
        result = self.launch_kernel.run_half(self._context.fields, self.slot,
                                             self._arguments, half)
        self.launches += 1
        if half == "near":
            self.last_launch = result
        else:
            self.last_far = result
        return result

    def run_near(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """``stepping.fill_symmetry_bc_*``: every folded axis's near plane, X, Y, Z."""
        return self._half("near")

    def run_far(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """``stepping.fill_folded_far_ghosts_*``: every folded PERIODIC axis's far
        plane. The driver runs ``zero_metal_*`` between this and :meth:`run_near`."""
        return self._half("far")

    def run(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """Both passes, near then far -- the array path's order.

        Correct in isolation only. A composed plan is consulted at the two driver
        sites instead, because the driver puts ``zero_metal_*`` between them; the
        dispatch seam branches on ``run_near`` and never reaches this method for a
        split plan.
        """
        return {"near": self.run_near(), "far": self.run_far()}


class _RealPmlCurlLauncher:
    """The certified real-PML curl wrappers behind one ``(fields, slot, arguments)``.

    ``step_curl_kernels._step_B_fused_pml_real`` / ``_step_D_fused_pml_real`` take
    exactly the keys :func:`resolve_curl_launch_args` returns, so the resolution and
    the launcher are two ends of one signature; the module imports CuPy at scope,
    hence the function-local imports. The block count is the wrapper's own
    arithmetic (``_step_fused_pml_real``) repeated on the slot's target, so the
    seam's ``programs_per_dispatch`` reads the grid the launch was given.
    """

    _WRAPPERS = {"step_B": "_step_B_fused_pml_real", "step_D": "_step_D_fused_pml_real"}
    _ENTRY_POINTS = {"step_B": "step_B_pml_real", "step_D": "step_D_pml_real"}
    _TARGETS = {"step_B": "Bx", "step_D": "Dx"}

    def __call__(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 ) -> Dict[str, Any]:
        from . import step_curl_kernels  # noqa: PLC0415 - imports cupy at scope

        getattr(step_curl_kernels, self._WRAPPERS[slot])(fields, **arguments)
        threads = step_curl_kernels._REAL_PML_THREADS  # noqa: SLF001
        count = int(getattr(fields, self._TARGETS[slot]).size)
        return {"launched": True, "blocks": (count + threads - 1) // threads}

    def warm(self, slot: str) -> Optional[str]:
        from . import step_curl_kernels  # noqa: PLC0415 - imports cupy at scope

        step_curl_kernels._get_kernel(self._ENTRY_POINTS[slot], False).compile()  # noqa: SLF001
        return None


class _RealPmlConstitutiveLauncher:
    """The certified real-PML constitutive wrappers, the twin of the curl launcher."""

    _WRAPPERS = {"update_H": "_update_H_fused_pml_real",
                 "update_E": "_update_E_fused_pml_real"}
    _ENTRY_POINTS = {"update_H": "update_H_pml_real", "update_E": "update_E_pml_real"}
    _TARGETS = {"update_H": "Hx", "update_E": "Ex"}

    def __call__(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 ) -> Dict[str, Any]:
        from . import constitutive_kernels  # noqa: PLC0415 - imports cupy at scope

        getattr(constitutive_kernels, self._WRAPPERS[slot])(fields, **arguments)
        threads = constitutive_kernels._CONSTITUTIVE_THREADS  # noqa: SLF001
        count = int(getattr(fields, self._TARGETS[slot]).size)
        return {"launched": True, "blocks": (count + threads - 1) // threads}

    def warm(self, slot: str) -> Optional[str]:
        from . import constitutive_kernels  # noqa: PLC0415 - imports cupy at scope

        constitutive_kernels._get_kernel(self._ENTRY_POINTS[slot]).compile()  # noqa: SLF001
        return None


#: The two launchers, one per resolver; ``registry._LAUNCHERS`` keys them by the same
#: ``"resolve"`` token the table rows carry.
launch_real_pml_curl = _RealPmlCurlLauncher()
launch_real_pml_constitutive = _RealPmlConstitutiveLauncher()


def resolve_curl_launch_args(context: StepContext, slot: str) -> Dict[str, Any]:
    """``(tables, boundary_codes, dtdx)`` for the real-field PML curl pair.

    LIFTED VERBATIM from the two harnesses that already drive these kernels —
    ``parity/meep_gpu/stress_cuda_end_to_end.py:766-784`` and
    ``parity/meep_gpu/stress_cuda_scale.py:230-241`` — with the harnesses' own
    reason for each call kept:

    * the codes come from the SHIPPED resolver, which is the copy
      ``covers_real_pml_curl`` itself consults, so a code split that drifted between
      predicate and launcher cannot hide here (``gate_cuda_folded_curl.py:567-580``);
    * the tables come from the SHIPPED flattener with the sub-lattice taken from
      :data:`CURL_SLOT_SUB_LATTICE` rather than re-derived;
    * ``dtdx`` is ``grid.dt / grid.dx`` as a python float, which is what the kernel
      takes.

    RAISES on a host with no CuPy, and on a grid the predicate refuses. Both are
    correct: callers gate on the predicate first, and ``_select_slot`` turns a
    raising builder into a NAMED refusal for the slot rather than an exception out
    of ``plan_step``.
    """
    from .step_curl_kernels import (  # noqa: PLC0415 - a NumPy host must be able to plan
        real_curl_boundary_codes,
        real_pml_curl_tables,
    )

    return {
        "tables": real_pml_curl_tables(context.pml, CURL_SLOT_SUB_LATTICE[slot]),
        "boundary_codes": real_curl_boundary_codes(context.grid),
        "dtdx": float(context.grid.dt / context.grid.dx),
    }


def resolve_constitutive_launch_args(context: StepContext,
                                     slot: str) -> Dict[str, Any]:
    """``(tables,)`` for the real-field PML constitutive pair.

    LIFTED VERBATIM from ``parity/meep_gpu/stress_cuda_end_to_end.py:787-791`` and
    ``parity/meep_gpu/stress_cuda_scale.py:243-244``, through the shipped
    ``constitutive_tables_for``, which asks ``coverage.constitutive_sub_lattice`` —
    the same function the predicate asks — so the two cannot disagree about which
    sub-lattice a side reads (``constitutive_kernels.py:444-462``). Binding the
    other one is a half-cell error in the absorber, not a crash.
    """
    from .constitutive_kernels import (  # noqa: PLC0415 - see the curl resolver
        constitutive_tables_for,
    )

    return {"tables": constitutive_tables_for(CONSTITUTIVE_SLOT_SIDES[slot],
                                              context.pml)}


# ---------------------------------------------------------------------------
# The three off-diagonal update_E singles (2026-09-27)
# ---------------------------------------------------------------------------
#
# ONE WRAPPER SHAPE, THREE MODULES. Each family ships a wrapper
# ``update_E_*(fields, pml=None, *, tables, codes, walls[, weights])`` that derives
# every keyword from the layer and the grid when it is handed ``pml`` -- "the
# launcher and the predicate cannot disagree" is enforced there, in the family
# module -- and takes all of them through the keyword door otherwise. The resolvers
# below call the SAME derivation functions the ``pml=`` branch calls, once per
# frozen configuration, and the launchers pass the answers through the keyword
# door: a launch is the wrapper's own ``pml=`` launch without six table views and
# their guards rebuilt inside every step. What each wrapper still reads per launch,
# it keeps reading per launch: the live row mask and its volumes, and for the
# dispersive family the pole chain, whose buffers rotate every step.
#
# CERTIFIED BY THE FAMILY'S OWN BYTE GATE, which drove these wrappers through the
# same derivations: ``offdiag_2026-08-16``, ``cuda_folded_offdiag_2026-08-21`` and
# ``cuda_dispersive_offdiag_2026-08-20``, all three re-gated in Round B
# (``certification.json`` ``cuda_regate_2026-09-25_roundb``). The seam they are
# adopted into is ``step_D`` 'PML' + ``update_E`` (``fastpath_cuda.SINGLE_ARM_SEAMS``).

class _OffdiagUpdateELauncher:
    """One certified off-diagonal ``update_E`` wrapper behind ``(fields, slot, args)``.

    The block count is the wrapper's own arithmetic -- one lane per cell of ``Ex``
    over the module's thread count -- repeated here so the seam's
    ``programs_per_dispatch`` reads the grid the launch was given.

    ``WARMS_FROM_CONTEXT``: the emitted source is specialised on the live row mask
    (and, for the dispersive family, the pole arity), which are properties of the
    ``Fields`` a plan was built for rather than of the slot, so the warm pass hands
    this launcher the plan's context (``CudaSlotPlan.warm``).
    """

    WARMS_FROM_CONTEXT = True
    _MODULE = ""
    _WRAPPER = ""
    _THREADS = ""
    _ENTRY_POINTS: Dict[str, str] = {}

    def _module(self) -> Any:
        from importlib import import_module  # noqa: PLC0415 - may import cupy at scope

        return import_module(f".{self._MODULE}", __package__)

    def _kernel_key(self, module: Any, fields: Any) -> Tuple[Any, ...]:
        raise NotImplementedError

    def __call__(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 ) -> Dict[str, Any]:
        module = self._module()
        result = getattr(module, self._WRAPPER)(fields, **arguments)
        threads = int(getattr(module, self._THREADS))
        count = int(fields.Ex.size)
        report: Dict[str, Any] = {"launched": True,
                                  "blocks": (count + threads - 1) // threads}
        if isinstance(result, dict):
            report["wrapper"] = dict(result)
        return report

    def warm(self, slot: str, context: StepContext) -> Optional[str]:
        module = self._module()
        module._get_kernel(*self._kernel_key(module, context.fields)).compile()  # noqa: SLF001
        return None


class _OffdiagLauncher(_OffdiagUpdateELauncher):
    """``offdiag_constitutive_kernels.update_E_offdiag_fused_pml_real`` (unfolded)."""

    _MODULE = "offdiag_constitutive_kernels"
    _WRAPPER = "update_E_offdiag_fused_pml_real"
    _THREADS = "_OFFDIAG_THREADS"
    _ENTRY_POINTS = {"update_E": "update_E_pml_real_offdiag"}

    def _kernel_key(self, module: Any, fields: Any) -> Tuple[Any, ...]:
        return (module.normalized_row_mask(module.offdiag_row_mask(fields)),)


class _FoldedOffdiagLauncher(_OffdiagUpdateELauncher):
    """``folded_offdiag_kernels.update_E_folded_offdiag_fused_pml_real``."""

    _MODULE = "folded_offdiag_kernels"
    _WRAPPER = "update_E_folded_offdiag_fused_pml_real"
    _THREADS = "_FOLDED_OFFDIAG_THREADS"
    _ENTRY_POINTS = {"update_E": "update_E_pml_real_folded_offdiag"}

    def _kernel_key(self, module: Any, fields: Any) -> Tuple[Any, ...]:
        return (module.normalized_row_mask(
            module._coverage.offdiag_row_mask(fields)),)  # noqa: SLF001


class _DispersiveOffdiagLauncher(_OffdiagUpdateELauncher):
    """``dispersive_offdiag_update_e.update_E_dispersive_offdiag_fused_pml_real``.

    THE POLE CHAIN IS NEVER AN ARGUMENT HERE. The wrapper resolves it on every call
    when ``plan`` is left at ``None``, because ``PolarizationState.update`` rotates
    the three buffers each step and a chain resolved at plan time would name last
    step's arrays -- stale in a way that still computes. A resolution that carried a
    ``plan`` key would be exactly that defect, so it is refused by name.
    """

    _MODULE = "dispersive_offdiag_update_e"
    _WRAPPER = "update_E_dispersive_offdiag_fused_pml_real"
    _THREADS = "_DISPERSIVE_OFFDIAG_THREADS"
    _ENTRY_POINTS = {"update_E": "update_E_pml_real_folded_offdiag_dispersive"}

    def __call__(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 ) -> Dict[str, Any]:
        if "plan" in arguments:
            raise ValueError(
                "the dispersive off-diagonal pole chain is resolved per launch by the "
                "wrapper; a resolved 'plan' argument would name last step's rotated "
                "buffers")
        return super().__call__(fields, slot, arguments)

    def _kernel_key(self, module: Any, fields: Any) -> Tuple[Any, ...]:
        mask = module.normalized_row_mask(
            module._coverage.offdiag_row_mask(fields))  # noqa: SLF001
        counts = module.normalized_pole_counts(
            module.pole_counts_of(module.resolve_pole_plan(fields)))
        return (mask, counts)


launch_offdiag = _OffdiagLauncher()
launch_folded_offdiag = _FoldedOffdiagLauncher()
launch_dispersive_offdiag = _DispersiveOffdiagLauncher()


def resolve_offdiag_launch_args(context: StepContext, slot: str) -> Dict[str, Any]:
    """``(tables, codes, walls)`` for the unfolded off-diagonal ``update_E``.

    The three calls ``update_E_offdiag_fused_pml_real`` makes when handed ``pml``
    (``offdiag_constitutive_kernels.py``, its ``if pml is not None`` branch), on the
    same objects: the layer, and the grid of the ``Fields`` being launched on.
    """
    from .offdiag_constitutive_kernels import (  # noqa: PLC0415 - imports cupy at scope
        offdiag_boundary_codes,
        offdiag_constitutive_tables,
        offdiag_wall_mask_flags,
    )

    grid = context.fields.grid
    return {"tables": offdiag_constitutive_tables(context.pml),
            "codes": offdiag_boundary_codes(grid, context.pml),
            "walls": offdiag_wall_mask_flags(grid)}


def resolve_folded_offdiag_launch_args(context: StepContext,
                                       slot: str) -> Dict[str, Any]:
    """``(tables, codes, walls, weights)`` for the folded off-diagonal ``update_E``.

    The four calls ``update_E_folded_offdiag_fused_pml_real`` makes when handed
    ``pml``: the half-integer tables, the codes with ``mirror`` in the table, the
    wall flags off the grid's declaration, and the ghost weights through
    ``fields.mirror_parity``.
    """
    from .coverage import offdiag_wall_mask_flags  # noqa: PLC0415
    from .folded_offdiag_kernels import (  # noqa: PLC0415
        folded_offdiag_boundary_codes,
        folded_offdiag_constitutive_tables,
        mirror_ghost_weights,
    )

    grid = context.fields.grid
    return {"tables": folded_offdiag_constitutive_tables(context.pml),
            "codes": folded_offdiag_boundary_codes(grid, context.pml),
            "walls": offdiag_wall_mask_flags(grid),
            "weights": mirror_ghost_weights(grid)}


def resolve_dispersive_offdiag_launch_args(context: StepContext,
                                           slot: str) -> Dict[str, Any]:
    """``(tables, codes, walls, weights)`` for the dispersive off-diagonal ``update_E``.

    The four calls ``update_E_dispersive_offdiag_fused_pml_real`` makes when handed
    ``pml``. The pole chain is deliberately NOT resolved here; see
    :class:`_DispersiveOffdiagLauncher`.
    """
    from .coverage import offdiag_wall_mask_flags  # noqa: PLC0415
    from .dispersive_offdiag_update_e import (  # noqa: PLC0415
        dispersive_offdiag_boundary_codes,
        dispersive_offdiag_constitutive_tables,
        mirror_ghost_weights,
    )

    grid = context.fields.grid
    return {"tables": dispersive_offdiag_constitutive_tables(context.pml),
            "codes": dispersive_offdiag_boundary_codes(grid, context.pml),
            "walls": offdiag_wall_mask_flags(grid),
            "weights": mirror_ghost_weights(grid)}


# ---------------------------------------------------------------------------
# The mirror fill (2026-09-27)
# ---------------------------------------------------------------------------
#
# THE SIX CERTIFIED IN-SEAM KERNELS, SERVED AS A FILL ARM. ``in_seam_passes`` ships
# ``fill_symmetry_{B,D}``, ``fill_folded_far_{B,D}`` and ``zero_metal_{B,D}``, each
# gated on its own against the ``stepping`` function it transcribes
# (``certification.json`` ``in_seam_2026-08-21``, re-gated in Round B). The fill
# slot runs the first two; ``zero_metal_*`` stays on the array path, where the driver
# runs it unconditionally BETWEEN them and where ``deposit_repair`` relies on it.

#: The mirror fill's refusal of an UNFOLDED grid, the head of the sentence
#: :func:`covers_mirror_fill` returns there. Named so a reader that classifies
#: refusals (``parity/meep_gpu/cuda_predicate_battery._refusal_kind``) compares
#: against the shipped words rather than a transcription of them: on an unfolded row
#: the fill slot has nothing to serve, which is not the same event as a fold no arm
#: admits.
MIRROR_FILL_UNFOLDED_REFUSAL = "the grid is not mirror-folded"


def covers_mirror_fill(fields: Any, grid: Any, family: str) -> Tuple[bool, str]:
    """May the certified in-seam fill kernels serve this family's fill slot?

    THE CONJUNCTION OF THE TWO PASS PREDICATES, because the slot runs both passes:
    ``in_seam_coverage.covers_fill_symmetry`` (real float32 C-contiguous storage, no
    cylindrical radial axis, every folded axis carrying a declared +/-1 phase) and
    ``in_seam_coverage.covers_fill_folded_far`` (the same, plus the fold's two
    termination readings agreeing and the far image row in bounds). Their reasons are
    reported deduplicated, since the second repeats the first's.

    AN UNFOLDED GRID IS REFUSED BY NAME, after those two, and the order is the point.
    Both pass predicates admit an unfolded Cartesian grid vacuously -- there is no
    folded axis to refuse -- and an arm that took the slot there would install a plan
    that launches nothing, on every unfolded row, in front of whatever the other
    table composes. The two pass predicates run FIRST so a grid they refuse (the
    cylindrical radial axis, complex storage) is refused for their reason rather
    than reported as "not folded".
    """
    from . import in_seam_coverage as _in_seam  # noqa: PLC0415 - CuPy-free sibling

    reasons: List[str] = []
    for verdict in (_in_seam.covers_fill_symmetry(fields, grid, family),
                    _in_seam.covers_fill_folded_far(fields, grid, family)):
        covered, reason = verdict
        if covered:
            continue
        for part in str(reason).split("; "):
            if part and part not in reasons:
                reasons.append(part)
    if reasons:
        return False, "; ".join(reasons)
    try:
        phases = _in_seam.mirror_fill_phases(grid)
    except Exception as exc:  # noqa: BLE001 - an unreadable fold is a refusal
        return False, (f"the grid could not state its mirror phases: "
                       f"{type(exc).__name__}: {exc}")
    if all(phase is None for phase in phases):
        return False, (f"{MIRROR_FILL_UNFOLDED_REFUSAL}: fill_symmetry_bc_{family} and "
                       f"fill_folded_far_ghosts_{family} touch no array here, so there "
                       "is nothing for a fill kernel to serve")
    return True, ("covered: every folded axis carries a declared +/-1 phase, its "
                  "termination agrees on both routes and its far image row is in "
                  "bounds, on real float32 storage")


class _MirrorFillLauncher:
    """The certified in-seam fill kernels behind the fill slot's two halves.

    ``in_seam_passes.run_pass`` walks a launch list in the array path's X, Y, Z order,
    one launch per folded axis, and returns each launch's geometry. The list is the
    resolver's (:func:`resolve_fill_launch_args`), read from
    ``in_seam_coverage.plan`` -- the function ``run_pass`` itself calls when handed a
    grid -- and handed through ``run_pass``'s ``launches`` argument, so it is derived
    once per frozen configuration rather than on every consult. The module imports
    CuPy at scope, hence the function-local imports.

    ``_ENTRY_POINTS`` names a PAIR per slot, because the slot is two passes.
    """

    _PASSES = {"near": "fill_symmetry", "far": "fill_folded_far"}
    _ENTRY_POINTS = {"fill_B": ("fill_symmetry_B", "fill_folded_far_B"),
                     "fill_D": ("fill_symmetry_D", "fill_folded_far_D")}

    def run_half(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 half: str) -> Dict[str, Any]:
        from . import in_seam_passes  # noqa: PLC0415 - imports cupy at scope

        launched = in_seam_passes.run_pass(self._PASSES[half], fields,
                                           arguments["family"],
                                           launches=arguments[half])
        return {"launched": any(entry.get("launched") for entry in launched),
                "pass": self._PASSES[half],
                "launches": len(launched),
                "blocks": sum(int(entry.get("blocks") or 0) for entry in launched
                              if entry.get("launched"))}

    def __call__(self, fields: Any, slot: str, arguments: Dict[str, Any],
                 ) -> Dict[str, Any]:
        near = self.run_half(fields, slot, arguments, "near")
        far = self.run_half(fields, slot, arguments, "far")
        return {"launched": bool(near["launched"] or far["launched"]),
                "blocks": near["blocks"], "near": near, "far": far}

    def warm(self, slot: str) -> Optional[str]:
        from . import in_seam_passes  # noqa: PLC0415 - imports cupy at scope

        for name in self._ENTRY_POINTS[slot]:
            in_seam_passes._get_kernel(name).compile()  # noqa: SLF001
        return None


launch_mirror_fill = _MirrorFillLauncher()


def resolve_fill_launch_args(context: StepContext, slot: str) -> Dict[str, Any]:
    """The family, and the near and far launch lists, for one fill slot.

    ``in_seam_coverage.plan`` is the one derivation of which axes each pass visits,
    what parity each carries and which row the far ghost images; the predicate's
    clauses are read from the same functions, so the launch list and the admission
    cannot disagree. Read off ``context.grid``, the grid the predicate was asked with.
    """
    from .in_seam_coverage import plan as in_seam_plan  # noqa: PLC0415 - CuPy-free

    return {"family": FILL_SLOT_FAMILIES[slot],
            "near": tuple(in_seam_plan("fill_symmetry", context.grid)),
            "far": tuple(in_seam_plan("fill_folded_far", context.grid))}


def plan_factory(family: str, label: str, kernel_label: str,
                 resolve: Optional[Callable[..., Dict[str, Any]]] = None,
                 launch: Optional[Any] = None,
                 plan_class: type = CudaSlotPlan) -> Callable[..., CudaSlotPlan]:
    """A ``plan(context, slot)`` closure for one family, for every slot it claims.

    The context is bound into the plan so ``run()`` can resolve its arguments at
    consult time; a family with no launcher gets a plan that is not launchable.
    ``plan_class`` is :class:`CudaSlotPlan` for every slot arm and
    :class:`CudaMirrorFillPlan` for the fill, whose slot is two driver passes.
    """

    def plan(context: StepContext, slot: str) -> CudaSlotPlan:
        return plan_class(family, slot, label, kernel_label, resolve,
                          launch=launch, context=context)

    return plan


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class ArmSpec:
    """One family's claim on one slot.

    ``coverage(context, slot)`` and ``plan(context, slot)`` take the slot as well as
    the context, so ONE pair of callables serves every slot a family claims — the
    curl families register the same two on ``step_B`` and ``step_D`` and let the
    slot->args mapping do the rest.

    ``gate(context)`` is the cheap precondition deciding whether the arm is
    CONSULTED. It is separate from ``coverage`` because a gated-out arm contributes
    NO reason, while a consulted-and-refusing arm contributes all of its reasons by
    name. NO ARM IN THIS TABLE CARRIES A GATE TODAY: every shipped CUDA predicate
    enumerates its own refusals positively — that is what makes the census
    attribution readable — so gating one would replace a named reason with silence.
    The parameter stays because the state it names is real on the sibling tracks.

    ``wired`` is not bookkeeping. A family may be built and byte-certified while its
    COMPOSITION is deliberately deferred; it registers ``wired=False`` and
    :func:`arms_for` skips it, so :func:`plan_step` cannot select it — while it stays
    ENUMERABLE, which is what lets a disjointness sweep ask every registered
    predicate and MEASURE that no unwired family co-admits a configuration a wired
    one admits. A family simply absent from the table could not be swept at all.

    ``replaces`` is the driver passes the arm covers, which is its own slot unless
    it says otherwise. It is what separates a WELD from a peer: a fused product
    claims one slot while replacing a whole seam, so its predicate is a conjunction
    containing the wired arm's and it ALWAYS co-admits with it. No CUDA arm is a
    weld today; the distinction is carried so that a sweep asserting "at most one
    arm admits" over the raw table does not silently become an assertion no fused
    product could ever satisfy.
    """

    __slots__ = ("family", "slot", "label", "gate", "coverage", "plan", "prefix",
                 "noun", "wired", "replaces", "shape", "predicate_name")

    def __init__(self, family: str, slot: str, label: str,
                 coverage: Callable[..., Any], plan: Callable[..., Any],
                 prefix: Optional[str] = None, noun: Optional[str] = None,
                 gate: Optional[Callable[[Any], bool]] = None,
                 wired: bool = True,
                 replaces: Optional[Tuple[str, ...]] = None,
                 shape: Optional[str] = None,
                 predicate_name: str = "") -> None:
        self.family = family
        self.slot = slot
        self.label = label
        self.coverage = coverage
        self.plan = plan
        self.prefix = f"{label}: " if prefix is None else prefix
        self.noun = label if noun is None else noun
        self.gate = gate
        self.wired = bool(wired)
        self.replaces = (slot,) if replaces is None else tuple(replaces)
        self.shape = shape
        self.predicate_name = predicate_name

    @property
    def is_weld(self) -> bool:
        """Does this arm REPLACE more driver passes than the one slot it sits on?"""
        return self.replaces != (self.slot,)

    def bind(self, context: Any) -> _Arm:
        """This spec against one composition context, as the shared ``_Arm``."""
        return _Arm(self.label,
                    True if self.gate is None else bool(self.gate(context)),
                    lambda: self.coverage(context, self.slot),
                    lambda: self.plan(context, self.slot),
                    self.prefix,
                    self.noun)

    def __repr__(self) -> str:
        return (f"ArmSpec({self.family}/{self.slot}/{self.label}, "
                f"wired={self.wired})")


#: slot -> the arms registered for it, in registration order. Order decides NOTHING
#: about selection — ``_select_slot`` refuses rather than picking by table order —
#: it only decides the order refusals are reported in.
_REGISTRY: Dict[str, List[ArmSpec]] = {}


def register(family: str, slot: str, label: str, coverage: Callable[..., Any],
             plan: Callable[..., Any], prefix: Optional[str] = None,
             noun: Optional[str] = None,
             gate: Optional[Callable[[Any], bool]] = None,
             wired: bool = True,
             replaces: Optional[Tuple[str, ...]] = None,
             shape: Optional[str] = None,
             predicate_name: str = "") -> ArmSpec:
    """Add one arm to the table. Re-registering a (family, label) on a slot is refused.

    A duplicate is refused rather than replaced: the second registration is almost
    always a module reached under two names, and silently keeping one of the two
    would make WHICH KERNEL RUNS depend on import order. Refusing means the tree
    fails at import, loudly, in one place.
    """
    spec = ArmSpec(family, slot, label, coverage, plan, prefix, noun, gate, wired,
                   replaces, shape, predicate_name)
    existing_arms = _REGISTRY.setdefault(slot, [])
    for existing in existing_arms:
        if (existing.family, existing.label) == (family, label):
            raise ValueError(
                f"{family!r} already registered {label!r} on slot {slot!r}; a "
                f"second registration would make the slot self-ambiguous and "
                f"leave it permanently UNSELECTED")
    existing_arms.append(spec)
    return spec


#: Where the table is in its one-time population. ``cold`` -> nothing imported;
#: ``running`` -> :mod:`.registry` is executing; ``done`` -> every family is in
#: :data:`_REGISTRY`.
_REGISTRATION_STATE = "cold"


def ensure_registered() -> None:
    """Populate the table once, so it is COMPLETE before it is read.

    REGISTRATION IS AN IMPORT SIDE EFFECT and that is the hazard this function
    exists to remove. Without it the table's contents depend on which modules the
    caller happened to import — and an arm missing from the table does not produce
    a refusal, it produces a DIFFERENT SELECTION with no ambiguity detected. That is
    not a fail-closed composer; it is one whose ambiguity detection can be switched
    off by import order.

    The import is INSIDE A FUNCTION BODY deliberately: :mod:`.registry` imports this
    module, so a module-scope import here would be a cycle.

    A reader that queries the table WHILE it is being populated is refused rather
    than served a half-populated answer — that read would be correct only by
    accident of registration order, which is the same defect one level down.
    """
    global _REGISTRATION_STATE

    if _REGISTRATION_STATE == "done":
        return
    if _REGISTRATION_STATE == "running":
        raise RuntimeError(
            "the arm table was read while it was still being populated; the answer "
            "would depend on registration order. Read it from a function body that "
            "runs after import instead")
    _REGISTRATION_STATE = "running"
    try:
        from . import registry  # noqa: F401, PLC0415 - import IS the registration

        registry.register_all()
    except BaseException:
        _REGISTRATION_STATE = "cold"
        raise
    _REGISTRATION_STATE = "done"


def registration_state() -> str:
    """``cold`` / ``running`` / ``done`` — for a probe that wants to record it."""
    return _REGISTRATION_STATE


def registered(slot: Optional[str] = None) -> Tuple[ArmSpec, ...]:
    """Every registered arm — WIRED OR NOT — for one slot, or for every slot."""
    ensure_registered()
    if slot is not None:
        return tuple(_REGISTRY.get(slot, ()))
    return tuple(spec for arms in _REGISTRY.values() for spec in arms)


def registered_slots() -> Tuple[str, ...]:
    ensure_registered()
    return tuple(_REGISTRY)


def families_on(slot: str) -> Tuple[str, ...]:
    ensure_registered()
    return tuple(spec.family for spec in _REGISTRY.get(slot, ()))


def arms_for(slot: str, context: Any) -> Tuple[_Arm, ...]:
    """The WIRED arms for one slot, bound to this composition context."""
    ensure_registered()
    return tuple(spec.bind(context) for spec in _REGISTRY.get(slot, ())
                 if spec.wired)


def ambiguity(slot: str) -> Callable[[Sequence[str]], str]:
    """The one home for the co-admission message, phrased per slot KIND.

    The wording is the shared composer's, unchanged, so a probe that recognises an
    ambiguous slot on one track recognises it on all three: two products admitting
    one slot is an over-covering dispatch, and the composition refuses rather than
    picking by table order.
    """
    kind = ("curl" if slot in ("step_B", "step_D")
            else "constitutive" if slot in ("update_H", "update_E")
            else slot)

    def message(labels: Sequence[str]) -> str:
        return (f"two {kind} products admit this configuration "
                f"({', '.join(labels)}); the composition refuses rather than "
                f"picking by table order")

    return message


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

#: Why a ``STEP_ORDER`` slot has no CUDA arm at all. An empty table is not the same
#: event as a table that refused, and a caller testing ``if plan.reasons.get(slot)``
#: must be able to tell them apart, so the absence gets a sentence of its own.
#:
#: EMPTY SINCE 2026-09-27, and kept as the place such a sentence goes. ``fill_B`` /
#: ``fill_D`` were the two entries until then: the in-seam passes take a FAMILY
#: rather than a slot (the seventh positional shape, ``fill_by_family``), they sit
#: outside the 759-slot coverage census, and no composition rule for them had been
#: written. The mirror fill arm (:func:`covers_mirror_fill`) is that rule -- both
#: passes conjoined, an unfolded grid refused by name, the plan split at the wall
#: wipe the driver runs between them -- and the merge adopts it only with its twin
#: (``fastpath_cuda.FILL_ARM_TWINS``). Every slot now has an arm, so a slot reaching
#: :func:`plan_step`'s fallback message is a new slot nobody registered.
NO_ARM_REASONS: Dict[str, str] = {}


class CudaStepPlan:
    """Which sub-steps of one frozen configuration the CUDA table covers.

    ``replaces`` reports the filled slots in the driver's own step order.
    ``selected`` names the ARM that won each one, and it is the only way to tell
    ``PML`` from ``BFAST`` from ``special_kz`` on ``step_B``: several arms build a
    :class:`CudaSlotPlan` of the same class, so a record that reported only the class
    would be reporting an ambiguity as a decision.

    ``launchable`` is the subset of ``replaces`` whose plan carries an established
    launch-argument resolution. It is smaller than ``replaces`` on purpose — see
    :class:`CudaSlotPlan`.
    """

    __slots__ = ("plans", "reasons", "selected")

    def __init__(self, plans: Dict[str, Any], reasons: Dict[str, Tuple[str, ...]],
                 selected: Optional[Dict[str, str]] = None) -> None:
        self.plans = dict(plans)
        self.reasons = dict(reasons)
        self.selected = dict(selected or {})

    @property
    def replaces(self) -> Tuple[str, ...]:
        return tuple(name for name in STEP_ORDER if name in self.plans)

    @property
    def launchable(self) -> Tuple[str, ...]:
        # READ THROUGH THE WRAPPER. After the fusion block runs, two of the four
        # occupants are wrappers -- a NoopPlan / TrailingRepairPlan in the absorbed
        # slot and a LeadingRepairPlan around the launch. None of the three declares
        # `launchable`, so read bare they would each fall back to False and a fused
        # seam would report as "not launchable" the moment it started carrying a
        # deposit. Every wrapper names the plan that does the work in `absorbed_by`,
        # so the declaration is read from THERE; an ordinary arm's plan, which has no
        # `absorbed_by`, answers for itself exactly as before.
        return tuple(name for name in self.replaces
                     if getattr(getattr(self.plans[name], "absorbed_by", None)
                                or self.plans[name], "launchable", False))

    def describe(self) -> str:
        covered = ", ".join(self.replaces) if self.replaces else "nothing"
        return f"CudaStepPlan(replaces=[{covered}])"

    def __repr__(self) -> str:
        return self.describe()


def plan_step(fields: Any, pml: Any, grid: Any = None,
              licenses: Optional[Dict[str, Any]] = None,
              subnormal_policy: Any = None, sources: Any = None,
              fuse: bool = False,
              fuse_labels: Optional[Sequence[str]] = None) -> CudaStepPlan:
    """Ask every registered arm and report the covered subset with its refusals.

    Never raises and never returns ``None``: a configuration nothing covers is a
    plan that replaces nothing, which is the array path, which is the correct
    answer. A predicate that raises, a builder that raises, and a builder that
    returns ``None`` are all NAMED refusals for their slot — the shared
    ``_select_slot``'s contract, not a second copy of it.

    THE TABLE IS COMPLETED BEFORE IT IS READ, by :func:`ensure_registered`, and that
    call is load-bearing rather than defensive: registration is an import side
    effect, and a missing arm changes the SELECTION rather than producing a refusal.

    ``fuse`` IS OPT-IN AND DEFAULTS TO OFF, so the composition this track's coverage
    census measured is what this function returns unless a caller asks for more. With
    it on, :mod:`.fused_pairs` runs AFTER every slot has been decided and may take a
    seam's two slots only where the arm table already gave them to the arms the fused
    kernel implements -- the fusion block is the one place a ``STEP_ORDER`` slot is
    filled without going through the table, and running it first would be a hole
    straight through the ambiguity and builder-refusal clauses.

    ``sources`` IS WHAT MAKES A FUSED SEAM ANSWERABLE. No slot arm reads it; a fused
    pair spans the driver's injection, so its predicate must be told. ``None`` is a
    refusal there, not an empty seam.

    ``fuse_labels`` IS THE OFFER, AND IT IS PER LABEL RATHER THAN PER RUN. ``fuse``
    alone can only say "install whatever admits", and a fused product occupies EVERY
    slot of its seam -- so a product the caller cannot run would take those slots and
    force the caller to refuse the whole step, taking the separate certified arms
    underneath it down as well. ``None`` means "every product", which is what the
    coverage census asks for and what every in-tree caller asked for before the
    dispatch seam existed; a sequence names exactly the products this caller may run
    and the rest are skipped with a NAMED reason under ``fused_pair_<family>``.

    THIS IS REACHED FROM THE DISPATCH SEAM, AND HAS BEEN SINCE PHASE 2.
    ``meep_gpu.fastpath._decide`` composes this table as the SECOND kernel table on a
    CuPy engine -- imported function-locally, below the backend rung -- and
    ``fastpath_cuda.merge_tables`` then adopts whole fused units of it into the
    merged plan, namespacing their labels ``cuda:<label>`` at that boundary. What is
    still true is the direction of the dependency: nothing here knows a dispatcher
    exists, no predicate can see who is asking, and the launch-argument resolvers
    attached to the returned plans are called by the DRIVER's consult rather than by
    anything in this function. Which units may be adopted -- fused products, the
    certified launchable single SEAMS and the certified fill TWINS -- and why any
    other single-arm plan may not be, is stated in
    :func:`meep_gpu.fastpath_cuda.merge_tables`.
    """
    plans: Dict[str, Any] = {}
    reasons: Dict[str, Tuple[str, ...]] = {}
    selected: Dict[str, str] = {}

    ensure_registered()

    context = StepContext(fields, pml, grid, licenses, subnormal_policy, sources)
    for slot in STEP_ORDER:
        table = arms_for(slot, context)
        if table:
            _select_slot(slot, table, ambiguity(slot), plans, reasons, selected)
        else:
            reasons[slot] = (NO_ARM_REASONS.get(
                slot, f"no CUDA arm is registered on {slot}"),)

    if fuse:
        # Imported in the function body, not at module scope: `fused_pairs` imports
        # this module's `StepContext` shape through the objects it is handed, and a
        # module-scope import here would make the composer's import graph a cycle
        # for a block that most callers never ask for.
        from . import fused_pairs  # noqa: PLC0415

        fused_pairs.install_fused_pairs(
            plans, reasons, selected, context,
            offered=fused_pairs.offered_labels(fuse_labels))

    return CudaStepPlan(plans, reasons, selected)

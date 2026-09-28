"""The COMPLEX curl with NO absorber: predicate and plan, and no kernel of its own.

WHAT THIS CLOSES, and it is four slots and not six. The 2026-08-16 group-(J)
round measured a complex, all-periodic, Bloch-phased, off-diagonal, no-absorber
configuration sub-step by sub-step on an RTX A6000
(``parity/meep_gpu/results/group_j_2026-08-16/``):

===============  ==========================  =========================
sub-step         verdict, both row shapes    max differing / compared
===============  ==========================  =========================
``step_B``       IDENTICAL                   0 / 281250  (3-D row)
``step_D``       IDENTICAL                   0 / 281250
``update_E``     DIVERGENT                   93750 / 281250
===============  ==========================  =========================

Three controls diverged on the same fixtures and bindings, which is what makes
the two nulls evidence rather than an untripped wire: PHASEOFF (the Bloch
rotation compiled away) diverged 7350 / 281250 on the wrapped lanes only, and
STALESOURCE (sources bound as a plan-time snapshot) diverged 93750 / 281250 —
the 2026-08-15 group-(G) false-pass mechanism, re-armed and caught.

So the arithmetic at the curls was already right, and what refused was SCOPE.
``update_E`` is a genuine kernel gap (complex storage AND the off-diagonal row
product AND the store arm; no certified body composes those) and stays refused —
this module ships no constitutive predicate at all, and
``test_triton_complex_no_pml_curl.py`` pins the refusal so a later widening
cannot take it by accident.

THIS MODULE SHIPS NO KERNEL. It re-uses ``complex_fields.bloch_pml_curl_step``
byte for byte, under a DEGENERATE NO-ABSORBER BINDING, and the whole of its
content is the predicate, that binding, and the binding's one measured caveat.

THE BINDING, and which of its three parts is a substitution
-----------------------------------------------------------
``stepping._apply_curl`` (stepping.py:508-539) has four branches. With
``_pml_is_active(pml)`` False and no conductivity the entire split-field tail
falls away and the sub-step ends in one line::

    target -= curl                                        # stepping.py:539

1. **Coefficients: substituted, and measured equal to what the round ran.** The
   group-(J) probe bound ``PML(thickness=0)``'s own ``kms``/``sinv`` tables,
   which are EXACTLY 1.0 on both Yee sub-lattices (measured, not assumed:
   ``local/verify_group_j_fixture.json``, ``pml_coefficients_all_exactly_one``).
   This module cannot bind those, because on a genuinely layer-less run ``pml``
   is ``None`` and there are no tables to bind — so it synthesizes unit columns
   instead. :func:`unit_coefficient_columns` is the one home for that, and the
   test file asserts the synthesized columns are bit-identical to
   ``PML(thickness=0)``'s, which is what keeps this binding the one the device
   measured rather than a second, unmeasured one.

2. **Sources: NOT substituted.** Both are read through the engine's OWN
   accessors, the ones ``stepping._read_component`` (stepping.py:2438-2453)
   calls: ``fields.get_E(name)`` for ``step_B`` (storage, guaranteed by the
   shared clause 9) and ``fields.get_H(name)`` for ``step_D``, which without PML
   returns the B array ITSELF and allocates nothing (fields.py:1164-1186).
   Binding the accessor rather than the attribute is deliberate: the attribute
   ``Hx`` is ``None`` here, and a plan-time snapshot of the value is exactly the
   group-(G) mechanism the STALESOURCE control re-armed.

3. **The auxiliary IS a substitution, and it is the only one.** The array path
   has no ``fu`` at all. The faithful stand-in is an auxiliary that is ZERO at
   entry to EVERY launch — zeroed in :meth:`ComplexNoPmlCurlPlan.run`, not once
   at plan time, because after one launch it holds ``-curl`` and a second launch
   would step the substitution instead of the body.

Under (1) and (3) the shipped recurrence (stepping.py:1975-1982)::

    fu_previous = fu.copy(); fu *= kms; fu -= curl; fu *= sinv
    field *= kms_u; field += fu; field -= fu_previous; field *= sinv_u

collapses to ``field - curl``. Re-measured on this tree rather than inherited:
0 differing words of 31250 on the group-(J) fixture fill (7814 negative-zero
words live in each operand), and 0 of 31250 on each of three physical bands
spanning 1e-6 to 1e3, each with a vacuity floor asserting the array path moved
every word it was compared on.

THE ONE MEASURED CAVEAT, stated here because it is a property of the
substitution and not of the kernel
-----------------------------------------------------------------------------
Over the 36-pattern cross product of ``{-0.0, +0.0, +-subnormal, +-normal}`` for
(target, curl), the degenerate binding reproduces ``target -= curl`` on 35 and
diverges on exactly one::

    target = -0.0, curl = +0.0   ->  array path 0x80000000, binding 0x00000000

The mechanism is forced and admits no coefficient repair: the auxiliary reaches
``+0.0 - (+0.0) = +0.0``, and ``-0.0 + (+0.0)`` is ``+0.0`` in round-to-nearest,
so the target's sign bit is lost. Seeding the auxiliary at ``-0.0`` moves the
loss to the ``field -= fu_previous`` line instead of removing it.

Reachability is an ARGUMENT, not a measurement, and is recorded as such: a
``-0.0`` word cannot be CREATED in B or D under this arm's own coverage —
``x - y`` yields ``-0.0`` only from ``-0.0 - (+0.0)``, so the subtraction
preserves the sign but never introduces it, and the volumes are allocated by
``xp.zeros``. It therefore needs a ``-0.0`` written from outside the sub-step.
The byte gate carries the pattern as a named fixture row
(``minus_zero_target_zero_curl``) so a device measures it instead of this
paragraph standing alone, and the released payload records its word count rather
than asserting it away.

WHAT IT IS FOR. The same demand list as the split-field complex product, minus
the absorber: group (J)'s two ``test_material_grid.py`` rows are the measured
demand, and any complex or Bloch-phased run whose boundaries are all periodic or
metallic and which declares no layer lands here. It accelerates no corpus
EXAMPLE script — of the 57 carrying boundary facts, 54 declare ``PML`` and 1
``Absorber`` — and that is stated first because it is the honest size of it.

DISJOINTNESS is the risk here, not arithmetic. Two admitters leave a slot
UNSELECTED and it falls back to the array path — a silent coverage LOSS, not an
error. This family INVERTS EXACTLY ONE numbered question,
``_complex_grid_reasons`` clause 3, and shares the other twelve by CALLING them
rather than copying them. Against every other curl family the separating clause
is named in the test file and measured there, not asserted here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .complex_fields import (
    ComplexPmlCurlPlan,
    DEFAULT_BLOCK,
    _complex_grid_reasons,
    _complex_layout_reasons,
    _resolve_expansion,
    bloch_phase_table,
)
from .coverage import CURL_SUB_STEPS, Coverage
from .launch import SUB_STEPS

__all__ = [
    "ComplexNoPmlCurlPlan",
    "complex_no_pml_curl_coverage",
    "plan_complex_no_pml_curl",
    "plan_complex_no_pml_curl_from_arrays",
    "source_arrays",
    "unit_coefficient_columns",
]

#: The engine accessor each sub-step's sources are read through, by sub-step.
#: ``stepping._read_component`` (stepping.py:2438-2453) is the transcription
#: source; naming them here keeps the builder from ever reaching for the raw
#: attribute, which is ``None`` for H on a layer-less run.
SOURCE_ACCESSOR: Dict[str, str] = {"step_B": "get_E", "step_D": "get_H"}


def unit_coefficient_columns(xp: Any, shape: Sequence[int]) -> Dict[str, Any]:
    """The six degenerate ``kms``/``sinv`` columns, keyed as the plan binds them.

    One float32 column per axis, broadcast-shaped exactly as ``PML`` shapes its
    own (``(nx,1,1)``, ``(1,ny,1)``, ``(1,1,nz)``) so ``launch._flat`` gets the
    contiguous view it requires. ``kms`` and ``sinv`` are the SAME allocation per
    axis: both are all-ones and the kernel only reads them, so a second buffer
    would carry no information and cost a second allocation.

    Under ``PML(thickness=0)`` the engine's own tables are already exactly 1.0 on
    both sub-lattices; the test file pins these against those, so the synthesized
    binding is the measured one and not a second one.
    """
    if len(shape) != 3:
        raise ValueError(f"grid shape {tuple(shape)!r} is not three-dimensional")
    columns: Dict[str, Any] = {}
    for index, axis in enumerate("xyz"):
        broadcast = [1, 1, 1]
        broadcast[index] = int(shape[index])
        ones = xp.ones(tuple(broadcast), dtype=xp.float32)
        columns[f"kms_{axis}"] = ones
        columns[f"sinv_{axis}"] = ones
    return columns


def source_arrays(fields: Any, sub_step: str) -> List[Any]:
    """This sub-step's three source volumes, through the engine's own accessor.

    Read at BUILD time here and re-read nowhere, which is safe for exactly the
    reason the group-(J) STALESOURCE control makes explicit: both accessors
    return a live reference to an array the engine mutates in place — the stored
    E, or the B array itself — never a fresh value. A plan that snapshotted the
    CONTENTS instead would be the group-(G) false pass.
    """
    accessor = getattr(fields, SOURCE_ACCESSOR[sub_step])
    return [accessor(name) for name in SUB_STEPS[sub_step]["sources"]]


def complex_no_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """May the degenerate no-absorber complex curl step this (fields, pml, sub_step)?

    Positive clauses only; a failing clause appends its reason and the scan
    continues. Off-diagonal epsilon is ADMITTED, as it is by the split-field
    twin and for the same reason: it is constitutive-only, its whole effect
    inside ``update_E`` (stepping.py:1001-1020), and it is refused there.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _complex_grid_reasons(fields, pml, grid, probe,
                                    require_active_pml=False)

    # 3b. ``Fields.enable_pml_storage`` is a ONE-WAY switch that also changes
    #     what get_E/get_H return (fields.py:678-700). A Fields whose storage was
    #     switched on while the layer handed to the stepper is inert reads H from
    #     the STORED array — which ``update_H`` declines to write, returning at
    #     stepping.py:944-945 — so the curl would difference a frozen H. The array
    #     path has the same hazard; this refuses the configuration rather than
    #     reproducing it, exactly as no_pml.py:434-437 does for the real twin.
    #     GATED on the layer being inert so the sentence is true where it fires;
    #     where the layer is active clause 3 has already refused.
    if (pml is None or not getattr(pml, "is_active", False)) and \
            bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields has PML storage enabled while the layer is inert "
                       "(get_H would serve a stored H that update_H never writes)")

    spec = SUB_STEPS[sub_step]

    # The targets are attributes and must exist. The SOURCES are not checked as
    # attributes: ``Hx`` is None on a layer-less run by construction, and the
    # question is whether the ACCESSOR answers — which is what the array path
    # asks (stepping.py:2438-2453). No ``fu_*`` clause: this product binds its
    # own zeroed auxiliary and never reads the engine's, allocated or not.
    for name in spec["targets"]:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    accessor = getattr(fields, SOURCE_ACCESSOR[sub_step], None)
    if not callable(accessor):
        reasons.append(
            f"fields.{SOURCE_ACCESSOR[sub_step]} is missing or not callable; the "
            f"source of {sub_step} is read through it, not from an attribute")
        sources: List[Any] = []
    else:
        sources = []
        for name in spec["sources"]:
            try:
                array = accessor(name)
            except Exception as exc:  # noqa: BLE001 - unanswerable is not covered
                reasons.append(f"fields.{SOURCE_ACCESSOR[sub_step]}({name!r}) raised "
                               f"{exc!r}; an unreadable source is not an absent one")
                array = None
            if array is None:
                reasons.append(f"{name} is not served by "
                               f"fields.{SOURCE_ACCESSOR[sub_step]}")
            sources.append(array)

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, spec["targets"]))
    # The sources are values, not attribute names, so they take the volume check
    # directly under the name the array path knows them by.
    if len(shape) == 3:
        from .complex_fields import _complex_volume_reasons  # noqa: PLC0415

        for name, array in zip(spec["sources"], sources):
            if array is not None:
                reasons.extend(_complex_volume_reasons(name, array, shape))

    return Coverage(not reasons, tuple(reasons))


class ComplexNoPmlCurlPlan:
    """A launchable no-absorber COMPLEX curl sub-step: the split-field plan, degenerate.

    Composition rather than a second plan class, so the bytes the gate certifies
    are the bytes ``ComplexPmlCurlPlan.run`` launches — the same kernel, the same
    argument order, the same ``enable_fp_fusion`` guard. This adds exactly one
    thing to it, and adds it on EVERY launch: the auxiliary is zeroed first.

    Allocation-free at launch, not at build: the three auxiliary volumes and the
    three unit columns are allocated once here.
    """

    __slots__ = ("sub_step", "shape", "_inner", "_auxiliaries")

    def __init__(self, inner: ComplexPmlCurlPlan, auxiliaries: Sequence[Any]) -> None:
        self.sub_step = inner.sub_step
        self.shape = inner.shape
        self._inner = inner
        self._auxiliaries = tuple(auxiliaries)

    def run(self, guard: Optional[bool] = None) -> None:
        """Zero the auxiliary, then launch the sub-step in place.

        The zeroing is NOT hoistable to build time. After one launch the
        auxiliary holds ``-curl``; a second launch over it would step
        ``fu*kms - curl`` from a nonzero ``fu`` and measure the substitution
        rather than the body. ``fill(0)`` writes ``+0.0``, which is the state the
        collapse to ``target -= curl`` was measured under.
        """
        for auxiliary in self._auxiliaries:
            auxiliary.fill(0)
        self._inner.run(guard)

    def __repr__(self) -> str:
        return (f"ComplexNoPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"inner={self._inner!r})")


def plan_complex_no_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             num_warps: Optional[int] = None,
                             probe: Any = None
                             ) -> Optional[ComplexNoPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a builder that raises into a caller which would
    otherwise have stepped correctly is strictly worse than the refusal it
    replaces, because a refusal falls back to the array path and is always
    right. That is not a style note — widening the predicate WITHOUT this
    builder is measured to raise ``ValueError: expected a complex64 volume, got
    dtype None`` behind an inactive layer object, and ``AttributeError:
    'NoneType' object has no attribute 'kms_x_h'`` behind no layer at all.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    if not complex_no_pml_curl_coverage(fields, pml, sub_step, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    xp = grid.xp
    # None, NOT the inert layer object: this is the argument the array path
    # passes at stepping.py:318 / :437 (``pml if pml_active else None``), and the
    # shared clause 4 resolved the kinds it admitted the same way.
    kinds = resolve(grid, None)
    phases = bloch_phase_table(grid, kinds)
    shape = tuple(grid.shape)
    columns = unit_coefficient_columns(xp, shape)
    auxiliaries = [xp.zeros(shape, dtype=xp.complex64) for _ in spec["targets"]]

    inner = ComplexPmlCurlPlan(
        sub_step, shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        *_phase_arguments_for(phases, spec),
        expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        auxiliaries,
        source_arrays(fields, sub_step),
        [columns[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )
    return ComplexNoPmlCurlPlan(inner, auxiliaries)


def _phase_arguments_for(phases, spec):
    """(phased flags, phase values) for this sub-step's difference direction."""
    from .complex_fields import _phase_arguments  # noqa: PLC0415

    return _phase_arguments(phases, backward=bool(spec["backward"]))


def plan_complex_no_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         phases: Sequence[Optional[complex]],
                                         dtdx: float, codes, expansion: int,
                                         xp: Any,
                                         block: Optional[int] = None,
                                         kernel: Any = None,
                                         num_warps: Optional[int] = None,
                                         ) -> ComplexNoPmlCurlPlan:
    """Build from bare device arrays — the byte gate's and the benchmark's route.

    ``arrays`` is keyed by component name and carries the TARGETS and the
    SOURCES only; the auxiliaries and the unit columns are this product's own and
    are synthesized here exactly as the engine route synthesizes them, so the
    gate certifies the shipped binding rather than a hand-built lookalike. No
    predicate runs: the caller is a harness that built the configuration
    deliberately, including the deliberately wrong ones, and ``kernel=`` carries
    the mutation override — dropping it silently disarms every mutation leg
    (launch.py:516-522's measured lesson).
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    columns = unit_coefficient_columns(xp, shape)
    auxiliaries = [xp.zeros(shape, dtype=xp.complex64) for _ in spec["targets"]]
    inner = ComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes,
        *_phase_arguments_for(tuple(phases), spec),
        int(expansion),
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        auxiliaries,
        [arrays[name] for name in spec["sources"]],
        [columns[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps,
    )
    return ComplexNoPmlCurlPlan(inner, auxiliaries)


# ---------------------------------------------------------------------------
# WIRING AND DEVICE CERTIFICATION — COMPLETE
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` consults this predicate for ``step_B`` and ``step_D`` when
# complex storage is active and the absorber is inert.  It is disjoint from the
# complex PML arm on ``stepping._pml_is_active``; the central composition sweep
# checks that neither slot can acquire two admitters.
#
# ``parity/meep_gpu/gate_triton_complex_no_pml_curl.py`` ran on the GPU host's RTX
# A6000 on 2026-08-17 under the installed ``keep`` policy.  All eight shipped
# product rows were byte-identical over eight launches; all three controls and
# all three mutations diverged.  The adversarial signed-zero leg records the
# known boundary honestly: step_B remained identical, while step_D differed by
# 86,450/281,250 words in 3-D and 3,552/11,250 in 2-D.  The readable provenance
# entry is ``triton_complex_no_pml_curl_device_gate`` in ``fingerprints.json``
# and licenses this arm at the opted-in fast-path seam.
# Nothing in ``metal_kernels`` or ``cuda_kernels`` changes: this family is
# Triton-only and those backends retain their independent certifications.

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
byte for byte, under a NO-ABSORBER BINDING, and the whole of its content is the
predicate and that binding.

THE BINDING: the kernel's dsig-only form on unit columns, which IS MEEP's plain curl
------------------------------------------------------------------------------------
``stepping._apply_curl`` (``target -= curl`` with no active layer and no
conductivity) is MEEP's plain form, ``f - curl`` (step_generic.cpp:104). The shipped
kernel selects its forms per cell from the chunk record (``forms.curl_form``): the
split-field recurrence where the cell's chunk carries the target's dsigu layer, and
the dsig-only form ``((f * kms) - curl) * sinv`` everywhere else. This binding makes
the second the only one, with three parts:

1. **The record: the absorber-free division's** (``launch.chunk_bounds_for(fields,
   None, chunk_layout)``). No chunk carries a layer, so no membership run exists and
   every cell takes the dsig-only form; the presence word is the grid's, so the
   stencil takes MEEP's one-partner forms on an invariant axis (``curl_stencil``).
2. **Coefficients: unit columns.** On a genuinely layer-less run ``pml`` is ``None``
   and there are no tables to bind, so :func:`unit_coefficient_columns` synthesizes
   them, and the test file pins them bit-identical to ``PML(thickness=0)``'s own
   tables. With ``kms = sinv = 1.0f`` the dsig-only form is ``(f * 1) - curl``
   rounded, times 1: ``f * 1.0`` and ``x * 1.0`` are exact on every word (signed
   zero included, and subnormals under the keep policy), so the form IS
   ``f - curl``. That holds on all 36 pairs of
   ``{-0.0, +0.0, +-subnormal, +-normal}`` for (target, curl); the previous
   binding (the recurrence on every cell with a zeroed auxiliary) lost one of them,
   ``(-0.0, +0.0)``, and the gate keeps that binding as a mutation its planted
   signed-zero leg must catch.
3. **Sources: the engine's own accessors**, the ones ``stepping._read_component``
   calls: ``fields.get_E(name)`` for ``step_B`` (storage, guaranteed by the shared
   clause 9) and ``fields.get_H(name)`` for ``step_D``, which without PML returns the
   B array ITSELF and allocates nothing (``Fields.get_H``). Binding the accessor
   rather than the attribute is deliberate: the attribute ``Hx`` is ``None`` here,
   and a plan-time snapshot of the value is exactly the group-(G) mechanism the
   gate's STALESOURCE control re-arms.

THE AUXILIARY IS A PLACEHOLDER. The kernel takes three ``fu`` pointers, and reads
and writes them only on the recurrence's cells (the auxiliary contract), of which
this record has none. The plan therefore binds three zeroed volumes of its own,
allocated once, and never touches them at launch; the engine has no ``fu`` on this
run. The gate poisons them before every launch and requires the bytes unchanged.

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
    bloch_phase_table,
)
from .coverage import CURL_SUB_STEPS, Coverage
from .launch import SUB_STEPS, chunk_bounds_for

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


def complex_no_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
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

    reasons = _complex_grid_reasons(fields, pml, grid, require_active_pml=False)

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
    """A launchable no-absorber COMPLEX curl sub-step: the split-field plan, unit-bound.

    Composition rather than a second plan class, so the bytes the gate certifies
    are the bytes ``ComplexPmlCurlPlan.run`` launches — the same kernel, the same
    argument order, the same ``enable_fp_fusion`` guard, under the absorber-free
    record and unit columns (see the module docstring).

    Allocation-free at launch, not at build: the three placeholder auxiliary volumes
    and the three unit columns are allocated once here.
    """

    __slots__ = ("sub_step", "shape", "_inner", "_auxiliaries")

    def __init__(self, inner: ComplexPmlCurlPlan, auxiliaries: Sequence[Any]) -> None:
        self.sub_step = inner.sub_step
        self.shape = inner.shape
        self._inner = inner
        self._auxiliaries = tuple(auxiliaries)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step in place.

        Nothing is done to the auxiliary: the absorber-free record has no
        recurrence cell, so the kernel neither reads nor writes it.
        """
        self._inner.run(guard)

    def __repr__(self) -> str:
        return (f"ComplexNoPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"inner={self._inner!r})")


def plan_complex_no_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             num_warps: Optional[int] = None, chunk_layout: Any = None,
                             ) -> Optional[ComplexNoPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a builder that raises into a caller which would
    otherwise have stepped correctly is strictly worse than the refusal it
    replaces, because a refusal falls back to the array path and is always
    right. That is not a style note — widening the predicate WITHOUT this
    builder is measured to raise ``ValueError: expected a complex64 volume, got
    dtype None`` behind an inactive layer object, and ``AttributeError:
    'NoneType' object has no attribute 'kms_x_h'`` behind no layer at all.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default.
    With no absorber every division binds the same words: no membership run, and the
    grid's presence word.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    if not complex_no_pml_curl_coverage(fields, pml, sub_step).covered:
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
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        auxiliaries,
        source_arrays(fields, sub_step),
        [columns[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        # The record of the absorber-free division: no run anywhere, so every cell
        # takes the dsig-only form, which on the unit columns is MEEP's plain curl.
        chunk_bounds=chunk_bounds_for(fields, None, chunk_layout),
    )
    return ComplexNoPmlCurlPlan(inner, auxiliaries)


def _phase_arguments_for(phases, spec):
    """(connection kinds, factor parts) for this sub-step's difference direction."""
    from .complex_fields import connection_arguments  # noqa: PLC0415

    return connection_arguments(phases, backward=bool(spec["backward"]))


def plan_complex_no_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         phases: Sequence[Optional[complex]],
                                         dtdx: float, codes,
                                         xp: Any,
                                         block: Optional[int] = None,
                                         kernel: Any = None,
                                         num_warps: Optional[int] = None,
                                         *, chunk_bounds: Any,
                                         ) -> ComplexNoPmlCurlPlan:
    """Build from bare device arrays — the byte gate's and the benchmark's route.

    ``arrays`` is keyed by component name and carries the TARGETS and the
    SOURCES only; the placeholder auxiliaries and the unit columns are this
    product's own and are synthesized here exactly as the engine route synthesizes
    them, so the gate certifies the shipped binding rather than a hand-built
    lookalike. ``chunk_bounds`` has no default: the harness states the record it
    binds (the engine route binds the absorber-free division's). No
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
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        auxiliaries,
        [arrays[name] for name in spec["sources"]],
        [columns[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
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
# ``parity/meep_gpu/gate_triton_complex_no_pml_curl.py`` first ran on the GPU host's
# RTX A6000 on 2026-08-17 under the installed ``keep`` policy, on the previous
# binding (the recurrence on every cell with a per-launch zeroed auxiliary): all
# eight product rows identical over eight launches, every control and mutation
# divergent, and the signed-zero leg's recorded residual on step_D (86,450/281,250
# words in 3-D, 3,552/11,250 in 2-D) -- the (-0.0, +0.0) pattern that binding lost.
# The readable provenance entry is ``triton_complex_no_pml_curl_device_gate`` in
# ``fingerprints.json``, which licenses this arm at the opted-in fast-path seam; it
# pins those bytes and is re-cut at certification. On the present binding the gate's
# planted leg requires that pattern identical.
# Nothing in ``metal_kernels`` or ``cuda_kernels`` changes: this family is
# Triton-only and those backends retain their independent certifications.

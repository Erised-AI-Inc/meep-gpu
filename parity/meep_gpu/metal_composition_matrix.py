"""The configuration matrix the Metal composition is MEASURED over.

ONE DEFINITION, TWO CONSUMERS. ``meep_gpu/test_metal_planner_composition.py`` runs
this matrix as a pinned expectation that cannot go stale, and
``parity/meep_gpu/probe_metal_planner_composition.py`` runs the same matrix as a
sweep that writes an artifact with the evaluation and overlap counts. Two copies of
a configuration list is how a test and its artifact quietly stop describing the same
thing.

WHAT A ROW IS: ``(label, build, expected_selected, expected_ambiguous)``.

* ``build()`` returns ``(fields, pml)`` from the ENGINE'S OWN classes. Nothing here
  is a duck type except where a row deliberately plants a state the engine cannot
  produce, and those rows say so in their name and their comment.
* ``expected_selected`` is the EXACT ``MetalStepPlan.selected`` dict — the arm label
  per slot. An empty dict is a legitimate and expected answer: every arm refused,
  the whole step is on the array path, which is always correct.
* ``expected_ambiguous`` names the slots that must be left UNSELECTED because TWO OR
  MORE arms admitted. A slot listed here and not ambiguous — or ambiguous and not
  listed — is a predicate defect.

WHY THE MATRIX IS BUILT RATHER THAN LIFTED. The corpus battery (the 186-row,
759-slot measurement) lifts real MEEP scripts and answers "how much of the corpus
does the composer reach". It cannot answer "do two arms ever admit the same slot",
because the corpus does not contain the adversarial pairs — a complex BETA run, a
BFAST run under an inactive absorber, an off-diagonal row planted past the
installer. Disjointness has to be measured on configurations CONSTRUCTED to be near
the boundary between two families, which is what this file is.

TWO ENVIRONMENT FACTS EVERY CONSUMER MUST SET, and both are refusals if unset
rather than silent defaults:

* ``MEEP_GPU_SUBNORMAL_POLICY=flush`` — on MPS the float32 subnormal flush is
  native and has no lever, so the resolved default ``keep`` is NOT OFFERABLE and
  every Metal predicate refuses by name. Setting ``flush`` is what puts the claim
  under the CHECKED subnormal-free precondition rather than under a pretence.
* the FOUR expansion probe artifacts — ``MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE``,
  ``MEEP_GPU_METAL_EXPANSION_PROBE``,
  ``MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE`` and
  ``MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE``. Which complex-multiply arm
  this host's NumPy reference takes is a MEASURED platform fact; with no artifact the
  complex, beta, folded-complex and cylindrical-complex arms refuse by name and six
  families vanish from the sweep, which would read as disjointness rather than as
  absence. EACH IS SEPARATE RATHER THAN SHARED because each family multiplies by an
  operand orientation the earlier artifacts never classified — the folded complex
  fill a complex scalar on the LEFT, the cylindrical complex curl a complex ROW on
  the left and a complex scalar on the left with a SIGNED-ZERO real word — and
  reading one of the others' would licence an arm from a record that never measured
  this call.

:func:`prepare_environment` sets all five and RETURNS what it set, so an artifact
can record it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: The probe artifacts this host has measured, newest first. A missing file is
#: skipped rather than fatal — the caller learns which one was used from the
#: dict :func:`prepare_environment` returns, and a row that needed an absent probe
#: fails its expectation loudly rather than passing as "nothing admitted".
COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_complex_audit_2026-08-16/complex_expansion_probe.json",
    "parity/meep_gpu/results/metal_complex_2026-08-15/complex_expansion_probe.json",
)
BETA_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_special_kz_2026-08-16_audit/expansion.json",
    "parity/meep_gpu/results/metal_special_kz_2026-08-15_certify/expansion.json",
    "parity/meep_gpu/results/metal_special_kz_2026-08-15/expansion.json",
)
#: The folded-complex artifact carries a SIXTH pattern the other two do not — the
#: mirror parity as a complex COEFFICIENT ON THE LEFT — so it is its own candidate
#: list rather than an alias for the complex one.
FOLDED_COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_folded_complex_2026-08-16/expansion.json",
)
#: The cylindrical-complex artifact carries TWO patterns none of the others do — the
#: i*m/r coefficient as a complex ROW on the LEFT, and the |m| = 1 axis-increment
#: scalar as a complex SCALAR on the LEFT whose real word is a signed zero — so it is
#: its own candidate list rather than an alias for the complex one.
CYLINDRICAL_COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_cylindrical_complex_2026-08-16/expansion.json",
)


def _first_existing(candidates: Sequence[str]) -> Optional[str]:
    for relative in candidates:
        path = os.path.join(API_ROOT, relative)
        if os.path.exists(path):
            return path
    return None


def prepare_environment() -> Dict[str, Optional[str]]:
    """Set the policy and every probe path; return exactly what was set."""
    os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    complex_probe = _first_existing(COMPLEX_PROBE_CANDIDATES)
    beta_probe = _first_existing(BETA_PROBE_CANDIDATES)
    folded_complex_probe = _first_existing(FOLDED_COMPLEX_PROBE_CANDIDATES)
    cylindrical_complex_probe = _first_existing(
        CYLINDRICAL_COMPLEX_PROBE_CANDIDATES)
    if cylindrical_complex_probe:
        os.environ.setdefault(
            "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
            cylindrical_complex_probe)
    if complex_probe:
        os.environ.setdefault("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE", complex_probe)
    if beta_probe:
        os.environ.setdefault("MEEP_GPU_METAL_EXPANSION_PROBE", beta_probe)
    if folded_complex_probe:
        os.environ.setdefault("MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
                              folded_complex_probe)
    return {
        "MEEP_GPU_SUBNORMAL_POLICY": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE": complex_probe,
        "MEEP_GPU_METAL_EXPANSION_PROBE": beta_probe,
        "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE": folded_complex_probe,
        "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE":
            cylindrical_complex_probe,
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _fill(fields: Any, seed: int = 5) -> Any:
    """Physical-band values in every allocated field volume.

    NOT COSMETIC. A predicate reads dtype, contiguity and base address, and a
    zero-filled volume passes every one of them — but a plan built on zeros and
    launched is a no-op agreeing with a no-op, which is the vacuity the gates
    refuse. The sweep only calls predicates, so this is here so the SAME builders
    can feed the whole-step gate without a second definition drifting from this one.
    """
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            values = rng.uniform(-0.4, 0.4, array.shape)
            array[...] = values.astype(array.dtype)
    return fields


def _epsilon(fields: Any, rows: Optional[Dict[str, Sequence[str]]] = None,
             seed: int = 7) -> Any:
    shape = tuple(fields.grid.shape)
    rng = np.random.default_rng(seed)
    names = ("Ex", "Ey", "Ez")
    epsilon = {n: np.full(shape, v, np.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    inverse = {n: np.full(shape, np.float32(1.0 / v), np.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    if rows is None:
        fields.set_epsilon_volumes(epsilon, inverse)
    else:
        built = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(np.float32)
                       for partner in partners}
                 for row, partners in rows.items()}
        fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=built)
    return fields


def _polarization(fields: Any, components: Sequence[str] = ("Ez",),
                  kind: str = "lorentzian") -> Any:
    """Build the real recurrence state so ``update_P`` predicates see its bytes."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility

    names = tuple(components)
    sigma = {name: (0.25 if name in names else 0.0)
             for name in ("Ex", "Ey", "Ez")}
    dtype = (np.complex64 if (fields.force_complex_fields
                              or fields.grid.has_bloch) else np.float32)
    return PolarizationState(
        Susceptibility(1.0, 0.1, kind), sigma, fields.grid, dtype)


def cart(cell: Tuple[float, float, float] = (2.0, 2.1, 1.2), pml: Any = 2,
         complex_storage: bool = False, storage: bool = True, eps: bool = True,
         rows: Optional[Dict[str, Sequence[str]]] = None,
         courant: float = 0.35, **grid_kwargs) -> Tuple[Any, Any]:
    """An ordinary Cartesian triple. ``pml=0`` gives an inactive layer."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=cell, courant=courant, xp=np,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if eps:
        _epsilon(fields, rows)
    if storage:
        fields.enable_pml_storage()
    return _fill(fields), PML(grid=grid, thickness=pml)


def flat(cell: Tuple[float, float, float] = (2.0, 2.1, 0.0), pml_cells: int = 2,
         **kwargs) -> Tuple[Any, Any]:
    """A 2-D Cartesian triple — the only shape ``grid.beta`` is defined on.

    The PML is declared PER AXIS because a two-cell layer does not fit on a
    zero-extent z, and a thickness that does not fit is a ``ValueError`` from the
    engine rather than an inactive layer.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    complex_storage = bool(kwargs.pop("complex_storage", False))
    storage = bool(kwargs.pop("storage", True))
    eps = bool(kwargs.pop("eps", True))
    rows = kwargs.pop("rows", None)
    courant = float(kwargs.pop("courant", 0.35))
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=2, courant=courant,
                xp=np, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if eps:
        _epsilon(fields, rows)
    if storage:
        fields.enable_pml_storage()
    thickness = ((pml_cells, pml_cells), (pml_cells, pml_cells), (0, 0))
    return _fill(fields), PML(grid=grid, thickness=thickness)


def folded(axis: str = "Y", phase: Any = 1,
           complex_storage: bool = False,
           boundaries: Any = None,
           rows: Optional[Dict[str, Sequence[str]]] = None,
           extent: float = 2.0, cross: float = 1.6,
           depth: float = 0.0, **grid_kwargs) -> Tuple[Any, Any]:
    """A folded grid, with all FOUR levers that change what the kernels do.

    ``boundaries`` decides MIRROR_METALLIC vs MIRROR_PERIODIC, which is the folded
    family's single point of failure and the axis a sweep must carry BOTH values
    of. Leaving it ``None`` gives the engine's default, which on the folded axis is
    PERIODIC-terminated (``_stored_past_owned`` True: the stored array carries
    MEEP's not-owned slot past ``big_corner``, measured stored 12 vs owned 11 at
    the default extent); declaring the folded axis ``metallic`` gives the other
    termination, where the top plane is OWNED and STEPPED and the far ghost pass
    does not run at all. A mutation caught on one is not caught on the other — the
    Triton gate measured exactly that (``drop_top_plane_mask`` fires only on
    MIRROR_PERIODIC, ``fold_ghost_wraps`` only on MIRROR_METALLIC).

    ``phase`` is the plane's declared parity and may be a per-axis tuple. MIXED
    phases on a two-axis fold are COMMON in the corpus (``solve-cw``,
    ``antenna-radiation``, ``antenna_pec_ground_plane``, ``perturbation_theory_2d``
    all drive one axis even and the other odd), so a matrix carrying only matched
    phases would miss the case where a doubly-unowned corner's two parities differ.

    ``axis`` may name MORE THAN ONE axis (``"XY"``). Two folded axes is what makes
    the fill's X, Y, Z ORDER measurable at all: a corner unowned on two planes must
    carry the product of both parities, and with one folded axis every ordering
    question is trivially null.

    ``extent`` decides the FULL COUNT'S PARITY, which decides the reflect row.
    ``_far_reflect_rows`` is ``n_full - stored + 2``, which is ``stored - 2`` at an
    even full count and ``stored - 3`` at an odd one — so at the default 2.0 the
    wrong ``n - 2`` formula HAPPENS TO BE RIGHT and a gate carrying only that
    extent measures nothing about it. 2.1 gives the odd count where it is a whole
    cell wrong.

    ``**grid_kwargs`` reaches ``Grid`` unchanged, which is how a folded row carries
    ``beta`` or a ``k_point``: those two are the levers the FOLDED COMPLEX family
    must refuse by name, and a matrix that could not build them would record the
    refusal as absence.

    ``cross`` and ``depth`` SIZE the grid and change no verdict — every predicate
    this matrix sweeps reads kinds, flags and dtypes, never an extent. They exist
    for the BYTE GATE, whose claim is only as strong as the number of words it
    moved: at the sweep's default 2-D shape a ghost-fill pass writes one 16-word
    plane, which is a thin thing to certify a family on. ``depth`` nonzero builds
    the 3-D grid (``dimensions=3``) the corpus's folded rows also carry, where the
    same pass writes 192. The defaults reproduce the previous shape exactly, so
    every recorded ``EXPECTED_WINNERS`` row is unchanged.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    axes = tuple(axis)
    phases = (phase,) * len(axes) if isinstance(phase, int) else tuple(phase)
    if len(phases) != len(axes):
        raise ValueError(f"{len(axes)} folded axes but {len(phases)} phases")
    size = [float(cross), float(cross), float(depth)]
    for name in axes:
        size["XYZ".index(name)] = extent
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=2 if not depth else 3,
                courant=0.35,
                symmetry=tuple(Mirror(name, int(value))
                               for name, value in zip(axes, phases)),
                boundaries=boundaries, xp=np, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    _epsilon(fields, rows)
    fields.enable_pml_storage()
    folded_indices = {"XYZ".index(name) for name in axes}
    thickness = []
    for index in range(3):
        if grid.shape[index] < 6:
            thickness.append((0, 0))
        elif index in folded_indices:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    return _fill(fields), PML(grid=grid, thickness=tuple(thickness))


def folded_no_pml(axis: str = "Y", phase: int = 1,
                  complex_storage: bool = False,
                  rows: Optional[Dict[str, Sequence[str]]] = None
                  ) -> Tuple[Any, Any]:
    """A folded grid with NO absorber — the partial-fold-coverage row.

    ``fill_symmetry_bc_*`` runs whether or not a PML is installed
    (stepping.py:1482-1483 gates on ``has_symmetry`` alone), so the mirror fill's
    predicate correctly carries no absorber clause while every curl and
    constitutive predicate requires an active one. The result is a composition that
    covers only the two seam slots, which is legal and has no throughput benefit,
    and this row is what keeps that visible.

    ``complex_storage`` and ``rows`` are ADDITIVE and default to the shape this
    builder had when it carried the real fold alone: the partial-coverage answer is
    a property of the ABSENT ABSORBER rather than of the storage, so each folded
    family owes its own row here — the alternative is inferring two families'
    behaviour from a third's measurement.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    size = [1.6, 1.6, 0.0]
    size["XYZ".index(axis)] = 2.0
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=2, courant=0.35,
                symmetry=(Mirror(axis, phase),), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    _epsilon(fields, rows)
    return _fill(fields), PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))


def cylindrical(m: int = 1, complex_storage: bool = True,
                z_kind: str = "metallic", accurate: bool = False,
                courant: float = 0.37) -> Tuple[Any, Any]:
    """A Dcyl grid. ``z_kind`` and ``accurate`` are the two axes the |m| >= 1 family
    specialises on beyond the sub-step: z PERIODIC drives its other ``BCZ`` arm, and
    ``accurate_fields_near_cylorigin`` selects a DIFFERENT near-axis row count at the
    same m (``stepping._cylindrical_axis_rows``, :601-622), which is the value the
    kernel takes as a runtime uniform rather than as a specialisation.

    ``courant`` is a parameter because ``Grid`` REFUSES the accurate branch above
    ``1/(|m| + 0.5)`` — a real constraint, not a tuning knob, so a row that wants
    that branch has to pass one it accepts.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(16.0, 0.0, 20.0), cylindrical=True,
                m=int(m), boundaries={"z": z_kind}, courant=float(courant),
                accurate_fields_near_cylorigin=bool(accurate), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    _epsilon(fields)
    fields.enable_pml_storage()
    return _fill(fields), PML(grid=grid, thickness={"x": (0, 4), "z": 4})


def conductive(pair: Tuple[Any, Any], value: float = 0.3,
               magnetic: bool = False) -> Tuple[Any, Any]:
    fields, pml = pair
    shape = tuple(fields.grid.shape)
    volumes = {name: np.full(shape, np.float32(value))
               for name in (("Bx", "By", "Bz") if magnetic
                            else ("Dx", "Dy", "Dz"))}
    if magnetic:
        fields.set_b_conductivity(volumes)
    else:
        fields.set_d_conductivity(volumes)
    return fields, pml


def complex_conductive_stored_e_no_pml() -> Tuple[Any, Any]:
    """Complex no-PML conductivity with stored E but no polarization state.

    This is the smallest complete-step intersection for the experimental Metal
    composition: complex conductive B/D curls, the null H return, and the
    zero-pole complex stored-E update.  It deliberately enables ordinary field
    storage *after* the inactive-PML Cartesian fixture is built, rather than PML
    storage: the latter changes ``get_H`` and is conservatively refused by every
    no-PML curl family.
    """
    fields, pml = cart(pml=0, complex_storage=True, storage=False, eps=True)
    fields.enable_field_storage()
    _fill(fields, seed=29)
    return conductive((fields, pml))


def real_stored_e_no_pml(*, with_polarization: bool = False) -> Tuple[Any, Any]:
    """Real no-PML stored-E fixture, optionally with a live ADE successor.

    The zero-pole form is the smallest full-step owner for the real stored-E arm:
    ordinary no-PML B/D curls, a genuine null H return, and pointwise stored E.
    The polarized form adds one seeded Lorentz pole so the same composition also
    exercises the live ``update_E -> update_P`` mirror handoff.  The pole arrays
    must be nonzero: a zero recurrence could agree with a missing ``update_P``.
    """
    fields, pml = cart(pml=0, storage=False, eps=True)
    fields.enable_field_storage()
    _fill(fields, seed=31)
    if with_polarization:
        state = _polarization(fields, ("Ez",))
        fields.polarizations.append(state)
        rng = np.random.default_rng(37)
        for component in state.driven():
            state.P[component][...] = rng.uniform(
                -0.2, 0.2, fields.grid.shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(
                -0.2, 0.2, fields.grid.shape).astype(np.float32)
        state._scratch[...] = rng.uniform(
            -0.2, 0.2, fields.grid.shape).astype(np.float32)
    return fields, pml


def complex_lossless_stored_e_no_pml() -> Tuple[Any, Any]:
    """Complex/Bloch lossless curls with the stored-E successor they require."""
    fields, pml = cart(pml=0, storage=False, eps=True, complex_storage=True,
                       k_point=(0.2, -0.125, 0.0))
    fields.enable_field_storage()
    _fill(fields, seed=41)
    return fields, pml


def complex_lossless_offdiag_no_pml() -> Tuple[Any, Any]:
    """Complex/Bloch no-PML tensor row with its lossless B/D/null-H spine.

    The complex tensor-row shader reads stored E and live off-diagonal inverse-
    permittivity rows directly.  It deliberately carries no poles: that isolates
    the no-absorber tensor product from the still-uncomposed dispersive variants.
    """
    fields, pml = cart(
        pml=0, storage=False, eps=True, complex_storage=True,
        k_point=(0.2, -0.125, 0.0), rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez",)})
    fields.enable_field_storage()
    _fill(fields, seed=43)
    return fields, pml


def dispersive(pair: Tuple[Any, Any],
               components: Sequence[str] = ("Ez",)) -> Tuple[Any, Any]:
    fields, pml = pair
    fields.polarizations.append(_polarization(fields, components))
    return fields, pml


def nonlinear(pair: Tuple[Any, Any], scalar: bool = False,
              chi2: float = 0.1, chi3: float = 0.2,
              components: Sequence[str] = ("Ez",)) -> Tuple[Any, Any]:
    """Install an instantaneous chi2/chi3 on the named E components.

    ``scalar`` picks the OTHER coefficient arm of the Metal Pade kernel — a
    uniform float, which is what ``mp.Medium(chi3=...)`` usually produces and
    which the kernel reads from a packed scalar buffer instead of a volume load.
    ``chi2``/``chi3`` are exposed so a row can sit ABOVE the family's magnitude
    ceiling on purpose; the default pair is far under it.
    """
    fields, pml = pair
    shape = tuple(fields.grid.shape)

    def value(magnitude: float) -> Any:
        return (np.float32(magnitude) if scalar
                else np.full(shape, np.float32(magnitude)))

    fields.set_nonlinear_volumes(
        {name: value(chi2) for name in components},
        {name: value(chi3) for name in components})
    return fields, pml


def plant_rows(pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """A live off-diagonal ROW SLOT with ``has_offdiagonal_epsilon`` still False.

    THE ENGINE CANNOT PRODUCE THIS — the flag is a read-only property over the very
    dict the row accessor reads, so the two move together — and that is exactly why
    it is planted. The ordinary constitutive refuses on the FLAG and the off-diagonal
    family requires a LIVE SLOT, so their disjointness rests on an engine coupling
    rather than on an inverted clause. Planting the disagreement is the only way to
    check that the pair FAILS CLOSED (``update_E`` unselected, both claimants named)
    rather than one of them silently winning.
    """
    fields, pml = pair
    row = np.full(tuple(fields.grid.shape), np.float32(0.1))

    class FlagFalse:
        def __init__(self, wrapped: Any) -> None:
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name: str) -> Any:
            if name == "has_offdiagonal_epsilon":
                return False
            if name == "chi1inv_offdiagonal_for":
                return lambda target: {"Ey": row} if target == "Ex" else {}
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    return FlagFalse(fields), pml


def flag_only(pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """``has_offdiagonal_epsilon`` True with every ROW SLOT dead — REACHABLE.

    The mirror of :func:`plant_rows` and the direction the engine can actually
    reach, via a row that installs to all zeros and is dropped. Both arms must
    refuse: the ordinary one on the flag, the off-diagonal one on the absent slot.
    No ambiguity to resolve, and ``update_E`` correctly falls to the array path.
    """
    fields, pml = pair

    class FlagTrue:
        def __init__(self, wrapped: Any) -> None:
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name: str) -> Any:
            if name == "has_offdiagonal_epsilon":
                return True
            if name == "chi1inv_offdiagonal_for":
                return lambda target: {}
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    return FlagTrue(fields), pml


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

Row = Tuple[str, Callable[[], Tuple[Any, Any]], Dict[str, str], Tuple[str, ...]]

#: Every configuration, with the EXACT selection the shipped composer produces.
#: Recorded from a run of the sweep on this host and then held as an expectation.
MATRIX: Tuple[Row, ...] = (
    # -- the tranche-1 real-field products -----------------------------------
    ("cart_pml_real", lambda: cart(),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"),
     ()),
    ("cart_pml_real_metallic", lambda: cart(boundaries="metallic"),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"),
     ()),
    ("cart_pml_real_odd_courant", lambda: cart(courant=0.2718281828),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"),
     ()),
    ("flat_pml_real_2d", lambda: flat(),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"),
     ()),
    # A conductivity changes the CURL recurrence only (stepping.py:508), and the
    # curl predicate is SUB-STEP AWARE about which one: an ELECTRIC conductivity is
    # read by step_D (the D advance) and leaves step_B alone, a MAGNETIC one the
    # other way round. The dedicated conductive-PML family owns that affected curl;
    # the other curl and both constitutive slots stay on their ordinary products,
    # because the constitutive sub-step never reads condfac_for.  The whole-step
    # source-welded gate proves the resulting B/H/D/E residency composition.
    ("cart_pml_conductive_electric", lambda: conductive(cart()),
     dict(step_B="PML", step_D="conductive PML curl",
          update_H="ordinary", update_E="ordinary"), ()),
    ("cart_pml_conductive_magnetic", lambda: conductive(cart(), magnetic=True),
     dict(step_B="conductive PML curl", step_D="PML",
          update_H="ordinary", update_E="ordinary"), ()),
    # A pole makes update_E's source (D - sum P); update_H is untouched.
    ("cart_pml_dispersive", lambda: dispersive(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_E="dispersive PML E", update_P="ADE update_P"), ()),

    # -- complex / Bloch ------------------------------------------------------
    ("cart_pml_complex_forced", lambda: cart(complex_storage=True),
     dict(step_B="complex/Bloch", step_D="complex/Bloch",
          update_H="complex/Bloch", update_E="complex/Bloch"), ()),
    ("cart_pml_bloch_complex",
     lambda: cart(complex_storage=True, k_point=(0.3, 0.0, 0.0)),
     dict(step_B="complex/Bloch", step_D="complex/Bloch",
          update_H="complex/Bloch", update_E="complex/Bloch"), ()),
    ("cart_pml_bloch_two_axes",
     lambda: cart(complex_storage=True, k_point=(0.3, -0.2, 0.0)),
     dict(step_B="complex/Bloch", step_D="complex/Bloch",
          update_H="complex/Bloch", update_E="complex/Bloch"), ()),
    # Complex conductivity changes B/D only.  Its specialized family claims both
    # curls whenever any target is conductive, then compiles a lossless tail for
    # the unaffected half-step; H/E reuse the ordinary complex pointwise body under
    # a scope view that hides conductivity and nothing else.
    ("cart_pml_complex_conductive_electric",
     lambda: conductive(cart(complex_storage=True)),
     dict(step_B="complex conductive PML curl",
          step_D="complex conductive PML curl",
          update_H="complex conductive PML constitutive",
          update_E="complex conductive PML constitutive"), ()),
    ("cart_pml_bloch_conductive_magnetic",
     lambda: conductive(
         cart(complex_storage=True, k_point=(0.3, -0.2, 0.0)), magnetic=True),
     dict(step_B="complex conductive PML curl",
          step_D="complex conductive PML curl",
          update_H="complex conductive PML constitutive",
          update_E="complex conductive PML constitutive"), ()),
    # A nonzero k with REAL storage: the complex family's clause 2 admits it (a
    # Bloch phase implies complex storage) and its LAYOUT clause then refuses the
    # float32 arrays. Nothing is selected — the array path, which is correct.
    ("cart_pml_bloch_real", lambda: cart(k_point=(0.3, 0.0, 0.0)), {}, ()),
    ("cart_pml_complex_metallic",
     lambda: cart(complex_storage=True, boundaries="metallic"),
     dict(step_B="complex/Bloch", step_D="complex/Bloch",
          update_H="complex/Bloch", update_E="complex/Bloch"), ()),
    # Off-diagonal is constitutive-only: the complex curls take it, the complex
    # update_E refuses it, and the off-diagonal family is real-storage only.
    ("cart_pml_complex_offdiag",
     lambda: cart(complex_storage=True, rows={"Ex": ("Ey",)}),
     dict(step_B="complex/Bloch", step_D="complex/Bloch",
          update_H="complex/Bloch"), ()),
    ("cart_pml_complex_dispersive", lambda: dispersive(cart(complex_storage=True)),
     dict(step_B="complex dispersive PML curl",
          step_D="complex dispersive PML curl",
          update_H="complex dispersive PML magnetic",
          update_E="complex dispersive PML E", update_P="ADE update_P"), ()),

    # -- the no-PML curl + null constitutive families ------------------------
    ("nopml_real_nostore", lambda: cart(pml=0, storage=False, eps=False),
     dict(step_B="no-PML curl", step_D="no-PML curl",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_real_eps_nostore", lambda: cart(pml=0, storage=False),
     dict(step_B="no-PML curl", step_D="no-PML curl",
          update_H="no-PML null", update_E="no-PML null"), ()),
    # The magnetic companion below completes the existing electric-conductivity
    # row.  The dedicated family claims both B/D slots whenever a conductivity
    # exists, compiling a lossless tail for the other half-step; inactive-PML H/E
    # remain true null plans.
    ("nopml_real_conductive_magnetic",
     lambda: conductive(cart(pml=0, storage=False, eps=False), magnetic=True),
     dict(step_B="conductive no-PML curl", step_D="conductive no-PML curl",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_complex_conductive_stored_e", complex_conductive_stored_e_no_pml,
     dict(step_B="complex conductive no-PML curl",
          step_D="complex conductive no-PML curl", update_H="no-PML null",
          update_E="complex no-PML stored E"), ()),
    ("nopml_complex_bloch_lossless", complex_lossless_stored_e_no_pml,
     dict(step_B="complex no-PML curl", step_D="complex no-PML curl",
          update_H="no-PML null", update_E="complex no-PML stored E"), ()),
    ("nopml_complex_bloch_offdiag", complex_lossless_offdiag_no_pml,
     dict(step_B="complex no-PML curl", step_D="complex no-PML curl",
          update_H="no-PML null", update_E="complex no-PML off-diagonal"), ()),
    ("nopml_real_stored_e", real_stored_e_no_pml,
     dict(step_B="no-PML curl", step_D="no-PML curl",
          update_H="no-PML null", update_E="no-PML stored E"), ()),
    ("nopml_real_stored_e_ade",
     lambda: real_stored_e_no_pml(with_polarization=True),
     dict(step_B="no-PML curl", step_D="no-PML curl",
          update_H="no-PML null", update_E="no-PML stored E",
          update_P="ADE update_P"), ()),
    ("nopml_none_layer",
     lambda: (cart(pml=0, storage=False, eps=False)[0], None),
     dict(step_B="no-PML curl", step_D="no-PML curl",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_complex_nostore",
     lambda: cart(pml=0, storage=False, eps=False, complex_storage=True),
     dict(update_H="no-PML null", update_E="no-PML null"), ()),
    # The absorber is inactive but the PML STORAGE is allocated. The null family
    # refuses the whole pair — a run that allocated f_w_* is not the run whose
    # update_H returns at :916-917 — so the step is entirely on the array path.
    # This is the same answer the Triton composer gives on the same shape.
    ("nopml_real_stored", lambda: cart(pml=0, storage=True, eps=False), {}, ()),
    # A BFAST run under an inactive absorber: the BFAST curl needs an ACTIVE
    # absorber, so the curls are on the array path and the null pair is correct.
    ("nopml_bfast",
     lambda: cart(pml=0, storage=False, eps=False,
                  bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_conductive",
     lambda: conductive(cart(pml=0, storage=False, eps=False)),
     dict(step_B="conductive no-PML curl", step_D="conductive no-PML curl",
          update_H="no-PML null", update_E="no-PML null"), ()),

    # -- special_kz (grid.beta) ----------------------------------------------
    ("beta_real_2d", lambda: flat(beta=0.33),
     dict(step_B="special_kz real beta", step_D="special_kz real beta",
          update_H="special_kz real beta", update_E="special_kz real beta"), ()),
    ("beta_real_2d_negative", lambda: flat(beta=-0.685),
     dict(step_B="special_kz real beta", step_D="special_kz real beta",
          update_H="special_kz real beta", update_E="special_kz real beta"), ()),
    # Metallic on x only: a 2-D run's z is translationally invariant and the engine
    # refuses a PEC declaration there outright (grid.py:863), which is a boundary
    # the matrix must respect rather than route around.
    ("beta_real_2d_metallic", lambda: flat(beta=0.33, boundaries={"x": "metallic"}),
     dict(step_B="special_kz real beta", step_D="special_kz real beta",
          update_H="special_kz real beta", update_E="special_kz real beta"), ()),
    # THE COMPLEX BETA ARM GAINED ITS CONSTITUTIVE COMPANION ON 2026-08-19 and this
    # row grew the two slots it had been missing. Until then the arm was registered
    # as an unconditional refusal so the gap was reported by name; the row listed
    # only the curls, and a whole complex beta step was half on the array path.
    ("beta_complex_2d", lambda: flat(beta=0.33, complex_storage=True),
     dict(step_B="special_kz complex beta",
          step_D="special_kz complex beta",
          update_H="special_kz complex beta",
          update_E="special_kz complex beta"), ()),
    ("beta_real_2d_dispersive", lambda: dispersive(flat(beta=0.33)),
     dict(step_B="special_kz real beta", step_D="special_kz real beta",
          update_H="special_kz real beta", update_P="ADE update_P"), ()),

    # -- BFAST ----------------------------------------------------------------
    ("bfast_real_pml", lambda: cart(bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(step_B="BFAST", step_D="BFAST",
          update_H="BFAST", update_E="BFAST"), ()),
    ("bfast_real_pml_metallic",
     lambda: cart(bfast_scaled_k=(0.2, 0.0, 0.0), boundaries="metallic"),
     dict(step_B="BFAST", step_D="BFAST",
          update_H="BFAST", update_E="BFAST"), ()),
    ("bfast_real_2d", lambda: flat(bfast_scaled_k=(0.2, 0.1, 0.0)),
     dict(step_B="BFAST", step_D="BFAST",
          update_H="BFAST", update_E="BFAST"), ()),
    ("bfast_complex", lambda: cart(bfast_scaled_k=(0.2, 0.0, 0.0),
                                   complex_storage=True), {}, ()),

    # -- the off-diagonal update_E --------------------------------------------
    ("offdiag_real_one_row", lambda: cart(rows={"Ex": ("Ey",)}),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_E="offdiag"), ()),
    ("offdiag_real_two_rows",
     lambda: cart(rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez",)}),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_E="offdiag"), ()),
    # THE PLANTED DOUBLE-ADMIT. Unreachable on the engine, kept permanently: it is
    # the only configuration on which two wired arms admit one slot, and the point
    # is that the composition REFUSES rather than picking.
    ("offdiag_flag_false_slot_live", lambda: plant_rows(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary"), ("update_E",)),
    # The reachable mirror image: both arms refuse, no ambiguity.
    ("offdiag_flag_true_slots_dead", lambda: flag_only(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary"), ()),

    # -- the fold (mirror symmetry) -------------------------------------------
    # SIX SLOTS, and the two fill rows are the first seam slots any Metal family
    # has filled. The two TERMINATIONS are carried separately and deliberately:
    # `fold_real_2d` is MIRROR_PERIODIC (codes (0, 3, 0)) so the top-plane mask and
    # the far ghost pass both fire, and `fold_real_2d_metallic` is MIRROR_METALLIC
    # (codes (0, 2, 0)) where neither does. A row carrying one proves nothing about
    # the other.
    ("fold_real_2d", lambda: folded(complex_storage=False),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("fold_real_2d_metallic",
     lambda: folded(complex_storage=False, boundaries={"y": "metallic"}),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # ODD PLANE. `mp.Mirror(d, phase=-1)` is the same fold with every sign
    # inverted, which on this backend is a DIFFERENT COMPILED FILL — the parity is
    # a source specialisation here because a runtime weight flushes subnormals.
    ("fold_real_2d_odd", lambda: folded(phase=-1, complex_storage=False),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("fold_real_2d_odd_metallic",
     lambda: folded(phase=-1, boundaries={"y": "metallic"}),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # X-FOLD. The fill's plane-index arithmetic is specialised per axis and axis 0
    # is the only one whose base is the flat index unchanged.
    ("fold_real_2d_x", lambda: folded(axis="X"),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # THE WALL SEAM. A folded PERIODIC axis needs the far ghost pass, and
    # `zero_metal_*` runs BETWEEN the two fill passes; the curl and constitutive
    # slots are unaffected and still compose, so this row shows the refusal is
    # SCOPED to the two seam slots rather than sinking the run.
    ("fold_real_2d_walled",
     lambda: folded(boundaries={"x": "metallic"}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded"), ()),
    ("fold_real_2d_dispersive", lambda: dispersive(folded()),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded dispersive PML E", update_P="ADE update_P", fill_B="mirror fill",
          fill_D="mirror fill"), ()),
    # A conductivity changes the CURL recurrence and NOT update_H/update_E, which
    # is why the two contracts are separate functions in the family module.
    ("fold_real_2d_conductive", lambda: conductive(folded()),
     dict(step_B="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # A folded NO-PML run: the two fill slots are covered while both CURLS fall to
    # the array path, because `fill_symmetry_bc_*` runs whether or not an absorber
    # is installed. A substitution with no throughput benefit, named here rather
    # than discovered later.
    #
    # THE NULL PAIR WINS update_H/update_E ON A FOLDED RUN, and it was MEASURED
    # here rather than assumed: `no_pml_constitutive` does not reach
    # `coverage._grid_reasons`, names no fold clause, and admits whenever the
    # absorber is inactive. That is not an ambiguity — the folded constitutive
    # requires an ACTIVE absorber, which is the inversion — and it is the correct
    # answer there, since `stepping.update_H` returns at :916-917 and `update_E` at
    # :954-955 before reading an array. The same shape `bfast_curl` records for a
    # BFAST run under an inactive absorber.
    ("fold_real_2d_no_pml", lambda: folded_no_pml(),
     dict(fill_B="mirror fill", fill_D="mirror fill",
          update_H="no-PML null", update_E="no-PML null"), ()),

    # -- the fold COMPOSED WITH AN OFF-DIAGONAL ROW ---------------------------
    # THE INTERSECTION OF TWO CERTIFIED FAMILIES, and the rows are chosen for what
    # only the composition can get wrong. `offdiag_update_e` refuses every fold
    # through clause 5 and `folded` refuses every live row because its constitutive
    # body is element-wise, so ONE arm must take `update_E` here and the other five
    # slots must stay with the folded curls and fills. A row where `update_E` fell
    # to the array path would be a silent coverage loss rather than a wrong answer.
    ("fold_offdiag_2d", lambda: folded(rows={"Ex": ("Ey",)}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded offdiag",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # THE CORPUS'S OWN SHAPE. All 19 folded off-diagonal corpus rows are 2-D and
    # folded METALLIC, over the two BC triples (METALLIC, MIRROR_METALLIC, PERIODIC)
    # and (MIRROR_METALLIC, MIRROR_METALLIC, PERIODIC). Not one folded PERIODIC row
    # exists, which is why the row above carries the other termination.
    ("fold_offdiag_2d_metallic",
     lambda: folded(boundaries={"y": "metallic"},
                    rows={"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded offdiag",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # THE ODD PLANE, and the polarity inverts the reading a reader arrives with:
    # the ghost weight is `-phase`, so an ODD plane weighs +1 and the emitted term
    # is the CERTIFIED one character for character, while an EVEN plane is the one
    # that needs the negated body.
    ("fold_offdiag_2d_odd", lambda: folded(phase=-1, rows={"Ex": ("Ey",)}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded offdiag",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # A POLE ON TOP: the source becomes (D - sum P) at every tensor-row gather.
    # The dedicated packed-pole product owns E while the folded B/D/H/fill spine
    # and ADE recurrence retain their independently verified slots.
    ("fold_offdiag_2d_dispersive",
     lambda: dispersive(folded(rows={"Ex": ("Ey",)})),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded off-diagonal dispersive PML E", update_P="ADE update_P", fill_B="mirror fill",
          fill_D="mirror fill"), ()),
    # THE SECOND PLANTED DOUBLE-ADMIT, and it is the SAME engine coupling as
    # `offdiag_flag_false_slot_live` one level down: `folded` refuses on the FLAG
    # and `folded offdiag` requires a LIVE ROW SLOT, and the two agree only because
    # `has_offdiagonal_epsilon` is a read-only property over the very dict the row
    # accessor reads. Unreachable on the engine, kept permanently, and the point is
    # that the composition REFUSES rather than picking.
    ("fold_offdiag_flag_false_slot_live", lambda: plant_rows(folded()),
     dict(step_B="folded", step_D="folded", update_H="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ("update_E",)),
    # The reachable mirror image: flag True, every slot dead. `folded` refuses on
    # the flag and `folded offdiag` is GATED OUT (no live slot), so `update_E` falls
    # to the array path with one claimant's reasons and no ambiguity.
    ("fold_offdiag_flag_true_slots_dead", lambda: flag_only(folded()),
     dict(step_B="folded", step_D="folded", update_H="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),

    # -- the fold UNDER COMPLEX STORAGE ---------------------------------------
    # THE COMPOSITION OF TWO CERTIFIED FAMILIES, and the rows are chosen for what
    # only the composition can get wrong. Both parents refuse every one of these BY
    # NAME — `symmetry` on storage, `complex_fields` on the fold — so a row where
    # neither the folded-complex arm nor a refusal appears is a silent coverage loss
    # rather than a wrong answer, which is why every one carries an expectation.
    ("fold_complex_2d", lambda: folded(complex_storage=True),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # THE OTHER TERMINATION. MIRROR_METALLIC runs no far pass at all, so the fill
    # plan's `far` list is empty and it declares only its own slot in
    # `replaces_sub_steps` — a different residency answer from the row above.
    ("fold_complex_2d_metallic",
     lambda: folded(complex_storage=True, boundaries={"y": "metallic"}),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # THE ODD PLANE. Under REAL storage this is a different compiled fill, because
    # the parity is a source specialisation there; under COMPLEX it is the SAME
    # compiled body with a different passed coefficient word, which is the visible
    # consequence of the parity having become arithmetic.
    ("fold_complex_2d_odd", lambda: folded(phase=-1, complex_storage=True),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # TWO FOLDED AXES AT MIXED PHASE — the only shape where the X, Y, Z fill order
    # is a real question. Under real storage every fill is a multiply by exactly
    # +/-1 and the order commutes (a measured null); complex multiplication is not
    # associative, so the doubly-unowned corner sees c_y (x) (c_x (x) z) against
    # c_x (x) (c_y (x) z). With c_x == c_y the two agree bit-exactly, so a
    # same-phase row would make the question vanish rather than answer it.
    ("fold_complex_2d_xy_mixed",
     lambda: folded(axis="XY", phase=(1, -1), complex_storage=True),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # ODD FULL COUNT. `_far_reflect_rows` is `stored - 2` at an even full count and
    # `stored - 3` at an odd one, so a matrix carrying only the default extent lets
    # the wrong `n - 2` formula pass.
    ("fold_complex_2d_odd_count",
     lambda: folded(complex_storage=True, extent=2.1),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # A BLOCH PHASE ON AN UNFOLDED AXIS — THE COMPOSITION'S OWN CASE, and the only
    # fold-plus-phase pairing the engine will build: `Grid` refuses a k component on
    # a mirror plane's own axis outright ("a mirror plane forces the field to be even
    # or odd about it, which only a zero Bloch phase allows"). Three of the five
    # non-beta folded complex corpus rows look like this, and without it every row
    # here would exercise the fold and the rotation only separately.
    ("fold_complex_2d_bloch",
     lambda: folded(complex_storage=True, k_point=(0.3, 0.0, 0.0)),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # AN OFF-DIAGONAL ROW. The dedicated complex folded tensor product owns E;
    # folded complex B/D/H and the two fills remain the verified spine. Three of
    # the eight folded complex corpus rows have this shape.
    ("fold_complex_2d_offdiag",
     lambda: folded(complex_storage=True, rows={"Ey": ["Ez"]}),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="complex folded off-diagonal PML E",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # A FOLDED COMPLEX BETA RUN — `folded_beta`'s complex arm.
    #
    # THIS ROW MOVED OUT OF `UNCARRIED` IN THE SAME CHANGE THAT BUILT THE FAMILY,
    # which is the only way a non-vacuity floor may be moved. Until then it pinned
    # the four arithmetic slots as selecting NOTHING, with the refusal named on both
    # sides (`folded_complex`'s "beta is nonzero", `special_kz`'s "a mirror plane is
    # active"), and that pin was correct: no product carried the intersection.
    #
    # THE TWO FILL SLOTS ARE UNCHANGED, and that is the point of keeping them
    # spelled here. `fill_symmetry_bc_*` copies `phase * field[2]` and touches no
    # beta term, so `folded_complex`'s fill predicate carries NO beta clause and
    # already admitted this row before the family existed. `folded_beta` therefore
    # registers NO fill arm: a third arm on those two slots would make them
    # AMBIGUOUS and drop them to the array path, which is a coverage LOSS dressed as
    # completeness. A row here that showed a folded-beta fill winning would be that
    # mistake.
    ("fold_complex_2d_beta",
     lambda: folded(complex_storage=True, beta=0.3),
     dict(step_B="folded beta complex", step_D="folded beta complex",
          update_H="folded beta complex", update_E="folded beta complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # THE SAME RUN WITH A BLOCH PHASE ON THE UNFOLDED AXIS — which is what two of
    # the four folded-beta corpus rows actually are
    # (`TestEigCoeffs.test_binary_grating_special_kz_*`, kx != 0 on x with the fold
    # on y). It exercises clause 9's THREE questions at once: the phase rides a
    # plain-PERIODIC axis (legal), the FOLDED axis carries none (required), and the
    # invariant z axis carries none because beta already carries its dependence.
    # Without this row every folded-beta complex verdict would rest on the
    # zero-phase specialisation, which is the arm `_phase_lines` emits as a comment.
    ("fold_complex_2d_beta_bloch",
     lambda: folded(complex_storage=True, beta=0.3, k_point=(0.3, 0.0, 0.0)),
     dict(step_B="folded beta complex", step_D="folded beta complex",
          update_H="folded beta complex", update_E="folded beta complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),

    # -- folded_beta, THE REAL ARM -------------------------------------------
    # A FOLDED REAL BETA RUN. Moved out of `UNCARRIED` in the same change that built
    # the family, for the same reason as its complex twin, and it is the row the
    # corpus's `TestSpecialKz.test_eigsrc_kz_1_real_imag` looks like.
    #
    # THE TWO ARMS BOTH SPEAK ON EVERY SLOT HERE AND EXACTLY ONE ADMITS. The gate
    # `_has_folded_beta` asks only about the fold and beta, deliberately, so the
    # complex arm is CONSULTED on this row and says "storage is real float32" while
    # the real arm is consulted on the row above and says
    # "force_complex_fields=True". That pair of sentences IS the clause-2 inversion,
    # and gating either out would hide which arm declined.
    ("fold_real_2d_beta", lambda: folded(beta=0.3),
     dict(step_B="folded beta real", step_D="folded beta real",
          update_H="folded beta real", update_E="folded beta real",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # THE FOLD OVER A METALLIC OUTER DECLARATION — MIRROR_METALLIC rather than
    # MIRROR_PERIODIC. `folded_top_plane_mask` emits NOTHING there, because the
    # stored array stops at `big_corner` and the top plane is owned and stepped, so
    # masking it would delete a real cell. A matrix carrying only the periodic
    # termination would let a top-plane mask that fires unconditionally pass —
    # and on THIS family that mask sits directly downstream of the beta term.
    ("fold_real_2d_beta_metallic",
     lambda: folded(beta=0.3, boundaries={"y": "metallic"}),
     dict(step_B="folded beta real", step_D="folded beta real",
          update_H="folded beta real", update_E="folded beta real",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # A DISPERSIVE folded beta run. The SCOPE row: dispersion changes the VALUES
    # `update_E` writes into the array the curl differences, not an operation the
    # curl performs, so the two curls AND `update_H` compose while `update_E` is
    # refused by name (its source becomes `D - sum P`). A pin showing all four
    # would be an arm stepping a recurrence it does not implement.
    ("fold_real_2d_beta_dispersive", lambda: dispersive(folded(beta=0.3)),
     dict(step_B="folded beta real", step_D="folded beta real",
          update_H="folded beta real", update_P="ADE update_P",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # THE WALLED FOLDED BETA RUN, on both storages. This is the TEN-PASS shape for
    # this family: `zero_metal_*` sits BETWEEN the near and far ghost passes
    # (driver.py:3285/:3286), so both fold families refuse the two fill slots by
    # name and the four arithmetic slots still compose. It is the row where the
    # fold's seam, the wall's seam and the beta term are all live in one step, and
    # the pin records that the fill refusal is SCOPED to the seam rather than
    # sinking the beta curls with it.
    ("fold_real_2d_beta_walled",
     lambda: folded(beta=0.3, boundaries={"x": "metallic"}),
     dict(step_B="folded beta real", step_D="folded beta real",
          update_H="folded beta real", update_E="folded beta real"), ()),
    ("fold_complex_2d_beta_walled",
     lambda: folded(complex_storage=True, beta=0.3,
                    boundaries={"x": "metallic"}),
     dict(step_B="folded beta complex", step_D="folded beta complex",
          update_H="folded beta complex", update_E="folded beta complex"), ()),

    # -- THE FOLD AGAINST THE FAMILIES IT MUST REFUSE -------------------------
    # THE NEGATIVE CONTROLS THE FOLDED ROWS ABOVE DO NOT CARRY. Every row above
    # puts a folded family beside families that refuse the FOLD; these put it
    # beside the two levers the folded families themselves refuse — BFAST and beta
    # — and the pins record the SCOPE of that refusal, which is the part a reader
    # would get wrong. Both fills still compose, because `fill_symmetry_bc_*`
    # copies `phase * field[2]` and reads neither a BFAST term nor a beta one, so a
    # refusal there would decline a pass those levers cannot reach.
    ("fold_real_2d_bfast", lambda: folded(bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(fill_B="mirror fill", fill_D="mirror fill"), ()),
    # A NONLINEARITY on a folded run: `update_E`'s source stops being (D - sum P)
    # and every constitutive arm refuses. The curls refuse too, because the
    # nonlinear builder's fields carry the flag the folded curl reads.
    ("fold_real_2d_nonlinear", lambda: nonlinear(folded()),
     dict(fill_B="mirror fill", fill_D="mirror fill"), ()),

    # -- THE FOLD IN THREE DIMENSIONS AND ON EVERY AXIS -----------------------
    # The corpus's folded rows are not all 2-D and not all folded on Y. The fill's
    # plane-index arithmetic is specialised per axis and the 3-D grid is where a
    # ghost pass writes 192 words rather than 16, so a matrix carrying only the
    # 2-D Y fold measures one specialisation of nine.
    ("fold_real_3d", lambda: folded(depth=1.2),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("fold_real_3d_z", lambda: folded(axis="Z", depth=2.0),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("fold_real_2d_x_metallic",
     lambda: folded(axis="X", boundaries={"x": "metallic"}),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # TWO FOLDED AXES AT MIXED PHASE under REAL storage. The complex row above is
    # where the fill ORDER is a live question; this is its control, and it is here
    # because a doubly-unowned corner is a different index path whatever the
    # storage.
    ("fold_real_2d_xy", lambda: folded(axis="XY", phase=(1, -1)),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # ODD FULL COUNT under real storage — `_far_reflect_rows` is `stored - 3` here
    # and `stored - 2` at the default extent, so the wrong formula passes on every
    # other real folded row in this matrix.
    ("fold_real_2d_odd_count", lambda: folded(extent=2.1),
     dict(step_B="folded", step_D="folded", update_H="folded", update_E="folded",
          fill_B="mirror fill", fill_D="mirror fill"), ()),

    # -- THE FOLDED COMPLEX FAMILY'S OWN BOUNDARIES ---------------------------
    # THE WALL SEAM, complex side. The mirror image of `fold_real_2d_walled`: the
    # two fills are refused BY NAME (`zero_metal_*` runs between the passes) while
    # every arithmetic slot still composes. Measured on both storages because the
    # complex fill imports the real family's helper and an import is not a proof
    # that the clause survived the composition.
    ("fold_complex_2d_walled",
     lambda: folded(complex_storage=True, boundaries={"x": "metallic"}),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex"), ()),
    # A CONDUCTIVITY, and the pin records a REAL ASYMMETRY between the two folded
    # families rather than a symmetry a reader would assume: the real fold keeps
    # `step_B`, `update_H` and `update_E` under an electric conductivity, and the
    # complex fold keeps NONE of them. That is the complex family's own conductivity
    # clause, inherited through `complex_fields`, and this row is where it is
    # visible instead of inferred.
    ("fold_complex_2d_conductive", lambda: conductive(folded(complex_storage=True)),
     dict(fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # A FOLDED COMPLEX NO-PML RUN — the partial-coverage shape, complex side: the
    # two fills compose and the NULL PAIR takes the constitutive slots, exactly as
    # it does on the real fold.
    ("fold_complex_2d_no_pml", lambda: folded_no_pml(complex_storage=True),
     dict(fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("fold_complex_2d_bfast",
     lambda: folded(complex_storage=True, bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    ("fold_complex_3d", lambda: folded(complex_storage=True, depth=1.2),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="folded complex",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    # THREE LEVERS AT ONCE: fold, Bloch phase on an unfolded axis, and a live
    # off-diagonal row. The same dedicated E body keeps its two phase stages while
    # the folded-complex spine supplies the other five slots.
    ("fold_complex_2d_offdiag_bloch",
     lambda: folded(complex_storage=True, k_point=(0.3, 0.0, 0.0),
                    rows={"Ey": ["Ez"]}),
     dict(step_B="folded complex", step_D="folded complex",
          update_H="folded complex", update_E="complex folded off-diagonal PML E",
          fill_B="folded complex fill", fill_D="folded complex fill"), ()),

    # -- THE FOLDED OFF-DIAGONAL FAMILY'S OWN BOUNDARIES ----------------------
    ("fold_offdiag_2d_walled",
     lambda: folded(boundaries={"x": "metallic"}, rows={"Ex": ("Ey",)}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded offdiag"), ()),
    ("fold_offdiag_2d_conductive", lambda: conductive(folded(rows={"Ex": ("Ey",)})),
     dict(step_B="folded", update_H="folded", update_E="folded offdiag",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("fold_offdiag_3d", lambda: folded(depth=1.2, rows={"Ex": ("Ey",)}),
     dict(step_B="folded", step_D="folded", update_H="folded",
          update_E="folded offdiag",
          fill_B="mirror fill", fill_D="mirror fill"), ()),
    # A FOLDED OFF-DIAGONAL NO-PML RUN, and the pin records the ONE PLACE the null
    # pair splits: `update_H` is the null's and `update_E` is NOBODY'S. The null
    # constitutive refuses an off-diagonal row on the E side, the folded
    # off-diagonal arm requires an ACTIVE absorber, and the slot correctly falls to
    # the array path. A row that showed the null taking BOTH sides would be the
    # null silently deleting the row multiply.
    ("fold_offdiag_2d_no_pml", lambda: folded_no_pml(rows={"Ex": ("Ey",)}),
     dict(fill_B="mirror fill", fill_D="mirror fill",
          update_H="no-PML null"), ()),

    # -- the chi2/chi3 Pade product -------------------------------------------
    # The Pade family owns update_E.  Its new nonlinear spine owns the B/D curls
    # and magnetic constitutive side through THREE separate, inverted admissions:
    # those sub-steps do not read chi2 or chi3, but the ordinary predicates retain
    # their global clause so a linear and nonlinear arm cannot co-admit.  This is
    # an added product rather than a widening of a shared predicate.
    ("nonlinear_real", lambda: nonlinear(cart()),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic", update_E="nonlinear"), ()),
    # A SCALAR chi pair, which is the commoner spelling of `mp.Medium(chi3=...)`
    # and takes the kernel's OTHER coefficient arm — the packed scalar buffer
    # rather than a volume load. The composition answer must not depend on which,
    # and a row that only ever exercised volumes would not say so.
    ("nonlinear_real_scalar_chi", lambda: nonlinear(cart(), scalar=True),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic", update_E="nonlinear"), ()),
    # A PARTLY nonlinear run: Ez nonlinear, Ex and Ey linear. The linear
    # components compile to the certified plain constitutive body verbatim, so
    # this row is where the per-component split is visible as a composition fact
    # rather than only as a kernel-source one.
    ("nonlinear_real_metallic", lambda: nonlinear(cart(boundaries="metallic")),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic", update_E="nonlinear"), ()),
    # A WALLED nonlinear run. `zero_metal_B`/`_D` clear stored cell 0 BETWEEN the
    # curl and the constitutive sub-step, so the four composed slots sit downstream
    # of a host pass that writes a volume their mirrors shadow. The wall is a
    # residency question, not a predicate one, and the whole-step gate is where it
    # is settled.
    ("nonlinear_real_walled", lambda: nonlinear(cart(boundaries="metallic")),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic", update_E="nonlinear"), ()),
    # THE NULL PAIR SPLITS ON A NONLINEAR RUN, and that is this row's whole content.
    # Under an inactive absorber `no_pml_constitutive` admits `update_H` — correct,
    # `stepping.update_H` returns at :916-917 — and REFUSES `update_E` by name,
    # because that side's return is not unconditional when a Pade factor is
    # installed. A row showing the null taking BOTH sides would be the null silently
    # deleting the nonlinearity, which is a wrong ANSWER rather than a missed slot.
    # The same shape `fold_offdiag_2d_no_pml` records one family over.
    ("nonlinear_no_pml",
     lambda: nonlinear(cart(pml=0, storage=False, eps=False)),
     dict(update_H="no-PML null"), ()),

    # -- the cylindrical complex product, |m| >= 1 ----------------------------
    # MOVED OUT OF `UNCARRIED` IN THE SAME CHANGE THAT MADE THE FAMILY REAL, which
    # is the only way a row is allowed to leave that list. ALL FOUR arithmetic
    # slots are composed, and the two constitutive ones are the CERTIFIED complex
    # body under a restated predicate rather than new arithmetic — `update_H` and
    # `update_E` carry no cylindrical branch at all.
    #
    # THE ROWS BELOW ARE THE FAMILY'S OWN SPECIALISATION AXES, one row each, so a
    # sweep that only ever built m = +1 with a metallic z would not measure the
    # composition of the arm the corpus's z-PERIODIC rows or its |m| >= 2 rows take:
    #   BACKWARD  — both sub-steps, every row
    #   BCZ       — `dcyl_m1_complex` (metallic) vs `dcyl_m1_complex_zperiodic`
    #   M_CLASS   — `dcyl_m1_complex` (M_ONE) vs `dcyl_m3_complex` (M_MANY)
    #   zero_rows — `dcyl_m3_complex` (3) vs `dcyl_m3_complex_accurate` (1); a
    #               RUNTIME UNIFORM, not a specialisation, which is why the two
    #               rows compose identically and only the bound value differs
    ("dcyl_m1_complex", lambda: cylindrical(m=1),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    # NEGATIVE m: seven of the sixteen corpus rows are m = -1, and the sign is not
    # cosmetic — it flips the i*m/r row's sign AND puts a -0.0 in the real word of
    # the |m| = 1 axis-increment scalar (`1j * X` for X < 0), which is the operand
    # class a plane-wise product gets wrong.
    ("dcyl_m1_complex_negative", lambda: cylindrical(m=-1),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m1_complex_zperiodic", lambda: cylindrical(m=1, z_kind="periodic"),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m3_complex", lambda: cylindrical(m=3),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m3_complex_accurate",
     lambda: cylindrical(m=3, accurate=True, courant=0.25),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),

    # -- cylindrical m = 0, real storage: the OTHER cylindrical kernel ---------
    # THIS ROW MOVED OUT OF `UNCARRIED` IN THE CHANGE THAT MADE THE FAMILY REAL, and
    # that direction matters: a row is never moved to turn a red floor green. What
    # it measures now is the separation between the two cylindrical families by
    # STORAGE (clause 2, inverted in both directions): since 2026-09-04 the complex
    # family carries complex64 at every m, so this row and `dcyl_m0_complex` below
    # are the same m under the two storages, and dropping either family's storage
    # clause would turn one of them AMBIGUOUS rather than silently pick one, which
    # is the outcome the composer is built to produce. (Until that round the pair's
    # separation was a DOUBLE inversion — clause 14 on m as well — and the complex
    # corner below was a named absence.)
    ("dcyl_m0_real", lambda: cylindrical(m=0, complex_storage=False),
     dict(step_B="cylindrical m=0", step_D="cylindrical m=0",
          update_H="cylindrical m=0", update_E="cylindrical m=0"), ()),
    # THE FAMILY'S ONE REMAINING SPECIALISATION AXIS, one row so it is driven rather
    # than merely enumerable: BCZ. The r axis is pinned METALLIC by construction (the
    # source emitter REFUSES any other code) and the wall-row arm is decided by the
    # sub-step, so z PERIODIC vs METALLIC is the only compiled axis a configuration
    # can move.
    ("dcyl_m0_real_zperiodic",
     lambda: cylindrical(m=0, complex_storage=False, z_kind="periodic"),
     dict(step_B="cylindrical m=0", step_D="cylindrical m=0",
          update_H="cylindrical m=0", update_E="cylindrical m=0"), ()),
    # A CONDUCTIVITY ON A CYLINDRICAL RUN — the SCOPE row this family did not have.
    # A conductivity changes the CURL recurrence only (stepping.py:508), so the two
    # curls fall to the array path and the constitutive pair still composes. The pin
    # records that the refusal is scoped to the two slots that read `condfac_for`
    # rather than sinking the run, which is the mistake a family-wide clause makes.
    ("dcyl_m0_real_conductive",
     lambda: conductive(cylindrical(m=0, complex_storage=False)),
     dict(update_H="cylindrical m=0", update_E="cylindrical m=0"), ()),

    # -- THE (m, storage) SQUARE'S OTHER TWO CORNERS ---------------------------
    # `dcyl_m0_complex` MOVED OUT OF `UNCARRIED` ON 2026-09-04, in the same change
    # that gave the cylindrical complex family its `M_ZERO` arm — the direction a
    # row may move. Until then it was a named absence: the complex family refused
    # m = 0 by name and the real family refused complex64 storage, so this
    # constructible configuration (and the corpus row `dipole_in_vacuum_cyl_off_
    # axis.py`) composed NOTHING and fell to the array path on all four slots. Now
    # all four come back `cylindrical complex`, and the real family's storage clause
    # is what keeps the row unambiguous. `dcyl_m1_real` stays the one dead corner:
    # nobody carries it, and the engine itself refuses to step it (stepping.py:
    # 731-736), so a family that admitted it would be admitting a run that cannot
    # run. Neither row is a plant — the engine builds both without complaint.
    ("dcyl_m0_complex", lambda: cylindrical(m=0, complex_storage=True),
     dict(step_B="cylindrical complex", step_D="cylindrical complex",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m1_real", lambda: cylindrical(m=1, complex_storage=False), {}, ()),
    # A POLE on a cylindrical run: `update_E`'s source becomes (D - sum P) and the
    # cylindrical dispersive update_E is still uncarried, while the geometry-free
    # ADE recurrence now composes independently in update_P.
    ("dcyl_m1_complex_dispersive", lambda: dispersive(cylindrical(m=1)),
     dict(update_P="ADE update_P"), ()),
    # A NONLINEARITY on a cylindrical run — the intersection of two families this
    # backend carries SEPARATELY and does not carry together. `nonlinear` refuses
    # the r = 0 axis by name and the cylindrical families refuse the Pade factor by
    # name, so the row measures that neither one over-reaches into the other's
    # geometry.
    ("dcyl_m0_real_nonlinear",
     lambda: nonlinear(cylindrical(m=0, complex_storage=False)), {}, ()),

    # -- intersections with deliberately partial Metal coverage ----------------
    ("nonlinear_complex", lambda: nonlinear(cart(complex_storage=True)), {}, ()),
    # THE MAGNITUDE CEILING, AS A COMPOSITION ROW. This is a real chi2/chi3 run on
    # a grid the family otherwise admits; the ONLY thing refusing it is clause (g),
    # `chi3 * |chi1inv|^3` above 1e29. Carried here so the refusal is measured as a
    # composition outcome — the slot falls to the array path — rather than only as
    # a predicate unit test, because a ceiling that refused in the predicate and
    # was then composed anyway would be the worst of both.
    #
    # THE VALUE IS COMPUTED AGAINST `_epsilon`, NOT PICKED TO LOOK BIG, and the
    # first spelling of this row got it wrong in the direction that matters: with
    # `chi3 = 1e30` and Ez's installed `chi1inv = 1/3`, the clause quantity is
    # 1e30 * (1/3)^3 = 3.7e28 — UNDER the ceiling — so the row was ADMITTED and the
    # pin of `{}` failed.  A ceiling row that does not clear the ceiling tests the
    # opposite of what it claims, and it was the composition sweep that said so.
    # 1e32 * (1/3)^3 = 3.7e30, two decades clear.  The ceiling belongs to update_E
    # only, so the unaffected nonlinear spine remains selectable.
    ("nonlinear_real_above_ceiling", lambda: nonlinear(cart(), chi3=1e32),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic"), ()),
    # A NONLINEARITY AND A LIVE OFF-DIAGONAL ROW — two `update_E` families in one
    # run, and the one pairing among them that could have overlapped. Both update_E
    # arms refuse — `nonlinear` because the row product reads neighbours and its
    # Pade body is element-wise, `offdiag_update_e` because clause 10 calls χ²/χ³
    # uncarried.  B/D/H are unaffected and remain a positive pin for the spine.
    ("nonlinear_offdiag", lambda: nonlinear(cart(rows={"Ex": ("Ey",)})),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic"), ()),
    # A NONLINEARITY AND A POLE. `update_E`'s source becomes (D - sum P) AND carries
    # a Pade factor; no product on this backend composes the two at update_E.  The
    # curl and magnetic sides are independent of both features; update_P remains
    # the ADE plan's own slot.
    ("nonlinear_dispersive", lambda: dispersive(nonlinear(cart())),
     dict(step_B="nonlinear PML curl", step_D="nonlinear PML curl",
          update_H="nonlinear PML magnetic", update_P="ADE update_P"), ()),
    # A NONLINEARITY ON AN UNFOLDED BETA RUN — the beta twin of
    # `fold_real_2d_nonlinear`, and it is here because the two are refused by
    # DIFFERENT clauses: the folded row by clause 5 on the nonlinear side, this one
    # by clause 12. A matrix carrying only the folded pairing would leave the
    # unfolded beta intersection untested in both directions.
    ("beta_real_2d_nonlinear", lambda: nonlinear(flat(beta=0.33)), {}, ()),
)

#: Every arm label that must WIN at least one row. A sweep in which an arm never
#: wins measures nothing about that arm; this is the non-vacuity floor and it is
#: asserted rather than eyeballed.
EXPECTED_WINNERS: Tuple[str, ...] = (
    "PML", "ordinary", "no-PML curl", "no-PML null", "complex/Bloch",
    "complex conductive no-PML curl", "complex no-PML curl",
    "complex no-PML stored E", "complex no-PML off-diagonal", "no-PML stored E",
    "dispersive PML E", "complex dispersive PML curl",
    "complex dispersive PML magnetic", "complex dispersive PML E",
    "special_kz real beta", "special_kz complex beta", "BFAST", "offdiag",
    "folded", "mirror fill", "folded complex", "folded complex fill",
    "folded offdiag", "folded dispersive PML E",
    "folded off-diagonal dispersive PML E", "complex folded off-diagonal PML E",
    "nonlinear", "cylindrical complex", "cylindrical m=0",
    "folded beta real", "folded beta complex", "ADE update_P",
    "nonlinear PML curl", "nonlinear PML magnetic",
)

#: The four arithmetic slots. The two fill slots are deliberately NOT here: a seam
#: pass that copies `phase * field[2]` reads no beta term, no BFAST term, no pole
#: and no nonlinearity, so a fill winning on a row whose ARITHMETIC this backend
#: cannot step is the correct answer rather than a violation.
ARITHMETIC_SLOTS: Tuple[str, ...] = ("step_B", "step_D", "update_H", "update_E")

#: THE OTHER HALF OF THE NON-VACUITY FLOOR, and the half a sweep normally forgets.
#: ``EXPECTED_WINNERS`` catches an arm that never fires; this catches an arm that
#: fires where NO PRODUCT EXISTS. Each row names a family this backend does not
#: carry, and the floor is that NOTHING is selected on any of the four arithmetic
#: slots — a quiet admission here is a kernel stepping a recurrence it does not
#: implement, which is a wrong ANSWER rather than a missed opportunity.
UNCARRIED: Dict[str, str] = {
    # `dcyl_m1_complex` LEFT THIS LIST when the cylindrical complex family landed,
    # `dcyl_m0_real` LEFT IT when the cylindrical REAL family landed, and
    # `dcyl_m0_complex` LEFT IT on 2026-09-04 when the complex family gained its
    # `M_ZERO` arm — each in the SAME change that made the product real, never to
    # make a red floor go green. Three of the (m, storage) square's four corners
    # are now positive pins above, and the two families are separated by STORAGE
    # alone (clause 2, inverted in both directions).
    # `nonlinear_real` LEFT THIS LIST when the Pade family landed, and the two
    # rows below stayed — deliberately, and each is refused BY NAME by a clause
    # that family keeps UN-inverted rather than by an omission. Complex storage:
    # MEEP nonlinearizes the two parts independently (the DOCMP split), which is a
    # second kernel body and not a wider dtype. Folded: `c2` is linear in D and so
    # carries D's parity while `c3` is even, which is why the engine itself
    # refuses a chi2 on a component a plane makes odd (fields.py:887).
    "nonlinear_complex": "chi2/chi3 nonlinear update_E, complex storage",
    "fold_real_2d_nonlinear": "chi2/chi3 nonlinear update_E, folded",
    # `fold_real_2d_beta` AND `fold_complex_2d_beta` LEFT THIS LIST when
    # `folded_beta` landed, in the SAME change that made the family real — which is
    # the only way this floor may be moved. Their rows above now pin all four
    # arithmetic slots to the new arms, and the two BFAST rows below stay because
    # nothing carries a folded BFAST curl: `symmetry` clause 11 and
    # `folded_complex` clause 8 still refuse it by name, and `bfast_curl` still
    # refuses the fold. THE PARALLEL IS EXACT AND THAT IS WHY THE TWO ARE KEPT
    # BESIDE EACH OTHER: beta and BFAST are both a term added between the curl and
    # the mask, and one of the two is now composed while the other is a named
    # absence. A reader who sees only the green one cannot tell those apart.
    "fold_real_2d_bfast": "folded BFAST",
    "fold_complex_2d_bfast": "folded complex BFAST",
    # THE SEVEN BELOW WERE ADDED IN THE TRANCHE-4 COMPOSITION ROUND, and adding to
    # this list is the direction that costs nothing to be wrong about: a row here
    # asserts that NOTHING is selected, so a family later built for it fails this
    # floor loudly and gets moved out deliberately, exactly as the four families
    # that landed this round were. Each was MEASURED to select nothing on all four
    # arithmetic slots before being written down; none is a guess about a clause.
    #
    # The first is the (m, storage) square's one dead corner (its complex64 m = 0
    # partner left this list on 2026-09-04): nobody carries |m| >= 1 under real
    # storage and the engine refuses to step it, so a family that admitted it here
    # would be admitting a run that cannot run.
    "dcyl_m1_real": "cylindrical |m| >= 1 under real float32 storage",
    "dcyl_m1_complex_dispersive": "cylindrical complex dispersive update_E",
    "dcyl_m0_real_nonlinear": "cylindrical chi2/chi3 update_E",
    # The two nonlinear intersections below retain a deliberately unselected
    # update_E, but their B/D/H spine is now selected.  They therefore cannot live
    # in this all-arithmetic-slots-negative table; their exact partial selections
    # above are the regression pins for the unfinished E product.
    "beta_real_2d_nonlinear": "chi2/chi3 update_E on an unfolded beta run",
}

#: All seven driver slots now have at least one registered product.  The tuple is
#: retained as the fail-closed census seam for future driver-slot additions.
UNREGISTERED_SLOTS: Tuple[str, ...] = ()

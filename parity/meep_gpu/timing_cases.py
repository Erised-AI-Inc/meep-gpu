"""The timing cases: one builder per case, read by the GPU bench and the MEEP bench alike.

Why this module exists. The identical-case comparison holds only if both sides build
the same ``mp.Simulation``. Until this module, the one case either side could time was
``pml_3d``, a 4 um cell that is 78.4 % PML by volume with its electric source off the
fused seams, so no timing exercised the deposit-repair bracket or a thin absorber. The
cases here are defined ONCE and resolved by name by every driver that builds them:

* ``bench_meep_identical_case.py`` (stock MEEP under ``mpirun``),
* ``digest_harness_lift.py`` (the geometry digest of the package's own lift),
* ``gpu_array_control.py`` (the array-path control as its own process),
* ``bench_timing_case.py`` (``bench_fused_products.py`` with the case injected).

A name that is already a case of ``gate_dispatch_fused_route.CASES`` resolves to THAT
object, so ``pml_3d`` is byte for byte the builder of 2026-09-28 and the GPU row for it
is the unmodified ``bench_fused_products.py``. The new cases live here and nowhere
else: they are injected into the gates' tables at run time by ``bench_timing_case.py``
and removed again, so no pinned file changes.

The builder contract is the route gates': ``builder(mp, res) -> (sim, [monitors],
until)``, with ``sim`` not yet initialised.

IMPORTS. This module imports neither MEEP nor ``meep_gpu`` nor a device library at
module scope; a builder receives ``mp`` from its caller, and the route gate (which imports
neither) is imported only when a built-in name is resolved or a case built on a route-gate
builder (:func:`route_case`) is built or described.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

#: The file every harness root holds; roots are found BY NAME, never by depth.
HARNESS_MARKER = "gate_dispatch_end_to_end.py"

#: The DRIVE tables a case can be injected into, as ``bench_fused_products`` names them.
TABLES = ("triton", "cuda", "metal")

Builder = Callable[..., Tuple[Any, List[Any], float]]


# ---------------------------------------------------------------------------
# The new cases
# ---------------------------------------------------------------------------

#: Shared by every new case: an 8 um cube with a 0.5 um PML on every side, so the
#: nominal PML fraction is 1 - (7/8)^3 = 0.330 against pml_3d's 1 - (2.4/4)^3 = 0.784.
#: Because 8 r' = 4 r, every resolution r of the 2026-09-28 ladder maps exactly to
#: r' = r / 2 here, and the two families can be named by cell count.
THIN_CELL = (8.0, 8.0, 8.0)
THIN_PML = 0.5
THIN_UNTIL = 35.0
#: The source's centre frequency and width, and the flux plane's three frequencies.
THIN_FCEN = 0.5
THIN_DF = 0.4
THIN_NFREQ = 3


def _source(mp: Any, component: Any) -> Any:
    """A Gaussian pulse 1 um inside the absorber's inner face on -x, on the axis."""
    return mp.Source(mp.GaussianSource(frequency=THIN_FCEN, fwidth=THIN_DF),
                     component=component, center=mp.Vector3(-2.5, 0, 0))


def _flux(mp: Any, sim: Any) -> Any:
    """One flux plane at x = 2.5 spanning 5 x 5 um of the interior."""
    return sim.add_flux(THIN_FCEN, THIN_DF, THIN_NFREQ, mp.FluxRegion(
        center=mp.Vector3(2.5, 0, 0), size=mp.Vector3(0, 5, 5)))


def _substrate(mp: Any) -> Any:
    """An axis-aligned slab under the scatterers that runs through the absorber in x
    and y, so the PML carries material as a user's substrate does."""
    return mp.Block(mp.Vector3(mp.inf, mp.inf, 1.0), center=mp.Vector3(0, 0, -2.5),
                    material=mp.Medium(epsilon=2.1))


def _curved_geometry(mp: Any) -> List[Any]:
    """The structured interior: curved surfaces and a rotated block, so MEEP's subpixel
    averaging writes off-diagonal chi1inv rows, as on ``pml_3d``'s sphere."""
    return [
        _substrate(mp),
        mp.Sphere(radius=0.8, center=mp.Vector3(0, 0, 0),
                  material=mp.Medium(epsilon=9)),
        mp.Cylinder(radius=0.5, height=3.0, axis=mp.Vector3(0, 0, 1),
                    center=mp.Vector3(1.5, 1.5, -0.5),
                    material=mp.Medium(epsilon=6)),
        mp.Block(mp.Vector3(1.5, 0.6, 0.6), center=mp.Vector3(-1.2, -1.5, 1.0),
                 e1=mp.Vector3(1, 1, 0), e2=mp.Vector3(-1, 1, 0),
                 e3=mp.Vector3(0, 0, 1), material=mp.Medium(epsilon=4)),
    ]


def _diagonal_geometry(mp: Any) -> List[Any]:
    """The same arrangement with axis-aligned blocks only, so the permittivity tensor
    stays diagonal and both ordinary pairs are admitted, as on ``pml_3d_diagonal``."""
    return [
        _substrate(mp),
        mp.Block(mp.Vector3(1.6, 1.6, 1.6), center=mp.Vector3(0, 0, 0),
                 material=mp.Medium(epsilon=9)),
        mp.Block(mp.Vector3(1.0, 1.0, 3.0), center=mp.Vector3(1.5, 1.5, -0.5),
                 material=mp.Medium(epsilon=6)),
        mp.Block(mp.Vector3(1.5, 0.6, 0.6), center=mp.Vector3(-1.2, -1.5, 1.0),
                 material=mp.Medium(epsilon=4)),
    ]


def _simulation(mp: Any, res: int, geometry: List[Any], component: Any) -> Any:
    return mp.Simulation(cell_size=mp.Vector3(*THIN_CELL),
                         boundary_layers=[mp.PML(THIN_PML)], geometry=geometry,
                         sources=[_source(mp, component)], resolution=res)


def case_thin_pml_3d(mp: Any, res: int = 6) -> Tuple[Any, List[Any], float]:
    """Thin PML, structured off-diagonal interior, Ez source (off the fused seams)."""
    sim = _simulation(mp, res, _curved_geometry(mp), mp.Ez)
    return sim, [_flux(mp, sim)], THIN_UNTIL


def case_thin_pml_3d_hz(mp: Any, res: int = 6) -> Tuple[Any, List[Any], float]:
    """``thin_pml_3d`` with an Hz source: the deposit lands on the fused B seam."""
    sim = _simulation(mp, res, _curved_geometry(mp), mp.Hz)
    return sim, [_flux(mp, sim)], THIN_UNTIL


def case_thin_pml_3d_diagonal(mp: Any, res: int = 6) -> Tuple[Any, List[Any], float]:
    """Thin PML, axis-aligned interior (diagonal epsilon), Ez source on the D seam."""
    sim = _simulation(mp, res, _diagonal_geometry(mp), mp.Ez)
    return sim, [_flux(mp, sim)], THIN_UNTIL


# ---------------------------------------------------------------------------
# The extended ladder's cases: one declared change to a route-gate builder each
# ---------------------------------------------------------------------------
#
# Each builder below calls a route gate's OWN builder (``pml_3d`` or
# ``pml_3d_diagonal``) and changes one stated thing on the returned simulation before
# ``init_sim``, so the base geometry is never re-typed and the cell stays the 4 um cube
# of the 2026-09-28 ladder: resolutions 20 / 32 / 48 hold 512,000 / 2,097,152 /
# 7,077,888 cells, the sizes of record. Measured on the package's NumPy lift
# (``test_seams_and_off_diagonal_flag_on_a_numpy_lift``) and per table by the first
# preflight row of a ladder.

#: The Lorentz medium of ``case_dispersive_2d``: eps_inf 2.25, one pole at 0.4 with
#: gamma 0.02 and sigma 1.1.
LORENTZ_EPSILON = 2.25
LORENTZ_POLE = {"frequency": 0.4, "gamma": 0.02, "sigma": 1.1}
#: The monitor load of ``pml_3d_monitors``: six flux planes closing a box of half-width
#: 1 um around the sphere, each at 20 frequencies, and one DFT field plane at z = 0 for
#: Ex, Ey and Ez at 5 frequencies, on top of ``pml_3d``'s own incident flux plane.
BOX_HALF_WIDTH = 1.0
BOX_NFREQ = 20
FIELD_PLANE_SIZE = (2.4, 2.4, 0.0)
FIELD_PLANE_NFREQ = 5


def route_case(name: str) -> Builder:
    """A route gate's own builder, imported by name from the harness this module sits
    in (the gate imports neither MEEP nor the package at module scope)."""
    root = find_harness_root(HERE)
    if not root:
        raise SystemExit(f"REFUSING: no {HARNESS_MARKER} above {HERE}; the base builders "
                         "are the route gate's, never re-typed")
    put_on_path(root)
    import gate_dispatch_fused_route as route  # noqa: PLC0415
    return route.CASES[name]


def case_pml_3d_lorentz(mp: Any, res: int = 20) -> Tuple[Any, List[Any], float]:
    """``pml_3d_diagonal`` with its slab a Lorentz medium: the dispersive arms in 3-D.

    The slab (1 um in y, through the absorber in x and z) keeps every chi1inv row
    diagonal. A curved or rotated dispersive body is refused by the 1.0 lift (the sigma
    check reads 1/chi1inv[c][c], which is eps_cc only for a diagonal tensor), measured
    on a Lorentz sphere and on the eps-9 sphere beside a Lorentz block. The Ez source is
    injected in the D seam; whether that seam is FUSED (and so bracketed) is the
    table's: :data:`TABLE_OVERRIDES` names what each table's smoke measured.
    """
    sim, monitors, _until = route_case("pml_3d_diagonal")(mp, res)
    sim.geometry[0].material = mp.Medium(
        epsilon=LORENTZ_EPSILON,
        E_susceptibilities=[mp.LorentzianSusceptibility(**LORENTZ_POLE)])
    return sim, monitors, 35.0


def case_pml_3d_monitors(mp: Any, res: int = 20) -> Tuple[Any, List[Any], float]:
    """``pml_3d`` with a realistic monitor load: 8 DFT objects where ``pml_3d`` has 1.

    Six separate flux planes (MEEP keeps a multi-region ``add_flux`` as ONE DFT object
    and the lift as one monitor per plane, so separate planes keep the two counts equal)
    and one ``add_dft_fields`` plane, every one stepped inside each timed window.
    """
    sim, monitors, until = route_case("pml_3d")(mp, res)
    monitors = list(monitors)
    fcen, df = 0.4, 0.4
    for axis in range(3):
        for sign in (1.0, -1.0):
            center = [0.0, 0.0, 0.0]
            center[axis] = sign * BOX_HALF_WIDTH
            size = [2.0 * BOX_HALF_WIDTH] * 3
            size[axis] = 0.0
            monitors.append(sim.add_flux(fcen, df, BOX_NFREQ, mp.FluxRegion(
                center=mp.Vector3(*center), size=mp.Vector3(*size))))
    monitors.append(sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], fcen, df, FIELD_PLANE_NFREQ,
                                       center=mp.Vector3(),
                                       size=mp.Vector3(*FIELD_PLANE_SIZE)))
    return sim, monitors, until


def case_pml_3d_hz(mp: Any, res: int = 20) -> Tuple[Any, List[Any], float]:
    """``pml_3d`` with an Hz source at the same point: the deposit on the fused B seam.

    The one change to the case of record, so the cost of the B-seam deposit-repair
    bracket is read against ``pml_3d`` directly: the same plan (fused pair B, the D seam
    on its off-diagonal singles; measured on the GPU host's smoke 2026-10-17, both NVIDIA
    tables, 3.0-3.3 launches per step) with the bracket around the B pair.
    """
    sim, monitors, until = route_case("pml_3d")(mp, res)
    sim.sources = [mp.Source(source.src, component=mp.Hz, center=source.center,
                             size=source.size, amplitude=source.amplitude)
                   for source in sim.sources]
    return sim, monitors, until


@dataclasses.dataclass(frozen=True)
class TimingCase:
    """What a case declares about itself, before anything is built.

    ``template`` is the DRIVE row the case's GPU spec is copied from on every table
    (``pml_3d``: the magnetic pair alone, the D seam keeping its off-diagonal singles;
    ``pml_3d_diagonal``: both ordinary pairs). ``expected_repair`` is the seam whose
    deposit-repair bracket the fused leg must carry (``None``: no bracket), checked on
    every GPU row by ``timing_records.py record-gpu``. ``off_diagonal_epsilon`` is what
    the lifted grid must read.
    """
    name: str
    builder: Optional[Builder]
    cell: Tuple[float, float, float]
    pml: float
    template: Optional[str]
    expected_repair: Optional[str]
    off_diagonal_epsilon: bool
    source_component: str
    isolates: str
    #: The route-gate case the builder changes (``None``: a builder of its own).
    base: Optional[str] = None
    #: DFT objects the initialised simulation carries (MEEP's ``sim.dft_objects``).
    dft_objects: int = 1
    #: Whether a susceptibility is registered (the dispersive arms run).
    dispersive: bool = False
    #: The factor the ladder scales a GPU or lift row's time budget (and so its timeout)
    #: by: a dispersive lift verifies sigma at every grid point (measured 2026-10-17 at
    #: 110,592 cells on the GPU host: 10-15 s a leg against 2-6 s for pml_3d; 55 us a cell
    #: at 512,000 cells on the Mac), which the 1.0 row times do not hold at 7,077,888 cells.
    budget_scale: float = 1.0

    @property
    def pml_fraction_nominal(self) -> float:
        interior = 1.0
        for extent in self.cell:
            interior *= max(0.0, extent - 2.0 * self.pml) / extent
        return 1.0 - interior


#: The cases this module defines. Geometry is confirmed per case by
#: ``test_timing_cases.py`` (on a NumPy lift: the off-diagonal flag and which seam the
#: source lands on) and, per table, by the first preflight row of a ladder.
NEW_CASES: Dict[str, TimingCase] = {
    "thin_pml_3d": TimingCase(
        "thin_pml_3d", case_thin_pml_3d, THIN_CELL, THIN_PML, "pml_3d", None, True,
        "Ez", "the PML fraction: pml_3d's arms and source seam on a thin absorber"),
    "thin_pml_3d_hz": TimingCase(
        "thin_pml_3d_hz", case_thin_pml_3d_hz, THIN_CELL, THIN_PML, "pml_3d", "B", True,
        "Hz", "a source on a fused seam: the B-seam deposit-repair bracket"),
    "thin_pml_3d_diagonal": TimingCase(
        "thin_pml_3d_diagonal", case_thin_pml_3d_diagonal, THIN_CELL, THIN_PML,
        "pml_3d_diagonal", "D", False, "Ez",
        "both ordinary pairs and the D-seam deposit-repair bracket"),
    # The extended ladder (2026-10-17): the pml_3d cell, so the sizes of record hold.
    # TEMPLATE pml_3d, NO BRACKET: on the GPU host's smoke (2026-10-17) both NVIDIA
    # tables compose fused pair B alone on this 3-D dispersive grid (Triton: fused pair B,
    # PML step_D, dispersive update_E, ADE update_P; CUDA: its fused magnetic pair with the
    # same Triton singles on the D side), so the Ez deposit is in no fused pair.
    "pml_3d_lorentz": TimingCase(
        "pml_3d_lorentz", case_pml_3d_lorentz, (4.0, 4.0, 4.0), 0.8, "pml_3d", None,
        False, "Ez",
        "a dispersive medium: pml_3d_diagonal's slab as a Lorentz medium (eps_inf 2.25, "
        "one pole at 0.4, gamma 0.02, sigma 1.1) through the absorber; fused pair B with "
        "the dispersive update_E and ADE update_P arms on the D side",
        base="pml_3d_diagonal", dispersive=True, budget_scale=3.0),
    "pml_3d_monitors": TimingCase(
        "pml_3d_monitors", case_pml_3d_monitors, (4.0, 4.0, 4.0), 0.8, "pml_3d", None,
        True, "Ez",
        "a monitor load: pml_3d plus six flux planes closing a 2 um box around the "
        "sphere at 20 frequencies each and one DFT field plane (Ex, Ey, Ez at z = 0, "
        "2.4 x 2.4 um) at 5 frequencies: 8 DFT objects against pml_3d's 1",
        base="pml_3d", dft_objects=8),
    "pml_3d_hz": TimingCase(
        "pml_3d_hz", case_pml_3d_hz, (4.0, 4.0, 4.0), 0.8, "pml_3d", "B", True, "Hz",
        "a source on a fused seam: pml_3d with its source an Hz dipole at the same point, "
        "so the deposit lands between step_B and update_H inside fused pair B (B-seam "
        "bracket); the D seam keeps pml_3d's off-diagonal singles",
        base="pml_3d"),
}

#: The cases the extended CERT-8 ladder adds to ``pml_3d``, in the order the catalogs
#: name them: dispersive, monitor-heavy, source on a fused seam.
EXTENDED_LADDER_CASES = ("pml_3d_lorentz", "pml_3d_monitors", "pml_3d_hz")

#: What the built-in cases a ladder may name declare. The builders themselves are
#: the route gate's and are never re-typed; only these declarations live here.
BUILTIN_CASES: Dict[str, TimingCase] = {
    "pml_3d": TimingCase(
        "pml_3d", None, (4.0, 4.0, 4.0), 0.8, None, None, True, "Ez",
        "the 2026-09-28 identical case"),
    "pml_3d_diagonal": TimingCase(
        "pml_3d_diagonal", None, (4.0, 4.0, 4.0), 0.8, None, "D", False, "Ez",
        "both ordinary pairs on a diagonal-epsilon grid"),
}

#: Per-table changes to a new case's copied DRIVE spec, applied after the copy. Empty
#: until a preflight on that table measures that a field the gate reads (``pairs``,
#: ``fills``, ``launches_per_step``, ``collapsed_single_launches_per_step``) differs
#: from the template's; each entry then names the row that measured it (``measured``,
#: kept out of the spec). ``expected_repair`` here overrides the case's declared seam for
#: that table's rows (``None``: no bracket) and is kept out of the spec too.
TABLE_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "triton": {}, "cuda": {}, "metal": {},
}
#: Keys of a :data:`TABLE_OVERRIDES` entry that describe the case on that table and are
#: never copied into the DRIVE spec.
OVERRIDE_META_KEYS = ("expected_repair", "measured")


def declared(name: str) -> TimingCase:
    """The declaration of a case, refused by name when there is none."""
    if name in NEW_CASES:
        return NEW_CASES[name]
    if name in BUILTIN_CASES:
        return BUILTIN_CASES[name]
    known = sorted(NEW_CASES) + sorted(BUILTIN_CASES)
    raise SystemExit(f"REFUSING: {name!r} is not a timing case; the cases with a "
                     f"declared cell are {', '.join(known)}")


def budget_scale(name: str) -> float:
    """The case's time-budget factor (1.0 for every built-in case)."""
    return float(declared(name).budget_scale)


def is_builtin(name: str) -> bool:
    return name not in NEW_CASES


def expected_repair(name: str, table: Optional[str] = None) -> Optional[str]:
    """``None``, ``"B"`` or ``"D"``: the seam the fused leg's bracket must sit on, on
    ``table`` when one is named (a measured :data:`TABLE_OVERRIDES` entry may differ from
    the case's declaration)."""
    case = declared(name)
    override = TABLE_OVERRIDES.get(table or "", {}).get(name, {})
    if "expected_repair" in override:
        return override["expected_repair"]
    return case.expected_repair


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

def counts(name: str, res: int) -> Tuple[int, int, int]:
    """Cells per axis as the comparison counts them: round(extent x resolution)."""
    case = declared(name)
    return tuple(max(1, int(round(extent * float(res)))) for extent in case.cell)  # type: ignore[return-value]


def cells(name: str, res: int) -> int:
    nx, ny, nz = counts(name, res)
    return nx * ny * nz


def res_for_cells(name: str, wanted: int) -> int:
    """The integer resolution at which ``name`` holds exactly ``wanted`` cells.

    Refused by name when none does: a size the two families cannot both hit is not a
    size the comparison can name by cell count.
    """
    case = declared(name)
    volume = 1.0
    for extent in case.cell:
        volume *= extent
    guess = (float(wanted) / volume) ** (1.0 / 3.0)
    for res in sorted({max(1, int(math.floor(guess))), max(1, int(math.ceil(guess))),
                       max(1, int(round(guess)))}):
        if cells(name, res) == int(wanted):
            return res
    raise SystemExit(f"REFUSING: no integer resolution gives {name} exactly "
                     f"{int(wanted):,} cells (its cell is {case.cell} um; resolution "
                     f"{int(math.floor(guess))} gives "
                     f"{cells(name, max(1, int(math.floor(guess)))):,})")


# ---------------------------------------------------------------------------
# Resolving a builder
# ---------------------------------------------------------------------------

def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _sha256_source(function: Any) -> Optional[str]:
    try:
        return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return None


def find_harness_root(start: str) -> Optional[str]:
    """The nearest directory at or above ``start`` holding the harness marker."""
    here = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(here, HARNESS_MARKER)):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def repository_root_of(harness_root: str) -> Optional[str]:
    """The nearest ancestor holding both ``meep_gpu/`` and ``parity/meep_gpu/``.

    The route gate's own rule: the repository root in the release layout, ``apps/api``
    in the development layout, and the same directory in any tree copied whole.
    """
    here = os.path.abspath(harness_root)
    while True:
        if (os.path.isdir(os.path.join(here, "meep_gpu"))
                and os.path.isdir(os.path.join(here, "parity", "meep_gpu"))):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def put_on_path(harness_root: str) -> None:
    """The harness directory and its repository root on ``sys.path``, by name."""
    for path in (harness_root, repository_root_of(harness_root)):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def case_identity(name: str) -> Dict[str, Any]:
    """What a row records about a NEW case: enough to tell two definitions apart."""
    case = declared(name)
    return {
        "name": case.name,
        "module": os.path.abspath(__file__),
        "module_sha256": _sha256_file(os.path.abspath(__file__)),
        "builder": (f"{case.builder.__module__}.{case.builder.__name__}"
                    if case.builder else None),
        "builder_source_sha256": _sha256_source(case.builder) if case.builder else None,
        "template": case.template, "expected_repair": case.expected_repair,
        "cell": list(case.cell), "pml": case.pml,
        "pml_fraction_nominal": round(case.pml_fraction_nominal, 6),
        "off_diagonal_epsilon": case.off_diagonal_epsilon,
        "source_component": case.source_component,
        "base": case.base,
        "base_builder_source_sha256": _base_source(case.base),
        "dft_objects": case.dft_objects,
        "dispersive": case.dispersive,
    }


def _base_source(base: Optional[str]) -> Optional[str]:
    """The source digest of the route-gate builder a case changes, so a row tells two
    base definitions apart as it tells two builders apart."""
    if not base:
        return None
    try:
        return _sha256_source(route_case(base))
    except (SystemExit, ImportError, KeyError):
        return None


#: The route gate whose ``CASES`` a table's GPU rows build from.
TABLE_GATES = {"triton": "gate_dispatch_fused_route", "cuda": "gate_dispatch_fused_route",
               "metal": "gate_dispatch_metal_route"}


def metal_overrides(harness_root: str) -> List[str]:
    """The case names ``gate_dispatch_metal_route.py`` assigns its OWN builder to, read
    from its source (``CASES["name"] = ...``) without importing it: for these names the
    Metal table's GPU rows build a different object from the route gate's."""
    path = os.path.join(harness_root, "gate_dispatch_metal_route.py")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    import re  # noqa: PLC0415
    return sorted(set(re.findall(r'^CASES\["([A-Za-z0-9_]+)"\]\s*=', text, re.M)))


def resolve(name: str, harness_root: Optional[str] = None,
            table: Optional[str] = None) -> Tuple[Builder, Dict[str, Any]]:
    """``(builder, facts)`` for a case name, the one lookup every timing driver makes.

    A name in ``gate_dispatch_fused_route.CASES`` returns THAT object (the route gate's
    table, which for ``pml_3d`` is ``gate_dispatch_end_to_end.case_pml_3d``); a name of
    :data:`NEW_CASES` returns this module's builder. ``facts`` says which, with the
    digests a row needs to show which code built its simulation.

    THE TABLE. The Metal table's GPU rows build from ``gate_dispatch_metal_route.CASES``,
    which inherits the route gate's builders and OVERRIDES some names
    (:func:`metal_overrides`). ``table="metal"`` resolves from it; ``triton``/``cuda``
    from the route gate; ``None`` from the route gate, and a name the Metal gate
    overrides is then REFUSED by name, because which simulation was meant is ambiguous.
    """
    root = os.path.abspath(harness_root) if harness_root else find_harness_root(HERE)
    if not root or not os.path.isfile(os.path.join(root, HARNESS_MARKER)):
        raise SystemExit(f"REFUSING: no {HARNESS_MARKER} under {harness_root!r}; the case "
                         "builders are imported from the harness, never re-typed")
    if table is not None and table not in TABLE_GATES:
        raise SystemExit(f"REFUSING: table {table!r}; one of {', '.join(TABLE_GATES)}")
    if name in NEW_CASES:
        case = NEW_CASES[name]
        facts = {"harness_root": root, "source": "timing_cases", "table": table,
                 "builder_is_end_to_end_case": False, **case_identity(name),
                 "modules": {"timing_cases": {"path": os.path.abspath(__file__),
                                              "sha256": _sha256_file(
                                                  os.path.abspath(__file__))}}}
        return case.builder, facts  # type: ignore[return-value]
    overridden = name in metal_overrides(root)
    if table is None and overridden:
        raise SystemExit(f"REFUSING: gate_dispatch_metal_route.CASES assigns {name!r} a "
                         "builder of its own, so the Metal table's GPU rows may build "
                         "another simulation than gate_dispatch_fused_route.CASES does; "
                         "name the table the rows must match (--table metal, triton or "
                         "cuda)")
    put_on_path(root)
    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415
    import gate_dispatch_fused_route as route  # noqa: PLC0415

    gates = [("gate_dispatch_end_to_end", e2e), ("gate_dispatch_fused_route", route)]
    if table == "metal" and overridden:
        import gate_dispatch_metal_route as metal  # noqa: PLC0415
        cases, source = metal.CASES, "gate_dispatch_metal_route.CASES"
        gates.append(("gate_dispatch_metal_route", metal))
    else:
        cases, source = route.CASES, "gate_dispatch_fused_route.CASES"
    if name not in cases:
        raise SystemExit(f"REFUSING: {name!r} is neither a case of {source} nor of "
                         "timing_cases.NEW_CASES")
    builder = cases[name]
    modules = {}
    for label, module in gates:
        path = os.path.abspath(getattr(module, "__file__", "") or "")
        modules[label] = {"path": path, "sha256": _sha256_file(path)}
    facts = {
        "harness_root": root, "source": source, "table": table,
        "name": name,
        "builder": f"{builder.__module__}.{builder.__name__}",
        "builder_is_end_to_end_case": bool(e2e.CASES.get(name) is builder),
        "builder_source_sha256": _sha256_source(builder),
        "expected_repair": (BUILTIN_CASES[name].expected_repair
                            if name in BUILTIN_CASES else None),
        "modules": modules,
    }
    return builder, facts


# ---------------------------------------------------------------------------
# The GPU spec of a new case
# ---------------------------------------------------------------------------

def drive_spec(table: str, name: str, rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A deep copy of the template's DRIVE row on ``table``, plus ``why`` and overrides.

    ``rows`` is the table's own DRIVE dict as read at run time
    (``bench_fused_products.drive_rows(table)``), so a template the release changes is
    followed without an edit here. The copy shares no object with the table.
    """
    if table not in TABLES:
        raise SystemExit(f"REFUSING: {table!r} is not a DRIVE table ({', '.join(TABLES)})")
    if name not in NEW_CASES:
        raise SystemExit(f"REFUSING: {name!r} is not a new timing case; a built-in case "
                         "is timed by bench_fused_products.py directly")
    case = NEW_CASES[name]
    if case.template not in rows:
        raise SystemExit(f"REFUSING: the template {case.template!r} of {name} is not a "
                         f"DRIVE row of the {table} table")
    spec = copy.deepcopy(rows[case.template])
    spec["why"] = (f"timing case {name} (timing_cases.py), spec copied from the "
                   f"{table} DRIVE row {case.template!r}: {case.isolates}")
    for key, value in TABLE_OVERRIDES.get(table, {}).get(name, {}).items():
        if key in OVERRIDE_META_KEYS:
            continue
        spec[key] = copy.deepcopy(value)
    return spec


def names() -> List[str]:
    """Every name this module declares, built-in first."""
    return sorted(BUILTIN_CASES) + sorted(NEW_CASES)


def describe(case_names: Sequence[str]) -> List[Dict[str, Any]]:
    """The declaration of each named case, as a dry run prints it."""
    out = []
    for name in case_names:
        case = declared(name)
        out.append({"name": name, "builtin": is_builtin(name), "cell": list(case.cell),
                     "pml": case.pml,
                     "pml_fraction_nominal": round(case.pml_fraction_nominal, 3),
                     "template": case.template, "expected_repair": case.expected_repair,
                     "source_component": case.source_component,
                     "base": case.base, "dft_objects": case.dft_objects,
                     "dispersive": case.dispersive,
                     "isolates": case.isolates})
    return out

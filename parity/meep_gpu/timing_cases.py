"""The timing cases: one builder per case, read by the GPU bench and the MEEP bench alike.

WHY THIS MODULE EXISTS. The identical-case comparison holds only if both sides build
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

THE BUILDER CONTRACT is the route gates': ``builder(mp, res) -> (sim, [monitors],
until)``, with ``sim`` not yet initialised.

IMPORTS. This module imports neither MEEP nor ``meep_gpu`` nor a device library at
module scope; a builder receives ``mp`` from its caller, and :func:`resolve` imports the
route gate (which imports neither) only when a built-in name is asked for.
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
}

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
#: from the template's; each entry then names the row that measured it.
TABLE_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "triton": {}, "cuda": {}, "metal": {},
}


def declared(name: str) -> TimingCase:
    """The declaration of a case, refused by name when there is none."""
    if name in NEW_CASES:
        return NEW_CASES[name]
    if name in BUILTIN_CASES:
        return BUILTIN_CASES[name]
    known = sorted(NEW_CASES) + sorted(BUILTIN_CASES)
    raise SystemExit(f"REFUSING: {name!r} is not a timing case; the cases with a "
                     f"declared cell are {', '.join(known)}")


def is_builtin(name: str) -> bool:
    return name not in NEW_CASES


def expected_repair(name: str) -> Optional[str]:
    """``None``, ``"B"`` or ``"D"``: the seam the fused leg's bracket must sit on."""
    return declared(name).expected_repair


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
    }


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
                     "isolates": case.isolates})
    return out

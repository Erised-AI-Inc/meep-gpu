"""SERVED IN DISPATCH: how many of a board's served seam-instances actually fuse
at runtime — measured off the shipped ladder, on every backend, for all three boards.

WHY THIS FILE EXISTS AT ALL. Until 2026-08-29 each of the three fusion boards
answered this question with a constant. The Metal board wrote
``"served_in_dispatch": 0`` as a literal (``build_fusion_matrix.py``); the Triton
board wrote the sentence "nothing dispatches a fused Triton product in production
today" into its caveat list; the CUDA board read a hardcoded ``wired: false``. All
three were TRUE when they were written and all three were assertions, not
measurements — a board that pins the pre-flip answer keeps printing it after the
flip, which is the failure this project ranks worst dressed up as a clean number.
They are now computed here, from the same constants the dispatcher reads, so a
board that says "0" is saying it because the tree still refuses.

THE TWO QUESTIONS, AND THEY ARE DIFFERENT.

  (1) CAN THIS BACKEND REACH THE SEAM AT ALL? A structural fact about
      ``meep_gpu/fastpath.py``, read off its parse tree rather than remembered:
      rung 3 refuses any backend that is not CuPy, rung 4 imports ``triton``, and
      the composer it calls is ``triton_kernels.plan_step``. No Metal or CUDA
      product is reachable from that function — however well gated, however total
      its fill cover. :func:`backend_reaches_the_dispatch_seam` measures it.

  (2) FOR THE BACKEND THAT CAN: WHICH SERVED CELLS DOES THE RELEASE COVER? A
      fused product only dispatches when its arm label is in
      ``fastpath.RELEASED_FUSED_ARMS`` and the row's configuration is inside
      ``fastpath.FUSED_RELEASE_ENVELOPE``. :func:`released_for_row` asks the
      shipped predicate itself — never a re-implementation of it, so a narrowing
      of the envelope moves every board on the next cut without touching them.

IT IS AN UPPER BOUND, and the board that publishes it must say so. Three clauses
below this one can still refuse a cell it counts: the composer's own
``specialized_family_owns_the_grid`` guard, ``_pair_may_absorb`` (the pair may only
absorb slots its constituent arms won), and the plan-time warm pass (a kernel that
will not compile unfills its slot). The census hosts have no Triton, so none of
the three was measurable when the census was cut — the same reason every board's
served count is already declared an upper bound.

A FOURTH CLAUSE EXISTED BETWEEN 2026-09-02's PRODUCT WAVE AND ITS RELEASE ROUND,
and it made this number an OVER-count rather than an upper bound, which is worse.
A fused product occupies both slots of its seam, so a product the ladder could not
admit forced clause (8) to refuse the WHOLE plan — including the released arm
beside it. Measured on the 2026-09-02 board: one row (``conductive_fused_electric_pair``)
took a credited ``fused pair B`` instance down with it, and widening the envelope
to folds would have taken four more (``folded_dispersive_fused_pair``). It is gone
rather than modelled: ``plan_step`` is handed the admitted label set
(``fastpath``'s ``fuse_labels``), so an un-admitted product is never installed and
the seam keeps its separate arms.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from meep_gpu import fastpath
import fusion_taxonomy as _fusion_taxonomy  # the one place seam spellings are reconciled
import h_to_d_seam as _h_to_d_seam  # the fourth seam names its own halves


#: The backend whose kernels the dispatch seam can run. Not a preference and not a
#: roadmap: it is what ``plan_fast_path`` does today, and
#: :func:`backend_reaches_the_dispatch_seam` re-measures it on every board run.
DISPATCHABLE_BACKEND = "triton"

#: The kernel packages a board may be built for, and the module each one would have
#: to appear in ``fastpath``'s composer call for its products to reach the seam.
BACKEND_PACKAGES: Mapping[str, str] = {
    "triton": "triton_kernels",
    "metal": "metal_kernels",
    "cuda": "cuda_kernels",
}


def _fastpath_source() -> str:
    return Path(fastpath.__file__).read_text(encoding="utf-8")


def _imported_names(source: str) -> Tuple[List[str], List[str]]:
    """Every dotted name a module imports, split into kernel packages and siblings.

    AST rather than a substring search: the three package names appear in this
    module's own prose (and in ``fastpath``'s), and a grep would find the sentence
    that says a package is UNREACHABLE and score it as reachable.
    """
    packages: List[str] = []
    siblings: List[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names: List[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and not base:
                # ``from . import metal_dispatch`` — the module names are the
                # siblings, and the relative level is what says they are siblings.
                names = [alias.name for alias in node.names]
            else:
                names = [base] + [f"{base}.{alias.name}" if base else alias.name
                                  for alias in node.names]
        for name in names:
            matched = False
            for package in BACKEND_PACKAGES.values():
                if (name == package or name.endswith(f".{package}")
                        or name.startswith(f"{package}.")):
                    packages.append(package)
                    matched = True
            if not matched and name and "." not in name:
                siblings.append(name)
    return packages, siblings


def _composer_packages(source: str) -> Tuple[str, ...]:
    """Every ``meep_gpu`` kernel package the dispatch seam can reach, transitively.

    ONE LEVEL DEEPER THAN THE FILE, and the depth is the 2026-09-10 correction. The
    Metal ladder does NOT live in ``fastpath.py``: its rungs, its release table and
    its residency install are in ``meep_gpu/metal_dispatch.py``, a sibling
    ``fastpath`` reaches through a function-local import at the backend rung. That
    placement is deliberate — a Metal release-row edit must not re-drift the Triton
    ``driver_dispatch`` record, which pins exactly ``driver.py``, ``fastpath.py``
    and ``fields.py`` — but a reader of ``fastpath.py``'s parse tree alone would
    conclude no Metal product is reachable while one dispatches.

    So the walk follows the ``meep_gpu`` SIBLINGS ``fastpath`` imports and unions
    their kernel packages in. One level, not a full closure: the question is which
    table the dispatch seam composes with, and a table's ladder is one hop from the
    seam by construction. A deeper walk would start scoring a module that merely
    reads a package's ledger as though it could dispatch from it.
    """
    packages, siblings = _imported_names(source)
    found: List[str] = list(packages)
    package_dir = Path(fastpath.__file__).parent
    for name in sorted(set(siblings)):
        sibling = package_dir / f"{name}.py"
        if not sibling.is_file():
            continue
        try:
            text = sibling.read_text(encoding="utf-8")
            nested, _ = _imported_names(text)
        except (OSError, SyntaxError):
            continue
        if not nested or not _calls_a_composer(text):
            continue
        found.extend(nested)
    return tuple(sorted(set(found)))


def _calls_a_composer(source: str) -> bool:
    """Does this module CALL a kernel package's ``plan_step``?

    THE DISCRIMINATOR THE SIBLING WALK NEEDS, and it was measured as necessary
    rather than added for tidiness. ``meep_gpu/subnormal_policy.py`` imports
    ``metal_kernels.subnormal`` — that is the ``mps`` executor arm of the policy
    installer, a lazy import of one function — and ``fastpath`` imports
    ``subnormal_policy``. Counting every package a sibling touches would therefore
    have scored ``metal`` as reaching the dispatch seam on the strength of a POLICY
    import, before the branch that routes to the Metal ladder existed at all: a
    board reporting ``reaches_the_seam: True`` while every Metal product is
    unreachable, which is the pre-flip-answer failure this whole file exists to end,
    inverted.

    What separates a ladder from a policy arm is that a ladder COMPOSES: it calls
    ``plan_step``. Read as a call rather than as a substring so the sentence in a
    docstring saying a module does not compose is not scored as though it did.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (function.attr if isinstance(function, ast.Attribute)
                else function.id if isinstance(function, ast.Name) else None)
        if name == "plan_step":
            return True
    return False


def backend_reaches_the_dispatch_seam(backend: str) -> Dict[str, Any]:
    """Whether a product of ``backend`` can be selected by ``plan_fast_path``.

    MEASURED, BOTH DIRECTIONS. The answer is not "triton is the dispatch backend"
    asserted; it is "``fastpath`` imports exactly these kernel packages, and this
    one is/is not among them" — so the day a second backend is wired in, this
    function says so on its own and every board's headline moves with it.
    """
    package = BACKEND_PACKAGES.get(backend)
    if package is None:
        raise SystemExit(f"unknown backend {backend!r}; expected one of "
                         f"{sorted(BACKEND_PACKAGES)}")
    imported = _composer_packages(_fastpath_source())
    reaches = package in imported
    return {
        "backend": backend,
        "kernel_package": package,
        "reaches_the_seam": reaches,
        "packages_fastpath_imports": list(imported),
        "measured_from": "meep_gpu/fastpath.py, parse tree",
        "why": ("the dispatch seam composes with "
                f"{', '.join(imported) or 'no kernel package'}; a product this "
                "function never imports cannot be selected, whatever its own gates "
                "say" if not reaches else
                "plan_fast_path imports this package and composes its plan_step"),
    }


#: The shape keys :func:`fastpath._run_shape` reports, assembled here from a
#: census ``configuration`` block. Every name on the left is the census's; every
#: name on the right is the one the dispatcher reads.
def run_shape_from_census(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    """A ``_run_shape``-shaped dict for one corpus row, from its census record.

    HOW ``dimensions`` IS OBTAINED, because it is the one axis the census does not
    carry and the one the release turns on: by MEEP's own rule over the lift facts
    the row carries (:func:`_dimensions_from_census`), and by the count of axes with
    extent greater than one only for a row handed over without them. On a
    cylindrical grid MEEP's normalization reports 2 whatever the extents are
    (``grid.py``: CYLINDRICAL sets ``dimensions = 2``). MEASURED against real lifts
    rather than assumed: a 2-D Cartesian lift reports ``dimensions 2`` with
    ``grid_shape (200, 120, 1)``, a 3-D one reports 3 with ``(48, 48, 48)``, a Dcyl
    m=0 one reports 2 with ``(80, 1, 80)``, a 1-D one DECLARED 1-D reports 1 with
    ``(1, 1, 2500)``, and — the two the count got wrong, 2026-09-12 — a
    ``kz_2d="3d"`` cell reports 3 with ``(135, 92, 1)`` and a ``(0, 0, L)`` cell
    declared 3-D reports 3 with ``(1, 1, N)``.

    THE DERIVATION IS CHECKED, NOT TRUSTED. :func:`assert_every_row_shape_is_read`
    refuses a corpus carrying a non-cylindrical row whose non-unit axes are not a
    prefix of (X, Y, Z) — the one pattern under which the extent count could name a
    grid MEEP builds along different axes — and records how many rows carried the
    facts MEEP's rule needs. On the 186-row corpus's 52
    example scripts the patterns were XY (34), cylindrical rz (9), Z (5), X (2),
    XYZ (1) and one all-unit row; on the 60 of the 194-row basis
    (``results/metal_coverage_2026-09-03_complete``) they are XY (37), cylindrical
    rz (11), Z (6), XYZ (3), X (2) and one all-unit row. None is ambiguous, and the
    check runs over whichever census it is handed.
    """
    extents = [int(value) for value in (configuration.get("shape") or ())]
    cylindrical = bool(configuration.get("cylindrical"))
    shape: Dict[str, Any] = {
        "dimensions": _dimensions_from_census(configuration, extents, cylindrical),
        "grid_shape": extents,
        "cylindrical": cylindrical,
        "complex_storage": bool(configuration.get("force_complex_fields")),
        "bloch": bool(configuration.get("has_bloch")),
        "beta": configuration.get("beta") or 0,
        "bfast": bool(configuration.get("bfast_active")),
        "conductivity": bool(configuration.get("has_conductivity")),
        "nonlinearity": bool(configuration.get("has_nonlinearity")),
        "off_diagonal_epsilon": bool(configuration.get("has_offdiagonal_epsilon")),
        "susceptibilities": configuration.get("n_polarizations") or 0,
        "pml_active": bool(configuration.get("pml_active")),
    }
    if configuration.get("has_symmetry") or any(configuration.get("mirrored") or ()):
        mirrored = configuration.get("mirrored") or ()
        axes = [name for name, on in zip("XYZ", mirrored) if on]
        shape["folded"] = ("mirror plane on " + ", ".join(axes) if axes
                           else "a mirror plane is active on an axis this record "
                                "could not name")
    return shape


def _dimensions_from_census(configuration: Mapping[str, Any], extents: Sequence[int],
                            cylindrical: bool) -> int:
    """MEEP's ``dimensions`` for a census row, by MEEP's own rule where the row carries it.

    THE EXTENT COUNT IS NOT THE RULE, and the 2026-09-12 ``complex`` round measured
    the two places it disagrees with what the runtime reads (``fastpath._run_shape``
    reports ``grid.dimensions``, which ``from_meep._effective_dimensions`` transcribes
    from ``Simulation._infer_dimensions``):

    * ``kz_2d="3d"`` — a zero-thickness cell declared 3-D whose ``k_point.z`` is
      nonzero and NOT lifted as a special-kz ``beta``. MEEP does not collapse it
      ("Working in 3D dimensions", one cell in z carrying a Bloch phase), the count
      called it 2-D, and the board credited both folded complex pairs on the one
      corpus row that has the shape (``TestEigCoeffs.test_binary_grating_special_kz_2_21_2``)
      while the release rows refuse it at runtime on ``dimensions=3, not 2``;
    * a ``(0, 0, L)`` cell declared 3-D — MEEP builds it in 3-D with one-cell x and
      y axes; only ``dimensions=1`` builds ``vol1d``. The count called every such row
      1-D. No credit turned on it (the arms that admit 1-D admit 3-D), but a refusal
      reason that names the wrong dimensionality is a reason about another grid.

    So the rule is MEEP's, read from the lift facts the census records beside the
    configuration (``dimensions_attr`` is the declared value, ``cell_size`` the
    declared cell, ``k_point`` and ``beta`` the lifted ones): cylindrical is 2;
    a declared 1 or 2 wins outright; a declared 3 collapses to 2 only when
    ``cell_size.z == 0`` and z carries no Bloch phase — every special-kz spelling
    lifts ``beta != 0`` and collapses, ``kz_2d="3d"`` lifts ``beta == 0`` with
    ``k_point.z != 0`` and does not. A row handed over WITHOUT its facts (an older
    caller) falls back to the extent count, which is what every board read until
    2026-09-13, and :func:`assert_every_row_shape_is_read` records how many rows
    took each route so a board cut cannot mix the two silently.
    """
    if cylindrical:
        return 2
    facts = configuration.get("facts") or {}
    declared = facts.get("dimensions_attr")
    cell = facts.get("cell_size")
    if declared in (1, 2):
        return int(declared)
    if declared == 3 and isinstance(cell, (list, tuple)) and len(cell) == 3:
        wavevector = configuration.get("k_point") or (0.0, 0.0, 0.0)
        beta = configuration.get("beta") or 0
        zero_z = float(cell[2]) == 0.0
        phased_z = float(wavevector[2]) != 0.0 and not beta
        return 2 if zero_z and not phased_z else 3
    return sum(1 for v in extents if v > 1)


def assert_every_row_shape_is_read(configurations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Refuse a corpus whose dimensionality the derivation above cannot name.

    A leg that measures zero must exit non-zero, and a leg that measures the WRONG
    thing must too. The extent count disagrees with MEEP's ``dimensions`` on a
    non-cylindrical grid whose non-unit axes are not a prefix of (X, Y, Z) — e.g.
    an XZ slab, which MEEP would carry as 3-D with one cell in Y while the count
    would call it 2-D and the release would credit it — and, measured 2026-09-12,
    on two prefix shapes too: a ``kz_2d="3d"`` cell and a ``(0, 0, L)`` cell declared
    3-D, both 3-D to MEEP. Those two are decided by MEEP's rule when the row carries
    its lift facts (:func:`_dimensions_from_census`); this check counts how many
    rows carried them (``with_facts``) beside the extent patterns, so a cut over a
    census that lost its facts reads as one rather than as the same board.
    """
    patterns: Dict[str, int] = {}
    ambiguous: List[Dict[str, Any]] = []
    with_facts = 0
    for configuration in configurations:
        extents = [int(value) for value in (configuration.get("shape") or ())]
        live = tuple(v > 1 for v in extents)
        pattern = "".join(axis for axis, on in zip("XYZ", live) if on) or "unit"
        key = ("cyl:" if configuration.get("cylindrical") else "") + pattern
        patterns[key] = patterns.get(key, 0) + 1
        facts = configuration.get("facts") or {}
        if facts.get("dimensions_attr") is not None and facts.get("cell_size") is not None:
            # MEEP's rule decides this row; the extents are not consulted.
            with_facts += 1
            continue
        if configuration.get("cylindrical"):
            continue
        # THE EXTENT COUNT IS TRUSTED ONLY ON THE SHAPES MEEP BUILDS THAT WAY: a
        # 2-D cell is XY, a 3-D one XYZ, and MEEP's ``vol1d`` is Z alone. An XZ or
        # YZ slab is a 3-D grid with one cell on the missing axis, which the count
        # would call 2-D and the release would credit.
        if pattern not in ("X", "XY", "XYZ", "Z", "unit"):
            ambiguous.append({"shape": extents, "live_axes": list(live)})
    if ambiguous:
        raise SystemExit(
            "the dimensionality derivation cannot name these rows: they carry no "
            "lift facts and their non-unit axes are not a shape MEEP builds by "
            f"count (X, XY, XYZ or Z), so the axis count may disagree with MEEP's "
            f"declared dimensions. {ambiguous}")
    return {"patterns": patterns, "ambiguous_rows": 0, "with_facts": with_facts,
            "rows": len(configurations)}


def _release_module(backend: str):
    """The module that owns ``backend``'s release table.

    TWO TABLES, TWO MODULES, AND THAT IS THE ARCHITECTURE RATHER THAN AN ACCIDENT.
    The Triton rows are in ``fastpath.py``; the Metal rows are in
    ``meep_gpu/metal_dispatch.py`` precisely so a Metal release edit does not
    re-drift the Triton ``driver_dispatch`` record. This function is the one place
    that knows which, so every reader below asks the SHIPPED predicate for the
    backend it is scoring rather than a board-local re-implementation of it.
    """
    if backend == "metal":
        from meep_gpu import metal_dispatch  # noqa: PLC0415

        return (metal_dispatch.released_fused_arms_metal,
                metal_dispatch.fused_release_reasons_metal,
                metal_dispatch.fused_release_arm_reasons_metal,
                metal_dispatch.METAL_RELEASED_FUSED_ARMS,
                metal_dispatch.METAL_DRIVER_ROUTE_GATE)
    if backend == "cuda":
        from meep_gpu import fastpath_cuda  # noqa: PLC0415

        return (fastpath_cuda.released_fused_arms,
                fastpath_cuda.fused_release_reasons,
                fastpath_cuda.fused_release_arm_reasons,
                fastpath_cuda.CUDA_RELEASED_FUSED_ARMS,
                fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE)
    return (fastpath.released_fused_arms, fastpath.fused_release_reasons,
            fastpath.fused_release_arm_reasons, fastpath.RELEASED_FUSED_ARMS,
            fastpath.DRIVER_ROUTE_FUSED_GATE)


def cuda_product_arm_labels() -> Mapping[str, str]:
    """Board product -> the NAMESPACED label the merge writes, read off the composer.

    DERIVED, NEVER TYPED, and doubly so: the ``label`` key on each
    ``FUSED_PRODUCTS`` row is compared against the ``label=`` literal its plan
    builder passes (the AST read below), so a row whose label had drifted from its
    builder's is a failure here rather than a board crediting a cell to a name
    nothing writes. The ``"cuda:"`` prefix is added at the same boundary
    ``fastpath_cuda.merge_tables`` adds it, so the board's join and the record's
    ``selected`` are the same string.
    """
    from meep_gpu import fastpath_cuda  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    source = (Path(fastpath.__file__).parent / "cuda_kernels"
              / "fused_pairs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    by_function: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in ("CudaFusedPairPlan",
                                        "CudaFusedTriplePlan")):
                for keyword in sub.keywords:
                    if (keyword.arg == "label"
                            and isinstance(keyword.value, ast.Constant)):
                        by_function.setdefault(node.name, []).append(
                            keyword.value.value)
    out: Dict[str, str] = {}
    for family, product in fused_pairs.FUSED_PRODUCTS.items():
        builder = by_function.get(product["plan"].__name__, [])
        row = product.get("label")
        if len(builder) != 1 or builder[0] != row:
            raise SystemExit(
                f"{family}'s FUSED_PRODUCTS row says label {row!r} and its plan "
                f"builder passes {builder!r}; the board's join would credit a cell "
                "to a name the composer never writes")
        out[family] = fastpath_cuda.namespaced(row)
    return out


def _cuda_installable_labels() -> Mapping[str, str]:
    """The join RESTRICTED to products the composer can actually install.

    THE TWO MAPS ARE DISJOINT BY CONTRACT (a board that found a product in both
    would be told it is installed and not installed at once), so the products
    :func:`cuda_certified_but_not_installed` answers for are taken out here. That is
    not a loss of coverage: their instances are counted in their own bucket with
    their own reason, which is the whole point of the three-way split.
    """
    refused = set(cuda_certified_but_not_installed())
    return {family: label for family, label in cuda_product_arm_labels().items()
            if family not in refused}


def cuda_certified_but_not_installed() -> Mapping[str, str]:
    """The CUDA products the composer REFUSES on every configuration, with the reason.

    TWO KINDS, and each is read from the tree rather than listed here:

    * the seven H->D products, which declare ``INSTALLABLE = False`` in their own
      modules — ``fused_pairs._declared_uninstallable`` quotes the declaration;
    * the six SUPERSEDED products, whose span is strictly contained by a longer one
      that admits wherever they do, so the composer refuses them by name at
      ``_superseded_by_a_longer_span`` or as a neighbouring-seam claimant. Their
      reasons are the release table's own pending text, which says the board serves
      zero instances through them.
    """
    from meep_gpu import fastpath_cuda  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    out: Dict[str, str] = {}
    labels = cuda_product_arm_labels()
    for family, product in fused_pairs.FUSED_PRODUCTS.items():
        declared = fused_pairs._declared_uninstallable(family, product)
        if declared is not None:
            out[family] = declared
            continue
        pending = fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS.get(labels[family])
        if pending and "serves 0 instances" in pending.lower():
            out[family] = pending
    return out


def released_for_row(arm_label: str, configuration: Mapping[str, Any],
                     backend: str = DISPATCHABLE_BACKEND) -> Dict[str, Any]:
    """Whether ``arm_label`` would DISPATCH on this row, asked of the shipped ladder.

    Calls the backend's own ``released_fused_arms``/``fused_release_reasons`` rather
    than re-deciding: the board's number and the dispatcher's decision are then the
    same predicate, and a narrowing of the envelope shows up on the next cut of every
    board without any of them being edited.
    """
    (released_fused_arms, fused_release_reasons, arm_reasons,
     released_table, _gate) = _release_module(backend)
    shape = run_shape_from_census(configuration)
    released = released_fused_arms(shape)
    reasons = fused_release_reasons(shape)
    # THE PREDICATE HAS TWO HALVES SINCE 2026-09-02 AND SO DOES THIS ANSWER. The
    # shared envelope refuses every arm at once; each arm's own axes refuse only it.
    # Reading the shared half alone left an instance refused by the per-arm half in
    # NO bucket at all — the counts stopped summing to the served total, which is
    # the accounting failure this file exists to prevent one level up.
    per_arm = (arm_reasons(shape, arm_label)
               if arm_label in released_table else ())
    return {
        "arm": arm_label,
        "dispatches": arm_label in released,
        "released_here": list(released),
        "outside_the_released_envelope": list(reasons),
        "outside_this_arms_own_cases": list(per_arm),
        "why_not": list(reasons) + [f"{arm_label}: {why}" for why in per_arm],
        "run_shape": shape,
    }


#: Board product name -> the arm label ``TritonStepPlan.selected`` writes when that
#: product wins its two slots. A board names its cells by MODULE; ``fastpath``
#: refuses and releases by ARM LABEL, and the two vocabularies have to be joined
#: somewhere. Here, once, with :func:`assert_the_label_map_covers_the_release`
#: measuring that the join is total in the direction that matters — every released
#: arm must be attributable to a product, or a board would silently report a
#: dispatching cell as not dispatching.
#:
#: A ROW HERE MEANS "THE SHIPPED COMPOSER INSTALLS THIS PRODUCT", and that is now
#: MEASURED rather than declared: :func:`composer_installed_labels` reads
#: ``triton_kernels/launch.py``'s parse tree, and the assertion below requires
#: every label in this map to be one the composer can actually write. A row for a
#: product no composer installs would be a claim that ``selected`` carries a label
#: nothing ever puts there — which is the shape of the join gap this map carried
#: until 2026-09-02, where ``fused_ade_state`` (a real product, but not one this
#: board scores) sat here while the board's E->P product ``fused_ade_chain`` had no
#: row at all and its instances landed in "cannot name".
#:
#: 6 -> 26 ON 2026-09-02 with the installer wave: 23 products routed by
#: ``launch.CERTIFIED_FUSED_PRODUCTS``, plus the two SECOND CELLS the board scores
#: under their own product names against the same module, the same gate and the
#: same label (one launch, two admitted arm pairs — ``launch``'s
#: ``CERTIFIED_FUSED_PAIR_EXTRA_ARMS``).
#:
#: 26 -> 28 ON 2026-09-07: the two cylindrical H->D products, routed with
#: ``INSTALLABLE = False``. The composer's table carries their labels, so the
#: assertion below REQUIRES the join (a label the composer can write and no product
#: maps to is the accounting gap it exists to refuse); what keeps the label out of
#: every slot is ``launch._declared_uninstallable``, reported by name on every row,
#: and ``fastpath.PENDING_DEVICE_GATE_ARMS`` names the flag beside the missing
#: ledger entry. :func:`served_in_dispatch` counts their rows under
#: ``served_by_an_arm_no_gate_released``; ``served_in_dispatch`` itself does not move.
PRODUCT_ARM_LABELS: Mapping[str, str] = {
    "fused_pair_B": "fused pair B",
    "fused_pair_D": "fused pair D",
    "dispersive_fused_pair": "dispersive fused pair",
    "folded_fused_magnetic_pair": "fused pair B (folded)",
    "folded_fused_pair": "fused pair D (folded)",
    # NOT a board product: ``fused_ade_state`` occupies ``update_P`` alone, which no
    # fusion board scores as a seam. The row is kept because the fact it records is
    # true and load-bearing — the ``fuse_ade`` block DOES install it and it IS a
    # fused label — and because dropping it would make the release assertion below
    # weaker without making any number more honest.
    "fused_ade_state": "fused ADE state",
    # The installer wave, 2026-09-02.
    "complex_fused_magnetic_pair": "fused pair B (complex)",
    "complex_fused_electric_pair": "fused pair D (complex)",
    "cylindrical_fused_magnetic_pair": "fused pair B (cylindrical complex)",
    "cylindrical_fused_electric_pair": "fused pair D (cylindrical complex)",
    "cylindrical_real_fused_magnetic_pair": "fused pair B (cylindrical)",
    "cylindrical_real_fused_electric_pair": "fused pair D (cylindrical)",
    "folded_complex_fused_magnetic_pair": "fused pair B (folded complex)",
    # The SECOND CELL of the same launch: one module, one label, two admitted arm
    # pairs, and the board scores each cell under its own product name.
    "folded_complex_fused_magnetic_pair_offdiag": "fused pair B (folded complex)",
    "folded_complex_fused_pair": "fused pair D (folded complex)",
    "folded_beta_complex_fused_magnetic_pair": "fused pair B (folded complex beta)",
    "folded_beta_complex_fused_pair": "fused pair D (folded complex beta)",
    "folded_beta_fused_magnetic_pair": "fused pair B (folded real beta)",
    "folded_beta_fused_electric_pair": "fused pair D (folded real beta)",
    "complex_beta_fused_magnetic_pair": "fused pair B (complex beta)",
    "complex_beta_fused_electric_pair": "fused pair D (complex beta)",
    "beta_fused_magnetic_pair": "fused pair B (real beta)",
    "beta_fused_electric_pair": "fused pair D (real beta)",
    "bfast_fused_magnetic_pair": "fused pair B (BFAST)",
    "bfast_fused_electric_pair": "fused pair D (BFAST)",
    "nonlinear_fused_magnetic_pair": "fused pair B (nonlinear)",
    "folded_dispersive_fused_pair": "fused pair D (folded dispersive)",
    "conductive_fused_electric_pair": "fused pair D (conductive)",
    "complex_conductive_fused_pair": "fused pair D (complex conductive no-PML)",
    "no_pml_fused_electric_pair": "fused pair D (no-PML stored E)",
    "no_pml_fused_electric_pair_lossless": "fused pair D (no-PML stored E)",
    # The two scratch-output off-diagonal welds, 2026-09-15: routed once their shared
    # plan base gained a ``warm`` that never rotates (they were in
    # CERTIFIED_BUT_NOT_INSTALLED for that reason until then).
    "offdiag_fused_electric_pair": "fused pair D (off-diagonal)",
    "folded_offdiag_fused_electric_pair": "fused pair D (folded off-diagonal)",
    # The H->D seam's two cylindrical products, 2026-09-07: routed, labelled, and
    # refused by their own INSTALLABLE = False on every row, so the label is one the
    # composer's tables carry and never write into a slot.
    "cylindrical_real_fused_hd_pair": "fused pair H->D (cylindrical)",
    "cylindrical_fused_hd_pair": "fused pair H->D (cylindrical complex)",
    "complex_fused_hd_pair": "fused pair H->D (complex)",
}

#: Board product name -> why the shipped composer installs NOTHING for it.
#:
#: A THIRD ANSWER, and it exists because the other two were both wrong for these
#: five. "Cannot name" reads as an accounting gap in this file; a row in
#: :data:`PRODUCT_ARM_LABELS` would claim ``selected`` carries a label nothing
#: writes. What is true of all five is that the product is CERTIFIED and the
#: COMPOSER does not install it, and a board should print that sentence rather
#: than either of the other two.
#:
#: The reasons are the composers' own, not this file's summary of them, and each
#: is checkable where it lives.
CERTIFIED_BUT_NOT_INSTALLED: Mapping[str, str] = {
    "fused_ade_chain":
        "an E->P chain: it replaces update_E -> update_P, whose second slot holds "
        "a LIST of per-susceptibility plans rather than a plan and whose first is a "
        "constitutive sub-step rather than a curl, so launch.py's pair seam loop "
        "(CERTIFIED_FUSED_PAIR_SEAMS) has no row for it",
    "complex_fused_ade_chain":
        "the same E->P seam as fused_ade_chain, on complex storage; the seam "
        "objection is the slot protocol and is unchanged by the storage",
    "fused_dispersive_chain":
        "an E->P chain over THREE slots (step_D, update_E, update_P); the same "
        "seam objection as the two above, one slot wider",
    "folded_offdiag_fused_ade_chain":
        "the same E->P seam as fused_ade_chain on the folded off-diagonal "
        "dispersive update_E body (one launch, all three components, one scratch "
        "per driven component); the seam objection is the SLOT PROTOCOL — "
        "update_P holds a LIST of per-susceptibility plans and update_E is a "
        "constitutive slot rather than a curl, so launch.py's pair seam loop "
        "(CERTIFIED_FUSED_PAIR_SEAMS) has no row for it — and that objection is "
        "unchanged by the body; it additionally declares INSTALLABLE = False",
    "fused_hd_pair":
        "the H->D weld (update_H + step_D, the fourth seam). launch.py's pair seam "
        "loop HAS a row for it (CERTIFIED_FUSED_PAIR_SEAMS['update_H']) and "
        "_install_fused_pair already routes that row to withdraw_hoist; what it is "
        "not offered is a LABEL, and on this backend a label plan_step can write must "
        "also appear in fastpath.FUSED_ARM_CONSTITUENTS and in "
        "PENDING_DEVICE_GATE_ARMS until a ledger entry exists. Both are in "
        "meep_gpu/fastpath.py, outside the round that built the product, so it is "
        "absent from CERTIFIED_FUSED_PRODUCTS and no composer installs it; it "
        "additionally declares INSTALLABLE = False",
    "conductive_bfast_fused_hd_pair":
        "the H->D weld on the conductive and BFAST curl tails -- ONE family, TWO "
        "emitted kernels, two board cells. The seam objection is fused_hd_pair's "
        "exactly (a composer label must also live in meep_gpu/fastpath.py), and it "
        "declares INSTALLABLE = False besides",
    "beta_real_fused_hd_pair":
        "the H->D weld on the two REAL beta cells (unfolded and folded), one family "
        "and two emitted kernels; the same label objection as fused_hd_pair, and "
        "INSTALLABLE = False",
    "folded_complex_fused_hd_pair":
        "the H->D weld on a folded complex grid, serving BOTH that cell's arm pairs "
        "with ONE launch (the shipped off-diagonal builders return K1's own plan "
        "class around K1's own kernel); the same label objection, and INSTALLABLE = "
        "False. Its arms are certified under the `keep` float32 subnormal policy "
        "only, so under `flush` the composer would select nothing complex on these "
        "rows in any case",
    "beta_complex_fused_hd_pair":
        "the H->D weld on the two COMPLEX beta cells, one family and two emitted "
        "kernels; the same label objection, INSTALLABLE = False, and the same "
        "keep-only expansion licence as the folded complex product",
    # THE SECOND BOARD CELL OF FOUR FAMILIES ABOVE. The board keys a row PER CELL and
    # three of these families emit two kernels from one transform, so each carries a
    # second cell under a suffixed name while the composer sees ONE family. The reason
    # is the base family's, unchanged -- a label in meep_gpu/fastpath.py that the round
    # which built the product does not own, plus INSTALLABLE = False -- and it is
    # repeated per cell rather than aliased so that a reader of any one cell finds it.
    "beta_complex_fused_hd_pair_unfolded":
        "the unfolded complex-beta cell of beta_complex_fused_hd_pair; same family, "
        "second emitted kernel, same label objection and INSTALLABLE = False",
    "beta_real_fused_hd_pair_folded":
        "the folded real-beta cell of beta_real_fused_hd_pair; same family, second "
        "emitted kernel, same label objection and INSTALLABLE = False",
    "conductive_bfast_fused_hd_pair_bfast":
        "the BFAST cell of conductive_bfast_fused_hd_pair; same family, second emitted "
        "kernel, same label objection and INSTALLABLE = False",
    "folded_complex_fused_hd_pair_offdiag":
        "the folded complex off-diagonal cell of folded_complex_fused_hd_pair, which "
        "ONE kernel serves alongside its plain cell (two admissions, one launch); same "
        "label objection and INSTALLABLE = False",
    "folded_fused_hd_pair":
        "the FOLDED H->D weld (folded update_H + folded step_D), the second product "
        "on the fourth seam and the largest cell this backend's board carries "
        "unbuilt: 78 seam-instances, 75 of them buildable_not_built. The objection is "
        "the plain product's exactly -- a LABEL in meep_gpu/fastpath.py that the "
        "round which built it does not own. Its own gate prices the DRIVEN rows at "
        "loss 55 / tie 19 / gain 0 (the lift leg's arbitration_counts over 74 rows); "
        "a predicate-level reading of the same cell says 67/11, which prices by "
        "admission rather than installation and cannot be a subset of the measured "
        "one. Both agree GAIN is ZERO on every row, and it declares "
        "INSTALLABLE = False for that",
    "nonlinear_fused_hd_pair":
        "an ADMISSION rather than a product: it adds one predicate and one plan "
        "builder and launches fused_hd_pair's own kernel through that family's own "
        "plan class, so the flag _declared_uninstallable reads is fused_hd_pair's "
        "and the entry above is the reason. Listed separately because the board "
        "scores it under its own name",
}


# ---------------------------------------------------------------------------
# The METAL join, derived live rather than typed
# ---------------------------------------------------------------------------
#
# NO TABLE FOR THIS BACKEND, AND THAT IS THE POINT. The Triton maps above are hand
# kept because that composer's labels live in literal dicts inside its ``launch.py``
# and the board's product names differ from the arm families in several places. The
# Metal composer has neither problem: every fused label is REGISTERED by the module
# that owns it (``metal_kernels/arms.py``), the registry carries ``family`` and
# ``label`` and ``is_weld`` on one object, and ``build_fusion_matrix._wiring``
# already REFUSES a board whose product keys are not registered families. So the
# join is the registry, read at cut time, and a family registered tomorrow is
# scored tomorrow instead of being silently dropped.
#
# THE SPLIT BETWEEN THE TWO BUCKETS IS ALSO MEASURED, off the composer's own
# tables rather than off a list of names:
#
#   * a weld family with a ``launch.FUSED_PAIR_ARMS`` row that does NOT declare
#     itself uninstallable is a product ``plan_step`` can install — it belongs in the
#     label map;
#   * a weld family with NO ``FUSED_PAIR_ARMS`` row can never be installed, because
#     the installer has nothing to absorb with (measured 2026-09-10: ten of the 47,
#     including the four E->P chains, whose seam has no row in FUSED_PAIR_SEAMS at
#     all);
#   * a family that declares itself uninstallable through
#     ``launch._declared_uninstallable`` is CERTIFIED AND NOT INSTALLED by its own
#     declaration (measured: the ten H->D products).
#
# 27 + 20 = 47, and the five-bucket floor below is what keeps that arithmetic honest.


def _metal_registry():
    from meep_gpu.metal_kernels import arms, launch  # noqa: PLC0415

    arms.ensure_registered()
    welds = {}
    for spec in arms.registered():
        if spec.is_weld:
            welds.setdefault(spec.family, spec)
    return welds, launch


def metal_product_arm_labels() -> Dict[str, str]:
    """Board product (an arm FAMILY on this backend) -> the label ``selected`` writes."""
    welds, launch = _metal_registry()
    return {family: spec.label for family, spec in sorted(welds.items())
            if family in launch.FUSED_PAIR_ARMS
            and launch._declared_uninstallable(spec) is None}  # noqa: SLF001


def metal_certified_but_not_installed() -> Dict[str, str]:
    """Board product -> why the shipped Metal composer installs NOTHING for it."""
    welds, launch = _metal_registry()
    out: Dict[str, str] = {}
    for family, spec in sorted(welds.items()):
        declared = launch._declared_uninstallable(spec)  # noqa: SLF001
        if declared is not None:
            out[family] = (f"the family declares itself uninstallable and the "
                           f"composer reports it by name on every row: {declared}")
        elif family not in launch.FUSED_PAIR_ARMS:
            # ASKED OF THE PRODUCT'S SLOT, NOT ITS NAME. This read `any(family in row
            # for row in FUSED_PAIR_SEAMS.values())` until 2026-09-17, which compared a
            # FAMILY NAME against that table's VALUES -- and those values are
            # (partner-slot, seam-letter) tuples like ("update_H", "B"). No family name
            # is ever an element of one, so the test was False for every product and the
            # clause was appended to all ten not-installed families. It is true of four:
            # the E->P ADE chains, whose leading slot `update_E` is not a key of
            # FUSED_PAIR_SEAMS at all. A sentence that reads as a structural finding and
            # fires unconditionally is worse than no sentence, because it is the reason
            # a reader would price an E->P chain the same as an H->D product.
            seam = ("; its seam has no row in launch.FUSED_PAIR_SEAMS either, so "
                    "there is no slot pair for an installer to occupy"
                    if spec.slot not in launch.FUSED_PAIR_SEAMS
                    else "")
            out[family] = ("the family registers a weld and has NO "
                           "launch.FUSED_PAIR_ARMS row, so _install_fused_pairs has "
                           "no arm pair to absorb and can never install it" + seam)
    return out


def _launch_source(backend: str = DISPATCHABLE_BACKEND) -> str:
    package = BACKEND_PACKAGES.get(backend, "triton_kernels")
    return (Path(fastpath.__file__).parent / package / "launch.py").read_text(
        encoding="utf-8")


def product_arm_labels(backend: str = DISPATCHABLE_BACKEND) -> Mapping[str, str]:
    """The board-product -> arm-label join for one backend."""
    if backend == "metal":
        return metal_product_arm_labels()
    if backend == "cuda":
        return _cuda_installable_labels()
    return PRODUCT_ARM_LABELS


def certified_but_not_installed(backend: str = DISPATCHABLE_BACKEND) -> Mapping[str, str]:
    if backend == "metal":
        return metal_certified_but_not_installed()
    if backend == "cuda":
        return cuda_certified_but_not_installed()
    return CERTIFIED_BUT_NOT_INSTALLED


def triton_composer_installed_labels() -> Tuple[str, ...]:
    """Every fused arm label ``launch.plan_step`` can write, from its parse tree.

    MEASURED, not declared, for the reason every other reader in this file is: the
    labels live in the composer and the join lives here, and a hand-kept copy of
    the first is a second place to forget an edit. Read as the union of

    * every ``label`` value in ``CERTIFIED_FUSED_PRODUCTS`` and
      ``FOLDED_FUSED_PAIRS`` — the two table-driven installers;
    * every WHOLE string literal in the file that ``fastpath.arm_is_fused``
      recognises — which is how the ``fuse_ade`` block's ``"fused ADE state"`` is
      caught, since it is written inline at the slot rather than in a table.

    F-STRING FRAGMENTS ARE EXCLUDED: the ordinary pairs' labels are built as
    ``f"fused pair {pair_name}"``, whose literal half ``"fused pair "`` matches the
    prefix and is a label no composer writes. They are added back explicitly.
    """
    tree = ast.parse(_launch_source("triton"))
    labels: List[str] = ["fused pair B", "fused pair D"]   # f"fused pair {name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for value in node.values:
                if not isinstance(value, ast.Dict):
                    continue
                for key, item in zip(value.keys, value.values):
                    if (isinstance(key, ast.Constant) and key.value == "label"
                            and isinstance(item, ast.Constant)):
                        labels.append(item.value)
    fragments = {id(node) for parent in ast.walk(tree)
                 if isinstance(parent, ast.JoinedStr)
                 for node in ast.walk(parent) if isinstance(node, ast.Constant)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in fragments and fastpath.arm_is_fused(node.value)):
            labels.append(node.value)
    return tuple(sorted(set(labels)))


def composer_installed_labels(backend: str = DISPATCHABLE_BACKEND) -> Tuple[str, ...]:
    """Every fused arm label ``backend``'s composer can write into a slot.

    TWO READINGS FOR TWO COMPOSERS, and neither is a copy of a table. Triton's
    labels live in literal dicts inside its ``launch.py`` and are read off that
    file's parse tree. The Metal composer registers its labels instead, so the
    registry IS the reading — and the installable half of it is the half that can
    reach a slot, which is what this function is asked for.
    """
    if backend == "metal":
        return tuple(sorted(metal_product_arm_labels().values()))
    if backend == "cuda":
        # THE CUDA READING IS THE PRODUCT TABLE ITSELF, which is the composer's own
        # declaration of what it can install: ``install_fused_pairs`` walks
        # ``FUSED_PRODUCTS`` and writes each row's builder label. Reading the labels
        # off that walk is the same measurement Triton's parse-tree read makes, one
        # level less indirect because this composer has no inline label.
        return tuple(sorted(_cuda_installable_labels().values()))
    return triton_composer_installed_labels()


def assert_the_label_map_covers_the_release(
        backend: str = DISPATCHABLE_BACKEND) -> Dict[str, Any]:
    """Every released arm is attributable to a product, and every label is fused.

    BOTH DIRECTIONS. A released arm missing from the map would be counted as not
    dispatching (an under-count that reads as caution and is a wrong number); a
    label the dispatcher does not recognise as fused would be counted as
    dispatching when clause (8) never even looks at it.

    AND A THIRD, ADDED WITH THE 2026-09-02 INSTALLER WAVE: every label in the map
    has to be one the SHIPPED COMPOSER can write. Before the wave the map carried a
    row for a product no board scores while the board's own E->P product had none,
    and nothing measured either fact. It does now, against ``launch.py``'s parse
    tree, so a product wired without a join entry and a join entry for a product
    nothing wires both fail here.
    """
    products = product_arm_labels(backend)
    not_installed = certified_but_not_installed(backend)
    _released, _reasons, _arm_reasons, released_table, _gate = _release_module(backend)
    # WHICH PREDICATE ANSWERS "IS THIS LABEL FUSED" IS PER BACKEND. ``arm_is_fused``
    # is the TRITON spelling — a "fused pair" prefix plus two names — and it answers
    # False for every Metal label ("fused magnetic B/H pair", "folded fused B/H
    # pair", ...). The Metal ladder reads the registry's own ``is_weld`` flag
    # instead, so this asks the registry there rather than widening one predicate to
    # answer for two vocabularies.
    if backend == "metal":
        welds, _launch = _metal_registry()
        fused_labels = {spec.label for spec in welds.values()}
        def _is_fused(label: str) -> bool:
            return label in fused_labels
    else:
        # ``arm_is_fused`` ANSWERS FOR BOTH NVIDIA TABLES since the hand-CUDA one
        # landed: a ``"cuda:"`` label is looked up in the typed
        # ``fastpath_cuda.CUDA_FUSED_LABELS`` (pinned to the composer's AST) and a
        # bare one takes the Triton prefix rule. So this branch is one predicate for
        # two vocabularies because the NAMESPACE, not a widened spelling, is what
        # keeps them apart.
        _is_fused = fastpath.arm_is_fused

    missing = sorted(set(released_table) - set(products.values()))
    if missing:
        raise SystemExit(
            f"[{backend}] these released fused arms map to no product: {missing}. A "
            "board cannot attribute a dispatching cell to a module it cannot name.")
    unrecognised = sorted(label for label in products.values()
                          if not _is_fused(label))
    if unrecognised:
        raise SystemExit(
            f"[{backend}] these labels are not recognised as fused: {unrecognised}; "
            "the no-fusion-except-by-name clause would never consult them.")
    installed = set(composer_installed_labels(backend))
    unwritable = sorted(set(products.values()) - installed)
    if unwritable:
        raise SystemExit(
            f"[{backend}] these labels are joined to a product but no composer "
            f"writes them: {unwritable}. A row here claims the step plan's "
            "`selected` carries the label.")
    unjoined = sorted(installed - set(products.values()))
    if unjoined:
        raise SystemExit(
            f"[{backend}] the composer writes these labels and no product maps to "
            f"them: {unjoined}. A board scoring that product would report it as not "
            "dispatching, whatever the release says.")
    both = sorted(set(products) & set(not_installed))
    if both:
        raise SystemExit(
            f"[{backend}] these products are declared both installed and not "
            f"installed: {both}")
    return {"backend": backend,
            "products": dict(products),
            "certified_but_not_installed": dict(not_installed),
            "composer_installed_labels": list(composer_installed_labels(backend)),
            "released_arms": sorted(released_table)}


def arm_label_for_product(product: str,
                          backend: str = DISPATCHABLE_BACKEND) -> Optional[str]:
    """The arm label a board's product writes, or None when this file cannot name it.

    ``None`` is NOT "not released" and — since 2026-09-02 — it is not "certified but
    not installed" either: :data:`CERTIFIED_BUT_NOT_INSTALLED` answers that case by
    name and :func:`served_in_dispatch` counts it in its own bucket. What is left
    for ``None`` is a product neither map has heard of, which is an accounting gap
    in THIS file and is reported as one.
    """
    return product_arm_labels(backend).get(product)


#: WHICH TWO SLOTS EACH SEAM SPANS, over the driver's step_B -> update_H -> step_D ->
#: update_E -> update_P path. Built from ``h_to_d_seam``'s own constants for the fourth
#: seam rather than a second spelling of them, the same join
#: ``build_cuda_fusion_matrix.SPANNABLE_SEAM_SLOTS`` makes.
#:
#: H_to_D IS THE ONLY INTERIOR SPAN, and that is the whole reason this table exists
#: here. B_to_H and E_to_P sit on the end edges and D_to_E's slots are its own, so on
#: today's tables no two dispatching products ever want the same slot. H_to_D takes
#: ``update_H`` from the B seam and ``step_D`` from the D seam -- one slot from EACH
#: neighbour -- so the moment an H->D arm is released, a row can have two products
#: counted where the composer will install one.
SEAM_SLOTS: Mapping[str, Tuple[str, ...]] = {
    "B_to_H": ("step_B", "update_H"),
    "D_to_E": ("step_D", "update_E"),
    "E_to_P": ("update_E", "update_P"),
    _h_to_d_seam.SEAM: tuple(_h_to_d_seam.HALVES),
}


def _slot_contention(counted: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Which counted instances want a slot another counted instance already wants.

    THE CLAUSE THIS EVALUATES, and until 2026-09-11 it was declared and not run:
    ``_pair_may_absorb`` (``<backend>_kernels/launch.py``) lets a fused pair take a
    slot only when the arm table already gave that slot to the arm the fused kernel
    implements, so a pair whose slot another product won is NOT installed. This count
    used to say so in ``clauses_below_this_one`` and then credit the instance anyway.

    WHY IT COST NOTHING UNTIL IT WOULD COST EVERYTHING. ``H_to_D`` is the only span
    that reaches into a neighbour's slots, and no H->D arm is released on any backend,
    so every counted instance today owns its slots outright and this function returns
    an empty contention set -- which is the assertion
    ``test_dispatch_reachability.py`` pins. Release one H->D arm and the same count
    would credit 173 instances per backend against at most 244 possible installs
    across all three, because the clause was never evaluated.

    WHAT IT IS NOT. This is the slot-contention HALF of ``_pair_may_absorb``: it
    refuses a credit where two counted products want one slot on one row. It does not
    evaluate which arm the arm table actually selected for a slot -- that needs the
    composer and the row's own arm resolution, and the remaining clause says so.
    """
    by_row: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    unknown: Dict[str, int] = {}
    for row_label, seam, product in counted:
        # THE BOARDS DO NOT SHARE A SEAM SPELLING, and discovering that by examining
        # zero rows while reporting evaluated: True is the vacuous-check failure this
        # whole floor exists to stop. Triton writes "B->H"/"D->E", the other two
        # "B_to_H"/"D_to_E"; ``fusion_taxonomy.canonical_seam`` is the one place that
        # knows, so it is asked rather than a second map being written here. A seam it
        # cannot name is COUNTED AS UNKNOWN and refuses below.
        canonical = _fusion_taxonomy.canonical_seam(seam)
        if canonical not in SEAM_SLOTS:
            unknown[seam] = unknown.get(seam, 0) + 1
            continue
        for slot in SEAM_SLOTS[canonical]:
            by_row.setdefault(row_label, {}).setdefault(slot, {}).setdefault(
                product, []).append(seam)
    contended: List[Dict[str, Any]] = []
    spanning: List[Dict[str, Any]] = []
    for row_label, slots in sorted(by_row.items()):
        for slot, claimants in sorted(slots.items()):
            if len(claimants) > 1:
                contended.append({"row": row_label, "slot": slot,
                                  "claimants": sorted(claimants)})
                continue
            # ONE PRODUCT UNDER TWO SEAMS IS NOT CONTENTION, it is a weld that spans
            # them. ``cuda_three_slot_dispersive_weld`` is the shipped example: it
            # takes step_D -> update_E -> update_P in ONE launch and is credited one
            # seam-instance at D_to_E and one at E_to_P, so ``update_E`` carries its
            # name twice. Keying contention on the SEAM rather than the PRODUCT called
            # that a double-claim and refused four correct CUDA rows -- measured
            # 2026-09-11 on examples:stochastic_emitter.py and three siblings. The
            # spans are recorded rather than dropped, because a reader counting
            # launches from this number needs to know two instances shared one.
            product, seams = next(iter(claimants.items()))
            if len(set(seams)) > 1:
                spanning.append({"row": row_label, "slot": slot,
                                 "product": product, "seams": sorted(set(seams))})
    return {"rows_examined": len(by_row), "contended_slots": contended,
            "slots_one_product_spans_under_two_seams": spanning,
            "instances_whose_seam_this_table_cannot_name": dict(sorted(unknown.items()))}


#: The guard the first clause below is about. Named once, and its PRESENCE is READ
#: rather than assumed, because it is not a property every backend has.
_SPECIALIZED_FAMILY_GUARD = "specialized_family_owns_the_grid"


def _specialized_family_guard_clause(backend: str) -> str:
    """What this count does not evaluate about that guard — ON THIS BACKEND.

    UNTIL 2026-09-11 THIS SENTENCE NAMED A FILE THAT DOES NOT CONTAIN THE GUARD. It
    was one string with the backend's own package interpolated into it, so the CUDA
    board published "the composer's specialized_family_owns_the_grid guard
    (cuda_kernels/launch.py) refuses fusion on a grid a specialized family owns" and
    the Metal board said the same of ``metal_kernels/launch.py`` — while the guard
    exists only in ``triton_kernels/launch.py``. Measured by grep across all three
    packages. The same false clause had also been copied into
    ``metal_dispatch.METAL_FUSED_RELEASE_ENVELOPE``'s off-diagonal refusal, where it
    turned a coverage boundary into a structural one and told a reader not to bother
    driving the case.

    A clause about a guard is worth nothing if it cannot say whether the guard is
    there, so this reads the package and says which. An over-declared clause is not
    harmless: it makes the count look MORE conservative than it is, and the next
    person to ask "what is this number missing?" is handed a mechanism that does not
    exist on two of the three backends.
    """
    package = BACKEND_PACKAGES.get(backend)
    # The same anchor every other reader in this file uses (see _composer_source and
    # the launch.py read at :701), rather than a second root constant.
    module = ((Path(fastpath.__file__).parent / package / "launch.py")
              if package else None)
    present = bool(module and module.is_file()
                   and _SPECIALIZED_FAMILY_GUARD in module.read_text(encoding="utf-8"))
    if present:
        return (f"the composer's {_SPECIALIZED_FAMILY_GUARD} guard "
                f"({package}/launch.py) refuses fusion on a grid a specialized family "
                f"owns, which this count does not evaluate")
    return (f"NOT A CLAUSE ON THIS BACKEND: {backend} has no "
            f"{_SPECIALIZED_FAMILY_GUARD} guard — it exists only in "
            f"triton_kernels/launch.py, and this sentence used to name "
            f"{package}/launch.py on every backend, which over-declared what the "
            f"count was missing. Read here rather than interpolated, so the day the "
            f"guard is ported the clause returns by itself")


def served_in_dispatch(backend: str,
                       instances: Sequence[Tuple[str, str, Mapping[str, Any]]],
                       ) -> Dict[str, Any]:
    """The headline number, per backend.

    ``instances`` is one ``(row_label, PRODUCT, configuration)`` triple per SERVED
    seam-instance the board counted — the board decides what "served" means, this
    function decides what "dispatches" means, and the two stay separable.

    THE SECOND ELEMENT IS THE PRODUCT, NOT THE ARM LABEL, since 2026-09-02, and the
    change is what lets the third answer exist. A board used to call
    :func:`arm_label_for_product` itself and hand the result over, so every product
    this file could not join arrived here as ``None`` — a certified product nobody
    routes and a product nobody has heard of, indistinguishable. Doing the join
    HERE keeps the vocabulary in one place and lets the two be counted apart.

    A backend that cannot reach the seam answers 0 WITH ITS REASON, not a bare
    zero: the difference between "no product of mine is reachable" and "my products
    are reachable and none is released" is the whole content of the number.
    """
    reach = backend_reaches_the_dispatch_seam(backend)
    result: Dict[str, Any] = {
        "backend": backend,
        "reachability": reach,
        "served_counted": len(instances),
        "served_in_dispatch": 0,
        "arms": {},
        "rows": 0,
        "refused_by_envelope": {},
        "is_an_upper_bound": True,
        "clauses_below_this_one": [
            _specialized_family_guard_clause(backend),
            "_pair_may_absorb: a pair may only absorb slots its constituent arms "
            "actually won. THE SLOT-CONTENTION HALF IS EVALUATED since 2026-09-11 "
            "when the board passes each instance's seam (see result.slot_contention): "
            "two counted products may not claim one slot on one row. What remains "
            "below this count is the other half -- WHICH arm the arm table selected "
            "for a slot, which needs the composer and the row's own arm resolution",
            "the plan-time warm pass: a kernel that will not compile unfills its "
            "slot, and no census host had the toolchain to compile on. On the METAL "
            "table there is no warm pass at all — every kernel is compiled inside "
            "plan_step's builders, so a source that will not compile has already "
            "refused its slot — but the census host's own refusals are unmeasured "
            "either way",
        ],
    }
    if not reach["reaches_the_seam"]:
        result["why_zero"] = reach["why"]
        return result

    result["label_map"] = assert_the_label_map_covers_the_release(backend)
    arms: Dict[str, int] = {}
    refused: Dict[str, int] = {}
    refused_detail: Dict[str, List[Dict[str, Any]]] = {}
    unreleased_arm = 0
    unnamed_product = 0
    not_installed: Dict[str, int] = {}
    rows: set = set()
    (_released, _reasons, _arm_reasons, released_table,
     route_gate) = _release_module(backend)
    not_installed_reasons = certified_but_not_installed(backend)
    counted: List[Tuple[str, str, str]] = []
    seams_given = 0
    for instance in instances:
        # THE SEAM IS OPTIONAL AND ITS ABSENCE IS RECORDED, never assumed. A board that
        # still hands three-tuples gets the old count and an honest
        # ``slot_contention.evaluated: False`` beside it, so the clause stays declared
        # rather than silently dropped for that backend.
        if len(instance) == 4:
            row_label, product, configuration, seam = instance
            seams_given += 1
        else:
            row_label, product, configuration = instance
            seam = None
        arm_label = arm_label_for_product(product, backend)
        if arm_label is None:
            # THREE ANSWERS, NOT TWO, and the split is the 2026-09-02 correction.
            # A certified product the composer does not install is a DECISION with
            # a reason (:data:`CERTIFIED_BUT_NOT_INSTALLED`); a product neither map
            # has heard of is an accounting gap in this file. Folding the first into
            # the second is what let ``fused_ade_chain``'s instances read as "cannot
            # name" while the tree knew exactly what it was.
            if product in not_installed_reasons:
                not_installed[product] = not_installed.get(product, 0) + 1
            else:
                unnamed_product += 1
            continue
        verdict = released_for_row(arm_label, configuration, backend)
        if verdict["dispatches"]:
            arms[arm_label] = arms.get(arm_label, 0) + 1
            rows.add(row_label)
            if seam is not None:
                counted.append((row_label, seam, product))
        elif arm_label not in released_table:
            unreleased_arm += 1
        else:
            # ONE BUCKET PER INSTANCE, not one per reason, and the change matters
            # for the arithmetic rather than for tidiness: a shape outside two axes
            # used to be counted twice, so the refusal buckets could exceed the
            # served total while an instance refused by the per-arm half was counted
            # zero times. The FIRST reason names the instance; the whole list stays
            # in ``refused_reasons`` for a reader.
            why = verdict["why_not"]
            axis = why[0].split(";")[0] if why else "no reason was given"
            refused[axis] = refused.get(axis, 0) + 1
            refused_detail.setdefault(axis, []).append(
                {"row": row_label, "arm": arm_label, "reasons": why})
    result["served_by_an_arm_no_gate_released"] = unreleased_arm
    result["served_by_a_certified_product_the_composer_does_not_install"] = dict(
        sorted(not_installed.items(), key=lambda item: (-item[1], item[0])))
    result["not_installed_total"] = sum(not_installed.values())
    result["why_not_installed"] = {name: not_installed_reasons[name]
                                   for name in sorted(not_installed)}
    result["served_by_a_product_this_file_cannot_name"] = unnamed_product
    # THE SLOT-CONTENTION HALF OF ``_pair_may_absorb``, evaluated rather than declared
    # whenever the board hands the seam. See :func:`_slot_contention`.
    evaluated = seams_given == len(instances) and bool(instances)
    contention = _slot_contention(counted) if evaluated else {}
    if evaluated and contention.get("instances_whose_seam_this_table_cannot_name"):
        raise SystemExit(
            f"{backend}: "
            f"{contention['instances_whose_seam_this_table_cannot_name']} — this table "
            f"could not name the seam of instances counted as dispatching, so the "
            f"_pair_may_absorb clause would have been reported as evaluated over a "
            f"population it never looked at. Add the spelling to "
            f"fusion_taxonomy._CANONICAL_SEAM, or the seam to "
            f"dispatch_reachability.SEAM_SLOTS.")
    if evaluated and sum(arms.values()) and not contention.get("rows_examined"):
        # ONLY WHEN SOMETHING WAS COUNTED. A backend that dispatches nothing examines
        # no rows and that is the right answer, not a vacuous check -- the first
        # version of this guard refused it and turned a correct zero into a crash,
        # which is the reverse defect of the one it was written for.
        raise SystemExit(
            f"{backend}: the slot-contention check examined 0 rows while "
            f"{sum(arms.values())} instances were counted as dispatching. An "
            f"evaluated clause that looked at nothing is worse than a declared one.")
    result["slot_contention"] = {
        "evaluated": evaluated,
        "why_not": None if evaluated else (
            f"{seams_given} of {len(instances)} instances carried a seam; the board "
            f"must pass (row, product, configuration, seam) for the "
            f"_pair_may_absorb clause to be evaluated rather than declared"),
        **contention}
    if evaluated and contention.get("contended_slots"):
        # A CONTENDED SLOT IS NOT SILENTLY DROPPED. Refusing here rather than
        # subtracting: which of two claimants the composer installs is the arm
        # table's answer, not this file's, and publishing a total that guessed
        # would be the over-credit in the other direction.
        raise SystemExit(
            f"{backend}: {len(contention['contended_slots'])} slot(s) are claimed by "
            f"more than one instance counted as dispatching — "
            f"{contention['contended_slots'][:4]}. _pair_may_absorb installs at most "
            f"one of them, so this count would credit an install that cannot happen. "
            f"Teach this function which arm won the slot before publishing a number "
            f"over a table where two products span one slot.")
    result["served_in_dispatch"] = sum(arms.values())
    result["arms"] = dict(sorted(arms.items()))
    result["rows"] = len(rows)
    result["refused_by_envelope"] = dict(
        sorted(refused.items(), key=lambda item: (-item[1], item[0])))
    result["refused_reasons"] = {axis: rows[:4] for axis, rows
                                 in sorted(refused_detail.items())}
    # THE BUCKETS MUST TOTAL THE SERVED COUNT, asserted rather than hoped. Every
    # served instance lands in exactly one of five: it dispatches, its arm is
    # released nowhere, its product is certified and not installed, this file
    # cannot name its product, or the release declines this row. A split that
    # stopped summing is how a board publishes a number nobody can reconstruct.
    parts = (result["served_in_dispatch"], unreleased_arm, sum(not_installed.values()),
             unnamed_product, sum(refused.values()))
    result["buckets_total_the_served_count"] = {
        "served_counted": len(instances), "parts": list(parts),
        "sum": sum(parts), "agrees": sum(parts) == len(instances)}
    if sum(parts) != len(instances):
        raise SystemExit(
            f"the dispatch buckets sum to {sum(parts)} and the board counted "
            f"{len(instances)} served instances; {parts} — a served instance is in "
            "no bucket or in two")
    result["released_arms"] = sorted(released_table)
    result["driver_route_gate"] = route_gate
    result["dispatch_by_default"] = fastpath.DISPATCH_BY_DEFAULT
    result["requires"] = dispatch_condition()
    if backend == "cuda":
        # EVERY CUDA DISPATCH NUMBER CARRIES ITS CONDITION IN THE SAME SENTENCE.
        # Under the SHIPPED precedence Triton composes first and holds every slot
        # both tables admit, so this count is what dispatches when the run asks for
        # the hand-CUDA table by name — or when the host has no validated Triton at
        # all. Published without that clause it would read as a default.
        result["requires"] += (
            f"; and {fastpath.BACKEND_PREFERENCE_SWITCH}=cuda (or a CuPy host with "
            "no validated Triton), because BACKEND_PRECEDENCE puts the "
            "release-gated Triton table first and a secondary unit may not take a "
            "slot the primary holds")
        result["counted_under"] = "by_preference"
        result["by_default_precedence"] = {
            "served_in_dispatch": 0,
            "why": ("no released CUDA product installs on any measured route case "
                    "under the shipped precedence: Triton's arms hold every slot, "
                    "and fastpath_cuda.YIELD_PENDING_PRIMARY_SLOTS is "
                    f"{fastpath_cuda_yield()} so a pending primary arm is not "
                    "yielded either"),
        }
    return result


def dispatch_condition() -> str:
    """The enable condition every dispatch count on a board is counted under.

    READ OFF ``fastpath.DISPATCH_BY_DEFAULT`` on every cut, so a board states the
    condition of the tree it was cut from. The sentence used to be typed — "the
    enable is still opt-in" — in this module and in both builders, and a flip of
    the constant would have left three boards describing a default that no longer
    shipped. The builders quote this function rather than restating it.
    """
    enable = fastpath.DISPATCH_ENABLE
    if fastpath.DISPATCH_BY_DEFAULT:
        # NOT "EVERY RUN": the default reaches a kernel only on a host and toolchain
        # a table's ledger certifies, and every other host takes the array path by a
        # named refusal, so the count is an upper bound on what a default run gets.
        return (f"nothing — dispatch is on by default ({enable} unset), so these "
                "dispatch for a run on a certified host and toolchain that does not "
                f"opt out with {enable}=0; an upper bound")
    return (f"{enable}=1 — the enable is still opt-in, so these dispatch for a run "
            "that asked for dispatch, not for every run")


def fastpath_cuda_yield() -> bool:
    """The shipped value of the yield clause, read rather than transcribed."""
    from meep_gpu import fastpath_cuda  # noqa: PLC0415

    return fastpath_cuda.YIELD_PENDING_PRIMARY_SLOTS

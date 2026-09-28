"""Sub-step byte-identity gate for the CURL HALVES of a chi2/chi3 run — a NULL.

THE CENSUS REFUSAL THIS ANSWERS, quoted from ``coverage.covers_real_pml_curl``::

    instantaneous chi2/chi3: a Pade factor on the constitutive product

It fires on BOTH curl sub-steps for the two corpus rows that carry an
instantaneous nonlinearity — ``3rd-harm-1d.py`` and
``Test3rdHarm1d.test_3rd_harm_1d`` — and it costs FOUR SLOTS at the census
(``step_B`` and ``step_D`` on each row; their ``update_H``/``update_E`` are
already served by ``nonlinear_constitutive.covers_nonlinear_pml_constitutive``).

READ AS A SENTENCE THE REFUSAL IS ABOUT A CONSTITUTIVE PRODUCT, and neither curl
sub-step forms one. ``stepping.step_B`` (stepping.py:261-376) and ``step_D``
(:379-457) difference the STORED E and H arrays; the Pade factor
(:func:`stepping.calc_nonlinear_u`) is applied where E is recovered from D, which
is ``update_E`` and nowhere else. So the clause looks INHERITED — the same shape
the 2026-08-20 nonlinear round found on ``update_H``, where the refusal was
retired after the certified H kernel came back byte-exact on grids carrying a
live chi and the premise was armed as a defect and caught.

THAT IS A READING. This gate is the measurement, and it has three legs the
reading does not:

=============================================================================
LEG 1 — THE STATIC READING, PERFORMED RATHER THAN QUOTED
=============================================================================

:func:`chi_reachability` parses ``meep_gpu/stepping.py`` with ``ast``, builds the
module-level call graph, and reports:

* every function whose body NAMES a chi2/chi3/nonlinear symbol;
* the transitive call closure of ``step_B`` and of ``step_D``;
* the intersection of the two.

An empty intersection is a release clause. SO IS A NON-EMPTY CONTROL: the same
scan must find the chi names inside ``update_E``'s closure, or it is a scan that
finds nothing anywhere and its silence on the curl says nothing. A scan whose
positive control failed would pass every negative it was ever asked, which is
exactly the shape of evidence this suite exists to refuse.

The scan is a fact about the ARRAY PATH — the oracle — not about the kernel. It
says the oracle's curl cannot see a chi; leg 2 says the kernel reproduces that
oracle bit for bit on a grid that has one.

=============================================================================
LEG 2 — THE SHIPPED CURL PAIR, UNCHANGED, ON GRIDS CARRYING A LIVE chi
=============================================================================

``step_curl_kernels._step_B_fused_pml_real`` / ``_step_D_fused_pml_real``, not
one byte changed, compared as raw uint32 against ``stepping.step_B`` /
``step_D`` from ONE frozen state, at one launch and at 60, under both float32
subnormal policies, at both courants and in both value classes.

=============================================================================
LEG 3 — THE PREMISE, ARMED
=============================================================================

``chi_pade_scale_the_curl`` rewrites the SHIPPED curl device string so that the
curl IS scaled by a quotient of the Pade factor's shape. It must be CAUGHT on
every leg. Without it, a 100%% pass rate on leg 2 is consistent with a comparator
that sees nothing, and "chi does not reach the curl" would rest on a leg nobody
showed could fail. It is the exact counterpart of the nonlinear round's
``pade_scale_the_H_source``, and it is spelled against this kernel's own text
rather than borrowed, because the constitutive needle does not appear here.

The rest of the certified curl battery runs beside it — the same shared
mutations the folded-curl round scored — so the comparator is shown to bite on
THESE grids and not only on the ones the certified record was cut against, plus
the two NULLS a battery needs if it is to distinguish a working comparator from
one that has degenerated into failing everything.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

A NULL GATE'S CHARACTERISTIC FAILURE IS A FIXTURE THAT NEVER CARRIED THE THING
IT CLAIMS DOES NOT MATTER. A grid whose chi is installed but inert would pass
leg 2 for a reason about the fixture, and the artifact would read exactly like a
measurement. So:

* ``chi_installed`` — ``fields.has_nonlinearity`` and the component list. A run
  where MEEP's both-or-neither rule deleted the pair carries no chi at all.
* ``chi_expansion`` — ``max(|c2| + |c3|)`` from ``stepping.nonlinear_margin``
  (stepping.py:1344-1386). Must be strictly inside ``(0, 1/3)``: zero is no live
  nonlinearity, and past 1/3 the run has left the domain
  ``driver._NonlinearityGuard`` admits, so the case would be certifying
  arithmetic the engine refuses to perform.
* ``chi_moves_update_E`` — THE DIFFERENTIAL FLOOR, and the one that makes this a
  measurement rather than a definition. Two triples are built from IDENTICAL rng
  streams, one with the chi and one without, seeded with the same host words, and
  ``stepping.update_E`` is run once on each. The number of differing output words
  is recorded. If it is zero, this grid's chi changes nothing anywhere in the
  step and the case is a linear grid wearing a nonlinear label — refused, not
  passed.
* ``oracle_moved`` — the fraction of output words the array path changed from the
  frozen input. Zero-init is a fixed point of this recurrence.
* ``absorber_departure`` — ``max|kms-1|`` / ``max|sinv-1|``. An identity profile
  cannot distinguish a coefficient-index error.

=============================================================================
WHAT THIS GATE DOES NOT MEASURE
=============================================================================

Nothing here touches ``update_E``. The nonlinear CONSTITUTIVE kernel is a
different family with its own gate (``gate_cuda_nonlinear.py``) and its own
predicate (``nonlinear_constitutive.covers_nonlinear_pml_constitutive``); this
gate neither strengthens nor weakens it. Nothing dispatches either: no module in
``meep_gpu`` imports ``cuda_kernels``, so a widened predicate licenses a
MEASUREMENT and not a production step.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=3 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_nonlinear_curl_null.py --subnormal-policy keep \\
        --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors, the same static scan and the same HOST mutations against a
transcription of the shipped device tree. IT COMPILES NOTHING AND CERTIFIES
NOTHING::

    python -u gate_cuda_nonlinear_curl_null.py --backend numpy \\
        --out /tmp/nl_curl_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the transcription and harness legs still run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
#: THE CURL MACHINERY IS THE FOLDED GATE'S, imported rather than respelled. A
#: second transcription of the device tree, the code triple, the table views and
#: the auxiliary bookkeeping is a second thing that must be kept in step with the
#: kernel forever, and the sibling gates' own notes record what a drifted copy
#: costs (a leg reporting a pass for bytes it never ran). What this file owns is
#: the chi: the fixture that installs one, the floors that show it is live, and
#: the mutation that arms the refusal.
import gate_cuda_folded_curl as curl  # noqa: E402
#: THE NONLINEAR FIXTURE IS THE CONSTITUTIVE GATE'S, for the same reason. MEEP's
#: both-or-neither chi2/chi3 rule and the chi magnitudes chosen against the pole
#: guard's derived bound are already written down and already measured there.
import gate_cuda_nonlinear as nl  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
subnormal_band_hosts = probe.subnormal_band_hosts
operand_census = probe.operand_census

SEED = 20260820

#: Consecutive launches in the multi-step leg. 60 is the budget every certified
#: hand-CUDA record in this tree is cut at. ``fu`` is STATE: a tree that gets the
#: field right and the auxiliary wrong is correct for exactly one launch and
#: wrong forever after.
MULTI_STEP_BUDGET = curl.MULTI_STEP_BUDGET

SUB_STEPS: Tuple[str, ...] = curl.SUB_STEPS

# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# THE GRIDS ARE THE NONLINEAR FAMILY'S, NOT THE FOLD FAMILY'S: unfolded, PML
# active, real storage, every boundary triple the constitutive round swept, and
# the corpus's own 1-D shape. A fold is refused here by a DIFFERENT clause with
# its own gate, and mixing the two would make a divergence ambiguous between
# them.
#
# ``linear_components`` puts one component on MEEP's ``else if (u)`` branch and
# ``chi_is_volume`` draws a spatially varying chi; both are carried because a
# uniform, wholly-nonlinear grid cannot distinguish a chi lookup that reads the
# wrong cell from one that reads no cell at all -- and this gate's claim is
# precisely that the curl reads no cell at all.

SPECS: Tuple[Dict[str, Any], ...] = nl.SPECS

COURANTS: Tuple[float, ...] = curl.COURANTS
INEXACT_COURANT = curl.INEXACT_COURANT
VALUE_CLASSES: Tuple[str, ...] = curl.VALUE_CLASSES

#: The chi magnitude classes. ``moderate`` is the workhorse; ``near_pole`` drives
#: the expansion parameter close to (and under) the guard's 1/3 bound, which is
#: where a chi that DID reach the curl would move the most words. A gate that
#: swept only a small chi would be reporting "no effect" from a fixture that had
#: the least chance of showing one.
AMPLITUDES: Tuple[str, ...] = ("moderate", "near_pole")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = curl.GUARD_SETS

#: The derived bound ``driver._NonlinearityGuard`` enforces (stepping.py:1356-1359).
POLE_EXPANSION_BOUND = 1.0 / 3.0


# ---------------------------------------------------------------------------
# LEG 3's mutation: the refusal's own premise, armed
# ---------------------------------------------------------------------------

#: The curl expression as the SHIPPED kernels spell it, transcribed from
#: ``step_curl_kernels.py`` (three sites in each of the two device strings). A
#: rewrite that stopped matching would be reported NOT ARMED rather than passing.
_CURL_EXPRESSION = "float curl = dtdx * ((sf - f1) + (f2 - ss));"


def _chi_pade_scale_the_curl(source: str) -> Tuple[str, int]:
    """ARM THE REFUSAL: make the certified curl kernel apply a Pade factor.

    The census refuses both curl sub-steps for "a Pade factor on the constitutive
    product". Neither ``stepping.step_B`` nor ``step_D`` contains one, so the
    refusal is inherited -- but "the comparator would notice if it did" is a claim
    that has to be shown rather than asserted. This rewrites the curl device
    string so the curl really is scaled by a quotient of the same shape as
    ``calc_nonlinear_u``'s, and it MUST BE CAUGHT.

    THE SHAPE IS THE PADE FACTOR'S, NOT AN ARBITRARY PERTURBATION:
    ``(1 + c2) / (1 + 2*c2)`` with ``c2`` standing in as a multiple of the
    operand, which is what ``calc_nonlinear_u`` computes with ``c2`` built from
    the chi and the displacement. A scale by a constant would be a coarser defect
    and would not answer the question the refusal asks.

    It cannot bind a chi POINTER, and that is the honest limit of the leg: the
    shipped kernels take no chi argument at all, so there is no chi in scope to
    scale by. What the leg shows is that a factor of this shape, applied where a
    nonlinearity would apply it, is visible to this comparator on these grids.

    Applied to ``step_curl_kernels._step_B_pml_real_kernel_code`` /
    ``..._D_...``, which are restored immediately afterwards.
    """
    replacement = (
        "float curl = dtdx * ((sf - f1) + (f2 - ss)) * "
        "__fdiv_rn(1.0f + 0.05f * f1, 1.0f + 0.10f * f1);")
    sites = source.count(_CURL_EXPRESSION)
    return source.replace(_CURL_EXPRESSION, replacement), sites


#: Device-text defects, in the order they are scored. The first is this gate's
#: own and is the refusal armed; the rest are the certified curl battery,
#: borrowed from the shared probe so this gate cannot own a second spelling of a
#: defect the certified record was cut against.
SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "chi_pade_scale_the_curl": _chi_pade_scale_the_curl,
    "regroup_stencil": probe.SOURCE_MUTATIONS["regroup_stencil"],
    "drop_metallic_mask": probe.SOURCE_MUTATIONS["drop_metallic_mask"],
    "swap_dsig_dsigu": probe.SOURCE_MUTATIONS["swap_dsig_dsigu"],
    "drop_fu_store": probe.SOURCE_MUTATIONS["drop_fu_store"],
    "read_fprev_after_store": probe.SOURCE_MUTATIONS["read_fprev_after_store"],
    "fortran_order_index_decomposition":
        probe.SOURCE_MUTATIONS["fortran_order_index_decomposition"],
    # THE TWO NULLS. A battery whose every leg must be caught scores identically
    # whether the comparator works or has degenerated into failing everything.
    "commute_dtdx_scale": probe.SOURCE_MUTATIONS["commute_dtdx_scale"],
    "reload_fu_from_memory": probe.SOURCE_MUTATIONS["reload_fu_from_memory"],
}

NULL_SOURCE_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS

#: A mutation is scored only on legs whose grid can make its lines FIRE.
#: ``drop_metallic_mask`` on an all-periodic grid changes nothing, and
#: ``fortran_order_index_decomposition`` is arithmetically the identity on the
#: 1-D corpus shape (i = j = 0 and k = idx either way) -- an UNCAUGHT that is a
#: fact about the fixture and would block release for the wrong reason. This is
#: NOT a way to excuse an uncaught leg: every mutation must still be CAUGHT
#: wherever it is asked, and :func:`_score` reports NO LEGS -- never "inert" --
#: if the filter leaves none.
SOURCE_MUTATION_SPEC_FILTER: Dict[str, Tuple[str, ...]] = {
    "drop_metallic_mask": ("all_metallic", "mixed_pmp", "mixed_mpm",
                           "partly_nonlinear", "volume_chi"),
    "fortran_order_index_decomposition": ("all_periodic", "all_metallic",
                                          "mixed_pmp", "mixed_mpm",
                                          "partly_nonlinear", "volume_chi"),
}

#: HOST mutations: they corrupt the TABLES rather than the device text, so they
#: run on both backends.
HOST_MUTATIONS: Tuple[str, ...] = ("swap_curl_sublattice",)


# ---------------------------------------------------------------------------
# LEG 1 — the static scan, performed rather than quoted
# ---------------------------------------------------------------------------

#: What counts as naming the nonlinearity, as a lowercase substring test over
#: every identifier and attribute the function body mentions. Deliberately WIDE:
#: a narrow needle that missed a helper would return the empty intersection this
#: gate is trying to earn.
CHI_NEEDLES: Tuple[str, ...] = ("chi2", "chi3", "nonlinear")

#: The scan's POSITIVE CONTROL. ``update_E`` is where the Pade factor is applied
#: (stepping.py:998-1000 branches into ``_nonlinear_constitutive``), so the scan
#: must find the needles inside its closure. A scan that found them nowhere would
#: report an empty intersection for the curl and mean nothing by it.
CHI_SCAN_POSITIVE_CONTROL = "update_E"


def _called_names(node: ast.AST) -> Set[str]:
    """Every bare name and attribute tail this function body calls."""
    out: Set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, ast.Attribute):
            out.add(target.attr)
    return out


def _docstring_nodes(node: ast.AST) -> Set[int]:
    """The ``id()`` of every string Constant that is a bare expression statement.

    A bare string statement is a docstring or a prose aside; it is never a
    reference to anything. Excluding it is what stops the scan from reporting a
    function as "naming a chi" because its prose EXPLAINS that it does not --
    which is exactly what ``stepping._boundary_kinds`` and ``_shift_up`` do, and
    what the first run of this scan reported before this function existed.
    """
    out: Set[int] = set()
    for child in ast.walk(node):
        if (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)):
            out.add(id(child.value))
    return out


def _mentioned_symbols(node: ast.AST, include_prose: bool = False) -> Set[str]:
    """Every identifier, attribute tail and CODE string constant in this body.

    ``include_prose`` keeps the docstrings in, and exists so the artifact can
    record both readings side by side: the code-only intersection is the claim,
    and the prose one is what shows the exclusion was needed rather than
    convenient.
    """
    prose = set() if include_prose else _docstring_nodes(node)
    out: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)
        elif isinstance(child, ast.Attribute):
            out.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # A NON-docstring string constant is kept: getattr(fields, "chi2_for")
            # is a real reference and a scan that dropped every string would miss
            # it.
            if id(child) not in prose:
                out.add(child.value)
    return out


def chi_reachability() -> Dict[str, Any]:
    """Does ``step_B``/``step_D``'s call closure name a chi anywhere in ``stepping``?

    The whole scan, performed here and recorded in the artifact, rather than a
    reading quoted from a docstring. Comments and docstrings are NOT scanned --
    ``ast`` drops the first and this treats the second as ordinary string
    constants, which is why a function that only MENTIONS chi in prose does not
    count as naming it. That is the conservative direction for a null: prose
    cannot rescue the claim, only code can sink it.
    """
    path = os.path.join(_REPO_API, "meep_gpu", "stepping.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    functions: Dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, node)

    def names_a_chi(node: ast.AST, include_prose: bool) -> bool:
        return any(needle in symbol.lower()
                   for symbol in _mentioned_symbols(node, include_prose)
                   for needle in CHI_NEEDLES)

    names_chi = {name for name, node in functions.items()
                 if names_a_chi(node, False)}
    #: The same scan with the docstrings LEFT IN. Recorded so the exclusion is
    #: auditable: the difference between the two sets is exactly the functions
    #: whose PROSE mentions a chi and whose CODE does not, and two of them sit
    #: inside the curl's closure.
    names_chi_with_prose = {name for name, node in functions.items()
                            if names_a_chi(node, True)}

    calls = {name: _called_names(node) & set(functions)
             for name, node in functions.items()}

    def closure(root: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in seen or current not in functions:
                continue
            seen.add(current)
            stack.extend(calls.get(current, ()))
        return seen

    out: Dict[str, Any] = {
        "source": "meep_gpu/stepping.py",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "functions_parsed": len(functions),
        "needles": list(CHI_NEEDLES),
        "functions_naming_chi": sorted(names_chi),
        "functions_naming_chi_including_docstrings": sorted(names_chi_with_prose),
        "named_in_prose_only": sorted(names_chi_with_prose - names_chi),
        "closures": {},
    }
    for root in tuple(SUB_STEPS) + (CHI_SCAN_POSITIVE_CONTROL,):
        reached = closure(root)
        touching = sorted(reached & names_chi)
        out["closures"][root] = {
            "functions_reached": len(reached),
            "reached": sorted(reached),
            "naming_chi": touching,
            # The prose reading, kept beside the code one. On the two curl roots
            # this is NON-empty and the code reading is empty, which is the
            # difference the docstring exclusion makes and the reason it is
            # recorded rather than applied silently.
            "naming_chi_in_prose_only": sorted(
                (reached & names_chi_with_prose) - names_chi),
        }
    control = out["closures"][CHI_SCAN_POSITIVE_CONTROL]["naming_chi"]
    out["positive_control"] = {
        "root": CHI_SCAN_POSITIVE_CONTROL,
        "naming_chi": control,
        # A scan that finds the needles nowhere reports an empty intersection for
        # every root it is asked about, the curl included, and means nothing by
        # it. This is what separates "measured absent" from "not looked for".
        "found_something": bool(control),
    }
    out["curl_closures_are_chi_free"] = all(
        not out["closures"][root]["naming_chi"] for root in SUB_STEPS)
    out["sound"] = bool(out["positive_control"]["found_something"]
                        and out["curl_closures_are_chi_free"])
    return out


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def _rng_for(spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, amplitude: str) -> np.random.Generator:
    """SEEDED FROM A DIGEST, never from ``hash()``.

    Python salts ``hash()`` of a tuple containing strings with ``PYTHONHASHSEED``,
    so a gate seeded that way draws a different fixture every process and a
    failing case cannot be replayed. Same form as
    ``gate_cuda_cylindrical_complex.py:1475-1477``.
    """
    label = f"{spec['label']}|{sub_step}|{courant}|{value_class}|{amplitude}"
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big"))


def build(xp, spec: Dict[str, Any], courant: float, amplitude: str, rng,
          with_chi: bool = True):
    """A frozen ``(fields, layer, grid)`` triple, with or without the chi.

    ``with_chi=False`` IS NOT A SEPARATE FIXTURE. It draws from the SAME rng in
    the same order -- the epsilon volumes first, then nothing -- so the linear
    twin and the nonlinear one differ in the chi and in nothing else, which is
    what makes :func:`chi_moves_update_E`'s word count attributable to the chi.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=xp, courant=courant)
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    nl.install_epsilon(fields, grid, rng)
    if with_chi:
        nl.install_nonlinearity(fields, grid, spec, amplitude, rng)
    return fields, layer, grid


def seed_electric_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Seed everything ``update_E`` touches, for the differential floor.

    ALL THREE D VOLUMES, because the nonlinearity's ``Dsqr`` reads the other two
    components at neighbouring cells: a probe that seeded one would leave two
    thirds of the stencil reading zeros and would under-report how much the chi
    moves.
    """
    xp = grid.xp
    names = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez", "Dx", "Dy", "Dz")
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    else:
        host = subnormal_band_hosts(names, tuple(grid.shape), rng)
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def chi_moves_update_E(xp, spec: Dict[str, Any], courant: float,
                       value_class: str, amplitude: str) -> Dict[str, Any]:
    """THE DIFFERENTIAL FLOOR: does this grid's chi change a single output word?

    Two triples from IDENTICAL rng streams -- one with the chi installed, one
    without -- seeded with the SAME host words, each stepped once through
    ``stepping.update_E``. The differing output words are counted as raw uint32.

    A NULL GATE'S CHARACTERISTIC FAILURE is a fixture that never carried the
    thing it claims does not matter, and this is the number that refuses it. Zero
    differing words means the chi is inert on this grid and the case would be a
    linear run wearing a nonlinear label -- and it would PASS leg 2 for a reason
    about the fixture.
    """
    names = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    out: Dict[str, Any] = {}
    produced: Dict[bool, Dict[str, np.ndarray]] = {}
    for with_chi in (True, False):
        rng = _rng_for(spec, "update_E_probe", courant, value_class, amplitude)
        fields, layer, grid = build(xp, spec, courant, amplitude, rng,
                                    with_chi=with_chi)
        seed_rng = _rng_for(spec, "update_E_state", courant, value_class, amplitude)
        seed_electric_state(fields, grid, value_class, seed_rng)
        if with_chi:
            out["has_nonlinearity"] = bool(fields.has_nonlinearity)
            out["nonlinear_components"] = list(fields.nonlinear_components)
            margin = stepping.nonlinear_margin(fields, layer)
            out["expansion"] = None if margin is None else float(margin.expansion)
            out["denominator"] = None if margin is None else float(margin.denominator)
            out["margin_component"] = None if margin is None else margin.component
        stepping.update_E(fields, layer)
        produced[with_chi] = {name: to_host(getattr(fields, name)).copy()
                              for name in names}
    differing = total = 0
    for name in names:
        a = np.ascontiguousarray(produced[True][name],
                                 dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(produced[False][name],
                                 dtype=np.float32).ravel().view(np.uint32)
        differing += int(np.count_nonzero(a != b))
        total += int(a.size)
    out["differing_words"] = differing
    out["total_words"] = total
    out["fraction"] = differing / total if total else 0.0
    expansion = out.get("expansion")
    out["expansion_in_domain"] = bool(
        expansion is not None and 0.0 < expansion < POLE_EXPANSION_BOUND)
    # TWO READINGS, NOT ONE, and the split was forced by a measurement rather
    # than designed in. The first full local run skipped every ``subnormal_band``
    # case here: with every operand between 1e-45 and 1e-38, ``c2`` and ``c3``
    # underflow, ``calc_nonlinear_u`` returns exactly 1.0 on both paths and the
    # chi moves NO word -- 0 of 5760, at expansions around 1e-32. That is the
    # same fact the constitutive round's header records for its own band class,
    # and collapsing the two into one floor threw away half the sweep.
    #
    # * ``admissible`` -- the chi is installed and the run is inside the domain
    #   ``driver._NonlinearityGuard`` admits. A case that fails THIS is refused:
    #   past the pole bound the case would certify arithmetic the engine will not
    #   perform, and with no chi installed there is nothing to be null about.
    # * ``chi_is_live`` -- the chi actually changes a word of ``update_E``'s
    #   output. Only cases in this arm can support "the curl is exact on a grid
    #   whose nonlinearity is DOING something"; the others measure the curl under
    #   the subnormal policy on a grid that carries an installed-but-underflowed
    #   chi, which is a real configuration and a weaker claim. Both are scored,
    #   the arm is recorded per case, and :func:`summarize` requires the live arm
    #   to be non-empty on both sub-steps and to contain the corpus shape.
    out["admissible"] = bool(out.get("has_nonlinearity")
                             and out["expansion_in_domain"])
    out["chi_is_live"] = bool(out["admissible"] and differing > 0)
    out["arm"] = ("chi_live" if out["chi_is_live"]
                  else "chi_underflowed" if out["admissible"] else "inadmissible")
    return out


def absorber_departure(sub_step: str, layer, grid) -> Dict[str, Any]:
    """Does any axis's coefficient profile differ from the identity?

    ``kms = sinv = 1`` is the interior pass-through. A case whose whole profile is
    that identity cannot distinguish a coefficient-index error, and the certified
    battery's table mutations would come back UNCAUGHT for a reason about the
    fixture.
    """
    tables = curl.tables_for(sub_step, layer)
    per_axis = {}
    worst = 0.0
    for name in "xyz":
        deviation = 0.0
        for stem in ("kms", "sinv"):
            values = to_host(tables[f"{stem}_{name}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        per_axis[name] = deviation
        worst = max(worst, deviation)
    return {"per_axis": per_axis, "max_deviation": worst,
            "meets_floor": worst > 0.0}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, amplitude: str, guard: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects, and it is the oracle on a grid that CARRIES THE CHI -- which is the
    whole point. If the array path's curl saw the nonlinearity, the reference
    would move with it and the kernel, which cannot see it, would diverge.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = _rng_for(spec, sub_step, courant, value_class, amplitude)

    fields, layer, grid = build(xp, spec, courant, amplitude, rng)
    host = curl.seed_state(fields, grid, sub_step, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, kinds = curl.boundary_codes_for(grid, curl.LICENSED_SUBSTITUTION)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "amplitude": amplitude, "guard": guard,
        "backend": backend, "host_mutation": host_mutation,
        "boundaries": list(spec["boundaries"]),
        "boundary_codes": [int(c) for c in codes],
        "resolved_kinds": list(kinds),
        "linear_components": list(spec.get("linear_components", ())),
        "chi_is_volume": bool(spec.get("chi_is_volume", False)),
        "dtdx": dtdx,
        "structure": curl.structure_facts(grid),
        "operand_census": operand_census(host),
    }

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement -- but a record that did not carry it
    # could not show WHICH clause the run was taken against.
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    chi = chi_moves_update_E(xp, spec, courant, value_class, amplitude)
    case["chi_liveness"] = chi
    case["chi_arm"] = chi["arm"]
    if not chi["admissible"]:
        case["skipped"] = (
            "the chi on this grid is not admissible: "
            f"installed={chi.get('has_nonlinearity')} "
            f"expansion={chi.get('expansion')!r} vs pole bound "
            f"{POLE_EXPANSION_BOUND!r}. With no chi installed there is nothing to "
            "be null about, and past the bound the case would certify arithmetic "
            "the engine refuses to perform")
        case["seconds"] = time.time() - started
        return case

    absorbs = absorber_departure(sub_step, layer, grid)
    case["absorber_departure"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("every axis's coefficient profile is the identity; this "
                           "case cannot distinguish a coefficient-index error")
        case["seconds"] = time.time() - started
        return case

    frozen = curl.snapshot(fields, sub_step)

    # Leg 1: the oracle, ON THE GRID CARRYING THE CHI.
    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    else:
        stepping.step_D(fields, layer)
    reference = curl.snapshot(fields, sub_step)

    moved = curl.oracle_moved(frozen, reference, sub_step)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the SHIPPED kernel, from the SAME frozen state.
    curl.restore(fields, frozen)
    tables = (curl.tables_for(sub_step, layer) if host_mutation is None
              else curl.host_mutated_tables(host_mutation, sub_step, layer, grid))
    runner = curl.run_kernel_cuda if backend == "cuda" else curl.run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in curl.outputs(sub_step)}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step. ``fu`` IS STATE. ONE ``Fields`` object, run twice
    # from the SAME frozen state, rather than a second object built beside it:
    # two objects means two epsilon draws unless the volumes are copied across,
    # and a fixture that solved two different problems would report a divergence
    # that says nothing about the kernel.
    if host_mutation is None:
        curl.restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if sub_step == "step_B":
                stepping.step_B(fields, layer)
            else:
                stepping.step_D(fields, layer)
            curl.advance_sources(fields, sub_step)
        oracle_multi = curl.snapshot(fields, sub_step)

        curl.restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx)
            curl.advance_sources(fields, sub_step)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in curl.outputs(sub_step)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str):
    specs = SPECS if product == "full" else SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    amplitudes = AMPLITUDES if product == "full" else ("near_pole",)
    out = []
    for spec in specs:
        for sub_step in SUB_STEPS:
            for courant in courants:
                for value_class in classes:
                    for amplitude in amplitudes:
                        out.append((spec, sub_step, courant, value_class, amplitude))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, entry in enumerate(plan, start=1):
        spec, sub_step, courant, value_class, amplitude = entry
        case = one_case(backend, spec, sub_step, courant, value_class, amplitude,
                        guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"{amplitude} SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        chi = case["chi_liveness"]
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} {amplitude} "
            f"shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"arm={chi['arm']} chi_moves_E={chi['differing_words']} "
            f"exp={chi['expansion']:.3e} "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: case product: a leg taken from the head of a sweep has landed entirely on
#: controls in two sibling gates, so the one mutation that mattered was skipped
#: on every leg and scored 0/0 UNCAUGHT -- a harness defect that reads exactly
#: like a kernel defect.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "all_periodic",       # the wrap on every axis
    "all_metallic",       # the wall mask on every axis
    "mixed_pmp",          # one of each
    "one_dimensional",    # THE CORPUS SHAPE, where two axes are a single cell
    "partly_nonlinear",   # a component on MEEP's ``else if (u)`` branch
    "volume_chi",         # a spatially varying chi
)


def mutation_plan(product: str):
    """One case per (spec, sub-step) for the mutation legs, at the INEXACT courant.

    Held to one courant, one value class and the ``near_pole`` amplitude
    deliberately: a mutation leg answers "can this gate see this defect at all",
    and multiplying it by the whole sweep buys repetitions of that answer rather
    than a second question. ``near_pole`` is the amplitude at which a chi that DID
    reach the curl would move the most, which is the right place to ask whether
    this comparator can see a factor of that shape.
    """
    by_label = {spec["label"]: spec for spec in SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    return [(by_label[label], sub_step, INEXACT_COURANT, "uniform", "near_pole")
            for label in labels for sub_step in SUB_STEPS]


def leg_caught(case: Dict[str, Any]) -> bool:
    """Did this mutated leg diverge from the array path at EITHER granularity?

    A defect that leaves the field right and the auxiliary wrong is bit-identical
    on the target at launch one; a defect in the auxiliary alone shows in ``fu``
    immediately and in the field from launch two.
    """
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never run
    # has not been shown inert; it has not been asked.
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict}


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       product: str) -> Dict[str, Any]:
    """Every table-level defect. A gate that cannot fail certifies nothing."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(product)
    for name in HOST_MUTATIONS:
        legs = [one_case(backend, spec, sub_step, courant, value_class, amplitude,
                         "fmad_false", host_mutation=name)
                for spec, sub_step, courant, value_class, amplitude in plan]
        out[name] = dict(_score(legs, True), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results: Dict[str, Any], out_path: str,
                         product: str) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE
    (``compile_cache.kernel_cache_key``), so a mutated body is a miss and reaches
    NVRTC. Every leg records how many constructions came from the mutated bytes,
    because a leg reporting a pass for a mutation it never applied is worse than
    no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache, step_curl_kernels  # noqa: PLC0415

    attribute_for = {"step_B": "_step_B_pml_real_kernel_code",
                     "step_D": "_step_D_pml_real_kernel_code"}
    originals = {name: getattr(step_curl_kernels, attribute)
                 for name, attribute in attribute_for.items()}
    out: Dict[str, Any] = {}
    plan = mutation_plan(product)

    try:
        for sub_step in SUB_STEPS:
            attribute = attribute_for[sub_step]
            for name, transform in SOURCE_MUTATIONS.items():
                mutated, sites = transform(originals[sub_step])
                key = f"{sub_step}:{name}"
                allowed = SOURCE_MUTATION_SPEC_FILTER.get(name)
                leg_plan = [entry for entry in plan
                            if entry[1] == sub_step
                            and (allowed is None or entry[0]["label"] in allowed)]
                if sites == 0 or mutated == originals[sub_step]:
                    out[key] = {"armed": False, "sites": sites,
                                "why": ("the rewrite matched 0 sites or changed "
                                        "nothing; the mutation and the kernel "
                                        "have drifted apart"),
                                "verdict": "NOT ARMED"}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                setattr(step_curl_kernels, attribute, mutated)
                step_curl_kernels._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = [one_case("cuda", spec, leg_sub_step, courant, value_class,
                                 amplitude, "fmad_false")
                        for spec, leg_sub_step, courant, value_class, amplitude
                        in leg_plan]
                setattr(step_curl_kernels, attribute, originals[sub_step])
                step_curl_kernels._clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, name not in NULL_SOURCE_MUTATIONS)
                out[key] = dict(scored, armed=True, sites=sites,
                                spec_filter=list(allowed) if allowed else None,
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    # A leg reporting a pass for a mutation it never applied is
                    # worse than no leg: it measured the SHIPPED kernel and its
                    # verdict is about nothing.
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for sub_step, attribute in attribute_for.items():
            setattr(step_curl_kernels, attribute, originals[sub_step])
        step_curl_kernels._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any], backend: str) -> Dict[str, Any]:
    """The release decision, with every clause it rests on named."""
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    skipped = [c for c in primary if c.get("skipped")]
    single_ok = [c for c in scored if c["single_launch"]["bit_identical"]]
    multi_cases = [c for c in scored if "multi_step" in c]
    multi_ok = [c for c in multi_cases if c["multi_step"]["bit_identical"]]

    if not scored:
        reasons.append("no case was scored at all")
    if len(single_ok) != len(scored):
        reasons.append(f"single-launch divergence on {len(scored) - len(single_ok)} "
                       f"of {len(scored)} cases")
    if len(multi_ok) != len(multi_cases):
        reasons.append(f"multi-step divergence on {len(multi_cases) - len(multi_ok)} "
                       f"of {len(multi_cases)} cases")

    # THE LIVE ARM IS WHAT SEPARATES A NULL FROM A TAUTOLOGY, and it is required
    # to be non-empty on EACH curl sub-step rather than over the sweep as a
    # whole. A run whose ``step_D`` cases all landed in the underflowed arm would
    # be releasing that sub-step on a grid whose chi does nothing, and the totals
    # alone could not tell the two apart.
    live = [c for c in scored if c["chi_liveness"]["chi_is_live"]]
    underflowed = [c for c in scored if c["chi_arm"] == "chi_underflowed"]
    for sub_step in SUB_STEPS:
        if not [c for c in live if c["sub_step"] == sub_step]:
            reasons.append(
                f"no case with a LIVE chi was scored on {sub_step}: every case "
                f"there carried a chi whose c2/c3 underflowed, so the identity "
                f"says nothing about a nonlinearity that is doing something")
    if not [c for c in live if c["label"] == "one_dimensional"]:
        reasons.append("the 1-D corpus shape was never scored with a LIVE chi; "
                       "the rows this gate is about are (1, 1, nz)")
    if skipped:
        # A SKIP IS NOT A PASS. Every one is reported, and a sweep that skipped a
        # whole spec has not measured it. Only INADMISSIBLE cases can reach here
        # -- the underflowed ones are scored, in their own arm.
        reasons.append(f"{len(skipped)} cases were skipped at the admissibility "
                       f"floor: " + "; ".join(sorted({c["skipped"][:80]
                                                      for c in skipped})))

    # BOTH CURL SUB-STEPS. They are different kernels and different device
    # strings; a run that scored one would be releasing the other by analogy.
    sub_steps = {c["sub_step"] for c in scored}
    if sub_steps != set(SUB_STEPS):
        reasons.append(f"only sub-steps {sorted(sub_steps)} were scored")

    labels = {c["label"] for c in scored}

    # THE STATIC SCAN, including its positive control.
    scan = results.get("chi_reachability") or {}
    if not scan.get("positive_control", {}).get("found_something"):
        reasons.append("the static scan found no chi symbol in update_E's closure; "
                       "a scan that finds nothing anywhere reports an empty "
                       "intersection for the curl and means nothing by it")
    if not scan.get("curl_closures_are_chi_free"):
        reasons.append("the static scan found a chi symbol inside step_B's or "
                       "step_D's call closure; the array path's curl CAN see the "
                       "nonlinearity and this null is false")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")

    # THE ARMED PREMISE IS A CLAUSE OF ITS OWN, not merely one entry in the
    # battery. It is the leg that separates "the kernel is exact under a chi"
    # from "this comparator cannot see a Pade factor", and a run in which it was
    # filtered away, unarmed or partial must not release.
    armed = {key: leg for key, leg in results.get("source_mutations", {}).items()
             if key.endswith(":chi_pade_scale_the_curl")}
    if len(armed) != len(SUB_STEPS):
        reasons.append(f"the armed premise was scored on {sorted(armed)}, not on "
                       f"both curl sub-steps")
    for key, leg in armed.items():
        if leg.get("verdict") != "CAUGHT":
            reasons.append(f"the armed premise {key} is {leg.get('verdict')}; "
                           f"without it the identity above is consistent with a "
                           f"comparator that sees nothing")

    if backend != "cuda":
        reasons.append("this backend compiles nothing through NVRTC and certifies "
                       "nothing; a laptop leg measures the transcription only")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded_identical = {(c["label"], c["sub_step"], c["courant"],
                          c["value_class"], c["amplitude"])
                         for c in single_ok}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"],
                      c["amplitude"]) in guarded_identical]
    control_diverged = [c for c in comparable if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where "
                    "the guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this sub-step"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    chi_words = [c["chi_liveness"]["differing_words"] for c in scored]
    expansions = [c["chi_liveness"]["expansion"] for c in scored
                  if c["chi_liveness"].get("expansion") is not None]
    return {
        "released": not reasons,
        "reasons": reasons,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "skipped_cases": len(skipped),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "sub_steps": sorted(sub_steps),
        "specs": sorted(labels),
        "value_classes": sorted({c["value_class"] for c in scored}),
        "amplitudes": sorted({c["amplitude"] for c in scored}),
        # THE NON-VACUITY NUMBERS, so a reader can see the chi was live rather
        # than take it on the verdict's word.
        "chi_arms": {"chi_live": len(live), "chi_underflowed": len(underflowed)},
        "chi_live_by_sub_step": {sub_step: len([c for c in live
                                                if c["sub_step"] == sub_step])
                                 for sub_step in SUB_STEPS},
        "chi_live_specs": sorted({c["label"] for c in live}),
        "chi_words_moved_in_update_E": {
            "min": min(chi_words) if chi_words else 0,
            "max": max(chi_words) if chi_words else 0,
            "min_on_the_live_arm": min(
                (c["chi_liveness"]["differing_words"] for c in live), default=0)},
        "chi_expansion": {"min": min(expansions) if expansions else None,
                          "max": max(expansions) if expansions else None,
                          "pole_bound": POLE_EXPANSION_BOUND},
        "static_scan": {
            "functions_naming_chi": (results.get("chi_reachability") or {}
                                     ).get("functions_naming_chi"),
            "curl_closures_are_chi_free": scan.get("curl_closures_are_chi_free"),
            "positive_control": scan.get("positive_control"),
        },
        "claim": ("the SHIPPED CUDA real-PML curl pair (step_B_pml_real / "
                  "step_D_pml_real) is byte-identical to stepping.step_B / "
                  "step_D, per sub-step, on grids carrying an instantaneous "
                  f"chi2/chi3 -- {len(live)} of the {len(scored)} scored cases "
                  "with the chi measurably LIVE (it moves words of update_E's "
                  f"output) and {len(underflowed)} with it installed but "
                  "underflowed by the subnormal value class -- at one launch and "
                  "at 60, with no change to the device source. So "
                  "covers_real_pml_curl's chi2/chi3 refusal is inherited from the "
                  "constitutive family rather than required by this arithmetic"),
        "does_not_claim": [
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all, so a widened predicate licenses a MEASUREMENT "
            "and not a production step",
            "the predicate is NOT changed by this gate; the change it licenses is "
            "reported, not made",
            "the nonlinear CONSTITUTIVE sub-step is a different family with its "
            "own kernel, its own predicate and its own gate; nothing here "
            "strengthens or weakens it",
            "a chi combined with a fold, a conductivity, BFAST, special_kz, a "
            "Bloch phase or complex storage is refused by other clauses and is "
            "untested here",
            "a MAGNETIC (H-side) nonlinearity: the engine does not implement one "
            "(fields.set_nonlinear_volumes refuses it by name), so there is "
            "nothing to measure",
            "no throughput claim: this is a correctness gate and times nothing",
            "the chi_underflowed arm claims nothing about a live nonlinearity: "
            "under the subnormal value class every operand is between 1e-45 and "
            "1e-38, c2 and c3 underflow and calc_nonlinear_u returns exactly 1.0 "
            "on both paths. Those cases measure the curl under the float32 "
            "subnormal policy on a grid that carries a chi, which is a real "
            "configuration and a weaker statement",
        ],
    }


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help=("import MEEP first so the HOST half of a 'flush' "
                              "policy can be attained; strict install refuses "
                              "otherwise"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--falsify", default=None, choices=tuple(SOURCE_MUTATIONS),
                        help=("plant this defect for the WHOLE RUN, sweep included, "
                              "so the RELEASE VERDICT ITSELF is shown to flip. A "
                              "gate whose individual legs can fail but whose "
                              "verdict cannot has not been shown to be a gate"))
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_nonlinear_curl_null",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("does covers_real_pml_curl's instantaneous chi2/chi3 refusal "
                     "describe a divergence in step_B/step_D, or is it inherited "
                     "from the constitutive family that DOES form a Pade "
                     "factor?"),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY, and the order is the design:
        # under 'keep' the policy's strip wraps this, so it records the option
        # tuple NVRTC was really given (post-strip). Installed afterwards it
        # would sit outside the strip and record the pre-strip tuple -- the one
        # thing that would make the two policy legs look alike.
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}

    # LEG 1 FIRST, and unconditionally: it needs no device and it is what makes
    # the identity below a null rather than a coincidence.
    results["chi_reachability"] = chi_reachability()
    scan = results["chi_reachability"]
    log(f"[scan] stepping.py: {scan['functions_parsed']} functions, "
        f"{len(scan['functions_naming_chi'])} name a chi; "
        f"step_B closure touches {scan['closures']['step_B']['naming_chi']}, "
        f"step_D closure touches {scan['closures']['step_D']['naming_chi']}, "
        f"update_E closure touches "
        f"{len(scan['closures']['update_E']['naming_chi'])} of them")
    # THE CURL TERM TABLE, pinned against stepping's, so the NumPy transcription
    # cannot drift from the oracle it is compared with.
    results["curl_terms_vs_stepping"] = curl.check_terms_against_stepping()
    save(results, args.out)

    # IMPORTED ONLY ON THE DEVICE BACKEND. ``step_curl_kernels`` imports ``cupy``
    # at module scope, so a laptop leg that touched it would die on the import
    # rather than run the transcription it exists to run.
    step_curl_kernels = None
    originals: Dict[str, str] = {}
    attribute_for = {"step_B": "_step_B_pml_real_kernel_code",
                     "step_D": "_step_D_pml_real_kernel_code"}
    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        originals = {name: getattr(step_curl_kernels, attribute)
                     for name, attribute in attribute_for.items()}
    try:
        if args.falsify:
            # THE VERDICT'S OWN FALSIFICATION LEG. Every mutation leg shows that a
            # LEG can fail; this shows that the RELEASE DECISION can. The defect is
            # planted for the whole run, sweep included, so a run under --falsify
            # that still reported released=True would mean the summary is not
            # reading the cases it claims to.
            transform = SOURCE_MUTATIONS[args.falsify]
            planted = {}
            for sub_step, attribute in attribute_for.items():
                mutated, sites = transform(originals[sub_step])
                planted[sub_step] = sites
                setattr(step_curl_kernels, attribute, mutated)
            step_curl_kernels._clear_kernel_cache()
            results["falsification"] = {
                "planted": args.falsify, "sites": planted,
                "requires": "summary.released must come out FALSE"}
            log(f"[falsify] planted {args.falsify} sites={planted}: the release "
                f"verdict must now be FALSE")
            save(results, args.out)

        for guard, options, _is_primary in GUARD_SETS:
            if args.backend == "numpy" and guard != "fmad_false":
                continue  # there is no compiler on this leg to guard
            if args.backend == "cuda":
                step_curl_kernels._COMPILE_OPTIONS = tuple(options)
                step_curl_kernels._clear_kernel_cache()
            log(f"[guard] {guard} options={options}")
            run_sweep(results, args.out, args.backend, args.product, guard)

        if args.backend == "cuda":
            step_curl_kernels._COMPILE_OPTIONS = ("--fmad=false",)
            step_curl_kernels._clear_kernel_cache()

        if not args.skip_mutations and not args.falsify:
            # NOT UNDER --falsify: the mutation runners restore the shipped strings
            # in their own ``finally``, which would silently disarm the planted
            # defect partway through and leave an artifact that measured two
            # different programs.
            run_host_mutations(results, args.out, args.backend, args.product)
            if args.backend == "cuda":
                run_source_mutations(results, args.out, args.product)
    finally:
        if step_curl_kernels is not None:
            for sub_step, attribute in attribute_for.items():
                setattr(step_curl_kernels, attribute, originals[sub_step])
            step_curl_kernels._clear_kernel_cache()

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results, args.backend)
    if args.falsify:
        results["summary"]["falsification"] = {
            "planted": args.falsify,
            "verdict_flipped": not results["summary"]["released"],
            "reading": ("this run is a FALSIFICATION and is not a release under "
                        "any reading; what it measures is whether the release "
                        "decision can come out False at all"),
        }
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} "
        f"scored={verdict['scored_cases']} skipped={verdict['skipped_cases']} "
        f"single_identical={verdict['single_launch_identical']} "
        f"multi_identical={verdict['multi_step_identical']}/"
        f"{verdict['multi_step_cases']} "
        f"chi_words={verdict['chi_words_moved_in_update_E']} "
        f"expansion={verdict['chi_expansion']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

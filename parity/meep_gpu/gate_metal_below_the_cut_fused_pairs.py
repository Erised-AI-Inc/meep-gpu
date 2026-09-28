#!/usr/bin/env python3
"""Native MPS byte gate for the THREE below-the-cut fused magnetic Metal welds.

WHAT IS BEING CERTIFIED. Three products that each span the same three driver passes
— ``step_B``, ``zero_metal_B`` and ``update_H`` — on three run kinds the shipped
:mod:`~meep_gpu.metal_kernels.fused_magnetic_pair` refuses BY NAME:

    nonlinear   meep_gpu/metal_kernels/nonlinear_fused_magnetic_pair.py
    beta        meep_gpu/metal_kernels/beta_fused_magnetic_pair.py
    bfast       meep_gpu/metal_kernels/bfast_fused_magnetic_pair.py

ONE GATE FOR THREE PRODUCTS, DELIBERATELY, AND IT IS THE POINT RATHER THAN A
SHORTCUT. All three are the SAME construction with one emitter swapped: the wall
clear, the three-line seam and the constitutive lift are literally the shipped weld's
own code, imported rather than copied. A harness that parameterises the emitter
therefore measures the thing that actually varies, and a drift in the shared half
fails all three legs at once instead of one. Each product still gets its OWN
specialisation, its OWN fixture, its OWN separate-dispatch control, its OWN refusal
questions and its OWN mutation set — nothing is inherited by assertion.

THE NINE THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove a fused path ran: a plan never
   launched is byte-identical to the oracle BY CONSTRUCTION. Every case asserts the
   exact LAUNCH COUNT and that every compared array MOVED from its seeded value.
2. **A hollow pass.** Every mutation must be CAUGHT, and leg ``disarm`` reruns the
   identical case with the shipped bytes and requires zero, so a "caught" cannot be a
   harness that diverges anyway.
3. **A dead-branch mutation.** Each product declares the WALLS its mutation case
   emits, and :func:`needle` raises when a needle is absent — so a wall mutation
   armed on an axis the specialisation does not wall fails LOUDLY at arm time rather
   than reporting itself uncaught while measuring nothing.
4. **A weld that is secretly the ordinary product.** The beta and BFAST emitters both
   have a ``has_beta=False`` / ``has_bfast=False`` arm whose body IS the certified
   ordinary curl. A lift that silently took it would pass every byte leg, because
   ``beta = 0`` and ``k = 0`` ARE the ordinary arithmetic. Both modules assert their
   inserted statements at emit time, and leg ``transcription`` re-measures it here
   against the shipped ordinary weld's own source: the fused source must differ from
   it, and differ only in the declared way.
5. **A transcription that drifted.** The swapped curl body must appear VERBATIM on
   both sides of the spliced wall clear, and the constitutive half must differ from
   the certified ``update_H`` body in EXACTLY the three seam lines.
6. **An unmeasured platform assumption.** Each product's separate-scalar signature
   must FAIL to compile and its packed one must COMPILE, LAUNCH and read every struct
   field back. For ``bfast`` that leg carries the whole family: 30 pointers plus one
   packed struct is 31 bindings and ``device.MAX_BUFFER_BINDINGS`` is 31 — there is
   ZERO margin, and this is where that is measured rather than argued.
7. **A seam that is not really a seam.** Leg ``byte_neutral_control`` replaces the
   three register reads with reloads of the words the curl half just stored. That
   mutant must NOT diverge. It is the one armed edit required to be UNCAUGHT.
8. **A refusal that is really an omission.** Leg ``refusal`` measures the magnetic
   source seam, the electric-source polarity, the fold — AND THE DISJOINTNESS: each
   product's configuration must be REFUSED by the shipped ordinary weld and by the
   other two, in both directions. Three welds on one slot that co-admitted anything
   would be a composition hazard, not a coverage gain.
9. **A vacuous subnormal precondition.** Leg ``value_classes`` runs a uniform seed
   and a SIGNED-ZERO LATTICE (both must be bit-identical, and the lattice is what
   exercises the ``flag ? 0.0f : v`` select and the ``curl - adv`` subtraction on
   negative zero) and then a SUBNORMAL-BAND seed, whose census must FIRE. A
   precondition that never triggers on any input this harness can build is not a
   precondition, and this leg is what stops it being one.

THE SUBNORMAL POLICY ON THIS BACKEND IS ONE-SIDED, and leg ``policy`` measures that
rather than leaving it as prose: on MPS the float32 flush is native and has no lever,
``keep`` is a REFUSAL (metal_kernels/subnormal.py), so "both policies" here means one
delivered and one refused BY NAME. That is checked.

Rule 7: one flushed line per case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. python -u \
      parity/meep_gpu/gate_metal_below_the_cut_fused_pairs.py \
      --out parity/meep_gpu/results/<fresh dir>/gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    beta_fused_magnetic_pair, bfast_curl as bfast_family,
    bfast_fused_magnetic_pair, fused_magnetic_pair as shipped,
    launch as metal_launch, nonlinear_fused_magnetic_pair, nonlinear_update_e,
    shaders, special_kz, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

#: The budget every case runs, matching the shipped weld's and for its reason: the
#: classes this gate exists for COMPOUND. ``fu_B`` and ``f_w_H`` are state carried
#: between steps, ``f_bfast_B`` is a marginally-stable IIR, and a coefficient index
#: off by one axis needs several steps to reach the low bits of the interior.
STEPS = 12

#: The gate's seed base. PER-CASE SEEDS ARE DIGESTS OF THE CASE LABEL, never
#: ``hash()``: Python salts ``hash()`` of a string with PYTHONHASHSEED, so a
#: hash-seeded gate draws a different fixture every process and a failing case cannot
#: be replayed.
SEED = 4_100_000


def seed_for(*parts: str) -> int:
    """A stable per-case seed from a digest of the case's identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


#: Which array-path function each live pass is.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every stored volume a complete step can touch. ``f_bfast_*`` is in the list for
#: every product, not only the BFAST one: a weld that wrote a state volume it has no
#: business touching would otherwise be invisible. ``getattr`` returns ``None`` where
#: a configuration does not allocate one and the comparison skips it.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"]
    + [f"f_bfast_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"])


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# The three products, as data
# ---------------------------------------------------------------------------

def _nonlinear_fixture(**keywords: Any) -> Tuple[Any, Any]:
    """A nonlinear run. ``matrix.nonlinear`` installs chi2/chi3 on Ez."""
    return matrix.nonlinear(matrix.cart(**keywords))


def _beta_fixture(**keywords: Any) -> Tuple[Any, Any]:
    """A real ``grid.beta`` run. 2-D is the only shape ``grid.beta`` is defined on."""
    return matrix.flat(**keywords)


def _bfast_fixture(**keywords: Any) -> Tuple[Any, Any]:
    return matrix.cart(**keywords)


class Product:
    """One weld, and every callable this harness needs to drive it.

    Spelled as data so the legs below iterate rather than branch. A leg that had to
    ask "which product is this" per row would be three legs wearing one name.
    """

    __slots__ = ("key", "module", "fixture", "folded_fixture", "cases",
                 "mutation_case", "refusal_case", "curl_plan", "constitutive_plan",
                 "certified_curl", "curl_emitter", "entry_point", "extra_mutations",
                 "notes")

    def __init__(self, key: str, module: Any, fixture: Callable[..., Any],
                 folded_fixture: Callable[[], Tuple[Any, Any]],
                 cases: Sequence[Tuple[str, Dict[str, Any]]], mutation_case: str,
                 refusal_case: Dict[str, Any],
                 curl_plan: Callable[..., Any],
                 constitutive_plan: Callable[..., Any],
                 certified_curl: Callable[..., str],
                 curl_emitter: Callable[..., str],
                 entry_point: str,
                 extra_mutations: Callable[[str], Dict[str, str]],
                 notes: str) -> None:
        self.key = key
        self.module = module
        self.fixture = fixture
        self.folded_fixture = folded_fixture
        self.cases = tuple(cases)
        self.mutation_case = mutation_case
        self.refusal_case = dict(refusal_case)
        self.curl_plan = curl_plan
        self.constitutive_plan = constitutive_plan
        self.certified_curl = certified_curl
        self.curl_emitter = curl_emitter
        self.entry_point = entry_point
        self.extra_mutations = extra_mutations
        self.notes = notes

    # -- the module's four public entry points, by name -------------------
    def source(self, codes: Sequence[int], walls: Sequence[bool],
               contract: str = shaders.CONTRACT_OFF) -> str:
        return getattr(self.module, f"{self.key}_fused_magnetic_pair_source")(
            codes, walls, contract)

    def coverage(self, fields: Any, pml: Any, sources: Any,
                 residency: Any) -> Any:
        return getattr(
            self.module, f"metal_{self.key}_fused_magnetic_pair_coverage")(
            fields, pml, sources, residency)

    def plan(self, fields: Any, pml: Any, sources: Any, residency: Any,
             functions: Optional[Mapping[str, Any]] = None) -> Any:
        return getattr(
            self.module, f"plan_metal_{self.key}_fused_magnetic_pair")(
            fields, pml, sources, residency, (shaders.CONTRACT_OFF,), functions)

    def compile_mutant(self, source: str) -> Any:
        return getattr(compile_source(source), self.entry_point)


def _beta_mutations(base: str) -> Dict[str, str]:
    """The two statements that make this weld a BETA weld (special_kz.py:392-393)."""
    return {
        # DROP THE BETA TERM ENTIRELY: the weld becomes the ordinary one, which is
        # the single most consequential silent defect this family can have.
        "beta_term_dropped_on_target_0":
            needle(base, "    curl0 = curl0 - (beta_plus * b);\n", ""),
        # THE PARTNERS ARE CROSS-ASSIGNED (step_db.cpp:148-176): target 0 takes the
        # SECOND source's centre load `b`, target 1 the FIRST source's `a`.
        "beta_partners_swapped":
            needle(base, "curl0 = curl0 - (beta_plus * b);",
                   "curl0 = curl0 - (beta_plus * a);"),
        # THE SIGN. `curl - (c*g)` IS the array path's `curl + (-(c*g))`; flipping it
        # is a converged, smooth, wrong field.
        "beta_sign_flipped_on_target_1":
            needle(base, "curl1 = curl1 - (beta_minus * a);",
                   "curl1 = curl1 + (beta_minus * a);"),
        # THE TERM CARRIES NO dtdx: it is an analytic derivative (stepping.py:758-762).
        "beta_term_scaled_by_dtdx":
            needle(base, "curl0 = curl0 - (beta_plus * b);",
                   "curl0 = curl0 - (dtdx * beta_plus * b);"),
        # THE COEFFICIENTS ARE SIGN PARTNERS, not one value used twice.
        "beta_minus_replaced_by_beta_plus":
            needle(base, "curl1 = curl1 - (beta_minus * a);",
                   "curl1 = curl1 - (beta_plus * a);"),
    }


def _bfast_mutations(base: str) -> Dict[str, str]:
    """The Tustin tail (stepping._bfast_term:896-904), as armed defects."""
    return {
        # DROP THE TAIL'S EFFECT: the weld becomes the ordinary one.
        "bfast_curl_correction_dropped":
            needle(base, "    curl0 = curl0 - adv0;\n", ""),
        # THE STATE UPDATE, dropped. A different defect from the one above and on the
        # same component: the curl is right this step and wrong forever after.
        "bfast_state_store_dropped":
            needle(base, "    s0[ii] = st0 + adv0;", "    // stale bfast state"),
        # k1 IS INDEXED BY THE SECOND TERM AND k2 BY THE FIRST, cross-assigned
        # (bfast_curl.py:26-28, vec.hpp:445). Swapping them is the port's likeliest slip.
        "bfast_k_pair_swapped_on_target_0":
            needle(base, "float total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b));",
                   "float total0 = (k2_0 * (c_y + c)) - (k1_0 * (b_z + b));"),
        # THE TAIL SUMS, THE CURL DIFFERENCES. `(c_y + c)` is not `(c_y - c)`.
        "bfast_sum_becomes_a_difference":
            needle(base, "float total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c));",
                   "float total1 = (k1_1 * (a_z - a)) - (k2_1 * (c_x + c));"),
        # THE ADVANCE IS `total - 2*state` — the Tustin recurrence, not `total`.
        "bfast_advance_drops_the_state_term":
            needle(base, "float adv2 = total2 - (2.0f * st2);",
                   "float adv2 = total2;"),
        # THE CURL CORRECTION IS SUBTRACTED (the caller-subtracts convention,
        # :904 returns -advance and :364-368 adds it).
        "bfast_correction_sign_flipped":
            needle(base, "curl1 = curl1 - adv1;", "curl1 = curl1 + adv1;"),
    }


def _no_extra_mutations(base: str) -> Dict[str, str]:
    """The nonlinear weld's curl body IS the ordinary one, so it adds no defect here.

    NOT AN OMISSION AND SAID SO. This family contributes an ADMISSION and nothing
    else: its emitted bytes are :func:`.fused_magnetic_pair.fused_magnetic_pair_source`'s
    for the same specialisation, which leg ``transcription`` measures as a string
    equality. A curl mutation invented for it would be one of the shared set under a
    second name. What IS specific to it — that a nonlinear run reaches this kernel at
    all, and that the ordinary weld refuses the same run — is measured by legs
    ``product``, ``equivalence`` and ``refusal``.
    """
    return {}


PRODUCTS: Tuple[Product, ...] = (
    Product(
        key="nonlinear", module=nonlinear_fused_magnetic_pair,
        fixture=_nonlinear_fixture,
        folded_fixture=lambda: matrix.nonlinear(
            matrix.folded(boundaries={"y": "metallic"}, depth=1.2)),
        # THE CORPUS ROWS ARE 1-D-LIKE WITH A METALLIC z (census.json), so
        # ``wall_z_3d`` is the corpus's own wall geometry and ``wall_xyz_3d`` is
        # where every wall-clear row is emitted at once.
        cases=(("all_periodic_3d", dict()),
               ("wall_z_3d", dict(boundaries={"z": "metallic"})),
               ("wall_xyz_3d", dict(boundaries="metallic")),
               ("scalar_chi_wall_x_3d", dict(boundaries={"x": "metallic"})),
               ("thin_pml_wall_xy_3d",
                dict(boundaries={"x": "metallic", "y": "metallic"}, pml=1))),
        mutation_case="wall_xyz_3d", refusal_case=dict(boundaries={"z": "metallic"}),
        curl_plan=lambda f, p, r: nonlinear_update_e.plan_nonlinear_run_pml_curl(
            f, p, "step_B", r),
        constitutive_plan=lambda f, p, r:
            nonlinear_update_e.plan_nonlinear_run_constitutive(f, p, "H", r),
        certified_curl=lambda codes, contract: shipped.certified_curl_body(
            codes, contract),
        curl_emitter=lambda codes, contract: shaders.curl_source(codes, False,
                                                                 contract),
        entry_point="fused_magnetic_pair_step",
        extra_mutations=_no_extra_mutations,
        notes="admission only: the emitted bytes ARE fused_magnetic_pair's"),
    Product(
        key="beta", module=beta_fused_magnetic_pair, fixture=_beta_fixture,
        folded_fixture=lambda: matrix.folded(boundaries={"y": "metallic"},
                                             depth=0.0, beta=0.33),
        # 2-D: z has zero extent, so no z wall exists and the z wall-clear row is
        # never emitted. The mutation case walls x AND y, which is every row this
        # specialisation CAN emit.
        cases=(("beta_positive_2d", dict(beta=0.33)),
               ("beta_negative_2d", dict(beta=-0.6850526103319672)),
               ("beta_wall_x_2d", dict(beta=0.33, boundaries={"x": "metallic"})),
               ("beta_wall_xy_2d",
                dict(beta=0.33, boundaries={"x": "metallic", "y": "metallic"})),
               ("beta_thin_pml_2d", dict(beta=-0.2, pml_cells=1))),
        mutation_case="beta_wall_xy_2d",
        refusal_case=dict(beta=0.33, boundaries={"x": "metallic"}),
        curl_plan=lambda f, p, r: special_kz.plan_beta_pml_curl(f, p, "step_B", r),
        constitutive_plan=lambda f, p, r: special_kz.plan_beta_run_constitutive(
            f, p, "H", r),
        certified_curl=lambda codes, contract:
            beta_fused_magnetic_pair.certified_beta_curl_body(codes, contract),
        curl_emitter=lambda codes, contract: special_kz.beta_curl_source(
            codes, False, True, contract),
        entry_point="beta_fused_magnetic_pair_step",
        extra_mutations=_beta_mutations,
        notes="one emitter swapped; +2 struct fields, same 27 pointers"),
    Product(
        key="bfast", module=bfast_fused_magnetic_pair, fixture=_bfast_fixture,
        folded_fixture=lambda: matrix.folded(boundaries={"y": "metallic"},
                                             depth=1.2,
                                             bfast_scaled_k=(0.2, 0.0, 0.0)),
        cases=(("bfast_kx_3d", dict(bfast_scaled_k=(0.2, 0.0, 0.0))),
               ("bfast_kx_wall_xyz_3d",
                dict(bfast_scaled_k=(0.2, 0.0, 0.0), boundaries="metallic")),
               ("bfast_kxyz_3d", dict(bfast_scaled_k=(0.13, -0.4, 0.07))),
               ("bfast_kxyz_wall_y_3d",
                dict(bfast_scaled_k=(0.13, -0.4, 0.07),
                     boundaries={"y": "metallic"})),
               ("bfast_thin_pml_3d",
                dict(bfast_scaled_k=(0.21, 0.05, 0.0), pml=1))),
        # ALL THREE AXES WALLED **AND** ALL THREE k COMPONENTS LIVE, so every
        # wall-clear row, the full ownership mask and all six k scalars are present
        # on the case the mutations are scored on.
        mutation_case="bfast_kxyz_wall_xyz_3d",
        refusal_case=dict(bfast_scaled_k=(0.2, 0.0, 0.0),
                          boundaries={"x": "metallic"}),
        curl_plan=lambda f, p, r: bfast_family.plan_bfast_pml_curl(
            f, p, "step_B", r),
        constitutive_plan=lambda f, p, r: bfast_family.plan_bfast_run_constitutive(
            f, p, "H", r),
        certified_curl=lambda codes, contract:
            bfast_fused_magnetic_pair.certified_bfast_curl_body(codes, contract),
        curl_emitter=lambda codes, contract: bfast_curl_source(codes, contract),
        entry_point="bfast_fused_magnetic_pair_step",
        extra_mutations=_bfast_mutations,
        notes="one emitter swapped; +3 pointers -> 30 of 30, ZERO margin"),
)

#: The BFAST mutation case is not in :attr:`Product.cases` — it is built only for the
#: mutation legs, because it is the geometry that emits every armed line at once and
#: is not otherwise interesting as a product row. Declared here so the case table and
#: the mutation case cannot drift apart.
EXTRA_CASES: Dict[str, Dict[str, Any]] = {
    "bfast_kxyz_wall_xyz_3d": dict(bfast_scaled_k=(0.13, -0.4, 0.07),
                                   boundaries="metallic"),
}


def bfast_curl_source(codes: Sequence[int], contract: str) -> str:
    """``bfast_curl.bfast_curl_source`` with this seam's fixed arguments bound."""
    return bfast_family.bfast_curl_source(codes, False, contract, True)


def case_keywords(product: Product, name: str) -> Dict[str, Any]:
    table = dict(product.cases)
    if name in table:
        return table[name]
    if name in EXTRA_CASES:
        return EXTRA_CASES[name]
    raise KeyError(f"{product.key}: no case named {name!r}")


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

#: The value classes leg's seeds, and what each is for.
VALUE_CLASSES: Tuple[str, ...] = ("gaussian", "uniform", "signed_zero_lattice",
                                  "subnormal_band")


def fill(fields: Any, seed: int, value_class: str = "gaussian") -> None:
    """Seed every stored volume in ONE value class, identically on every side."""
    rng = np.random.default_rng(seed)
    shape = tuple(fields.grid.shape)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if value_class == "gaussian":
            values = rng.standard_normal(shape) * 0.37
        elif value_class == "uniform":
            values = rng.uniform(-1.0, 1.0, size=shape)
        elif value_class == "signed_zero_lattice":
            # +0.0 and -0.0 on a checkerboard. The two words differ (0x00000000 vs
            # 0x80000000) and a byte gate SEES the difference, so this is what
            # exercises `flag ? 0.0f : v` and every subtraction's sign convention.
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover - the table above is the whole domain
            raise ValueError(f"unknown value class {value_class!r}")
        array[...] = np.asarray(values, dtype=np.float32)


def build(product: Product, keywords: Mapping[str, Any], seed: int,
          value_class: str = "gaussian") -> Tuple[Any, Any]:
    """One seeded engine. Called twice (or three times) per case, identically."""
    fields, pml = product.fixture(**dict(keywords))
    fill(fields, seed, value_class)
    return fields, pml


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is."""
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def array_step(fields: Any, pml: Any, live: Sequence[str]) -> None:
    for name in live:
        ARRAY_PATH[name](fields, pml)


def metal_step(fields: Any, pml: Any, dispatch: Mapping[str, Any],
               owned: Sequence[str], residency: Residency,
               live: Sequence[str]) -> None:
    """One complete step, on the device where a plan owns the pass.

    ``owned`` and ``dispatch`` are SEPARATE because a fused plan owns THREE passes
    and dispatches at one of them. Deriving the skip set from the dispatch keys would
    leave ``zero_metal_B`` running on the host on top of the clear the kernel already
    carried — which is IDEMPOTENT and would hide a dropped carry.
    """
    skip = set(owned)
    for name in live:
        if name in skip:
            plan = dispatch.get(name)
            if plan is not None:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


def run_case(product: Product, keywords: Mapping[str, Any], seed: int, steps: int,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = "gaussian",
             require_every_volume_moved: bool = True) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(product, keywords, seed, value_class)
    actual, actual_pml = build(product, keywords, seed, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = product.plan(actual, actual_pml, (), residency, functions)
    if plan is None:
        reasons = product.coverage(actual, actual_pml, (), residency).reasons
        return {"passed": False, "reason": f"{product.key} weld was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    # MOVEMENT IS MEASURED AT EVERY STEP, NEVER FIRST-VERSUS-LAST, and this is a
    # correction a measurement made it: with a k vector whose target-0 coefficients
    # are both zero, the BFAST state's advance is exactly ``-2 * state``, so the
    # store is ``s = -s`` and after an EVEN budget the volume is back at its seed.
    # A first-versus-last check reported ``f_bfast_Bx`` as never moving on a run
    # where it changed sign twelve times — a false alarm that, on a different
    # volume, would have been a false ALL-CLEAR. The running maximum below is the
    # question the floor actually means: did this word ever differ from its seed?
    moved: Dict[str, int] = {name: 0 for name in before}
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(actual, actual_pml, {plan.replaces_sub_steps[0]: plan},
                   plan.replaces_sub_steps, residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(value)
                     for value in state_of(reference).values())
        live_state = state_of(actual)
        for name in moved:
            moved[name] = max(moved[name], differing(before[name],
                                                     live_state[name]))
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    # THE NON-VACUITY FLOOR, and it is per-VOLUME rather than a total: a weld that
    # never wrote H at all would still move B and pass a summed check. The SEAM
    # volumes are the ones this weld is responsible for and they must move on every
    # value class.
    #
    # ``require_every_volume_moved`` covers the WHOLE state and is the right floor on
    # a random seed, where a volume that did not move means a pass that did not run.
    # It is switched off for the SIGNED-ZERO LATTICE and for it alone, and the reason
    # is arithmetic rather than convenience: with every word at +-0 the D-side curl
    # is exactly zero, so ``fu_D`` legitimately stays at +-0 for the whole budget.
    # Failing on that would be scoring the fixture, not the kernel — while the seam
    # floor below still has to hold, so the leg cannot go vacuous.
    must_move = [name for name in ("Bx", "By", "Bz", "Hx", "Hy", "Hz",
                                   "f_w_Hx", "f_w_Hy", "f_w_Hz")
                 if name in before]
    dead = sorted(name for name in must_move if moved[name] == 0)
    return {
        "passed": bool(identical and clean and launches_ok and not dead
                       and (not still or not require_every_volume_moved)),
        "required_every_volume_to_move": require_every_volume_moved,
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "reference_subnormals": per_step[-1]["reference_subnormals"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "seam_volumes_that_never_moved": dead,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.codes),
        "zero_metal": list(plan.zero_metal),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
        "value_class": value_class,
    }


def run_separate_control(product: Product, keywords: Mapping[str, Any], seed: int,
                         steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, same state.

    THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three engines from
    one seed: the array path, the two ALREADY CERTIFIED Metal sub-step products for
    THIS run kind stepping the same seam as separate dispatches with the wall clear
    left on the HOST between them, and the fused product. All three must agree word
    for word at every complete step. The separate side is not a strawman: those two
    plans are exactly what ``plan_step`` composes for this configuration today.
    """
    reference, reference_pml = build(product, keywords, seed)
    separate, separate_pml = build(product, keywords, seed)
    fused, fused_pml = build(product, keywords, seed)

    separate_residency = Residency()
    curl = product.curl_plan(separate, separate_pml, separate_residency)
    magnetic = product.constitutive_plan(separate, separate_pml,
                                         separate_residency)
    fused_residency = Residency()
    plan = product.plan(fused, fused_pml, (), fused_residency)
    if curl is None or magnetic is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "constitutive": magnetic is not None,
                "fused": plan is not None}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(separate, separate_pml, {"step_B": curl, "update_H": magnetic},
                   ("step_B", "update_H"), separate_residency, live)
        metal_step(fused, fused_pml, {plan.replaces_sub_steps[0]: plan},
                   plan.replaces_sub_steps, fused_residency, live)
        separate_residency.sync_out()
        fused_residency.sync_out()
        per_step.append({
            "step": step,
            "separate_vs_array": sum(compare(reference, separate).values()),
            "fused_vs_array": sum(compare(reference, fused).values()),
            "fused_vs_separate": sum(compare(separate, fused).values()),
        })
        if any(value for key, value in per_step[-1].items() if key != "step"):
            break

    seam = ("zero_metal_B",)
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and magnetic.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + magnetic.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + magnetic.launches,
        "fused_launches": plan.launches,
        "separate_curl_family": getattr(type(curl), "family", type(curl).__name__),
        "separate_constitutive_family": getattr(type(magnetic), "family",
                                                type(magnetic).__name__),
        "seam_host_passes_for_separate": [n for n in live if n in seam],
        "seam_host_passes_for_fused": [],
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
    """The (codes, walls) pair the shipped plan compiles from."""
    from meep_gpu.stepping import _boundary_kinds

    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(fields.grid, pml))
    return codes, zero_metal_axes(fields.grid)


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit.

    THE DEAD-BRANCH GUARD. A mutation whose needle is absent — because the line sits
    under a specialisation this case does not emit — would launch the shipped kernel
    and report the defect as uncaught while measuring nothing. That is the single
    failure a mutation leg cannot see from its own result, so it is refused HERE, at
    arm time, with the needle named.
    """
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


def shared_mutations(product: Product, codes: Sequence[int],
                     walls: Sequence[bool]) -> Dict[str, str]:
    """The defects EVERY one of these three welds can carry, since all three splice
    the same wall clear, the same three-line seam and the same constitutive lift.

    THE WALL ROWS ARE ARMED ONLY WHERE THE SPECIALISATION EMITS THEM. ``zero_metal_mask``
    writes one line per WALLED axis, so on the beta family's 2-D mutation case (x and
    y walled, z of zero extent) the ``at_z`` row does not exist. Arming it there would
    raise in :func:`needle`; skipping it silently would be the dead-branch trap. The
    axis is therefore chosen from ``walls`` and RECORDED in the artifact.
    """
    base = product.source(codes, walls)
    walled = [axis for axis in range(3) if walls[axis]]
    assert walled, (
        f"{product.key}: the mutation case walls no axis; the wall mutations would "
        f"be armed on lines the shipped kernel does not emit")
    first = walled[0]
    other = next((axis for axis in range(3) if axis != first), first)
    flag = ("at_x", "at_y", "at_z")[first]
    other_flag = ("at_x", "at_y", "at_z")[other]

    edits: Dict[str, str] = {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float src0 = v0;", "float src0 = curl0;"),
        # THE SEAM, WRONG COMPONENT.
        "seam_takes_the_wrong_component":
            needle(base, "float src1 = v1;", "float src1 = v0;"),
        # THE WALL CLEAR, DROPPED on the first walled axis.
        "zero_metal_dropped":
            needle(base, f"    v{first} = {flag} ? 0.0f : v{first};\n", ""),
        # THE WALL TABLE IS THE DIAGONAL FOR B, the off-diagonal for D. Reusing the
        # D-side table clears the wrong component on every walled run.
        "zero_metal_uses_the_d_side_table":
            needle(base, f"    v{first} = {flag} ? 0.0f : v{first};",
                   f"    v{first} = {other_flag} ? 0.0f : v{first};"),
        # THE AUXILIARY IS NOT CLEARED: `zero_metal_B` passes B_COMPONENTS only.
        "zero_metal_also_masks_the_auxiliary":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   f"    u0[ii] = {flag} ? 0.0f : n0; u1[ii] = n1; u2[ii] = n2;"),
        # THE OWNERSHIP MASK, dropped. A different pass from the wall clear, on the
        # same component and axis — which is exactly why both are armed.
        "ownership_mask_dropped":
            needle(base, f"    curl{first} = {flag} ? 0.0f : curl{first};\n", ""),
        # STEP_B IS FORWARD. step_D's negated strides are a different product.
        "curl_direction_reversed":
            needle(base, "int si = i + 1, sj = j + 1, sk = k + 1;",
                   "int si = i - 1, sj = j - 1, sk = k - 1;"),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            needle(base, "dtdx * ((c_y - c) + (b - b_z))",
                   "dtdx * (c_y - c + b - b_z)"),
        # THE RECURRENCE PAIRS are vec.hpp's cycle_direction: target 0 takes (y, z).
        "recurrence_axis_pair_swapped":
            needle(base, "float n0 = ((p0 * km_y) - curl0) * si_y;",
                   "float n0 = ((p0 * km_z) - curl0) * si_z;"),
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "flux_store_dropped":
            needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
                   "    f1[ii] = v1; f2[ii] = v2;"),
        "fw_store_dropped":
            needle(base, "    w0[ii] = src0;", "    // stale f_w_Hx"),
        # The dsigw index is the component's OWN axis (stepping.py:227-228).
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = km0[i];",
                   "float kp_0 = kp0[j], km_0 = km0[j];"),
        "constitutive_accumulations_reversed":
            needle(base, "    a0 = a0 + kp_0 * src0;\n"
                         "    a0 = a0 - km_0 * prev0;",
                   "    a0 = a0 - km_0 * src0;\n"
                   "    a0 = a0 + kp_0 * prev0;"),
    }
    return edits


def byte_neutral_source(product: Product, codes: Sequence[int],
                        walls: Sequence[bool]) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0`` executes three lines above, so ``f0[ii]`` and ``v0``
    hold the same float32 word. This edit is the fusion's central claim written as a
    program, and leg ``byte_neutral_control`` requires it NOT to diverge.
    """
    source = product.source(codes, walls)
    for target in range(3):
        source = needle(source, f"float src{target} = v{target};",
                        f"float src{target} = f{target}[ii];")
    return source


def coefficient_slots(plan: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The two coefficient groups' argument indices, read off the plan's volumes.

    ``plan.volumes`` is built in the SIGNATURE'S ORDER by every one of the three plan
    builders, so the index of a named mirror in that list IS its argument index.
    Reading it here is what keeps the host mutations correct across a weld whose
    pointer layout differs — and an assertion that the groups came out six long is
    what stops a rename disabling the mutation silently.
    """
    volumes = list(plan.volumes)
    curl = tuple(volumes.index(f"pml:{stem}_{axis}_h")
                 for axis in "xyz" for stem in ("kms", "sinv"))
    constitutive = tuple(volumes.index(f"pml:{stem}_{axis}")
                         for axis in "xyz" for stem in ("kps", "kms"))
    assert len(set(curl)) == 6 and len(set(constitutive)) == 6, (curl, constitutive)
    return curl, constitutive


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

    ``step_B`` reads HALF-INTEGER positions and ``step_D`` integer ones
    (launch.py:109-124). The kernel takes six pointers and never asks which lattice
    they came from, so no shader mutation can reach this: it is a half-cell error in
    the absorber profile, not a crash.
    """
    slots, _ = coefficient_slots(plan)
    _rebind(plan, slots,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


def swap_h_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the HALF-INTEGER constitutive coefficients to the H half.

    ``update_H`` takes ``kps_a``/``kms_a`` and ``update_E`` takes ``kps_a_h``
    (stepping.py:948 vs :1015).
    """
    _, slots = coefficient_slots(plan)
    _rebind(plan, slots,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling(product: Product) -> Dict[str, Any]:
    """The separate-scalar signature must FAIL; the packed one must COMPILE + LAUNCH.

    For ``bfast`` this leg carries the whole family: 30 pointers plus one packed
    struct is 31 bindings against a ceiling of 31, so there is ZERO margin and a
    signature that compiled but bound the struct wrongly would produce a smooth,
    plausible, wrong field.
    """
    import torch  # noqa: PLC0415

    module = product.module
    separate = module.SEPARATE_SCALAR_BINDINGS
    packed = module.PACKED_BINDINGS
    # THE REFUTED SIGNATURE BELONGS TO WHICHEVER MODULE OWNS THE KERNEL. The
    # nonlinear family owns none — its emitted bytes ARE the shipped weld's — so the
    # shipped weld's refuted signature IS its refuted signature, and reading it from
    # there is the same statement as the one-line source forward. A per-family copy
    # would be a second string nothing measures against the first.
    refuted = getattr(module, "refuted_separate_scalar_source", None)
    row: Dict[str, Any] = {"separate_scalar_bindings": separate,
                           "packed_bindings": packed, "ceiling": 31,
                           "pointer_margin": 31 - packed,
                           "refuted_signature_owner":
                               module.__name__.rsplit(".", 1)[-1] if refuted
                               else "fused_magnetic_pair (this weld emits its bytes)"}
    if refuted is None:
        refuted = shipped.refuted_separate_scalar_source
        assert separate == shipped.SEPARATE_SCALAR_BINDINGS, (
            "this weld declares a separate-scalar width the shipped weld it borrows "
            "its kernel from does not; one of the two numbers is wrong")
    try:
        compile_source(refuted())
        row.update(separate_compiled=True, separate_error="", passed=False,
                   note=f"the {separate}-binding signature COMPILED; this family's "
                        f"shape rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    pointers_needed = packed - 1
    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(pointers_needed))
    # THE STRUCT IS READ OFF THE TEMPLATE THAT ACTUALLY EMITS THIS WELD, which for
    # the nonlinear family is the SHIPPED weld's — that family owns no template and
    # a fallback that quietly built a five-field struct for an eleven-field kernel
    # would make this leg measure a different signature from the shipped one.
    template = getattr(module, "_TEMPLATE", None) or shipped._TEMPLATE
    struct = template.split("struct Params", 1)[1].split(";\n\nkernel", 1)[0]
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        f"struct Params{struct};", "",
        "kernel void packed_probe(", pointers,
        f"    constant Params&    prm     [[buffer({pointers_needed})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;",
        f"    b5[idx] = b{pointers_needed - 1}[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(pointers_needed)]
    buffers[-1] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx = (3, 4, 5), 0.125
    # THE REAL PACKER, not a hand-built record: the field the kernel reads is then
    # the field the plan builder writes. n_elem is the product of the shape (60),
    # deliberately not the eight-element probe buffers, so the guard is exercised.
    params = _params_for(product, shape, dtdx)
    function(*buffers, params)
    torch.mps.synchronize()
    read = [buffers[index].cpu().numpy() for index in range(6)]
    expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx)), 8.0)
    fields_ok = all(bool(np.all(read[index] == expected[index]))
                    for index in range(6))
    row.update(packed_launched=True,
               packed_fields_read_back=[float(value[0]) for value in read],
               packed_fields_expected=[float(value) for value in expected],
               packed_fields_correct=fields_ok,
               passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok))
    return row


def _params_for(product: Product, shape: Sequence[int], dtdx: float) -> Any:
    """The product's OWN packer, called with its own extra scalars."""
    if product.key == "nonlinear":
        return shipped._params_tensor(shape, dtdx, "mps")
    if product.key == "beta":
        return beta_fused_magnetic_pair._params_tensor(shape, dtdx, 0.25, -0.25,
                                                        "mps")
    return bfast_fused_magnetic_pair._params_tensor(
        shape, dtdx, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), "mps")


def leg_transcription(product: Product, codes: Sequence[int],
                      walls: Sequence[bool]) -> Dict[str, Any]:
    """Both halves must be the CERTIFIED emitters' own bytes, and the weld must not
    be the ordinary one under another name.

    Four measurements:

    * the swapped curl body appears VERBATIM on both sides of the spliced wall clear;
    * the constitutive half differs from the certified ``update_H`` body in EXACTLY
      the three ``float srcN =`` lines;
    * the lifted curl body IS its own emitter's output (a string equality, so a lift
      that dropped a line is caught rather than inferred);
    * for ``beta`` and ``bfast``, the fused source DIFFERS from the shipped ordinary
      weld's for the same specialisation — which is what rules out the silent
      ``has_beta=0`` / ``has_bfast=0`` arm — and for ``nonlinear`` it is EQUAL to it,
      which is that family's whole claim.
    """
    source = product.source(codes, walls)
    curl = product.certified_curl(codes, shaders.CONTRACT_OFF)
    emitted = product.curl_emitter(codes, shaders.CONTRACT_OFF)
    lift_is_the_emitters_own = curl in emitted

    head, tail = curl.split(shipped._CURL_STORE, 1)
    curl_head_present = head in source
    curl_tail_present = (shipped._CURL_STORE + tail) in source

    certified = shipped.certified_constitutive_body().splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    expected = [(f"    float src{t} = g{t}[ii];", f"    float src{t} = v{t};")
                for t in range(3)]
    lengths_match = len(certified) == len(spliced)

    ordinary = shipped.fused_magnetic_pair_source(codes, walls)
    equals_the_ordinary_weld = (source == ordinary)
    should_equal = (product.key == "nonlinear")

    return {
        "passed": bool(curl_head_present and curl_tail_present and lengths_match
                       and changed == expected and lift_is_the_emitters_own
                       and equals_the_ordinary_weld == should_equal),
        "curl_body_head_verbatim": curl_head_present,
        "curl_body_tail_verbatim": curl_tail_present,
        "lifted_curl_is_its_emitters_own_output": lift_is_the_emitters_own,
        "certified_constitutive_lines": len(certified),
        "spliced_constitutive_lines": len(spliced),
        "constitutive_lines_changed": [list(pair) for pair in changed],
        "constitutive_lines_expected": [list(pair) for pair in expected],
        "equals_the_shipped_ordinary_weld": equals_the_ordinary_weld,
        "must_equal_the_shipped_ordinary_weld": should_equal,
    }


def _volume_source(fields: Any, component: str) -> Any:
    """A REAL engine source on ``component``, so ``field_type`` is the engine's."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0))


def leg_refusal(product: Product) -> Dict[str, Any]:
    """The source seam, the fold, and the DISJOINTNESS — each measured by name.

    Five questions:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real MAGNETIC ``VolumeSource`` must be refused, naming the driver line;
    3. a real ELECTRIC ``VolumeSource`` must NOT be refused — it is injected in the
       D/E half, and treating it as disqualifying would throw away every row this
       family exists to serve;
    4. a FOLDED grid must be refused, naming both B-side fills;
    5. THE SHIPPED ORDINARY WELD MUST REFUSE this product's configuration, and this
       product must refuse a plain configuration the ordinary weld admits. Three
       welds on one slot that co-admitted anything would be a composition hazard.
    """
    fields, pml = build(product, product.refusal_case,
                        seed_for(product.key, "refusal"))
    residency = Residency()

    undeclared = product.coverage(fields, pml, None, residency)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = product.coverage(fields, pml, (magnetic,), residency)
    magnetic_named = [r for r in magnetic_coverage.reasons
                      if "is magnetic" in r and "driver.py:3283-3284" in r]
    magnetic_plan = product.plan(fields, pml, (magnetic,), residency)

    electric = _volume_source(fields, "Ez")
    electric_coverage = product.coverage(fields, pml, (electric,), residency)

    # THE FOLD, ON A CONFIGURATION OF THIS RUN KIND, and that qualifier is the whole
    # correction. A weld whose first clause is its run-kind inversion — the nonlinear
    # one returns on ``_nonlinear_spine_reasons`` before it reaches any grid clause —
    # refuses a PLAIN folded grid for the wrong reason and never evaluates the fold
    # clause at all. Scoring that as "the fold is refused" would be the name-drift
    # failure: a check that passes while measuring nothing. So each product folds ITS
    # OWN run kind, and the leg requires the two B-side fills to be NAMED.
    folded_fields, folded_pml = product.folded_fixture()
    fill(folded_fields, seed_for(product.key, "refusal", "folded"))
    folded_coverage = product.coverage(folded_fields, folded_pml, (), Residency())
    folded_named = [r for r in folded_coverage.reasons
                    if "fill_symmetry_bc_B" in r and "fill_folded_far_ghosts_B" in r]

    # DISJOINTNESS, both directions.
    ordinary_on_this = shipped.metal_fused_magnetic_pair_coverage(
        fields, pml, (), residency)
    plain_fields, plain_pml = matrix.cart()
    fill(plain_fields, seed_for(product.key, "refusal", "plain"))
    plain_residency = Residency()
    ordinary_on_plain = shipped.metal_fused_magnetic_pair_coverage(
        plain_fields, plain_pml, (), plain_residency)
    this_on_plain = product.coverage(plain_fields, plain_pml, (), plain_residency)

    siblings = {}
    for other in PRODUCTS:
        if other.key == product.key:
            continue
        siblings[other.key] = {
            "covered": bool(other.coverage(fields, pml, (), residency).covered),
            "reasons": list(other.coverage(fields, pml, (), residency).reasons)[:3],
        }

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and not magnetic_coverage.covered and magnetic_named
                       and magnetic_plan is None
                       and electric_coverage.covered
                       and not folded_coverage.covered and folded_named
                       and not ordinary_on_this.covered
                       and ordinary_on_plain.covered
                       and not this_on_plain.covered
                       and not any(v["covered"] for v in siblings.values())),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_refused": not magnetic_coverage.covered,
        "magnetic_plan_is_none": magnetic_plan is None,
        "magnetic_named": magnetic_named,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "folded_refused": not folded_coverage.covered,
        "folded_named": folded_named,
        "shipped_ordinary_weld_refuses_this_configuration":
            not ordinary_on_this.covered,
        "shipped_ordinary_weld_reasons": list(ordinary_on_this.reasons),
        "shipped_ordinary_weld_admits_a_plain_run": ordinary_on_plain.covered,
        "this_weld_refuses_a_plain_run": not this_on_plain.covered,
        "this_weld_reasons_on_a_plain_run": list(this_on_plain.reasons),
        "sibling_welds_on_this_configuration": siblings,
    }


def leg_equivalence(product: Product) -> Dict[str, Any]:
    """The weld's predicate is the conjunction of its two halves', never wider.

    Every one of the three welds is built as ``half AND half AND seam clauses``, and
    the halves are the shipped SUB-STEP predicates for that run kind. This leg
    evaluates all three on the product's own case table and requires: wherever the
    weld admits, BOTH halves admit. A weld wider than a half it claims to inherit
    would be serving rows the certified sub-step refuses — the one failure this
    construction can have that bytes would not show, because the harness only ever
    builds configurations the weld admits.
    """
    rows: List[Dict[str, Any]] = []
    for name, keywords in product.cases:
        fields, pml = build(product, keywords, seed_for(product.key, "equiv", name))
        residency = Residency()
        weld = product.coverage(fields, pml, (), residency)
        curl = product.curl_plan(fields, pml, residency)
        magnetic = product.constitutive_plan(fields, pml, residency)
        rows.append({
            "case": name,
            "weld_covered": bool(weld.covered),
            "weld_reasons": list(weld.reasons),
            "half_curl_planned": curl is not None,
            "half_constitutive_planned": magnetic is not None,
            "never_wider": bool((not weld.covered)
                                or (curl is not None and magnetic is not None)),
        })
    return {"passed": all(row["never_wider"] for row in rows)
                      and any(row["weld_covered"] for row in rows),
            "cases": rows}


def leg_policy(product: Product) -> Dict[str, Any]:
    """One policy delivered, one REFUSED BY NAME — the whole lever this backend has.

    On MPS the float32 subnormal flush is native and uncontrollable and ``keep`` is a
    refusal (metal_kernels/subnormal.py). "Both policies" here is therefore: the
    resolved policy must be ``flush``, and asking for ``keep`` must produce a named
    refusal rather than a silent downgrade. A gate that ran under a policy it never
    checked would be certifying under an assumption.
    """
    from meep_gpu.metal_kernels.subnormal import mps_policy_reasons  # noqa: PLC0415
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    policy, origin = subnormal_policy.resolve_policy()
    keep_reasons = list(mps_policy_reasons("keep"))
    flush_reasons = list(mps_policy_reasons("flush"))
    return {
        "passed": bool(policy == "flush" and keep_reasons and not flush_reasons),
        "resolved_policy": policy,
        "policy_origin": origin,
        "keep_refused": bool(keep_reasons),
        "keep_reasons": keep_reasons,
        "flush_reasons": flush_reasons,
    }


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['product']}/{row['leg']}] "
        f"{row['label']}: passed={row.get('passed')} "
        f"diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--only", default="",
                        help="comma-separated product keys, for a partial rerun")
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    matrix.prepare_environment()

    selected = tuple(p for p in PRODUCTS
                     if not args.only or p.key in args.only.split(","))
    if not selected:
        raise SystemExit(f"--only={args.only!r} selects no product")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0

    # ARM EVERYTHING FIRST, so a needle absent from a specialisation fails before any
    # device time is spent and names itself in the traceback.
    armed: Dict[str, Dict[str, Any]] = {}
    for product in selected:
        keywords = case_keywords(product, product.mutation_case)
        probe_fields, probe_pml = build(product, keywords,
                                        seed_for(product.key, "probe"))
        codes, walls = _specialisation(probe_fields, probe_pml)
        sources = dict(shared_mutations(product, codes, walls))
        sources.update(product.extra_mutations(product.source(codes, walls)))
        armed[product.key] = {
            "codes": codes, "walls": walls,
            "mutants": {name: {shaders.CONTRACT_OFF: product.compile_mutant(text)}
                        for name, text in sources.items()},
            "neutral": {shaders.CONTRACT_OFF: product.compile_mutant(
                byte_neutral_source(product, codes, walls))},
        }
        log(f"armed {product.key}: {len(sources)} shader mutations on "
            f"{product.mutation_case} codes={codes} walls={walls}")

    total = sum(1 + 1 + 1 + 1 + len(p.cases) + 2 + 3 + 1
                + len(armed[p.key]["mutants"]) + len(HOST_MUTATIONS) + 1
                for p in selected)

    with jsonl.open("w", encoding="utf-8") as handle:
        for product in selected:
            codes = armed[product.key]["codes"]
            walls = armed[product.key]["walls"]
            mutation_keywords = case_keywords(product, product.mutation_case)

            def record(leg: str, label: str, payload: Dict[str, Any]) -> None:
                nonlocal index
                index += 1
                row = {"index": index, "total": total, "product": product.key,
                       "leg": leg, "label": label, **payload}
                rows.append(row)
                emit(handle, row)

            record("binding_ceiling",
                   f"{product.module.PACKED_BINDINGS}_bindings_packed",
                   leg_binding_ceiling(product))
            record("transcription", "both_halves_are_the_certified_bytes",
                   leg_transcription(product, codes, walls))
            record("equivalence", "the_weld_is_never_wider_than_its_halves",
                   leg_equivalence(product))
            record("policy", "flush_delivered_keep_refused", leg_policy(product))

            for name, keywords in product.cases:
                record("product", name,
                       {"steps": args.steps,
                        **run_case(product, keywords,
                                   seed_for(product.key, "product", name),
                                   args.steps)})

            for name in [n for n, _ in product.cases][:2]:
                record("separate_control", name,
                       {"steps": args.steps,
                        **run_separate_control(
                            product, case_keywords(product, name),
                            seed_for(product.key, "separate", name), args.steps)})

            record("refusal", "the_source_seam_the_fold_and_the_disjointness",
                   leg_refusal(product))

            # VALUE CLASSES. The first two must be bit-identical; the third must FIRE
            # the subnormal census, which is what stops the precondition being
            # vacuous. Its `passed` is therefore the census being NONZERO.
            for value_class in ("uniform", "signed_zero_lattice"):
                record("value_classes", value_class,
                       {"steps": args.steps,
                        **run_case(product, mutation_keywords,
                                   seed_for(product.key, "value", value_class),
                                   args.steps, value_class=value_class,
                                   require_every_volume_moved=(
                                       value_class != "signed_zero_lattice"))})
            band = run_case(product, mutation_keywords,
                            seed_for(product.key, "value", "subnormal_band"),
                            args.steps, value_class="subnormal_band")
            record("value_classes", "subnormal_band_precondition_fires",
                   {"steps": args.steps,
                    "reference_subnormals": band.get("reference_subnormals"),
                    "bit_identical": band.get("bit_identical"),
                    "passed": bool(band.get("reference_subnormals")),
                    "note": "the precondition must FIRE on a band seed; identity is "
                            "NOT claimed here and the census is the measurement"})

            # THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT.
            neutral = run_case(product, mutation_keywords,
                               seed_for(product.key, "neutral"), args.steps,
                               functions=armed[product.key]["neutral"])
            record("byte_neutral_control",
                   "register_replaced_by_a_reload_of_the_same_word",
                   {"diverged": not neutral.get("bit_identical", False),
                    "passed": bool(neutral.get("passed")),
                    "differing_words": neutral.get("differing_words"),
                    "launches": neutral.get("launches")})

            for name, functions in armed[product.key]["mutants"].items():
                result = run_case(product, mutation_keywords,
                                  seed_for(product.key, "mutation", name),
                                  args.steps, functions=functions)
                caught = not result.get("bit_identical", False)
                record("mutation", name,
                       {"caught": caught,
                        "passed": bool(caught and result.get("launches")),
                        "first_divergence": result.get("first_divergence"),
                        "differing_words": result.get("differing_words"),
                        "differing_arrays": result.get("differing_arrays"),
                        "launches": result.get("launches")})

            for name, patch in HOST_MUTATIONS.items():
                result = run_case(product, mutation_keywords,
                                  seed_for(product.key, "host_mutation", name),
                                  args.steps, patch=patch)
                caught = not result.get("bit_identical", False)
                record("mutation", name,
                       {"host_defect": True, "caught": caught,
                        "passed": bool(caught and result.get("launches")),
                        "first_divergence": result.get("first_divergence"),
                        "differing_words": result.get("differing_words"),
                        "differing_arrays": result.get("differing_arrays"),
                        "launches": result.get("launches")})

            # THE DISARM CHECK. Identical harness, identical case, SHIPPED bytes.
            record("disarm", "shipped_bytes_on_the_mutation_case",
                   {**run_case(product, mutation_keywords,
                               seed_for(product.key, "disarm"), args.steps)})

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "products": {p.key: {
            "module": Path(p.module.__file__).name,
            "packed_bindings": p.module.PACKED_BINDINGS,
            "separate_scalar_bindings": p.module.SEPARATE_SCALAR_BINDINGS,
            "mutation_case": p.mutation_case,
            "mutation_specialisation": {
                "codes": list(armed[p.key]["codes"]),
                "zero_metal": list(armed[p.key]["walls"])},
            "shader_mutations": len(armed[p.key]["mutants"]),
            "cases": [name for name, _ in p.cases],
            "notes": p.notes,
        } for p in selected},
        "host_mutations": sorted(HOST_MUTATIONS),
        "seed_base": SEED,
        "seed_rule": "SEED + sha256(label)[:4] — never hash(), which PYTHONHASHSEED "
                     "salts and which would make a failing case unreplayable",
        "corpus_reachable_seam_instances": {
            "census": "parity/meep_gpu/results/below_the_cut_census_2026-08-20",
            "nonlinear": 2, "beta": 1, "bfast": 1,
            "note": "reachable B->H instances; each cell's D->E partner is worth "
                    "ZERO (every row injects electrically inside that seam)",
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
        "source_sha256": {
            **{p.key: hashlib.sha256(
                Path(p.module.__file__).read_bytes()).hexdigest() for p in selected},
            "shipped_weld": hashlib.sha256(
                Path(shipped.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    failures = [f"{r['product']}/{r['leg']}/{r['label']}"
                for r in rows if not r["passed"]]
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"{len(rows)} rows, {len(failures)} failed; artifact {args.out}")
    for name in failures:
        log(f"  FAILED {name}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))

"""The FOLDED BETA COMPLEX fused D/E pair: a folded ``special_kz`` Bloch ``step_D``
welded into ``update_E``.

:mod:`.folded_complex_fused_pair` WITH MEEP'S BETA PASS CARRIED IN, and the pass is
the whole module -- the D-side mirror of what :mod:`.folded_beta_complex_fused_magnetic_pair`
is to :mod:`.folded_complex_fused_magnetic_pair` on the other seam. The source is the
certified weld's own output (:func:`.folded_complex_fused_pair.folded_complex_fused_pair_source`)
with :func:`.special_kz.beta_pass` inserted in register by
:func:`.folded_pass.insert_split_pass`. MEEP runs the beta pass after every curl
(step_db.cpp:161-175) through the additive ladder (``stepping._apply_folded_increment``),
and the pass reads the curl's partner fields only, so the weld's lines are untouched and
only the pass is added: the near and far carries, the parity chain, the wall clear, the
ghost coefficient reload and the constitutive blocks are the weld's own.

This is the cell ``(folded beta complex, folded beta complex)`` at ``D_to_E``, and it
is the one D/E cell neither shipped folded weld can stand in for.
:mod:`.folded_complex_fused_pair` refuses ``grid.beta`` through both halves' own clause
(``folded_complex._folded_complex_grid_reasons`` clause 8, which names "the folded beta
product" as the owner); :mod:`.beta_fused_electric_pair` refuses a fold by name and is
real-storage besides. This module is the product they name.

===========================================================================
WHAT IT REACHES
===========================================================================

THREE corpus rows, and unlike its magnetic twin ALL THREE are reachable:

* ``tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2``
* ``tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7``
* ``tests:TestSpecialKz.test_eigsrc_kz_0_complex``

The magnetic twin's docstring records that third row as UNREACHABLE ON ANY BACKEND
because it deposits a MAGNETIC source inside the B/H seam. On THIS seam the deposit is
electric and is CARRIED (see :data:`CARRIES_DEPOSIT_REPAIR`), so the row is in scope
here. **THE ROW COUNT IS NOT RE-DERIVED IN THIS MODULE AND IT CLAIMS NO CELL MOVE** --
that is a matrix run, and this module asserts only its predicate and its binding
arithmetic, both measured on every build.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

:data:`PACKED_BINDINGS` is IMPORTED from :mod:`.folded_complex_fused_pair` rather than
re-spelled, because the beta pass adds NO POINTER: its two words are real SCALARS and
they ride inside the same ``constant Params&`` the phases and parities already ride in,
in front of the chunk record. The record grows from 184 to :data:`PARAMS_ITEMSIZE` bytes
and the binding count does not move -- 31, which is :data:`.device.MAX_BUFFER_BINDINGS`
exactly. The three inverse-epsilon volumes are what put this family ON the ceiling
where the magnetic twin sits three under it: ``update_H`` is ``H = B`` with mu = 1
baked in, ``update_E`` reads ``D * inverse_epsilon_for(component)``.

``beta_plus``/``beta_minus`` are ``2*pi*beta*dt`` at each sign, rounded ONCE to float32
by :func:`.special_kz.beta_curl_coefficients` with ``magnetic=False``, which carries the
D side's negation in the word; the quarter turn is the pass's own spelling. That flag is
not a default: the two seams take OPPOSITE signs, and a weld that reused the magnetic
pair's words would run a correctly-shaped, converged, wrong dispersion.

NOT WIRED AS AN ARM: ``plan_step`` assigns at most one arm per slot and this product
spans five. It registers ``wired=False`` and reaches its two slots through
``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, _call, zero_metal_axes
from ..triton_kernels.symmetry import folded_axis_kinds
from . import folded_beta, folded_pass, shaders, special_kz
from . import chunk_bounds as _chunk_bounds
from . import folded_complex_fused_pair as _folded_complex_fused_pair
from .device import Residency, compile_source
from .folded_complex_fused_magnetic_pair import far_parity_words, parity_words
from .folded_complex_fused_pair import NEAR_SOURCE_INDEX, PACKED_BINDINGS
from .folded_fused_pair import _far_carry_reasons, _mirror_phase_reasons
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit. Flipping it is a claim
#: about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row, and it is what makes the cell worth THREE rows
#: rather than none: every corpus row here carries an electric source inside the seam.
#: The machinery is ``deposit_repair.repair_cells``' image closure over a FOLDED seam
#: (deposit_repair.py:217-258) -- the inverse of the forward carry this kernel
#: implements -- and it is storage-agnostic, indexing and accumulating element-wise over
#: one ``complex64`` array per component (fields.py:573). ``repairable`` still refuses
#: BY NAME the folds whose fill map it cannot read: the cylindrical r = 0 axis, and a
#: folded axis storing no more than ``MIRROR_SOURCE_INDEX`` cells.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_beta_complex_fused_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on the
#: slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: ``backward`` and ``has_beta`` for the lifted curl. Named rather than passed: a
#: ``has_beta=False`` build is the shipped folded complex D/E weld under this family's
#: name, and ``step_B``'s strides are the magnetic twin.
BACKWARD = True
HAS_BETA = True

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3303).
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The host record's total size in bytes: the parent's record
#: (:func:`.folded_complex_fused_pair.params_record_dtype`) with the two float beta
#: words in front of the chunk record -- 80 + 8 + 100 = 188, padded to the ``float2``
#: members' 8-byte alignment.
PARAMS_ITEMSIZE = 192

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAMILY", "HAS_BETA", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES", "SLOT",
    "MetalFoldedBetaComplexFusedPairPlan", "beta_words",
    "compile_folded_beta_complex_fused_pair",
    "folded_beta_complex_fused_pair_source",
    "metal_folded_beta_complex_fused_pair_coverage", "params_record_dtype",
    "plan_metal_folded_beta_complex_fused_pair", "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: The two beta words, as the ``Params`` member line and the unpack line this family
#: adds to the parent's own (:func:`.folded_pass.with_params_members`), in front of
#: the embedded chunk record and its alias.
_BETA_MEMBERS = "    float beta_plus; float beta_minus;"
_BETA_UNPACK = "    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;"


def folded_beta_complex_fused_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    THE CERTIFIED :func:`.folded_complex_fused_pair.folded_complex_fused_pair_source`'S OWN OUTPUT, with
    MEEP's beta pass carried in (:func:`.folded_pass.insert_split_pass`) and the two
    beta words added to its ``Params`` (:func:`.folded_pass.with_params_members`).
    The pass's statements are :func:`.special_kz.beta_pass`'s complex ones on the mask
    rows the folded head carries -- the cell-0 rows and, on a folded PERIODIC axis, the
    top-plane rows -- so the increment is masked exactly where the curl is. Its
    increment and ``fu`` update land before the ``fu`` store; each target's
    displacement update lands right after the displacement is formed, inside the
    ownership guard and before the wall clear and the store. Every other line is the
    parent's: the folded head and the per-chunk forms, the near and far carries, the
    parity chain, the wall clear and the constitutive blocks.
    """
    owner = "folded_beta_complex_fused_pair.folded_beta_complex_fused_pair_source"
    parent = _folded_complex_fused_pair.folded_complex_fused_pair_source(
        codes, phased, zero_metal, contract)
    source = folded_pass.with_params_members(
        parent, ("folded_complex_fused_pair_step", "folded_beta_complex_fused_pair_step"), _BETA_MEMBERS,
        _BETA_UNPACK, owner)
    block = special_kz.beta_pass(folded_pass.ownership_rows(source), True)
    return folded_pass.insert_split_pass(source, block, owner)


def compile_folded_beta_complex_fused_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_beta_complex_fused_pair_source(
        codes, phased, zero_metal,
        contract)).folded_beta_complex_fused_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def metal_folded_beta_complex_fused_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None) -> Coverage:
    """May ONE dispatch span folded beta complex ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    The halves are :mod:`.folded_beta`'s ROUTING verdicts (a fold MANDATORY), which is
    what keeps this product disjoint from :mod:`.beta_fused_electric_pair` and from
    :mod:`.folded_complex_fused_pair` in both directions.
    """
    reasons: List[str] = []

    curl = folded_beta.folded_beta_composition_bloch_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"folded beta complex curl half: {reason}"
                       for reason in curl.reasons)
    electric = folded_beta.folded_beta_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not electric.covered:
        reasons.extend(f"folded beta complex constitutive half: {reason}"
                       for reason in electric.reasons)

    # THE PARITY ARM IS THIS PRODUCT'S OWN QUESTION and neither half asks it: the two
    # halves are folded_beta's, which never emits a mirror-parity product because that
    # family registers NO fill arm. The fill arithmetic this weld carries inline is
    # folded_complex's, so folded_complex's artifact is required too.

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless the
    # deposit repair brackets the launch -- which for this family it does.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver injects "
            f"it BETWEEN step_D and update_E (driver.py:3294-3299), which is work "
            f"inside the seam this kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)
    reasons.extend(_far_carry_reasons(grid, codes))
    reasons.extend(_mirror_phase_reasons(grid, codes))

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedBetaComplexFusedPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded complex beta run."""

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "near_parities", "far_parities", "beta_values", "zero_metal",
                 "shape", "n_elem", "dtdx", "reflect", "params",
                 "chunk_bounds")

    family = "folded beta complex PML D-curl/mirror-fill/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "zero_metal", "reflect")

    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 near_parities: Sequence[float], far_parities: Sequence[float],
                 beta_values: Sequence[float],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 reflect: Sequence[Optional[int]], params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any], *,
                 chunk_bounds: Any) -> None:
        self.residency = residency
        # The host words of the record ``params`` carries, kept for inspection.
        self.chunk_bounds = chunk_bounds
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.near_parities = tuple(float(word) for word in near_parities)
        self.far_parities = tuple(float(word) for word in far_parities)
        # float(): `beta_curl_coefficients` already rounded ONCE to float32 and
        # float() of a numpy.float32 preserves the word.
        self.beta_values = tuple(float(word) for word in beta_values)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.reflect = tuple(None if row is None else int(row) for row in reflect)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


#: The two beta words this family's ``Params`` adds to the parent's, in front of the
#: embedded chunk record.
_BETA_FIELDS = (("beta_plus", "<f4"), ("beta_minus", "<f4"))


def params_record_dtype() -> Any:
    """The parent's ``Params`` record with the two beta words in front of ``cb``.

    :func:`.folded_pass.extend_params_dtype` of :func:`.folded_complex_fused_pair.params_record_dtype`,
    so every parent member keeps its offset (the three ``float2`` phases first, for
    the alignment reason the parent records) and the chunk record keeps its place
    last, as the source's struct does: itemsize :data:`PARAMS_ITEMSIZE`.
    """
    record = folded_pass.extend_params_dtype(_folded_complex_fused_pair.params_record_dtype(),
                                             _BETA_FIELDS)
    if record.itemsize != PARAMS_ITEMSIZE:  # pragma: no cover - an invariant
        raise AssertionError((record.itemsize, PARAMS_ITEMSIZE))
    return record


def beta_words(grid: Any) -> Tuple[float, float]:
    """``(beta_plus, beta_minus)`` -- the CURL ARM'S OWN words, called.

    :func:`.special_kz.beta_curl_coefficients` is the array path's coefficient
    evaluated on the objects the array path evaluates it on, so the words this kernel
    receives are its bits. ``magnetic`` is ``CURL_SUB_STEP == "step_B"`` -- NOT a
    default, because the D side's word carries the negation the B side's does not;
    ``complex_storage=True`` because this is the complex arm. Neither is a parameter.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=True)


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   near_parities: Sequence[float], far_parities: Sequence[float],
                   beta_values: Sequence[float], reflect: Sequence[Optional[int]],
                   device: str, chunk_words: Any) -> Any:
    """The parent's record, built by the parent's own ``_params_tensor`` on the host,
    re-laid with the two beta words (:func:`.folded_pass.extend_params_words`).

    ``-1`` stands for an axis with no reflect row, the parent's sentinel and for its
    reason. ``chunk_words`` is the chunk record's checked words.
    """
    import torch  # noqa: PLC0415

    beta = tuple(beta_values)
    if len(beta) != 2:
        raise ValueError(
            f"beta_values must be the (plus, minus) pair "
            f"special_kz.beta_curl_coefficients returns, got {beta_values!r}")
    parent = _folded_complex_fused_pair._params_tensor(
        shape, dtdx, phase_values, near_parities, far_parities, reflect, "cpu",
        chunk_words)
    words = folded_pass.extend_params_words(
        _folded_complex_fused_pair.params_record_dtype(), parent.numpy(), _BETA_FIELDS, beta)
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_beta_complex_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None, *, chunk_bounds: Any,
        ) -> Optional[MetalFoldedBetaComplexFusedPairPlan]:
    """Build the folded beta complex fused D/E plan, or ``None`` when refused."""
    if not metal_folded_beta_complex_fused_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    chunk_words = _chunk_bounds.plan_words(
        chunk_bounds, fields.grid.shape, "MetalFoldedBetaComplexFusedPairPlan")

    from ..stepping import _boundary_kinds, _far_reflect_rows  # noqa: PLC0415
    from . import complex_fields  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    walls = zero_metal_axes(grid)
    kinds = _boundary_kinds(grid, pml)
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=BACKWARD)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    near_parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    beta_values = beta_words(grid)
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    flux = [bind_complex(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in ("Ex", "Ey", "Ez")]
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + electric + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_beta_complex_fused_pair(
                codes, flags, walls, mode)

    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaComplexFusedPairPlan(
        residency, volumes, codes, flags, values, near_parities, far_parities,
        beta_values, walls, grid.shape, dtdx, reflect,
        _params_tensor(grid.shape, dtdx, values, near_parities, far_parities,
                       beta_values, reflect, residency.device, chunk_words),
        pointers, selected, chunk_bounds=chunk_words)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded beta complex fused D/E pair cannot fill {slot}",))
    return metal_folded_beta_complex_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedBetaComplexFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_complex_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS EVERY OTHER FUSED PAIR'S. ``plan_step`` assigns at most one arm per
    slot and this product spans five. Registering unwired keeps it ENUMERABLE for the
    disjointness sweep while ``arms.arms_for`` skips it, so no existing arm's selection
    changes. It reaches its two slots through ``launch.FUSED_PAIR_ARMS``.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded beta complex fused D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta complex fused D/E pair: ",
                          noun=("folded beta complex PML D-curl/mirror-fill/"
                                "stored-E pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()

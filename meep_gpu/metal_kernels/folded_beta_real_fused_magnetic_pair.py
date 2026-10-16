"""The FOLDED BETA REAL fused B/H pair: a folded ``special_kz`` ``step_B`` welded into
``update_H``.

:mod:`.folded_fused_magnetic_pair` WITH ONE EMITTER SWAPPED, and the swap is the whole
module — the B-side twin of :mod:`.folded_beta_real_fused_pair` and the real-storage
sibling of :mod:`.folded_beta_complex_fused_magnetic_pair`. The folded real curl body
is replaced by the certified FOLDED BETA real curl body
(:func:`.folded_beta.folded_beta_curl_source` with ``backward=False`` and
``has_beta=True``) and everything below the split-field cut is IMPORTED from that
shipped weld rather than re-spelled: the near and far carries, the wall clear, the
ghost coefficient reload and the constitutive block emitter are all that module's own
functions, called here.

This is the cell ``(folded beta real, folded beta real)`` at ``B_to_H``.
:mod:`.folded_fused_magnetic_pair` refuses ``grid.beta`` (its halves name "the folded
beta product" as the owner) and :mod:`.beta_fused_magnetic_pair` refuses a fold by
name, so this is the product they name.

WHY THE SWAP IS SOUND, measured rather than argued. The folded beta curl and the folded
curl differ by TWO STATEMENTS and two scalars::

    curl0 = curl0 - (beta_plus  * b);
    curl1 = curl1 - (beta_minus * a);

inserted AFTER the ``dtdx`` curl and BEFORE BOTH ownership masks — the array path's own
order (stepping.py:384-391 against :397). Everything below the cut is untouched.

WHY THE CONSTITUTIVE HALF NEEDS NOTHING. ``update_H`` reads nothing beta-dependent:
``grid.beta`` enters the two CURLS only, never ``_apply_constitutive_pml``. That is the
measurement :func:`.folded_beta.folded_beta_constitutive_coverage` already ships on.

THE SIGNATURE IS :data:`.fused_magnetic_pair.PACKED_BINDINGS` UNMOVED — 27 pointers
plus one packed ``constant Params&``, 28 of the platform's 31. The beta term adds NO
POINTER: its two coefficients are floats and ride inside the same record the five
scalars and three reflect rows already ride in.

``beta_plus``/``beta_minus`` are ``2*pi*beta*dt`` at each sign, rounded ONCE to float32
by :func:`.special_kz.beta_curl_coefficients` with ``magnetic=True``. That flag is not
a default: the two seams take OPPOSITE signs.

NOT WIRED AS AN ARM: this product spans five driver passes. It reaches its two slots
through ``launch.FUSED_PAIR_ARMS``.
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
from .device import Residency, compile_source
from . import folded_fused_magnetic_pair as _folded_fused_magnetic_pair
from .folded_fused_magnetic_pair import (
    NEAR_SOURCE_INDEX,
    PACKED_BINDINGS,
    _far_carry_reasons,
    _mirror_phase_reasons,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row. The claim is exactly the one
#: :mod:`.folded_fused_magnetic_pair` has carried since 2026-08-28, on the same seam
#: with the same geometry: ``deposit_repair.repair_cells`` (deposit_repair.py:217-258)
#: restores the CLOSURE of the cells ``fill_symmetry_bc_B`` and
#: ``fill_folded_far_ghosts_B`` image each deposit point into, which is the inverse of
#: the forward carry this kernel implements. ``repairable`` still refuses BY NAME the
#: folds whose fill map it cannot read.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_beta_real_fused_magnetic_pair"

SLOT = "step_B"
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: ``backward`` and ``has_beta`` for the lifted curl. Named rather than passed: a
#: ``has_beta=False`` build is :mod:`.folded_fused_magnetic_pair` under this family's
#: name, and every device leg would still pass because beta = 0 IS that arithmetic.
BACKWARD = False
HAS_BETA = True

#: The driver passes ONE launch of this plan performs (driver.py:3281-3289).
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAMILY", "HAS_BETA", "PACKED_BINDINGS", "REPLACES", "SLOT",
    "MetalFoldedBetaRealFusedMagneticPairPlan", "beta_words", "compile_folded_beta_real_fused_magnetic_pair",
    "folded_beta_real_fused_magnetic_pair_source",
    "metal_folded_beta_real_fused_magnetic_pair_coverage",
    "plan_metal_folded_beta_real_fused_magnetic_pair", "register_arms",
]


#: The two beta words, as the ``Params`` member line and the unpack line this family
#: adds to the parent's own (:func:`.folded_pass.with_params_members`), in front of
#: the embedded chunk record and its alias.
_BETA_MEMBERS = "    float beta_plus; float beta_minus;"
_BETA_UNPACK = "    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;"


def folded_beta_real_fused_magnetic_pair_source(
        codes: Sequence[int], phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, parities, walls, mode).

    THE CERTIFIED :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_source`'S OWN OUTPUT, with
    MEEP's beta pass carried in (:func:`.folded_pass.insert_split_pass`) and the two
    beta words added to its ``Params`` (:func:`.folded_pass.with_params_members`).
    The pass's statements are :func:`.special_kz.beta_pass`'s on the mask rows the
    folded head carries -- the cell-0 rows and, on a folded PERIODIC axis, the
    top-plane rows -- so the increment is masked exactly where the curl is. Every
    other line is the parent's: the folded head and the per-chunk forms, the near and
    far carries, the wall clear and the constitutive blocks.

    ``phases`` is ``grid.mirror_phase(axis)`` per axis, +1 or -1 on a folded axis, a
    SOURCE specialisation of the parent's (a runtime weight flushes every subnormal
    on this backend at BOTH signs, symmetry.py:657-686).
    """
    owner = "folded_beta_real_fused_magnetic_pair.folded_beta_real_fused_magnetic_pair_source"
    parent = _folded_fused_magnetic_pair.folded_fused_magnetic_pair_source(codes, phases, zero_metal, contract)
    source = folded_pass.with_params_members(
        parent, ("folded_fused_magnetic_pair_step", "folded_beta_real_fused_magnetic_pair_step"), _BETA_MEMBERS,
        _BETA_UNPACK, owner)
    block = special_kz.beta_pass(folded_pass.ownership_rows(source), False)
    return folded_pass.insert_split_pass(source, block, owner)


def compile_folded_beta_real_fused_magnetic_pair(
        codes: Sequence[int], phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, parities, walls, mode)."""
    return compile_source(folded_beta_real_fused_magnetic_pair_source(
        codes, phases, zero_metal,
        contract)).folded_beta_real_fused_magnetic_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_beta_real_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None) -> Coverage:
    """May ONE dispatch span folded beta real ``step_B`` -> fills -> wall -> ``update_H``?"""
    reasons: List[str] = []

    curl = folded_beta.folded_beta_composition_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"folded beta real curl half: {reason}"
                       for reason in curl.reasons)
    magnetic = folded_beta.folded_beta_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not magnetic.covered:
        reasons.extend(f"folded beta real constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B unless the
    # deposit repair brackets the launch -- which for this family it does. An ELECTRIC
    # source is injected in the D/E half and does NOT disqualify the pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the driver injects "
            f"it BETWEEN step_B and update_H (driver.py:3283-3284)"),
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
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedBetaRealFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded real beta run."""

    __slots__ = ("residency", "volumes", "codes", "phases", "zero_metal", "shape",
                 "dtdx", "beta_plus", "beta_minus", "reflect", "params",
                 "chunk_bounds")

    family = "folded beta real PML B-curl/mirror-fill/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phases", "zero_metal", "reflect",
                   "beta_plus", "beta_minus")

    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phases: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 beta_plus: float, beta_minus: float,
                 reflect: Sequence[Optional[int]], params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any], *,
                 chunk_bounds: Any) -> None:
        self.residency = residency
        # The host words of the record ``params`` carries, kept for inspection.
        self.chunk_bounds = chunk_bounds
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phases = tuple(int(value) for value in phases)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.reflect = tuple(None if row is None else int(row) for row in reflect)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def beta_words(grid: Any) -> Tuple[float, float]:
    """``(beta_plus, beta_minus)`` — the CURL ARM'S OWN coefficients, called.

    ``magnetic=True`` because :data:`CURL_SUB_STEP` is ``step_B`` -- NOT a default,
    because the two seams take opposite signs; ``complex_storage=False`` because this
    is the real arm.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=False)


#: The two beta words this family's ``Params`` adds to the parent's, in front of the
#: embedded chunk record.
_BETA_FIELDS = (("beta_plus", "<f4"), ("beta_minus", "<f4"))


def params_record_dtype() -> Any:
    """The parent's ``Params`` record with the two beta words in front of ``cb``.

    :func:`.folded_pass.extend_params_dtype` of
    :func:`.folded_fused_magnetic_pair.params_record_dtype`, so every parent member keeps its
    offset and the record keeps its place last, as the source's struct does.
    """
    return folded_pass.extend_params_dtype(_folded_fused_magnetic_pair.params_record_dtype(),
                                           _BETA_FIELDS)


def _params_tensor(shape: Sequence[int], dtdx: float, beta_plus: float,
                   beta_minus: float, reflect: Sequence[Optional[int]],
                   record: Any, device: str) -> Any:
    """The parent's record, built by the parent's own ``_params_tensor`` on the host,
    re-laid with the two beta words (:func:`.folded_pass.extend_params_words`).

    ``-1`` stands for an axis with no reflect row, the parent's sentinel and for its
    reason. ``record`` is the chunk record's checked words.
    """
    import torch  # noqa: PLC0415

    parent = _folded_fused_magnetic_pair._params_tensor(shape, dtdx, reflect, record, "cpu")
    words = folded_pass.extend_params_words(
        _folded_fused_magnetic_pair.params_record_dtype(), parent.numpy(), _BETA_FIELDS,
        (beta_plus, beta_minus))
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_beta_real_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None, *, chunk_bounds: Any,
        ) -> Optional[MetalFoldedBetaRealFusedMagneticPairPlan]:
    """Build the folded beta real fused B/H plan, or ``None`` when refused."""
    if not metal_folded_beta_real_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _far_reflect_rows  # noqa: PLC0415

    grid = fields.grid
    record = _chunk_bounds.plan_words(chunk_bounds, grid.shape,
                                      "plan_metal_folded_beta_real_fused_magnetic_pair")
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    walls = zero_metal_axes(grid)
    beta_plus, beta_minus = beta_words(grid)
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    pointers = (
        [bind(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
        + [bind("fu_" + name, getattr(fields, "fu_" + name))
           for name in ("Bx", "By", "Bz")]
        + [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
        + [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
        + [bind("f_w_" + name, getattr(fields, "f_w_" + name))
           for name in ("Hx", "Hy", "Hz")]
        # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
        # one — the OPPOSITE pairing to the D/E pair, and the kernel cannot tell.
        + [bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                constant=True)
           for axis in "xyz" for stem in ("kms", "sinv")]
        + [bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
           for axis in "xyz" for stem in ("kps", "kms")])
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_beta_real_fused_magnetic_pair(
                codes, phases, walls, mode)

    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaRealFusedMagneticPairPlan(
        residency, volumes, codes, phases, walls, grid.shape, dtdx,
        beta_plus, beta_minus, reflect,
        _params_tensor(grid.shape, dtdx, beta_plus, beta_minus, reflect, record,
                       residency.device),
        pointers, selected, chunk_bounds=record)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded beta real fused B/H pair cannot fill {slot}",))
    return metal_folded_beta_real_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedBetaRealFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_real_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False`` — every other fused pair's reason."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded beta real fused B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta real fused B/H pair: ",
                          noun=("folded beta real PML B-curl/mirror-fill/"
                                "stored-H pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()

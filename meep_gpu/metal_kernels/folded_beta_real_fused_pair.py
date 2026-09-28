"""The FOLDED BETA REAL fused D/E pair: a folded ``special_kz`` ``step_D`` welded into
``update_E``.

:mod:`.folded_fused_pair` WITH ONE EMITTER SWAPPED, and the swap is the whole module —
the real-storage sibling of :mod:`.folded_beta_complex_fused_pair` and the D-side twin
of :mod:`.folded_beta_real_fused_magnetic_pair`. The folded real curl body is replaced
by the certified FOLDED BETA real curl body
(:func:`.folded_beta.folded_beta_curl_source` with ``backward=True`` and
``has_beta=True``) and everything below the split-field cut is IMPORTED from that
shipped weld rather than re-spelled: the near and far carries, the wall clear, the
ghost coefficient reload and the constitutive block emitter are all
:mod:`.folded_fused_pair`'s own functions, called here.

This is the cell ``(folded beta real, folded beta real)`` at ``D_to_E``.
:mod:`.folded_fused_pair` refuses ``grid.beta`` (its halves name "the folded beta
product" as the owner) and :mod:`.beta_fused_electric_pair` refuses a fold by name, so
this is the product they name.

WHY THE SWAP IS SOUND, measured rather than argued. The folded beta curl and the folded
curl differ by TWO STATEMENTS and two scalars::

    curl0 = curl0 - (beta_plus  * b);
    curl1 = curl1 - (beta_minus * a);

inserted AFTER the ``dtdx`` curl and BEFORE BOTH ownership masks — the array path's own
order (stepping.py:467-474 against :479). Everything below the cut is untouched, which
is why every carry emitter is called rather than copied.

THE SIGNATURE IS :data:`.folded_fused_pair.PACKED_BINDINGS` UNMOVED — 30 pointers plus
one packed ``constant Params&``, 31, the platform ceiling exactly. The beta term adds
NO POINTER: its two coefficients are floats and ride inside the same record the five
scalars and three reflect rows already ride in, so the struct grows from eight fields
to ten and the binding count does not move.

``beta_plus``/``beta_minus`` are ``2*pi*beta*dt`` at each sign, rounded ONCE to float32
by :func:`.special_kz.beta_curl_coefficients` with ``magnetic=False``. That flag is not
a default: the two seams take OPPOSITE signs.

NOT WIRED AS AN ARM: ``plan_step`` assigns at most one arm per slot and this product
spans five. It reaches its two slots through ``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, _call, zero_metal_axes
from ..triton_kernels.symmetry import folded_axis_kinds
from . import folded_beta, shaders, special_kz, templates
from .device import Residency, compile_source
from .folded_fused_pair import (
    NEAR_SOURCE_INDEX,
    PACKED_BINDINGS,
    _COORDINATE,
    _LAST_FLAG,
    _carry_blocks,
    _constitutive_block,
    _far_carry_reasons,
    _mirror_phase_reasons,
    far_fill_axes,
    near_fill_axes,
    zero_metal_lines,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row, and it is what makes the cell worth anything: the one
#: corpus row here carries an electric source inside the seam. The machinery is
#: ``deposit_repair.repair_cells``' image closure over a FOLDED seam
#: (deposit_repair.py:217-258) -- the inverse of the forward carry
#: :func:`.folded_fused_pair.carried_destinations` implements -- and it is the same
#: claim :mod:`.folded_fused_pair` has carried since 2026-08-28 on the same seam with
#: the same geometry. ``repairable`` still refuses BY NAME the folds whose fill map it
#: cannot read.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_beta_real_fused_pair"

SLOT = "step_D"
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: ``backward`` and ``has_beta`` for the lifted curl. Named rather than passed: a
#: ``has_beta=False`` build is :mod:`.folded_fused_pair` under this family's name, and
#: every device leg would still pass because beta = 0 IS that family's arithmetic.
BACKWARD = True
HAS_BETA = True

#: The driver passes ONE launch of this plan performs (driver.py:3292-3303).
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The comment line the folded beta curl body opens its split-field block with — the
#: CUT POINT of the lift, matched as a PREFIX.
_RECURRENCE_MARK = "    // --- split-field recurrence"

#: The marker that separates a certified shader's signature from its body.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The two statements the real beta insert contributes to the lifted head, as
#: ``special_kz`` spells them. Checked present on every build.
_BETA_STATEMENTS: Tuple[str, ...] = (
    "    curl0 = curl0 - (beta_plus * b);",
    "    curl1 = curl1 - (beta_minus * a);",
)

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAMILY", "HAS_BETA", "PACKED_BINDINGS", "REPLACES", "SLOT",
    "MetalFoldedBetaRealFusedPairPlan", "beta_words", "certified_curl_head",
    "certified_curl_statements", "compile_folded_beta_real_fused_pair",
    "folded_beta_real_fused_pair_source",
    "metal_folded_beta_real_fused_pair_coverage",
    "plan_metal_folded_beta_real_fused_pair", "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

// TEN SCALARS, ONE BINDING, AND THE BINDING COUNT IS UNCHANGED AT 31. The two beta
// coefficients ride inside this record beside the three reflect rows the far carry
// needs, so the beta term costs nothing against the platform's ceiling (device.py:72)
// — which this family sits EXACTLY on and could not have paid.
struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                float beta_plus; float beta_minus;
                int rx; int ry; int rz; };

kernel void folded_beta_real_fused_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       e0      [[buffer(9)]],
    device float*       e1      [[buffer(10)]],
    device float*       e2      [[buffer(11)]],
    device float*       w0      [[buffer(12)]],
    device float*       w1      [[buffer(13)]],
    device float*       w2      [[buffer(14)]],
    device const float* ie0     [[buffer(15)]],
    device const float* ie1     [[buffer(16)]],
    device const float* ie2     [[buffer(17)]],
    device const float* kmx     [[buffer(18)]],
    device const float* sinvx   [[buffer(19)]],
    device const float* kmy     [[buffer(20)]],
    device const float* sinvy   [[buffer(21)]],
    device const float* kmz     [[buffer(22)]],
    device const float* sinvz   [[buffer(23)]],
    device const float* kp0     [[buffer(24)]],
    device const float* km0     [[buffer(25)]],
    device const float* kp1     [[buffer(26)]],
    device const float* km1     [[buffer(27)]],
    device const float* kp2     [[buffer(28)]],
    device const float* km2     [[buffer(29)]],
    constant Params&    prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE TEN SCALARS ARE UNPACKED INTO THE LIFTED BODIES' OWN NAMES, once, before
    // any of the spliced text runs. `beta_plus`/`beta_minus` are the beta parent's own
    // spelling, so the lifted head gets the names it wrote.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""


def _body_of(source: str, what: str) -> str:
    """One certified shader's body: everything between its signature and its brace."""
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            f"the certified {what} source no longer carries the body anchor; this "
            f"family lifts that body and would otherwise splice a truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"the certified {what} source does not end with '}}'")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than retyping "
            f"it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def certified_curl_head(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded BETA real ``step_D`` curl body DOWN TO the split-field recurrence.

    :func:`.folded_beta.folded_beta_curl_source`'s own output with the
    ``#include``/signature preamble removed and the cut taken at
    :data:`_RECURRENCE_MARK`. It carries the guard, the index decode, the ghost gather,
    the curl grouping, THE BETA INSERT, the cell-0 ownership mask and the folded
    top-plane mask.
    """
    body = _body_of(
        folded_beta.folded_beta_curl_source(codes, BACKWARD, HAS_BETA, contract),
        "folded beta real curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_RECURRENCE_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError("the lifted folded beta real curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded beta real curl body no longer opens its split-field "
        f"block with {_RECURRENCE_MARK!r}; this family cuts the lift there")


def certified_curl_statements(codes: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """The split-field recurrence's own lines, pulled out of the certified beta body.

    THREE statements per target here, not the complex beta family's five: the REAL beta
    template nests the intermediates exactly as ``shaders._CURL_TEMPLATE`` does, so the
    anchors are the shipped folded real weld's.
    """
    body = _body_of(
        folded_beta.folded_beta_curl_source(codes, BACKWARD, HAS_BETA, contract),
        "folded beta real curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    recurrence = []
    for target in range(3):
        recurrence.append((
            _line_starting(body, f"    float p{target} = ",
                           f"target {target}'s previous split field"),
            _line_starting(body, f"    float n{target} = ",
                           f"target {target}'s split-field recurrence"),
            _line_starting(body, f"    float v{target} = ",
                           f"target {target}'s stepped displacement"),
        ))
    aux_store = _line_starting(body, "    u0[ii] = ", "the three fu stores")
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded beta real curl no longer stores three targets on "
            f"one line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "recurrence": tuple(recurrence),
            "aux_store": aux_store, "stores": stores}


def folded_beta_real_fused_pair_source(
        codes: Sequence[int], phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, parities, walls, mode).

    THE SPLICE IS :func:`.folded_fused_pair.folded_fused_pair_source`'S, with the
    lifted beta head where its inline curl block stands. Every carry below the cut is
    that module's OWN emitter, called.

    ``phases`` is ``grid.mirror_phase(axis)`` per axis, +1 or -1 on a folded axis. It
    is a SOURCE specialisation and not a runtime scalar, which is measured: a runtime
    weight flushes every subnormal on this backend at BOTH signs
    (symmetry.py:657-686).
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills inside "
            "the D seam of a BETA run, and an unfolded real beta grid's D/E seam is "
            "beta_fused_electric_pair")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    phases = tuple(int(value) if value is not None else 0 for value in phases)
    if len(phases) != 3:
        raise ValueError(f"phases must be a per-axis triple, got {phases!r}")
    for axis, code in enumerate(codes):
        if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; a "
                f"plane's parity is +1 or -1 and an even-mirror default standing in "
                f"for a plane that declared otherwise is a run wrong by twice the "
                f"field wherever the parity mattered")
        if int(code) in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"this kernel's ghost carry relies on the two sets being disjoint")

    head = certified_curl_head(codes, contract)
    # THE BETA TERM MUST BE PRESENT, and IN THE LIFTED HEAD.
    for statement in _BETA_STATEMENTS:
        if statement not in head:
            raise AssertionError(
                f"the lifted folded beta real curl head does not carry "
                f"{statement.strip()!r}; this weld would be folded_fused_pair under "
                f"another name, and every device leg would still pass because "
                f"beta = 0 IS that family's arithmetic")

    statements = certified_curl_statements(codes, contract)
    body: List[str] = [head.rstrip("\n")]
    body.extend([
        "",
        "    (void)last_x; (void)last_y; (void)last_z;",
        "",
        "    // --- split-field recurrence (stepping._apply_pml_update:1905) ------",
    ])
    body.extend(statements["coefficients"])
    body.extend([
        "",
        "    // --- the constitutive coefficient, on the component's OWN axis -----",
        "    // shaders.py:310-312. The NEAR imaged cell takes the SAME entry: it",
        "    // differs from its source only on a folded axis whose iyee is 0, which",
        "    // is never the component's own. The FAR imaged cell does NOT: it moves",
        "    // along exactly the axis this is indexed on",
        "    // (folded_fused_pair._far_coefficient_lines).",
        "    float kp_0 = kp0[i], km_0 = km0[i];",
        "    float kp_1 = kp1[j], km_1 = km1[j];",
        "    float kp_2 = kp2[k], km_2 = km2[k];",
        "",
    ])

    for target in range(3):
        previous, recurrence, _ = statements["recurrence"][target]
        body.extend([previous, recurrence])
    body.append("")
    body.append(statements["aux_store"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        far = far_fill_axes(codes, target)
        if len(near) > 2:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the D "
                f"family that set is at most two axes (fields.IYEE_SHIFTS)")
        if len(far) > 1 or set(far) & set(near):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against near "
                f"axes {near}; for the D family the two sets are COMPLEMENTARY")
        _, _, displacement = statements["recurrence"][target]
        body.append("")
        owned = ([f"({_COORDINATE[axis]} == 0)" for axis in near]
                 + [_LAST_FLAG[axis] for axis in far])
        if owned:
            body.extend([
                f"    // component {target}: the fills image "
                + ", ".join(
                    [f"stored cell 0 on {_COORDINATE[axis]} (near)" for axis in near]
                    + [f"the top plane on {_COORDINATE[axis]} (far)"
                       for axis in far]) + ",",
                "    // so those cells are OWNED BY THEIR SOURCE THREADS. This one "
                "stops after fu.",
                f"    if (!({' || '.join(owned)})) {{",
            ])
            indent = "        "
        else:
            body.append(f"    // component {target}: no folded axis images a cell "
                        f"of this component")
            indent = "    "
        body.append(indent + displacement.strip())
        cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
        body.extend(cleared or [f"{indent}// no walled axis clears this target"])
        body.append(indent + statements["stores"][target])
        body.extend(_constitutive_block(target, "ii", f"v{target}", indent, "o"))
        if owned:
            body.append("")
            body.extend(_carry_blocks(target, near, far, phases, zero_metal, indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__BODY__": "\n".join(body),
    })


def compile_folded_beta_real_fused_pair(
        codes: Sequence[int], phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, parities, walls, mode)."""
    return compile_source(folded_beta_real_fused_pair_source(
        codes, phases, zero_metal, contract)).folded_beta_real_fused_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_beta_real_fused_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        beta_probe: Any = None) -> Coverage:
    """May ONE dispatch span folded beta real ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of :mod:`.folded_beta`'s two ROUTING verdicts (a fold MANDATORY, a
    nonzero beta MANDATORY, REAL storage) plus the seam clauses, which is what keeps
    this product disjoint from :mod:`.folded_fused_pair` and from
    :mod:`.beta_fused_electric_pair` in both directions.
    """
    reasons: List[str] = []

    # THE REAL BETA CURL PREDICATE TAKES NO PROBE, and the absence is the arm's own
    # fact rather than an omission here: a real beta curl emits no complex product, so
    # there is no expansion arm to bind. `beta_probe` is accepted on this signature for
    # one reason -- the arm-table and gate call sites are uniform across the beta
    # families -- and is deliberately unused.
    del beta_probe
    curl = folded_beta.folded_beta_composition_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"folded beta real curl half: {reason}"
                       for reason in curl.reasons)
    electric = folded_beta.folded_beta_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not electric.covered:
        reasons.extend(f"folded beta real constitutive half: {reason}"
                       for reason in electric.reasons)

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
            f"it BETWEEN step_D and update_E (driver.py:3294-3299)"),
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

class MetalFoldedBetaRealFusedPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded real beta run."""

    __slots__ = ("residency", "volumes", "codes", "phases", "zero_metal", "shape",
                 "dtdx", "beta_plus", "beta_minus", "reflect", "params")

    family = "folded beta real PML D-curl/mirror-fill/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phases", "zero_metal", "reflect",
                   "beta_plus", "beta_minus")

    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phases: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 beta_plus: float, beta_minus: float,
                 reflect: Sequence[Optional[int]], params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phases = tuple(int(value) for value in phases)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        # float(): `beta_curl_coefficients` already rounded ONCE to float32 and float()
        # of a numpy.float32 preserves the word, the sign of a zero included.
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

    ``magnetic=False`` because :data:`CURL_SUB_STEP` is ``step_D`` -- NOT a default,
    because the two seams take opposite signs; ``complex_storage=False`` because this
    is the real arm. Neither is a parameter.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=False)


def _params_tensor(shape: Sequence[int], dtdx: float, beta_plus: float,
                   beta_minus: float, reflect: Sequence[Optional[int]],
                   device: str) -> Any:
    """The ten scalars as one 40-byte device record, built once at plan time.

    ``-1`` stands for an axis with no reflect row, the sentinel
    :func:`.folded_fused_pair._params_tensor` uses and for its reason: the guard that
    would read it is emitted only for an axis :func:`far_fill_axes` returned.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
         ("dtdx", "<f4"), ("beta_plus", "<f4"), ("beta_minus", "<f4"),
         ("rx", "<i4"), ("ry", "<i4"), ("rz", "<i4")]))
    nx, ny, nz = (int(n) for n in shape)
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record[0] = ((nx, ny, nz, nx * ny * nz, np.float32(dtdx),
                  np.float32(beta_plus), np.float32(beta_minus)) + rows)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_beta_real_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        beta_probe: Any = None,
        ) -> Optional[MetalFoldedBetaRealFusedPairPlan]:
    """Build the folded beta real fused D/E plan, or ``None`` when refused."""
    if not metal_folded_beta_real_fused_pair_coverage(
            fields, pml, sources, residency, beta_probe).covered:
        return None
    from ..stepping import _far_reflect_rows  # noqa: PLC0415

    grid = fields.grid
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
        [bind(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
        + [bind("fu_" + name, getattr(fields, "fu_" + name))
           for name in ("Dx", "Dy", "Dz")]
        + [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
        + [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
        + [bind("f_w_" + name, getattr(fields, "f_w_" + name))
           for name in ("Ex", "Ey", "Ez")]
        + [bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
           for name in ("Ex", "Ey", "Ez")]
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015).
        + [bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
           for axis in "xyz" for stem in ("kms", "sinv")]
        + [bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                constant=True)
           for axis in "xyz" for stem in ("kps", "kms")])
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_beta_real_fused_pair(
                codes, phases, walls, mode)

    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaRealFusedPairPlan(
        residency, volumes, codes, phases, walls, grid.shape, dtdx,
        beta_plus, beta_minus, reflect,
        _params_tensor(grid.shape, dtdx, beta_plus, beta_minus, reflect,
                       residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded beta real fused D/E pair cannot fill {slot}",))
    return metal_folded_beta_real_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedBetaRealFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_real_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — every other fused pair's reason."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded beta real fused D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta real fused D/E pair: ",
                          noun=("folded beta real PML D-curl/mirror-fill/"
                                "stored-E pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()

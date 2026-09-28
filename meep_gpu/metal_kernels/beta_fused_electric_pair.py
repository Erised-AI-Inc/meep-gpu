"""The fused electric D/E pair on a REAL ``grid.beta`` run: ``step_D`` -> ``update_E``.

:mod:`.fused_electric_pair` WITH ONE EMITTER SWAPPED, and the swap is the whole
module -- the D-side mirror of what :mod:`.beta_fused_magnetic_pair` is to
:mod:`.fused_magnetic_pair`. The certified ordinary curl body is replaced by the
certified REAL beta curl body (:func:`.special_kz.beta_curl_source` with
``backward=True``) and NOTHING ELSE CHANGES: the same wall-clear carry, the same
three-line seam, the same constitutive lift, the same 30 pointers, the same plan
class shape.

WHY THE SWAP IS SOUND, measured rather than argued. The beta curl and the certified
curl differ by TWO STATEMENTS and two scalars (special_kz.py:392-393)::

    curl0 = curl0 - (beta_plus  * b);
    curl1 = curl1 - (beta_minus * a);

inserted AFTER the ``dtdx`` curl and BEFORE the ownership mask -- the array path's
own order (stepping.py:467-474 against :479 on this side). Everything the weld
depends on is unchanged: the beta template ends its parameter list with the same
``uint idx [[thread_position_in_grid]])`` anchor, declares ``at_x``/``at_y``/``at_z``
in the same place, and stores the three targets on the same single line
``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` that the wall clear splices in front of.
Those three are ASSERTED at emit time, not assumed.

WHY THE CONSTITUTIVE HALF NEEDS NOTHING. ``update_E`` reads nothing beta-dependent:
``grid.beta`` enters the two CURLS only (stepping.py:384-391, :467-474), never
``_apply_constitutive_pml`` (:2130). That is the same measurement
:func:`.special_kz.beta_run_constitutive_coverage` already ships on, and the reason
a real beta run's ``update_E`` is served by the CERTIFIED ordinary constitutive
today. So this weld's constitutive half is lifted through
:func:`.fused_electric_pair.certified_constitutive_body`, the shipped weld's own
lift, imported rather than copied -- including its three-way crossing rename, which
is what puts the E targets on ``e`` and the inverse epsilon on ``ie``.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

    3 D  +  3 fu_D  +  3 H  +  6 curl coefficients (kms/sinv per axis)
  + 3 E  +  3 f_w_E  +  3 inverse epsilon  +  6 constitutive coefficients  =  30

identical to :mod:`.fused_electric_pair`'s, because the beta term takes NO BUFFER:
its two coefficients are SCALARS and they ride in the same ``constant Params&`` the
other five already ride in. The struct grows from five fields to seven and the
POINTER count does not move -- 31 bindings, which is
:data:`.device.MAX_BUFFER_BINDINGS` exactly. That is the difference between this
cell and the magnetic one: the magnetic twin sits at 28 with three slots spare,
this one has none, and the three inverse-epsilon volumes are why (``update_H`` is
``H = B`` with mu = 1 baked into the array path; ``update_E`` reads
``D * inverse_epsilon_for(component)``, stepping.py:1011-1013).

``beta_plus``/``beta_minus`` are ``2*pi*beta*dt`` at each sign, rounded ONCE to
float32 by :func:`.special_kz.beta_curl_coefficients` -- stepping.py:797-811, and
that function is CALLED rather than re-derived here, so the words this kernel
receives are the array path's own bits, the sign of a zero included. It is called
with ``magnetic=False``: the two signs SWAP between the seams
(``step_db.cpp``'s ``cc`` sign follows the sub-step), and a weld that reused the
magnetic pair's pair would run a correctly-shaped, converged, wrong dispersion.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH, and why it used to be zero
===========================================================================

ONE seam-instance: ``examples:refl-angular-kz2d.py``, the same row the magnetic
twin serves on the other seam. :mod:`.beta_fused_magnetic_pair`'s docstring records
that "the D/E partner cell is worth ZERO -- that row injects electrically inside the
D/E seam -- and is not built", and THAT SENTENCE IS RETIRED BY THIS MODULE rather
than contradicted: it was true of the products shipping when it was written, before
``meep_gpu.deposit_repair`` existed. The deposit is now carried (see
:data:`CARRIES_DEPOSIT_REPAIR`), so the row is reachable and the cell is worth one.

REGISTERED UNWIRED, for :func:`.fused_electric_pair.register_arms`'s reason and the
one specific to a weld that inherits a wired half: were it wired it would contend on
``step_D`` with ``special_kz``'s real-beta curl arm, which admits exactly the same
configurations, and ``_select_slot`` would leave the slot UNSELECTED -- taking the
certified curl off the device as well. It reaches its two slots through
``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import shaders, special_kz
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_electric_pair import (
    _BODY_ANCHOR,
    _CURL_STORE,
    certified_constitutive_body,
    zero_metal_mask,
)
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit. Flipping it is a claim
#: about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, and it is the whole reason the cell is worth
#: building: the one corpus row this pair reaches carries an ELECTRIC source, so
#: without the bracket it would serve nothing at all -- which is exactly what
#: :mod:`.beta_fused_magnetic_pair`'s docstring recorded when the repair did not
#: exist. Nothing here is new: the electric branch is the one
#: :mod:`.fused_electric_pair` already drives, with the same ``'D'`` pair label and
#: the same two wrappers, and this family refuses every mirrored axis by name in its
#: own predicate below, so no configuration it admits has a mirror image for a POINT
#: repair to miss.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "beta_fused_electric_pair"

#: The slot this arm holds a row on, and the driver passes one launch performs
#: (driver.py:3292-3303). Declared, never inferred from the slot name.
SLOT = "step_D"
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The bindings this shape needs with the seven scalars bound SEPARATELY, and with
#: them packed into one ``constant Params&``. Spelled as data so the gate compiles
#: the refuted signature from the number this docstring argues from.
SEPARATE_SCALAR_BINDINGS = 37
PACKED_BINDINGS = 31

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the beta fused electric pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "MetalBetaFusedElectricPairPlan",
    "beta_fused_electric_pair_source", "certified_beta_curl_body",
    "compile_beta_fused_electric_pair",
    "metal_beta_fused_electric_pair_coverage",
    "plan_metal_beta_fused_electric_pair", "refuted_separate_scalar_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                float beta_plus; float beta_minus; };

kernel void beta_fused_electric_pair_step(
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
    // THE SEVEN SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any of the lifted text runs. Everything below this line is then
    // character-for-character what `special_kz.beta_curl_source` and
    // `shaders.constitutive_source` emit, plus the wall clear.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;

__BODY__
}
"""


def certified_beta_curl_body(codes: Sequence[int],
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """The REAL beta ``step_D`` curl kernel's BODY, lifted from its own emitter.

    Not a transcription: this is :func:`.special_kz.beta_curl_source`'s output with
    the ``#include``/signature preamble and the closing brace removed, exactly as
    :func:`.fused_electric_pair.certified_curl_body` lifts the ordinary one.
    ``backward`` is True and never a parameter -- this is the D seam, and
    ``step_B``'s forward strides are :mod:`.beta_fused_magnetic_pair` -- and
    ``has_beta`` is True and never a parameter either: a ``has_beta=False`` weld
    would be the ordinary product under a second name, which is what
    :mod:`.fused_electric_pair` already is.
    """
    source = special_kz.beta_curl_source(codes, True, True, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified beta curl source no longer carries the body anchor; "
            "this family lifts that body and would otherwise splice a truncated "
            "kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified beta curl source does not end with '}'")
    return body[: -len("}\n")]


def beta_fused_electric_pair_source(codes: Sequence[int],
                                    zero_metal: Sequence[bool],
                                    contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    THE SPLICE IS :func:`.fused_electric_pair.fused_electric_pair_source`'S, step for
    step, with :func:`certified_beta_curl_body` where its ``certified_curl_body``
    stands. The anchors it depends on are checked here rather than inherited: an
    emitter that moved the store line or renamed a source read would otherwise
    produce a kernel that compiles and is wrong.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_beta_curl_body(codes, contract)
    # THE BETA TERM MUST BE PRESENT. A lift that silently produced the has_beta=0
    # arm would be the ordinary weld wearing this family's name, and every device
    # leg would still pass because beta = 0 IS the ordinary arithmetic.
    for statement in ("curl0 = curl0 - (beta_plus * b);",
                      "curl1 = curl1 - (beta_minus * a);"):
        if statement not in curl:
            raise AssertionError(
                f"the lifted beta curl body does not carry {statement!r}; this "
                f"weld would be the ordinary product under another name")
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified beta curl body no longer stores the three targets on "
            "one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines — the shipped D/E weld's, unchanged.
    # The inverse-epsilon factor stays on the right and D stays on the LEFT, which is
    # the order the array path writes (stepping.py:1011-1013).
    for target in range(3):
        old = f"    float src{target} = g{target}[ii] * ie{target}[ii];\n"
        if old not in electric:
            raise AssertionError(
                f"the certified E constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float src{target} = v{target} * ie{target}[ii];\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D. After the seam there must
    # be no `g` left in the electric half at all -- one surviving read would take the
    # magnetic field as a displacement, which is a smooth, plausible, entirely wrong
    # answer rather than a crash.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted E constitutive half still reads g{target}, which in the "
                f"fused signature is H{'xyz'[target]} and not D{'xyz'[target]}")

    body = "".join((
        curl,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_beta_fused_electric_pair(codes: Sequence[int],
                                     zero_metal: Sequence[bool],
                                     contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(beta_fused_electric_pair_source(
        codes, zero_metal, contract)).beta_fused_electric_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 37-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and require
    the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an
    argument into a measurement. The body touches every buffer -- what is being
    measured is the SIGNATURE, and a body the compiler could drop would let dead-code
    elimination decide the answer. Two bindings wider than
    :func:`.fused_electric_pair.refuted_separate_scalar_source`'s, which is exactly
    the beta pair.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "e0", "e1", "e2",
               "w0", "w1", "w2"]
    read = ["g0", "g1", "g2", "ie0", "ie1", "ie2",
            "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
            "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    lines: List[str] = []
    slot = 0
    for name in written:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&      {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("dtdx", "beta_plus", "beta_minus"):
        lines.append(f"    constant float&     {name:<10}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a design invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch = " + ".join(f"{name}[0]" for name in read)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz)"
        " + beta_plus + beta_minus;",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_beta_fused_electric_pair_coverage(fields: Any, pml: Any,
                                            sources: Any = None,
                                            residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> wall -> ``update_E`` on a real beta run?

    The conjunction of the two halves' OWN shipped predicates plus the seam clauses,
    which is :func:`.fused_electric_pair.metal_fused_electric_pair_coverage`'s
    construction with each half's predicate replaced by the real-beta arm's:

    * :func:`.special_kz.beta_pml_curl_coverage` on ``step_D`` -- the certified curl
      predicate with the beta clause INVERTED;
    * :func:`.special_kz.beta_run_constitutive_coverage` on side ``"E"`` -- the
      certified constitutive predicate with the same clause inverted.

    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.
    """
    reasons: List[str] = []

    curl = special_kz.beta_pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = special_kz.beta_run_constitutive_coverage(fields, pml, "E", residency)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch -- which for this family it does, and
    # which is the only reason the cell is worth anything at all.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does NOT disqualify the pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. `special_kz`'s own clause 5 already refuses a mirror plane on
    # both halves -- a FOLDED beta run belongs to :mod:`.folded_beta` -- so this
    # restates by NAME what that refusal buys on THIS seam.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled. A grid that cannot answer would silently be treated as unwalled, which
    # is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # The E-side predicate already refuses it; restated because this family's kernel
    # bakes the plain product and a reader should not have to chase the other
    # predicate to learn that.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBetaFusedElectricPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes on a real beta run.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.
    """

    __slots__ = ("residency", "volumes", "codes", "zero_metal", "shape", "dtdx",
                 "beta_plus", "beta_minus", "params")

    family = "fused real-beta PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "zero_metal", "beta_plus", "beta_minus")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, beta_plus: float,
                 beta_minus: float, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        # float(): `beta_curl_coefficients` already rounded ONCE to float32 and
        # float() of a numpy.float32 preserves the word, the sign of a zero included.
        # Rounding a second time here would be a second rounding.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def _params_tensor(shape: Sequence[int], dtdx: float, beta_plus: float,
                   beta_minus: float, device: str) -> Any:
    """The seven scalars as one 28-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for
    :func:`.beta_fused_magnetic_pair._params_tensor`'s reason:
    :meth:`.Residency.mirror` binds float32 and complex64 volumes and refuses
    anything else BY NAME, and this record is neither. Built here, once, at plan
    time. Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4"), ("dtdx", "<f4"),
                                         ("beta_plus", "<f4"),
                                         ("beta_minus", "<f4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx),
                 np.float32(beta_plus), np.float32(beta_minus))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_beta_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalBetaFusedElectricPairPlan]:
    """Build the fused beta D/E plan, or ``None`` when the seam is refused.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING.
    """
    if not metal_beta_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)
    # THE COEFFICIENTS ARE THE CURL ARM'S OWN, called rather than re-derived:
    # stepping.py:797-811 evaluated on the objects the array path evaluates it on, so
    # the intermediate precision is the array path's too. `magnetic=False` is the D
    # seam's sign convention and is NOT a default: the two seams take opposite signs.
    beta_plus, beta_minus = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=False, complex_storage=False)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
    stored = [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
        for name in ("Ex", "Ey", "Ez")]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one (stepping.py:948 vs :1015). That is the OPPOSITE pairing to the B/H pair,
    # and the kernel cannot tell.
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + stored + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_beta_fused_electric_pair(codes, walls, mode)

    dtdx = grid.dt / grid.dx
    return MetalBetaFusedElectricPairPlan(
        residency, volumes, codes, walls, grid.shape, dtdx, beta_plus, beta_minus,
        _params_tensor(grid.shape, dtdx, beta_plus, beta_minus, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"beta fused electric pair cannot fill {slot}",))
    return metal_beta_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalBetaFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_beta_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    Unwired for :func:`.fused_electric_pair.register_arms`'s reason, and for the one
    specific to a weld that inherits a wired half: were it wired it would contend on
    ``step_D`` with ``special_kz``'s real-beta curl arm, which admits exactly the
    same configurations, and ``_select_slot`` would leave the slot UNSELECTED -- the
    certified curl would come off the device as well.

    THE OTHER HALF IS THE ABSORB TABLE:
    ``launch.FUSED_PAIR_ARMS["beta_fused_electric_pair"]`` declares which arm each of
    the two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "beta fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="beta fused electric D/E pair: ",
                          noun="fused real-beta PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()

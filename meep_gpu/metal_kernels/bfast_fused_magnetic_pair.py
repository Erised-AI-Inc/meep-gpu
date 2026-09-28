"""The fused magnetic B/H pair on a BFAST run: ``step_B`` -> ``update_H``.

:mod:`.fused_magnetic_pair` WITH ONE EMITTER SWAPPED, exactly as
:mod:`.beta_fused_magnetic_pair` is, and the second-cheapest weld in the tree for the
same reason: the certified ordinary curl body is replaced by the certified BFAST curl
body — :func:`.bfast_curl.bfast_curl_source` with ``backward=False`` — and nothing
else about the construction changes. Same wall-clear carry, same three-line seam, same
constitutive lift, same plan shape.

WHERE IT IS NOT FREE, AND THE NUMBER IS THE WHOLE STORY: **this weld sits EXACTLY on
the platform's binding ceiling.** The BFAST curl carries THREE MORE POINTERS than the
ordinary one — the Tustin filter's per-component state ``f_bfast_Bx/By/Bz``
(bfast_curl.py:287-290) — so the fused signature is

    3 B  +  3 fu_B  +  3 E  +  3 f_bfast_B  +  6 curl coefficients
  + 3 H  +  3 f_w_H  +  6 constitutive coefficients                    =  30

pointers, plus one packed ``constant Params&`` at buffer(30). That is 31 bindings,
and ``device.MAX_BUFFER_BINDINGS`` is 31 — attribute indices run 0..30 and a 32nd is
a COMPILE ERROR. There is NO margin: one more pointer and this cell would be
UNFUSABLE ON METAL rather than merely tight, which is the verdict the 2026-08-20
matrix hands the D/E off-diagonal cells. The gate's ``binding_ceiling`` leg compiles
the 35-binding separate-scalar signature, requires the FAILURE, and then compiles,
launches and reads back the packed 31-binding one, so the margin-of-zero is a
measurement and not an inference from a docstring. :mod:`.folded_fused_pair` already
ships at this exact width, which is why 30 is known to be reachable rather than
hoped for.

WHY THE SWAP IS SOUND, measured rather than argued. The BFAST curl and the certified
curl differ by ONE INSERTED BLOCK — the Tustin tail (bfast_curl.py, transcribing
``stepping._bfast_term`` :896-904) — placed AFTER the ``dtdx`` curl and BEFORE the
ownership mask, which is the array path's own fold order (:364-368 after :342, before
:369). Everything the weld depends on is unchanged: the same
``uint idx [[thread_position_in_grid]])`` body anchor, the same
``bool at_x = (i == 0), ...`` declaration, and the same single store line
``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` the wall clear splices in front of. All
three are ASSERTED at emit time.

ONE ORDERING FACT THIS WELD INHERITS AND MUST NOT DISTURB. The BFAST curl declares
``at_x``/``at_y``/``at_z`` EARLY — before the tail, because the tail's advance carries
the same ownership predicate the curl's mask does (:902 precedes :903) — where the
ordinary curl declares them just before its mask. The wall clear this family splices
in reads those same three flags and is spliced in front of the STORE, which is after
both. So the flags are in scope at the splice on either emitter, and the splice site
is unchanged. That is checked at emit time by the store-line anchor, not assumed.

WHY THE CONSTITUTIVE HALF NEEDS NOTHING. ``update_H`` reads nothing BFAST-dependent:
:func:`.bfast_curl.bfast_run_constitutive_coverage` is already the certified
constitutive predicate with the BFAST clause inverted, and its builder calls
``launch.ConstitutivePlan`` unchanged. So a BFAST run's ``update_H`` is served by the
CERTIFIED ordinary constitutive today, and this weld lifts it through
:func:`.fused_magnetic_pair.certified_constitutive_body` — the shipped weld's own
lift, imported rather than copied.

WHAT THE CORPUS SAYS THIS IS WORTH. ONE seam-instance, reachable:
``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``, measured at
``parity/meep_gpu/results/below_the_cut_census_2026-08-20/census.json``. It carries an
ELECTRIC source only, so the magnetic seam (driver.py:3283-3284) is clear; it is
all-periodic, so the inline ``zero_metal_B`` carry is compiled OUT on that row and the
gate exercises the walled specialisation separately. The D/E partner cell is worth
ZERO and is not built.

REGISTERED UNWIRED, for :func:`.fused_magnetic_pair.register_arms`'s reason.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.bfast_curl import BFAST_STATE_NAMES, bfast_curl_coefficients
from ..triton_kernels.coverage import Coverage, MAGNETIC_FIELD_TYPE
from . import bfast_curl, shaders
from .coverage import zero_metal_axes
from .device import Residency, compile_source
from .fused_magnetic_pair import (
    certified_constitutive_body, zero_metal_mask, _BODY_ANCHOR, _CURL_STORE,
)
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "bfast_fused_magnetic_pair"

#: The slot this arm holds a row on, and the driver passes one launch performs.
SLOT = "step_B"
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The bindings this shape needs with the eleven scalars bound SEPARATELY, and with
#: them packed into one ``constant Params&``. Spelled as data so the gate compiles
#: the refuted signature from the number this docstring argues from. 41 is four over
#: :data:`.device.MAX_BUFFER_BINDINGS` even before the packing; 31 is exactly AT it.
SEPARATE_SCALAR_BINDINGS = 41
PACKED_BINDINGS = 31

__all__ = [
    "FAMILY", "PACKED_BINDINGS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalBfastFusedMagneticPairPlan", "certified_bfast_curl_body",
    "compile_bfast_fused_magnetic_pair", "bfast_fused_magnetic_pair_source",
    "metal_bfast_fused_magnetic_pair_coverage",
    "plan_metal_bfast_fused_magnetic_pair", "refuted_separate_scalar_source",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                float k1_0; float k2_0; float k1_1; float k2_1;
                float k1_2; float k2_2; };

kernel void bfast_fused_magnetic_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       s0      [[buffer(9)]],
    device float*       s1      [[buffer(10)]],
    device float*       s2      [[buffer(11)]],
    device float*       h0      [[buffer(12)]],
    device float*       h1      [[buffer(13)]],
    device float*       h2      [[buffer(14)]],
    device float*       w0      [[buffer(15)]],
    device float*       w1      [[buffer(16)]],
    device float*       w2      [[buffer(17)]],
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
    // THE ELEVEN SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any of the lifted text runs. Everything below this line is then
    // character-for-character what `bfast_curl.bfast_curl_source` and
    // `shaders.constitutive_source` emit, plus the wall clear.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float k1_0 = prm.k1_0, k2_0 = prm.k2_0;
    float k1_1 = prm.k1_1, k2_1 = prm.k2_1;
    float k1_2 = prm.k1_2, k2_2 = prm.k2_2;

__BODY__
}
"""


def certified_bfast_curl_body(codes: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF) -> str:
    """The BFAST ``step_B`` curl kernel's BODY, lifted from its own emitter.

    Not a transcription: this is :func:`.bfast_curl.bfast_curl_source`'s output with
    the ``#include``/signature preamble and the closing brace removed, exactly as
    :func:`.fused_magnetic_pair.certified_curl_body` lifts the ordinary one.
    ``backward`` is False and never a parameter — this is the B seam — and
    ``has_bfast`` is True and never a parameter either: a ``has_bfast=False`` weld
    would be the ordinary product under a second name.
    """
    source = bfast_curl.bfast_curl_source(codes, False, contract, True)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified BFAST curl source no longer carries the body anchor; "
            "this family lifts that body and would otherwise splice a truncated "
            "kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified BFAST curl source does not end with '}'")
    return body[: -len("}\n")]


def bfast_fused_magnetic_pair_source(codes: Sequence[int],
                                     zero_metal: Sequence[bool],
                                     contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    THE SPLICE IS :func:`.fused_magnetic_pair.fused_magnetic_pair_source`'S, step for
    step, with :func:`certified_bfast_curl_body` where its ``certified_curl_body``
    stands.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_bfast_curl_body(codes, contract)
    # THE TAIL MUST BE PRESENT. A lift that silently produced the has_bfast=0 arm
    # would be the ordinary weld wearing this family's name, and every device leg
    # would still pass because a zero k vector IS the ordinary arithmetic.
    for statement in ("float st0 = s0[ii];",
                      "s0[ii] = st0 + adv0;",
                      "curl0 = curl0 - adv0;"):
        if statement not in curl:
            raise AssertionError(
                f"the lifted BFAST curl body does not carry {statement!r}; this "
                f"weld would be the ordinary product under another name")
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified BFAST curl body no longer stores the three targets on "
            "one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z.\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    magnetic = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines — the shipped weld's, unchanged.
    for target in range(3):
        old = f"    float src{target} = g{target}[ii];\n"
        if old not in magnetic:
            raise AssertionError(
                f"the certified H constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        magnetic = magnetic.replace(
            old,
            f"    // THE SEAM: the register step_B just wrote, not a reload of "
            f"B{'xyz'[target]}.\n"
            f"    float src{target} = v{target};\n")

    body = "".join((
        curl,
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2083) --\n",
        magnetic,
    ))
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_bfast_fused_magnetic_pair(codes: Sequence[int],
                                      zero_metal: Sequence[bool],
                                      contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(bfast_fused_magnetic_pair_source(
        codes, zero_metal, contract)).bfast_fused_magnetic_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 41-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and require
    the failure. Ten bindings over the ceiling rather than the shipped weld's one,
    which is the point — this family's packing is not a preference, it is the only
    signature that exists at all.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "s0", "s1", "s2",
               "h0", "h1", "h2", "w0", "w1", "w2"]
    read = ["g0", "g1", "g2",
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
    for name in ("dtdx", "k1_0", "k2_0", "k1_1", "k2_1", "k1_2", "k2_2"):
        lines.append(f"    constant float&     {name:<8}[[buffer({slot})]],")
        slot += 1
    assert slot == SEPARATE_SCALAR_BINDINGS, (slot, SEPARATE_SCALAR_BINDINGS)
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
        " + k1_0 + k2_0 + k1_1 + k2_1 + k1_2 + k2_2;",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_bfast_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None,
                                             residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_B`` -> wall -> ``update_H`` on a BFAST run?

    The conjunction of the two halves' OWN shipped predicates plus the seam clauses,
    which is :func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage`'s
    construction with each half's predicate replaced by the BFAST arm's:

    * :func:`.bfast_curl.bfast_pml_curl_coverage` on ``step_B``;
    * :func:`.bfast_curl.bfast_run_constitutive_coverage` on side ``"H"``.

    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.
    """
    reasons: List[str] = []

    curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_B", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    magnetic = bfast_curl.bfast_run_constitutive_coverage(fields, pml, "H",
                                                          residency)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE STATE VOLUMES. The Tustin filter's per-component state is bound as three
    # POINTERS and there is no room to discover at launch time that one is missing:
    # the curl half's own predicate checks them, and this restates the check by NAME
    # because the weld is what BINDS them.
    for name in BFAST_STATE_NAMES["step_B"]:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated; this weld binds it directly")

    # THE TWO FILLS. `bfast_curl`'s own clause 5 already refuses a mirror plane on
    # both halves, so this restates by NAME what that refusal buys on THIS seam.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_B and "
                f"stepping.fill_folded_far_ghosts_B both run inside this seam "
                f"(driver.py:3285-3287) and neither is carried by this family")

    # zero_metal_B is CARRIED, so the grid must be able to answer which axes are
    # walled. A grid that cannot answer would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBfastFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes on a BFAST run.

    ``launches_per_run`` is the base's 1: this plan unpacks one argument tuple and
    calls one function.
    """

    __slots__ = ("residency", "volumes", "codes", "zero_metal", "shape", "dtdx",
                 "ks", "params")

    family = "fused BFAST PML B-curl/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "zero_metal", "ks")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, ks: Sequence[float],
                 params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        # Already f32-rounded by `bfast_curl_coefficients`; float() keeps the bits.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def _params_tensor(shape: Sequence[int], dtdx: float, ks: Sequence[float],
                   device: str) -> Any:
    """The eleven scalars as one 44-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for
    :func:`.fused_magnetic_pair._params_tensor`'s reason. Built here, once, at plan
    time. Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
         ("dtdx", "<f4"), ("k1_0", "<f4"), ("k2_0", "<f4"), ("k1_1", "<f4"),
         ("k2_1", "<f4"), ("k1_2", "<f4"), ("k2_2", "<f4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx),
                 *(np.float32(value) for value in ks))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_bfast_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalBfastFusedMagneticPairPlan]:
    """Build the fused BFAST B/H plan, or ``None`` when the seam is refused.

    ``functions`` is the mutation seam and is FORWARDED; dropping it would silently
    disarm every mutation leg.
    """
    if not metal_bfast_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)
    # THE SIX SCALARS ARE THE CURL ARM'S OWN, called with the grid's OWN declared
    # invariance flags (never a shape test) and the magnetic side.
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant, magnetic=True)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Bx", "By", "Bz")]
    electric = [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
    state = [bind(name, getattr(fields, name))
             for name in BFAST_STATE_NAMES["step_B"]]
    magnetic = [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Hx", "Hy", "Hz")]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
    # one (launch.py:109-124; CONSTITUTIVE_SIDES['H']['half_integer'] is False).
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + state + magnetic + workspace
                + curl_coefficients + constitutive_coefficients)
    assert len(pointers) + 1 == PACKED_BINDINGS, (len(pointers), PACKED_BINDINGS)

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_bfast_fused_magnetic_pair(codes, walls, mode)

    dtdx = grid.dt / grid.dx
    return MetalBfastFusedMagneticPairPlan(
        residency, volumes, codes, walls, grid.shape, dtdx, ks,
        _params_tensor(grid.shape, dtdx, ks, residency.device), pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"BFAST fused magnetic pair cannot fill {slot}",))
    return metal_bfast_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalBfastFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_bfast_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    Unwired for :func:`.fused_magnetic_pair.register_arms`'s reason, and for the one
    specific to a weld that inherits a wired half: were it wired it would contend on
    ``step_B`` with ``bfast_curl``'s own curl arm, which admits exactly the same
    configurations, and ``_select_slot`` would leave the slot UNSELECTED — taking
    the certified curl off the device as well.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "BFAST fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="BFAST fused magnetic B/H pair: ",
                          noun="fused BFAST PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()

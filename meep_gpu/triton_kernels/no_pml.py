"""The NO-PML curl path: kernel, predicate and plan for a run with no absorber.

The public experimental planner composes this product for each curl only when
its predicate admits the configuration and the PML curl predicate refuses it.
An overlap fails closed. Production dispatch remains disabled:
``fastpath.plan_fast_path`` still returns ``None`` on every branch. Host coverage
and refusal reporting stay importable when the optional Triton dependency is
absent; kernel helpers are imported lazily only after admission or at launch.

WHAT THIS IS FOR, stated before the physics, because the honest answer is narrow.
It accelerates ZERO scripts of either validation corpus. Of the 57 example scripts
carrying boundary facts (``parity/meep_gpu/results/survey_after_monitors/survey.jsonl``)
exactly TWO declare no boundary layer (54 are ``PML``, 1 ``Absorber``) —
``material-dispersion.py``, which this predicate refuses anyway on clauses 7 and 2
(``run_k_points`` sweeps 100 nonzero k-points, which also forces complex storage)
and whose cell is one voxel, and ``phase_in_material.py``, 120x120 with no source.
MEEP's ``python/tests`` adds nothing steppable. The value is two things:

1. it closes ``2d_plain``, one of the nine benchmark cases, and closes it FULLY —
   both heavy sub-steps, 90.3% of the Triton whole step at tier 4;
2. it is the package's best bit-identity TEST VEHICLE, carrying no PML arithmetic
   at all, which is what let it locate the step-67 divergence in the simplest
   configuration the engine has (see BUDGET below).

WHAT COLLAPSES WHEN THE ABSORBER GOES AWAY
------------------------------------------
``stepping._apply_curl`` (stepping.py:460-509) has four branches. With
``_pml_is_active(pml)`` False and ``fields.condfac_for(target)`` None, all of the
machinery falls away and the whole sub-step tail is one line::

    target -= curl                                       # stepping.py:508

There is no ``fu`` auxiliary (``_require_pml_storage`` is not called, and
``fu_*`` is None), no ``kms``/``sinv`` pair, and no split-field recurrence. The
curl itself — the term table, the ghost rule, the grouping and the ownership
mask — is UNCHANGED, which is why this kernel's first two thirds are a verbatim
copy of ``kernels.pml_curl_step``'s.

``update_H`` returns immediately (stepping.py:916): without PML, H is not stored
and ``get_H`` serves the B array itself. ``update_E`` returns immediately too
unless ``fields.stores_E`` (stepping.py:958). So on a plain run
``FdtdDriver.step``'s four heavy sub-steps reduce to TWO, and this one kernel
covers both of them.

THE THING THAT IS NOT A SIMPLIFICATION: THE E SOURCE
----------------------------------------------------
With PML, ``step_B`` differences the STORED Ex/Ey/Ez arrays. Without PML and
without a susceptibility, ``fields.stores_E`` is False and
``stepping._read_component`` (stepping.py:2383-2397) DERIVES the source::

    E = D * fields.inverse_epsilon_for(name)

into a pooled scratch buffer, once per component per sub-step — three
full-volume multiplies plus three full-volume writes that the PML path never
pays. ``step_D`` has no analogue: ``get_H`` returns the B array itself, mu = 1
(fields.py:1164-1186).

``DERIVE`` is the constexpr for it. With ``DERIVE = 1`` the kernel forms
``D * inv_eps`` at each load site instead, and the whole snapshot pass
disappears. It is bit-safe because the product is ELEMENTWISE: the value at flat
index j is ``D[j] * inv_eps[j]`` whether it is computed into a scratch array and
read back or computed in a register, the operands are the same float32 bits in
the same order, and ``ENABLE_FP_FUSION = False`` stops the multiply from
contracting with the subtraction that consumes it. The operand ORDER is kept as
the array path writes it (D on the left, stepping.py:2393
``xp.multiply(displacement, inverse, out=target)``); float multiplication is
bitwise commutative, but a transcription is not the place to rely on that.

``DERIVE = 0`` binds whatever the host resolved — the stored E arrays when
``stores_E``, or the array path's own snapshot when a caller wants the
conservative composition. Both are gated.

SEPARATE KERNEL OR A FLAG ON THE EXISTING ONE?
----------------------------------------------
A ``PML: tl.constexpr`` flag on ``kernels.pml_curl_step`` is the right END state
and this module is the interim: the curl half — ghost rule, grouping, ownership
mask — is the part the byte gate is really about, and keeping two copies of it
means two places that must stay bit-identical to ``stepping._curl_from_operands``
forever. The auxiliary and coefficient pointer arguments cost nothing to carry
under a false constexpr (the loads are dead-code-eliminated), so the merged
kernel would take the same eleven-argument prefix and bind placeholders, exactly
as ``constitutive_step`` already binds its sources as inverse-epsilon
placeholders on the H side (launch.py:272-277). ``kernels.py`` is owned by
another live track this round, so the merge is SPECIFIED, not performed; see the
module ``MERGE`` note at the bottom.

BUDGET — 66 STEPS, AND IT IS NOT A PROPERTY OF THIS KERNEL
----------------------------------------------------------
Bit-identity is claimable only FOR A STATED NUMBER OF STEPS (plan section 16).
Measured on the GPU host, RTX A6000, ``2d_plain`` at resolution 40 (640x640), every
state array compared bytewise after every one of 6000 whole ``FdtdDriver.step``
calls: **identical through step 66, FIRST DIFFERENT AT STEP 67** — 28 floats of
2,457,600, in Bx/By/Dz, the kernel writing ``0x004fbb9e`` where the array path
writes ``0x00000000``. ``allclose`` would report True.

That is section 16's ``2d_pml`` result exactly — same step, same array, same bit
pattern — in a run with no absorber, no split-field recurrence and no PML
coefficient anywhere in this file. Section 16 established the divergence is not
dispersion-specific; this establishes it is not absorber-specific either. The
seed is CuPy's flush-to-zero meeting ``dtdx * (a difference)`` at the wavefront.
The spatial spread saturates from step 1500 at a median 49.09% of stored floats,
against 49.65% for ``2d_pml``: the same plateau.

So: 108/108 single-launch, 12/12 x 8 multi-step and 12/12 whole driver steps are
what the gate measured and they pass; 66 is what may be CLAIMED. Nobody should
quote an unqualified "bit-identical" about a run of this kernel, and none of this
is progress toward unblocking dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class _UnavailableKernel:
    """Fail at the launch expression while leaving host predicates importable."""

    __slots__ = ("name", "error")

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def __getitem__(self, grid):
        raise ImportError(
            f"the optional triton package is required to launch {self.name}; "
            f"host coverage remains available without it: {self.error}") from self.error


try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    COVERED_BOUNDARIES,
    CURL_TARGETS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _boundary_kinds,
    _call,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)
from .launch import CupyPointer

# Same constexpr code as ``kernels.METALLIC``. Importing kernels.py here would
# import Triton and defeat this module's host-only coverage route.
METALLIC = tl.constexpr(1)

#: Targets and sources per sub-step. Same table as ``launch.SUB_STEPS`` minus the
#: PML sub-lattice suffix, which does not exist on this path.
SUB_STEPS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "targets": ("Bx", "By", "Bz"),
        "sources": ("Ex", "Ey", "Ez"),
        "displacement": ("Dx", "Dy", "Dz"),   # what DERIVE reads instead
        "backward": 0,
    },
    "step_D": {
        "targets": ("Dx", "Dy", "Dz"),
        # get_H returns the B array itself without PML (fields.py:1184-1186), so
        # the source pointers ARE the B pointers. Not an optimisation here — it is
        # what the array path differences.
        "sources": ("Bx", "By", "Bz"),
        "displacement": None,                 # nothing to derive on the D side
        "backward": 1,
    },
}


#: sha256 of :func:`plain_curl_step`'s source segment — signature through final
#: store, decorator excluded — as the bit-identity gate certified it on the GPU host
#: (RTX A6000, physical device 7, Triton 3.1.0, CuPy 13.5.1) on 2026-08-10.
#:
#: ``fingerprints.json`` cannot carry this yet: its kernel census reads
#: ``kernels.py`` alone (see WIRING note 3), so it would silently swallow this
#: module. Until the record enumerates modules, ``test_triton_no_pml`` welds the
#: kernel to its verdict here instead — the kernel cannot be edited without failing
#: that suite, and this constant cannot be re-cut without re-running the gate on a
#: CUDA host (``parity/meep_gpu/gate_triton_no_pml.py``).
KERNEL_SOURCE_SHA256 = "83ee5373ad9a3d2320bf63acf0c0c6bbbf38d4203aaad2b25ff84ffecccc8cd8"

#: What the gate measured on that source, verbatim from ``results/gate.json``.
#:
#: Two entries need reading carefully rather than skimming.
#:
#: * ``single_launch_unguarded`` is **38/108 identical, not 0/108**, which is a
#:   WEAKER control than the PML curl's 0/120 and is reported as such. 70 of 108
#:   are caught; the 38 that survive are cases where contracting the multiply into
#:   the subtraction happens to land on the same float32. The split by Courant is
#:   the reason the sweep is not a power of two: 12/36 differ at 0.5 against 29/36
#:   at 0.35 and 29/36 at 0.2673. A gate testing only 0.5 would have certified this
#:   kernel on a control that bit less than half as hard.
#: * ``step_budget`` is the number of consecutive whole driver steps over which
#:   bit-identity is CLAIMED. It first differs at ``step_budget + 1``; see BUDGET
#:   in the module docstring.
GATE_VERDICT = {
    "device": "the GPU host RTX A6000 (physical device 7), triton 3.1.0, cupy 13.5.1",
    # The gate compiles a SLICE of this file (``gate_triton_no_pml.kernel_only_source``
    # takes the decorated kernel through its last store), and that slice is byte-equal
    # to the one the recon measured. It did not have to be TAKEN on trust, though: the
    # four fast legs were re-run on the LANDED module and reproduced every number
    # below exactly (``results/triton_no_pml_2026-08-10/landed_rerun/``), which is
    # also what confirmed the clause-3b fix on the device — ``2d_pml``'s refusal now
    # carries one true reason where it used to carry a true one and a false one.
    "gate_slice_sha256":
        "666676cf7d6ac21b4b21b8d0764adc9d354203504d5171fac9619004b471aaab",
    "single_launch_guarded": "108/108 identical, 0 ulp, all targets moved (84 skipped)",
    "single_launch_unguarded": "38/108 identical = 70/108 CAUGHT, worst 419433 ulp",
    "unguarded_by_courant": {"0.5": "12/36 differ", "0.35": "29/36", "0.2673": "29/36"},
    "multistep_guarded": "12/12 configurations, identical at each of 8 steps",
    "engine_route": "2d_plain: 12/12 whole FdtdDriver.step calls, every state array",
    "mutations_caught": {
        "add_instead_of_subtract": "54/54",
        "flatten_grouping": "36/54",            # 18 uncaught are all (64,1,64)
        "drop_inverse_epsilon": "18/54",        # exactly the 18 DERIVE=1 cases
        "derive_centre_only": "18/54",          # likewise
        "drop_ownership_mask": "18/54",         # exactly the metallic-x B-side cases
        "periodic_ghost_on_metallic": "18/54",  # likewise
        "swap_derive_operand_order": "0/54",    # EXPECTED-uncaught control
    },
    "courants": (0.5, 0.35, 0.2673),
    "step_budget": 66,
    "first_divergent_step": 67,
}

# A DEGENERATE AXIS HIDES A GROUPING DEFECT, and ``2d_plain`` is a degenerate-axis
# case. ``flatten_grouping`` — the mutation that reassociates
# ``dtdx * ((c_y - c) + (b - b_z))`` into ``dtdx * (c_y - c + b - b_z)`` — was caught
# 36/36 on the 3-D shape (32,24,16) and 0/18 on (64,1,64), at EVERY Courant and on
# both sub-steps. The cause is structural, not statistical: the mutation edits
# ``curl0``, and on a one-cell y axis ``(c_y - c)`` is an exact zero, so the
# reassociation is exact. EVERY 2-D benchmark case has an invariant axis, so a gate
# that swept only the shapes it was unblocking would certify a regrouped kernel.
# The 3-D shape is in the mutation leg for that reason and for no other, and
# ``test_the_curl_grouping_is_the_one_stepping_uses`` pins the same thing
# structurally on a laptop, where no shape is being swept at all.


@triton.jit
def plain_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    g0, g1, g2,                       # sources: Ex,Ey,Ez (or Dx,Dy,Dz if DERIVE) / Bx,By,Bz
    e0, e1, e2,                       # inverse epsilon; read only when DERIVE
    nx, ny, nz, n_elem, dtdx,
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    DERIVE: tl.constexpr,             # 1 = sources are D and E = D * inv_eps here
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One curl sub-step of all three components with NO absorber: ``f -= curl``.

    Transcribed, and from where (re-checked against the tree):

    * term table          ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (:213-223)
    * ghost rule          ``stepping._shift_up`` (:1723) / ``_shift_down`` (:1787)
    * curl grouping       ``stepping._curl_from_operands`` (:1601)
    * ownership mask      ``stepping._mask_non_owned_cells`` (:1865)
    * the update          ``stepping._apply_curl``'s plain branch (:508)
    * the E derivation    ``stepping._read_component`` (:2383)

    The first three are IDENTICAL to :func:`kernels.pml_curl_step`; only the tail
    differs. ``ENABLE_FP_FUSION`` still applies and still matters: ``f - curl``
    with ``curl = dtdx * (...)`` is a multiply feeding a subtract, which is the
    exact shape LLVM contracts into ``fma.rn.f32``. It matters MORE here than in
    the split-field tail, because there the multiply's consumer was a separate
    coefficient product.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == METALLIC:
        vx = live & (si >= 0) & (si < nx)
    else:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    if BCY == METALLIC:
        vy = live & (sj >= 0) & (sj < ny)
    else:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    # --- the source operands ----------------------------------------------------
    # DERIVE forms stepping._read_component's D * inv_eps here rather than reading
    # a scratch volume the host wrote. D ON THE LEFT — the array path's operand
    # order (stepping.py:2393), kept literally.
    if DERIVE:
        a = tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0) * tl.load(e1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0) * tl.load(e2 + idx, mask=live, other=0.0)
        a_y = tl.load(g0 + oy, mask=vy, other=0.0) * tl.load(e0 + oy, mask=vy, other=0.0)
        a_z = tl.load(g0 + oz, mask=vz, other=0.0) * tl.load(e0 + oz, mask=vz, other=0.0)
        b_x = tl.load(g1 + ox, mask=vx, other=0.0) * tl.load(e1 + ox, mask=vx, other=0.0)
        b_z = tl.load(g1 + oz, mask=vz, other=0.0) * tl.load(e1 + oz, mask=vz, other=0.0)
        c_x = tl.load(g2 + ox, mask=vx, other=0.0) * tl.load(e2 + ox, mask=vx, other=0.0)
        c_y = tl.load(g2 + oy, mask=vy, other=0.0) * tl.load(e2 + oy, mask=vy, other=0.0)
    else:
        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_y = tl.load(g0 + oy, mask=vy, other=0.0)
        a_z = tl.load(g0 + oz, mask=vz, other=0.0)
        b_x = tl.load(g1 + ox, mask=vx, other=0.0)
        b_z = tl.load(g1 + oz, mask=vz, other=0.0)
        c_x = tl.load(g2 + ox, mask=vx, other=0.0)
        c_y = tl.load(g2 + oy, mask=vy, other=0.0)

    # A METALLIC ghost is an EXACT 0.0 past the wall, and under DERIVE the array
    # path shifts the DERIVED volume, so its ghost is 0.0 too — not 0.0*inv_eps,
    # which would be the same value but is worth saying: `other=0.0` on both loads
    # makes the product 0.0*0.0 = +0.0, and the array path's zero is +0.0. Signed
    # zero is the exact class §10.5 records a defect in, so it is pinned by the
    # metallic legs of the gate rather than argued here.

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    curl0 = dtdx * ((c_y - c) + (b - b_z))
    curl1 = dtdx * ((a_z - a) + (c - c_x))
    curl2 = dtdx * ((b_x - b) + (a - a_y))

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0 = tl.where(at_y, 0.0, curl0)
        if BCZ == METALLIC:
            curl0 = tl.where(at_z, 0.0, curl0)
        if BCX == METALLIC:
            curl1 = tl.where(at_x, 0.0, curl1)
        if BCZ == METALLIC:
            curl1 = tl.where(at_z, 0.0, curl1)
        if BCX == METALLIC:
            curl2 = tl.where(at_x, 0.0, curl2)
        if BCY == METALLIC:
            curl2 = tl.where(at_y, 0.0, curl2)
    else:
        if BCX == METALLIC:
            curl0 = tl.where(at_x, 0.0, curl0)
        if BCY == METALLIC:
            curl1 = tl.where(at_y, 0.0, curl1)
        if BCZ == METALLIC:
            curl2 = tl.where(at_z, 0.0, curl2)

    # --- the plain update (stepping._apply_curl, the `target -= curl` branch) ---
    v0 = tl.load(f0 + idx, mask=live, other=0.0) - curl0
    v1 = tl.load(f1 + idx, mask=live, other=0.0) - curl1
    v2 = tl.load(f2 + idx, mask=live, other=0.0) - curl2

    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
#
# The clause numbering below deliberately MIRRORS ``coverage._grid_reasons`` so
# the two can be diffed line by line. Clause 3 is INVERTED (this kernel requires
# the absence of an absorber, where every other kernel here requires one) and
# clauses 13-16 are new: they exist only on this path.
#
# The duplication is the merge's job to remove — see MERGE at the bottom.


def _no_pml_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The shared clauses, with the absorber requirement turned around."""
    reasons: List[str] = []

    # 1. CuPy backend.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3 (INVERTED). NO absorber that absorbs. An active layer routes to the
    #    split-field recurrence, which is ``kernels.pml_curl_step``'s, not this one.
    #    ``stepping._pml_is_active`` is the same test the array path branches on, so
    #    an all-zero-face layer lands HERE, which is what makes the two kernels'
    #    coverage sets a partition rather than an overlap.
    if pml is not None and getattr(pml, "is_active", False):
        reasons.append("an active PML layer is installed (that is the split-field "
                       "kernel's path, not this one)")

    # 3b (NEW, and the one an "absence of a blocker" reading would miss).
    #    ``Fields.enable_pml_storage`` is a ONE-WAY switch that also changes what
    #    ``get_E``/``get_H`` return (fields.py:678-700). A Fields whose storage was
    #    switched on while the layer handed to the stepper is inert reads H from the
    #    STORED array — which ``update_H`` then declines to write — so the curl would
    #    difference a frozen H. The array path has the same hazard; this kernel
    #    refuses the configuration rather than reproducing it.
    #
    #    GATED ON THE LAYER BEING INERT, because the sentence has to be TRUE where it
    #    fires. The recon build tested ``_pml_active`` alone, which appended "while
    #    the layer is inert" to every active-PML refusal too — a right verdict with a
    #    false reason, which is the failure mode a reasons list exists to avoid. The
    #    verdict is unchanged by this guard in every configuration: where the layer IS
    #    active, clause 3 has already refused.
    if (pml is None or not getattr(pml, "is_active", False)) and \
            bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields has PML storage enabled while the layer is inert "
                       "(get_H would serve a stored H that update_H never writes)")

    # 4. Only the two ghost rules the kernel writes.
    kinds = _boundary_kinds(grid, None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {COVERED_BOUNDARIES}")

    # 5. No mirror plane anywhere.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 8. No conductivity on any curl target. Without PML a conductivity routes to
    #    ``_apply_conductive_update`` (stepping.py:1938) — a DIFFERENT three-factor
    #    update, not this one's ``f -= curl``. That is the conductivity kernel's
    #    coverage set; naming it here keeps the two disjoint by construction.
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in CURL_TARGETS:
            if reader(target) is not None:
                reasons.append(f"a conductivity is installed on {target} "
                               f"(mp.Absorber routes to _apply_conductive_update)")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 10. No instantaneous nonlinearity.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST and beta.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def _derive_reasons(fields: Any, grid: Any) -> List[str]:
    """What ``DERIVE = 1`` additionally requires — the E source formed in-kernel.

    Every clause here is about the SHAPE and TYPE of what the array path would
    have multiplied, because the kernel reproduces that multiply per element and
    a broadcast or a promoted dtype changes the answer, not the speed.
    """
    reasons: List[str] = []
    shape = tuple(int(n) for n in getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not a 3-tuple")
        return reasons
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        reasons.append("Fields does not expose inverse_epsilon_for()")
        return reasons
    for name in ELECTRIC_COMPONENTS:
        try:
            inverse = reader(name)
        except Exception as exc:  # noqa: BLE001 - unreadable is not covered
            reasons.append(f"inverse_epsilon_for({name!r}) raised {type(exc).__name__}")
            continue
        # A SCALAR or a broadcast row is not steppable by a kernel that indexes it
        # with the cell's own flat index. The array path would broadcast it; this
        # would read out of bounds or read the wrong cell. ``_volume_reasons`` is
        # coverage.py's own check — shape, dtype AND C-contiguity — reused rather
        # than re-stated, because the recon build re-stated it and dropped the
        # contiguity clause on the way: a reversed view has the right shape and the
        # right dtype and is read in the wrong order by a flat-indexed kernel, which
        # is a silent wrong answer of exactly the class this predicate exists to
        # refuse. (The PML curl reaches the same helper through ``_grid_reasons``.)
        reasons.extend(_volume_reasons(f"inverse_epsilon_for({name!r})", inverse, shape))
        displacement = getattr(fields, "D" + name[1], None)
        if displacement is None:
            reasons.append(f"D{name[1]} is not allocated (DERIVE differences D * inv_eps)")
        else:
            reasons.extend(_volume_reasons("D" + name[1], displacement, shape))
    # 15. Off-diagonal chi1inv. The PML curl ADMITS it because its whole effect is
    #     inside update_E; DERIVE moves part of update_E's product INTO this kernel,
    #     so the row-coupling term — which reads the other components — would be
    #     silently dropped. It is refused for DERIVE and only for DERIVE.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("off-diagonal chi1inv is installed (the derived E source is a "
                       "row product that reads the other components)")
    return reasons


def plain_curl_coverage(fields: Any, pml: Any, sub_step: str = "step_B") -> Coverage:
    """May the no-absorber curl kernel step this ``(fields, pml)`` for this sub-step?

    Written positively and per sub-step, the same two rules the rest of this
    package follows. The sub-step matters here in a way it did not for the PML
    curl: ``step_B``'s source is E, whose storage mode decides ``DERIVE``, while
    ``step_D``'s source is the B array itself and has no mode at all.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields has no grid",))
    reasons = _no_pml_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))

    # 13. Every array the sub-step touches must exist and be a C-contiguous float32
    #     volume of the grid's shape, and the grid must be inside the kernel's int32
    #     index range. Stated rather than assumed: without PML the auxiliaries are
    #     None by design, so "getattr returned something" is a weaker check here than
    #     it is on the PML path — and the layout half goes through coverage.py's own
    #     ``_layout_reasons`` so the two paths cannot drift apart.
    shape = tuple(int(n) for n in getattr(grid, "shape", ()))
    spec = SUB_STEPS[sub_step]
    for name in spec["targets"]:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, spec["targets"]))

    # 14. The E storage mode, named. ``stores_E`` is what ``_read_component``
    #     branches on, so it is what DERIVE must equal — and getting it backwards is
    #     not a crash: with stores_E True and DERIVE 1 the kernel would difference
    #     D*inv_eps while the engine's monitors read a stored E that a polarization
    #     has already been subtracted from, a smooth field one polarization out of
    #     date. With stores_E False and DERIVE 0 the host would bind Ex, which is
    #     None, and the launcher's failure would be a TypeError far from its cause.
    if sub_step == "step_B":
        stores_E = bool(getattr(fields, "stores_E", False))
        if stores_E:
            for name in ELECTRIC_COMPONENTS:
                if getattr(fields, name, None) is None:
                    reasons.append(f"stores_E is True but {name} is not allocated")
            reasons.extend(_layout_reasons(fields, shape, ELECTRIC_COMPONENTS))
        else:
            reasons.extend(_derive_reasons(fields, grid))
            # 16. ``_read_component``'s derived branch is taken only when a scratch
            #     pool exists; with ``fields.scratch is None`` it calls ``get_E``,
            #     which allocates and computes the SAME product. Both are covered —
            #     but the fact that they are the same product is a measurement, not
            #     an assumption, so the gate runs both and this clause records that
            #     neither is refused.
            pass
    else:
        for name in spec["sources"]:
            if getattr(fields, name, None) is None:
                reasons.append(f"{name} is not allocated (step_D differences B directly)")
        reasons.extend(_layout_reasons(fields, shape, spec["sources"]))
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class PlainCurlPlan:
    """A launchable, allocation-free no-absorber curl sub-step.

    Same two construction routes and one launch route as ``launch.PmlCurlPlan``:
    :func:`plan_plain_curl` from the engine's objects through the predicate, and
    :func:`plan_plain_curl_from_arrays` from bare device arrays for the gate.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "derive", "bc",
                 "block", "_targets", "_sources", "_inv_eps", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, block: int,
                 targets, sources, inverse_epsilon=None, derive: int = 0,
                 kernel=None) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply; Triton types a
        # Python float argument as fp32. Same bits. (launch.PmlCurlPlan, verbatim.)
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.derive = int(derive)
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._sources = tuple(CupyPointer(a) for a in sources)
        # Placeholders when DERIVE is 0: the loads sit behind a constexpr branch and
        # are compiled away, but a pointer argument still has to type. Same device
        # as ConstitutivePlan's H side (launch.py:272-277).
        self._inv_eps = (tuple(CupyPointer(a) for a in inverse_epsilon)
                         if inverse_epsilon is not None else self._sources)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else plain_curl_step
        kernel[self._grid](
            *self._targets, *self._sources, *self._inv_eps,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            DERIVE=self.derive,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"PlainCurlPlan({self.sub_step}, shape={self.shape}, bc={self.bc}, "
                f"derive={self.derive}, block={self.block})")


def plan_plain_curl(fields: Any, pml: Any, sub_step: str,
                    block: Optional[int] = None) -> Optional[PlainCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage."""
    if not plain_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    # ``pml=None``, and that is faithful rather than convenient: the absorber does
    # not DECIDE the ghost rule (stepping.py:2099's own docstring — "the rule is the
    # symmetry and nothing else"); the argument exists so the layer can be checked
    # for AGREEMENT with a fold. This path refuses every fold at clause 5 and every
    # active layer at clause 3, so there is nothing left for that check to say.
    # ``launch.plan_pml_curl`` passes the real layer for the same reason in reverse.
    kinds = resolve(grid, None)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    derive = int(sub_step == "step_B" and not bool(getattr(fields, "stores_E", False)))
    if derive:
        sources = [getattr(fields, n) for n in spec["displacement"]]
        inverse = [fields.inverse_epsilon_for(n) for n in ELECTRIC_COMPONENTS]
    else:
        sources = [getattr(fields, n) for n in spec["sources"]]
        inverse = None
    return PlainCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        sources, inverse, derive)


def plan_plain_curl_from_arrays(sub_step: str, arrays: Dict[str, Any], codes,
                                dtdx: float, derive: int = 0,
                                block: Optional[int] = None,
                                kernel: Any = None) -> PlainCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``kernel=`` is plumbed
    through because a mutation leg that silently launches the SHIPPED kernel
    reports every defect as uncaught — the exact harness defect §13.4 records.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    if derive:
        sources = [arrays[n] for n in spec["displacement"]]
        inverse = [arrays["inv_eps_" + n] for n in ELECTRIC_COMPONENTS]
    else:
        sources = [arrays[n] for n in spec["sources"]]
        inverse = None
    return PlainCurlPlan(sub_step, shape, dtdx, codes,
                         DEFAULT_BLOCK if block is None else block,
                         [arrays[n] for n in spec["targets"]],
                         sources, inverse, derive, kernel=kernel)


class PlainStepPlan:
    """The composition: which no-absorber sub-steps this run may put on Triton.

    On a plain run there are only two heavy sub-steps to compose —
    ``stepping.update_H`` returns immediately and ``stepping.update_E`` returns
    immediately unless ``stores_E`` — so ``covered`` being ``{step_B, step_D}`` is
    FULL coverage of the step's arithmetic, not a subset of it. That is the one
    respect in which this path is better placed than the PML one, and it is worth
    stating because "2 of 4" reads like less than it is.
    """

    __slots__ = ("plans", "refusals")

    def __init__(self, plans: Dict[str, PlainCurlPlan],
                 refusals: Dict[str, Tuple[str, ...]]) -> None:
        self.plans = plans
        self.refusals = refusals

    def run(self) -> None:  # pragma: no cover - the gate drives the sub-steps itself
        raise NotImplementedError(
            "the whole no-PML step is not a single launch: the driver's boundary "
            "passes (sources, fill_symmetry_bc_*, zero_metal_*) run between step_B "
            "and step_D on the array path, so a caller composes the two plans in "
            "that order rather than asking this object to.")


def plan_plain_step(fields: Any, pml: Any,
                    block: Optional[int] = None) -> PlainStepPlan:
    """Ask each sub-step's predicate independently and report the covered subset."""
    plans: Dict[str, PlainCurlPlan] = {}
    refusals: Dict[str, Tuple[str, ...]] = {}
    for sub_step in SUB_STEPS:
        verdict = plain_curl_coverage(fields, pml, sub_step)
        if verdict.covered:
            plan = plan_plain_curl(fields, pml, sub_step, block=block)
            if plan is not None:
                plans[sub_step] = plan
                continue
            refusals[sub_step] = ("coverage passed but the plan builder refused",)
        else:
            refusals[sub_step] = verdict.reasons
    return PlainStepPlan(plans, refusals)


# ---------------------------------------------------------------------------
# WIRING — experimental composition is live; production dispatch is deferred
# ---------------------------------------------------------------------------
#
# ``__init__.py`` exports the predicate and one-curl builder lazily, and
# ``launch.plan_step`` evaluates the PML and no-PML predicates independently for
# each curl. A plain run composes ``{step_B, step_D}``, which is FULL coverage of
# its step arithmetic because ``update_H`` and ``update_E`` are array-path no-ops.
# The CUDA composition gate and ``fingerprints.json`` bind this exact module and
# the central host seam to that result.
#
# This remains an experimental plan consumed by gates and benchmarks. Enabling
# ``fastpath.plan_fast_path`` requires the package-wide numerical release policy,
# including the documented long-run CuPy/Triton subnormal divergence; this module
# must not turn itself on independently.
#
# Nothing else changes: ``run()`` uses the package's exact ``GUARD_SPELLING``, the
# module builds no ``options={...}`` dict, and ``launch.py`` keeps its four launch
# sites.

# ---------------------------------------------------------------------------
# MERGE — what this module owes ``kernels.py`` once that file is free again
# ---------------------------------------------------------------------------
#
# 1. ``plain_curl_step`` folds into ``kernels.pml_curl_step`` as a
#    ``PML: tl.constexpr``. Lines 1-190 of the two bodies are already identical;
#    the merged tail is::
#
#        if PML:
#            <the split-field recurrence, unchanged>
#        else:
#            v0 = tl.load(f0 + idx, mask=live, other=0.0) - curl0
#            ...
#
#    The auxiliary and coefficient pointers stay in the signature and are bound to
#    placeholders when PML is 0, the way ``ConstitutivePlan`` already binds its
#    sources as inverse-epsilon placeholders on the H side.
#
# 2. ``DERIVE`` folds in with it and is orthogonal to ``PML`` — but only one of the
#    four combinations is reachable: ``PML=1`` forces stored E, so ``DERIVE=1``
#    exists only under ``PML=0``. The merged predicate must REFUSE ``PML=1,
#    DERIVE=1`` by name rather than leave it unreachable-by-construction.
#
# 3. ``coverage._grid_reasons`` splits into ``_shared_grid_reasons`` (clauses 1, 2,
#    4-12, with the ``pml`` argument used only to resolve the ghost rule) plus two
#    one-clause callers: ``_requires_active_pml`` and ``_requires_no_pml``.
#    ``_no_pml_grid_reasons`` above is then deleted, not aliased.
#
# 4. ``fingerprints.json`` gains the merged kernel's hash. Note that the merge
#    CHANGES ``kernels.py``'s bytes, so every fingerprint in that record is
#    re-cut, which by §13.5's own rule means re-running the whole byte gate on a
#    CUDA host. Merging is therefore a gate event, not a refactor.

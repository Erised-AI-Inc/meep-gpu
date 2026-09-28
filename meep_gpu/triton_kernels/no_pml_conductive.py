"""The CONDUCTIVE no-absorber curl: ``step_B``/``step_D`` with a sigma and no PML.

Closes residual group (E) and the CURL half of residual group (H) — 6 of the 9
slots the no-absorber trio is about. It is a genuine kernel: the closure round
measured the incumbent plain curl DIVERGENT on this configuration at both
sub-steps (1536/4608 and 1356/4608,
``results/residual_closure_2026-08-15/device/bodies/bodies.json``), and this file
reproduces those verdicts locally before claiming to close them
(see :data:`LOCAL_DIVERGENCE`).

WHAT THE ARRAY PATH DOES, AND WHERE THE FOURTH BRANCH LIVES
-----------------------------------------------------------
``stepping._apply_curl`` (stepping.py:489-539) has FOUR branches and this module
owns the third::

    target  = getattr(fields, term.target)                    # stepping.py:507
    condfac = fields.condfac_for(term.target)                 # stepping.py:508
    if pml_active:
        ...                                                   # :480-506  (two branches)
    elif condfac is not None:
        _apply_conductive_update(target, curl, condfac,
                                 fields.condinv_for(term.target))   # :507-508
    else:
        target -= curl                                        # :509-510

and ``_apply_conductive_update`` (stepping.py:1985-1998) is three in-place passes,
in this order and no other::

    field *= condfac        # stepping.py:1996
    field -= curl           # stepping.py:1997
    field *= condinv        # stepping.py:1998

i.e. ``f <- ((f * condfac) - curl) * condinv``, MEEP step_generic.cpp:87-97, with
``condfac = 1 - sigma*dt/2`` and ``condinv = 1/(1 + sigma*dt/2)`` (fields.py:821-822,
MEEP structure.cpp:693-706), both FULL VOLUMES of ``grid.shape``. There is no
``fu``, no ``f_cond`` and no coefficient table on this path: without an absorber
``Fields._ensure_conductive_pml_storage`` returns at its first line
(fields.py:732-733) and ``f_cond_*`` is never allocated. That is precisely why
``conductivity.conductive_pml_curl_coverage`` refuses these rows today, and the
refusal it prints — "f_cond_Bx is not allocated, but Bx is conductive" — is TRUE
and is not a bug in that predicate.

THE CURL HALF IS UNCHANGED, AND THAT IS THE WHOLE REASON THIS IS CHEAP.
Conductivity enters ``_apply_curl`` only AFTER the curl is formed, so the term
table, the ghost rule, the stencil grouping, the ownership mask and the derived-E
source are ``no_pml.plain_curl_step``'s, byte for byte. Only the tail differs, and
it is one expression.

CONDUCTIVITY IS READ PER TERM. NOT PER SIDE, AND NOT PER RUN.
--------------------------------------------------------------
``_apply_curl`` reads ``fields.condfac_for(term.target)`` at stepping.py:508, once
per TERM, inside a loop that ``step_B`` runs over ``B_CURL_TERMS`` (stepping.py:214,
targets Bx/By/Bz) and ``step_D`` over ``D_CURL_TERMS`` (stepping.py:219, targets
Dx/Dy/Dz). **A D sigma therefore cannot reach step_B**, and a component whose sigma
is absent takes ``target -= curl`` while its neighbours are lossy — MEEP's own
allocation granularity (``s->conductivity[c][d]``).

:data:`COND0`-style constexprs carry that per component, from
:func:`conductive_no_pml_targets`, which is the SINGLE place the question is asked
so the clause and the compile-time choice cannot disagree
(``conductivity.conductive_targets`` exists for the same reason and this is its
no-PML sibling). Where a component is lossless the conductive branch compiles out
entirely and the kernel emits ``no_pml.plain_curl_step``'s ``f - curl`` verbatim,
so a partially conductive launch is identical on its lossless components BY
CONSTRUCTION rather than by argument.

**THE ADMISSION GATE IS A DIFFERENT QUESTION FROM THE ARITHMETIC, and conflating
the two is the error this paragraph exists to prevent.** The arithmetic reads per
TERM, above. The ADMISSION reads the same question ``no_pml.plain_curl_coverage``
clause 8 reads — is a conductivity installed on ANY curl target, or is
``has_magnetic_conductivity`` set — because that clause is what this predicate
INVERTS, and inverting it exactly is what makes the two coverage sets a partition
instead of leaving a hole. Asking the narrower question in the admission would
leave ``step_B`` on a D-only-conductive run refused by BOTH predicates: the
incumbent refuses it (its clause 8 is per RUN) and a narrow inversion would too.
Measured: that hole is 0 corpus slots today (all three conductive no-PML rows carry
both sides, ``predicate_coverage_2026-08-16_wired_convention``), and it is still not
opened, because a partition that depends on which rows exist is not a partition.

WHAT THIS CLOSES, AND THE ROWS BY NAME
--------------------------------------
From ``results/predicate_coverage_2026-08-16_wired_convention/remaining.txt``, the
186-row census after the residual-group wiring:

* ``TestAbsorber.test_absorber_2d`` — [200,200,1], metallic/metallic/periodic,
  conductivity on BOTH sides, ``stores_E`` False, no poles. Group (E) exactly. Its
  ``step_B`` derives E (``DERIVE = 1``); its ``update_H``/``update_E`` are already
  ``no_pml_constitutive``'s null. **2 slots, and the row goes whole-step.**
* ``TestAbsorber.test_absorber`` and ``absorber-1d.py`` — [1,1,400],
  periodic/periodic/metallic, conductivity on BOTH sides, ``stores_E`` True, 5
  registered poles. Group (H)'s curls. ``stores_E`` makes ``DERIVE = 0``; the poles
  never reach the curl. **4 slots**, and with ``no_pml_stored_e``'s Arm S at
  ``update_E`` and the already-wired ``no_pml_ade`` at ``update_P`` those two rows
  go whole-step as well.

GROUP (H)'s CURLS WERE NEVER MEASURED DIRECTLY BY THE CLOSURE ROUND — the
assumption was that they diverge for group (E)'s reason. That assumption is now a
MEASUREMENT and it is local, so it needs no device: see :data:`LOCAL_DIVERGENCE`
and ``test_triton_no_pml_conductive``'s divergence leg. Dispersion cannot reach a
curl at all — ``stepping.step_B`` (stepping.py:261-376) and ``step_D`` (:379-457)
name ``fields.polarizations`` nowhere, and the only route from a susceptibility to
the curl is ``stores_E`` flipping ``_read_component`` (stepping.py:2438-2451) from
the derived branch to the stored one, which is a boolean about STORAGE and not
about poles. So group (H)'s curl and group (E)'s curl are ONE product at two
settings of ``DERIVE``, which is why one kernel closes both.

DEVICE RESULT
-------------
``parity/meep_gpu/gate_triton_no_pml_conductive.py`` ran on the GPU host's RTX A6000
on 2026-08-17 under the installed ``keep`` policy. All eight product rows were
byte-identical over eight launches, including D-only and B-only runs at BOTH
curls. The three live mutations diverged by 1,536, 1,308 and 1,344 words. The
run is transcribed into ``fingerprints.json`` as
``triton_no_pml_conductive_device_gate`` and the arm is eligible for opted-in
dispatch. This is a correctness result, not a throughput claim.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

# ONE DEFINITION OF THE SUB-STEP TABLE, IMPORTED. The targets, the sources, the
# displacement fallback and the difference direction are all identical to the
# lossless no-PML curl's — conductivity changes the tail and nothing above it — so
# re-spelling them here would be a second transcription of a table whose whole
# purpose is that there is only one.
from .no_pml import SUB_STEPS

__all__ = [
    "LOCAL_DIVERGENCE",
    "ConductivePlainCurlPlan",
    "conductive_no_pml_targets",
    "conductive_plain_curl_coverage",
    "conductive_plain_curl_step",
    "explain_conductive_no_pml",
    "plan_conductive_plain_curl",
    "plan_conductive_plain_curl_from_arrays",
    "plan_conductive_plain_step",
]


#: THE GAP THIS MODULE CLOSES, MEASURED WHERE IT COULD BE — on NumPy, on this
#: laptop, against the real ``stepping`` sub-steps.
#:
#: The question a divergence measurement has to answer is "what would the INCUMBENT
#: have left in the target", and the honest way to ask it is to run the real array
#: path with the one reader that selects the tail cleared: ``_apply_curl`` consults
#: ``fields.condfac_for(term.target)`` at stepping.py:508 and takes ``target -= curl``
#: at :539 when it answers None, so a ``condfac_for`` that answers None turns
#: ``stepping.step_B`` into exactly the product ``no_pml.plain_curl_step`` writes —
#: with the real curl, the real ghost rule, the real ownership mask and the real E
#: source, none of which this module changes.
#:
#: ``differing`` is uint32-compared over the three targets of the sub-step, so a
#: signed zero counts as a difference and a NaN never matches itself.
#: ``moved`` is what the array path itself changed from the seed, on the same words:
#: it is the VACUITY FLOOR, and a case whose ``moved`` is 0 measures nothing.
#:
#: The two CONTROLS are the load-bearing half. With no conductivity anywhere the
#: two tails are the same branch of the same function, so a DIVERGENT verdict there
#: would mean the harness is measuring something other than the conductive tail.
LOCAL_DIVERGENCE: Dict[str, Dict[str, Any]] = {
    "_measured": ("NumPy float32, arm64 laptop, real Grid/Fields/PML objects, one "
                  "sub-step per case, uint32 compare. Reproduced by "
                  "test_triton_no_pml_conductive rather than quoted from here."),
    # Group (E): the closure round's own shape. 8x8x8 x 3 targets = 1536 words, and
    # its 4608 denominator is the whole 9-volume stored inventory.
    "E_cond_no_poles_3d_step_B": {"verdict": "DIVERGENT", "differing": 1536,
                                  "compared": 1536, "moved": 1536},
    "E_cond_no_poles_3d_step_D": {"verdict": "DIVERGENT", "differing": 1536,
                                  "compared": 1536, "moved": 1536},
    # Group (H): the same shape with poles registered and stored E. THE ASSUMPTION
    # THE CLOSURE ROUND LEFT OPEN, now measured.
    "H_cond_2_poles_3d_step_B": {"verdict": "DIVERGENT", "differing": 1536,
                                 "compared": 1536, "moved": 1536},
    "H_cond_2_poles_3d_step_D": {"verdict": "DIVERGENT", "differing": 1536,
                                 "compared": 1536, "moved": 1536},
    # The corpus rows' own 1-D shape, [1,1,400] x 3 targets = 1200 words.
    "H_cond_5_poles_1d_step_B": {"verdict": "DIVERGENT", "differing": 1200,
                                 "compared": 1200, "moved": 1200},
    "H_cond_5_poles_1d_step_D": {"verdict": "DIVERGENT", "differing": 1200,
                                 "compared": 1200, "moved": 1200},
    "CONTROL_no_cond_2_poles_1d_step_B": {"verdict": "IDENTICAL", "differing": 0,
                                          "compared": 1200, "moved": 800},
    "CONTROL_no_cond_2_poles_1d_step_D": {"verdict": "IDENTICAL", "differing": 0,
                                          "compared": 1200, "moved": 800},
}


def conductive_no_pml_targets(fields: Any, sub_step: str) -> Tuple[bool, bool, bool]:
    """Which of THIS sub-step's three targets carry a conductivity.

    THE SINGLE PLACE THAT DECIDES IT, for the reason
    :func:`conductivity.conductive_targets` gives on the PML path: the predicate's
    per-target clauses and the ``COND0``/``COND1``/``COND2`` constexprs both read
    this function, so a lossless component cannot be stepped through the conductive
    branch — which would dereference a placeholder pointer, multiply by whatever it
    found and produce a smooth, plausible, wrong field with no exception anywhere.

    PER SUB-STEP, and that is the point. ``step_B`` loops ``B_CURL_TERMS``
    (stepping.py:214) and ``step_D`` loops ``D_CURL_TERMS`` (stepping.py:219), and
    ``_apply_curl`` reads ``condfac_for(term.target)`` inside that loop
    (stepping.py:508). A D sigma cannot reach ``step_B``; asking "is this run
    conductive" instead of "is THIS target conductive" is an over-refusal on one
    side and an over-admission on the other, and both are silent.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return (False, False, False)
    flags: List[bool] = []
    for target in SUB_STEPS[sub_step]["targets"]:
        try:
            flags.append(reader(target) is not None)
        except Exception:  # noqa: BLE001 - an unanswerable component is not conductive
            flags.append(False)
    return (flags[0], flags[1], flags[2])


def _any_curl_conductivity(fields: Any) -> bool:
    """``no_pml.plain_curl_coverage`` clause 8's question, asked the same way.

    Clause 8 refuses on ``condfac_for(target) is not None`` for every name in
    ``coverage.CURL_TARGETS`` — all SIX, both sides — and again on
    ``fields.has_magnetic_conductivity``. This function is the same disjunction, so
    that clause fires exactly where this one is True and the two predicates
    partition the conductive/lossless axis with no gap and no overlap.

    An unreadable reader answers False, i.e. "not this family's": the incumbent
    reads the same attribute with the same permissiveness (a ``condfac_for`` that is
    not callable makes its loop a no-op), so answering True here would put both
    predicates on the same configuration.
    """
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in CURL_TARGETS:
            try:
                if reader(target) is not None:
                    return True
            except Exception:  # noqa: BLE001 - matches the incumbent's own permissiveness
                continue
    return bool(getattr(fields, "has_magnetic_conductivity", False))


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Deferred behind a flag so the module imports with Triton absent: the coverage
# predicate below is imported by tests that run on the merge-bar laptop, where
# Triton is not installable at all.
#
# The kernel is defined at MODULE scope and not inside a builder. That is measured
# rather than stylistic — `@triton.jit` resolves a body's names, including the
# `tl.constexpr` annotations, through the defining module's `__globals__`, so a
# kernel defined inside a function compiles to `NameError('tl is not defined')` at
# first launch (dispersive_update_e.py:138-143 records the run that showed it, 320
# of 320 cases).

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the byte gate

    #: Boundary codes, identical to ``kernels``' and ``no_pml``'s own. A plan built
    #: here and a plan built there index the same table; a test pins them equal.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def conductive_plain_curl_step(
        f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
        g0, g1, g2,                       # sources: Ex,Ey,Ez (or Dx,Dy,Dz if DERIVE) / Bx,By,Bz
        e0, e1, e2,                       # inverse epsilon; read only when DERIVE
        cf0, cf1, cf2,                    # condfac  (a placeholder where COND == 0)
        ci0, ci1, ci2,                    # condinv  (a placeholder where COND == 0)
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
        DERIVE: tl.constexpr,             # 1 = sources are D and E = D * inv_eps here
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One curl sub-step with a material conductivity and NO absorber.

        Transcribed, and from where (re-checked against the tree, 2026-08-16):

        * term table          ``stepping.B_CURL_TERMS`` (:213) / ``D_CURL_TERMS`` (:218)
        * ghost rule          ``stepping._shift_up`` (:1723) / ``_shift_down`` (:1787)
        * curl grouping       ``stepping._curl_from_operands`` (:1601)
        * ownership mask      ``stepping._mask_non_owned_cells`` (:1865)
        * the E derivation    ``stepping._read_component`` (:2383-2396)
        * the LOSSLESS tail   ``stepping._apply_curl``'s plain branch (:509-510)
        * the LOSSY tail      ``stepping._apply_conductive_update`` (:1938-1951)

        Everything above the tail is byte-copied from :func:`no_pml.plain_curl_step`,
        because conductivity enters ``_apply_curl`` only after the curl is formed
        (stepping.py:508 reads the coefficient; :536 chooses the branch). That
        duplication is forced — the merge that would remove it is a ``COND``
        constexpr on the plain kernel, which is a ``no_pml.py`` edit gated on that
        module's own byte gate — and ``test_triton_no_pml_conductive`` diffs the two
        curl bodies textually so they cannot drift.

        ``ENABLE_FP_FUSION`` applies and matters MORE than on the lossless path.
        ``((f * cf) - curl) * ci`` is a multiply feeding a subtract feeding a
        multiply, with ``curl`` itself a multiply — three separate contraction sites
        where the plain tail has one. DO NOT FLATTEN THE PARENTHESES: the array path
        performs three SEQUENTIAL in-place passes (stepping.py:1996, :1997, :1998),
        each rounding to float32 before the next reads it.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
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

        # --- the source operands ------------------------------------------------
        # DERIVE forms stepping._read_component's D * inv_eps here rather than
        # reading a scratch volume the host wrote. D ON THE LEFT — the array path's
        # operand order (stepping.py:2450), kept literally.
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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens -
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask (stepping._mask_non_owned_cells) --------------------
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

        # --- the tail, PER COMPONENT (stepping.py:536-539) ----------------------
        # COND == 0 emits no condfac load, no condinv load and no multiply: the
        # component takes ``no_pml.plain_curl_step``'s own line, which is what makes
        # a mixed launch identical on its lossless components by construction.
        v0 = tl.load(f0 + idx, mask=live, other=0.0)
        if COND0:
            v0 = ((v0 * tl.load(cf0 + idx, mask=live, other=0.0)) - curl0) \
                * tl.load(ci0 + idx, mask=live, other=0.0)
        else:
            v0 = v0 - curl0

        v1 = tl.load(f1 + idx, mask=live, other=0.0)
        if COND1:
            v1 = ((v1 * tl.load(cf1 + idx, mask=live, other=0.0)) - curl1) \
                * tl.load(ci1 + idx, mask=live, other=0.0)
        else:
            v1 = v1 - curl1

        v2 = tl.load(f2 + idx, mask=live, other=0.0)
        if COND2:
            v2 = ((v2 * tl.load(cf2 + idx, mask=live, other=0.0)) - curl2) \
                * tl.load(ci2 + idx, mask=live, other=0.0)
        else:
            v2 = v2 - curl2

        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

else:  # pragma: no cover - the laptop path
    conductive_plain_curl_step = None  # type: ignore[assignment]


def conductive_plain_curl_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError.

    The accessor exists so a caller that needs the kernel gets an explanation rather
    than a ``None`` that fails later as a ``TypeError`` far from its cause.
    """
    if conductive_plain_curl_step is None:
        raise ImportError(
            "the conductive no-PML curl kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast path "
            f"is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return conductive_plain_curl_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
#
# The clauses below MIRROR ``no_pml._no_pml_grid_reasons`` line for line, because
# this predicate's coverage set is that one's complement on exactly one axis. Two
# clauses are INVERTED and everything else is restated verbatim:
#
#   clause 3   an active PML layer  -> refused, as in no_pml (NOT inverted; this
#              family is the no-absorber one too)
#   clause 8   a conductivity       -> REQUIRED here, refused there
#
# RESTATED, not imported-and-subtracted, for the reason
# ``conductivity._shared_grid_reasons`` gives: subtracting a reason string from
# another predicate's output would make this file's coverage a function of that
# file's phrasing, and a clause renamed there would silently widen coverage here.


def _no_pml_conductive_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """``no_pml``'s clause set with clause 8 turned around."""
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. The tail is the same shape in complex64, the STORAGE is not,
    #    and a complex run stepped as float32 reads the wrong stride.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. NO absorber that absorbs — the same test ``no_pml`` makes, NOT inverted.
    #    An active layer routes to ``_apply_conductive_pml_update`` (stepping.py:2001),
    #    the four-case recurrence with fu and f_cond, which is
    #    ``conductivity.conductive_pml_curl_step``'s and not this one's.
    #    ``stepping._pml_is_active`` is the same test the array path branches on, so
    #    an all-zero-face layer lands HERE.
    if pml is not None and getattr(pml, "is_active", False):
        reasons.append("an active PML layer is installed (that is the conductive "
                       "split-field kernel's path, not this one)")

    # 3b. ``Fields.enable_pml_storage`` is a ONE-WAY switch that also changes what
    #     get_E/get_H return (fields.py:678-700). A Fields whose storage was switched
    #     on while the layer handed to the stepper is inert reads H from the STORED
    #     array — which ``update_H`` then declines to write — so the curl would
    #     difference a frozen H. Gated on the layer being inert so the sentence is
    #     TRUE where it fires (``no_pml.py`` clause 3b was rewritten for exactly that).
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

    # 5. No mirror plane anywhere: a fold changes the ghost rule, adds a parity mask
    #    at cell 0 and changes the stored extent.
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

    # 8 (INVERTED). A conductivity SOMEWHERE on the curl targets is REQUIRED, and
    #    the question is asked exactly as ``no_pml`` clause 8 asks it — over all six
    #    targets plus ``has_magnetic_conductivity`` — so the two coverage sets are a
    #    partition rather than an approximation of one. The per-TARGET question,
    #    which is what the arithmetic reads, is :func:`conductive_no_pml_targets`
    #    and is asked in the caller.
    if not _any_curl_conductivity(fields):
        reasons.append(
            "no curl target carries a conductivity: without one the array path takes "
            "stepping._apply_curl's plain branch (:509-510), which is "
            "no_pml.plain_curl_step's, and two kernels must not claim one slot")

    # 10. No instantaneous nonlinearity. The Pade factor is update_E's, but the
    #     clause is written per RUN in every sibling predicate and is kept that way
    #     here so the sets stay comparable line by line.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST adds a second additive term to every curl; beta adds out-of-plane
    #        couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def _derive_reasons(fields: Any, grid: Any) -> List[str]:
    """What ``DERIVE = 1`` additionally requires — the E source formed in-kernel.

    ``no_pml._derive_reasons``'s clause set, restated for the same reason the grid
    clauses are. Every clause is about the SHAPE and TYPE of what the array path
    would have multiplied, because the kernel reproduces that multiply per element
    and a broadcast or a promoted dtype changes the answer, not the speed.
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
        reasons.extend(_volume_reasons(f"inverse_epsilon_for({name!r})", inverse, shape))
        displacement = getattr(fields, "D" + name[1], None)
        if displacement is None:
            reasons.append(f"D{name[1]} is not allocated (DERIVE differences D * inv_eps)")
        else:
            reasons.extend(_volume_reasons("D" + name[1], displacement, shape))
    # An off-diagonal row makes the derived E source a row product that reads the
    # OTHER components. Refused for DERIVE and only for DERIVE, exactly as on the
    # lossless path.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("off-diagonal chi1inv is installed (the derived E source is a "
                       "row product that reads the other components)")
    return reasons


def conductive_plain_curl_coverage(fields: Any, pml: Any,
                                   sub_step: str = "step_B") -> Coverage:
    """May the conductive no-absorber curl kernel step this sub-step of this run?

    Written POSITIVELY and PER SUB-STEP. In one sentence, what is admitted is a
    real-field Cartesian CuPy run with NO active split-field layer, at k = 0 on
    every axis, with periodic or metallic ghost rules, no fold, no cylindrical axis,
    no BFAST, no special_kz, no instantaneous nonlinearity, whose susceptibilities
    (if any) are electric Lorentz/Drude poles, **which carries a material
    conductivity on at least one curl target**, and whose ``condfac``/``condinv``
    volumes for the conductive targets of THIS sub-step are present, float32,
    C-contiguous and grid-shaped.

    DISPERSION IS ADMITTED, and that is this predicate's group-(H) half. The curl
    differences the STORED E and H arrays and never reads a polarization —
    ``stepping.step_B`` and ``step_D`` name ``fields.polarizations`` nowhere. What a
    pole does change is ``stores_E``, and that is read below as the ``DERIVE``
    clause, which is a question about STORAGE. Measured rather than argued: with
    poles registered and a conductivity installed the incumbent plain tail diverges
    by the same count as with no poles at all (:data:`LOCAL_DIVERGENCE`).

    ``f_cond_*`` IS NOT A CLAUSE, and its absence is the reason this module exists.
    ``Fields._ensure_conductive_pml_storage`` returns at fields.py:732-733 without an
    active layer, so no no-PML run allocates one; ``_apply_conductive_update``
    (stepping.py:1985-1998) keeps no history and needs none.
    ``conductivity.conductive_pml_curl_coverage`` requires it and is right to —
    that is what holds the two conductive predicates apart on the absorber axis.

    NEITHER IS A COEFFICIENT TABLE. There is no ``kms``/``sinv`` on this path;
    requiring one would be a clause about an object the sub-step never reads.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields has no grid",))

    spec = SUB_STEPS[sub_step]
    targets = spec["targets"]
    reasons = _no_pml_conductive_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))

    # 13. Every array the sub-step touches must exist and be a C-contiguous float32
    #     volume of the grid's shape, and the grid must be inside the kernel's int32
    #     index range.
    shape = tuple(int(n) for n in getattr(grid, "shape", ()))
    for name in targets:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, targets))

    # 14. The E storage mode, named. ``stores_E`` is what ``_read_component``
    #     branches on (stepping.py:2440), so it is what DERIVE must equal. Getting it
    #     backwards is not a crash: with stores_E True and DERIVE 1 the kernel would
    #     difference D*inv_eps while the engine's monitors read a stored E a
    #     polarization has already been subtracted from — a smooth field one
    #     polarization out of date. THIS IS THE CLAUSE THE GROUP (H) ROWS EXERCISE:
    #     they store E because they carry poles, so they take the DERIVE = 0 arm.
    if sub_step == "step_B":
        if bool(getattr(fields, "stores_E", False)):
            for name in ELECTRIC_COMPONENTS:
                if getattr(fields, name, None) is None:
                    reasons.append(f"stores_E is True but {name} is not allocated")
            reasons.extend(_layout_reasons(fields, shape, ELECTRIC_COMPONENTS))
        else:
            reasons.extend(_derive_reasons(fields, grid))
    else:
        for name in spec["sources"]:
            if getattr(fields, name, None) is None:
                reasons.append(f"{name} is not allocated (step_D differences B directly)")
        reasons.extend(_layout_reasons(fields, shape, spec["sources"]))

    # 15. The conductivity coefficients, PER TARGET of THIS sub-step, and only for
    #     the targets that carry one. A lossless component compiles to the plain
    #     line and binds a placeholder, so demanding a volume for it would refuse
    #     the mixed launch the constexprs exist to serve.
    reader = getattr(fields, "condfac_for", None)
    inverse_reader = getattr(fields, "condinv_for", None)
    if not callable(reader) or not callable(inverse_reader):
        reasons.append("fields does not expose condfac_for/condinv_for; a "
                       "conductivity this predicate cannot read is not a covered one")
    else:
        flags = conductive_no_pml_targets(fields, sub_step)
        for index, target in enumerate(targets):
            try:
                condfac = reader(target)
                condinv = inverse_reader(target)
            except Exception as exc:  # noqa: BLE001 - unreadable is not covered
                reasons.append(f"condfac_for/condinv_for({target!r}) raised "
                               f"{type(exc).__name__}")
                continue
            if (condfac is None) != (condinv is None):
                reasons.append(
                    f"{target} has condfac="
                    f"{'a volume' if condfac is not None else None} but condinv="
                    f"{'a volume' if condinv is not None else None}; the update needs "
                    f"both or neither")
                continue
            if not flags[index]:
                continue  # A lossless component compiles to the plain tail.
            if len(shape) == 3:
                reasons.extend(_volume_reasons(f"condfac[{target}]", condfac, shape))
                reasons.extend(_volume_reasons(f"condinv[{target}]", condinv, shape))

    return Coverage(not reasons, tuple(reasons))


def explain_conductive_no_pml(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return conductive_plain_curl_coverage(fields, pml, sub_step)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ConductivePlainCurlPlan:
    """A launchable, allocation-free conductive no-absorber curl sub-step.

    Deliberately a sibling of :class:`no_pml.PlainCurlPlan` rather than a subclass:
    the two hold different bindings (this one carries six more pointers) and a
    subclass that inherited ``run`` would launch the LOSSLESS kernel on a conductive
    grid — a wrong answer that never raises. Built two ways —
    :func:`plan_conductive_plain_curl` from the engine's own objects and
    :func:`plan_conductive_plain_curl_from_arrays` from bare device arrays for the
    gate — and launched ONE way, so the bytes the gate certifies are the bytes the
    engine would launch.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET AS THE PLACEHOLDER for its two
    coefficient slots, never a null. The loads sit behind a ``tl.constexpr`` branch
    and are compiled away, but a pointer argument still has to type, and passing
    ``None`` makes the launcher's failure a ``TypeError`` far from its cause. The
    placeholder is never read: ``COND == 0`` emits no conductive load at all.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "derive", "bc",
                 "cond", "block", "_targets", "_sources", "_inv_eps", "_condfac",
                 "_condinv", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, cond, block: int,
                 targets, sources, condfac, condinv, inverse_epsilon=None,
                 derive: int = 0, kernel=None) -> None:
        from .launch import CupyPointer  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply; Triton types a
        # Python float argument as fp32. Same bits. (no_pml.PlainCurlPlan, verbatim.)
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.derive = int(derive)
        self.bc = tuple(int(code) for code in bc)
        self.cond = tuple(int(bool(flag)) for flag in cond)
        if len(self.cond) != 3:
            raise ValueError("a conductive curl plan carries one COND flag per target")
        # All three flags may legitimately be false for ONE sub-step of a
        # one-sided conductive run.  Admission is intentionally per run because
        # ``no_pml.plain_curl_coverage`` refuses both curls as soon as any B or D
        # conductivity exists; the compile-time flags remain per target because
        # stepping._apply_curl reads conductivity inside the term loop.  Refusing
        # here would therefore leave (for example) step_B of a D-only run with no
        # plan even though this family's predicate admitted it.  In that slot this
        # family owns the lossless specialization; there is no overlap with the
        # incumbent predicate.
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._inv_eps = (tuple(CupyPointer(a) for a in inverse_epsilon)
                         if inverse_epsilon is not None else self._sources)
        # A lossless component's slot takes its OWN target pointer, which is always
        # allocated; ``None`` would type-fail at the launch expression.
        self._condfac = tuple(
            CupyPointer(volume) if volume is not None else self._targets[index]
            for index, volume in enumerate(condfac))
        self._condinv = tuple(
            CupyPointer(volume) if volume is not None else self._targets[index]
            for index, volume in enumerate(condinv))
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else conductive_plain_curl_kernel())
        kernel[self._grid](
            *self._targets, *self._sources, *self._inv_eps,
            *self._condfac, *self._condinv,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            DERIVE=self.derive,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"ConductivePlainCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, cond={self.cond}, derive={self.derive}, "
                f"block={self.block})")


def plan_conductive_plain_curl(fields: Any, pml: Any, sub_step: str,
                               block: Optional[int] = None
                               ) -> Optional[ConductivePlainCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage."""
    if not conductive_plain_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    # ``pml=None``, and that is faithful rather than convenient: the absorber does
    # not DECIDE the ghost rule (stepping.py:2146 — "the rule is the symmetry and
    # nothing else"); the argument exists so the layer can be checked for AGREEMENT
    # with a fold. This path refuses every fold at clause 5 and every active layer
    # at clause 3, so there is nothing left for that check to say.
    kinds = resolve(grid, None)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    derive = int(sub_step == "step_B" and not bool(getattr(fields, "stores_E", False)))
    if derive:
        sources = [getattr(fields, n) for n in spec["displacement"]]
        inverse = [fields.inverse_epsilon_for(n) for n in ELECTRIC_COMPONENTS]
    else:
        sources = [getattr(fields, n) for n in spec["sources"]]
        inverse = None
    targets = spec["targets"]
    return ConductivePlainCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes,
        conductive_no_pml_targets(fields, sub_step),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in targets],
        sources,
        [fields.condfac_for(n) for n in targets],
        [fields.condinv_for(n) for n in targets],
        inverse, derive)


def plan_conductive_plain_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                           codes, dtdx: float, cond=(1, 1, 1),
                                           derive: int = 0,
                                           block: Optional[int] = None,
                                           kernel: Any = None
                                           ) -> ConductivePlainCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``kernel=`` IS LOAD-BEARING
    and is not decoration — a mutation leg that silently launches the SHIPPED kernel
    reports every defect as uncaught, which is the harness defect the no-PML gate's
    §13.4 records at 4/4 real defects reported as identical.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    targets = spec["targets"]
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    if derive:
        sources = [arrays[n] for n in spec["displacement"]]
        inverse = [arrays["inv_eps_" + n] for n in ELECTRIC_COMPONENTS]
    else:
        sources = [arrays[n] for n in spec["sources"]]
        inverse = None
    flags = tuple(int(bool(flag)) for flag in cond)
    return ConductivePlainCurlPlan(
        sub_step, shape, dtdx, codes, flags,
        DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in targets], sources,
        [arrays["condfac_" + n] if flags[i] else None
         for i, n in enumerate(targets)],
        [arrays["condinv_" + n] if flags[i] else None
         for i, n in enumerate(targets)],
        inverse, derive, kernel=kernel)


def plan_conductive_plain_step(fields: Any, pml: Any,
                               block: Optional[int] = None) -> Dict[str, Any]:
    """Ask each curl sub-step's predicate independently, report the covered subset.

    Returns ``{"plans": {...}, "refusals": {...}}`` rather than a step-plan class:
    unlike ``no_pml.PlainStepPlan`` there is nothing this composition could
    usefully ``run()`` — the driver's boundary passes sit between ``step_B`` and
    ``step_D`` (driver.py:3212-3225) — and a class whose only method raises
    ``NotImplementedError`` is a class that exists to be misread.
    """
    plans: Dict[str, ConductivePlainCurlPlan] = {}
    refusals: Dict[str, Tuple[str, ...]] = {}
    for sub_step in SUB_STEPS:
        verdict = conductive_plain_curl_coverage(fields, pml, sub_step)
        if verdict.covered:
            plan = plan_conductive_plain_curl(fields, pml, sub_step, block=block)
            if plan is not None:
                plans[sub_step] = plan
                continue
            refusals[sub_step] = ("coverage passed but the plan builder refused",)
        else:
            refusals[sub_step] = verdict.reasons
    return {"plans": plans, "refusals": refusals}


# ---------------------------------------------------------------------------
# WIRING AND DEVICE CERTIFICATION — COMPLETE
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` now has one ``conductive no-PML`` arm, gated on the same
# inert-layer value its predicate reads.  ``__init__.py`` exposes the predicate
# and builder through the lazy package seam, and ``launch.FAMILY_MODULES`` names
# this module.  Its dedicated 2026-08-17 A6000 byte gate is recorded under
# ``triton_no_pml_conductive_device_gate`` and licenses the opted-in fast-path
# seam; it is not credited to
# ``conductivity`` or ``no_pml``, neither of which compiled this body.
#
# DISJOINTNESS, stated as the pair of clauses that carries it rather than as a
# claim. Against ``no_pml.plain_curl_coverage``: its clause 8 and this one's are
# the same question with opposite signs (:func:`_any_curl_conductivity`), so
# exactly one of the two admits any given curl slot. Against
# ``conductivity.conductive_pml_curl_coverage``: its clause 3 requires an active
# layer and this one's refuses one, which is the same ``pml.is_active`` call.
# Against ``kernels.pml_curl_step``'s predicate: clause 3 again. The gate's
# ``overlap`` leg measures all three rather than asserting them.

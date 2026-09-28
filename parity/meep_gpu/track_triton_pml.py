"""Triton track adapter for the shared real-field PML bit-identity gate.

Loaded by ``probe_fused_kernel_bit_identity.py --track triton --module <this>``.
Everything that decides pass or fail — the reference transcription, the byte
comparison, the sweep product, the multi-step leg, the mutation bookkeeping —
lives in that probe, so the two tracks' verdicts are commensurable. This file is
only the plug: it says how to install Triton's contraction guard, how to launch
the shipped kernel on the harness's arrays, and how to mutate the shipped kernel
source.

THE LAUNCH GOES THROUGH THE SHIPPED CODE. ``plan_from_arrays`` builds the same
``PmlCurlPlan`` the engine route builds and calls the same ``run``, so the bytes
the gate certifies are the bytes ``plan_pml_curl`` would launch — not a
re-implementation written for the harness.

THE MUTATION LEG COMPILES A DERIVED COPY. ``inspect.getsource`` on the shipped
``@triton.jit`` function, textually mutated, written to a temporary module and
imported (Triton parses a kernel through ``inspect.getsource``, so an ``exec``-ed
function has no source for it to read). The mutated kernel is therefore genuinely
the shipped kernel with one defect introduced, which is the only version of a
mutation test worth running.

Preconditions on a fresh box::

    mkdir -p ~/triton_libcuda_stub
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/triton_libcuda_stub/libcuda.so
    export TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub

Triton links its ``cuda_utils`` extension with ``-lcuda``, which needs a
development symlink a driver-only CUDA install does not ship.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import sys
import tempfile
import textwrap
from typing import Any, Dict, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from meep_gpu.triton_kernels import kernels as _kernels  # noqa: E402
from meep_gpu.triton_kernels import dispersive_update_e as _dispersive  # noqa: E402
from meep_gpu.triton_kernels.launch import (  # noqa: E402
    plan_constitutive_from_arrays,
    plan_from_arrays,
    plan_fused_pair_from_arrays,
)

#: The shipped kernels' sources, for the mutation leg to rewrite. One entry per
#: kernel the gate can drive; ``apply_source_mutation`` requires a mutation to
#: match SOMEWHERE, not everywhere, because a needle aimed at one kernel does not
#: appear in the others.
PML_CURL_KERNEL_SOURCE = textwrap.dedent(inspect.getsource(_kernels.pml_curl_step.fn))
CONSTITUTIVE_KERNEL_SOURCE = textwrap.dedent(
    inspect.getsource(_kernels.constitutive_step.fn))
FUSED_PAIR_KERNEL_SOURCE = textwrap.dedent(
    inspect.getsource(_kernels.fused_curl_constitutive_B.fn))
#: The dispersive ``update_E`` kernel. It is built on demand rather than decorated
#: at module scope — its module has to import on a laptop with no Triton — so the
#: source is taken off the built function, which is the same text either way.
DISPERSIVE_KERNEL_SOURCE = textwrap.dedent(inspect.getsource(
    _dispersive.constitutive_step_dispersive_kernel().fn))

#: Set by :func:`begin_guard`; consumed by :func:`launch`.
_GUARD: bool = False
_MUTATED_KERNEL: Any = None
_MUTATED_CONSTITUTIVE: Any = None
_MUTATED_FUSED_PAIR: Any = None
_MUTATED_DISPERSIVE: Any = None
_MUTATED_SOURCE: Optional[str] = None
_TEMPORARY: list = []


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
# Triton's `enable_fp_fusion` gates BOTH stages of its pipeline, where NVRTC's
# `--fmad=false` reaches only ptxas:
#   compiler.py:259  llvm.translate_to_asm(..., opt.enable_fp_fusion, False)
#   compiler.py:284  fmad = '' if opt.enable_fp_fusion else ' --fmad=false'
# The token is a one-string tuple because the probe records `list(guard)`.

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fp_fusion_off", ("enable_fp_fusion=False",), True),
    ("fp_fusion_default_on", ("enable_fp_fusion=True",), False),
)
GUARD_DESCRIPTION = (
    "Triton CUDAOptions.enable_fp_fusion — gates the LLVM contraction pass AND "
    "passes ptxas --fmad=false; default True")


def begin_guard(token: Sequence[str]) -> None:
    global _GUARD
    _GUARD = str(token[0]).split("=")[1].strip().lower() == "true"


def end_guard() -> None:
    global _GUARD
    _GUARD = False


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any],
           codes: Sequence[Any], dtdx: float,
           extra: Optional[Dict[str, Any]] = None) -> None:
    """One sub-step, in place, through the shipped plan and the shipped ``run``.

    ``sub_step`` is one of ``step_B``/``step_D`` (the curl), ``update_H``/
    ``update_E`` (the constitutive accumulation) or ``fused_pair_B`` (both magnetic
    sub-steps in one launch). ``codes`` and ``dtdx`` are the curl's; the
    constitutive sub-step reads no neighbour and has no Courant scalar, so it
    ignores both — the shared signature is what lets one adapter method serve every
    leg of the gate.

    ``extra`` carries what only the fused pair needs and nothing else has:
    ``h_flat`` (the constitutive half's own kps/kms, on ITS OWN sub-lattice, chosen
    by the caller so the gate can hand over a swapped one), ``zero_metal`` (the
    three wall flags the kernel carries as constexprs) and ``num_warps``.

    ``num_warps`` CHANGES THE COMPILED KERNEL, so it is a gate axis and not a
    tuning knob: a value that was never swept is a value the byte gate never
    certified. ``None`` means Triton's default.
    """
    if sub_step == "fused_pair_B":
        bindings = extra or {}
        plan = plan_fused_pair_from_arrays(
            "B", arrays, flat, bindings["h_flat"], [int(c) for c in codes],
            tuple(bool(f) for f in bindings["zero_metal"]), float(dtdx),
            kernel=_MUTATED_FUSED_PAIR, num_warps=bindings.get("num_warps"))
        plan.run(guard=_GUARD)
        return
    if sub_step == "update_E_dispersive":
        bindings = extra or {}
        plan = _dispersive.plan_dispersive_constitutive_from_arrays(
            arrays, flat, bindings["poles"], kernel=_MUTATED_DISPERSIVE)
        plan.run(guard=_GUARD)
        return
    if sub_step in ("update_H", "update_E"):
        plan = plan_constitutive_from_arrays(
            sub_step.split("_", 1)[1], arrays, flat, kernel=_MUTATED_CONSTITUTIVE)
        plan.run(guard=_GUARD)
        return
    plan = plan_from_arrays(sub_step, arrays, flat, [int(c) for c in codes],
                            float(dtdx), kernel=_MUTATED_KERNEL)
    plan.run(guard=_GUARD)


# ---------------------------------------------------------------------------
# Source mutations — the Triton spellings of the gate's four defects
# ---------------------------------------------------------------------------
# The gate owns WHICH defects are injected and what must happen when they are;
# only the needles are per-track, because one track's kernel is a CUDA string and
# the other's is Python. Each transform returns (mutated source, sites hit), and
# the gate refuses a mutation that matched nothing — a mutation that has drifted
# away from the kernel is not exercising anything.

_STENCIL = re.compile(
    r"dtdx \* \(\((\w+) - (\w+)\) \+ \((\w+) - (\w+)\)\)")
_MASK = re.compile(r"tl\.where\(at_[xyz], 0\.0, (curl\d)\)")
_RECURRENCE = re.compile(
    r"n(?P<t>\d) = \(\(p(?P=t) \* km_(?P<s>[xyz])\) - curl(?P=t)\) \* si_(?P=s)\n"
    r"(?P<pad>\s+)v(?P=t) = \(\(\(tl\.load\(f(?P=t) \+ idx, mask=live, other=0\.0\)"
    r" \* km_(?P<u>[xyz])\) \+ n(?P=t)\) - p(?P=t)\) \* si_(?P=u)")
_FU_STORE = re.compile(r"\n *tl\.store\(u\d \+ idx, n\d, mask=live\)")


def _regroup_stencil(source: str) -> Tuple[str, int]:
    """Mutation 3: ``dtdx*((a-b)+(c-d))`` -> ``dtdx*(a-b+c-d)``, i.e. ``((a-b)+c)-d``."""
    return _STENCIL.sub(r"dtdx * (\1 - \2 + \3 - \4)", source), \
        len(_STENCIL.findall(source))


def _drop_metallic_mask(source: str) -> Tuple[str, int]:
    """Mutation 4: the wall's cell-0 curl is no longer zeroed.

    Rewritten to ``curl0 = curl0`` rather than deleted, because the statement is
    the whole body of a ``constexpr`` ``if`` and removing it is a syntax error —
    a mutation that fails to compile measures nothing.
    """
    return _MASK.sub(r"\1", source), len(_MASK.findall(source))


def _swap_dsig_dsigu(source: str) -> Tuple[str, int]:
    """Mutation 5: the recurrence reads dsigu's coefficients where dsig's belong."""
    def swap(match: "re.Match[str]") -> str:
        t, s, u, pad = match["t"], match["s"], match["u"], match["pad"]
        return (f"n{t} = ((p{t} * km_{u}) - curl{t}) * si_{u}\n"
                f"{pad}v{t} = (((tl.load(f{t} + idx, mask=live, other=0.0) "
                f"* km_{s}) + n{t}) - p{t}) * si_{s}")
    return _RECURRENCE.sub(swap, source), len(_RECURRENCE.findall(source))


def _drop_fu_store(source: str) -> Tuple[str, int]:
    """The auxiliary-only defect: ``f`` reads the register, ``fu`` is never written.

    Right in ``f`` on every single launch, wrong in ``fu`` from the first one —
    the failure only an auxiliary comparison, or a multi-step run, can see.
    """
    return _FU_STORE.sub("", source), len(_FU_STORE.findall(source))


# --- the constitutive kernel's spellings of the same defect families ---------
# Each is aimed at ``constitutive_step`` and matches nothing in the curl kernel,
# which is why the gate's drift check is on the TOTAL across attributes rather
# than per attribute.

_CONST_ACCUMULATE = re.compile(
    r"a(?P<t>\d) = a(?P=t) \+ kp_(?P=t) \* src(?P=t)\n"
    r"(?P<pad>\s+)a(?P=t) = a(?P=t) - km_(?P=t) \* prev(?P=t)")
_CONST_FW_STORE = re.compile(r"\n *tl\.store\(w(\d) \+ idx, src\1, mask=live\)")
_CONST_INV_EPS = re.compile(
    r"src(?P<t>\d) = tl\.load\(g(?P=t) \+ idx, mask=live, other=0\.0\)"
    r" \* tl\.load\(e(?P=t) \+ idx, mask=live,\n\s+other=0\.0\)")


def _regroup_constitutive(source: str) -> Tuple[str, int]:
    """``((f + kps*src) - kms*prev)`` -> ``f + (kps*src - kms*prev)``.

    The array path spells the update as two accumulations (``field += kps * fw``
    then ``field -= kms * fw_previous``), so this is the same defect family as the
    curl's regroup: a different float32 number, no error, no crash.
    """
    return (_CONST_ACCUMULATE.sub(
        lambda m: (f"a{m['t']} = a{m['t']} + (kp_{m['t']} * src{m['t']} "
                   f"- km_{m['t']} * prev{m['t']})\n"
                   f"{m['pad']}a{m['t']} = a{m['t']}"), source),
        len(_CONST_ACCUMULATE.findall(source)))


def _drop_fw_store(source: str) -> Tuple[str, int]:
    """The auxiliary-only defect on this sub-step: ``f_w`` is never written.

    ``f`` is right on every single launch — the accumulation reads the register —
    and ``f_w`` is wrong from the first one. §12.3 measured the identical defect on
    the curl as 120/120 UNCAUGHT when the auxiliary was not compared.
    """
    return _CONST_FW_STORE.sub("", source), len(_CONST_FW_STORE.findall(source))


def _store_fw_before_reading_prev(source: str) -> Tuple[str, int]:
    """RISK 4: the kernel reads its own write in place of ``fw_previous``.

    ``_apply_constitutive_pml`` copies ``fw`` BEFORE overwriting it, and this is
    what happens when a fused kernel does not: ``prev`` becomes ``src``. Wrong only
    where ``kms != 0`` — i.e. inside the PML only — so it looks like a slightly
    worse absorber rather than like a bug.
    """
    count = 0
    out = source
    for index in (0, 1, 2):
        # Reading g (the source) in place of w (the auxiliary) IS reading the value
        # the store is about to write, without reordering statements — the same
        # wrong number, expressed as a substitution the mutation leg can count.
        pattern = re.compile(
            rf"prev{index} = tl\.load\(w{index} \+ idx, mask=live, other=0\.0\)")
        out, hits = pattern.subn(
            f"prev{index} = tl.load(g{index} + idx, mask=live, other=0.0)",
            out, count=1)
        count += hits
    return out, count


def _drop_inverse_epsilon(source: str) -> Tuple[str, int]:
    """E side only: ``source`` becomes ``D`` instead of ``D * inv_eps``.

    Invisible whenever the inverse-epsilon table is 1.0, which is why the gate's
    E-side arrays draw it from [0.2, 0.9).
    """
    return (_CONST_INV_EPS.sub(
        lambda m: f"src{m['t']} = tl.load(g{m['t']} + idx, mask=live, other=0.0)",
        source),
        len(_CONST_INV_EPS.findall(source)))


# --- the DISPERSIVE update_E kernel's own defect families ---------------------
# These are aimed at ``constitutive_step_dispersive``. Most of them match nothing in
# the other three kernels, which is why the gate's drift check is on the TOTAL across
# attributes rather than per attribute.
#
# TWO OF THEM ALSO HIT THE NON-DISPERSIVE KERNELS, and the site counts in the
# artifact say so rather than the prose claiming otherwise:
# ``drop_fw_store_dispersive``'s needle is the same statement shape
# ``constitutive_step`` and ``fused_curl_constitutive_B`` write (3 sites each), and
# ``store_fw_before_reading_prev_dispersive``'s likewise. That is harmless — the
# dispersive leg launches only the dispersive kernel — and it is worth having in the
# record: a needle that matched only the intended kernel because it was written
# around a rename would be a weaker anchor, not a stronger one.
#
# The kernel's subtraction chain is a run of statements guarded by ``tl.constexpr``
# conditions, one per compiled pole slot::
#
#     s0 = tl.load(g0 + idx, mask=live, other=0.0)
#     if NP0 > 0:
#         s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)
#     if NP0 > 1:
#         s0 = s0 - tl.load(a1 + idx, mask=live, other=0.0)
#     ...
#     src0 = s0 * tl.load(e0 + idx, mask=live, other=0.0)
#
# so every needle below is anchored on that shape and reports its site count.

_DISP_FIRST_POLE = re.compile(
    r"s(?P<t>\d) = s(?P=t) - tl\.load\((?P<p>[abc])0 \+ idx, mask=live, other=0\.0\)")
_DISP_LATER_POLE = re.compile(
    r"s(?P<t>\d) = s(?P=t) - tl\.load\((?P<p>[abc][1-7]) \+ idx, mask=live, other=0\.0\)")
_DISP_SRC_LINE = re.compile(
    r"(?P<pad>\n(?P<ind> *))src(?P<t>\d) = s(?P=t)"
    r" \* tl\.load\(e(?P=t) \+ idx, mask=live, other=0\.0\)")
_DISP_ACCUMULATE = re.compile(
    r"v(?P<t>\d) = v(?P=t) \+ kp_(?P=t) \* src(?P=t)\n"
    r"(?P<pad>\s+)v(?P=t) = v(?P=t) - km_(?P=t) \* prev(?P=t)")
_DISP_FW_STORE = re.compile(r"\n *tl\.store\(w(\d) \+ idx, src\1, mask=live\)")
_DISP_PREV_USE = re.compile(r"v(?P<t>\d) = v(?P=t) - km_(?P=t) \* prev(?P=t)")
_DISP_FIRST_GUARD = re.compile(r"if NP(?P<t>\d) > 0:")


def _sum_then_subtract(source: str) -> Tuple[str, int]:
    """THE trap this kernel exists to avoid: ``D - (P0 + P1)`` instead of ``(D - P0) - P1``.

    The natural design — accumulate one "sum of P" volume and subtract it once — is
    what anyone writes from ``update_E``'s docstring sentence, and it is a DIFFERENT
    float32 number at two or more poles. Measured on the real engine at 20/20 caught
    with two poles and 0/10 with one; a leg reporting it caught on a single-pole row
    under the uniform value class is reporting a harness defect.

    Spelled without a zero seed — the first pole initialises the accumulator — so the
    zero-pole arm is byte-unchanged and the mutation is exactly the multi-pole defect
    and nothing else.
    """
    out, first = _DISP_FIRST_POLE.subn(
        lambda m: (f"q{m['t']} = tl.load({m['p']}0 + idx, mask=live, other=0.0)"),
        source)
    out, later = _DISP_LATER_POLE.subn(
        lambda m: (f"q{m['t']} = q{m['t']} + tl.load({m['p']} + idx, "
                   f"mask=live, other=0.0)"), out)
    out, srcs = _DISP_SRC_LINE.subn(
        lambda m: (f"\n{m['ind']}if NP{m['t']} > 0:"
                   f"\n{m['ind']}    s{m['t']} = s{m['t']} - q{m['t']}"
                   f"{m['pad']}src{m['t']} = s{m['t']} * tl.load(e{m['t']} + idx, "
                   f"mask=live, other=0.0)"), out)
    return out, first + later + srcs


def _regroup_constitutive_dispersive(source: str) -> Tuple[str, int]:
    """``((E + kps*src) - kms*prev)`` -> ``E + (kps*src - kms*prev)``.

    The array path spells the update as two accumulations (``field += kps * fw``
    then ``field -= kms * fw_previous``), so this is the constitutive regroup with
    the dispersive source. Caught 30/30 on the real engine, at BOTH Courant numbers —
    the non-representable multiplicands are the PML coefficients themselves.
    """
    return (_DISP_ACCUMULATE.sub(
        lambda m: (f"v{m['t']} = v{m['t']} + (kp_{m['t']} * src{m['t']} "
                   f"- km_{m['t']} * prev{m['t']})\n"
                   f"{m['pad']}v{m['t']} = v{m['t']}"), source),
        len(_DISP_ACCUMULATE.findall(source)))


def _drop_fw_store_dispersive(source: str) -> Tuple[str, int]:
    """``f_w_E`` is never written — and ``f_w_E`` is ``update_P``'s input.

    Worse than the non-dispersive version of the same defect: ``Fields.drive_field``
    hands ``f_w_<c>`` to every polarization under PML (fields.py:1140-1162), so a
    dropped store leaves ``E`` right on the launch and every subsequent P wrong. The
    blindness control runs this same mutation with ``--no-fu-compare`` and it must
    then come back UNCAUGHT.
    """
    return _DISP_FW_STORE.sub("", source), len(_DISP_FW_STORE.findall(source))


def _store_fw_before_reading_prev_dispersive(source: str) -> Tuple[str, int]:
    """The aliasing trap ``_apply_constitutive_pml``'s ``fw.copy()`` exists to prevent.

    The array path copies ``fw`` BEFORE overwriting it. A kernel that reads it after
    the store gets ``src`` where ``fw_previous`` belongs — which is what this
    substitutes, without reordering statements, so the mutation leg can count it.
    Wrong only where ``kms != 0``, i.e. INSIDE THE PML ONLY, so it looks like a
    slightly worse absorber rather than like a bug.
    """
    return (_DISP_PREV_USE.sub(
        lambda m: f"v{m['t']} = v{m['t']} - km_{m['t']} * src{m['t']}", source),
        len(_DISP_PREV_USE.findall(source)))


def _drop_one_pole(source: str) -> Tuple[str, int]:
    """The first pole of every component silently falls out of the chain.

    ``if NP > 0`` becomes a condition no compiled configuration satisfies, so slot 0
    is never subtracted. A component the predicate admitted with NO pole is
    unaffected, which is the right signature: this is a defect in carrying poles, not
    in the arithmetic around them.
    """
    return (_DISP_FIRST_GUARD.sub(lambda m: f"if NP{m['t']} > 99:", source),
            len(_DISP_FIRST_GUARD.findall(source)))


def _drop_inverse_epsilon_dispersive(source: str) -> Tuple[str, int]:
    """``source`` becomes ``D - sum P`` instead of ``(D - sum P) * inv_eps``.

    Invisible whenever the inverse-epsilon table is 1.0, which is why the gate's
    arrays draw it from [0.2, 0.9).
    """
    return (_DISP_SRC_LINE.sub(
        lambda m: f"{m['pad']}src{m['t']} = s{m['t']}", source),
        len(_DISP_SRC_LINE.findall(source)))


def _inv_eps_left(source: str) -> Tuple[str, int]:
    """THE NULL. ``inv_eps * s`` in place of ``s * inv_eps``. It MUST come back UNCAUGHT.

    float32 multiplication is bitwise commutative — measured 0/30 caught on real
    engine cases — so the array path's operand order is transcription discipline and
    NOT a gate axis. This mutation is carried so that claim is measured rather than
    asserted: a leg that reports it CAUGHT is comparing something other than bytes.
    """
    return (_DISP_SRC_LINE.sub(
        lambda m: (f"{m['pad']}src{m['t']} = tl.load(e{m['t']} + idx, mask=live, "
                   f"other=0.0) * s{m['t']}"), source),
        len(_DISP_SRC_LINE.findall(source)))


# --- the cross-sub-step pair's own defect families ---------------------------
# These exist only because two sub-steps now share one launch. They are aimed at
# `fused_curl_constitutive_B` and match nothing in the other two kernels.

_FUSED_SEAM = re.compile(r"src(?P<t>\d) = v(?P=t)$", re.MULTILINE)
_FUSED_WIPE = re.compile(
    r"v(?P<t>\d) = tl\.where\(at_(?P<ax>[xyz]), 0\.0, v(?P=t)\)")
_FUSED_STORE_F = re.compile(
    r"tl\.store\(f(?P<t>\d) \+ idx, v(?P=t), mask=live\)")
_FUSED_STORE_BLOCK = re.compile(
    r"(?P<pad>\n    )tl\.store\(u0 \+ idx, n0, mask=live\)"
    r"(?:\n    tl\.store\(\w\d \+ idx, \w\d, mask=live\)){5}")
#: Which v-register each axis's wall wipe belongs to, for the two ZM mutations.
_ZM_AXIS_OF = {"0": "x", "1": "y", "2": "z"}


def _reload_B_from_memory(source: str) -> Tuple[str, int]:
    """THE TRUNCATION QUESTION, asked as a mutation. It MUST come back UNCAUGHT.

    Puts the float32 round trip back — the constitutive half re-loads ``B`` from
    the array the curl half just stored, at the same index — while keeping every
    other consequence of the fusion. On NVIDIA a Triton fp32 value lives in a
    32-bit register, so ``store(reg); load()`` is the identity on the bits and this
    leg must be 64/64 identical. Its shifted sibling is what stops that from being
    a tautology.
    """
    return (_FUSED_SEAM.sub(
        lambda m: (f"src{m['t']} = tl.load(f{m['t']} + idx, mask=live, other=0.0)"),
        source), len(_FUSED_SEAM.findall(source)))


def _reload_B_from_memory_but_shift(source: str) -> Tuple[str, int]:
    """The discriminating control: the reload reads the NEXT cell.

    Same edit as :func:`_reload_B_from_memory` with the index advanced by one
    (wrapped, so nothing reads past the allocation). If the harness could not catch
    this, the "identical" verdict on the unshifted reload would be a statement
    about the comparison rather than about the seam.
    """
    return (_FUSED_SEAM.sub(
        lambda m: (f"src{m['t']} = tl.load(f{m['t']} + (idx + 1) % n_elem, "
                   f"mask=live, other=0.0)"),
        source), len(_FUSED_SEAM.findall(source)))


def _drop_zero_metal_inline(source: str) -> Tuple[str, int]:
    """``zero_metal_B`` is no longer carried across the seam.

    Rewritten to ``v0 = v0`` rather than deleted: the statement is the whole body
    of a ``constexpr`` ``if`` and removing it is a syntax error, and a mutation that
    fails to compile measures nothing (the same reason ``_drop_metallic_mask`` is
    spelled this way). INVISIBLE ON EVERY BENCHMARK CASE — they are all periodic
    under PML, MEEP's default being the Bloch wrap — so this one only bites on a
    walled configuration, which is why the gate builds one.
    """
    return (_FUSED_WIPE.sub(lambda m: f"v{m['t']} = v{m['t']}", source),
            len(_FUSED_WIPE.findall(source)))


def _zero_metal_only_on_the_store(source: str) -> Tuple[str, int]:
    """The ORDERING defect a naive fusion introduces: B is wiped, the H read is not.

    ``B`` still lands in memory with its wall plane cleared, so an inspection of the
    field array looks right; but ``update_H`` accumulated from the UNWIPED value,
    because in the array path the wipe happened between the two sub-steps and here
    it did not. One plane of one component, on a walled run only.

    Note this is NOT the same mutation as moving the wipe below the stores: nothing
    reads ``f0..f2`` back, so that edit is unobservable — see
    ``hoist_constitutive_above_curl_store``. Carrying the wipe on the store alone is
    the version that is a real defect.
    """
    moved = _FUSED_STORE_F.sub(
        lambda m: (f"tl.store(f{m['t']} + idx, "
                   f"tl.where(at_{_ZM_AXIS_OF[m['t']]}, 0.0, v{m['t']}), mask=live)"),
        source)
    dropped, count = _FUSED_WIPE.subn(lambda m: f"v{m['t']} = v{m['t']}", moved)
    return dropped, count


def _hoist_constitutive_above_curl_store(source: str) -> Tuple[str, int]:
    """The ordering, as a NULL mutation: it MUST come back UNCAUGHT.

    Moves the curl half's six stores below the whole constitutive block. Nothing
    reads ``f0..f2`` or ``u0..u2`` back, so the edit cannot change a bit — and
    measuring that is how "the seam is a pure register substitution, not a
    reordering hazard" stops being an assertion. A leg that reports this CAUGHT is
    reporting a defect in the kernel or in the harness.
    """
    match = _FUSED_STORE_BLOCK.search(source)
    if match is None:
        return source, 0
    block = match.group(0)
    return source.replace(block, "", 1).rstrip("\n") + block + "\n", 1


SOURCE_MUTATIONS = {
    "regroup_stencil": _regroup_stencil,
    "drop_metallic_mask": _drop_metallic_mask,
    "swap_dsig_dsigu": _swap_dsig_dsigu,
    "drop_fu_store": _drop_fu_store,
    "regroup_constitutive": _regroup_constitutive,
    "drop_fw_store": _drop_fw_store,
    "store_fw_before_reading_prev": _store_fw_before_reading_prev,
    "drop_inverse_epsilon": _drop_inverse_epsilon,
    "reload_B_from_memory": _reload_B_from_memory,
    "reload_B_from_memory_but_shift": _reload_B_from_memory_but_shift,
    "drop_zero_metal_inline": _drop_zero_metal_inline,
    "zero_metal_only_on_the_store": _zero_metal_only_on_the_store,
    "hoist_constitutive_above_curl_store": _hoist_constitutive_above_curl_store,
    "sum_then_subtract": _sum_then_subtract,
    "regroup_constitutive_dispersive": _regroup_constitutive_dispersive,
    "drop_fw_store_dispersive": _drop_fw_store_dispersive,
    "store_fw_before_reading_prev_dispersive":
        _store_fw_before_reading_prev_dispersive,
    "drop_one_pole": _drop_one_pole,
    "drop_inverse_epsilon_dispersive": _drop_inverse_epsilon_dispersive,
    "inv_eps_left": _inv_eps_left,
}


def source_attributes() -> Tuple[str, ...]:
    return ("PML_CURL_KERNEL_SOURCE", "CONSTITUTIVE_KERNEL_SOURCE",
            "FUSED_PAIR_KERNEL_SOURCE", "DISPERSIVE_KERNEL_SOURCE")


#: Which shipped function each mutable source string defines, so the compiled
#: replacement is routed to the right launcher.
_MUTABLE_KERNELS = {
    "PML_CURL_KERNEL_SOURCE": "pml_curl_step",
    "CONSTITUTIVE_KERNEL_SOURCE": "constitutive_step",
    "FUSED_PAIR_KERNEL_SOURCE": "fused_curl_constitutive_B",
    "DISPERSIVE_KERNEL_SOURCE": "constitutive_step_dispersive",
}


def set_source_mutation(mutated: Dict[str, str]) -> None:
    """Compile the mutated kernel(s) from a real file, and route launches to them.

    Triton reads a kernel's text with ``inspect.getsource``, so the mutated
    function has to live in a file on disk; an ``exec``-ed one raises
    ``OSError: could not get source code`` at first launch.
    """
    global _MUTATED_KERNEL, _MUTATED_CONSTITUTIVE, _MUTATED_FUSED_PAIR
    global _MUTATED_DISPERSIVE, _MUTATED_SOURCE

    _MUTATED_KERNEL = None
    _MUTATED_CONSTITUTIVE = None
    _MUTATED_FUSED_PAIR = None
    _MUTATED_DISPERSIVE = None
    _MUTATED_SOURCE = None
    for attribute, function_name in _MUTABLE_KERNELS.items():
        source = mutated.get(attribute)
        if source is None:
            continue
        # The curl kernel reads PERIODIC/METALLIC as tl.constexpr globals; the
        # mutated copy must declare them the same way or Triton refuses the name.
        header = ("import triton\nimport triton.language as tl\n"
                  "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
        handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_pml.py",
                                             delete=False)
        handle.write(header + source)
        handle.close()
        _TEMPORARY.append(handle.name)
        spec = importlib.util.spec_from_file_location(
            "triton_mutated_pml_" + str(len(_TEMPORARY)), handle.name)
        module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
        sys.modules[spec.name] = module                      # type: ignore[union-attr]
        spec.loader.exec_module(module)                      # type: ignore[union-attr]
        compiled = getattr(module, function_name)
        if function_name == "pml_curl_step":
            _MUTATED_KERNEL = compiled
        elif function_name == "constitutive_step":
            _MUTATED_CONSTITUTIVE = compiled
        elif function_name == "constitutive_step_dispersive":
            _MUTATED_DISPERSIVE = compiled
        else:
            _MUTATED_FUSED_PAIR = compiled
        _MUTATED_SOURCE = source


def mutated_kernels() -> Dict[str, Any]:
    """The mutated kernel per PLAN CLASS, so the engine-route legs are armed too.

    The synthetic legs receive a mutated kernel through
    ``plan_*_from_arrays(kernel=...)``. The whole-step leg does not — it builds its
    plans with ``plan_step``, which passes no kernel — so without this mapping every
    whole-step mutation leg launches the SHIPPED kernel and reports a pass. See
    ``ExternalTrackAdapter.mutated_kernels`` for the measurement that found it.
    """
    return {"PmlCurlPlan": _MUTATED_KERNEL,
            "ConstitutivePlan": _MUTATED_CONSTITUTIVE,
            "FusedPairPlan": _MUTATED_FUSED_PAIR,
            "DispersiveConstitutivePlan": _MUTATED_DISPERSIVE}


def describe() -> Dict[str, Any]:
    """Versions and knobs that decide this track's answer, for the artifact."""
    import triton  # noqa: PLC0415
    out: Dict[str, Any] = {
        "triton_version": triton.__version__,
        "triton_path": os.path.dirname(triton.__file__),
        "TRITON_LIBCUDA_PATH": os.environ.get("TRITON_LIBCUDA_PATH", "<unset>"),
        "block": _kernels.DEFAULT_BLOCK,
        "enable_fp_fusion_default_in_source": _kernels.ENABLE_FP_FUSION,
    }
    try:
        import torch  # noqa: PLC0415
        out["torch_version"] = torch.__version__
    except Exception as exc:  # noqa: BLE001
        out["torch_version"] = f"ABSENT: {type(exc).__name__}"
    return out


def kernel_lines() -> Dict[str, int]:
    """Non-blank, non-comment, non-docstring lines of the kernel and its host side."""
    import ast  # noqa: PLC0415

    def measure(text: str) -> int:
        tree = ast.parse(textwrap.dedent(text))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) \
                    and ast.get_docstring(node):
                node.body = node.body[1:]
        return sum(1 for line in ast.unparse(tree).splitlines() if line.strip())

    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415
    return {
        "kernel_pml_curl_step": measure(PML_CURL_KERNEL_SOURCE),
            "kernel_ade_update_p": measure(inspect.getsource(_kernels.ade_update_p.fn)),
        "kernel_fused_curl_constitutive_B": measure(FUSED_PAIR_KERNEL_SOURCE),
        "kernel_constitutive_step_dispersive": measure(DISPERSIVE_KERNEL_SOURCE),
        "host_dispersive_module": measure(
            open(_dispersive.__file__, encoding="utf-8").read()),
        "host_launch_module": measure(
            open(launch_module.__file__, encoding="utf-8").read()),
        "coverage_predicate": measure(
            open(os.path.join(os.path.dirname(launch_module.__file__),
                              "coverage.py"), encoding="utf-8").read()),
    }


PROBE_ADAPTER: Dict[str, Any] = {
    "track": "triton",
    "guard_sets": GUARD_SETS,
    "guard_description": GUARD_DESCRIPTION,
    "begin_guard": begin_guard,
    "end_guard": end_guard,
    "launch": launch,
    "source_attributes": source_attributes,
    "set_source_mutation": set_source_mutation,
    "source_mutations": SOURCE_MUTATIONS,
    "mutated_kernels": mutated_kernels,
    "describe": describe,
    "kernel_lines": kernel_lines,
}

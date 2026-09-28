"""The CONDUCTIVE hand-CUDA fused pair: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE BOARD'S CELL is ``D_to_E (cuda_conductive/conductive, cuda_constitutive/ordinary)``
and it owns exactly one corpus row -- ``tests:TestAdjointSolver.test_damping`` -- read
from ``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-02_tierfix``'s
``taxonomy.instances.buildable_not_built``. Both halves were already CERTIFIED
(``conductive_kernels.step_D_pml_conductive`` and
``constitutive_kernels.update_E_pml_real``) and the verdict was POINTWISE-BUILDABLE, so
what was missing was the weld and nothing else.

SIBLINGS. ``metal_kernels/conductive_fused_electric_pair.py`` and
``triton_kernels/conductive_fused_electric_pair.py``. This is a PORT of that shape, not
an invention -- CUDA was the only backend without it.

This module is the twin of :mod:`.fused_electric_pair` with ONE half exchanged: the
curl is the THREE-HISTORY conductive-PML recurrence instead of the two-history real
one. The constitutive half is the same certified ``update_E_pml_real``.

=============================================================================
THIS IS THE SIGNED-ZERO ROW, AND WHAT MADE IT SERVABLE
=============================================================================

``FdtdDriver._inject_electric_through_conductivity`` scales an injected current by
``condinv`` (MEEP step.cpp:294-317). It USED to do that as three WHOLE-VOLUME passes --
``array -= before; array *= condinv; array += before`` -- which are the identity at
every finite value EXCEPT that they canonicalise ``-0.0`` to ``+0.0`` at every cell of
the component, and on a device that flushes subnormals also ZERO every non-deposit
subnormal word. Both rewrites land at cells no deposit closure can name, so a fused
launch computing E from pre-rewrite values diverged in words no repair could reach.
THAT is why the fused electric pairs on three backends refused conductive rows by name,
and why this cell stood empty on every backend until 2026-09-01.

THE DRIVER NO LONGER DOES THAT. It snapshots the target at the deposit cells the
sources publish (``deposit_repair._deposit_index`` reads the same tables) and replays
the rescale per deposit cell in the retired passes' exact operand order -- exact at
every deposit cell, untouched everywhere else, and closer to stock MEEP, which scales
only the injected current at the source's own points. Triton's and Metal's refusal
clauses were lifted on that measurement, and this family is built on it.

RE-MEASURED FOR THIS PRODUCT RATHER THAN INHERITED.
``parity/meep_gpu/probe_cuda_conductive_electric_pair.py``'s ``rescale`` leg drives the
SHIPPED method on a fixture seeded with all four word classes and an exactly-zero curl
(so a ``-0.0`` survives ``step_D`` to the injection point) and measures that NOT ONE
non-deposit word moves, on 4 of 4 configurations. The same leg replays the RETIRED
dense composition as a control: it diverges on every configuration -- 18, 189, 243 and
4 words -- and every differing word classifies as SIGNED-ZERO, with ``other`` at zero.
The classifier admits the SUBNORMAL class too and records it ``predicted_null`` with
its reason (NumPy does not flush subnormals, so only a device run can arm that half;
the published CuPy figure is 68 subnormal beside 54 signed-zero of 588).

WHAT SURVIVES IS THE DRIVER'S OWN FALLBACK. A scaled source publishing NO deposit table
(no ``_point_ix``) still takes the whole-volume passes, because an unnameable deposit
must still be scaled. No in-tree electric source class is one; the clause below stays
for the duck-typed stranger, and it is a TRUE refusal rather than a leftover.

=============================================================================
THE SEAM, PASS BY PASS
=============================================================================

* **the electric injection -- CARRIED, through the SPLIT-FIELD deposit repair.** The
  layer is ACTIVE on this cell, so ``update_E`` runs the dsigw accumulation
  (``stepping.py:1014-1018``) and the repair that inverts it is the default one.
  :data:`REPAIR_PATHS` is therefore the default and is spelled anyway, so a reader can
  see that it was decided rather than inherited. The corpus row declares both a ``D``
  and a ``B`` source; only the ``D`` one is in THIS seam.
* **``zero_metal_D`` -- CARRIED INLINE.** The corpus row is METALLIC on x AND y, so two
  wall rows are live and declining the pass would cost the whole cell.
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` -- INERT, because a mirror
  plane is REFUSED**, by name, as on both siblings. The corpus row has no symmetry.

=============================================================================
THE CONDUCTIVE MASK IS A COMPILE-TIME AXIS
=============================================================================

``conductive_kernels`` bakes the per-component conductivity into three ``COND`` defines
and takes the certified ``pml_apply`` behind ``#else`` where a component is lossless.
This family emits one device string per mask, and the mask comes from
``conductive_kernels.conductive_targets`` -- the same function the predicate reads,
which is the discipline that keeps a clause and a compile-time choice from drifting
apart.

THE FULLY LOSSLESS MASK IS NOT EMITTED, and that is a refusal rather than an omission:
``covers_conductive_curl`` refuses a run where no target carries a sigma BY NAME, and
that run belongs to :mod:`.fused_electric_pair`. Emitting ``(0, 0, 0)`` here would be a
body no admitted configuration can reach.

=============================================================================
THE ALIASING HAZARD
=============================================================================

D appears EXACTLY ONCE in the signature, for :mod:`.fused_electric_pair`'s reason: the
curl writes it and the constitutive half reads it from a register, so binding it a
second time as ``const __restrict__`` would be two restrict pointers to one allocation,
which NVRTC miscompiles without a diagnostic.

THE CONDUCTIVITY POINTERS ARE NOT ``__restrict__``, and that is a DELIBERATE DEPARTURE
from the certified curl's own signature, which does mark them so. A run whose three
targets share one conductivity profile hands the same device pointer three times
(``Fields.set_d_conductivity`` installs ONE volume and ``condfac_for`` returns it for
every component), and the certified kernel gets away with the promise only because a
mask of three identical read-only pointers is never written through. This family binds
strictly less than the certified one promises, which is sound in that direction and not
the other; :func:`assert_disjoint_bindings` checks them against the restrict group.

THE INVERSE-EPSILON POINTERS ARE NOT ``__restrict__`` either, lifted from the certified
``update_E`` rather than decided here: an isotropic run hands the same pointer three
times (fields.py:1321-1326).

=============================================================================
DISJOINT FROM THE REAL TWIN BY ONE MEASURED CLAUSE
=============================================================================

``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so an overlap
with :mod:`.fused_electric_pair` would COST the 79 rows that one serves rather than
adding one. The two partition on the CONDUCTIVITY: ``coverage.covers_real_pml_curl``
refuses a target carrying a sigma by name ("Dx carries a conductivity: routes to the
three-history conductive-PML recurrence" -- the stamped census's own text on this row)
and ``conductive_kernels.covers_conductive_curl`` REQUIRES one. Nothing else is needed
and nothing here restates it -- both predicates are conjoined whole. Against the
dispersive electric pair it is disjoint on ``fields.polarizations``, which that
family's constitutive half requires and this one's refuses.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import deposit_repair as _deposit_repair
from . import compile_cache
from . import fused_electric_pair as real

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

try:  # pragma: no cover
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:  # pragma: no cover
    from . import conductive_kernels
except Exception:  # pragma: no cover
    conductive_kernels = None  # type: ignore

from . import own_cell_hoist

try:  # pragma: no cover
    from . import constitutive_kernels
except Exception:  # pragma: no cover
    constitutive_kernels = None  # type: ignore

try:  # pragma: no cover
    from . import step_curl_kernels
except Exception:  # pragma: no cover
    step_curl_kernels = None  # type: ignore

from .in_seam_coverage import zero_metal_axes

#: RELEASED 2026-09-02 under BOTH float32 subnormal policies. What moved this kernel
#: across the partition is the RUN, not an argument:
#: ``gate_cuda_conductive_fused_electric_pair.py`` on one RTX A6000, 8 of 8 cases
#: bit-identical over 60 COMPLETE driver steps, every stored volume compared as uint32
#: words INCLUDING both curl histories (``fu_D`` and ``f_cond_D``), 48 of 62 mutation
#: legs caught with three confirmed nulls and NONE unarmed, the deposit legs
#: byte-identical with the unbracketed control diverging, and the RESCALE leg
#: measuring on the device what the off-device probe measured on the host: the
#: shipped condinv injection leaves every non-deposit word untouched, and the retired
#: whole-volume control diverges in 232 words that classify wholly as SIGNED-ZERO
#: (``other`` at zero). Recorded as
#: ``cuda_conductive_fused_electric_pair_2026-09-02``.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "conductive_fused_electric_pair_pml_real",
)

#: EMPTY since 2026-09-02; what emptied it was the run above.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE. The corpus
#: row declares an electric deposit inside the seam (``source_field_types`` includes
#: ``"D"``), so without the bracket this module would serve ZERO -- which is why
#: ``CARRIES_DEPOSIT_REPAIR`` is asserted as a per-cell MEASUREMENT off the census in
#: the host suite rather than read back as a constant.
CARRIES_DEPOSIT_REPAIR = True

#: WHICH repair the two slots install. THE DEFAULT, and spelled anyway so a reader can
#: see it was decided: the layer is ACTIVE on every configuration this family admits
#: (``covers_real_pml_constitutive`` requires it), so ``update_E`` runs the split-field
#: dsigw accumulation and the repair that inverts it is the split-field one.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.SPLIT_FIELD_PATH,)

FAMILY = "cuda_conductive_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once. FIXED ACROSS MASKS -- the conductive
#: triple varies the SOURCE, not the name.
KERNEL_NAME = "conductive_fused_electric_pair_pml_real"

#: The driver passes ONE launch performs, in driver order (driver.py:3306, :3310,
#: :3313). The two symmetry fills are absent because this family REFUSES a mirror
#: plane and they are inert on every row it admits.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The conductive masks this family emits for a digest, and the ones its gate sweeps.
#: ``(False, False, False)`` IS ABSENT DELIBERATELY -- see the module docstring: that
#: run is refused by ``covers_conductive_curl`` and belongs to the real twin, so
#: emitting it would be a body no admitted configuration can reach. ``(T, T, T)`` is
#: the corpus row's own mask.
COND_MASKS_SWEPT: Tuple[Tuple[bool, bool, bool], ...] = (
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
)

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"what": "the certified curl helpers pml_apply and cond_pml_apply",
     "from": "void pml_apply(...) { ... f[idx] = ...; }",
     "to": "float pml_apply_reg(...) { ... float value = ...; return value; }",
     "why": "the stepped displacement is CARRIED in a register instead of being "
            "written and read back. THE AUXILIARY STORES STAY: fu and f_cond are the "
            "curl's own histories, zero_metal_D does not touch them, and update_E "
            "does not read them"},
    {"what": "the curl's per-target tail",
     "from": "pml_apply(Dx, fu_Dx, idx, curl, ...);  /  cond_pml_apply(Dx, ...);",
     "to": "d_x = pml_apply_reg(...);  /  d_x = cond_pml_apply_reg(...);",
     "why": "both branches of the certified #if are captured; capturing one would "
            "leave the other writing to global memory where the wall clear below "
            "could not reach it"},
    {"what": "the constitutive half's index decomposition and bounds guard",
     "from": "int idx = ...; if (idx >= nx * ny * nz) return; int k = ...; ...",
     "to": "(dropped)",
     "why": "the curl half above already declares idx, i, j and k and its guard has "
            "already returned; a second decomposition would redeclare them"},
    {"what": "the constitutive half's flux-density read",
     "from": "float src_x = Dx[idx] * inv_eps_Ex[idx];",
     "to": "float src_x = d_x * inv_eps_Ex[idx];",
     "why": "THE SEAM. A float32 stored to global and reloaded is the identity on the "
            "bits, so the substitution is exact rather than close; it is also what "
            "removes this half's only use of Dx/Dy/Dz and lets D be bound once"},
    {"what": "the constitutive half's kms_* coefficient names",
     "from": "constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);",
     "to": "... kps_x[i], kms_half_x[i]);",
     "why": "kms_x is the INTEGER sigma sub-lattice in the curl body and the "
            "HALF-INTEGER one here (stepping.py:1015); on the bare name the two "
            "parameters would collide and one launch would bind a half-cell error"},
    {"what": "zero_metal_D",
     "from": "(a separate pass, in_seam_passes.zero_metal_D at driver.py:3310)",
     "to": "an inline clear of the register and the store",
     "why": "the pass sits between the two halves and update_E reads what it wrote; "
            "the corpus row is METALLIC on x AND y, so declining it would cost the "
            "whole cell"},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "COND_MASKS_SWEPT", "FAMILY",
    "KERNEL_NAME", "LIFT_EDITS", "REPAIR_PATHS", "REPLACES", "SLOT",
    "UNCERTIFIED_KERNELS", "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "conductive_bindings",
    "conductive_fused_electric_pair_prelude",
    "conductive_fused_electric_pair_signature",
    "conductive_fused_electric_pair_source",
    "covers_conductive_fused_electric_pair", "device_sources",
    "launch_conductive_fused_electric_pair", "run_conductive_fused_electric_pair",
    "zero_metal_carry",
]

# =============================================================================
# THE DEVICE TEXT
# =============================================================================

_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified conductive template's name for this sub-step's PML kernel.
_CURL_TEMPLATE = "step_D_pml_conductive"

_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")

#: ``stepping._zero_metal``'s component table -- the OFF-DIAGONAL, two D components per
#: walled axis. See :mod:`.no_pml_dispersive_fused_electric_pair`'s copy for the Yee
#: derivation; it is restated rather than imported so neither family's carry can be
#: changed by an edit aimed at the other.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("y", "z")),
    ("wall_y", "j", ("x", "z")),
    ("wall_z", "k", ("x", "y")),
)

#: The two certified curl helpers and the edits that turn each into a value the seam
#: can carry. THE AUXILIARY STORES ARE UNTOUCHED in both: ``fu`` and ``f_cond`` are the
#: curl's own histories, ``zero_metal_D`` does not write them (it writes D alone,
#: stepping.py:2206-2247) and ``update_E`` does not read them, so they stay exactly
#: where the certified helper puts them.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
_PML_APPLY_STORE = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
_PML_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its parenthesisation\n"
    "    // and every auxiliary store are the certified ones; the displacement is\n"
    "    // additionally NAMED and RETURNED so the weld can clear it on a metallic\n"
    "    // wall before storing it, which is where driver.py:3310 puts zero_metal_D.\n"
    "    return (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n")

_COND_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void cond_pml_apply(\n"
_COND_PML_APPLY_SIGNATURE_REG = (
    "__device__ __forceinline__ float cond_pml_apply_reg(\n")
_COND_PML_APPLY_STORE = (
    "    f[idx] = dsigu ? f_split : (dsig ? f_first : f_direct);\n")
_COND_PML_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER, and it is the SELECTION that is returned\n"
    "    // rather than stored -- the three-way case choice, its predicates and both\n"
    "    // history stores below are the certified ones, untouched.\n"
    "    float value = dsigu ? f_split : (dsig ? f_first : f_direct);\n")
_COND_PML_APPLY_TAIL = "    if (dsig) c[idx] = c_new;\n"
_COND_PML_APPLY_TAIL_REG = (
    "    if (dsig) c[idx] = c_new;\n"
    "    return value;\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature: no arithmetic lives here.
_SIGNATURE = r'''
extern "C" __global__ void __NAME__(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the constitutive
    // half reads it from a register; binding it a second time as a source would be
    // two pointers to one allocation with a __restrict__ promise on one of them,
    // which is UB and which NVRTC miscompiles without a diagnostic.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    // The curl's split-field auxiliary. Written by the certified helper and read by
    // nothing else in this launch.
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    // H, the curl's operands. Under an active layer H is STORED
    // (enable_pml_storage) and the certified D kernel differences exactly these.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // NOT __restrict__, and a DELIBERATE DEPARTURE from the certified curl, which
    // marks them so: Fields.set_d_conductivity installs ONE volume and condfac_for
    // returns it for every component, so a run hands the same pointer three times.
    // Binding strictly less than the certified kernel promises is sound in that
    // direction and not the other.
    const float* cf0, const float* cf1, const float* cf2,
    const float* ci0, const float* ci1, const float* ci2,
    // The conductive recurrence's THIRD history, and what makes this curl arm a
    // family of its own (Fields._ensure_conductive_pml_storage).
    float* __restrict__ fc0, float* __restrict__ fc1, float* __restrict__ fc2,
    // E and its dsigw auxiliary, the constitutive half's targets.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__, lifted from the certified update_E rather than decided here:
    // an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326).
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    int nx, int ny, int nz, float dtdx,
    // THE CURL'S COEFFICIENTS, on the INTEGER sigma sub-lattice (step_D, half_integer
    // False).
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // THE CONSTITUTIVE'S COEFFICIENTS, on the HALF-INTEGER one (stepping.py:986).
    // kms is RENAMED because the bare name collides with the curl's above and the
    // two are different sub-lattices -- a half-cell error, not a crash.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    int bc_x, int bc_y, int bc_z,
    // zero_metal_D's three per-axis flags (driver.py:3310). A FOLDED metallic axis
    // carries no wall (stepping._zero_metal, :2237-2239) and this family refuses a
    // fold outright, so the two can never disagree here.
    int wall_x, int wall_y, int wall_z
) {
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 986->1015


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable."""
    missing = [name for name, module in (
        ("conductive_kernels", conductive_kernels),
        ("constitutive_kernels", constitutive_kernels)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host (they "
            f"import CuPy at module scope), so there is no certified text to splice. "
            f"covers_conductive_fused_electric_pair needs neither and still answers")


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body."""
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Substitute exactly one occurrence, or fail by name."""
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"{what}: the anchor {old!r} appears {count} times in the certified text, "
            f"not once; this family LIFTS that line rather than retyping it and "
            f"cannot splice around its absence")
    return text.replace(old, new, 1)


def _code_lines(body: str) -> List[str]:
    """``body`` without its comment lines, for checks that must read what compiles."""
    return [line for line in body.splitlines()
            if not line.lstrip().startswith("//")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line starting "
            f"{prefix!r}); this family LIFTS that line rather than retyping it")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def conductive_fused_electric_pair_prelude() -> str:
    """Both certified preludes, with the two curl helpers turned into values.

    ``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE`` is emitted UNTOUCHED --
    ``constitutive_apply`` needs no edit at all, because the seam changes where its
    ``src`` argument comes from and not what it does with it.
    """
    _certified_halves()
    template = conductive_kernels.kernel_template(_CURL_TEMPLATE)
    prelude = template.split('extern "C" __global__ void', 1)[0]
    for signature, store in ((_PML_APPLY_SIGNATURE, _PML_APPLY_STORE),
                             (_COND_PML_APPLY_SIGNATURE, _COND_PML_APPLY_STORE)):
        if prelude.count(signature) != 1:
            raise AssertionError(
                f"the certified curl prelude declares {signature.strip()!r} "
                f"{prelude.count(signature)} times, not once; the value rewrite has "
                f"no anchor")
        if prelude.count(store) != 1:
            raise AssertionError(
                f"the certified helper no longer closes with the store this module "
                f"turns into a named value ({store.strip()!r}); the arithmetic may "
                f"have moved")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    prelude = prelude.replace(_COND_PML_APPLY_SIGNATURE,
                              _COND_PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_COND_PML_APPLY_STORE, _COND_PML_APPLY_STORE_REG, 1)
    prelude = _replace_once(prelude, _COND_PML_APPLY_TAIL, _COND_PML_APPLY_TAIL_REG,
                            "cond_pml_apply's return")
    # THE HISTORY STORES SURVIVE, and that is checked rather than trusted: fu and
    # f_cond are the curl's own state and a rewrite that swallowed them would leave
    # the recurrence one step behind on every cell, which no seam test would notice.
    for store in ("    fu[idx] = fu_new;\n", "    if (dsigu) u[idx] = u_new;\n",
                  "    if (dsig) c[idx] = c_new;\n"):
        if store not in prelude:
            raise AssertionError(
                f"the auxiliary store {store.strip()!r} did not survive the value "
                f"rewrite; the curl's histories would stop advancing")
    return prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE


def conductive_fused_electric_pair_signature() -> str:
    """The fused signature."""
    source = _SIGNATURE.replace("__NAME__", KERNEL_NAME)
    if "__" in source.replace("__restrict__", "").replace("__global__", ""):
        raise AssertionError("an unsubstituted placeholder survived the signature")
    return source


def certified_curl_body() -> str:
    """``step_D_pml_conductive``'s body, with the three registers captured."""
    _certified_halves()
    template = conductive_kernels.kernel_template(_CURL_TEMPLATE)
    prelude = template.split('extern "C" __global__ void', 1)[0]
    body = _split_body(template, prelude, f"kernel_template({_CURL_TEMPLATE!r})")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the carried "
            "registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each certified\n"
        "    // component below is a braced scope, and a value declared inside one\n"
        "    // does not outlive it.\n"
        "    float d_x = 0.0f;\n"
        "    float d_y = 0.0f;\n"
        "    float d_z = 0.0f;\n"
        "\n",
        tail,
    ))
    # BOTH BRANCHES OF THE CERTIFIED #if ARE REWRITTEN, matched as WHOLE LINES so a
    # moved argument list is a named failure rather than a partial substitution.
    for index, (axis, target) in enumerate(zip(_CARRIED, _TARGETS)):
        for helper in ("cond_pml_apply", "pml_apply"):
            prefix = (f"        {helper}({target}, fu_{target}, fc{index}, "
                      if helper == "cond_pml_apply"
                      else f"        {helper}({target}, fu_{target}, idx, ")
            call = _line_starting(body, prefix, f"target {target}'s {helper} tail")
            if not call.endswith(");"):
                raise AssertionError(
                    f"the certified curl body no longer closes target {target}'s "
                    f"{helper} on one line ({call!r})")
            # THE ARGUMENT LIST IS CARRIED WHOLE. Only the callee's NAME changes and a
            # destination is prepended -- every operand, its order and the coefficient
            # indices are the certified ones. Slicing the arguments off by an offset
            # would silently drop the leading ones on any call whose spelling moved.
            captured = (f"        d_{axis} = "
                        + call.strip().replace(f"{helper}(", f"{helper}_reg(", 1))
            body = _replace_once(body, call + "\n", captured + "\n",
                                 f"target {target}'s {helper} tail")
    code = "\n".join(_code_lines(body))
    for helper in ("pml_apply(", "cond_pml_apply("):
        if helper in code.replace(f"{helper[:-1]}_reg(", ""):
            raise AssertionError(
                f"a call to {helper[:-1]} survived the capture rewrite; its value "
                f"would be written to global memory and never read into the seam")
    for target in _TARGETS:
        if f"{target}[idx] = " in code:
            raise AssertionError(
                f"a store into {target} survived the capture rewrite; the wall clear "
                f"below writes the register and this store would race it")
    stencil = "        float curl = dtdx * ((sf - f1) + (f2 - ss));"
    if body.count(stencil) != 3:
        raise AssertionError(
            f"the certified curl body carries {body.count(stencil)} of the three curl "
            f"stencil lines; this family exists to reuse them unchanged")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    The register is cleared BESIDE the store, because the constitutive half reads the
    register and not the volume. NO OWNERSHIP GUARD, and that is a consequence of the
    fold refusal rather than a simplification: this family refuses a mirror plane
    outright, so no cell is imaged and every thread owns its own.

    ``fu`` AND ``f_cond`` ARE NOT CLEARED, and that is the array path's own behaviour:
    ``stepping._zero_metal`` writes the D components alone (:2206-2247). A carry that
    cleared the histories too would damp the absorber differently at every wall cell.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: two D components are wiped per walled\n"
        "    // axis. Stored cell 0 is the LOW wall; the high wall is the zero ghost\n"
        "    // shift_dn already supplies. The curl's fu and f_cond histories are\n"
        "    // NOT touched -- the array path's own pass writes D alone.\n"
    ]
    for flag, coordinate, axes in _ZERO_METAL_ROWS:
        sets = " ".join(f"d_{axis} = 0.0f;" for axis in axes)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {sets} }}\n")
    lines.append(
        "\n    // THE STORE, after the clear and before the constitutive read, which\n"
        "    // is exactly where driver.py:3310 puts zero_metal_D relative to :3313.\n")
    for axis, target in zip(_CARRIED, _TARGETS):
        lines.append(f"    {target}[idx] = d_{axis};\n")
    return "".join(lines)


def certified_constitutive_body() -> str:
    """``update_E_pml_real``'s body, lifted, reading the registers.

    Drops the second index decomposition and turns the three flux-density reads into
    the seam. The ``kms_*`` rename is this family's, for the collision the signature
    names: the curl binds the INTEGER sub-lattice under that name and this half needs
    the HALF-INTEGER one.
    """
    _certified_halves()
    body = _split_body(own_cell_hoist.unhoisted_kernel_code(
                           constitutive_kernels._update_E_pml_real_kernel_code,
                           "update_E_pml_real", "_update_E_pml_real_kernel_code"),
                       constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,
                       "_update_E_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own line; the "
            "duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]

    # THE SEAM. ONE TOKEN PER LINE is substituted and the rest of the certified
    # statement -- the operand order, the inv_eps index and the line's own annotation
    # -- is left exactly as it stands.
    for axis, target in zip(_CARRIED, _TARGETS):
        anchor = f"    float src_{axis} = {target}[idx] * inv_eps_E{axis}[idx];"
        line = _line_starting(tail, anchor, f"{target}'s inverse-permittivity product")
        if line.count(f"{target}[idx]") != 1:
            raise AssertionError(
                f"{target}[idx] appears {line.count(f'{target}[idx]')} times in "
                f"{line!r}; the seam substitution has no unambiguous token")
        tail = tail.replace(line, line.replace(f"{target}[idx]", f"d_{axis}", 1), 1)
    first = _line_starting(tail, "    float src_x = d_x * inv_eps_Ex[idx];",
                           "the rewritten first inverse-permittivity product")
    tail = tail.replace(
        first,
        "    // THE SEAM. The certified body reads D[idx] here -- the very word the\n"
        "    // curl half stored a few lines above -- and this reads the register\n"
        "    // instead. A float32 stored to global and reloaded is the identity on\n"
        "    // the bits, so the substitution is exact rather than close; it is also\n"
        "    // what removes this half's only use of Dx/Dy/Dz, which is what lets D\n"
        "    // be bound exactly once.\n" + first, 1)
    for target in _TARGETS:
        if f"{target}[idx]" in tail:
            raise AssertionError(
                f"a read of {target} survived the seam rewrite; the constitutive half "
                f"would need D bound a second time, which is the aliasing hazard this "
                f"signature exists to avoid")

    # THE ONE RENAME, on one line each.
    for axis, coordinate in zip(_CARRIED, _COORDINATE):
        prefix = f"    constitutive_apply(E{axis}, f_w_E{axis}, idx, src_{axis}, "
        call = _line_starting(tail, prefix, f"E{axis}'s constitutive_apply")
        expected = f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);"
        if call != expected:
            raise AssertionError(
                f"E{axis}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an "
                f"argument list this module has not read")
        tail = tail.replace(call, call.replace(f"kms_{axis}[", f"kms_half_{axis}[", 1),
                            1)
    for axis, coordinate in zip(_CARRIED, _COORDINATE):
        if f"kms_{axis}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER "
                f"vector in the curl body and the HALF-INTEGER one here, and the two "
                f"would collide on the bare name")
    return tail


def conductive_fused_electric_pair_source(cond: Sequence[bool]) -> str:
    """The whole fused kernel at one conductive mask.

    THE THREE ``#define`` LINES ARE THIS FILE'S ONLY ADDITION TO THE MASK, exactly as
    ``conductive_kernels.kernel_source`` spells it, and for that function's own reason:
    the conductive tail helper lives INSIDE the emitted text behind
    ``#if COND0 || COND1 || COND2``, so a gate that mutates the emitted source reaches
    it.
    """
    mask = tuple(bool(flag) for flag in cond)
    if len(mask) != 3:
        raise ValueError(
            f"cond must be a triple of per-target conductivity flags, got {cond!r}; "
            f"it is conductive_kernels.conductive_targets' own answer and a shorter "
            f"tuple would leave a #define unset")
    if not any(mask):
        raise ValueError(
            "the fully lossless mask (False, False, False) is not a body this family "
            "emits: covers_conductive_curl refuses a run where no target of step_D "
            "carries a sigma BY NAME, and that run belongs to fused_electric_pair. "
            "Emitting it would be a body no admitted configuration can reach")
    defines = "".join(f"#define COND{index} {int(flag)}\n"
                      for index, flag in enumerate(mask))
    source = "".join((
        defines,
        conductive_fused_electric_pair_prelude(),
        conductive_fused_electric_pair_signature(),
        certified_curl_body(),
        zero_metal_carry(),
        certified_constitutive_body(),
        "}\n",
    ))
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by mask -- for a digest."""
    return {"".join("1" if flag else "0" for flag in mask):
            conductive_fused_electric_pair_source(mask)
            for mask in COND_MASKS_SWEPT}


_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear()


def _get_kernel(cond: Sequence[bool], name: str = KERNEL_NAME):
    """Compile (or fetch) the kernel at one mask.

    ``compile_cache.kernel_cache_key`` keys the memo ON THE SOURCE, so a gate that
    mutates the emitted text is a cache MISS and reaches NVRTC.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = conductive_fused_electric_pair_source(cond)
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def _by_path(module_name: str, attribute: str):
    """One predicate, importable without a device."""
    import importlib.util as _importlib_util  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = _importlib_util.spec_from_file_location(
        f"cuda_kernels_{module_name}_predicate",
        _os.path.join(here, f"{module_name}.py"))
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def _conductive_predicate():
    if conductive_kernels is not None:
        return conductive_kernels.covers_conductive_curl
    return _by_path("conductive_kernels", "covers_conductive_curl")


def _constitutive_predicate():
    from . import coverage as _coverage  # noqa: PLC0415 - laptop-importable
    return _coverage.covers_real_pml_constitutive


def covers_conductive_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                          sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. The curl half is
    ``conductive_kernels.covers_conductive_curl``, whole -- which REQUIRES a target
    carrying a sigma and so partitions this family from :mod:`.fused_electric_pair`,
    whose curl predicate refuses one by name. The constitutive half is
    ``coverage.covers_real_pml_constitutive(side='E')``, whole -- which refuses
    ``fields.polarizations`` truthy and so partitions it from the dispersive pair.

    THE SEAM CLAUSES ARE THIS PREDICATE'S OWN: the fold refusal, the wall/fold
    disjointness, the whole-volume-rescale fallback and the in-seam source routing.
    """
    covered, reason = _conductive_predicate()(fields, pml, grid, "step_D")
    if not covered:
        return False, f"conductive curl half: {reason}"
    covered, reason = _constitutive_predicate()(fields, pml, grid, side="E")
    if not covered:
        return False, f"ordinary constitutive half: {reason}"

    # THE TWO SYMMETRY PASSES. Both return at their first line without a mirror
    # (stepping.py:1481-1482, :1565-1566). RESTATED HERE rather than inferred from
    # either half's guard, as on both siblings.
    if bool(getattr(grid, "has_symmetry", bool)()):
        return False, ("a mirror plane is active: stepping.fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam and this weld carries neither")
    for axis in range(3):
        if bool(getattr(grid, "is_mirrored", lambda _a: False)(axis)):
            return False, (f"axis {axis} is folded by a mirror plane: the two symmetry "
                           f"passes image ghost cells inside this seam and this weld "
                           f"carries neither")

    try:
        walls = zero_metal_axes(grid)
    except Exception as error:
        return False, (f"the metallic-wall axes cannot be read from the grid "
                       f"({type(error).__name__}: {error}), and this weld carries "
                       f"zero_metal_D inline")
    for axis in range(3):
        if walls[axis] and bool(getattr(grid, "is_mirrored", lambda _a: False)(axis)):
            return False, (f"axis {axis} is reported BOTH walled and folded; "
                           f"stepping._zero_metal carries no wall on a folded axis "
                           f"(:2237-2239) and this weld would clear a plane the array "
                           f"path leaves alone")

    # THE DRIVER'S WHOLE-VOLUME FALLBACK, refused by name. See the module docstring:
    # the sparse replay is what made this cell servable, and a scaled source that
    # publishes NO deposit table still takes the three whole-volume passes, which
    # rewrite -0.0 to +0.0 at every cell of the component and not only at the deposit.
    # A point repair cannot reconstruct that. NO IN-TREE ELECTRIC SOURCE CLASS IS ONE;
    # the clause stays for the duck-typed stranger.
    for index, source in enumerate(_deposit_repair.in_seam_sources(
            tuple(sources) if sources is not None else (), "D")):
        if (not getattr(source, "is_integrated", False)
                and not hasattr(source, "_point_ix")):
            component = getattr(source, "component", None)
            return False, (
                f"source {index} ({type(source).__name__}) is a NON-INTEGRATED "
                f"electric source on a conductive run and publishes NO deposit table "
                f"(no _point_ix): the driver's fallback applies its condinv scaling as "
                f"three WHOLE-VOLUME passes "
                f"(_inject_electric_through_conductivity's dense branch), which "
                f"rewrites -0.0 to +0.0 at every cell of "
                f"D{component[1] if component else '*'} and not only at the deposit, "
                f"and a point repair cannot reconstruct that")

    # THE WORDING IS THIS SITE'S OWN, which is what ``seam_source_reasons`` asks for:
    # the shared helper decides WHICH sources count as in-seam and whether a deposit
    # is carryable, and each product keeps the text its own gate and tests pin.
    reasons = _deposit_repair.seam_source_reasons(
        fields, sources, "D", pml=pml, carries_repair=CARRIES_DEPOSIT_REPAIR,
        repair_paths=REPAIR_PATHS,
        undeclared=("the source list was not declared: this weld spans the driver's "
                    "electric injection (driver.py:3306-3313) and Fields does not "
                    "hold the sources, so an undeclared set is a refusal rather than "
                    "an empty seam"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) deposits into D inside this "
            f"seam and this weld does not carry it"))
    if reasons:
        return False, reasons[0]
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz")
_HISTORY_BINDINGS: Tuple[str, ...] = ("f_cond_Dx", "f_cond_Dy", "f_cond_Dz")
_ELECTRIC_BINDINGS: Tuple[str, ...] = (
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The curl's coefficient views, in signature order -- the INTEGER sub-lattice.
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
#: The constitutive's, on the HALF-INTEGER one.
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def conductive_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, built once per frozen configuration.

    THE TWO SUB-LATTICES ARE THE POINT. ``step_D`` reads the INTEGER one
    (``half_integer=False``) and ``update_E`` the HALF-INTEGER one (stepping.py:1015);
    binding either group where the other belongs is a half-cell error in the absorber
    profile, not a crash. The real twin's tables are reused whole -- its constitutive
    group is the same certified ``update_E``'s.
    """
    return real.fused_electric_pair_tables(pml)


def conductive_bindings(fields: "Fields") -> Tuple[Tuple[Any, ...], Tuple[Any, ...],
                                                   Tuple[bool, bool, bool]]:
    """``(condfac + condinv, the three f_cond histories, the COND mask)``.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET ARRAY in all three slots, which is
    ``conductive_kernels.step_conductive_curl``'s own convention: ``#if CONDn`` is 0
    for it, so those pointers are never dereferenced.
    """
    if conductive_kernels is None:
        raise RuntimeError(
            "conductive_kernels is not importable on this host, so the per-component "
            "conductivity mask cannot be read from the family that defines it")
    mask = conductive_kernels.conductive_targets(fields, "step_D")
    targets = tuple(getattr(fields, name) for name in _TARGETS)
    condfac = tuple(fields.condfac_for(name) if flag else target
                    for name, flag, target in zip(_TARGETS, mask, targets))
    condinv = tuple(fields.condinv_for(name) if flag else target
                    for name, flag, target in zip(_TARGETS, mask, targets))
    histories = tuple(getattr(fields, name) if flag else target
                      for name, flag, target in zip(_HISTORY_BINDINGS, mask, targets))
    return (tuple(condfac) + tuple(condinv), histories,
            tuple(bool(flag) for flag in mask))


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             conductivity: Sequence[Any],
                             histories: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in this signature makes.

    THE RESTRICT GROUP IS: the nine curl volumes, the three ``f_cond`` histories, the
    six electric volumes and both coefficient groups. The inverse-epsilon and
    conductivity pointers are NOT restrict and may legitimately repeat, so they are
    checked only AGAINST the restrict group.

    A history slot holding a TARGET array is this launcher's placeholder convention for
    a lossless component and is expected; it is therefore checked against the restrict
    group only when the mask says it is a real history.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []
    for name in _FIELD_BINDINGS + _ELECTRIC_BINDINGS:
        pointer = int(getattr(fields, name).data.ptr)
        if pointer in bound:
            collisions.append(f"{name} and {bound[pointer]} are the same allocation")
            continue
        bound[pointer] = name
    for key in _CURL_TABLE_KEYS:
        bound[int(tables["curl"][key].data.ptr)] = f"curl:{key}"
    for key in _CONSTITUTIVE_TABLE_KEYS:
        bound[int(tables["constitutive"][key].data.ptr)] = f"constitutive:{key}"
    for index, array in enumerate(histories):
        pointer = int(array.data.ptr)
        existing = bound.get(pointer)
        if existing is not None and not existing.startswith("D"):
            collisions.append(f"f_cond {index} and {existing} are the same allocation")
            continue
        if existing is None:
            bound[pointer] = f"f_cond {index}"
    for label, group in (("inv_eps", real.inverse_epsilon_bindings(fields)),
                         ("conductivity", tuple(conductivity))):
        for index, volume in enumerate(group):
            existing = bound.get(int(volume.data.ptr))
            if existing is not None and existing.startswith("f_cond "):
                collisions.append(
                    f"{label} {index} and {existing} are the same allocation, and the "
                    f"second is bound __restrict__ and WRITTEN")
    if collisions:
        raise ValueError(
            "the conductive fused electric pair binds every field, history and table "
            "argument __restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_conductive_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int], dtdx: float,
        conductivity: Sequence[Any], histories: Sequence[Any],
        cond: Sequence[bool], kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch, at this run's mask.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is.
    """
    if len(tuple(conductivity)) != 6 or len(tuple(histories)) != 3:
        raise ValueError(
            f"{len(tuple(conductivity))} conductivity and {len(tuple(histories))} "
            f"history pointers were resolved, not 6 and 3; the signature binds condfac "
            f"and condinv per target plus one f_cond each, and a mismatch would shift "
            f"every argument after the group")
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + tuple(
        conductivity
    ) + tuple(histories) + tuple(
        getattr(fields, name) for name in _ELECTRIC_BINDINGS
    ) + tuple(real.inverse_epsilon_bindings(fields)) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3))
    (kernel or _get_kernel(cond))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "cond": tuple(bool(flag) for flag in cond),
            "walls": tuple(int(bool(walls[axis])) for axis in range(3))}


def run_conductive_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path.
    """
    covered, reason = covers_conductive_fused_electric_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = conductive_fused_electric_pair_tables(pml)
    conductivity, histories, cond = conductive_bindings(fields)
    assert_disjoint_bindings(fields, tables, conductivity, histories)
    return launch_conductive_fused_electric_pair(
        fields, tables, step_curl_kernels.real_curl_boundary_codes(grid),
        zero_metal_axes(grid), dtdx, conductivity, histories, cond, kernel)

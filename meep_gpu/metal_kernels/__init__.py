"""Hand-written Metal kernels for the real-field PML step path (Apple GPUs, MPS).

The Metal counterpart of :mod:`meep_gpu.triton_kernels`, and a deliberate port of
its DESIGN rather than of its code: the coverage predicates, the plan objects, the
composition and seam rules, the fail-closed arm table, the mutation harness and the
gate's vacuity discipline all transfer, because none of them is about a backend.
Only the launch layer and the kernel bodies are Metal-specific.

WHAT IS HERE, in the first tranche:

* :mod:`.shaders` — the two kernel sources (real-field PML curl, ``dsigw``
  constitutive) and the specialisation substitution that replaces Triton's
  ``tl.constexpr`` arguments;
* :mod:`.coverage` — which configurations either kernel may step, re-using every
  backend-free leaf helper and table from the Triton predicate module;
* :mod:`.launch` — plan objects, builders, the residency mirror layer and a
  ``plan_step`` arm table carrying the PML curl, the constitutive pair and the
  no-kernel null family;
* :mod:`.subnormal` — the MPS executor column: flush is NATIVE and
  UNCONTROLLABLE here and keep is a REFUSAL.

DISPATCH IS NOT WIRED FOR THIS PACKAGE, and since 2026-09-02 that is no longer the
same sentence as "dispatch is not wired". ``meep_gpu.fastpath.plan_fast_path``
returns a ``FastPathPlan``, the driver consults it at seven slots, and the Triton
track dispatches nine certified families plus five fused products inside an
opted-in run. It reaches NONE of this package: the function imports
``triton_kernels`` and composes its ``plan_step``, which is a structural fact rather
than a coverage verdict and is re-measured on every board cut
(``parity/meep_gpu/dispatch_reachability``). Which track earns the step path on
Apple hardware is Phase 3 of the dispatch-expansion plan — a residency-aware seam —
and a benchmark informs it; a port does not make it.

NO TORCH AT MODULE LEVEL, here or in any submodule that a predicate reaches.
:mod:`.coverage` and :mod:`.shaders` are importable, readable and testable on a
host with no GPU and no optional dependency; only :mod:`.launch`'s launch path
imports torch, and it does so inside the functions that need it.

THE RESIDENCY LAYER IS NEW AND HAS NO TRITON COUNTERPART. The CuPy path is zero
copy — ``CupyPointer`` renames an existing device allocation and the kernel writes
the engine's own array in place. On this host the engine holds NumPy, and a torch
MPS tensor is a separate allocation, so a plan owns persistent device MIRRORS of
the volumes it steps. That buys back the round trip (measured 12x-64x the kernel
time) and creates one invariant this package must check rather than comment on: a
mirror is valid only while nothing else writes the host array. See
:func:`.coverage.residency_reasons`.
"""

from __future__ import annotations

# EVERY SUBMODULE, because this list is the lazy loader's whole namespace: a family
# absent from it raises AttributeError on `metal_kernels.<family>` while importing
# fine as `from meep_gpu.metal_kernels import <family>`, which is a difference no
# reader expects and which three families had acquired.
__all__ = [
    "ade_update_p",
    "arms",
    "bfast_curl",
    "complex_conductive_pml",
    "complex_fields",
    "complex_folded_offdiag_update_e",
    "complex_fused_hd_pair",
    "complex_no_pml_curl",
    "complex_no_pml_conductive",
    "complex_no_pml_offdiag_update_e",
    "complex_dispersive_update_e",
    "complex_dispersive_spine",
    "complex_no_pml_stored_e",
    "conductive_pml",
    "coverage",
    "cylindrical_complex",
    "cylindrical_complex_fused_hd_pair",
    "cylindrical_complex_scan",
    "cylindrical_real",
    "device",
    "dispersive_update_e",
    "folded_beta",
    "folded_complex",
    "folded_dispersive_update_e",
    "folded_offdiag_dispersive_update_e",
    "folded_offdiag_update_e",
    "launch",
    "no_pml_curl",
    "no_pml_constitutive",
    "no_pml_conductive",
    "no_pml_stored_e",
    "offdiag_update_e",
    "plans",
    "preconditions",
    "registry",
    "shaders",
    "special_kz",
    "subnormal",
    "symmetry",
    "templates",
]


def __getattr__(name: str):
    """Import submodules lazily so ``import meep_gpu.metal_kernels`` needs no torch."""
    if name in __all__:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

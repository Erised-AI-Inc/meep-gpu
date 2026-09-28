"""The NO-ABSORBER three-slot weld: ``step_D`` -> ``update_E`` -> ``update_P``.

THE NO-PML TWIN of :mod:`.three_slot_dispersive_weld`, over the board's second
collision cell. Where that module welds the PML D->E product to the PML-ADE
recurrence for the SEVEN rows of ``D_to_E (cuda_curl/PML,
cuda_dispersive/dispersive)``, this one welds
:mod:`.no_pml_dispersive_fused_electric_pair` to the same recurrence for the THREE
rows of ``D_to_E (cuda_conductive/conductive, cuda_no_pml_dispersive/no-PML
dispersive store)`` and its extra cell ``D_to_E (cuda_no_pml_curl/no-PML curl,
...)``: ``absorber-1d.py``, ``TestAbsorber.test_absorber`` and
``material-dispersion.py``. Those three are EXACTLY the three
``cuda_no_pml_fused_polarization_pair`` serves at ``E_to_P``, which is the collision
this weld exists to resolve.

=============================================================================
WHY THIS IS A SECOND MODULE AND NOT A SECOND FAMILY IN THE FIRST
=============================================================================

ONE DECLARATION, AND IT IS THE ONE THAT CANNOT BE SHARED. :data:`REPAIR_PATHS` is
``deposit_repair.PLAIN_PATH`` here and ``SPLIT_FIELD_PATH`` there, because
``update_E`` with an INACTIVE layer is a pure overwrite (stepping.py:1019-1022) and the
split-field repair inverts a recurrence that does not run. ``fused_pairs``
``_repair_paths_of`` reads that declaration off the MODULE, so one module cannot
carry two answers -- and a wrong answer is a ``deposit_repair.repairable`` refusal BY
NAME rather than a mis-repair, measured 7 of 7 in the probe's ``wrong_repair_path``
row. It is the same reason :mod:`.no_pml_dispersive_fused_electric_pair` is a
separate module from :mod:`.dispersive_fused_electric_pair`.

EVERYTHING ELSE IS IMPORTED, NOT COPIED. The emitter, the launcher, the rotation and
the binding check all live in :mod:`.three_slot_dispersive_weld` and are re-exported
here, because with the ``update_E`` half removed from the fused kernel the device
text is ARM-INDEPENDENT: what is left is the certified ADE recurrence, and the
absorber reaches the polarization through the DRIVE and nowhere else
(stepping.py:1424). ``Fields.drive_field`` hands back ``f_w_<c>`` under an active
layer and the stored ``E<c>`` without one (fields.py:1140-1163); the launcher binds
whichever it is handed, and the ARM only names which of the two the predicate
requires.

=============================================================================
WHAT THE ARM CHANGES, IN FULL
=============================================================================

* the D->E half launched in group 1 --
  ``no_pml_dispersive_fused_electric_pair.launch_no_pml_dispersive_fused_electric_pair``,
  released as ``cuda_no_pml_dispersive_fused_electric_pair_2026-09-02``, whose
  ``REPLACES`` is THREE passes rather than five (this family refuses a mirror plane,
  so the two symmetry fills are inert on every row it admits);
* the repair path, above;
* the two predicates conjoined, both of which refuse an ACTIVE layer by name where
  the PML module's require one -- which is what keeps the two welds disjoint on a
  single boolean, exactly as their four halves already are.

Nothing else. :data:`KERNEL_NAME`, :data:`LIFT_EDITS` and the emitted device text are
:mod:`.three_slot_dispersive_weld`'s, and the digest that pins them is one digest.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` composes ``triton_kernels.plan_step`` and
reaches no hand-CUDA product. This module ships a predicate and two declarations;
``fused_pairs`` can plan it opt-in (``arms.plan_step(..., fuse=True)``).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# THE PML TWIN. Every piece of device text, every launcher and the rotation come from
# it rather than a copy: the kernel is arm-independent (module docstring), and two
# copies of the rotation is two places for the buffer sequence to be wrong.
try:
    from . import three_slot_dispersive_weld as shared
    from . import fused_polarization_pair as ep
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util

    def _load(stem):
        here = os.path.dirname(os.path.abspath(__file__))
        spec = _importlib_util.spec_from_file_location(
            f"cuda_kernels_{stem}", os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    shared = _load("three_slot_dispersive_weld")
    ep = _load("fused_polarization_pair")

try:
    from .. import deposit_repair as _deposit_repair
except ImportError:  # loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "meep_gpu_deposit_repair",
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "deposit_repair.py"))
    _deposit_repair = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_deposit_repair)


# =============================================================================
# THE PARTITION -- BOTH HALVES EMPTY, BY OWNERSHIP
# =============================================================================

#: BOTH HALVES EMPTY, AND THAT IS AN OWNERSHIP FACT RATHER THAN AN OMISSION. This
#: module writes NO device text: the kernel this arm launches is
#: ``three_slot_dispersive_weld``'s, emitted by that module's emitter, and one kernel
#: name is owned by exactly one family
#: (``test_kernel_partition.test_every_kernel_in_the_package_has_exactly_one_owning_
#: module``) so that ``certification.json`` can never be ambiguous about which body a
#: verdict saw. ``three_slot_polarization_real`` therefore sits in the PML twin's
#: :data:`~.three_slot_dispersive_weld.CERTIFIED_KERNELS`, where it has been since
#: 2026-09-03: ``parity/meep_gpu/gate_cuda_three_slot_weld.py`` cut its block
#: (``cuda_three_slot_dispersive_weld_2026-09-03``) as one gate run over BOTH arms,
#: because the emitted text is arm-independent (module docstring) and the arms
#: differ only in the bindings and the group-1 launch; this arm's 14 of 14 cases are
#: in that block beside the PML arm's 18 of 18.
CERTIFIED_KERNELS = ()
UNCERTIFIED_KERNELS = {}

#: TRUE, AND IT IS THE WHOLE PRODUCT. All three rows of this weld's cells carry an
#: ELECTRIC deposit inside the ``step_D`` -> ``update_E`` span, so without the bracket
#: it would serve ZERO. The flag and the wiring change together or not at all;
#: ``fused_pairs._install_fused_triple`` is what installs the bracket.
CARRIES_DEPOSIT_REPAIR = True

#: THE PLAIN REPAIR, and this is the declaration that makes this a separate module.
#: ``update_E`` under an inactive layer is a pure overwrite (stepping.py:1019-1022);
#: declaring the default split-field one would make ``deposit_repair.repairable``
#: refuse every configuration this product exists for. Inherited from the D->E half
#: this arm welds (``no_pml_dispersive_fused_electric_pair.REPAIR_PATHS``) rather than
#: re-decided.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.PLAIN_PATH,)

FAMILY = "cuda_three_slot_no_pml_dispersive_weld"

#: The PML twin's symbol, spelled once THERE. The device text is arm-independent.
KERNEL_NAME = shared.KERNEL_NAME

#: THE TWIN, and this is the declaration a gate must read before it plants a defect.
#: The emitted text, the compile memo and :data:`~.three_slot_dispersive_weld.
#: SOURCE_TRANSFORM` all live in that module and NOT here, because one kernel name is
#: owned by exactly one family (:data:`CERTIFIED_KERNELS` above). Assigning a source
#: transform to THIS module would set an attribute no compile path reads: the launch
#: still goes through the twin's ``_get_kernel``, the mutated body would never be
#: emitted, and the gate would score a defect as applied-and-null. Naming the owner
#: makes that a lookup instead of a thing the gate has to know.
KERNEL_OWNER_MODULE = shared.KERNEL_OWNER_MODULE

#: The driver passes this product performs, in driver order (driver.py:3316, :3327,
#: :3332, :3334). FOUR rather than the PML twin's six: the two symmetry fills are
#: absent because the D->E half REFUSES a mirror plane and they are inert on every
#: row it admits -- this family's ``REPLACES`` is its own D->E half's three passes
#: plus the polarization advance, not a shortened copy of the other arm's.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E", "update_P")

#: The slot the composer holds this on (the span's first).
SLOT = "step_D"

#: ALL THREE slots this product owns, in driver order.
SLOTS: Tuple[str, str, str] = ("step_D", "update_E", "update_P")

#: The PML twin's, because the emitted text is the same text.
LIFT_EDITS = shared.LIFT_EDITS

ELECTRIC_TERMS = shared.ELECTRIC_TERMS

#: The emitter and the launch, re-exported rather than reimplemented.
three_slot_polarization_source = shared.three_slot_polarization_source
device_sources = shared.device_sources
component_specs = shared.component_specs
assert_disjoint_bindings = shared.assert_disjoint_bindings
launch_three_slot_polarization_component = (
    shared.launch_three_slot_polarization_component)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "KERNEL_OWNER_MODULE", "LIFT_EDITS", "REPAIR_PATHS", "REPLACES", "SLOT",
    "SLOTS", "UNCERTIFIED_KERNELS", "assert_disjoint_bindings", "component_specs",
    "covers_no_pml_three_slot_dispersive_weld", "device_sources",
    "launch_three_slot_polarization_component", "run_three_slot_polarization",
    "three_slot_polarization_source",
]


def run_three_slot_polarization(fields: Any, kernel: Optional[Any] = None
                                ) -> Dict[str, Any]:
    """THE THIRD SLOT on this arm: the shared launcher, pinned to ``no_pml``.

    A one-line wrapper rather than an alias, so this module's callers cannot pass the
    other arm's name into a launcher whose only use for it is
    :func:`~.three_slot_dispersive_weld._drive_identity_problem` -- the clause that
    asks whether ``Fields.drive_field`` hands back the stored ``E<c>``, which is the
    one thing about the recurrence that differs between the two arms.
    """
    return shared.run_three_slot_polarization(fields, "no_pml", kernel)


# =============================================================================
# COVERAGE
# =============================================================================

def covers_no_pml_three_slot_dispersive_weld(fields: Any, pml: Any, grid: Any,
                                             sources: Any = None
                                             ) -> Tuple[bool, str]:
    """May ONE product span ``step_D`` -> ``update_E`` -> ``update_P`` with no absorber?

    A CONJUNCTION OF TWO SHIPPED PREDICATES, AND NOTHING IS WEAKENED. The D->E half
    is ``no_pml_dispersive_fused_electric_pair.covers_no_pml_dispersive_fused_
    electric_pair`` whole -- which is where the SOURCE question is asked, through
    ``deposit_repair.seam_source_reasons`` carrying that module's own
    ``CARRIES_DEPOSIT_REPAIR`` and its PLAIN repair path, the same declaration this
    module makes. The E->P half is
    ``fused_polarization_pair.covers_no_pml_fused_polarization_pair`` whole.

    DISJOINT FROM THE PML TWIN ON THE ABSORBER ALONE: both of this one's halves
    refuse an active layer by name and both of that one's require it, so no run can
    be claimed by two -- the same single boolean the four halves already partition on.

    THE ADMISSION SET IS A SUBSET OF EACH HALF'S, which is what
    ``fused_pairs._superseded_by_a_longer_span`` needs.

    NOTHING IS ADDED, for the PML twin's measured reason: see
    :func:`~.three_slot_dispersive_weld.covers_three_slot_dispersive_weld`, including
    which fact about this product no predicate can express.
    """
    from . import no_pml_dispersive_fused_electric_pair as de  # noqa: PLC0415

    covered, reason = de.covers_no_pml_dispersive_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return False, f"step_D->update_E half: {reason}"
    covered, reason = ep.covers_no_pml_fused_polarization_pair(
        fields, pml, grid, sources)
    if not covered:
        return False, f"update_E->update_P half: {reason}"
    return True, "covered"

"""A Metal kernel gate's arm table: the families this process imported, and no others.

:func:`meep_gpu.metal_kernels.arms.ensure_registered` imports every family before the
table is read, which is right for a run: selection must not depend on import order. For
a gate it puts every family in the import record, so an edit to any kernel voids every
gate that plans a step. A gate whose record reads the same under both tables closes the
table first, and its import record is then its own dependency set. The package has no
such mode, so nothing a run can reach closes the table; ``test_metal_closed_arm_table``
pins which gates close it and that the whole-table readers do not.
"""

from __future__ import annotations

from meep_gpu.metal_kernels import arms, launch


def close_table() -> None:
    """Refuse an unarmed slot by name and never import the rest of the table."""
    state = arms.registration_state()
    if state != "cold":
        raise RuntimeError(
            f"the arm table cannot close once it is {state}: a reader has already seen "
            "the complete table")
    arms._REGISTRATION_STATE = "done"
    launch.NO_ARM_REASONS.update(
        {slot: f"{slot}: no family in this gate's closed table" for slot in launch.STEP_ORDER})

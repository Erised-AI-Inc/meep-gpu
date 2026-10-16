"""Which vacuity floors do not apply under the ``flush`` policy, decided by measurement.

WHY THIS EXISTS. Several Triton gates refuse a value class whose operands hold no float32
subnormal ("VACUOUS OPERAND CLASS: value_class='subnormal_band' seeded 0 subnormals"),
because under ``keep`` such a row measured nothing about the class it is named for. The
Triton fleet never ran under ``flush`` before the 1.0 round, and on its first flush run
eight of them failed on exactly that floor while every byte comparison was identical
(pre-certification pass FPC, 2026-10-10). Under ``flush`` the class cannot occur: the host
FPU flushes (MEEP's ``set_zero_subnormals``, FTZ and DAZ) and every executor compiles
flush-to-zero arithmetic, so a subnormal is never the result of a computation -- not in a
gate that seeds its band with host arithmetic, and not in any field MEEP itself steps. A
floor demanding subnormal operands then fails a correct kernel for a property of the
policy.

WHAT IS DECLARED, AND ONLY WHEN THREE MEASUREMENTS AGREE.
:func:`subnormal_class_not_applicable` names the floor NOT APPLICABLE when

1. the installed subnormal policy is ``flush`` (an install happened and resolved flush);
2. the host FPU is flushing subnormals at the moment the class is seeded
   (``backends.subnormals_flushed``, which multiplies, not reads a flag); and
3. the class's own census counted ``0`` subnormal words.

Any one false and the floor applies unchanged: always under ``keep``, and under ``flush``
whenever the seeding did reach the band (a bit-planted subnormal is counted, and the
floor is then met). Only floors about SUBNORMAL operands are in scope: signed-zero and
zero floors stand under both policies (``-0`` survives a flush), and every other check
a row carries -- its byte comparison first -- is unchanged; under ``flush`` that
comparison still measures that every executor flushes the band's small normal
products the same way.

:func:`seed_conditions` takes the first two measurements; a gate records them in the row
beside its census, so the verdict is a function of the artifact.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

FLUSH = "flush"

#: The words a row carries when the floor is declared not applicable.
NOT_APPLICABLE = "NOT-APPLICABLE under flush"


def seed_conditions() -> Dict[str, Any]:
    """The policy in force and the host FPU's behaviour, measured now."""
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    installed = bool(subnormal_policy.policy_is_installed())
    return {"policy_installed": installed,
            "policy": subnormal_policy.get_subnormal_policy() if installed else None,
            "host_flushing": bool(backends.subnormals_flushed())}


def subnormal_class_not_applicable(subnormal_words: Any,
                                   conditions: Optional[Mapping[str, Any]]
                                   ) -> Optional[str]:
    """Why a subnormal-operand floor does not apply to this row, or ``None`` if it does."""
    if not isinstance(conditions, Mapping):
        return None
    if not conditions.get("policy_installed") or conditions.get("policy") != FLUSH:
        return None
    if conditions.get("host_flushing") is not True:
        return None
    try:
        count = int(subnormal_words)
    except (TypeError, ValueError):
        return None
    if count != 0:
        return None
    return (f"{NOT_APPLICABLE} (measured): the installed policy is flush, the host FPU "
            f"flushes subnormals, and the class's census counted 0 subnormal words; "
            f"the class cannot occur under this policy, and every other check on the "
            f"row is unchanged")

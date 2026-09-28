"""The arm table, as DATA — which product may fill which sub-step slot.

WHY THIS EXISTS. ``plan_step`` spelled its arm table as two literal loops while the
package carried one family. With a tree of families that table has to be data, or
the fail-closed contract gets re-implemented once per family and the ambiguity
message acquires seven homes. Here a family REGISTERS its arms and ``plan_step``
iterates.

THE FAIL-CLOSED CONTRACT IS NOT REDEFINED HERE. ``_Arm`` and ``_select_slot`` are
IMPORTED from :mod:`meep_gpu.triton_kernels.launch` — they are a label plus three
closures and the (a)-(f) clause contract, both backend-free and already shared —
and this module only replaces the hard-coded tables that fed them. Exactly one
admitter fills a slot; two or more leave the slot UNSELECTED naming all admitters;
a predicate that RAISES is a refusal; an unselected slot is the array path.

THE NULL ARM GETS NO PREFERENCE, in either direction, and its gate is a condition
on the CONFIGURATION rather than on the table: a null preferred over a kernel
silently deletes a real sub-step, and a kernel preferred over the null launches the
``dsigw`` accumulation on a step that allocates no ``f_w_*`` at all.

REGISTRATION IS NOT DISPATCH, and that is still true — but the second half of the
sentence this paragraph used to carry is not. Registering an arm here decides what
``plan_step`` composes; it does not decide what DISPATCHES, which since Phase 3 is
``meep_gpu.metal_dispatch``'s ladder and the release rows beside it.
``meep_gpu.fastpath.plan_fast_path`` no longer returns ``None`` on every branch:
it routes a NumPy engine on a host with an MPS device to that ladder, so a row
registered here can now reach the driver's consults, and what stops one that has
not been driven through them is a named refusal there rather than the absence of a
seam.

A FAMILY THAT IS BUILT AND GATED BUT NOT WIRED STILL REGISTERS — with
``wired=False``. It is the ``wired`` flag, not absence from the table, that keeps
it out of composition: :func:`arms_for` skips it so ``plan_step`` cannot launch
it, while :func:`registered` still returns it so a composition probe can sweep
every predicate in the table and MEASURE disjointness rather than assume it.

AS OF TRANCHE 2 EVERY CERTIFIED FAMILY IS WIRED and the flag has no live user;
it is kept because the state it names is real and will recur — the next family to
be built and gated before its composition is settled registers ``wired=False`` and
is swept without being selectable. What changed is not the flag but the ORDER: the
disjointness sweep ran first, over configurations built to admit each family, and
wiring followed the measurement.

THE TABLE IS POPULATED BY IMPORT, WHICH IS WHY :func:`ensure_registered` EXISTS.
A family registers at its own module scope, so the table's contents used to depend
on which modules a caller happened to import — ``import metal_kernels.launch``
alone produced a three-arm table, and the same configuration could therefore
select a different product, or fail to detect an ambiguity at all, depending on an
unrelated earlier import. Every reader of the table now goes through
:func:`ensure_registered` first.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# One definition of the fail-closed machinery, imported rather than re-implemented.
from ..triton_kernels.launch import _Arm, _select_slot  # noqa: F401


class StepContext:
    """Everything an arm's coverage and plan closures need, in one object.

    A registry entry cannot close over ``fields``/``pml``/``residency`` — it is
    built at import time — so the context is passed at composition time and each
    entry is a pair of functions of it. That is the whole reason the table can be
    data.
    """

    __slots__ = ("fields", "pml", "residency", "contract_variants", "sources",
                 "synced", "extra")

    def __init__(self, fields: Any, pml: Any, residency: Any = None,
                 contract_variants: Sequence[str] = (),
                 sources: Any = None, synced: Sequence[str] = (),
                 extra: Optional[Dict[str, Any]] = None) -> None:
        self.fields = fields
        self.pml = pml
        self.residency = residency
        self.contract_variants = tuple(contract_variants)
        self.sources = sources
        self.synced = tuple(synced)
        self.extra = dict(extra or {})


class ArmSpec:
    """One family's claim on one slot.

    ``coverage(context, slot)`` and ``plan(context, slot)`` take the slot as well
    as the context, so ONE pair of functions can serve every slot a family claims
    — the null family registers the same two callables on ``update_H`` and
    ``update_E`` and looks the side up from the slot. ``gate(context)`` is the
    cheap precondition that decides whether the arm is even CONSULTED; it is
    separate from ``coverage`` because a gated-out arm contributes NO reason, while
    a consulted-and-refusing arm contributes all of its reasons by name.

    ``wired`` is not bookkeeping. A family may be built, gated and byte-certified
    while its COMPOSITION is deliberately deferred; it registers with
    ``wired=False`` and :func:`arms_for` skips it, so ``plan_step`` cannot launch
    it — but it stays ENUMERABLE, which is what lets the composition probe sweep
    every registered predicate and assert that no unwired family co-admits a
    configuration a wired one admits. A family simply absent from the table could
    not be swept at all, and "it does not overlap" would be an assumption.
    """

    __slots__ = ("family", "slot", "label", "gate", "coverage", "plan", "prefix",
                 "noun", "wired", "replaces")

    def __init__(self, family: str, slot: str, label: str,
                 coverage: Callable[..., Any], plan: Callable[..., Any],
                 prefix: Optional[str] = None, noun: Optional[str] = None,
                 gate: Optional[Callable[[Any], bool]] = None,
                 wired: bool = True,
                 replaces: Optional[Tuple[str, ...]] = None) -> None:
        self.family = family
        self.slot = slot
        self.label = label
        self.coverage = coverage
        self.plan = plan
        self.prefix = f"{label}: " if prefix is None else prefix
        self.noun = label if noun is None else noun
        self.gate = gate
        self.wired = bool(wired)
        self.replaces = (slot,) if replaces is None else tuple(replaces)

    @property
    def is_weld(self) -> bool:
        """Does this arm REPLACE more driver passes than the one slot it sits on?

        THIS IS THE DISTINCTION THE DISJOINTNESS SWEEP NEEDS, and ``wired`` is not
        it. A fused product registers ``wired=False`` on ONE slot while replacing
        the whole seam -- its curl, the in-seam passes, and the constitutive that
        closes it -- so its predicate is a CONJUNCTION whose first conjunct is
        verbatim the wired curl arm's. Containment is therefore by construction and
        a weld ALWAYS co-admits with the arm it welds. A sweep that asserts "at most
        one arm admits" over the raw table is asserting something no weld can ever
        satisfy.

        ``wired`` cannot stand in for this. Today every unwired arm happens to be a
        weld (measured: 16 of 16, 9 on ``step_B``, 3 on ``step_D``, 4 on
        ``update_E``), so filtering on ``not wired`` would look identical and green
        -- while silently deleting the case the sweep was BUILT for: a deferred
        SINGLE-SLOT peer, registered unwired ahead of its wiring, that would genuinely
        collide with an existing arm the moment it is wired. That peer is a real
        pattern here, not a hypothetical; see the anticipatory check in
        ``parity/meep_gpu/gate_metal_special_kz.py``.
        """
        return self.replaces != (self.slot,)

    def bind(self, context: Any) -> _Arm:
        """This spec against one composition context, as the shared ``_Arm``."""
        return _Arm(self.label,
                    True if self.gate is None else bool(self.gate(context)),
                    lambda: self.coverage(context, self.slot),
                    lambda: self.plan(context, self.slot),
                    self.prefix,
                    self.noun)

    def __repr__(self) -> str:
        return (f"ArmSpec({self.family}/{self.slot}/{self.label}, "
                f"wired={self.wired})")


#: slot -> the arms registered for it, in registration order. Order decides NOTHING
#: about selection — ``_select_slot`` refuses rather than picking by table order —
#: it only decides the order refusals are reported in.
_REGISTRY: Dict[str, List[ArmSpec]] = {}


def register(family: str, slot: str, label: str, coverage: Callable[..., Any],
             plan: Callable[..., Any], prefix: Optional[str] = None,
             noun: Optional[str] = None,
             gate: Optional[Callable[[Any], bool]] = None,
             wired: bool = True,
             replaces: Optional[Tuple[str, ...]] = None) -> ArmSpec:
    """Add one arm to the table. Re-registering a (family, slot, label) is refused.

    A duplicate is refused rather than replaced: the second registration is almost
    always a module re-imported under two names, and silently keeping one of the
    two would make WHICH KERNEL RUNS depend on import order. Refusing means the
    tree fails at import, loudly, in one place.
    """
    spec = ArmSpec(family, slot, label, coverage, plan, prefix, noun, gate, wired,
                   replaces)
    existing_arms = _REGISTRY.setdefault(slot, [])
    for existing in existing_arms:
        if (existing.family, existing.label) == (family, label):
            raise ValueError(
                f"{family!r} already registered {label!r} on slot {slot!r}; a "
                f"second registration would make the slot self-ambiguous and "
                f"leave it permanently UNSELECTED")
    existing_arms.append(spec)
    return spec


#: Where the table is in its one-time population. ``cold`` -> nothing imported;
#: ``running`` -> :mod:`.registry` is executing; ``done`` -> every family module has
#: been imported and every arm it registers is in :data:`_REGISTRY`.
_REGISTRATION_STATE = "cold"


def ensure_registered() -> None:
    """Import every family module, so the table is COMPLETE before it is read.

    REGISTRATION IS AN IMPORT SIDE EFFECT and that is the whole hazard this
    function exists to remove. A family registers its arms at its own module scope,
    so before tranche 2 the table's contents depended on WHICH MODULES THE CALLER
    HAPPENED TO IMPORT: ``import metal_kernels.launch`` alone produced a three-arm
    table, and the same ``plan_step`` call could therefore select a different
    product depending on an unrelated earlier import. That is not a fail-closed
    composer; it is a composer whose ambiguity detection can be switched off by
    import order.

    :mod:`.registry` is the one module that imports them all, and every reader of
    the table goes through here first. The import is INSIDE A FUNCTION BODY
    deliberately: two families import ``.launch`` for ``SUB_STEPS``, so a
    module-scope import here would be a cycle, and the deferral is also what keeps
    ``import meep_gpu.metal_kernels`` free of torch.

    A family that queries the table during its OWN import is refused rather than
    served a half-populated answer — that read would be correct only by accident of
    registration order, which is the same defect one level down.
    """
    global _REGISTRATION_STATE

    if _REGISTRATION_STATE == "done":
        return
    if _REGISTRATION_STATE == "running":
        raise RuntimeError(
            "a family module read the arm table while the table was still being "
            "populated; the answer would depend on registration order. Read it "
            "from a function body that runs after import instead")
    _REGISTRATION_STATE = "running"
    try:
        from . import registry  # noqa: F401, PLC0415 - import IS the registration
    except BaseException:
        _REGISTRATION_STATE = "cold"
        raise
    _REGISTRATION_STATE = "done"


def registration_state() -> str:
    """``cold`` / ``running`` / ``done`` — for a probe that wants to record it."""
    return _REGISTRATION_STATE


def registered(slot: Optional[str] = None) -> Tuple[ArmSpec, ...]:
    """Every registered arm — WIRED OR NOT — for one slot, or for every slot."""
    ensure_registered()
    if slot is not None:
        return tuple(_REGISTRY.get(slot, ()))
    return tuple(spec for arms in _REGISTRY.values() for spec in arms)


def registered_slots() -> Tuple[str, ...]:
    ensure_registered()
    return tuple(_REGISTRY)


def families_on(slot: str) -> Tuple[str, ...]:
    ensure_registered()
    return tuple(spec.family for spec in _REGISTRY.get(slot, ()))


def arms_for(slot: str, context: Any) -> Tuple[_Arm, ...]:
    """The WIRED arms for one slot, bound to this composition context."""
    ensure_registered()
    return tuple(spec.bind(context) for spec in _REGISTRY.get(slot, ())
                 if spec.wired)


def ambiguity(slot: str) -> Callable[[Sequence[str]], str]:
    """The one home for the co-admission message, phrased per slot KIND.

    The wording is the composer's own, unchanged: two products admitting one slot is
    an over-covering dispatch, and the composition refuses rather than picking by
    table order. Keeping it here is what stops seven families from writing seven
    slightly different sentences about the same contract.
    """
    kind = ("curl" if slot in ("step_B", "step_D")
            else "constitutive" if slot in ("update_H", "update_E")
            else slot)

    def message(labels: Sequence[str]) -> str:
        return (f"two {kind} products admit this configuration "
                f"({', '.join(labels)}); the composition refuses rather than "
                f"picking by table order")

    return message

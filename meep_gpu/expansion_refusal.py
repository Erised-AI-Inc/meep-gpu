"""What a dispatch has already DECIDED about the complex-expansion probe.

TWO CASES THAT DESERVE OPPOSITE OUTCOMES, AND USED TO BE THE SAME VALUE.

A kernel family's plan builders and coverage predicates all take ``probe=``, and
until 2026-08-16 the only two things a caller could say through it were "here is a
record" and ``None``. ``None`` was read everywhere as *nothing was offered*, which
licenses the obvious fallback::

    record = probe if probe is not None else load_expansion_probe()

That fallback is correct for a caller with no opinion. It is catastrophic for a
caller that HAD an artifact, JUDGED it, and REFUSED it — because the refusal was
also spelled ``None``, so the very next rung read the environment variable the
refused artifact came from and got it straight back. Measured on the shipped
dispatch path: ``fastpath`` dropped a flush-cut artifact against its required
``keep`` policy, passed ``None``, and ``triton_kernels.launch.plan_step`` re-read
the same file and handed it to every complex arm. The drop was advisory where it
had to bind.

So the two cases are given two values:

``None``
    NOTHING WAS OFFERED. A rung may read the environment for itself, and refuses
    by name when that finds nothing either.

:class:`RefusedExpansionProbe`
    SOMETHING WAS OFFERED AND REFUSED. It may never bind an arm, never fall back
    to the reader, and never reach an environment-default table. It is not
    ``None``, so every ``probe if probe is not None else ...`` site in every
    family honours it without being edited, and every licence funnel refuses it
    by name while quoting the reasons it carries.

THE STANDING REFUSAL is the other half. Passing the value forward only protects
rungs the value reaches; a rung that was handed ``None`` by an older caller could
still open the artifact itself. So a refusal is also recorded, for the duration
of the dispatch that made it, against the ENVIRONMENT VARIABLE it refused — and
each backend's reader consults that in its own function body before returning a
record. In its FUNCTION BODY and not by rebinding the module attribute: sibling
modules bind the reader with ``from .complex_fields import load_expansion_probe``
at import time, so a rebound attribute would be invisible to them.

THE DECLARED RUN POLICY is the module's other half, and it is here rather than
beside the policy module because it is what makes the refusal above JUDGEABLE: a
probe artifact's licence is conditional on the subnormal policy it was cut under,
and dispatch consults the artifact several rungs before it installs a policy, so
at every rung that binds an arm there is nothing to read. A dispatch therefore
DECLARES the policy it requires and will install. Because a declaration is the
softest link in that chain, it is also the one an attacker reaches for, and
:func:`declaring_run_policy` says what it may and may not do: any policy name may
be declared — and it must BE one, not merely spell one — but a declaration made
inside a window that declared something else does not replace it: the two
disagree, and disagreement refuses.

WHAT A DECLARATION IS NOT is a licence for anybody's kernels. It is one
process-wide fact about arithmetic, pooled across backends on purpose (see
:data:`_DECLARATIONS`), while the arms of different backends are CERTIFIED under
different policies — Metal's device flushes float32 subnormals natively and
exposes no lever, the Triton complex families were certified under ``'keep'``.
So each family's licence funnel owes a second comparison this module cannot make
for it: the policy in force against the policy THAT family was certified under.
This module reports what was declared; it does not say whose arms that licenses.

THIS MODULE DEPENDS ON NO BACKEND. No CuPy, no Triton, no Metal, no kernel
module — it is a value and two dictionaries. (It reads its sibling
:mod:`meep_gpu.subnormal_policy` for the list of policy NAMES, lazily and inside
a function, so that list has one home; nothing runs at import time and no backend
package is involved.) Every backend can adopt it by (i) returning
:func:`standing_refusal` from its reader, (ii) refusing
:class:`RefusedExpansionProbe` by name in its licence funnel. Keeping it out of
any one backend's package is the point: no backend should have to import
another's to say "I refused this".
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

__all__ = [
    "RefusedExpansionProbe",
    "refuse_expansion_probe",
    "standing_refusal",
    "refusing_expansion_probe",
    "is_expansion_refusal",
    "ContradictedRunPolicy",
    "is_contradicted_run_policy",
    "declared_run_policy",
    "declaring_run_policy",
]


class RefusedExpansionProbe:
    """An expansion probe artifact that was offered to this dispatch and REFUSED.

    Deliberately NOT a ``dict`` and deliberately NOT ``None``:

    * not ``None``, so the twelve-plus ``probe if probe is not None else
      load_expansion_probe()`` sites and ``plan_step``'s ``resolved_probe is
      None`` short-circuit on it rather than re-reading the environment;
    * not a ``dict``, so every licence funnel's ``isinstance(record, dict)``
      guard already refuses it even before the by-name clause that quotes
      :attr:`reasons`.

    ``key`` names the environment variable whose artifact was refused, so one
    backend's refusal never silences another's (the Triton and Metal families
    read different variables and score different pattern sets).
    """

    __slots__ = ("key", "reasons")

    def __init__(self, key: str, reasons: Iterable[str]) -> None:
        self.key = str(key)
        self.reasons: Tuple[str, ...] = tuple(str(reason) for reason in reasons)
        if not self.reasons:
            # A refusal with no reason is indistinguishable from an accident, and
            # this value's whole job is to be distinguishable. Refuse to be one.
            raise ValueError("a refused expansion probe must carry at least one "
                             "reason; a refusal nobody can read is a drop")

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"RefusedExpansionProbe(key={self.key!r}, "
                f"reasons={list(self.reasons)!r})")


def is_expansion_refusal(value: object) -> bool:
    """True when ``value`` is a refusal — the check every binding rung owes."""
    return isinstance(value, RefusedExpansionProbe)


def refuse_expansion_probe(key: str, reasons: Iterable[str]) -> RefusedExpansionProbe:
    """Mint the refusal value for ``key``. Recording it is :func:`refusing_expansion_probe`."""
    return RefusedExpansionProbe(key, reasons)


#: Refusals in force right now, keyed by the environment variable each refuses.
#: Written only through :func:`refusing_expansion_probe`, which restores what it
#: found, so a refusal is scoped to the dispatch that made it and never leaks
#: into unrelated later work in the same process.
#:
#: PROCESS-SCOPED, NOT THREAD-SCOPED, and named as such rather than left to be
#: discovered: two dispatches composing plans concurrently in one process would
#: share this dictionary. That is the same scope the thing it guards already has
#: — the environment variable it keys on, and ``subnormal_policy``'s install,
#: are both process state — so a per-thread refusal would be narrower than the
#: fact it is about. The day a second thread composes plans, the policy install
#: is the harder half of the same problem.
_STANDING: Dict[str, RefusedExpansionProbe] = {}


def standing_refusal(key: str) -> Optional[RefusedExpansionProbe]:
    """The refusal in force for ``key``, or None.

    A reader calls this IN ITS FUNCTION BODY, before it opens anything. Reading
    module state rather than being monkeypatched is what makes the refusal reach
    sibling modules that bound the reader with ``from ... import`` at import time.
    """
    return _STANDING.get(key)


@contextmanager
def refusing_expansion_probe(key: str, reasons: Iterable[str]
                             ) -> Iterator[RefusedExpansionProbe]:
    """Record a refusal of ``key``'s artifact for the duration of this block.

    Yields the value to pass DOWNWARD. Both halves matter and neither is
    redundant: the value binds the rungs it reaches, the record binds the rungs
    it does not.
    """
    refusal = RefusedExpansionProbe(key, reasons)
    previous = _STANDING.get(key)
    _STANDING[key] = refusal
    try:
        yield refusal
    finally:
        if previous is None:
            _STANDING.pop(key, None)
        else:
            _STANDING[key] = previous


class ContradictedRunPolicy:
    """Two OPEN declarations disagree about this process's one subnormal policy.

    Returned by :func:`declared_run_policy` instead of picking a winner. It is a
    value rather than an exception because this module's whole discipline is to
    refuse LICENCES, not to abort runs: a dispatch that cannot say which
    arithmetic it will use must fall back to the array path, which is what every
    policy-conditional clause does when handed something it cannot judge. An
    exception here would take down a simulation that would otherwise have run
    correctly, just without a fused kernel.

    It carries the disagreeing names, outermost declaration first, so the
    refusal it causes can quote what was actually said rather than reporting the
    policy as merely unreadable.
    """

    __slots__ = ("policies",)

    def __init__(self, policies: Iterable[str]) -> None:
        self.policies: Tuple[str, ...] = tuple(
            dict.fromkeys(str(policy) for policy in policies))

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ContradictedRunPolicy(policies={list(self.policies)!r})"


def is_contradicted_run_policy(value: object) -> bool:
    """True when ``value`` is a contradiction rather than a policy name."""
    return isinstance(value, ContradictedRunPolicy)


def _known_policies() -> Optional[Tuple[str, ...]]:
    """The policy names :func:`declaring_run_policy` accepts, or None.

    Read from :mod:`meep_gpu.subnormal_policy` so the list of policies has ONE
    home — that module resolves what this process does to float32 subnormals and
    owns the names. Imported in the body, not at module scope, to keep this
    module's stated property true: it is importable by any backend without
    dragging in anything, and nothing here runs at import time.

    None when the sibling cannot be read, and the caller then accepts any
    string. That is deliberately the weaker half of the check: an unimportable
    ``subnormal_policy`` means the package is broken, and turning that into an
    exception at every declaration would replace a broken package with a broken
    dispatch. The check that actually closes the reported hole is the STRING
    check in the caller, which needs nothing imported — and an unrecognised name
    still fails closed downstream, because it can match neither an artifact's
    stamp nor a family's certification policy.
    """
    try:
        from . import subnormal_policy  # noqa: PLC0415
        return tuple(subnormal_policy.POLICIES)
    except Exception:  # noqa: BLE001 - a list that cannot be read is not a veto
        return None


#: Every run-policy declaration currently open, outermost first. ALL OF THEM
#: rather than a single slot, and NOT SEGREGATED BY BACKEND the way
#: :data:`_STANDING` is segregated by artifact. Both are decisions, and both have
#: a measurement behind them. (The mapping IS keyed, by a per-window token — but
#: that key is a LIFETIME HANDLE and carries no meaning about who declared; see
#: the last section. It is not a semantic key and does not segregate anything.)
#:
#: WHY ALL OF THEM. A single slot let an inner declaration silently replace an outer
#: one, and that is a laundering route rather than a nuisance. MEASURED
#: 2026-08-16 against the five seams that gate a plan builder: inside a dispatch
#: that had declared ``'keep'``, a nested ``declaring_run_policy('flush')`` made
#: ``complex_fields._expansion_reasons``, ``folded_complex``'s three predicates
#: and ``special_kz._beta_expansion_reasons`` all return ZERO reasons on a
#: flush-cut record, and ``_resolve_expansion`` bind FMA_V1 — the artifact
#: ``fastpath``'s own policy clause had refused. It held whether the record was
#: passed explicitly or read back off the environment variable. Keeping every
#: open declaration is what lets that disagreement be SEEN instead of silently
#: resolved in the inner caller's favour.
#:
#: WHY NOT SEGREGATED BY BACKEND, when a refusal is. They are keyed on what each
#: fact is about. A refusal is about ONE BACKEND'S ARTIFACT — each backend reads its own
#: environment variable and scores its own pattern set — so one backend's
#: refusal must not silence another's. A subnormal policy is not like that: it
#: is ONE PROCESS-WIDE fact. :mod:`meep_gpu.subnormal_policy` drives the host
#: FPU, the device compiler's flags and the JIT's arithmetic lowering from a
#: single decision, precisely because a half-applied policy is worse than either
#: policy applied uniformly. Two backends therefore cannot legitimately be under
#: different ones at the same instant, and keying the declaration per backend
#: would let two disagreeing statements about a single fact both be believed —
#: the same laundering hole wearing a key. Pooled and CONTRADICTION-AWARE is
#: the shape that matches the fact.
#:
#: THE PRICE OF POOLING THEM IS PAID ELSEWHERE, and it has to be, because this
#: shape is right about the policy and wrong about the KERNELS. Two backends
#: cannot be under different subnormal policies — but their arms can be, and are,
#: CERTIFIED under different ones: the MPS executor flushes natively and exposes
#: no lever, so Metal must declare ``'flush'``, while every Triton complex family
#: was certified under ``'keep'``. A declaration is therefore not by itself a
#: licence for anybody's arm, and a funnel that treats it as one lets one
#: backend's honest declaration bind another's kernels. That is why each family's
#: licence funnel ALSO compares the policy in force against the policy IT was
#: certified under — see ``triton_kernels.complex_fields``'s
#: ``CERTIFIED_UNDER_SUBNORMAL_POLICY`` and ``expansion_certification_reasons``.
#: This module says what is declared; it does not say whose kernels that licenses.
#:
#: Process-scoped, not thread-scoped, for the same reason :data:`_STANDING` is:
#: the fact it is about is process state.
#:
#: WHY IT IS KEYED BY A TOKEN AND NOT INDEXED BY POSITION. This was a list of
#: names, and each window removed its entry by the POSITION that entry took::
#:
#:     depth = len(_DECLARATIONS)          # at entry
#:     if len(_DECLARATIONS) >= depth:     # at exit
#:         del _DECLARATIONS[depth - 1]
#:
#: Positions SHIFT when an earlier window closes, so a window whose entry had
#: moved down no longer indexed itself: the guard saw a stack shorter than the
#: depth it remembered, skipped the delete, and the entry stood FOR THE LIFE OF
#: THE PROCESS. MEASURED 2026-08-16 three ways — an abandoned generator holding a
#: declaration across a yield, three windows with the middle closed first, and
#: two threads whose ordinary ``with`` blocks overlap because they are on
#: different stacks. All three end with ``_DECLARATIONS == ['flush']`` and NO
#: window open, and in that state ``_policy_in_force()`` answers ``'flush'``, all
#: five gating seams score zero reasons on a flush-cut record and
#: ``_resolve_expansion`` binds FMA_V1. That is the laundering hole the
#: contradiction rule closes, restored by the cleanup that was meant to protect
#: it: the ``'keep'`` that made the pair a refusal is erased on the way out and
#: the foreign ``'flush'`` is left standing alone.
#:
#: The position guard was not gratuitous — it was defending a REAL case, which
#: is why the fix may not simply be to pop the end. A generator finalized out of
#: order pops somebody else's declaration and leaves its own behind, which is the
#: same leak arriving from the other side. BOTH properties are required:
#:
#:   (a) no entry may outlive its window, and
#:   (b) a window finalized out of order must retire ITS OWN entry, nobody else's.
#:
#: A TOKEN HELD BY THE WINDOW has both, and has them without a rule. Each window
#: mints a fresh :class:`object` — unique, hashable by identity, unforgeable —
#: uses it as the key, and deletes THAT KEY on the way out. (a) holds because the
#: delete is unconditional: there is no predicate that can decline to fire. (b)
#: holds because a key is an identity and never a position, so nothing another
#: window does can make this window's handle name a different entry. Both fall
#: out of the data structure rather than being enforced by a guard that has to be
#: right, which is the difference that matters. The guard that was here shipped
#: with a comment correctly naming the hazard it was defending against, and was
#: still wrong about the other one; an invariant that has to be re-derived by
#: every reader is one a reader can agree with and still not check.
#:
#: A dict rather than a list of pairs because deletion by key is O(1) and, more
#: to the point, TOTAL: ``pop(token, None)`` cannot fail to remove its own entry
#: and cannot remove another. Insertion order is declaration order, outermost
#: first, which is what :func:`declared_run_policy` reports.
_DECLARATIONS: Dict[object, str] = {}


def declared_run_policy() -> Any:
    """What the surrounding dispatch DECLARED — and whether it said one thing.

    THREE RETURNS, and the third is the 2026-08-16 correction:

    ``None``
        nothing is declared.
    a policy name
        every open declaration agrees. Re-declaring the SAME policy is
        idempotent and stays legal, because it already happens in this tree:
        ``conftest``'s ``run_policy_declared_keep`` fixture wraps
        ``fastpath._composition_window``, and both say ``'keep'``.
    :class:`ContradictedRunPolicy`
        open declarations disagree. NOT resolved to either of them and above all
        not to the innermost: they are two statements about one process-wide
        fact, so at most one is true and nothing here can say which. Every
        policy-conditional clause refuses on it, which is the fail-closed answer
        the house default asks for.
    """
    if not _DECLARATIONS:
        return None
    distinct = tuple(dict.fromkeys(_DECLARATIONS.values()))
    if len(distinct) == 1:
        return distinct[0]
    return ContradictedRunPolicy(distinct)


@contextmanager
def declaring_run_policy(policy: Optional[str]) -> Iterator[Optional[str]]:
    """State the subnormal policy this dispatch will run under, before installing it.

    WHY A DECLARATION EXISTS AT ALL. A probe artifact's licence is conditional on
    the subnormal policy it was cut under, so the rung that consumes one has to
    know which policy is in force. On the dispatch path it cannot READ one:
    ``fastpath`` consults the probe four rungs before it installs a policy, so
    ``policy_is_installed()`` is False at every rung that binds an arm. That is
    not an edge case, it is the state of every dispatch.

    A clause that cannot judge must refuse (nothing else is honest — see
    ``complex_fields.expansion_policy_reasons``), and refusing every dispatch is
    not a fix. The resolution is to give the clause something to judge with:
    dispatch knows exactly which policy it requires and installs, because it
    refuses any process that installed anything else. Declaring it is stating a
    fact dispatch already owns, not guessing one.

    The declaration is NOT a substitute for the install and never outranks it:
    an installed policy is a measurement of this process's arithmetic, a
    declaration is only an intent, so readers prefer the install whenever there
    is one.

    WHAT A DECLARATION MAY NOT DO IS OVERRULE ANOTHER ONE. Any policy name may
    be declared — a backend whose device cannot keep float32 subnormals has to
    be able to say ``'flush'``, and refusing that would be refusing the truth —
    but declaring it INSIDE a window that declared something else does not make
    it the answer. The two are added, not swapped, and
    :func:`declared_run_policy` then reports a :class:`ContradictedRunPolicy`
    that every licence clause refuses on. That is what stops a declaration from
    being used to launder a record refused under a different policy: the
    laundering move is exactly "declare the policy the artifact was cut under,
    inside the dispatch that refused it".

    ``policy=None`` DECLARES NOTHING, and specifically does not RETRACT: under
    the old single-slot implementation an inner ``None`` erased the dispatch's
    declaration for its whole body. Nothing in this tree passes it, and the
    honest reading of "I have nothing to say" is silence, not a retraction of
    what someone else said.

    WHAT MAY BE DECLARED IS CHECKED, because this used to be ``str(policy)`` and
    a conversion is not a check. An object whose ``__str__`` returns ``'keep'``
    declared ``'keep'`` and bound an arm; ``True`` became the string ``'True'``
    and refused. Neither is a policy, and the first is the shape an attacker
    wants — a proxy that SPELLS a real name. A name is required to BE a string
    and to be one of :mod:`meep_gpu.subnormal_policy`'s policies. Violations
    raise rather than refuse quietly: every other value this module rejects is a
    licence question decided at a funnel, but this one is a caller passing
    something that is not a policy at all, and the honest place to report that is
    the call that made it — the same stance :class:`RefusedExpansionProbe` takes
    on being minted without a reason.
    """
    if policy is None:
        yield None
        return
    if not isinstance(policy, str):
        raise ValueError(
            f"a run policy declaration must be a policy NAME, not "
            f"{type(policy).__name__}; str() is a conversion and not a check, so "
            f"an object whose __str__ spells a policy would declare it")
    known = _known_policies()
    if known is not None and policy not in known:
        raise ValueError(
            f"{policy!r} is not a float32 subnormal policy; expected one of "
            f"{list(known)!r}")
    token = object()
    # THE TOKEN IS THE HANDLE, and its IDENTITY is the whole of it — see
    # _DECLARATIONS for why a position was not. Fresh per window, so no other
    # window can hold it, and unforgeable, so nothing can retire this entry but
    # the block that opened it.
    _DECLARATIONS[token] = policy
    try:
        yield policy
    finally:
        # UNCONDITIONAL and EXACT. Unconditional so no entry can outlive its
        # window; exact — by key, never by position — so a window finalized out
        # of order retires its own entry and nobody else's. Both properties, and
        # neither depends on a guard being right about the shape of the stack.
        _DECLARATIONS.pop(token, None)

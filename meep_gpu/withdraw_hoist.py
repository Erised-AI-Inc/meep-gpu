"""Hoist the electric integrated-source withdraw over ``update_H``, for a product that
fuses the ``update_H -> step_D`` seam.

WHY THIS IS A SIBLING OF ``deposit_repair`` AND NOT A BRANCH OF IT. The two modules
answer the same-shaped question about two opposite passes, and the difference is not a
detail of wording:

* ``deposit_repair`` is about an INJECTION. ``source.inject`` lands between the curl and
  the constitutive half (``driver.py:3293-3294`` magnetic, ``:3318-3322`` electric), so a
  product fusing those halves consumed a PRE-injection field and something has to be put
  back AFTERWARDS. Its whole contract follows from that: two slots, a
  :class:`~.deposit_repair.LeadingRepairPlan` that saves state before the launch and a
  :class:`~.deposit_repair.TrailingRepairPlan` that recomputes the constitutive at the
  deposit points after the driver has injected, filled and cleared.
* THIS module is about a WITHDRAWAL, and there is no injection in this seam at all. The
  only statement between the ``update_H`` consult (``driver.py:3311``) and the ``step_D``
  consult (``:3315``) is ``for source in electric: getattr(source, "withdraw",
  _no_withdraw)(self.fields)`` (``:3313-3314``), which returns the PREVIOUS step's
  standing dipole to D before the curl reads it. Nothing is deposited into the seam, so
  there is nothing to reconstruct: the pass is MOVED to before the launch and the
  driver's own loop then no-ops. ONE slot, not two; a hoist, not a bracket.

Keeping them apart buys two things. First, the declarations stay separate: a product
that declares ``CARRIES_DEPOSIT_REPAIR`` must not thereby acquire a withdraw licence, and
one that declares it hoists the withdraw must not acquire the deposit repair's -- each
flag is a statement about wiring that product actually has, and a shared flag would let
one measurement license the other. Second, the deposit repair's refusals are all clauses
about RECONSTRUCTING a constitutive product -- the off-diagonal stencil, the
nonlinearity, the absorber whose recurrence it cannot invert, the folded fills that image
a deposit -- and not one of them is a reason a withdraw cannot be hoisted. Inheriting
them here would refuse rows the campaign below measured byte-identical, on grounds that
have nothing to do with this pass.

===========================================================================
THE MEASURED LICENCE, AND IT IS NARROW
===========================================================================

``parity/meep_gpu/results/h_to_d_withdraw_order_2026-09-04/`` (probe
``parity/meep_gpu/probe_h_to_d_withdraw_order.py``). Seven corpus rows -- the seven of
the seam campaign's 194 whose electric withdraw does work -- lifted and stepped 60
COMPLETE driver steps each under five orderings in lockstep from one seed, compared per
complete step as uint32 WORDS over every stored volume the row allocates (D, B, E, H, the
``f_u`` / ``f_cond`` / ``f_bfast`` / ``f_w`` auxiliaries, the ``D - P`` scratches and
every susceptibility's ``P`` / ``P_prev``):

* ``hoisted`` -- the withdraw suppressed in the seam and re-run immediately BEFORE the
  ``update_H`` consult -- **0 differing words of 7,364,789,280**, over 420 complete
  driver steps (7 rows x 60), on all 7 rows;
* ``driver_repeat`` -- the restore control, which proves the save/restore carries
  everything the evolution depends on -- 0 differing words;
* ``hoisted_drop_first`` -- hoisted with the first working withdraw DROPPED -- diverges
  at step 2 on all 7 rows;
* ``after_step_D`` -- the withdraw moved to after the curl, still before the injection
  slot -- diverges at step 2 on all 7 rows.

WHAT THAT LICENSES: the ARRAY PATH, and the placement :data:`BEFORE_UPDATE_H` and no
other. WHAT IT DOES NOT: any device, any kernel, and any other placement. It says nothing
about whether a backend's fused ``update_H + step_D`` launch is bit-identical to the two
array-path halves, and nothing about the MAGNETIC withdraw, which the driver runs over
``step_B`` in a different seam (``driver.py:3289-3290``) and which this campaign never
moved. Those refusals are :data:`PLACEMENT_REFUSALS` and the magnetic clause below,
by name.

===========================================================================
THE MECHANISM, WHICH A BYTE COMPARISON CANNOT GIVE
===========================================================================

``VolumeSource.withdraw`` (``sources.py:2092-2121``) reads ``envelope.is_integrated``,
``_n_source_points`` and ``_applied_dipole``, resolves ONE array with ``_array_for``
(``sources.py:236-237``; an electric component maps to ``Dx``/``Dy``/``Dz`` and to
nothing else, ``_ARRAY_FOR_COMPONENT`` at ``sources.py:119-133``) and hands it to
``_inject_points`` (``sources.py:1297-1305``), whose whole body is
``D[ix, iy, iz] -= _step_source_values(...)``. It then writes ``_applied_dipole = 0j``.
So it reads and writes the component's own D array at its own deposit points and nothing
else; the ``f_u`` mirror is on the NON-integrated inject path only
(``sources.py:2153-2154``, under ``if not self.envelope.is_integrated``).

``stepping.update_H`` (``stepping.py:907-923``) reads B, ``f_w_H*``, the PML's
integer-position constitutive coefficients and a pooled ``fields.scratch``, and writes
``H*`` and ``f_w_H*``.

The two touch disjoint arrays, which is why the hoist ought to be free -- and "ought to
be" is not a measurement, which is why the campaign above exists rather than this
paragraph standing alone.

===========================================================================
WHY THE HOIST NEEDS NO DRIVER EDIT, AND WHAT THAT RESTS ON
===========================================================================

``withdraw`` is IDEMPOTENT within a step: it returns immediately when
``_applied_dipole == 0`` and zeroes that offset after applying it
(``sources.py:2116-2121``). So a product's leading plan may call it for every standing
electric source immediately before its launch and the driver's own unconditional loop
then no-ops -- byte-exactly the ``hoisted`` mode the campaign measured. Nothing between
the two calls re-arms the offset: only ``inject`` writes ``_applied_dipole`` a non-zero
value (``sources.py:2145-2146``) and it runs after ``step_D``. That premise is what
:func:`_span_reasons` protects: a span wider than the seam swallows passes this
campaign never moved.

The idempotence is a property of THAT BODY, so :func:`_idempotence_reasons` refuses by
name a source carrying some other ``withdraw`` implementation rather than assuming the
early return is there. ``meep_gpu/test_withdraw_hoist.py`` DRIVES the idempotence -- two
consecutive withdraws against one injected offset, byte-compared -- rather than asserting
it, so a change to that body fails a test instead of silently widening this licence.

===========================================================================
THE CONFIGURATION TEST IS NOT THE PER-STEP TEST
===========================================================================

Whether a withdraw stands in this seam is a property of the RUN's configuration --
``is_integrated`` and a non-empty deposit-point set -- and :func:`standing_withdraws`
uses exactly that, mirroring the seam probe's own ``withdraw_does_work``
(``parity/meep_gpu/probe_h_to_d_seam.py``). It deliberately does NOT read
``_applied_dipole``, which is the PER-STEP gate: it is zero before the first injection
and non-zero from the second step on, so a predicate that asked it at plan-build time
would answer "no withdraw stands here" for a row whose withdraw does work on every
subsequent step -- and would admit that row into a product with no hoist at all.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

__all__ = ["SEAM", "SEAM_SPAN", "MAGNETIC_FIELD_TYPE", "BEFORE_UPDATE_H", "AFTER_STEP_D",
           "IN_LAUNCH", "PLACEMENTS", "LICENSED_PLACEMENTS", "PLACEMENT_REFUSALS",
           "MEASUREMENT", "WHAT_THIS_DOES_NOT_LICENSE", "WithdrawNotHoistable",
           "standing_withdraws", "hoistable", "hoist", "seam_withdraw_reasons",
           "LeadingWithdrawPlan"]

#: ``sources.FIELD_TYPE_B``. Spelled here rather than imported, the ``deposit_repair``
#: precedent, so this module stays importable by a predicate on a host with no engine
#: objects built yet.
MAGNETIC_FIELD_TYPE = "B"

#: The seam this module serves, as the boards spell it
#: (``parity/meep_gpu/h_to_d_seam.py``).
SEAM = "H_to_D"

#: The driver slots a product must own for the seam's withdraw to need hoisting at all:
#: both halves, and nothing else. A product owning one of them leaves the driver's own
#: loop running between the two launches, and one owning more swallows passes the
#: campaign never moved. See :func:`_span_reasons`.
SEAM_SPAN: Tuple[str, str] = ("update_H", "step_D")

#: THE ONE MEASURED PLACEMENT: the withdraw runs immediately before the ``update_H``
#: consult, so the fused launch and then the driver's now-no-op loop follow it.
BEFORE_UPDATE_H = "before_update_H"

#: NULL CONTROL, refused. The campaign moved the withdraw to after the curl -- still
#: before the injection slot -- and it diverged at step 2 on all 7 rows.
AFTER_STEP_D = "after_step_D"

#: Refused as UNMEASURED, not as wrong. Carrying the withdraw inside the fused launch is
#: the other shape a product could take, and no campaign has compared it against
#: anything.
IN_LAUNCH = "in_launch"

#: Every placement this module has a decided answer for. A name outside it is a
#: declaration error rather than a silent no-op: a caller naming a placement that does
#: not exist has said something about its own wiring that is not true (the
#: ``deposit_repair._path_declaration_reasons`` precedent).
PLACEMENTS: Tuple[str, ...] = (BEFORE_UPDATE_H, AFTER_STEP_D, IN_LAUNCH)

#: What the campaign licenses, and it is one placement.
LICENSED_PLACEMENTS: Tuple[str, ...] = (BEFORE_UPDATE_H,)

#: Why each named placement outside :data:`LICENSED_PLACEMENTS` is refused. Each reason
#: is a measurement or the absence of one, never a preference.
PLACEMENT_REFUSALS: Dict[str, str] = {
    AFTER_STEP_D: (
        "REFUSED PLACEMENT after_step_D: this is one of the campaign's two null "
        "controls, and it is required to diverge. Moving the electric withdraw to after "
        "the curl -- still before the injection slot -- makes step_D read a D that is "
        "missing the previous step's standing dipole, and the measurement diverged at "
        "step 2 on all 7 rows "
        "(results/h_to_d_withdraw_order_2026-09-04, mode `after_step_D`)"),
    IN_LAUNCH: (
        "REFUSED PLACEMENT in_launch: carrying the withdraw inside the fused launch has "
        "never been measured against anything. The 2026-09-04 campaign moved the pass "
        "on the array path only, and its licence is the BEFORE placement; a product "
        "that performed the deposit-point subtraction in-kernel would need its own byte "
        "comparison, with the unwithdrawn launch as the null control"),
}

#: The campaign this module encodes, as data rather than as prose a test would have to
#: parse. ``meep_gpu/test_withdraw_hoist.py`` compares every number here against the
#: record's own ``summary.json`` when that tree is present, so a widened claim fails a
#: test rather than standing in a docstring.
MEASUREMENT: Dict[str, Any] = {
    "campaign": "h_to_d_withdraw_order_2026-09-04",
    "record": "parity/meep_gpu/results/h_to_d_withdraw_order_2026-09-04",
    "probe": "parity/meep_gpu/probe_h_to_d_withdraw_order.py",
    "verdict": "order_equivalent",
    "rows": 7,
    "rows_order_equivalent": 7,
    "steps_per_leg": 60,
    "complete_driver_steps": 420,
    "words_compared": 7364789280,
    "differing_words": 0,
    "must_be_identical": ("driver_repeat", "hoisted"),
    "must_diverge": ("hoisted_drop_first", "after_step_D"),
    "null_controls_first_divergence_step": 2,
    "path": "the NumPy array path",
    "placement": BEFORE_UPDATE_H,
}

#: Said once, here, so every artifact that quotes this module quotes the same sentence.
WHAT_THIS_DOES_NOT_LICENSE = (
    "an ordering measurement on the array path, at the BEFORE placement, for the "
    "ELECTRIC withdraw. It licenses nothing about a device kernel, nothing about "
    "whether a fused update_H+step_D launch is bit-identical to the two array-path "
    "halves, nothing about any other placement, and nothing about the magnetic "
    "withdraw, which the driver runs over step_B in a different seam.")


class WithdrawNotHoistable(RuntimeError):
    """The seam's withdraw may not be moved under the licence this module carries."""


# ---------------------------------------------------------------------------
# Does a withdraw stand in this seam?
# ---------------------------------------------------------------------------


def _is_electric(source: Any) -> bool:
    """The driver's own split: ``field_type != FIELD_TYPE_B`` (``driver.py:3283``).

    A source that publishes no ``field_type`` is electric here for the same reason it is
    electric there -- the driver's list comprehension puts it in the electric list -- so
    this module answers for exactly the sources that loop will iterate.
    """
    return str(getattr(source, "field_type", "")) != MAGNETIC_FIELD_TYPE


def _withdraw_does_work(source: Any) -> bool:
    """Would ``withdraw`` reach its first array write on this source, ever?

    The three conditions ``VolumeSource.withdraw`` itself gates on
    (``sources.py:2114-2117``), minus ``_applied_dipole``, which is the PER-STEP gate --
    see the module docstring's last section. ``callable(source.withdraw)`` is this side's
    spelling of the driver's ``getattr(source, "withdraw", _no_withdraw)``: the no-op is
    only ever reached when the attribute is ABSENT, so a source that carries one is a
    source whose own body runs.
    """
    if not callable(getattr(source, "withdraw", None)):
        return False
    if not bool(getattr(source, "is_integrated", False)):
        return False
    try:
        points = int(getattr(source, "_n_source_points", 0) or 0)
    except (TypeError, ValueError):
        return False
    return points > 0


def standing_withdraws(sources: Any) -> Tuple[Tuple[int, Any], ...]:
    """The electric sources whose withdraw does work, with their index in ``sources``.

    THE INDEX IS THE CALLER'S, not the subset's -- the ``deposit_repair._in_seam_indexed``
    lesson: "source 1" has to name the second source the RUN declared, or a diagnostic on
    a mixed run points at the wrong source.

    Magnetic sources are absent from the result and that is not an omission: the driver
    withdraws them over ``step_B`` (``driver.py:3289-3290``), one seam earlier, where a
    product spanning ``update_H`` and ``step_D`` never sees them.
    """
    return tuple((index, source)
                 for index, source in enumerate(tuple(sources or ()))
                 if _is_electric(source) and _withdraw_does_work(source))


# ---------------------------------------------------------------------------
# The three refusals: the placement, the span, the source
# ---------------------------------------------------------------------------


def _placement_reasons(placement: Any) -> Tuple[str, ...]:
    """Why THIS placement is not the one the campaign measured."""
    if placement in LICENSED_PLACEMENTS:
        return ()
    if placement in PLACEMENT_REFUSALS:
        return (PLACEMENT_REFUSALS[placement],)
    return (f"UNKNOWN PLACEMENT {placement!r}: this module decides "
            f"{PLACEMENTS} and licenses {LICENSED_PLACEMENTS}. A caller naming another "
            f"has declared wiring that does not exist, which is a refusal rather than a "
            f"silent fall-through to the licensed one",)


def _span_reasons(span: Any) -> Tuple[str, ...]:
    """Why THIS span is not the one the hoist was measured for.

    FAIL CLOSED in both directions, and the two directions fail for different reasons.
    A span NARROWER than the seam leaves the driver's own withdraw loop running between
    the two launches, so there is nothing to hoist and a hoist would be an unmeasured
    reordering of a pass that was never swallowed. A span WIDER than the seam swallows
    passes this campaign never moved -- the magnetic withdraw before ``step_B``, the
    electric injection and the D fills after ``step_D`` -- and the licence says nothing
    about any of them.
    """
    try:
        named = tuple(str(slot) for slot in span)
    except TypeError:
        return (f"the span {span!r} is not a sequence of driver slot names, so whether "
                f"it swallows the seam's withdraw cannot be established",)
    if named == SEAM_SPAN:
        return ()
    held = tuple(slot for slot in SEAM_SPAN if slot in named)
    if len(held) == len(SEAM_SPAN):
        extra = tuple(slot for slot in named if slot not in SEAM_SPAN)
        return (f"SPAN WIDER THAN THE SEAM: {named} holds {SEAM_SPAN} and also {extra}. "
                f"The 2026-09-04 campaign moved ONE pass over ONE half -- the electric "
                f"withdraw over update_H -- and a launch spanning further also swallows "
                f"passes it never moved (the magnetic withdraw before step_B at "
                f"driver.py:3289-3290, the electric injection and the D fills after "
                f"step_D at driver.py:3318-3330). Those need their own measurement, not "
                f"this one",)
    if not held:
        return (f"SPAN OUTSIDE THE SEAM: {named} holds neither half of {SEAM_SPAN}, so "
                f"the seam's withdraw is not swallowed by this product at all and there "
                f"is nothing here to hoist",)
    return (f"SPAN HOLDS ONE HALF: {named} holds {held} of {SEAM_SPAN}, so the driver's "
            f"own loop still runs between this product's launch and the other half's "
            f"(driver.py:3313-3314). Nothing is swallowed, so nothing may be moved: a "
            f"hoist here would be a reordering the campaign did not measure",)


def _idempotence_reasons(index: int, source: Any) -> Tuple[str, ...]:
    """Why a SECOND call to this source's ``withdraw`` cannot be shown to be a no-op.

    THE WHOLE HOIST RESTS ON THIS. The product calls ``withdraw`` before its launch and
    the driver's unconditional loop calls it again inside the seam; if the second call
    did work, the hoisted order would subtract the offset twice and the campaign's
    measurement would not describe it. The early return at ``sources.py:2116-2117`` and
    the zeroing at ``:2121`` are what make the second call free, and they are properties
    of THAT BODY -- so a source carrying another implementation is refused by name rather
    than assumed to share them.
    """
    from . import sources as sources_module  # noqa: PLC0415 - avoids a cycle at import

    reasons = []
    measured = getattr(sources_module.VolumeSource, "withdraw", None)
    body = getattr(type(source), "withdraw", None)
    if measured is None or body is not measured:
        reasons.append(
            f"FOREIGN WITHDRAW: source {index} ({type(source).__name__}) does not use "
            f"sources.VolumeSource.withdraw. The hoist rests on that body's idempotence "
            f"-- it returns when _applied_dipole == 0 and zeroes the offset after "
            f"applying it (sources.py:2116-2121) -- which is what lets the driver's own "
            f"loop no-op after a hoisted call. Another implementation has never been "
            f"driven twice in one step and cannot be assumed to")
    if not hasattr(source, "_applied_dipole"):
        reasons.append(
            f"NO STANDING OFFSET: source {index} ({type(source).__name__}) publishes no "
            f"_applied_dipole, which is the attribute the second call's early return "
            f"reads (sources.py:2116); without it the idempotence that makes the "
            f"driver's loop free cannot be established")
    else:
        try:
            bool(getattr(source, "_applied_dipole") == 0)
        except Exception as error:  # noqa: BLE001 - an offset that cannot be compared
            reasons.append(
                f"UNREADABLE STANDING OFFSET: source {index}'s _applied_dipole could not "
                f"be compared against zero ({error!r}), so whether a second withdraw "
                f"returns early (sources.py:2116-2117) cannot be established")
    return tuple(reasons)


def _array_reasons(fields: Any, index: int, source: Any) -> Tuple[str, ...]:
    """Why ``withdraw`` could not reach the array it writes, on THESE fields.

    ``withdraw`` resolves one array with ``_array_for(fields, self.component)``
    (``sources.py:2118``, resolver at ``:236-237``) and subtracts at its deposit points.
    A component this ``Fields`` cannot resolve would raise INSIDE the hoisted call, after
    the composer has already answered that the whole sub-step will run, so it is refused
    here instead.
    """
    from . import sources as sources_module  # noqa: PLC0415 - avoids a cycle at import

    component = getattr(source, "component", None)
    try:
        array = sources_module._array_for(fields, component)  # noqa: SLF001
    except Exception as error:  # noqa: BLE001 - fail closed on anything unresolvable
        return (f"UNRESOLVED ARRAY: source {index}'s component {component!r} does not "
                f"resolve to a primary array on these fields ({error!r}); withdraw "
                f"subtracts its standing dipole from exactly that array "
                f"(sources.py:2118-2120)",)
    if array is None:
        return (f"UNRESOLVED ARRAY: source {index}'s component {component!r} resolves to "
                f"None on these fields, so the array withdraw subtracts from "
                f"(sources.py:2118-2120) is not allocated",)
    return ()


def _declaration_reasons(sources: Any, hoisted: Sequence[Any],
                         standing: Tuple[Tuple[int, Any], ...]) -> Tuple[str, ...]:
    """Why the caller's declared hoist list is not the set the campaign moved.

    TWO NULL CONTROLS LIVE HERE. ``hoisted_drop_first`` dropped one working withdraw and
    diverged at step 2 on all 7 rows, so a declaration missing any standing withdraw is
    refused; and the campaign moved the ELECTRIC list only, so a magnetic source in the
    declaration is refused by name rather than quietly hoisted over a seam it does not
    sit in.
    """
    declared = tuple(hoisted)
    reasons = []
    known = tuple(sources or ())
    for position, source in enumerate(declared):
        if not any(source is candidate for candidate in known):
            reasons.append(
                f"UNDECLARED SOURCE: entry {position} of the hoist list "
                f"({type(source).__name__}) is not in the run's source list, so the "
                f"driver's own loop will never call its withdraw and the hoisted call "
                f"would be the only one -- an ordering nothing measured")
            continue
        if not _is_electric(source):
            reasons.append(
                f"NON-ELECTRIC SOURCE: entry {position} of the hoist list "
                f"({type(source).__name__}, component "
                f"{getattr(source, 'component', None)!r}) is magnetic. The driver "
                f"withdraws magnetic sources over step_B (driver.py:3289-3290), one "
                f"seam earlier; the 2026-09-04 campaign moved the ELECTRIC list only "
                f"and licenses nothing about the magnetic one")
    for index, source in standing:
        if not any(source is candidate for candidate in declared):
            reasons.append(
                f"DROPPED WITHDRAW: source {index} "
                f"({type(source).__name__}, component "
                f"{getattr(source, 'component', None)!r}) has a standing withdraw in "
                f"this seam and is not in the hoist list. Dropping one working withdraw "
                f"is the campaign's `hoisted_drop_first` null control, which diverged at "
                f"step 2 on all 7 rows; the hoist is licensed for the WHOLE electric "
                f"list or not at all")
    return tuple(reasons)


# ---------------------------------------------------------------------------
# The two questions
# ---------------------------------------------------------------------------


def hoistable(fields: Any, sources: Any, *,
              span: Sequence[str] = SEAM_SPAN,
              placement: str = BEFORE_UPDATE_H,
              hoisted: Optional[Sequence[Any]] = None) -> Tuple[bool, Tuple[str, ...]]:
    """May this run's in-seam electric withdraw be moved? ``(ok, reasons)``, never raises.

    FAIL CLOSED, like ``deposit_repair.repairable``: anything this cannot establish is a
    refusal, because a clause that admits on ignorance is the over-covering failure the
    fusion predicates exist to prevent.

    ``span`` is the driver slots the candidate product occupies and ``placement`` is
    where it will run the withdraw; both are checked before any source is looked at,
    because a product with the wrong span or the wrong placement is refused whether or
    not a withdraw stands.

    ``hoisted`` is the caller's declaration of WHICH sources it will move. ``None`` means
    "every standing electric withdraw", which is the shape the campaign measured; an
    explicit sequence is checked against :func:`standing_withdraws` for completeness, so
    a partial hoist is the refusal its null control earned.

    A run with NO standing withdraw is hoistable vacuously -- there is nothing in the
    seam and the driver's loop is the no-op on every electric source. Use
    :func:`standing_withdraws` to ask whether one stands; the two questions are separate
    on purpose.
    """
    reasons = list(_placement_reasons(placement))
    reasons.extend(_span_reasons(span))
    if sources is None:
        reasons.append(
            "NO SOURCE LIST: the run declares none, so whether a withdraw stands in this "
            "seam -- and therefore whether this product may span it -- cannot be "
            "established")
        return False, tuple(reasons)
    standing = standing_withdraws(sources)
    for index, source in standing:
        reasons.extend(_idempotence_reasons(index, source))
        reasons.extend(_array_reasons(fields, index, source))
    if hoisted is not None:
        reasons.extend(_declaration_reasons(sources, hoisted, standing))
    return (not reasons), tuple(reasons)


def hoist(fields: Any, sources: Any, *,
          span: Sequence[str] = SEAM_SPAN,
          placement: str = BEFORE_UPDATE_H) -> int:
    """Perform the hoisted withdraw. Returns the number of sources withdrawn.

    Refuses BEFORE the first array write: a partly hoisted seam is worse than an
    unhoisted one, because the composer has already answered that the whole sub-step will
    run. The whole standing electric list is withdrawn, in the driver's own order, which
    is the set and the order the campaign measured.

    Called from :class:`LeadingWithdrawPlan`, immediately before the fused launch. The
    driver's own loop (``driver.py:3313-3314``) then runs and no-ops on every source this
    touched, by the idempotence at ``sources.py:2116-2121``.
    """
    ok, reasons = hoistable(fields, sources, span=span, placement=placement)
    if not ok:
        raise WithdrawNotHoistable("; ".join(reasons))
    withdrawn = 0
    for _index, source in standing_withdraws(sources):
        source.withdraw(fields)
        withdrawn += 1
    return withdrawn


def seam_withdraw_reasons(fields: Any, sources: Any, *,
                          undeclared: str,
                          refusal: Callable[[int, Any], str],
                          hoists_the_withdraw: bool = False,
                          span: Sequence[str] = SEAM_SPAN,
                          placement: str = BEFORE_UPDATE_H) -> Tuple[str, ...]:
    """Why this run's withdraw bars a fused ``update_H + step_D``, or ``()`` if it does not.

    THE SHAPE IS ``deposit_repair.seam_source_reasons``', and deliberately so: the
    decision lives here and the PROSE stays at the site. ``undeclared`` is the site's own
    no-source-list text and ``refusal(index, source)`` returns its own in-seam text,
    because those strings are pinned by that product's gate and its tests.

    ``hoists_the_withdraw`` is the product's own declaration that its leading slot runs
    :func:`hoist` -- the ``CARRIES_DEPOSIT_REPAIR`` contract's shape, and IT IS NOT A
    HINT for the same reason. A product that passes True without wiring
    :class:`LeadingWithdrawPlan` lets the fused launch read a D still holding the
    previous step's standing dipole and reports success, which is the failure this module
    exists to prevent. The wiring and the flag change together or not at all.

    A product that does not hoist keeps refusing every row with a standing withdraw,
    which is the answer the three boards file today under ``withdraw_seam``.
    """
    if sources is None:
        return (undeclared,)
    standing = standing_withdraws(sources)
    if not standing:
        return ()
    if not hoists_the_withdraw:
        return tuple(refusal(index, source) for index, source in standing)
    ok, reasons = hoistable(fields, sources, span=span, placement=placement)
    if ok:
        return ()
    return tuple(f"the withdraw hoist cannot carry this seam: {reason}"
                 for reason in reasons)


class LeadingWithdrawPlan:
    """The fused launch, with the seam's electric withdraw performed immediately before.

    Occupies the FIRST of the two slots a product spanning this seam owns (``update_H``),
    exactly as :class:`~.deposit_repair.LeadingRepairPlan` occupies ``step_B`` /
    ``step_D``. The hoist happens here rather than at plan-build time because it writes
    the field as of THIS step, and a plan is built once per configuration freeze.

    THERE IS NO TRAILING PLAN, AND THAT IS THE POINT. The deposit repair needs its second
    slot because something was injected after its launch and has to be reconstructed;
    nothing is injected in this seam, and the pass this module moves is undone by
    nobody -- the driver's own loop runs after the launch and no-ops
    (``sources.py:2116-2121``). A second slot here would have nothing to do, and building
    one would suggest a symmetry with the repair that does not exist.
    """

    __slots__ = ("inner", "absorbed_by", "span", "placement", "withdrawn",
                 "_fields", "_sources")

    def __init__(self, inner: Any, fields: Any, sources: Sequence[Any], *,
                 span: Sequence[str] = SEAM_SPAN,
                 placement: str = BEFORE_UPDATE_H) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self.span = tuple(span)
        self.placement = placement
        self._fields = fields
        self._sources = tuple(sources or ())
        #: How many sources the last run withdrew, for a gate to read back.
        self.withdrawn = 0

    def run(self, *args: Any, **kwargs: Any) -> None:
        # THE REFUSAL IS INSIDE hoist, before its first write: this is the last moment a
        # configuration outside the licence can be refused BEFORE the fused launch reads
        # a D that still holds the previous step's standing dipole.
        self.withdrawn = hoist(self._fields, self._sources,
                               span=self.span, placement=self.placement)
        self.inner.run(*args, **kwargs)

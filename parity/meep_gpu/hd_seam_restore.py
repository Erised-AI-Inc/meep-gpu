#!/usr/bin/env python3
"""The H->D seam on the 0.10 source path: one set of measurements for every CUDA and
Triton H->D gate.

WHAT STANDS IN THE SEAM NOW. An integrated source enters at the constitutive update
through ``f_minus_p`` (``sources.f_minus_p_begin`` / ``sources.f_minus_p_end``) and stands
no withdraw in D or B: ``withdraw_hoist.standing_withdraws`` answers ``()`` for every
engine source. The one statement the driver runs between the ``update_H`` and ``step_D``
consults is the MAGNETIC ``f_minus_p`` restore, where H is stored
(``h_to_d_seam.EXPECTED_BETWEEN``); it writes B alone and ``step_D`` reads no B array
(``withdraw_hoist.SEAM_PASS``). A launch spanning both halves therefore runs the restore
after ``step_D`` and must be byte-identical to the driver's order.

The H->D gates were cut on the driver before ``f_minus_p``, where an integrated
electric source stood a withdraw in this seam, and four of their legs measured that
withdraw: ``driver_order`` required the withdraw loop between the consults, ``refusal``
required an engine integrated source to be refused by it, ``withdraw`` drove the hoist
and its two null controls around it, and ``lift`` required the rows the seam record
flags ``withdraw_in_seam`` to be refused. This module is what each of those legs reads
now, so the CUDA and the Triton gates make one claim in one way:

:func:`driver_seam_facts`
    the seam's statements read off ``driver.py`` by :func:`h_to_d_seam.driver_seam_fact`,
    which raises unless they are exactly the magnetic restore.
:func:`withdraw_refusal_checks`
    an engine integrated electric source is ADMITTED and stands no withdraw; a source
    class bringing a standing integrated withdraw of its own
    (:class:`ForeignStandingWithdraw`) is still REFUSED by name, because no launch was
    ever measured beside its body.
:func:`seam_restore_leg`
    the seam's one pass driven on the device: integrated electric AND magnetic sources
    under an active absorber, the weld force-installed, stepped in lockstep with its own
    array-path reference and required identical; and the same weld with the restore
    moved before the launch, and with the magnetic ``f_minus_p`` dropped, each required
    to diverge. Those two are the leg's planted defects: they are what a weld that read B
    on the wrong side of the restore would look like, and a leg in which neither
    diverged would prove nothing.
:func:`lift_withdraw_accounting`
    which lifted rows a predicate must refuse by a standing withdraw, DERIVED from each
    row's own recorded sources (a source whose withdraw does work), never from the
    predicate's answer; the rows the seam record flags ``withdraw_in_seam`` are admitted.
:func:`lifter_refusal`
    a corpus row the engine's own lifter refuses by name (``MeepSimulationNotLiftable``),
    which no arrangement can step, told apart from a hole.
:func:`plans_bind_the_driver_chunk_layout`
    on a LIFTED row, every Triton engine-route plan a gate builds binds the driver's own
    chunk division (MEEP's), the one the array path steps with, not the one-rank default.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# Sibling module: every consumer already has parity/meep_gpu on sys.path.
import h_to_d_seam

from meep_gpu import withdraw_hoist

#: The words every predicate's withdraw refusal carries, read by the refusal and lift
#: accounting below. A predicate that refused a foreign withdraw in other words would
#: be refusing it unnamed.
STANDING_WITHDRAW = "standing integrated"

#: The two orderings of the magnetic restore that the commutation does NOT license, each
#: run with the weld installed. Each must diverge from the driver's order, or the leg
#: cannot tell a weld that kept the restore's place from one that moved it.
SEAM_NULL_CONTROLS: Tuple[str, ...] = ("restore_before_the_launch", "no_magnetic_f_minus_p")

#: The exception the engine's lifter raises, by name, for a simulation it will not step.
LIFTER_REFUSAL = "MeepSimulationNotLiftable"

#: Integrated sources for :func:`seam_restore_leg`'s drivers, as ``FdtdDriver.add_source``
#: specs. Point sources off the cell's centre and off each other, so neither lands on a
#: mirror plane, and CONTINUOUS so neither switches off inside the budget. The magnetic
#: one is the whole reason the leg exists: its ``f_minus_p`` restore is the pass between
#: the two consults wherever H is stored.
INTEGRATED_ELECTRIC_SOURCE: Dict[str, Any] = {
    "component": "Ez", "center": (0.15, -0.1, 0.05), "size": (0.0, 0.0, 0.0),
    "frequency": 1.0, "source_type": "continuous", "is_integrated": True,
}
INTEGRATED_MAGNETIC_SOURCE: Dict[str, Any] = {
    "component": "Hy", "center": (-0.1, 0.15, -0.05), "size": (0.0, 0.0, 0.0),
    "frequency": 1.1, "source_type": "continuous", "is_integrated": True,
}


class ForeignStandingWithdraw:
    """A source class from outside this package that brings an integrated electric
    ``withdraw`` of its own.

    No engine source stands one any more, and the predicates still refuse a class that
    does, by name, because nothing measured a fused launch beside its body. Built to
    ``withdraw_hoist._withdraw_does_work``'s three conditions (a callable ``withdraw``,
    ``is_integrated``, a positive ``_n_source_points``) and electric by the driver's
    split, so a predicate that did not recognise it would make the refusal check pass
    vacuously -- the check therefore asks ``standing_withdraws`` too.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = True
    _n_source_points = 3

    def withdraw(self, fields: Any) -> None:  # noqa: ARG002 - never called by a predicate
        raise AssertionError("a coverage predicate called a source's withdraw")


def driver_seam_facts(driver_path: Optional[Any] = None) -> Dict[str, Any]:
    """The statements between the ``update_H`` and ``step_D`` consults, read off the tree.

    :func:`h_to_d_seam.driver_seam_fact` locates the two consults in ``FdtdDriver.step``
    and raises unless what sits between them is exactly the magnetic ``f_minus_p``
    restore; the raise is recorded as a failure with its text rather than allowed to end
    the gate. ``driver_path`` reads another copy of ``driver.py`` (the host tests plant
    a pass there).
    """
    try:
        fact = (h_to_d_seam.driver_seam_fact() if driver_path is None
                else h_to_d_seam.driver_seam_fact(driver_path))
    except SystemExit as error:
        return {"passed": False, "error": str(error)[:600]}
    statements = [entry["statement"] for entry in fact["between"]]
    record = {
        "seam_fact": fact,
        "statements_between_the_two_consults": statements,
        "the_seam_holds_only_the_magnetic_restore":
            tuple(statements) == h_to_d_seam.EXPECTED_BETWEEN,
        "a_withdraw_stands_between_the_two_consults":
            any("withdraw" in line for line in statements),
        "an_injection_stands_between_the_two_consults":
            any("inject" in line for line in statements),
        "the_restore_writes_B_alone_and_step_D_reads_none_of_it": bool(
            tuple(withdraw_hoist.SEAM_PASS.get("writes", ())) == ("Bx", "By", "Bz")
            and withdraw_hoist.SEAM_PASS.get("step_D_reads_none_of_it")),
    }
    record["passed"] = bool(record["the_seam_holds_only_the_magnetic_restore"]
                            and not record["a_withdraw_stands_between_the_two_consults"]
                            and not record["an_injection_stands_between_the_two_consults"]
                            and record["the_restore_writes_B_alone_and_step_D_reads_none_of_it"])
    return record


def withdraw_refusal_checks(ask: Callable[[Tuple[Any, ...]], Tuple[bool, Any]],
                            engine_integrated: Any) -> Dict[str, Dict[str, Any]]:
    """The seam clause of a predicate, asked twice.

    ``ask(sources)`` returns the predicate's ``(covered, reason)`` for a source tuple on
    a configuration it otherwise admits; ``engine_integrated`` is an engine integrated
    ELECTRIC source built on that configuration. The engine source must be admitted and
    stand no withdraw; :class:`ForeignStandingWithdraw` must be refused, by name.
    """
    def one(source: Any) -> Tuple[Optional[bool], str, Optional[str]]:
        try:
            covered, reason = ask((source,))
        except Exception as error:  # noqa: BLE001 - a raise is not a refusal by name
            return None, "", repr(error)[:300]
        return bool(covered), str(reason), None

    engine_covered, engine_reason, engine_error = one(engine_integrated)
    foreign = ForeignStandingWithdraw()
    foreign_covered, foreign_reason, foreign_error = one(foreign)
    engine_standing = withdraw_hoist.standing_withdraws((engine_integrated,))
    foreign_standing = withdraw_hoist.standing_withdraws((foreign,))
    return {
        "admits_an_engine_integrated_electric_source": {
            "covered": engine_covered, "expected": True, "reason": engine_reason[:300],
            "error": engine_error,
            "is_integrated": bool(getattr(engine_integrated, "is_integrated", False)),
            "standing_withdraws": len(engine_standing),
            "passed": bool(engine_covered is True and not engine_standing
                           and getattr(engine_integrated, "is_integrated", False)
                           and "withdraw" not in engine_reason),
        },
        "refuses_a_foreign_standing_integrated_electric_withdraw": {
            "covered": foreign_covered, "expected": False, "reason": foreign_reason[:300],
            "error": foreign_error,
            "standing_withdraws": len(foreign_standing),
            "names_the_reason": STANDING_WITHDRAW in foreign_reason,
            "passed": bool(foreign_covered is False and len(foreign_standing) == 1
                           and STANDING_WITHDRAW in foreign_reason),
        },
    }


def seam_restore_leg(*, build: Callable[[], Any],
                     install_weld: Callable[[Any], Mapping[str, Any]],
                     coverage: Callable[[Any], Tuple[bool, Any]],
                     state_of: Callable[[Any], Mapping[str, Any]],
                     compare: Callable[[Mapping[str, Any], Mapping[str, Any]],
                                       Mapping[str, int]],
                     synchronize: Callable[[], None], steps: int,
                     spec_label: str = "",
                     settle: Optional[Callable[[Any], None]] = None) -> Dict[str, Any]:
    """The seam's one pass on the device: the weld against the array path, and two
    planted orderings that must diverge.

    ``build()`` returns a fresh seeded driver carrying an integrated electric AND an
    integrated magnetic source under an active absorber (every call the same state);
    ``install_weld(driver)`` force-installs the product as the driver's fast path and
    returns a live book -- a mapping whose ``"launches"`` (or ``"runs"``) counts one per
    weld step, or a zero-argument callable returning that count;
    ``coverage(driver)`` is
    the product's predicate on the driver's own sources; ``state_of`` and ``compare`` are
    the gate's own stored-state snapshot and word comparator; ``settle(driver)``, when
    given, runs after every subject step (a gate whose plans rotate bindings settles
    them there, as its own ``drive`` does).

    Every arm is a subject driver stepped IN LOCKSTEP with its own array-path reference
    built from the same seed, compared after every complete step:

    ``weld``                       the driver's order: MUST be identical every step;
    ``restore_before_the_launch``  the magnetic bracket formed and restored before the
                                   update_H consult, so the launch reads B without the
                                   dipole: MUST diverge;
    ``no_magnetic_f_minus_p``      the magnetic bracket dropped: MUST diverge.
    """
    from meep_gpu.sources import (FIELD_TYPE_B, FIELD_TYPE_D,  # noqa: PLC0415
                                  f_minus_p_end, has_integrated)

    probe = build()
    sources = tuple(getattr(probe, "_sources", ()) or ())
    try:
        covered, reason = coverage(probe)
    except Exception as error:  # noqa: BLE001 - recorded, and it fails the leg
        covered, reason = False, f"raised: {error!r}"[:300]
    standing = withdraw_hoist.standing_withdraws(sources)
    stores_h = bool(probe._stores_H())  # noqa: SLF001
    magnetic = bool(has_integrated(sources, FIELD_TYPE_B))
    electric = bool(has_integrated(sources, FIELD_TYPE_D))
    seam = driver_seam_facts()

    def control(name: str, driver: Any) -> None:
        original = driver._f_minus_p_begin_B  # noqa: SLF001 - the bound method

        def restore_before_the_launch(fast: Any, time: float) -> Any:
            f_minus_p_end(driver.fields, original(fast, time))
            return None

        def no_magnetic_f_minus_p(fast: Any, time: float) -> Any:  # noqa: ARG001
            return None

        # An INSTANCE attribute shadows the method for this subject driver alone; the
        # reference beside it keeps the driver's own bracket.
        driver._f_minus_p_begin_B = {  # noqa: SLF001
            "restore_before_the_launch": restore_before_the_launch,
            "no_magnetic_f_minus_p": no_magnetic_f_minus_p}[name]

    arms: Dict[str, Any] = {}
    words = 0
    for arm in ("weld",) + SEAM_NULL_CONTROLS:
        reference = build()
        subject = build()
        book = install_weld(subject)
        if arm != "weld":
            control(arm, subject)
        first: Optional[Dict[str, Any]] = None
        compared = 0
        error: Optional[str] = None
        for step in range(steps):
            try:
                reference.step()
                subject.step()
                if settle is not None:
                    settle(subject)
                synchronize()
            except Exception as exc:  # noqa: BLE001 - recorded, and it fails the leg
                error = repr(exc)[:400]
                break
            left, right = state_of(reference), state_of(subject)
            moved = dict(compare(left, right))
            compared = step + 1
            words += sum(int(getattr(value, "size", 0)) for value in left.values())
            if moved:
                first = {"step": step + 1, "volumes": moved}
                break
        arms[arm] = {"first_divergence": first, "identical": first is None and error is None,
                     "steps_compared": compared, "step_error": error,
                     "launches": _launches(book)}
    weld = arms["weld"]
    controls = {name: arms[name]["first_divergence"] for name in SEAM_NULL_CONTROLS}
    record = {
        "spec": spec_label, "steps": steps,
        "sources": [{"component": getattr(source, "component", None),
                     "field_type": str(getattr(source, "field_type", "")),
                     "is_integrated": bool(getattr(source, "is_integrated", False))}
                    for source in sources],
        "an_integrated_electric_source_runs": electric,
        "an_integrated_magnetic_source_runs": magnetic,
        "predicate_admits_the_integrated_sources": bool(covered),
        "predicate_reason": str(reason)[:300],
        "standing_withdraws": len(standing),
        "H_is_stored_so_the_restore_stands_in_the_seam": stores_h,
        "driver_seam": {key: seam.get(key) for key in
                        ("statements_between_the_two_consults",
                         "the_seam_holds_only_the_magnetic_restore", "passed", "error")},
        "arms": arms,
        "weld_identical": weld["identical"],
        "weld_launched_every_step": weld["launches"] == steps,
        "null_controls_first_divergence": controls,
        "every_null_control_diverged": all(step is not None for step in controls.values()),
        "words_compared": words,
        "seam_pass": dict(withdraw_hoist.SEAM_PASS),
        "superseded_campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "this weld beside the 0.10 seam pass on this device: with integrated sources "
            "entering through f_minus_p, the launch spanning update_H and step_D leaves "
            "the magnetic restore after step_D and is byte-identical to the driver's "
            "order, while the restore moved before the launch and the magnetic "
            "f_minus_p dropped both diverge. It says nothing about a source class that "
            "brings a withdraw of its own, which the predicate refuses by name (the "
            "refusal leg)"),
    }
    record["passed"] = bool(
        covered and not standing and stores_h and magnetic and electric
        and seam.get("passed")
        and weld["identical"] and weld["steps_compared"] == steps
        and record["weld_launched_every_step"]
        and record["every_null_control_diverged"])
    return record


def _launches(book: Any) -> int:
    """A weld book's launch count: a zero-argument callable, or a mapping's
    ``"launches"`` (else ``"runs"``)."""
    if callable(book):
        return int(book() or 0)
    return int(book.get("launches", book.get("runs", 0)) or 0)


def electric_withdraw_sources(block: Mapping[str, Any]) -> List[int]:
    """Indices of the row's recorded sources that are electric and whose withdraw does
    work -- the sources a predicate must refuse the row by."""
    out: List[int] = []
    for index, source in enumerate(block.get("sources") or ()):
        source = source or {}
        if (bool(source.get("withdraw_does_work"))
                and str(source.get("field_type", "")) != withdraw_hoist.MAGNETIC_FIELD_TYPE):
            out.append(index)
    return out


def lift_withdraw_accounting(blocks: Mapping[str, Mapping[str, Any]],
                             seam_withdraw_rows: Iterable[str] = (),
                             reasons_of: Optional[Callable[[Mapping[str, Any]],
                                                           Sequence[str]]] = None,
                             tolerated: Optional[Callable[[str, Mapping[str, Any]],
                                                          bool]] = None,
                             ) -> Dict[str, Any]:
    """Which lifted rows the predicate must refuse by a standing withdraw, and whether it
    refused exactly those, by name.

    THE EXPECTED SET IS DERIVED FROM EACH ROW'S OWN RECORDED SOURCES -- a source whose
    ``withdraw_does_work`` is true and whose field type is not B -- never from the
    predicate's answer, so a predicate that refused too much or too little reads as a
    failure here. A scored row with no source list is ignorance, and ignorance is never
    an empty set: it fails. On the 0.10 engine no source's withdraw does work, so the
    rows the seam record flags ``withdraw_in_seam`` (cut on the driver before
    ``f_minus_p``) must be ADMITTED; a refusal naming a standing withdraw on any row
    that carries no such source is a failure.

    ``reasons_of(block)`` returns a block's refusal texts (default: its
    ``predicate_reasons`` list, else its ``predicate_reason``). ``tolerated(label,
    block)`` names a refusal the CALLER owns and has already accounted for (a policy
    refusal it scores by its own rule); such a row is reported, not scored here. Rows
    whose block has no ``predicate_admits`` are the caller's unmeasured rows and are not
    scored here.
    """
    def reasons(block: Mapping[str, Any]) -> List[str]:
        if reasons_of is not None:
            return [str(text) for text in reasons_of(block) or ()]
        listed = block.get("predicate_reasons")
        if listed:
            return [str(text) for text in listed]
        single = block.get("predicate_reason")
        return [str(single)] if single not in (None, "", "covered") else []

    scored = {label: block for label, block in blocks.items() if "predicate_admits" in block}
    no_sources = sorted(label for label, block in scored.items()
                        if not isinstance(block.get("sources"), list))
    expected = sorted(label for label, block in scored.items()
                      if electric_withdraw_sources(block))
    refused = sorted(label for label, block in scored.items()
                     if block.get("predicate_admits") is False)
    caller_owned = sorted(label for label in refused
                          if tolerated is not None and tolerated(label, scored[label])
                          and label not in expected)
    scored_refused = sorted(set(refused) - set(caller_owned))
    by_withdraw = sorted(label for label in scored
                         if any(STANDING_WITHDRAW in text for text in reasons(scored[label])))
    seam_rows = sorted(set(seam_withdraw_rows) & set(scored))
    seam_rows_refused = sorted((set(seam_rows) & set(refused)) - set(expected)
                               - set(caller_owned))
    return {
        "passed": bool(not no_sources and scored_refused == expected
                       and sorted(set(by_withdraw) - set(caller_owned)) == expected
                       and not seam_rows_refused),
        "rows_scored": len(scored),
        "expected_refused_from_the_row_sources": expected,
        "refused": refused,
        "refused_by_a_standing_withdraw": by_withdraw,
        "refused_by_a_refusal_the_caller_owns": caller_owned,
        "rows_without_a_source_list": no_sources,
        "rows_the_seam_record_flags_withdraw_in_seam": seam_rows,
        "of_those_refused_without_a_working_withdraw": seam_rows_refused,
        "rule": ("refused == the rows carrying an electric source whose withdraw does "
                 "work (read from the row's own sources); every such refusal names a "
                 "standing withdraw; the seam record's withdraw_in_seam rows, cut on the "
                 "driver before f_minus_p, are admitted"),
    }


def lifter_refusal(record: Mapping[str, Any]) -> Optional[str]:
    """The engine's own lifter refused this row BY NAME, or ``None``.

    ``meep_gpu.lift_simulation`` raises ``MeepSimulationNotLiftable`` with its reasons for
    a simulation it will not step (e.g. a BinaryPartition chunk layout beside mirror
    symmetry). Nothing runs on any path, so the row is no product's to measure; it is
    named with the lifter's reason and counted out of the denominator, never scored as a
    hole and never as a pass. Any other lift failure is not this, and stays a hole.
    """
    if not record.get("has_simulation"):
        return None
    for key in ("lift_error", "child_error"):
        text = str(record.get(key) or "")
        if text.startswith(LIFTER_REFUSAL + ":"):
            return text[:600]
    return None


@contextlib.contextmanager
def plans_bind_the_driver_chunk_layout(driver: Any) -> Iterator[Any]:
    """Every Triton engine-route plan built inside binds ``driver.chunk_layout``.

    A lifted corpus row steps with MEEP's own chunk division (``FdtdDriver.chunk_layout``)
    on the array path, and the kernels key their per-chunk forms on the record of that
    division. The Triton plan builders take the division as ``chunk_layout`` (``None``
    for the one-rank default, ``launch.chunk_bounds_for``), and the H->D weld builders
    (``plan_fused_hd_pair`` and its siblings, not dispatched) bind the one-rank default
    with no argument to pass. A gate building its arrangements from the engine's objects
    on a lifted row therefore bound the one-rank record in EVERY arrangement, and on a row
    whose division differs (measured 2026-10-10 on the folded rows
    ``absorbed_power_density.py`` and ``finite_grating.py``) all of them left the array
    path together -- the same harness defect the CUDA lift had in its certified singles.

    Inside this context ``launch.chunk_bounds_for`` answers a ``None`` division with the
    driver's (an explicit division is passed through unchanged), so a plan built here
    binds the record the array path steps with. Several kernel modules bind the function
    by name at import (``from .launch import chunk_bounds_for``), so every loaded
    ``meep_gpu.triton_kernels`` module holding the engine's function is pointed at the
    wrapper too, and on exit every module holding the wrapper -- one imported inside the
    context included -- gets the engine's function back. Plans bind their record at
    construction; the context is held across the drive so a plan built later binds it
    too. A driver built without MEEP carries the one-rank division, so a synthetic
    fixture binds exactly what it did before.
    """
    import sys  # noqa: PLC0415

    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    layout = getattr(driver, "chunk_layout", None)
    original = launch.chunk_bounds_for

    def chunk_bounds_for(fields: Any, pml: Any, chunk_layout: Any) -> Any:
        return original(fields, pml, layout if chunk_layout is None else chunk_layout)

    def modules() -> List[Any]:
        return [module for name, module in list(sys.modules.items())
                if module is not None and (name == "meep_gpu.triton_kernels"
                                           or name.startswith("meep_gpu.triton_kernels."))]

    for module in modules():
        if getattr(module, "chunk_bounds_for", None) is original:
            module.chunk_bounds_for = chunk_bounds_for
    try:
        yield layout
    finally:
        for module in modules():
            if getattr(module, "chunk_bounds_for", None) is chunk_bounds_for:
                module.chunk_bounds_for = original

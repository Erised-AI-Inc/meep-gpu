"""The METAL table: its dispatch ladder, its residency seam, and its release rows.

:mod:`meep_gpu.fastpath` decides WHICH KERNEL TABLE may replace array sub-steps.
On a CuPy engine that is the Triton table and the ladder stays in that file; on a
NumPy engine with an MPS device it is the Metal table, and the rungs below the
backend rung live HERE, as does everything that says which Metal products have been
driven through the driver's consults.

WHY THE METAL RELEASE ROWS ARE NOT IN ``fastpath.py``, which is the whole reason
this file exists rather than a section of that one. Every gate artifact in this
tree records its own provenance by walking ``sys.modules``
(``parity/meep_gpu/gate_provenance.provenance``), so a module a gate imports is
PINNED by that gate's artifact and any later edit to it re-drifts every record that
pinned it. ``fastpath.py`` is pinned by the Triton ``driver_dispatch`` record —
which pins exactly ``driver.py``, ``fastpath.py`` and ``fields.py`` — and by all 42
gate-bound Metal artifacts. Putting :data:`METAL_RELEASED_FUSED_ARMS` there would
mean that every future Metal release edit (a widened envelope, an off-diagonal
case, the group-key arms) re-cut the TRITON record too, for a change Triton has no
stake in. A NEW file is pinned by NO EXISTING artifact — verified against all 62
entries of ``metal_kernels/fingerprints.json`` and all 47 board bindings — so this
one costs nothing today, and from the next Metal campaign onward it is pinned by
the Metal artifacts and by them alone. That is the honest cost of any Metal-side
file and it is the cost this file is placed to pay.

WHAT THIS MODULE OWES ``fastpath`` AND WHAT IT BORROWS. It owes: the ladder
(:func:`decide`), the Metal tables below, and the residency bracket's installation.
It borrows, by function-local import and never re-implements: ``_axis_reasons``
(the four kinds of pinned axis), ``_run_shape``, ``_fold_description``,
``_refuse``, ``DRIVER_SLOTS``, ``FAR_FILL_PASSES``, the subnormal gate and the
factored tail ``_finish``. One definition of each decision, two tables it is
applied to.

THE SEAM THIS FILE WAITS ON. The rung-3 branch that reaches :func:`decide` and the
per-table plumbing beneath it are a separate batch on ``fastpath.py``. Until it
lands, nothing calls this module on the dispatch path and the symbols named below
are the join:

* ``fastpath.candidate_tables`` / ``fastpath.METAL_TABLE`` /
  ``fastpath.metal_hardware_present`` — the rung-3 branch that routes a NumPy
  engine with an MPS device here (from ``FdtdDriver``, an Apple GPU driver built
  with ``prefer_gpu=True``; a ``prefer_gpu=False`` driver is the NumPy reference and
  never reaches the planner);
* ``fastpath._finish`` — the factored tail, passed to :func:`decide` as ``finish``
  so both ladders emit through one implementation;
* ``fastpath.TABLE_SUBNORMAL_POLICY`` — ``{"triton": CERTIFICATION_SUBNORMAL_POLICY,
  "metal": "flush"}``; read here through :func:`certification_policy`, which falls
  back to this module's own :data:`SUBNORMAL_POLICY` while the map is absent;
* ``fastpath.TABLE_GOVERNED_EXECUTORS`` — the Metal row is
  :data:`GOVERNED_EXECUTORS`;
* ``fastpath._subnormal_gate(record, xp, table)`` and
  ``fastpath._drive_one_policy(record, xp, policy_module, table)`` — the ``table``
  argument rung 8bM passes;
* ``fastpath._certification_for(arm, table)`` and ``fastpath._fingerprints(table)``
  — the per-table lookup that reads :data:`ARM_CERTIFICATION` and
  ``metal_kernels/fingerprints.json`` for this table;
* ``fastpath._environment_block(grid, probe, probe_error, table)`` — the Metal half
  of which is :func:`environment_block` here.

IMPORTS NOTHING METAL AND NOTHING TORCH AT MODULE LEVEL. Every reference to
``metal_kernels`` and to ``torch`` below is function-local, so importing this
module on a machine with neither costs nothing and pulls in nothing —
``test_package_boundary`` measures exactly that kind of needless pull-in.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import sys
from contextlib import ExitStack, contextmanager
from typing import (Any, Dict, FrozenSet, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

#: The name this table answers to in ``record["table"]`` and in every per-table
#: mapping. The twin of ``fastpath.METAL_TABLE``, which is what rung 3 compares.
TABLE = "metal"

#: The float32 subnormal policy the Metal table certifies under, and the second
#: relaxation of the one-policy rule — NOT an exemption from it.
#:
#: THE RULE IS UNCHANGED: dispatch never runs two executors under disagreeing
#: policies. What changes is the VALUE, per table, because the value has to be one
#: BOTH executors can attain. MPS flushes denormals natively and exposes no lever
#: at all (``metal_kernels/subnormal.py``: both denormal pragma spellings are
#: compile errors on this toolchain), so ``keep`` is unattainable on the device
#: half; the arm64 host, whose MEEP knob is the ``#if HAVE_IMMINTRIN_H`` no-op, is
#: driven to flush by ``subnormal_policy``'s fenv lever. Under that one policy the
#: whole run is uniform.
#:
#: WHAT A KEEPING HOST WOULD COST, MEASURED 2026-09-10 rather than argued: the
#: released pairs are byte-identical to the array path at step 12 and DIVERGE at
#: step 24 (pml_2d 12 arrays / 108 words on ``fields.Ez``; folded_2d 12 arrays / 48
#: words), with every differing element inside a few ULP of the subnormal band
#: (largest differing magnitude 2.3e-31 against a smallest normal of 1.18e-38). A
#: Metal dispatch under host-keep would therefore be a certified 12-step
#: composition and a documented divergence forever after.
SUBNORMAL_POLICY = "flush"

#: The executors rung 8bM drives and verifies for this table. No ``cupy`` and no
#: ``triton``: neither is present in a Metal dispatch, and probing for them would
#: make an absent optional dependency into a refusal.
GOVERNED_EXECUTORS: Tuple[str, str] = ("host", "mps")

#: THE DRIVER-ROUTE GATE THIS TABLE'S RELEASE IS CUT FROM, named BEFORE the
#: campaign runs and not after it.
#:
#: THAT ORDER IS THE TRAP, and the Triton constant records paying for it: every
#: source edit of a round must land BEFORE the round's campaign starts, and editing
#: this line is itself such an edit — so a run that discovers the constant is wrong
#: has to be repeated, not repointed. The campaign driver
#: (``parity/meep_gpu/run_metal_dispatch_campaign.sh``) writes its legs under
#: ``results/<this name>_<architecture>/`` (see the 2026-10-05 entry below), and
#: ``recut_driver_dispatch_record.py --backend metal`` refuses to cut the record
#: unless the directory it reads is the one this constant spells for the
#: architecture its legs recorded.
#: REPOINTED 2026-09-15, BEFORE the relaunch, which is the order the paragraph above
#: demands. The ``2026-09-14_weldgrid`` campaign died with its host at 44 of 62 fleet
#: gates, before its dispatch stage, and it had been launched while this line still
#: named ``_target`` -- so it could not have been cut even had it finished.
# ``_2026-09-15_offdiag``: the ``_2026-09-15_weldgrid`` campaign was stopped part-way
# through its fleet, because the Triton off-diagonal fix moved fastpath.py, which
# every Metal gate records.
# ``_2026-09-15_gapclose``: one Triton batch moved fastpath.py again -- the release of
# ``fused pair D (complex conductive no-PML)`` and the lift of ``complex folded
# off-diagonal`` with the widening of ``fused pair B (folded complex)`` -- which every
# Metal gate and this table's record bind, so the fleet and the dispatch stage re-run
# under a fresh name. The ``_2026-09-15_offdiag`` campaign is kept as the record of the
# tree before it.
#: REPOINTED 2026-09-17 to ``_2026-09-17_envelope``, BEFORE the campaign runs, for the
#: batch that moved ``pml_active`` and ``bfast`` off the shared envelope and onto 29
#: per-arm pins (16 added, values read off each arm's own release cases). The
#: constant had read ``_2026-09-15_gapclose`` through the whole 2026-09-17_allpaths
#: campaign, so that campaign's record could never have bound; this one can.
#: ``_2026-09-17_envelope`` ran and refused on three of four legs, on two defects of
#: the ROUTE GATE's instrument rather than of any product: its launch witness never
#: wrapped the real-cylindrical plans' function tables (``cylindrical``,
#: ``cylindrical_m0`` VACUOUS-PASS on kernels that ran), and its withheld-consult
#: control had no NOT-APPLICABLE rung for an inactive layer (``absorber_1d``,
#: ``no_pml_dispersive_2d`` NULL on an array-against-array comparison). It stays as
#: the record of that. REPOINTED 2026-09-19 to ``_2026-09-19_witness`` for the
#: re-run with both repaired.
#: ``_2026-09-19_witness`` refused ONE case on its probe leg, and on the route gate's
#: EXPECTATION rather than on any product: ``complex_no_pml_3d`` (first fused on this
#: round) read DROPPED-BUT-NOT-EXACT on a drop of 3.0 against an expected 1, because
#: its unfused ``complex no-PML stored E`` single launches once per component (1800
#: over 600 steps) and the pair folds all three into its one launch -- 8.0 -> 5.0 a
#: step, compiled calls agreeing, byte-identical to the array path. The gate's
#: "one launch fewer per pair" was the N = 1 case of ``pairs + sum(N - 1)``; that one
#: drive row now DECLARES ``collapsed_single_launches_per_step = {"update_E": 3}``,
#: cross-checked against the unfused leg's own counter (``DECLARATION-MISMATCH``
#: refuses). Replayed over every stored row of that campaign's four case legs, it is
#: the only verdict that moves. REPOINTED 2026-09-19 to ``_2026-09-19_components``,
#: before that campaign ran; it released every leg (7.65 h) and cut the 327 / 597 board.
#: REPOINTED 2026-09-20 to ``_2026-09-20_restrict``, before that campaign runs: the
#: deposit repair this table brackets its source-seam pairs with (host NumPy here) now
#: repairs only the component the source writes, bit-identical, so every bracketed row
#: executes different bytes and the fleet and the route are re-run on them. The campaign
#: runs its three long legs as concurrent lanes for the first time; the gate makes no
#: timing claim, and each leg remains one process with its own policy and environment.
#: REPOINTED 2026-09-27 to ``_2026-09-27_flip``, before that campaign runs: dispatch
#: is on by default and held residency is this table's default, with the release at
#: every freeze, the ``reset`` layer and the admission rung, so every leg executes
#: different bytes and the fleet and the route are re-run on them.
#: REPOINTED 2026-09-30 to ``_2026-09-30_091`` for release 0.9.1, whose certification
#: round re-runs the fleet and the route on the released files.
#: REPOINTED 2026-10-02 to ``_2026-10-02_arch3``, before that campaign runs: the
#: 0.9.1 Metal round never ran, and this one is the first whose welds record the GPU
#: architecture each ran on, which the route record requires every row to read
#: certified. (``_2026-10-02_arch`` was abandoned during its fleet: its lift legs
#: found no MEEP corpus. ``_2026-10-02_arch2`` was killed 40 gates in, every one
#: released, when the shell that started it exited.)
#: REPOINTED 2026-10-05 to ``_2026-10-05_perarch``, before that campaign runs: the
#: admission reads the per-architecture run records and the route record keeps one
#: run per GPU architecture, so this file moved and the route is re-run on it. THE
#: CONSTANT IS NOW THE STAMP, not the directory: each architecture's campaign writes
#: its legs under ``results/<metal_runs.route_campaign(this, architecture)>/``
#: (``dispatch_metal_route_2026-10-05_perarch_applegpu_g13s``), the way the NVIDIA
#: constants name ``<stamp>_cc86``, so a second Mac drives the same release on the
#: same bytes without editing this line.
METAL_DRIVER_ROUTE_GATE = "dispatch_metal_route_2026-10-05_perarch"

#: The four expansion-probe families, as ``(plan_step keyword, module)``. Each
#: module owns its own ``PROBE_PATH_ENVIRONMENT`` and ``load_expansion_probe``; the
#: four are SEPARATE because the four families require different pattern sets and
#: one shared artifact would let a record that classified five orientations licence
#: a kernel that performs seven (``metal_kernels/launch.plan_step``).
PROBE_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("complex_probe", "complex_fields"),
    ("beta_probe", "special_kz"),
    ("folded_complex_probe", "folded_complex"),
    ("cylindrical_complex_probe", "cylindrical_complex"),
)


# ---------------------------------------------------------------------------
# The ledger this table quotes
# ---------------------------------------------------------------------------

_FINGERPRINTS: Optional[Mapping[str, Any]] = None


def fingerprints() -> Mapping[str, Any]:
    """``metal_kernels/fingerprints.json``, read once and cached.

    The Metal twin of ``fastpath._fingerprints``, kept here so this module can
    check its own rows (every :data:`ARM_CERTIFICATION` gate must resolve) and
    derive :func:`cited_environments` without waiting on the per-table argument
    ``fastpath._fingerprints`` is due to grow.
    """
    global _FINGERPRINTS
    if _FINGERPRINTS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "metal_kernels", "fingerprints.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _FINGERPRINTS = json.load(handle)
        except Exception:  # noqa: BLE001 - an unreadable ledger is not a refusal
            _FINGERPRINTS = {}
    return _FINGERPRINTS


#: How a Metal weld's run records the machine it ran on: ``"this machine: Apple MPS
#: device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10"``. Parsed rather than
#: typed, for the reason ``fastpath.validated_triton_versions`` is read off the
#: Triton ledger: the certified set is a fact about which gates have run, and a
#: hand-kept copy of it drifts the moment one does. The writers parse the line they
#: write (``metal_runs.environment_of``) to key the run by its architecture and to
#: record its torch and frontend beside it; the admission reads those fields
#: (:func:`cited_environments`). Lines written before the GPU architecture was
#: stamped name no ``applegpu_`` token, and that fact then reads as not recorded.
_HOST_ARCHITECTURE = re.compile(r"\b(applegpu_[A-Za-z0-9_]+)")
_HOST_TORCH = re.compile(r"torch\s+([^\s,]+)")
_HOST_FRONTEND = re.compile(r"metalfe-([^\s,]+)")

#: THE THREE FACTS THAT MAKE A METAL ENVIRONMENT, each of which changes the code
#: that runs: the GPU architecture is the unit Metal compiles for, torch supplies
#: ``mps.compile_shader`` and picks the language version, and the Metal frontend
#: (one per macOS build, whatever the GPU) turns the source into instructions.
ENVIRONMENT_FACTS: Tuple[str, ...] = ("architecture", "torch", "metal_frontend")

#: torch's switch that compiles every Metal source, ``compile_shader`` included, in
#: fast-math mode. Measured 2026-10-02 (``parity/meep_gpu/probe_metal_fast_math.py``,
#: M1 Max, torch 2.10.0): ``1`` changed 274,523 of 1,048,576 float32 divide words
#: and 327,338 square roots; ``0`` changed none. Every weld ran with it unset, so a
#: run with any value but ``0`` is outside the certified environment.
FAST_MATH = "PYTORCH_MPS_FAST_MATH"


def frontend_key(value: Optional[str]) -> Optional[str]:
    """One spelling of a Metal frontend version, whichever spelling arrived.

    TWO SPELLINGS OF ONE FACT, AND COMPARING THEM RAW REFUSED THE ONLY CERTIFIED
    HOST. ``device.metal_frontend_version()`` returns the string the shipped
    compiler library reports — ``"metalfe-32023.850.10"``, prefix included — while
    the ledger's ``host`` line embeds the same value and :data:`_HOST_FRONTEND`
    captures only what follows the prefix, ``"32023.850.10"``. Measured 2026-09-10:
    rung 4M therefore answered ``frontend_certified: False`` on this machine, the
    one every Metal weld in the ledger ran on, and refused the whole table with
    "Metal frontend metalfe-32023.850.10 is not in the toolchains any Metal weld
    recorded running on [['2.10.0', None], ['2.10.0', '32023.850.10']]" — a refusal
    whose own message shows the two halves are the same version.

    Normalised on the READ side only. What the artifact records stays verbatim: an
    environment block should say what the host said, and a comparison should not
    depend on which of two writers formatted it.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("metalfe-"):
        text = text[len("metalfe-"):]
    return text or None


def host_environment(host: Any) -> Dict[str, Optional[str]]:
    """The Metal environment one run's ``host`` line records.

    A fact the line does not name is ``None``, and so is one it names as ``?`` or
    ``unknown``: a record that could not tell is not a record of a value. The
    frontend is normalised through :func:`frontend_key`.
    """
    text = host if isinstance(host, str) else ""

    def first(pattern: Any) -> Optional[str]:
        match = pattern.search(text)
        value = None if match is None else match.group(1)
        return None if value in (None, "?", "unknown") else value

    return {"architecture": first(_HOST_ARCHITECTURE),
            "torch": first(_HOST_TORCH),
            "metal_frontend": frontend_key(first(_HOST_FRONTEND))}


def _run_environment(architecture: str, run: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """The environment one per-architecture run records: its key, torch and frontend."""
    torch_version = run.get("torch")
    return {"architecture": architecture,
            "torch": None if torch_version in (None, "?", "unknown") else str(torch_version),
            "metal_frontend": frontend_key(run.get("metal_frontend"))}


def cited_environments(ledger: Optional[Mapping[str, Any]] = None,
                       gates: Optional[Sequence[str]] = None
                       ) -> Tuple[Tuple[str, Optional[Tuple[Dict[str, Optional[str]], ...]]],
                                  ...]:
    """``(gate, runs)`` for every weld :data:`ARM_CERTIFICATION` cites.

    ``runs`` is the environment of each LIVE run the weld's entry records under
    ``runs[<architecture>]``, in architecture order: the architecture it is keyed by,
    the torch and the Metal frontend it recorded. A run is live while its
    ``bound_sha256`` is ``fastpath.bound_digest`` of the entry, and
    ``fastpath.live_capabilities`` is the one function that answers that, for this
    table as for the NVIDIA ones. An entry whose bytes moved after a run was cut no
    longer offers that run.

    ``runs`` is ``None`` when the entry cannot be read as per-architecture runs:
    the gate has no entry, or the entry holds no ``runs`` mapping (the one-run shape
    the per-architecture records replaced, which is not read both ways). Such a weld
    recorded nothing this ladder can judge: every verdict it takes part in is
    ``None``, so by default the plan dispatches with ``certified: None`` and prints no
    NOTE line, and only ``MEEP_GPU_ALLOW_UNCERTIFIED=0`` refuses it. What catches a
    ledger in that shape is the test suite
    (``test_the_shipped_ledger_cites_a_readable_entry_for_every_gate``), not the run;
    :func:`unresolved_certification_rows` refuses a row with no entry at all.

    THE CERTIFIED SET IS READ FROM THE CITED WELDS ONLY. An entry no dispatched arm
    quotes certifies nothing this table runs, so its runs cannot widen what counts as
    certified. ``ledger`` defaults to :func:`fingerprints` and ``gates`` to the welds
    :data:`ARM_CERTIFICATION` cites; the migration tool passes the document it is
    about to write, to check entry by entry the admission it would produce.
    """
    from . import fastpath  # noqa: PLC0415

    ledger = fingerprints() if ledger is None else ledger
    gates = sorted({gate for _family, gate in ARM_CERTIFICATION.values()}
                   if gates is None else set(gates))
    out = []
    for gate in gates:
        entry = ledger.get(gate) if isinstance(ledger, Mapping) else None
        runs = entry.get(fastpath.RUNS) if isinstance(entry, Mapping) else None
        if not isinstance(runs, Mapping):
            out.append((gate, None))
            continue
        out.append((gate, tuple(_run_environment(architecture, runs[architecture])
                                for architecture in fastpath.live_capabilities(entry))))
    return tuple(out)


def recorded_environments(ledger: Optional[Mapping[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
    """The distinct environments the cited welds' live runs recorded, with their weld counts.

    What a reader is shown beside a verdict: one row per distinct
    ``(architecture, torch, metal_frontend)`` among the live runs of the cited welds,
    ``welds`` counting the cited welds with a live run that recorded it, most common
    first. A cited weld that cannot be read as per-architecture runs is one row of
    ``None`` facts. A weld with runs for two architectures counts in two rows, so the
    counts need not sum to the number of cited welds.
    """
    counts: Dict[Tuple[Optional[str], ...], int] = {}
    for _gate, runs in cited_environments(ledger=ledger):
        environments = ([{fact: None for fact in ENVIRONMENT_FACTS}] if runs is None
                        else runs)
        for key in {tuple(environment[fact] for fact in ENVIRONMENT_FACTS)
                    for environment in environments}:
            counts[key] = counts.get(key, 0) + 1
    return [dict(zip(ENVIRONMENT_FACTS, key), welds=count)
            for key, count in sorted(counts.items(),
                                     key=lambda item: (-item[1], repr(item[0])))]


def environment_judgements(fact: str, value: Optional[str],
                           architecture: Optional[str] = None, *,
                           ledger: Optional[Mapping[str, Any]] = None
                           ) -> Tuple[Tuple[str, Optional[bool]], ...]:
    """``(gate, verdict)`` per cited weld: does that weld's record certify ``value``?

    * ``architecture``: ``True`` when the weld has a live run keyed by ``value``;
      ``False`` when it can be read and has none (it never ran on this GPU on these
      bytes, or that run was superseded); ``None`` when it cannot be read.
    * ``torch`` / ``metal_frontend``: judged against the weld's live run for
      ``architecture`` (this host's), because the facts are recorded inside each
      architecture's run. ``None`` when the weld has no such run or the run does not
      record the fact. With ``architecture`` unread, against every live run the weld
      holds: ``True`` when all recorded ``value``, ``False`` when one recorded another.
    """
    wanted = frontend_key(value) if fact == "metal_frontend" else value
    out = []
    for gate, runs in cited_environments(ledger=ledger):
        verdict: Optional[bool]
        if wanted is None or runs is None:
            verdict = None
        elif fact == "architecture":
            verdict = any(run["architecture"] == wanted for run in runs)
        else:
            if architecture is not None:
                candidates = [run for run in runs if run["architecture"] == architecture]
            else:
                candidates = list(runs)
            seen = [run[fact] for run in candidates]
            if any(recorded is not None and recorded != wanted for recorded in seen):
                verdict = False
            elif seen and all(recorded == wanted for recorded in seen):
                verdict = True
            else:
                verdict = None
        out.append((gate, verdict))
    return tuple(out)


def environment_verdict(fact: str, value: Optional[str],
                        architecture: Optional[str] = None, *,
                        ledger: Optional[Mapping[str, Any]] = None) -> Optional[bool]:
    """Is ``value`` the ``fact`` EVERY cited weld certifies? THREE-VALUED.

    The intersection rule of ``fastpath.capability_report``, per fact
    (:func:`environment_judgements`): ``True`` when every cited weld certifies it;
    ``False`` when one can be read and does not, so the table as a whole is not
    certified here; ``None`` when the value was not read, when no weld is cited, or
    when some cited welds cannot judge it and none contradicts it: the records
    cannot tell, and saying ``True`` would certify what no gate recorded.

    So this host's GPU architecture is certified when every cited weld has a live
    run for it, and its torch and Metal frontend when every such run recorded this
    host's. ``architecture`` is this host's, for judging torch and the frontend.
    """
    if value is None:
        return None
    verdicts = [verdict for _gate, verdict in environment_judgements(
        fact, value, architecture, ledger=ledger)]
    if not verdicts:
        return None
    if any(verdict is False for verdict in verdicts):
        return False
    if all(verdict is True for verdict in verdicts):
        return True
    return None


def certified_values(fact: str, architecture: Optional[str] = None, *,
                     ledger: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The values of ``fact`` the cited welds CERTIFY, as a NOTE line or a refusal names them.

    A candidate is a value some live run of a cited weld recorded: for torch and the
    frontend, a run for ``architecture`` (this host's) where any weld has one, else
    any run. It is listed only when :func:`environment_verdict` says ``True`` for it,
    so the list obeys the intersection rule the verdict obeys. During a partial round
    an architecture some cited welds ran on and others did not is therefore NOT
    listed: naming every architecture any weld recorded would print
    ``certified: applegpu_g13s, applegpu_g15s`` on the very GPU the verdict has just
    found not certified (measured on a ledger holding the second architecture on 44
    of 45 cited welds).
    """
    rows = recorded_environments(ledger=ledger)
    if fact in ("torch", "metal_frontend") and any(
            row["architecture"] == architecture for row in rows):
        rows = [row for row in rows if row["architecture"] == architecture]
    candidates = sorted({str(row[fact]) for row in rows if row[fact] is not None})
    return [value for value in candidates if environment_verdict(
        fact, value, None if fact == "architecture" else architecture,
        ledger=ledger) is True]


def certification_policy() -> str:
    """The policy this table certifies under, preferring ``fastpath``'s own map.

    WAITS ON ``fastpath.TABLE_SUBNORMAL_POLICY``. Once that map exists it is the
    single source and this function reports it; until then :data:`SUBNORMAL_POLICY`
    is the only statement there is. The two must agree, and the join is one line in
    the fastpath batch — ``{"triton": CERTIFICATION_SUBNORMAL_POLICY, "metal":
    "flush"}`` — so a disagreement is a typo rather than a decision.
    """
    from . import fastpath  # noqa: PLC0415

    table_policy = getattr(fastpath, "TABLE_SUBNORMAL_POLICY", None)
    if isinstance(table_policy, Mapping) and TABLE in table_policy:
        return str(table_policy[TABLE])
    return SUBNORMAL_POLICY


# ---------------------------------------------------------------------------
# Which certification each dispatched Metal arm rides on
# ---------------------------------------------------------------------------

#: Arm label -> ``(family, the key in metal_kernels/fingerprints.json)``.
#:
#: A ROW HERE IS A PROMISE THAT THE LOOKUP RESOLVES, exactly as it is on the Triton
#: table, and ``test_metal_dispatch`` holds it to that with no exemption: an
#: artifact whose dispatched arm names a gate that resolves to nothing is the
#: over-claim this map exists to prevent. The family names are the ARM families
#: (``metal_kernels.arms`` registers them); several arms share one gate because the
#: gate certifies a wider unit than the arm — ``PML`` and ``ordinary`` are both cut
#: by ``metal_pml_device_gate``, ``mirror fill`` and ``folded`` by
#: ``metal_symmetry_device_gate`` — and the row says which family the arm belongs
#: to so the two facts stay separable.
#:
#: THE SINGLE ARMS ARE HERE AND MOST OF THE FUSED LABELS ARE NOT, and the split is
#: the same one the Triton table makes. A single arm can win a slot on any
#: dispatching case, so every one of them needs a row. A fused label gets a row only
#: when it can DISPATCH: the FOURTEEN below are the ones :data:`METAL_RELEASED_FUSED_ARMS`
#: releases — five of them on a GROUP key, and the ruling is written on the rows —
#: and every other registry weld label is refused by name at clause 8M before any
#: certification is looked up. :data:`METAL_PENDING_DEVICE_GATE_ARMS` has been EMPTY
#: since 2026-09-12: the last two families it held were released on the same ruling.
ARM_CERTIFICATION: Mapping[str, Tuple[str, str]] = {
    # --- THE ELEVEN ARMS THE ALL-PATHS ROUND RELEASES, 2026-09-17. Each gate name is
    # DERIVED rather than transcribed: it is the ledger entry whose pin set contains
    # that family's own module, read out of metal_kernels/fingerprints.json. Four of
    # the eleven share one entry, which is correct -- metal_residue_fused_pairs and
    # metal_tranche7_fused_pairs each certified several products in one device round.
    "BFAST fused electric D/E pair": (
        "bfast_fused_electric_pair", "metal_residue_fused_pairs_device_gate"),
    "complex-beta fused electric D/E pair": (
        "beta_complex_fused_electric_pair", "metal_residue_fused_pairs_device_gate"),
    "complex-beta fused magnetic B/H pair": (
        "beta_complex_fused_magnetic_pair", "metal_residue_fused_pairs_device_gate"),
    "conductive no-PML fused electric D/E pair": (
        "no_pml_conductive_fused_electric_pair",
        "metal_no_pml_fused_electric_pairs_device_gate"),
    "cylindrical complex fused magnetic B/H pair": (
        "cylindrical_fused_magnetic_pair",
        "metal_cylindrical_fused_magnetic_pair_device_gate"),
    "cylindrical m=0 fused magnetic B/H pair": (
        "cylindrical_real_fused_magnetic_pair",
        "metal_cylindrical_real_fused_magnetic_pair_device_gate"),
    "folded beta complex fused B/H pair": (
        "folded_beta_complex_fused_magnetic_pair",
        "metal_folded_beta_complex_fused_magnetic_pair_device_gate"),
    "folded beta complex fused D/E pair": (
        "folded_beta_complex_fused_pair", "metal_tranche7_fused_pairs_device_gate"),
    "folded beta real fused B/H pair": (
        "folded_beta_real_fused_magnetic_pair",
        "metal_tranche7_fused_pairs_device_gate"),
    "folded beta real fused D/E pair": (
        "folded_beta_real_fused_pair", "metal_tranche7_fused_pairs_device_gate"),
    "folded off-diagonal fused electric D/E pair": (
        "folded_offdiag_fused_electric_pair",
        "metal_offdiag_stencil_welds_device_gate"),
    "off-diagonal fused electric D/E pair": (
        "offdiag_fused_electric_pair", "metal_offdiag_stencil_welds_device_gate"),
    "complex no-PML off-diagonal fused electric D/E pair": (
        "complex_no_pml_offdiag_fused_electric_pair",
        "metal_offdiag_stencil_welds_device_gate"),
    "no-PML fused electric D/E pair": (
        "no_pml_fused_electric_pair",
        "metal_no_pml_fused_electric_pairs_device_gate"),
    "fused complex conductive no-PML curl -> complex stored E": (
        "complex_conductive_fused_pair",
        "metal_complex_conductive_fused_pair_device_gate"),
    # --- the single arms, one row each, every gate read off the ledger ---------
    "ADE update_P": ("ade_update_p", "metal_ade_update_p_device_gate"),
    "BFAST": ("bfast_curl", "metal_bfast_device_gate"),
    "PML": ("pml_curl", "metal_pml_device_gate"),
    "complex conductive PML constitutive": ("complex_conductive_pml",
                                            "metal_complex_conductive_pml_device_gate"),
    "complex conductive PML curl": ("complex_conductive_pml",
                                    "metal_complex_conductive_pml_device_gate"),
    "complex conductive no-PML curl": ("complex_no_pml_conductive",
                                       "metal_complex_no_pml_conductive_device_gate"),
    "complex dispersive PML E": ("complex_dispersive_update_e",
                                 "metal_complex_dispersive_update_e_device_gate"),
    "complex dispersive PML curl": ("complex_dispersive_spine",
                                    "metal_complex_dispersive_update_e_device_gate"),
    "complex dispersive PML magnetic": ("complex_dispersive_spine",
                                        "metal_complex_dispersive_update_e_device_gate"),
    "complex folded off-diagonal PML E": ("complex_folded_offdiag_update_e",
                                          "metal_complex_folded_offdiag_device_gate"),
    "complex no-PML curl": ("complex_no_pml_curl",
                            "metal_complex_no_pml_curl_device_gate"),
    "complex no-PML off-diagonal": ("complex_no_pml_offdiag_update_e",
                                    "metal_complex_no_pml_offdiag_update_e_device_gate"),
    "complex no-PML stored E": ("complex_no_pml_stored_e",
                                "metal_complex_no_pml_stored_e_device_gate"),
    "complex/Bloch": ("complex_fields", "metal_complex_device_gate"),
    "conductive PML curl": ("conductive_pml", "metal_conductive_pml_device_gate"),
    "conductive no-PML curl": ("no_pml_conductive",
                               "metal_no_pml_conductive_device_gate"),
    "cylindrical complex": ("cylindrical_complex",
                            "metal_cylindrical_complex_device_gate"),
    "cylindrical m=0": ("cylindrical_real", "metal_cylindrical_real_device_gate"),
    "dispersive PML E": ("dispersive_update_e",
                         "metal_dispersive_update_e_device_gate"),
    "folded": ("folded", "metal_symmetry_device_gate"),
    "folded beta complex": ("folded_beta_complex", "metal_folded_beta_device_gate"),
    "folded beta real": ("folded_beta_real", "metal_folded_beta_device_gate"),
    "folded complex": ("folded_complex", "metal_folded_complex_device_gate"),
    "folded complex fill": ("folded_complex", "metal_folded_complex_device_gate"),
    "folded dispersive PML E": ("folded_dispersive_update_e",
                                "metal_folded_dispersive_update_e_device_gate"),
    "folded off-diagonal dispersive PML E": (
        "folded_offdiag_dispersive_update_e",
        "metal_folded_offdiag_dispersive_device_gate"),
    "folded offdiag": ("folded_offdiag_constitutive",
                       "metal_folded_offdiag_device_gate"),
    "mirror fill": ("folded", "metal_symmetry_device_gate"),
    "no-PML curl": ("no_pml_curl", "metal_no_pml_curl_device_gate"),
    "no-PML null": ("no_pml_constitutive", "metal_no_pml_constitutive_device_gate"),
    "no-PML stored E": ("no_pml_stored_e", "metal_no_pml_stored_e_device_gate"),
    "nonlinear": ("nonlinear_constitutive", "metal_nonlinear_device_gate"),
    "nonlinear PML curl": ("nonlinear_constitutive", "metal_nonlinear_device_gate"),
    "nonlinear PML magnetic": ("nonlinear_constitutive",
                               "metal_nonlinear_device_gate"),
    "offdiag": ("offdiag_constitutive", "metal_offdiag_device_gate"),
    "ordinary": ("constitutive", "metal_pml_device_gate"),
    "special_kz complex beta": ("special_kz_complex", "metal_special_kz_device_gate"),
    "special_kz real beta": ("special_kz_real", "metal_special_kz_device_gate"),
    # --- the eight released fused products, each with its OWN family weld ------
    "fused magnetic B/H pair": ("fused_magnetic_pair",
                                "metal_fused_magnetic_pair_device_gate"),
    "fused electric D/E pair": ("fused_electric_pair",
                                "metal_fused_electric_pair_device_gate"),
    "fused dispersive D/E pair": ("fused_dispersive_pair",
                                  "metal_fused_dispersive_pair_device_gate"),
    "folded fused B/H pair": ("folded_fused_magnetic_pair",
                              "metal_folded_fused_magnetic_pair_device_gate"),
    "folded fused D/E pair": ("folded_fused_pair",
                              "metal_folded_fused_pair_device_gate"),
    "complex fused magnetic B/H pair": (
        "complex_fused_magnetic_pair",
        "metal_complex_fused_magnetic_pair_device_gate"),
    "folded complex fused B/H pair": (
        "folded_complex_fused_magnetic_pair",
        "metal_folded_complex_fused_magnetic_pair_device_gate"),
    "cylindrical m=0 fused electric D/E pair": (
        "cylindrical_real_fused_electric_pair",
        "metal_cylindrical_real_fused_electric_pair_device_gate"),
    # --- the three cited by a GROUP weld, ruled 2026-09-12 ---------------------
    # THE RULING AND ITS EVIDENCE, because a group key is a weaker-LOOKING citation
    # and the next reader is owed the reason it is not a weaker one. These three
    # families have no per-family weld; each is bound by a group weld that is PASS.
    # What was checked before citing it:
    #   * the group weld's `code_sha256` is a PER-FILE map and it carries the
    #     family's OWN kernel (conductive_fused_electric_pair.py,
    #     folded_fused_dispersive_pair.py, beta_fused_electric_pair.py), so the
    #     binding is per-file exactly as a per-family weld's is; and
    #   * the group ARTIFACT carries a per-family verdict under `families[<name>]` —
    #     `identity` bit-identical on every case, `mutation` with 3-5 mutations
    #     CAUGHT, `deposit` repaired, `passed: true` — which is the same structure a
    #     per-family gate emits, under a different filename.
    # A group key is therefore a citation of the same evidence, not less of it. What
    # it does NOT buy is a per-family ARTIFACT NAME, so a future reader tracing one
    # family lands in a directory holding seven; that is the whole cost and it is
    # recorded here rather than discovered.
    "conductive fused electric D/E pair": ("conductive_fused_electric_pair",
                                           "metal_residue_fused_pairs_device_gate"),
    "folded fused dispersive D/E pair": ("folded_fused_dispersive_pair",
                                         "metal_tranche7_fused_pairs_device_gate"),
    "beta fused electric D/E pair": ("beta_fused_electric_pair",
                                     "metal_tranche7_fused_pairs_device_gate"),
    # --- two more on the SAME group weld, released later on 2026-09-12 ---------
    # The same two facts were checked before citing it: the group weld's
    # code_sha256 map carries complex_fused_electric_pair.py and
    # folded_complex_fused_pair.py, and the group artifact
    # (metal_tranche7_fused_pairs_2026-09-11_dispatch/gate.json) carries
    # families[<name>] for both — identity bit-identical, declared equivalences
    # passed, deposit repaired, passed: true. They waited one campaign only because
    # their dispatch is licensed by the complex expansion probe, so their route rows
    # are driven on the probe leg alone and refused by name on the shipped leg.
    "complex fused electric D/E pair": ("complex_fused_electric_pair",
                                        "metal_tranche7_fused_pairs_device_gate"),
    "folded complex fused D/E pair": ("folded_complex_fused_pair",
                                      "metal_tranche7_fused_pairs_device_gate"),
    # --- a per-family weld whose FILE is not named for its family -----------
    # The registry family is cylindrical_complex_fused_electric_pair; its kernel
    # is metal_kernels/cylindrical_fused_electric_pair.py and its weld key follows
    # the file. Both resolve, which is what the row promises.
    "cylindrical complex fused electric D/E pair": (
        "cylindrical_complex_fused_electric_pair",
        "metal_cylindrical_fused_electric_pair_device_gate"),
}

#: Fused labels the shipped composer SELECTS on a driver-route case, that were
#: measured byte-identical through ``driver.step``, and that are WITHHELD from the
#: release for a reason that is a decision rather than a measurement. The rung that
#: reads this table stops such a label before the warm pass, before any kernel
#: compiles, and without borrowing a sibling family's provenance — and it still
#: refuses when a run opts into the label by name through
#: ``fastpath.FUSE_ARMS_SWITCH``.
#:
#: EMPTY SINCE 2026-09-12. It held two families whose cells were bound by a GROUP
#: weld only — ``complex_fused_electric_pair`` and ``folded_complex_fused_pair`` —
#: while "may the release table cite a group key" was an open question. That call
#: was taken on 2026-09-12 for three sibling families (the ruling is on their
#: :data:`ARM_CERTIFICATION` rows) and applied to these two later the same day,
#: after the same two facts were checked for them. The table stays, empty, because
#: the rung that reads it is the right place for the NEXT family that is driven,
#: bound, and withheld on a decision; an empty table is a fact about the release,
#: not a vestige.
METAL_PENDING_DEVICE_GATE_ARMS: Mapping[str, str] = {}


def fused_labels() -> Tuple[str, ...]:
    """Every label the Metal registry's WELD rows can write, read off the registry.

    ``fastpath.arm_is_fused`` is the TRITON spelling — a ``"fused pair"`` prefix
    plus two names — and it answers False for every label here (verified: it
    returns True for ``"fused pair B"`` and ``"dispersive fused pair"``, and False
    for ``"fused magnetic B/H pair"``, ``"folded fused B/H pair"`` and the rest).
    Widening that function would make one predicate answer for two vocabularies;
    reading the registry's own ``is_weld`` flag instead means a new Metal weld is
    fused here the moment it registers, with nothing to remember to update.

    Torch-free: ``metal_kernels.arms`` registers labels and predicates, and
    importing it compiles nothing and touches no device.
    """
    from .metal_kernels import arms  # noqa: PLC0415

    return tuple(sorted({spec.label for spec in arms.registered() if spec.is_weld}))


def fused_arm_constituents() -> Mapping[str, Tuple[str, ...]]:
    """Fused label -> the ARMS its kernel implements, derived from the composer.

    THE SEE-THROUGH THE PENDING RUNG NEEDS. A fused product writes ONE label into
    BOTH slots it absorbs, so the arms it substitutes vanish from ``selected`` and a
    rung reading the label alone would pass over them — which is how a wiring edit
    could quietly lift a standing refusal by covering the arms it names.

    DERIVED, NOT TYPED, from ``launch.FUSED_PAIR_ARMS`` plus every
    ``FUSED_PAIR_EXTRA_ARMS`` row, keyed by the registry's label for the family. The
    Triton track keeps this as a literal table and checks it against the composer;
    here the composer IS the table, which is the same invariant with nothing to keep
    in step. A family with no ``FUSED_PAIR_ARMS`` row gets no entry — the composer
    refuses to install it for exactly that reason, so a label it could write is a
    label with a row.
    """
    from .metal_kernels import arms, launch  # noqa: PLC0415

    out: Dict[str, Tuple[str, ...]] = {}
    for spec in arms.registered():
        if not spec.is_weld:
            continue
        pair_arms = launch.FUSED_PAIR_ARMS.get(spec.family)
        if pair_arms is None:
            continue
        names = set(pair_arms)
        for extra in launch.FUSED_PAIR_EXTRA_ARMS.get(spec.family, ()):
            names |= set(extra)
        out[spec.label] = tuple(sorted(names))
    return out


# ---------------------------------------------------------------------------
# The release: which fused arms have been driven through the driver seam
# ---------------------------------------------------------------------------

#: THE FUSED METAL ARMS THAT HAVE BEEN DRIVEN THROUGH THE DRIVER SEAM, and the
#: gate cases each one was driven on. This is the per-arm allow-list clause 8M
#: reads, and it is the whole of what the Metal table ships enabled.
#:
#: WHAT "DRIVEN THROUGH THE SEAM" MEANS HERE, and it is the same sentence the
#: Triton table makes: the arm's kernel LAUNCHED from ``FastPathPlan.dispatch`` at
#: the leading consult inside ``driver.step()``, on a driver lifted by
#: ``lift_simulation(prefer_gpu=False)`` from a real ``mp.Simulation`` under
#: ``MEEP_GPU_DISPATCH=1`` (HISTORY: the rows below were driven that way, before
#: 2026-09-27; since then a ``prefer_gpu=False`` driver is the NumPy reference and
#: never plans, and the Metal route gate lifts ``prefer_gpu=True`` -- the same
#: NumPy-owned engine on the same ladder from ``plan_fast_path`` on); the absorbed
#: consult answered True off the sentinel or the repair; the whole run was
#: byte-compared as uint32 against the same driver with ``MEEP_GPU_FUSED=0`` at
#: every rung of a ten-rung ladder plus flux spectra, with
#: ``first_divergent_checkpoint`` null everywhere; and compiled-function calls per
#: complete step DROPPED by exactly one per pair against a third leg that dispatched
#: the same slot set with fusion off.
#:
#: TYPED ROWS, AND THE ONLY TYPED STATEMENT IN THE WHOLE RELEASE. Everything else
#: — the record, the weld, the board's numbers — is cut from a run. These rows are
#: typed because the campaign has to be TOLD which (arm, case) pairs to drive, and
#: ``recut_driver_dispatch_record.py --backend metal`` refuses to write the record
#: unless the artifact shows every one of them actually driven. That is the ordering
#: trap this file's :data:`METAL_DRIVER_ROUTE_GATE` names, and the mitigation is
#: ``parity/meep_gpu/probe_metal_dispatch_dryrun.py``: the rows are typed only after
#: a dry run has shown each case dispatching and byte-identical.
#:
#: ``pml_3d_diagonal`` IS A DISPATCH CASE ON THIS TABLE and not the envelope
#: control, which is where the Metal and Triton gates deliberately differ: measured,
#: it selects both ordinary pairs and is byte-identical to 192 steps, so
#: ``dimensions`` is released as ``{1, 2, 3}`` below. The envelope control is
#: ``pml_3d`` — the SPHERE, whose subpixel averaging writes off-diagonal ``chi1inv``
#: rows — which is the shape where the release genuinely does not apply.
#:
#: NINE CASE NAMES JOINED THIS TABLE ON 2026-09-13 AND NONE OF THEM HAS RUN YET,
#: which is stated here rather than left to be discovered. Every one was built and
#: LIFTED off-device that day against stock MEEP 1.33.0 — so the run shape each one
#: produces, and therefore the axis value each row below cites, is a measurement —
#: but the composer's SELECTION, the launch counts and the byte comparison are what
#: the campaign takes, and the fused install does not run on a NumPy host at all.
#: ``recut_driver_dispatch_record.py --backend metal`` is the backstop: it walks
#: every ``(arm, case)`` pair in this table against the campaign's own
#: ``arms_driven`` and refuses to write the record for any pair no dispatching leg
#: shows. A name here the campaign does not drive is therefore a loud failure and
#: not a silent over-claim, which is the property that makes it safe to type the
#: rows before the run rather than after it. The nine: ``complex_1d``,
#: ``complex_3d_thinline`` and ``complex_3d`` on the two complex pairs;
#: ``folded_complex_3d`` and ``folded_complex_kz2d_3d`` on the two folded complex
#: pairs; ``folded_complex_offdiag_2d`` and ``folded_complex_nobloch_offdiag_2d`` on
#: the folded complex MAGNETIC pair alone, because no D-seam product is admitted on
#: an off-diagonal grid; ``dispersive6_2d`` on the ordinary magnetic pair and the
#: dispersive electric pair; and ``folded_dispersive5_2d`` on the folded magnetic
#: pair and the folded dispersive electric pair. The last two each name TWO arms
#: because the dispersive D/E products leave ``susceptibilities`` unpinned and are
#: admitted on a multi-pole grid as well — measured on the lifts, and the same
#: shape ``dispersive_2d`` and ``folded_dispersive_2d`` already have.
METAL_RELEASED_FUSED_ARMS: Mapping[str, Tuple[str, ...]] = {
    "fused magnetic B/H pair": ("pml_2d", "magnetic_seam_2d", "conductive_2d",
                                "dispersive_2d", "dispersive6_2d", "pml_1d",
                                "pml_3d_diagonal", "pml_3d", "offdiag_2d",
                                # 2026-09-17: the magnetic-source off-diagonal cell,
                                # added with the off-diagonal weld it lets fuse on
                                # the D seam. Inside this arm's existing corner
                                # (dimensions 2, zero poles, lossless).
                                "offdiag_magnetic_2d"),
    "fused electric D/E pair": ("pml_2d", "magnetic_seam_2d", "pml_1d",
                                "pml_3d_diagonal"),
    "fused dispersive D/E pair": ("dispersive_2d", "dispersive6_2d"),
    "folded fused B/H pair": ("folded_2d", "folded_dispersive_2d",
                              "folded_dispersive5_2d", "folded_offdiag_2d",
                              "folded_3d", "folded_dispersive_3d",
                              # 2026-09-17, with the folded off-diagonal weld: inside
                              # this arm's corner (dimensions 2, zero poles).
                              "folded_offdiag_magnetic_2d"),
    "folded fused D/E pair": ("folded_2d", "folded_3d"),
    "complex fused magnetic B/H pair": ("bloch_2d", "complex_nobloch_2d",
                                        "complex_1d", "complex_3d_thinline",
                                        "complex_3d"),
    "complex fused electric D/E pair": ("bloch_2d", "complex_nobloch_2d",
                                        "complex_1d", "complex_3d_thinline",
                                        "complex_3d"),
    "folded complex fused B/H pair": ("folded_complex_2d", "folded_complex_3d",
                                      "folded_complex_kz2d_3d",
                                      "folded_complex_offdiag_2d",
                                      "folded_complex_nobloch_offdiag_2d"),
    "folded complex fused D/E pair": ("folded_complex_2d", "folded_complex_3d",
                                      "folded_complex_kz2d_3d"),
    "cylindrical m=0 fused electric D/E pair": ("cylindrical",),
    "cylindrical complex fused electric D/E pair": ("cylindrical_m1",
                                                    "cylindrical_m0_complex"),
    "conductive fused electric D/E pair": ("conductive_2d",),
    # WIDENED 2026-09-17 BY THE CONFIRMING PASS, which measured this arm holding
    # step_D/update_E on all three folded dispersive cases under the release's own
    # offer while the row named one. The other two were always driven; the row simply
    # did not say so, and folded_dispersive_3d's DRIVE row read `pairs: 1` as a
    # result.
    "folded fused dispersive D/E pair": ("folded_dispersive_2d",
                                         "folded_dispersive5_2d",
                                         "folded_dispersive_3d"),
    "beta fused electric D/E pair": ("special_kz_2d",),
    # --- THE ALL-PATHS ROUND, 2026-09-17. Eleven arms, each released on the cases the
    # dry run MEASURED the composer selecting it on, with every axis value below read
    # off that case's own ``fastpath._run_shape`` lift rather than transcribed from a
    # sibling row. Six of the eleven became reachable only when their family gained a
    # ``launch.FUSED_PAIR_ARMS`` row in the same batch: they were certified and welded
    # the whole time, and the seam loop had no arm pair to absorb, so it never asked.
    "BFAST fused electric D/E pair": ("bfast_1d",),
    "complex-beta fused electric D/E pair": ("complex_beta_2d",
                                             "complex_beta_bloch_2d"),
    "complex-beta fused magnetic B/H pair": ("complex_beta_2d",
                                             "complex_beta_bloch_2d"),
    "conductive no-PML fused electric D/E pair": ("absorber_1d",),
    "cylindrical complex fused magnetic B/H pair": ("cylindrical_m0_complex",
                                                    "cylindrical_m1"),
    "cylindrical m=0 fused magnetic B/H pair": ("cylindrical_m0", "cylindrical"),
    "folded beta complex fused B/H pair": ("folded_complex_beta_2d",
                                           "folded_complex_beta_bloch_2d"),
    "folded beta complex fused D/E pair": ("folded_complex_beta_2d",
                                           "folded_complex_beta_bloch_2d"),
    "folded beta real fused B/H pair": ("folded_special_kz_2d",),
    "folded beta real fused D/E pair": ("folded_special_kz_2d",),
    "folded off-diagonal fused electric D/E pair": ("folded_offdiag_magnetic_2d",),
    # THE UNFOLDED TWIN, and the case name is the whole finding of this round's Metal
    # dry run. `offdiag_2d` and `pml_3d` are the off-diagonal cases this table already
    # drives and NEITHER can carry this weld: both put an ELECTRIC source in the D->E
    # seam, and `deposit_repair.repairable` refuses an off-diagonal chi1inv
    # constitutive by name, because a point repair recomputes E at the deposit cell
    # from THAT cell's displacement while an off-diagonal constitutive reads its
    # neighbours'. The magnetic-source variants are what the Triton and CUDA tables
    # release these two welds on, for exactly this reason, and they are new Metal
    # DRIVE cases here.
    "off-diagonal fused electric D/E pair": ("offdiag_magnetic_2d",),
    # THE TWO COMPLEX NO-ABSORBER ARMS, measured last because they are probe-gated:
    # without the expansion licence the composer refuses each half by name -- "no
    # complex-multiply expansion probe artifact is available ... which arm the numpy
    # reference takes is a measured platform fact and may not be guessed" -- so the
    # first pass over these two cases read SELECTED-NOTHING and said so.
    "complex no-PML off-diagonal fused electric D/E pair": ("complex_no_pml_offdiag",),
    # THE NO-ABSORBER STORED-E PAIR. Measured on the all-offered pass and then MISSED
    # from this table, which the confirming pass caught by name: with the arm unoffered
    # the composer kept `no-PML curl` and `no-PML stored E` as singles on
    # no_pml_dispersive_2d while that case's DRIVE row claimed a pair. A release row
    # and a route row are two halves of one statement and this round had typed one.
    "no-PML fused electric D/E pair": ("no_pml_dispersive_2d",),
    "fused complex conductive no-PML curl -> complex stored E": ("complex_no_pml_3d",),
}

#: THE AXES EVERY RELEASED METAL ARM WAS DRIVEN ON WITH THE SAME VALUE. Four rows,
#: and the table is short on purpose: this backend's released set spans Cartesian,
#: folded, complex-Bloch and cylindrical shapes, so ``cylindrical``,
#: ``complex_storage``, ``bloch`` and ``dimensions`` DISAGREE between arms and one
#: shared row could only say one of those. They are per-arm rows in
#: :data:`METAL_FUSED_RELEASE_ARM_AXES` instead, which is what lets the cylindrical
#: product require a Dcyl grid while the seven Cartesian ones refuse one.
#:
#: What is left is what every arm agreed on, and each row is a fact about the whole
#: case set rather than about one case.
#:
#: ``off_diagonal_epsilon`` LEFT THIS TABLE ON 2026-09-12 and is a per-arm CORNER
#: now, because the arms stopped agreeing on it: ``pml_3d`` drives the ORDINARY
#: MAGNETIC PAIR on a sphere whose chi1inv carries off-diagonal rows (measured: that
#: case's run shape reads ``off_diagonal_epsilon=True``) while most released arms
#: are never driven off-diagonal at all, and a shared row could only have said one
#: of those.
#:
#: THREE ARMS HAVE OFF-DIAGONAL COVERAGE NOW, which is worth stating because the
#: sentence here counted ONE until 2026-09-12 and TWO until 2026-09-13, and a reader
#: who takes a stale count for a structural fact reaches the wrong conclusion about
#: what is still open. They are the ordinary magnetic pair (``pml_3d``,
#: ``offdiag_2d``), the folded magnetic pair (``folded_offdiag_2d``) and, from
#: 2026-09-13, the folded COMPLEX magnetic pair (``folded_complex_offdiag_2d`` and
#: ``folded_complex_nobloch_offdiag_2d``). Every one is bounded by its own entry in
#: :data:`METAL_OFF_DIAGONAL_CORNERS`, and every arm ABSENT from that table is still
#: refused on any off-diagonal grid — which is every D-seam product without
#: exception, so a fused pair on the D side of an off-diagonal grid has never been
#: admitted here.
#:
#: THE SENTENCE THAT ROW USED TO CARRY WAS FALSE, and is recorded here rather than
#: quietly deleted. Until 2026-09-11 it read "and none CAN be on the products
#: released here: the composer's specialized-family guard refuses every ordinary and
#: folded pair on a grid with off-diagonal chi1inv rows". That guard is
#: ``specialized_family_owns_the_grid``; it lives in ``triton_kernels/launch.py`` and
#: NOWHERE in the Metal package — grep ``metal_kernels/launch.py`` and this file for
#: it — so the refusal was resting on another backend's composer. The two readings
#: prescribe opposite work: a structural "none can be" says do not bother, while a
#: coverage boundary says drive the case. Driving it is what ``pml_3d`` now does,
#: and the magnetic pair dispatching there is the measurement that settles it.
METAL_FUSED_RELEASE_ENVELOPE: Tuple[Tuple[str, Any, str], ...] = (
    ("nonlinearity", False, "no nonlinear case was driven"),
)

#: THE AXES ONE ARM WAS DRIVEN ON, beyond the shared table above.
#:
#: THE RULE FOR A ROW HERE, applied uniformly and the same rule the Triton table
#: states: an axis is pinned for an arm when that arm's OWN case list drove exactly
#: one of its values. An axis whose cases drove BOTH values is left out — which is
#: what the missing ``conductivity`` and ``susceptibilities`` rows on ``fused
#: magnetic B/H pair`` mean, since ``conductive_2d`` and ``dispersive_2d`` are in
#: its list and ``pml_2d`` is too.
#:
#: AND THE RULE HAS A PRECONDITION THAT COST AN OVER-CLAIM TO LEARN, 2026-09-12.
#: "Left out because both values were driven" is only sound when the axis is
#: INDEPENDENT of the others — when driving both values somewhere licenses both
#: values everywhere. ``conductivity`` and ``susceptibilities`` qualify: the arm
#: runs at 1-D, 2-D and 3-D with and without each. ``off_diagonal_epsilon`` does
#: NOT: both its values were driven, but the True side by exactly ONE case at ONE
#: dimensionality, so leaving it out admitted 1-D and 2-D off-diagonal grids that
#: nothing drives — and the shipped composer duly installed a fused pair on one.
#: An axis like that needs the pair recorded, not the axis dropped, which is what
#: :data:`METAL_OFF_DIAGONAL_CORNERS` does. Before omitting an axis here, ask
#: whether its two values were driven ACROSS the arm's other axes or only at one
#: corner of them.
#:
#: AN ARM WITH NO ROW FAILS CLOSED (:func:`fused_release_arm_reasons_metal`),
#: because the shared table above is deliberately only the axes every arm agrees
#: on: an arm missing from here has no statement about the axes that separate the
#: arms from each other, and admitting on silence is the shape of every over-claim
#: this file exists to prevent.
#:
#: ``susceptibilities`` IS DELIBERATELY UNPINNED ON THE THREE DISPERSIVE-ADJACENT
#: ROWS — ``fused dispersive D/E pair``, ``folded fused D/E pair`` and, from
#: 2026-09-12, ``folded fused dispersive D/E pair`` — and that is an admitted
#: asymmetry rather than an oversight. ``dispersive_2d``
#: drives ONE Lorentzian pole while the corpus rows this arm serves carry up to six,
#: so pinning the axis to the driven value would release nothing on the rows the
#: composer actually selects it for. It is the Triton table's recorded precedent for
#: the dispersive fused pair, extended here and recorded as a release decision rather
#: than presented as a measurement.
#: THE CORNER EACH ARM'S OWN OFF-DIAGONAL CASES SAT AT, on every axis the arm's row
#: leaves open. An arm is admitted on an off-diagonal grid only INSIDE its corner.
#:
#: WHY THIS IS A TABLE AND NOT A ROW IN :data:`METAL_FUSED_RELEASE_ARM_AXES`.
#: ``fastpath._axis_reasons`` asks one question per axis and cannot express a
#: CONJUNCTION, and off-diagonal coverage is one: it is coverage "at dimensions=3
#: AND susceptibilities=0 AND conductivity=False", not coverage of the axis. The
#: ordinary magnetic pair really does run at 1-D, 2-D and 3-D, with and without a
#: conductivity, with and without a susceptibility — so its row leaves those axes
#: open, correctly — but its OFF-DIAGONAL cases are the sphere (``pml_3d``) and the
#: cylinder (``offdiag_2d``), both lossless, both dispersion-free. As independent
#: rows that is unsayable: an ``off_diagonal_epsilon`` row could only say False,
#: refusing both cases and losing the axis, or be omitted, admitting every
#: off-diagonal grid at every corner of the open axes, which is what happened.
#:
#: OMISSION IS THE OVER-CLAIM THIS TABLE EXISTS TO STOP, and it was a live one for
#: the length of one audit. When ``off_diagonal_epsilon`` moved out of the shared
#: envelope on 2026-09-12 with no per-arm replacement, ``released_fused_arms_metal``
#: admitted the magnetic pair on EVERY off-diagonal shape at EVERY dimensionality,
#: and the shipped composer installed it on a 2-D off-diagonal grid — measured on
#: device, not argued. The first repair bounded the DIMENSIONALITY alone, which was
#: the axis the audit caught; the table now records the whole corner, because the
#: rule that makes omission sound ("both values were driven") has a precondition
#: the row-table's note states — driven ACROSS the arm's other axes, not at one
#: corner of them — and a bound that names one axis of the corner is silent about
#: the rest.
#:
#: THE RULE FOR WHAT A CORNER LISTS, and it is checkable: every axis the arm's row
#: in :data:`METAL_FUSED_RELEASE_ARM_AXES` leaves OPEN — absent, or a set of more
#: than one value — appears here with the values its off-diagonal cases actually
#: carried; an axis the row already pins to one value is not repeated, because one
#: fact with two homes is one that can disagree with itself. ``folded fused B/H
#: pair`` therefore lists ``susceptibilities`` and, since ``folded_3d`` opened its
#: ``dimensions`` row to {2, 3}, ``dimensions``: it drives both susceptibility values
#: (``folded_2d``, ``folded_dispersive_2d``) and both dimensionalities, but
#: off-diagonal only at zero poles in 2-D (``folded_offdiag_2d``). Measured on the
#: 2026-09-04 census, that corner costs
#: exactly one served instance — ``absorbed_power_density.py``, folded,
#: off-diagonal, one Lorentzian pole — and admits the nineteen beside it.
#:
#: AN ARM ABSENT FROM THIS TABLE IS REFUSED ON ANY OFF-DIAGONAL GRID, which is why
#: the other arms carry NO ``off_diagonal_epsilon`` row of their own: absence
#: here already says it. And an axis of the corner that did not READ refuses too:
#: admitting on a fact that did not read would assert the fact.
#:
#: THE THIRD ENTRY ARRIVED 2026-09-13 and is the step at which the Metal dispatch
#: count reaches its target. ``folded complex fused B/H pair`` was absent from this
#: table, so it was refused on EVERY off-diagonal grid — which is the whole of what
#: kept three served instances (``examples:solve-cw.py``,
#: ``tests:TestArrayMetadata.test_array_metadata`` and
#: ``TestHoleyWvgBands.test_fields_at_kx``, all folded, complex and off-diagonal)
#: off the dispatch route. Two route cases drive it: ``folded_complex_offdiag_2d``
#: at a nonzero k_point and ``folded_complex_nobloch_offdiag_2d`` unphased, both
#: 2-D, so the corner reads ``dimensions`` {2} and ``bloch`` {True, False}. The
#: ``bloch`` key is there because dropping that arm's ``bloch`` pin OPENED the axis,
#: and an axis the row leaves open is one this table has to speak about; listing
#: both values is not a widening, it is the record that both were driven here and
#: only here. The two levers pay only TOGETHER: measured on the 2026-09-13 census,
#: the corner entry alone recovers one instance and the ``bloch`` drop alone
#: recovers none, because ``solve-cw.py`` and ``TestArrayMetadata`` are refused
#: twice over — on ``bloch`` AND on ``off_diagonal_epsilon``.
#:
#: WHAT WOULD WIDEN IT: a case at the missing corner. A folded off-diagonal grid
#: over a susceptibility would put 1 in the folded pair's set; a conductive
#: off-diagonal grid would put True in the magnetic pair's; a 3-D folded complex
#: off-diagonal grid would put 3 in the folded complex pair's. NONE OF THE THREE
#: EXISTS ON THE ROUTE, and the first of them was attempted and could not be built:
#: a folded cylinder over a Lorentzian pole is refused by the lift on this MEEP
#: (``chi1inv`` is non-diagonal at a lattice point and stock MEEP 1.33.0 carries no
#: ``fields.get_susceptibility_sigma``), and neither escape the refusal names
#: serves — ``eps_averaging=False`` deletes the off-diagonal rows the case exists to
#: drive, and moving the dispersion onto an axis-aligned block beside a
#: non-dispersive cylinder still refuses, because the check is not per-region. It
#: needs a MEEP built with ``MEEP_SIGMA_PATCH=1``; measured 2026-09-13, and the one
#: served instance behind it (``examples:absorbed_power_density.py``) stays refused
#: by name until such a build exists on the campaign host.
METAL_OFF_DIAGONAL_CORNERS: Mapping[str, Mapping[str, FrozenSet[Any]]] = {
    "fused magnetic B/H pair": {
        "dimensions": frozenset({2, 3}),
        "susceptibilities": frozenset({0}),
        "conductivity": frozenset({False}),
    },
    "folded fused B/H pair": {
        "dimensions": frozenset({2}),
        "susceptibilities": frozenset({0}),
    },
    "folded complex fused B/H pair": {
        "dimensions": frozenset({2}),
        "bloch": frozenset({True, False}),
    },
    # THE FOURTH ENTRY, 2026-09-17, AND IT IS EMPTY ON PURPOSE. ``folded off-diagonal
    # fused electric D/E pair`` is the one arm here that exists ONLY for off-diagonal
    # grids -- its kernel is the off-diagonal constitutive welded to the folded curl --
    # so ABSENCE from this table would refuse it on every grid it can run on, which is
    # the opposite of what absence means for the three arms above.
    #
    # EMPTY IS THE COMPLETE ANSWER UNDER THIS TABLE'S OWN RULE rather than a shortcut.
    # The rule is that every axis the arm's row in METAL_FUSED_RELEASE_ARM_AXES leaves
    # OPEN appears here with the values its off-diagonal cases carried, and that row
    # leaves nothing open: dimensions 2, fold required, real storage, no Bloch, beta 0,
    # not BFAST, lossless, zero poles, PML active -- every one a single value read off
    # folded_offdiag_magnetic_2d's own lift. There is no axis left for a corner to
    # bound. If a later round widens any of those rows, the axis it opens must arrive
    # here in the same edit, or this entry silently starts admitting a corner nothing
    # drove.
    "folded off-diagonal fused electric D/E pair": {},
    # AND ITS UNFOLDED TWIN, empty for the same reason: its row pins every axis to a
    # single value read off offdiag_magnetic_2d's lift, so there is no open axis for a
    # corner to bound, and absence would refuse the arm on the only grids it serves.
    "off-diagonal fused electric D/E pair": {},
    # AND THE COMPLEX UNFOLDED ONE, 2026-09-17, empty for the same reason as the two
    # above: its kernel IS the off-diagonal constitutive, and its row pins every axis
    # to a single value read off complex_no_pml_offdiag's lift.
    "complex no-PML off-diagonal fused electric D/E pair": {},
}


METAL_FUSED_RELEASE_ARM_AXES: Mapping[str, Tuple[Tuple[str, Any, str], ...]] = {
    # --- THE ALL-PATHS ROUND'S ELEVEN, 2026-09-17. Every value below is
    # ``fastpath._run_shape`` on that case's own lift (prefer_gpu=False, stock MEEP
    # 1.33.0; the run shape is the lift's and is the same for prefer_gpu=True on an
    # Apple GPU, whose engine is the same NumPy arrays), read back verbatim rather
    # than inherited from the sibling row it
    # resembles -- which is what catches the places the two differ, such as
    # ``absorber_1d`` carrying FIVE susceptibility states and no boundary layer where
    # its PML sibling carries neither.
    "BFAST fused electric D/E pair": (
        ("dimensions", 3, "bfast_1d lifts to dimensions 3 on a (1, 1, 250) cell"),
        ("cylindrical", False, "bfast_1d is a Cartesian grid"),
        ("folded", None, "bfast_1d is unfolded"),
        ("complex_storage", False, "bfast_1d stores real fields"),
        ("bloch", False, "bfast_1d carries no Bloch phase"),
        ("beta", 0, "bfast_1d is not a special-kz run"),
        ("bfast", True,
         "this arm IS the BFAST product; on a non-BFAST grid the ordinary electric "
         "pair is what the composer selects"),
        ("conductivity", False, "bfast_1d is lossless"),
        ("susceptibilities", 0, "bfast_1d carries no susceptibility"),
        ("pml_active", True, "bfast_1d carries a PML"),
    ),
    "complex-beta fused electric D/E pair": (
        ("dimensions", 2, "both cases are 2-D, (120, 120, 1)"),
        ("cylindrical", False, "neither case is a Dcyl grid"),
        ("folded", None,
         "neither case is folded; the folded beta-complex pairs are separate arms"),
        ("complex_storage", True,
         "this arm IS the complex beta product; the real spelling routes to the "
         "beta pairs above"),
        ("bloch", frozenset({False, True}),
         "BOTH VALUES DRIVEN, which is why there are two cases: complex_beta_2d "
         "lifts bloch False and complex_beta_bloch_2d lifts it True on the same "
         "cell, so the axis is an enumeration on evidence rather than an omission"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN; both cases run beta=0.4"),
        ("bfast", False, "neither case is a BFAST grid"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        ("pml_active", True, "both carry a PML"),
    ),
    "complex-beta fused magnetic B/H pair": (
        ("dimensions", 2, "both cases are 2-D, (120, 120, 1)"),
        ("cylindrical", False, "neither case is a Dcyl grid"),
        ("folded", None, "neither case is folded"),
        ("complex_storage", True, "this arm IS the complex beta product"),
        ("bloch", frozenset({False, True}),
         "both values driven, as on the electric twin and for the same reason"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN; both cases run beta=0.4"),
        ("bfast", False, "neither case is a BFAST grid"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        ("pml_active", True, "both carry a PML"),
    ),
    "conductive no-PML fused electric D/E pair": (
        ("dimensions", 1, "absorber_1d lifts to dimensions 1 on (1, 1, 400)"),
        ("cylindrical", False, "absorber_1d is a Cartesian grid"),
        ("folded", None, "absorber_1d is unfolded"),
        ("complex_storage", False, "absorber_1d stores real fields"),
        ("bloch", False, "absorber_1d carries no Bloch phase"),
        ("beta", 0, "absorber_1d is not a special-kz run"),
        ("bfast", False, "absorber_1d is not a BFAST grid"),
        ("conductivity", True,
         "this arm IS the conductive no-absorber product; the lossless spelling "
         "routes to the plain no-PML electric pair"),
        ("susceptibilities", 5,
         "FIVE, read off the lift and pinned to it. absorber_1d carries an "
         "mp.Absorber, whose conductivity profile MEEP expresses as susceptibility "
         "states; an omitted axis here would admit every pole count, which is the "
         "defect the 2026-09-12 verifier found on the ordinary magnetic pair"),
        ("pml_active", False,
         "this arm IS the NO-ABSORBER product: absorber_1d carries an mp.Absorber "
         "rather than a PML, so pml_active lifts False -- the axis that made this "
         "case reachable at all once the shared envelope stopped pinning it"),
    ),
    "cylindrical complex fused magnetic B/H pair": (
        ("cylindrical", True, "this arm IS the Dcyl product"),
        ("dimensions", 2, "a Dcyl grid reports dimensions 2"),
        ("folded", None,
         "Grid refuses a mirror plane on a Dcyl cell, so no fold can reach it"),
        ("complex_storage", True,
         "|m| >= 1 and m = 0 complex both store complex fields. The census carries "
         "no m, so the arm is bounded on STORAGE exactly as its electric twin is, "
         "and cylindrical_m0_complex is the case that drives it"),
        ("bloch", False, "the Dcyl cases carry no k_point"),
        ("beta", 0, "the Dcyl cases are not special-kz runs"),
        ("conductivity", False, "the Dcyl cases are lossless"),
        ("susceptibilities", 0, "the Dcyl cases carry no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "cylindrical m=0 fused magnetic B/H pair": (
        ("cylindrical", True, "this arm IS the Dcyl product"),
        ("dimensions", 2, "a Dcyl grid reports dimensions 2"),
        ("folded", None, "the cylindrical case is unfolded"),
        ("complex_storage", False,
         "the m=0 product is the REAL cylindrical one; |m|>=1 goes to the complex "
         "family, exactly as on the electric twin"),
        ("bloch", False, "cylindrical_m0 carries no k_point"),
        ("beta", 0, "cylindrical_m0 is not a special-kz run"),
        ("conductivity", False, "cylindrical_m0 is lossless"),
        ("susceptibilities", 0, "cylindrical_m0 carries no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded beta complex fused B/H pair": (
        ("dimensions", 2, "both cases are 2-D, (120, 62, 1)"),
        ("cylindrical", False, "neither case is a Dcyl grid"),
        ("folded", "required",
         "this arm IS the folded product: both cases carry 'mirror plane on Y'"),
        ("complex_storage", True, "this arm IS the complex beta product"),
        ("bloch", frozenset({False, True}),
         "both values driven: folded_complex_beta_2d False, "
         "folded_complex_beta_bloch_2d True on the same folded cell"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN; both cases run beta=0.4"),
        ("bfast", False, "neither case is a BFAST grid"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        ("pml_active", True, "both carry a PML"),
    ),
    "folded beta complex fused D/E pair": (
        ("dimensions", 2, "both cases are 2-D, (120, 62, 1)"),
        ("cylindrical", False, "neither case is a Dcyl grid"),
        ("folded", "required", "this arm IS the folded product"),
        ("complex_storage", True, "this arm IS the complex beta product"),
        ("bloch", frozenset({False, True}), "both values driven, as on its B twin"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN; both cases run beta=0.4"),
        ("bfast", False, "neither case is a BFAST grid"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        ("pml_active", True, "both carry a PML"),
    ),
    "folded beta real fused B/H pair": (
        ("dimensions", 2, "folded_special_kz_2d is 2-D, (120, 62, 1)"),
        ("cylindrical", False, "folded_special_kz_2d is a Cartesian grid"),
        ("folded", "required",
         "this arm IS the folded product: 'mirror plane on Y'"),
        ("complex_storage", False,
         "this arm IS the REAL beta product; the complex spelling routes to the "
         "folded beta COMPLEX pair above"),
        ("bloch", False,
         "MEASURED, not assumed: a kz-only k_point on a 2-D cell folds into beta "
         "rather than into Bloch phases, so the run shape reads bloch False"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN"),
        ("bfast", False, "folded_special_kz_2d is not a BFAST grid"),
        ("conductivity", False, "folded_special_kz_2d is lossless"),
        ("susceptibilities", 0, "folded_special_kz_2d carries no susceptibility"),
        ("pml_active", True, "folded_special_kz_2d carries a PML"),
    ),
    "folded beta real fused D/E pair": (
        ("dimensions", 2, "folded_special_kz_2d is 2-D, (120, 62, 1)"),
        ("cylindrical", False, "folded_special_kz_2d is a Cartesian grid"),
        ("folded", "required", "this arm IS the folded product"),
        ("complex_storage", False, "this arm IS the REAL beta product"),
        ("bloch", False, "as on its B twin, and for the same measured reason"),
        ("beta", 0.4, "PINNED TO THE ONE VALUE DRIVEN"),
        ("bfast", False, "folded_special_kz_2d is not a BFAST grid"),
        ("conductivity", False, "folded_special_kz_2d is lossless"),
        ("susceptibilities", 0, "folded_special_kz_2d carries no susceptibility"),
        ("pml_active", True, "folded_special_kz_2d carries a PML"),
    ),
    "no-PML fused electric D/E pair": (
        ("dimensions", 2, "no_pml_dispersive_2d is 2-D, (120, 120, 1)"),
        ("cylindrical", False, "no_pml_dispersive_2d is a Cartesian grid"),
        ("folded", None, "no_pml_dispersive_2d is unfolded"),
        ("complex_storage", False, "no_pml_dispersive_2d stores real fields"),
        ("bloch", False, "no_pml_dispersive_2d carries no Bloch phase"),
        ("beta", 0, "no_pml_dispersive_2d is not a special-kz run"),
        ("bfast", False, "no_pml_dispersive_2d is not a BFAST grid"),
        ("conductivity", False,
         "no_pml_dispersive_2d is lossless; the conductive spelling routes to the "
         "conductive no-PML pair above"),
        ("susceptibilities", 2,
         "TWO, read off the lift and pinned to it -- update_P keeps `ADE update_P` "
         "beside the weld"),
        ("pml_active", False,
         "this arm IS the no-absorber product: the case carries no boundary layer"),
    ),
    "complex no-PML off-diagonal fused electric D/E pair": (
        ("dimensions", 2, "complex_no_pml_offdiag is 2-D, (25, 25, 1)"),
        ("cylindrical", False, "complex_no_pml_offdiag is a Cartesian grid"),
        ("folded", None,
         "this is the UNFOLDED complex weld; a mirror plane goes to the folded "
         "complex off-diagonal arm, which this round does not release"),
        ("complex_storage", True,
         "this arm IS the complex weld; the real spelling routes to the plain "
         "off-diagonal stencil weld"),
        ("bloch", True,
         "complex_no_pml_offdiag carries k_point (0.3892, 0.1597, 0)"),
        ("beta", 0, "complex_no_pml_offdiag is not a special-kz run"),
        ("bfast", False, "complex_no_pml_offdiag is not a BFAST grid"),
        ("conductivity", False, "complex_no_pml_offdiag is lossless"),
        ("susceptibilities", 0,
         "complex_no_pml_offdiag carries no susceptibility"),
        ("pml_active", False,
         "this arm IS the NO-ABSORBER weld: the case carries no boundary layer"),
    ),
    "fused complex conductive no-PML curl -> complex stored E": (
        ("dimensions", 3, "complex_no_pml_3d is 3-D, (23, 21, 27)"),
        ("cylindrical", False, "complex_no_pml_3d is a Cartesian grid"),
        ("folded", None, "complex_no_pml_3d is unfolded"),
        ("complex_storage", True,
         "this arm IS the complex product; both halves bind complex storage as "
         "float2 volumes"),
        ("bloch", True,
         "complex_no_pml_3d carries k_point (0.4, -1.3, 0.7)"),
        ("beta", 0, "complex_no_pml_3d is not a special-kz run"),
        ("bfast", False, "complex_no_pml_3d is not a BFAST grid"),
        ("conductivity", True,
         "this arm IS the conductive product; the lossless spelling routes to the "
         "plain complex no-PML arms"),
        ("susceptibilities", 1,
         "ONE, read off the lift and pinned to it -- complex_no_pml_3d carries a "
         "single pole, and update_P keeps `ADE update_P` beside the weld"),
        ("pml_active", False,
         "this arm IS the NO-ABSORBER product: complex_no_pml_3d carries no "
         "boundary layer"),
    ),
    "off-diagonal fused electric D/E pair": (
        ("dimensions", 2, "offdiag_magnetic_2d is 2-D, (200, 120, 1)"),
        ("cylindrical", False, "offdiag_magnetic_2d is a Cartesian grid"),
        ("folded", None,
         "this is the UNFOLDED weld; a mirror plane goes to its folded twin below"),
        ("complex_storage", False, "offdiag_magnetic_2d stores real fields"),
        ("bloch", False, "offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "offdiag_magnetic_2d is not a special-kz run"),
        ("bfast", False, "offdiag_magnetic_2d is not a BFAST grid"),
        ("conductivity", False, "offdiag_magnetic_2d is lossless"),
        ("susceptibilities", 0, "offdiag_magnetic_2d carries no susceptibility"),
        ("pml_active", True, "offdiag_magnetic_2d carries a PML"),
    ),
    "folded off-diagonal fused electric D/E pair": (
        ("dimensions", 2, "folded_offdiag_magnetic_2d is 2-D, (160, 61, 1)"),
        ("cylindrical", False, "folded_offdiag_magnetic_2d is a Cartesian grid"),
        ("folded", "required",
         "this arm IS the folded weld: 'mirror plane on Y'. An unfolded "
         "off-diagonal grid goes to the plain stencil weld"),
        ("complex_storage", False, "folded_offdiag_magnetic_2d stores real fields"),
        ("bloch", False, "folded_offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "folded_offdiag_magnetic_2d is not a special-kz run"),
        ("bfast", False, "folded_offdiag_magnetic_2d is not a BFAST grid"),
        ("conductivity", False, "folded_offdiag_magnetic_2d is lossless"),
        ("susceptibilities", 0,
         "folded_offdiag_magnetic_2d carries no susceptibility"),
        ("pml_active", True, "folded_offdiag_magnetic_2d carries a PML"),
    ),
    "fused magnetic B/H pair": (
        ("dimensions", frozenset({1, 2, 3}),
         "the gate ran pml_1d, five 2-D cases (offdiag_2d among them), "
         "pml_3d_diagonal and pml_3d"),
        ("folded", None, "no folded case drove the ORDINARY pair; the folded grid "
                         "has its own pair, released separately below"),
        ("cylindrical", False, "no Dcyl case drove this arm"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no k_point case drove it"),
        ("beta", 0, "no special-kz case drove it"),
        # AN INTEGER AXIS IS NOT BINARY (2026-09-13). The 2026-09-12 round's
        # verifier found four TestLoadDump rows carrying FIVE Lorentz poles admitted
        # on evidence of 0 and 1: an omitted count axis admits every count. The
        # driven values are the row, and a pole count no case drove is refused.
        ("susceptibilities", frozenset({0, 1, 6}),
         "pml_2d and its siblings carry no susceptibility; dispersive_2d carries "
         "one pole; dispersive6_2d carries SIX, which is the count the three "
         "stochastic_emitter rows carry, and it lifted off-device 2026-09-13 to "
         "susceptibilities 6 on a (90, 90, 1) grid — driven through the seam by "
         "the Metal driver-route campaign this batch runs. THE SET IS AN "
         "ENUMERATION, NOT A RANGE: two, three, four and five poles are refused "
         "because no case drove them. And the counts above one are evidenced at "
         "2-D only, since dispersive_2d and dispersive6_2d are both 2-D grids — "
         "measured over this board's 531-instance row set 2026-09-13, no served "
         "instance sits at three dimensions and more than one pole, so nothing is "
         "credited on that corner today"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "fused electric D/E pair": (
        ("dimensions", frozenset({1, 2, 3}),
         "the gate ran pml_1d, pml_2d/magnetic_seam_2d and pml_3d_diagonal"),
        ("folded", None, "no folded case drove the ORDINARY pair"),
        ("cylindrical", False, "no Dcyl case drove this arm"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no k_point case drove it"),
        ("beta", 0, "no special-kz case drove it"),
        ("conductivity", False,
         "its cases are all lossless; conductive_2d drives the magnetic pair and "
         "leaves the D seam to the conductive fused electric product"),
        ("susceptibilities", 0,
         "no case that drove this arm carried a susceptibility; dispersive_2d's D "
         "seam goes to the dispersive fused pair"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "fused dispersive D/E pair": (
        ("dimensions", 2, "dispersive_2d is a 2-D Cartesian grid"),
        ("folded", None, "no folded case drove it"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False, "dispersive_2d stores real fields"),
        ("bloch", False, "dispersive_2d carries no k_point"),
        ("beta", 0, "dispersive_2d is not a special-kz run"),
        ("conductivity", False, "dispersive_2d is lossless"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded fused B/H pair": (
        ("dimensions", frozenset({2, 3}),
         "folded_2d, folded_dispersive_2d, folded_offdiag_2d and "
         "folded_dispersive5_2d are 2-D grids; folded_3d and folded_dispersive_3d "
         "are 3-D. The second 3-D case is what keeps the susceptibilities row below "
         "honest across this axis: 0 and 1 are both driven at BOTH "
         "dimensionalities, not only at 2-D. The 5 that joined that row on "
         "2026-09-13 is the exception and is recorded rather than repaired — "
         "folded_dispersive5_2d is 2-D, so a folded 3-D grid at five poles is "
         "admitted on no case; measured over this board's 531-instance row set the "
         "same day, no served instance sits at three dimensions and more than one "
         "pole, folded or unfolded, so nothing is credited on that corner"),
        ("folded", "required",
         "this arm IS the folded product; on an unfolded grid the ordinary pair is "
         "what the composer selects"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False, "all three folded cases store real fields"),
        ("bloch", False, "none of the three folded cases carries a k_point"),
        ("beta", 0, "none of the three folded cases is a special-kz run"),
        ("conductivity", False, "none of the three folded cases carries a "
                                "conductivity"),
        # AN INTEGER AXIS IS NOT BINARY (2026-09-13): the same correction as the
        # ordinary pair's, on the four five-pole folded TestLoadDump rows.
        ("susceptibilities", frozenset({0, 1, 5}),
         "folded_2d, folded_3d and folded_offdiag_2d carry no susceptibility; "
         "folded_dispersive_2d and folded_dispersive_3d carry one pole; "
         "folded_dispersive5_2d carries FIVE, which is the count the four folded "
         "TestLoadDump 2-D rows carry, and it lifted off-device 2026-09-13 to "
         "susceptibilities 5 on an (80, 61, 1) grid under a mirror plane on Y — "
         "driven through the seam by the Metal driver-route campaign this batch "
         "runs. AN ENUMERATION, NOT A RANGE: two, three, four and six poles are "
         "refused because no case drove them, and the 5 is evidenced at 2-D alone "
         "(see the dimensions row above for what that does and does not admit)"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded fused D/E pair": (
        ("dimensions", frozenset({2, 3}), "folded_2d is a 2-D grid; folded_3d is 3-D"),
        ("folded", "required", "this arm IS the folded product"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False, "folded_2d stores real fields"),
        ("bloch", False, "folded_2d carries no k_point"),
        ("beta", 0, "folded_2d is not a special-kz run"),
        ("conductivity", False, "folded_2d is lossless"),
        ("susceptibilities", 0,
         "folded_dispersive_2d gives its D seam to the folded dispersive product, "
         "so no case drove THIS arm on a susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "complex fused magnetic B/H pair": (
        # THE 1 AND THE 3 ARRIVED 2026-09-13, and the third case is why the 3 is
        # honest. bloch_2d and complex_nobloch_2d are 2-D; complex_1d and
        # complex_3d_thinline are both (1, 1, N) grids, so on those two alone the 3
        # would be evidenced by a DEGENERATE grid while the census carries genuine
        # 3-D complex rows (TestLoadDump's four 3-D shapes and
        # TestMaterialGrid.test_matgrid_3d) that the widened row admits at runtime.
        # complex_3d is the (40, 40, 40) block that closes that gap. Neither family
        # gate has ever driven dimensions=1 on these arms, so complex_1d is the
        # FIRST 1-D evidence they carry and the route gate's byte-identity leg is
        # what supplies it.
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d and complex_nobloch_2d are the 2-D grids. complex_1d is a cell "
         "DECLARED 1-D — a declared 1 wins outright in the dimensions reader — and "
         "lifted off-device 2026-09-13 to dimensions 1 on a (1, 1, 480) grid; "
         "complex_3d_thinline is a (0, 0, L) cell declared 3-D that MEEP does not "
         "collapse, lifted the same day to dimensions 3 on (1, 1, 275); complex_3d "
         "is a genuine 3-D block, lifted to dimensions 3 on (40, 40, 40). The "
         "Metal driver-route campaign this batch runs (the directory "
         "METAL_DRIVER_ROUTE_GATE names) is what drives all three through the "
         "seam"),
        ("folded", None, "both cases are unfolded; the folded complex grid has its "
                         "own pair, released separately"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", True, "this arm IS the complex-storage product"),
        # NO ``bloch`` ROW FROM 2026-09-12: bloch_2d drives True and
        # complex_nobloch_2d drives False, and the precondition for leaving an axis
        # out holds — every OTHER axis of this row is pinned to one value, so both
        # values were driven at the same corner rather than at two different ones.
        # Until then the row pinned True on the strict rule (one case, one value).
        ("beta", 0, "neither case is a special-kz run"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "complex fused electric D/E pair": (
        # The same three cases as the magnetic twin above, and the same 2026-09-13
        # reading; see that row for why complex_3d is in the set beside the two
        # (1, 1, N) cells.
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d and complex_nobloch_2d are the 2-D grids. complex_1d is a cell "
         "DECLARED 1-D, lifted off-device 2026-09-13 to dimensions 1 on a "
         "(1, 1, 480) grid; complex_3d_thinline is a (0, 0, L) cell declared 3-D "
         "that MEEP does not collapse, lifted to dimensions 3 on (1, 1, 275); "
         "complex_3d is a genuine 3-D block, lifted to dimensions 3 on "
         "(40, 40, 40). The Metal driver-route campaign this batch runs is what "
         "drives all three through the seam"),
        ("folded", None, "both cases are unfolded; the folded complex grid has its "
                         "own pair, released separately"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", True, "this arm IS the complex-storage product"),
        # No ``bloch`` row, for the reason on the magnetic twin above.
        ("beta", 0, "neither case is a special-kz run"),
        ("conductivity", False, "both cases are lossless"),
        ("susceptibilities", 0, "neither case carries a susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded complex fused B/H pair": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d, folded_complex_offdiag_2d and "
         "folded_complex_nobloch_offdiag_2d are 2-D folds. folded_complex_3d is a "
         "genuine 3-D fold — the triangular-lattice-oblique cell, lifted "
         "off-device 2026-09-13 to dimensions 3 on an (8, 21, 126) grid under a "
         "mirror plane on X — and folded_complex_kz2d_3d is a zero-thickness z "
         "cell asked for with kz_2d='3d', so the z component of its k_point does "
         "NOT lift as a beta and the declared 3 stands: lifted the same day to "
         "dimensions 3 on (90, 62, 1). Both are driven by the Metal driver-route "
         "campaign this batch runs"),
        ("folded", "required", "this arm IS the folded complex product"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", True, "complex storage is what makes this the folded "
                                  "COMPLEX product"),
        # THE bloch PIN LEFT THIS ROW ON 2026-09-13 AND STAYED ON THE D/E TWIN,
        # which is a deliberate asymmetry and not an oversight. The rule for
        # omitting an axis is that the arm's OWN cases drove both of its values:
        # folded_complex_2d, folded_complex_3d, folded_complex_kz2d_3d and
        # folded_complex_offdiag_2d carry a nonzero k_point and
        # folded_complex_nobloch_offdiag_2d carries none, so this arm has driven
        # both. The D/E pair has NOT: the one unphased folded-complex case is
        # off-diagonal, and no D-seam product is admitted on an off-diagonal grid,
        # so that arm's row keeps its pin. The precondition the row-table's note
        # states is met here in the weaker sense only — bloch=False was driven at
        # ONE corner of the open dimensions axis (2-D) and off-diagonal at that —
        # which is why METAL_OFF_DIAGONAL_CORNERS carries the bloch values for this
        # arm rather than leaving the conjunction unsaid.
        ("beta", 0, "none of the folded complex cases is a special-kz run"),
        ("conductivity", False, "every folded complex case is lossless"),
        ("susceptibilities", 0, "no folded complex case carries a susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded complex fused D/E pair": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d is the 2-D fold; folded_complex_3d is a genuine 3-D "
         "fold, lifted off-device 2026-09-13 to dimensions 3 on (8, 21, 126), and "
         "folded_complex_kz2d_3d is a zero-thickness z cell asked for with "
         "kz_2d='3d', lifted to dimensions 3 on (90, 62, 1). THE CAVEAT THIS ROW "
         "CARRIES AND ITS MAGNETIC TWIN DOES NOT: this arm has no device gate file "
         "of its own — it binds to the tranche-7 GROUP weld, whose four fixtures "
         "are all 2-D folds — so these two route cases are the FIRST 3-D evidence "
         "it has ever carried. The magnetic twin is backed by eleven 3-D fixtures "
         "on its own family gate. If that is ruled too thin for a route case to "
         "supply alone, withhold this half: the two instances behind it are "
         "disjoint from every other lever's"),
        ("folded", "required", "this arm IS the folded complex product"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", True, "complex storage is what makes this the folded "
                                  "COMPLEX product"),
        # THE PIN STAYS HERE WHILE THE MAGNETIC TWIN DROPPED IT, 2026-09-13. The
        # only unphased folded-complex case on this route is
        # folded_complex_nobloch_offdiag_2d, and it is OFF-DIAGONAL: no D-seam
        # product is admitted on a grid whose chi1inv carries off-diagonal rows, so
        # the release does not admit this arm there and the case cannot be evidence
        # for it. "Both values were driven" is false for this arm, and writing the
        # drop on the twin's licence would be claiming a case that never drove it.
        ("bloch", True,
         "every folded complex case this arm is admitted on carries a nonzero "
         "k_point — folded_complex_2d, folded_complex_3d and "
         "folded_complex_kz2d_3d. An unphased fold would need a DIAGONAL "
         "folded-complex case, which this route does not have"),
        ("beta", 0, "none of the folded complex cases is a special-kz run"),
        ("conductivity", False, "every folded complex case is lossless"),
        ("susceptibilities", 0, "no folded complex case carries a susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "cylindrical m=0 fused electric D/E pair": (
        ("cylindrical", True, "this arm IS the Dcyl product"),
        ("dimensions", 2, "a Dcyl grid reports dimensions 2"),
        ("folded", None, "the cylindrical case is unfolded"),
        ("complex_storage", False, "the m=0 product is the REAL cylindrical one; "
                                   "|m|>=1 goes to the complex family"),
        ("bloch", False, "the cylindrical case carries no k_point"),
        ("beta", 0, "the cylindrical case is not a special-kz run"),
        ("conductivity", False, "the cylindrical case is lossless"),
        ("susceptibilities", 0, "the cylindrical case carries no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "cylindrical complex fused electric D/E pair": (
        ("cylindrical", True, "this arm IS the Dcyl product"),
        ("dimensions", 2, "a Dcyl grid reports dimensions 2"),
        ("folded", None, "Grid refuses a mirror plane on a Dcyl cell, so no fold "
                         "can reach this arm"),
        ("complex_storage", True, "|m| >= 1 stores complex fields; cylindrical_m1 "
                                  "runs m = 1. The census carries no m, so the "
                                  "arm is bounded on storage, as the m=0 twin is"),
        ("bloch", False, "cylindrical_m1 carries no k_point"),
        ("beta", 0, "cylindrical_m1 is not a special-kz run"),
        ("conductivity", False, "cylindrical_m1 is lossless"),
        ("susceptibilities", 0, "cylindrical_m1 carries no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "conductive fused electric D/E pair": (
        ("dimensions", 2, "conductive_2d is a 2-D Cartesian grid"),
        ("folded", None, "no folded case drove it"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False, "conductive_2d stores real fields"),
        ("bloch", False, "conductive_2d carries no k_point"),
        ("beta", 0, "conductive_2d is not a special-kz run"),
        ("conductivity", True,
         "THE ONLY ARM THAT REQUIRES A LOSS. Its single case installs a "
         "D_conductivity and the product exists to own that D curl, so a lossless "
         "grid is not a configuration it was driven on"),
        ("susceptibilities", 0, "conductive_2d carries no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "folded fused dispersive D/E pair": (
        ("dimensions", frozenset({2, 3}),
         "WIDENED 2026-09-17: folded_dispersive_2d and folded_dispersive5_2d are "
         "2-D and folded_dispersive_3d is 3-D, and the confirming pass measured the "
         "arm holding both seams on all three"),
        ("folded", "required", "this arm exists only on a fold"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False, "all three cases store real fields"),
        ("bloch", False, "none of the three carries a k_point"),
        ("beta", 0, "none of the three is a special-kz run"),
        ("conductivity", False, "all three are lossless"),
        ("susceptibilities", frozenset({1, 5}),
         "AN ENUMERATION, NOT A RANGE, and the reason this axis appears at all: "
         "folded_dispersive_2d and folded_dispersive_3d carry ONE pole and "
         "folded_dispersive5_2d carries FIVE. An omitted count axis admits every "
         "count, which is the defect the 2026-09-12 verifier found on the ordinary "
         "magnetic pair; two, three and four poles are refused because no case "
         "drove them"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
    "beta fused electric D/E pair": (
        ("dimensions", 2, "special_kz_2d is a 2-D Cartesian grid"),
        ("folded", None, "no folded case drove it"),
        ("cylindrical", False, "no Dcyl case drove it"),
        ("complex_storage", False,
         "special_kz_2d asks kz_2d='real/imag', so storage is float32; the default "
         "'complex' spelling lifts the same beta onto COMPLEX storage and routes "
         "the slot to the complex special-kz arm instead, which this is not"),
        ("bloch", False,
         "MEASURED, not assumed: the case carries k_point=(0,0,0.4) and the run "
         "shape still reads bloch=False, because a kz-only k_point on a 2-D cell "
         "folds into beta rather than into Bloch phases"),
        ("beta", 0.4,
         "PINNED TO THE ONE VALUE DRIVEN. special_kz_2d runs beta=0.4 and nothing "
         "else on this route, so the axis admits 0.4 and refuses every other beta. "
         "The per-family device gate additionally drove beta_2d, beta_metallic_2d "
         "and beta_negative_2d, so widening this row is available on evidence "
         "already recorded — but widening it HERE would claim route-gate coverage "
         "the route gate does not have"),
        ("conductivity", False, "special_kz_2d is lossless"),
        ("susceptibilities", 0, "special_kz_2d carries no susceptibility"),
        # PINNED 2026-09-17 when `pml_active` and `bfast` left the shared envelope.
        # Values READ OFF THIS ARM'S OWN RELEASE CASES (metal_case_shapes,
        # 2026-09-17), not copied from the envelope: a shared axis removed fails
        # OPEN for every arm that does not pin it, and a copied value would admit
        # the arm on evidence it never had.
        ("pml_active", True, "every case this arm drove carries pml_active=True (measured)"),
        ("bfast", False, "every case this arm drove carries bfast=False (measured)"),
    ),
}


def fused_release_reasons_metal(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """Why this configuration is OUTSIDE what the Metal driver-route gate measured.

    Empty means the shared half of the release applies here. FAILS CLOSED ON AN
    UNREAD FACT, which is the opposite of the device rung's rule and deliberately
    so: that rung does not refuse a capability it could not read, because refusing
    on an unread fact asserts one; here the decision is an ADMISSION, and admitting
    on an unread fact asserts one.
    """
    from . import fastpath  # noqa: PLC0415

    if "unreadable" in shape:
        return (f"the run shape did not read ({shape['unreadable']}); admitting a "
                "released arm on a configuration this record cannot describe would "
                "be asserting the configuration",)
    return fastpath._axis_reasons(shape, METAL_FUSED_RELEASE_ENVELOPE)  # noqa: SLF001


def fused_release_arm_reasons_metal(shape: Mapping[str, Any],
                                    arm: str) -> Tuple[str, ...]:
    """Why THIS arm is outside what its own cases drove, beyond the shared envelope.

    An arm with no row in :data:`METAL_FUSED_RELEASE_ARM_AXES` is refused by name
    rather than admitted on the shared table alone — see that table for why silence
    cannot be an admission here.
    """
    from . import fastpath  # noqa: PLC0415

    axes = METAL_FUSED_RELEASE_ARM_AXES.get(arm)
    if axes is None:
        return (f"{arm} has no per-arm axis row (METAL_FUSED_RELEASE_ARM_AXES), so "
                "the configurations its own gate cases drove have never been "
                "written down; the shared envelope alone cannot admit it",)
    reasons = list(fastpath._axis_reasons(shape, axes))  # noqa: SLF001

    # THE ONE CONJUNCTION THE ROW VOCABULARY CANNOT SPELL. See
    # METAL_OFF_DIAGONAL_CORNERS for why it is here and not a row: off-diagonal
    # coverage is a CORNER of the axes the row leaves open, and asking the axes
    # independently either refuses every off-diagonal case driven or admits every
    # one that is not. FAILS CLOSED TWICE: an arm absent from the table is refused
    # off-diagonal, and so is a run on which any axis of the corner did not read.
    if shape.get("off_diagonal_epsilon"):
        corner = METAL_OFF_DIAGONAL_CORNERS.get(arm)
        if corner is None:
            reasons.append(
                f"off_diagonal_epsilon={shape.get('off_diagonal_epsilon')!r}; no "
                f"case drove {arm} on a grid whose chi1inv carries off-diagonal "
                "rows")
        else:
            for axis, driven in corner.items():
                if axis not in shape:
                    reasons.append(
                        f"off_diagonal_epsilon=True and {axis} did not read, so "
                        f"{arm}'s off-diagonal corner cannot be checked")
                elif shape[axis] not in driven:
                    reasons.append(
                        f"off_diagonal_epsilon=True at {axis}={shape[axis]!r}, "
                        f"and {arm} was driven off-diagonal only at {axis} in "
                        f"{sorted(driven, key=repr)!r}")
    return tuple(reasons)


def released_fused_arms_metal(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """The Metal fused labels admitted here: the shared envelope AND the arm's own axes.

    Both halves, never one. The shared table is the axes every released arm was
    driven on with the SAME value; the per-arm table is what lets the folded and
    cylindrical products REQUIRE their geometry while the Cartesian ones refuse it.
    """
    if fused_release_reasons_metal(shape):
        return ()
    return tuple(sorted(arm for arm in METAL_RELEASED_FUSED_ARMS
                        if not fused_release_arm_reasons_metal(shape, arm)))


# ---------------------------------------------------------------------------
# The environment this table reads
# ---------------------------------------------------------------------------

def metal_toolchain() -> Dict[str, Any]:
    """torch, MPS, the Metal frontend and the GPU architecture, as READ. Never raises.

    ``importable``/``built``/``available``/``compile_shader`` are the same three
    reads ``metal_kernels.coverage._metal_backend_reasons`` makes, kept in step by
    being the same three questions rather than by a shared symbol — the coverage
    predicate answers for a GRID and this answers for a HOST, and one function
    serving both would have to be told which it was being asked.
    """
    block: Dict[str, Any] = {"importable": False, "version": None, "built": None,
                             "available": None, "compile_shader": None,
                             "metal_frontend": None, "architecture": None,
                             "device_name": None,
                             "fast_math": os.environ.get(FAST_MATH),
                             "machine": platform.machine()}
    # THE GPU IS READ BEFORE TORCH, because it needs no torch: a host whose torch
    # is broken still says which GPU it has.
    try:
        from .metal_kernels import device  # noqa: PLC0415

        gpu = device.apple_gpu_identity()
        block["architecture"] = gpu.get("architecture")
        block["device_name"] = gpu.get("name")
        if gpu.get("error"):
            block["architecture_error"] = gpu["error"]
    except Exception as exc:  # noqa: BLE001 - an unreadable GPU is recorded, not refused
        block["architecture_error"] = repr(exc)
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - a broken install is as absent as a missing one
        block["import_error"] = repr(exc)
        return block
    block["importable"] = True
    block["version"] = str(getattr(torch, "__version__", "unknown"))
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        block["built"] = bool(backend is not None and backend.is_built())
        block["available"] = bool(backend is not None and backend.is_available())
    except Exception as exc:  # noqa: BLE001 - an unreadable backend is not available
        block["mps_error"] = repr(exc)
    block["compile_shader"] = hasattr(getattr(torch, "mps", None), "compile_shader")
    try:
        from .metal_kernels import device  # noqa: PLC0415

        block["metal_frontend"] = device.metal_frontend_version()
    except Exception as exc:  # noqa: BLE001 - an unreadable frontend is recorded, not refused
        block["metal_frontend_error"] = repr(exc)
    return block


def environment_block(grid: Any, toolchain: Mapping[str, Any]) -> Dict[str, Any]:
    """The Metal half of the dispatch record's ``environment``.

    Every fact in it is a Metal fact: which GPU, which torch, which frontend, and
    whether each is the one every cited Metal weld ran on (:func:`environment_verdict`).
    The GPU is recorded as ``device``, with its verdict as ``device_certified``, in
    the keys the NVIDIA tables use for theirs: the architecture is to a Metal
    environment what the compute capability is to an NVIDIA one.

    ``device_certified``, ``torch_certified`` and ``frontend_certified`` are
    THREE-VALUED (:func:`environment_verdict`): ``None`` means the fact was not read
    or the cited welds' runs cannot judge it. torch and the frontend are judged
    against the runs recorded for THIS architecture, because each architecture's
    run records its own.

    THE ARCHITECTURE IS ALSO WRITTEN AS ``device.compute_capability``, the key
    ``fastpath._finish`` hands to ``fastpath._certification_for``, so each
    dispatched arm quotes the run of the weld that certified it on this GPU (its
    ``host``, ``recorded_utc``, ``records``), and no other architecture's run. The
    Metal ledger keys its runs by architecture exactly where the NVIDIA ledgers key
    theirs by compute capability. Measured 2026-10-05 against every reader of the
    key that a Metal plan reaches: the decision, the slots, the arms, ``certified``,
    the status line and the field are byte-equal with and without it; the NVIDIA
    rungs that also read it are never reached by a Metal plan
    (``fastpath.candidate_tables`` offers ``("metal",)`` alone on an MPS host).
    Written only when the architecture was read, so an unread one still quotes no
    run and says why.
    """
    architecture = toolchain.get("architecture")
    torch_version = toolchain.get("version")
    frontend = toolchain.get("metal_frontend")
    device: Dict[str, Any] = {"name": toolchain.get("device_name"),
                              "architecture": architecture}
    if architecture is None:
        device["unreadable"] = toolchain.get("architecture_error") or "unknown"
    else:
        device["compute_capability"] = architecture
    return {
        "backend": getattr(getattr(grid, "xp", None), "__name__", None),
        "table": TABLE,
        "device": device,
        "torch": torch_version,
        "mps_built": toolchain.get("built"),
        "mps_available": toolchain.get("available"),
        "mps_compile_shader": toolchain.get("compile_shader"),
        "metal_frontend": frontend,
        "machine": toolchain.get("machine"),
        "device_supported": device_supported(toolchain),
        "device_certified": environment_verdict("architecture", architecture),
        "torch_certified": environment_verdict("torch", torch_version, architecture),
        "frontend_certified": environment_verdict("metal_frontend", frontend,
                                                  architecture),
        "fast_math": toolchain.get("fast_math"),
        "fast_math_certified": toolchain.get("fast_math") in (None, "0"),
        "recorded_environments": recorded_environments(),
        "certified_only": certified_only(),
    }


#: WHY AN UNCERTIFIED METAL ENVIRONMENT RAN, in the words the run's NOTE line uses.
#: The Metal table runs by default on an environment the cited welds did not record,
#: and says so; ``fastpath.UNCERTIFIED_SWITCH=0`` restricts it to certified ones. On
#: an Apple GPU the reason is that every Apple GPU is SUPPORTED: the kernels are
#: meant to run there and only have not been measured there.
SUPPORTED_BECAUSE = ("this Apple GPU is supported, and {switch}=0 would restrict the "
                     "Metal kernels to certified environments")
UNCERTIFIED_BECAUSE = ("the Metal table runs on environments outside the certified "
                       "set unless {switch}=0")


def device_supported(toolchain: Mapping[str, Any]) -> Optional[bool]:
    """Is this GPU one the Metal table SUPPORTS: an Apple GPU? THREE-VALUED.

    SUPPORTED IS NOT CERTIFIED. Every Apple GPU is supported: the same Metal source
    compiles for each of them and the kernels are meant to run there. Certified is
    narrower: the environment is one every cited weld measured
    (:func:`environment_verdict`). Read off the architecture Metal names
    (``applegpu_*``) and, where the architecture cannot be read (before macOS 14),
    off the device name. ``None`` when neither was read.
    """
    architecture = toolchain.get("architecture")
    if architecture:
        return str(architecture).startswith("applegpu_")
    name = toolchain.get("device_name")
    if name:
        return str(name).startswith("Apple ")
    return None


def certified_only() -> bool:
    """Does this run restrict the Metal kernels to certified environments? Never raises.

    ``True`` for ``fastpath.UNCERTIFIED_SWITCH=0`` only. Unset and ``1`` both run an
    uncertified environment, recorded as uncertified; any other value is refused by
    name at rung 3b before this is asked.
    """
    from . import fastpath  # noqa: PLC0415

    return os.environ.get(fastpath.UNCERTIFIED_SWITCH) == "0"


# ---------------------------------------------------------------------------
# The residency mode: how state gets between the host and the device
# ---------------------------------------------------------------------------

#: HOW THE METAL RESIDENCY TRANSPORTS STATE, chosen by the environment and DEFAULT
#: HELD. ``held`` lets the device keep the state between launches and makes the
#: host acquire it -- measured 2026-09-21 on a real pml_2d composition at 2.0 copies
#: a step against 278, byte-identical over all 24 arrays. ``shipped`` is the bracket
#: the Metal kernel gates certified their bytes under: every mirror copied both ways
#: around every launch, the host authoritative at every instant, and 96-98% of the
#: step.
#:
#: HELD IS THE DEFAULT BECAUSE SHIPPED LOSES TO NOT DISPATCHING. The shipped fused
#: step fits ``28.16 + 37.61 N`` ms/step against the array path's ``0.71 + 16.65 N``
#: (N in Mcells; ``the design notes (metal-residency-fix-plan)`` section 1) and the two
#: lines never cross, so a shipped default was a dispatch that was slower than the
#: array path at every measured size. A configuration a hold cannot serve is
#: therefore refused to the ARRAY PATH by name at admission
#: (:func:`_held_admission`), never to shipped.
#:
#: BOTH VALUES STAY EXPLICIT because one tree has to run both: the route campaign
#: pins ``held``, and a run that wants the bracket the kernel gates certified under
#: asks for ``shipped`` by name. Any other spelling is refused by name.
RESIDENCY_ENV = "MEEP_GPU_METAL_RESIDENCY"
RESIDENCY_SHIPPED = "shipped"
RESIDENCY_HELD = "held"
RESIDENCY_MODES = (RESIDENCY_SHIPPED, RESIDENCY_HELD)
RESIDENCY_DEFAULT = RESIDENCY_HELD

#: The seam passes a held composition may leave on the array path. Every one of
#: them reaches a held volume only through a ``host_writes`` door -- the source
#: deposit and the trailing repair (``fill_*``), the metallic wall clear
#: (``zero_metal_*``, ``zero_faces``) and the folded far-ghost fills
#: (``copy_face``) -- so under a hold each moves the cells it touches and nothing
#: more. The held route campaign of 2026-09-25 found ``synced`` inside this set on
#: 72 of 72 residency blocks over 42 cases, and the 2026-09-25 census found 0 of
#: 179 composer plans leaving a whole sub-step on the array path.
HELD_DOOR_WIRED_PASSES: FrozenSet[str] = frozenset({
    "fill_B", "fill_D", "zero_metal_B", "zero_metal_D",
    "fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D",
})


def _residency_mode() -> str:
    """The requested mode, refusing an unknown spelling BY NAME.

    A typo that silently fell back to the default would make a leg record a mode
    it never asked for, which is the one failure a timing or parity artifact cannot
    survive. :func:`decide` reads this at its admission rung, BEFORE rung 8bM
    installs the subnormal policy, and turns the ``ValueError`` into a named
    refusal: raised any later it would escape the ladder as ``internal error`` with
    the policy already installed.
    """
    import os  # noqa: PLC0415

    requested = (os.environ.get(RESIDENCY_ENV) or RESIDENCY_DEFAULT).strip().lower()
    if requested not in RESIDENCY_MODES:
        raise ValueError(
            f"{RESIDENCY_ENV}={requested!r} is not a residency mode; this package "
            f"has {RESIDENCY_MODES}. Refused by name rather than defaulted, because "
            f"a leg that recorded one mode and ran the other would be a false "
            f"artifact")
    return requested


def _engine_owned_names(residency, fields) -> tuple:
    """Mirror names whose host array the ENGINE allocated, not a plan's builder.

    Reachability from ``fields`` is the test, walked shallowly: the primaries and
    auxiliaries are attributes, the polarization arrays hang off the states in
    ``polarizations`` through two mappings and one attribute, and everything else a
    registry holds -- packs, prefixes, weld twins, ADE scratch the plan allocated --
    is unreachable that way and is therefore the plan's.

    Deliberately conservative: a volume this misses stays on the shipped bracket and
    costs copies, while a plan-owned array this wrongly claimed would be sealed
    against its owner's next write.
    """
    engine = set()

    def note(value):
        if value is not None and hasattr(value, "shape"):
            engine.add(id(value))

    for name in dir(fields):
        if name.startswith("__"):
            continue
        try:
            note(getattr(fields, name))
        except Exception:  # noqa: BLE001 - a property that refuses is not engine state
            continue
    for state in (getattr(fields, "polarizations", ()) or ()):
        for attribute in ("P", "P_prev"):
            mapping = getattr(state, attribute, None)
            if isinstance(mapping, dict):
                for value in mapping.values():
                    note(value)
        note(getattr(state, "_scratch", None))
    return tuple(name for name, mirror in residency._mirrors.items()
                 if id(mirror.host) in engine)


def _arm_residency_hold(residency, fields, synced, step_plan=None, mode=None):
    """Put the device in charge of the held set, or leave the shipped bracket alone.

    ``mode`` is the residency mode :func:`decide` already read and admitted at its
    admission rung; ``None`` reads it here (a caller outside the ladder).

    A SYNCED PASS DOES NOT REFUSE THE HOLD, and the first version of this function
    had it the other way round. ``synced`` names the array-path passes this composer
    did not replace and brackets by hand, and refusing on it looked like the safe
    default -- but measured against the route gate it refuses ``pml_2d``, whose only
    synced pass is ``fill_D``, the electric source seam. That is the seam the hold
    was designed around: it is ~6 sparse operations, and every one of them now
    declares itself.

    What makes that safe is not a model of which volumes each pass reaches -- the
    kind of static claim that was measured wrong on 28 of 49 families -- but three
    guards that leave no silent path:

    * every host WRITE to a held mirror either goes through ``host_writes.acquire``
      (the two sites a held composition actually reaches: the source inject and the
      trailing deposit repair) or RAISES on the seal. There is no third outcome;
    * every host READ through a ``Fields`` attribute or a polarization mapping
      acquires first, so it cannot return stale words;
    * anything else -- a walk over an instance ``__dict__``, a view taken before the
      hold was armed -- is named in ``barrier``'s own documentation and is why every
      state-collecting caller flushes first.

    So the residual risk is an UNWIRED write site, and that fails closed with a
    traceback at the offending statement rather than as a wrong field. The
    enumeration was done by running a held composition and reading what raised.
    """
    if (mode if mode is not None else _residency_mode()) != RESIDENCY_HELD:
        return (), (), (), []
    # EVERY SEAM PASS IS SPARSE NOW, so no grid refuses the hold by name: the
    # source deposit, the trailing repair and the metallic wall zero were made
    # sparse first (plan section 10.6), and the mirror-fold fills, the folded
    # far-ghost fills and the cylindrical axis constraint followed once the 2026-09-23
    # preflight showed that holding a third of the case table on the shipped
    # bracket was a demo, not a result. A site not yet sparse still fails CLOSED --
    # the seal raises at its own line -- so a newly-added host pass is enumerated by
    # its traceback, never by a wrong field.
    from . import host_writes  # noqa: PLC0415
    from .metal_kernels import barrier as barrier_module  # noqa: PLC0415

    # SEAL ONLY WHAT THE ENGINE OWNS. A mirror's host array is either a volume the
    # ENGINE allocated (reachable from ``fields``) or scratch the PLAN allocated in
    # its own builder -- a cylindrical radial prefix, a coefficient pack, a weld
    # twin. The two need opposite treatment, and holding them alike broke the
    # cylindrical rows: ``cylindrical_fused_magnetic_pair.refresh_prefix`` writes
    # ``self._prefix_host[...]`` on the HOST every launch, so sealing it turned an
    # ordinary refresh into ``ValueError: assignment destination is read-only``.
    #
    # Plan-owned scratch is host-written and device-read, every launch, by
    # construction -- there is nothing to hold, and no host read of it can ever be
    # stale because the plan just wrote it. So it stays on the shipped bracket,
    # which costs one small copy a launch and cannot be wrong. The engine's volumes
    # are the ones worth holding and the only ones the barrier can reach anyway.
    engine_owned = _engine_owned_names(residency, fields)
    # CONSTANTS ARE HOISTED REGARDLESS OF WHO OWNS THEM. The engine-owned rule
    # exists to keep plan-owned SCRATCH off the hold (the cylindrical prefix is
    # host-written every launch), and it is right about that -- but it also
    # excluded every mirror that lives on ``pml`` rather than ``fields``: the PML
    # coefficient vectors and the inverse-epsilon volumes, all registered
    # ``constant=True``. Unheld, they fall to the shipped ``sync_in``, which has
    # no constant check, so every launch re-uploaded all of them. Measured
    # 2026-09-22 on ``pml_3d`` at 512k cells: 84.6 copies IN per step, of which
    # the seam passes accounted for about 8 -- the other ~76 were 27 constants
    # times 3 launches. A constant is safe to hoist whoever allocated it: the
    # device never writes it, and the seal turns a post-freeze host rewrite into
    # an exception rather than a stale device.
    constants = tuple(name for name, mirror in residency._mirrors.items()
                      if mirror.constant)
    refused = residency.arm_hold(tuple(dict.fromkeys(engine_owned + constants)))
    host_writes.register(residency)
    barriered = barrier_module.install_read_barrier(fields, residency,
                                                   residency.names)
    # ``reset`` RELEASES FIRST. ``Fields.reset`` zeroes every field array IN PLACE
    # (``array.fill(0)``), so under a hold it raised ``assignment destination is
    # read-only`` at its first array -- and it runs BEFORE the driver's own
    # ``invalidate_fast_path``, so no freeze-time release can reach it. The layer
    # is the one in-place engine mutator that needs this: the enumeration that
    # called every driver mutator under a hold found ``reset`` the only one raising
    # at the mutator itself (the rest rebind, re-plan, and are released at the next
    # freeze by :func:`_release_prior_hold`). ``remove_read_barrier`` restores the
    # pre-swap class, which drops this layer together with the barrier.
    if barriered:
        fields.__class__ = _reset_layer(type(fields))
    # THE POLARIZATION MAPPINGS NEED THEIR OWN BARRIER and leaving this out cost
    # every dispersive row in the first held campaign (7 of 42 cases DIVERGED).
    # ``PolarizationState`` is a plain object that is not a ``Fields`` attribute,
    # and its P arrays live in DICTS -- ``dispersion.subtract_into`` reads
    # ``self.P[component]`` every step on the deposit-repair path. The class swap
    # above reaches neither: a property is an attribute descriptor and cannot
    # intercept a subscript. Without this the host reads pre-launch polarization
    # words and the field comes out smooth, plausible and wrong.
    dict_barriers = 0
    polarization_report = []
    by_host = {id(m.host) for m in residency._mirrors.values()}
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        replaced = barrier_module.install_dict_barrier(state, residency)
        dict_barriers += len(replaced)
        # RECORDED, because "the dict barrier installed nothing" and "this row has
        # no polarization dicts" are different facts and the first held campaign
        # could not tell them apart: seven dispersive rows diverged and the
        # artifact said only ``read_barrier: 26``, every entry a Fields attribute.
        mapping = getattr(state, "P", None)
        polarization_report.append({
            "index": index,
            "type": type(state).__name__,
            "replaced": list(replaced),
            "p_is_mapping": isinstance(mapping, dict),
            "p_keys": sorted(mapping) if isinstance(mapping, dict) else None,
            "p_values_mirrored": (
                sum(1 for v in mapping.values() if id(v) in by_host)
                if isinstance(mapping, dict) else None),
            "has_scratch": hasattr(state, "_scratch"),
            "scratch_mirrored": (id(getattr(state, "_scratch", None)) in by_host),
        })
    # NO ATTRIBUTE IS ATTACHED TO THE PLAN. ``MetalStepPlan`` declares ``__slots__``,
    # so stashing the residency on it raises AttributeError INSIDE the freeze -- and
    # a raise there does not merely fail the row: the dispatch refuses to the array
    # path with the subnormal policy already installed, leaving the host FPU in a
    # state the next freeze then refuses on as well. Callers reach the residency
    # through ``finish(..., residency=...)``, which is where it already was.
    held = tuple(sorted(residency.held))
    return (held, refused,
            tuple(barriered) + tuple(f"dict:{n}" for n in range(dict_barriers)),
            polarization_report)


def release_residency_hold(residency, fields) -> tuple:
    """Flush the device's words back, drop EVERY guard, hand the host authority back.

    EVERY CALLER THAT READS STATE OUTSIDE AN ATTRIBUTE MUST CALL THIS FIRST. The
    read barrier covers attribute access and the dict barrier covers the
    polarization mappings, but a walk over an instance ``__dict__`` -- which is how
    the end-to-end byte comparison collects state -- reads straight out of the
    mapping with no descriptor and would compare host arrays the device has moved
    past. Flushing first is what makes that comparison mean anything under a hold.

    THE RELEASE IS FULL: the ``Fields`` class swap AND each polarization state's
    dict barrier and ``_scratch`` attribute barrier. Stripping only the first was
    measured silently wrong on ``dispersive_2d`` (200 + 200 steps, reference max|P|
    0.768): the ``P``/``P_prev`` mappings stayed ``BarrierDict``\\ s bound to the
    released residency, ``install_dict_barrier`` skips a mapping that already is
    one, so the next hold installed no dict barrier of its own and the host read
    pre-launch polarization words -- 1239 of 4800 words wrong on each of four
    ``P``/``P_prev`` arrays after an ``invalidate_fast_path`` re-arm, and on three
    after a ``reset``. With the strip, both are byte-identical.

    ORDER: flush first, then strip. ``remove_dict_barrier`` hands the mapping back
    through :meth:`BarrierDict.plain`, which acquires every value, and that is a
    no-op only once nothing is ``DEVICE_OWNED``.

    ONLY THIS RESIDENCY'S GUARDS ARE STRIPPED. A barrier bound to a different
    residency is that residency's to release; stripping it here would leave a live
    hold with no read barrier. Safe to call twice (the second call flushes nothing
    and finds no guard bound to it), and ``fields=None`` -- a driver after
    ``close()`` -- releases the residency and strips nothing.
    """
    from . import host_writes  # noqa: PLC0415
    from .metal_kernels import barrier as barrier_module  # noqa: PLC0415

    moved = residency.flush()
    residency.release_hold()
    host_writes.unregister(residency)
    if fields is None:
        return moved
    key = barrier_module._BARRIER_RESIDENCY  # noqa: SLF001
    namespace = getattr(fields, "__dict__", None) or {}
    for state in (namespace.get("polarizations") or ()):
        attributes = tuple(
            attribute for attribute in ("P", "P_prev")
            if isinstance(getattr(state, attribute, None), barrier_module.BarrierDict)
            and getattr(state, attribute)._residency is residency)  # noqa: SLF001
        if attributes:
            barrier_module.remove_dict_barrier(state, attributes)
        if (getattr(state, "__dict__", None) or {}).get(key) is residency:
            barrier_module.remove_read_barrier(state)
    if namespace.get(key) is residency:
        barrier_module.remove_read_barrier(fields)
    return moved


#: The ``reset`` layer over each generated barrier class, built once per class so a
#: freeze never mints a new type.
_RESET_LAYERS: Dict[Any, Any] = {}


def _reset_layer(barriered: Any) -> Any:
    """``barriered`` with one override, ``reset``: release the hold fully, then reset.

    THE ONLY OVERRIDE IS ``reset``. Every other engine mutator either rebinds (the
    barrier setter handles it), re-plans (the next freeze's
    :func:`_release_prior_hold` handles it), or is refused after the first step by
    the driver API on the array path too.

    THE PROPERTIES ARE COPIED INTO THIS LAYER'S OWN NAMESPACE, because
    ``barrier.barriered_names`` reads ``type(instance).__dict__`` only; a layer
    carrying ``reset`` alone would make it report ``()`` for a barriered instance.
    Same class name as the barrier class, so reprs and tracebacks read as before.
    """
    layer = _RESET_LAYERS.get(barriered)
    if layer is not None:
        return layer
    from .metal_kernels import barrier as barrier_module  # noqa: PLC0415

    original = barriered.__bases__[0]
    key = barrier_module._BARRIER_RESIDENCY  # noqa: SLF001

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        residency = self.__dict__.get(key)
        if residency is not None:
            # Restores ``__class__`` to ``original`` as part of the release.
            release_residency_hold(residency, self)
        return original.reset(self, *args, **kwargs)

    namespace: Dict[str, Any] = {
        name: value for name, value in barriered.__dict__.items()
        if isinstance(value, property)}
    namespace["reset"] = reset
    namespace["__module__"] = barriered.__module__
    namespace["__qualname__"] = barriered.__qualname__
    namespace["__doc__"] = (
        f"{barriered.__doc__} Its reset releases the hold before zeroing; layered "
        f"by meep_gpu.metal_dispatch and removed with the barrier.")
    layer = type(barriered.__name__, (barriered,), namespace)
    _RESET_LAYERS[barriered] = layer
    return layer


def _engine_arrays_raw(fields: Any) -> List[Tuple[str, Any]]:
    """This ``Fields``' own arrays, read RAW: no barrier fires, nothing is acquired.

    The instance ``__dict__`` (an attribute, or a value of a mapping attribute) and
    each polarization state's ``P``/``P_prev`` values and ``_scratch``, read through
    ``__dict__`` and ``dict.items`` so neither the class-swap properties nor a
    ``BarrierDict`` fires a whole-volume acquire on the way.
    """
    out: List[Tuple[str, Any]] = []
    seen = set()

    def note(label: str, value: Any) -> None:
        if value is None or id(value) in seen:
            return
        if hasattr(value, "shape") and hasattr(value, "flags"):
            seen.add(id(value))
            out.append((label, value))

    namespace = getattr(fields, "__dict__", None)
    if not isinstance(namespace, dict):
        return out
    for name, value in namespace.items():
        if isinstance(value, dict):
            for key, item in dict.items(value):
                note(f"{name}[{key!r}]", item)
        else:
            note(name, value)
    for index, state in enumerate(namespace.get("polarizations") or ()):
        state_namespace = getattr(state, "__dict__", None) or {}
        for attribute in ("P", "P_prev"):
            mapping = state_namespace.get(attribute)
            if isinstance(mapping, dict):
                for key, item in dict.items(mapping):
                    note(f"polarizations[{index}].{attribute}[{key!r}]", item)
        note(f"polarizations[{index}]._scratch", state_namespace.get("_scratch"))
    return out


def _release_prior_hold(fields: Any, why: str) -> Optional[Dict[str, Any]]:
    """Release, fully, every live hold on this ``Fields``' arrays; report what it did.

    CALLED AT THE TOP OF EVERY METAL FREEZE, before the new ``Residency`` exists.
    ``invalidate_fast_path`` -- the funnel ``set_field``, ``set_epsilon*`` and
    ``add_source`` all pass through -- drops the plan and releases nothing, so the
    next freeze met engine arrays the old residency still sealed: ``arm_hold``
    refused them as unsealable (1 name held in the ``set_field`` probe, 0 after
    ``invalidate_fast_path``), they fell to the shipped bracket, and its copy-out
    raised ``assignment destination is read-only`` at the next launch. A freeze is
    the deterministic release point: no launch is in flight, and the dropped plan
    cannot be relied on to die (a ``weakref.finalize`` on it was measured never to
    fire -- the first plan in a process is pinned by the frames a lazy import's
    stored traceback keeps alive).

    THE PRIOR RESIDENCY IS FOUND TWO WAYS, and both are needed:

    * ``barrier_key`` -- the residency the ``Fields`` read barrier acquires through;
    * ``host_writes_identity`` -- any residency the door registry still holds that
      mirrors one of this ``Fields``' arrays BY IDENTITY, which finds a hold whose
      barrier installed no names and so set no key.

    Keyed on this ``Fields``' own arrays, so a hold armed by another driver is never
    touched.

    A THIRD CASE IS DETECTED AND DELIBERATELY NOT HEALED: an array whose write flag
    is down with no live residency holding it. That happens only when a residency
    died without a release -- its barrier key removed or replaced -- and its device
    words died with it. Unsealing such arrays was MEASURED to turn a loud failure
    into a silent one: a probe that killed a held residency that way and unsealed
    the orphans completed without raising and differed from the array path on 12 of
    32 arrays (``pml_2d``, 10 + 10 steps; 634-669 of 6000 words each) and on 14 of
    41 (``dispersive_2d``, 150 + 50). So they are reported under
    ``sealed_without_a_live_holder``, the admission rung refuses the hold on them by
    name, and the array path raises at its first write to one rather than computing
    on stale words.

    Returns ``None`` when there was nothing to release or report.
    """
    from . import host_writes  # noqa: PLC0415

    if fields is None:
        return None
    arrays = _engine_arrays_raw(fields)
    candidates: List[Tuple[Any, List[str]]] = []

    def admit(residency: Any, route: str) -> None:
        if residency is None:
            return
        for known, routes in candidates:
            if known is residency:
                routes.append(route)
                return
        candidates.append((residency, [route]))

    # The barrier module is imported only by an armed hold, so a process that never
    # held has no key to look for -- and this rung sits above the torch checks.
    barrier_module = sys.modules.get(f"{__package__}.metal_kernels.barrier")
    if barrier_module is not None:
        namespace = getattr(fields, "__dict__", None) or {}
        admit(namespace.get(barrier_module._BARRIER_RESIDENCY),  # noqa: SLF001
              "barrier_key")
    for residency in host_writes._live():  # noqa: SLF001
        if any(residency._by_host_identity(array) is not None  # noqa: SLF001
               for _label, array in arrays):
            admit(residency, "host_writes_identity")

    released: List[Dict[str, Any]] = []
    for residency, routes in candidates:
        held_before = len(getattr(residency, "held", ()) or ())
        hoisted_before = len(getattr(residency, "hoisted", ()) or ())
        moved = release_residency_hold(residency, fields)
        released.append({"found_by": routes, "held_before": held_before,
                         "hoisted_before": hoisted_before,
                         "names_flushed": list(moved)})

    live = host_writes._live()  # noqa: SLF001
    orphaned = []
    for label, array in arrays:
        flags = getattr(array, "flags", None)
        if flags is None or flags.writeable:
            continue
        if any(mirror.host is array and mirror.sealed
               for residency in live
               for mirror in residency._mirrors.values()):  # noqa: SLF001
            continue
        orphaned.append(label)

    if not released and not orphaned:
        return None
    return {
        "why": why,
        "released": released,
        "names_flushed": sorted({name for entry in released
                                 for name in entry["names_flushed"]}),
        "sealed_without_a_live_holder": sorted(orphaned),
    }


def _held_admission(residency: Any, fields: Any, synced: Sequence[str],
                    released_prior: Optional[Mapping[str, Any]]
                    ) -> Tuple[List[str], Dict[str, Any]]:
    """Rung 8aM: may THIS composition be held? Returns ``(refusals, block)``.

    Asked only when the mode is ``held``, and BEFORE rung 8bM installs anything, so a
    refused configuration never has its arithmetic moved. Every refusal lands on the
    ARRAY PATH, never on shipped: shipped is slower than the array path at every
    measured size, so a hold refusal that fell to it would be a slowdown presented
    as safety.

    (a) THE SYNCED PASSES must lie inside :data:`HELD_DOOR_WIRED_PASSES`. A PERFORMANCE
        guard, not a correctness one: an array-path pass outside the set writes a held
        volume through ``stepping._acquire_written``, a whole-volume acquire per
        sub-step, and the array path under a live hold was measured byte-identical.
        Refuses 0 of 179 census rows and 0 of 42 route cases today.
    (b) EVERY ENGINE-OWNED MIRROR MUST BE SEALABLE, checked dry (``Mirror.sealable``
        round-trips the write flag and leaves it up). ``arm_hold`` would refuse such a
        name and leave it on the shipped bracket, whose copy-out writes the host
        array; when the array is one a dead residency left sealed that copy-out
        raises mid-step, after the launch.
    (c) NO POLARIZATION MAPPING MAY ALREADY BE A ``BarrierDict`` bound to a residency
        other than this one. ``install_dict_barrier`` skips such a mapping, so the
        new hold would get no dict barrier and read stale ``P`` -- the partial
        release's silent failure. The backstop that turns a regressed release into a
        named refusal.
    """
    from .metal_kernels import barrier as barrier_module  # noqa: PLC0415

    refusals: List[str] = []
    outside = sorted(set(synced) - HELD_DOOR_WIRED_PASSES)
    if outside:
        refusals.append(
            f"held residency admits only array-path seam passes that reach a held "
            f"volume through a host_writes door ({', '.join(sorted(HELD_DOOR_WIRED_PASSES))}); "
            f"this composition leaves {', '.join(outside)} on the array path, which "
            f"under a hold would acquire whole volumes every sub-step. A performance "
            f"guard, not a correctness one (the array path under a live hold is "
            f"measured byte-identical); refused to the array path rather than to "
            f"shipped, which is slower than the array path at every measured size. "
            f"{RESIDENCY_ENV}={RESIDENCY_SHIPPED} dispatches it on the shipped "
            f"bracket anyway")

    engine_owned = _engine_owned_names(residency, fields)
    unsealable = sorted(name for name in engine_owned
                        if not residency._mirrors[name].sealable())  # noqa: SLF001
    if unsealable:
        orphaned = list((released_prior or {}).get("sealed_without_a_live_holder") or ())
        detail = (f"; {', '.join(orphaned)} carry a seal no live residency holds -- "
                  f"a hold that ended without a release, whose device words are "
                  f"gone -- so the array path will raise at its first write to one "
                  f"rather than compute on them" if orphaned else "")
        refusals.append(
            f"held residency cannot seal the engine-owned mirror(s) "
            f"{', '.join(unsealable)}: arm_hold would leave them on the shipped "
            f"bracket beside a hold, and that bracket's copy-out writes the host "
            f"array{detail}")

    foreign = []
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for attribute in ("P", "P_prev"):
            mapping = getattr(state, attribute, None)
            if (isinstance(mapping, barrier_module.BarrierDict)
                    and mapping._residency is not residency):  # noqa: SLF001
                foreign.append(f"polarizations[{index}].{attribute}")
    if foreign:
        refusals.append(
            f"the polarization mapping(s) {', '.join(foreign)} still carry a read "
            f"barrier bound to another residency -- a prior hold was released only "
            f"in part -- and install_dict_barrier skips such a mapping, so this hold "
            f"would read stale P (measured on dispersive_2d: 1239 of 4800 words "
            f"wrong on each of four P/P_prev arrays)")

    block = {"rules": ["synced_within_door_wired_set", "engine_owned_sealable",
                       "no_foreign_polarization_barrier"],
             "synced_outside_door_wired_set": outside,
             "unsealable_engine_owned": unsealable,
             "foreign_polarization_barriers": foreign,
             "refused": bool(refusals)}
    return refusals, block


# ---------------------------------------------------------------------------
# The composition window: what this dispatch knows that the composer cannot read
# ---------------------------------------------------------------------------

@contextmanager
def _composition_window(drops: Sequence[Tuple[str, str, Sequence[str]]]
                        ) -> Iterator[Dict[str, Any]]:
    """Declare this dispatch's policy, and stand its probe refusals up, for the composition.

    TWO FACTS THE RUNGS BELOW CANNOT READ FOR THEMSELVES, and the Triton window
    exists for the same two.

    THE POLICY. Rung 8bM installs :data:`SUBNORMAL_POLICY` and refuses any process
    that installed anything else, so which arithmetic this dispatch will use is a
    fact it already owns — but it is not INSTALLED yet at the composer rung, and
    nothing under ``plan_step`` can read an intention. On this table the
    declaration is not a nicety: ``metal_kernels.subnormal.mps_policy_reasons``
    gates the whole Metal backend clause, and this arm64 host's own resolution is
    ``keep``, so WITHOUT the declaration the composer selects NOTHING on any case —
    measured on all fourteen driver-route cases, every reason being that one.

    THE REFUSALS. A probe artifact that is not RELEASED is dropped as a refusal
    VALUE rather than as ``None``, and the refusal is also recorded against that
    family's own environment variable for the duration. Refusal and absence deserve
    opposite outcomes: nothing offered may fall back to the environment, something
    offered and refused may not — and the rung that broke that invariant on the
    Triton track was one that read the variable rather than the argument.

    Restored on the way out, exception included, so neither fact leaks into
    unrelated later work in this process.
    """
    from .expansion_refusal import (  # noqa: PLC0415
        declaring_run_policy, refusing_expansion_probe)

    with ExitStack() as scope:
        scope.enter_context(declaring_run_policy(certification_policy()))
        refusals: Dict[str, Any] = {}
        for keyword, env, reasons in drops:
            refusals[keyword] = scope.enter_context(
                refusing_expansion_probe(env, tuple(reasons)))
        yield refusals


def _load_probes(record: Dict[str, Any]) -> Tuple[Dict[str, Any],
                                                  List[Tuple[str, str, List[str]]]]:
    """Rung 4cM: the four expansion licences, read and judged. Never raises.

    Returns ``(probes handed to the composer, the drops the window must stand up)``.
    A probe that is ABSENT stays ``None`` — the composer's own by-name refusal then
    reports which family could not bind its multiply arm, which is the array path
    for that family and no change at all for a run with no complex storage.
    """
    probes: Dict[str, Any] = {keyword: None for keyword, _ in PROBE_FAMILIES}
    drops: List[Tuple[str, str, List[str]]] = []
    block: Dict[str, Any] = {}
    for keyword, module_name in PROBE_FAMILIES:
        entry: Dict[str, Any] = {}
        try:
            module = importlib.import_module(
                f".metal_kernels.{module_name}", __package__)
            env = module.PROBE_PATH_ENVIRONMENT
            entry["variable"] = env
            entry["path"] = os.environ.get(env)
            artifact = module.load_expansion_probe()
        except Exception as exc:  # noqa: BLE001 - an unreadable licence is a refusal
            entry["error"] = repr(exc)
            entry["state"] = "unreadable"
            block[module_name] = entry
            continue
        if artifact is None:
            entry["state"] = "absent" if not entry.get("path") else "unreadable"
            block[module_name] = entry
            continue
        release = artifact.get("release") if isinstance(artifact, Mapping) else None
        released = isinstance(release, Mapping) and release.get("released") is True
        entry["release"] = None if release is None else dict(release)
        if released:
            entry["state"] = "released"
            probes[keyword] = artifact
        else:
            # DROPPED, NOT IGNORED. An expansion artifact whose own runner did not
            # release it has no licence to bind a multiply arm, and handing back
            # ``None`` would let the rung below re-read the same file off the
            # environment and consume it anyway.
            reasons = [
                f"the expansion probe at {entry.get('path')!r} carries no released "
                f"verdict (release={entry['release']!r}); a licence its own gate "
                f"runner did not release cannot bind an expansion arm"]
            entry["state"] = "dropped"
            entry["dropped_because"] = reasons
            drops.append((keyword, entry["variable"], reasons))
        block[module_name] = entry
    record.setdefault("environment", {})["metal_expansion_probes"] = block
    return probes, drops


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def decide(record: Dict[str, Any], fields: Any, pml: Any, grid: Any,
           sources: Any, xp: Any, finish: Any) -> Any:
    """Rungs 4M-9M: whether Metal kernels may serve this frozen configuration.

    Reached from ``fastpath._decide``'s backend rung once
    ``fastpath.candidate_tables`` has answered ``("metal",)``. Rungs 1-3 — the kill
    switch, the enable and the backend — are that ladder's and are not repeated
    here.

    EVERY RUNG REFUSES TO THE ARRAY PATH WITH A NAMED REASON AND NONE OF THEM
    RAISES, which is the contract the Triton ladder states and the one the driver
    depends on: after the freeze there is NO runtime fallback, so a plan that
    reaches ``dispatch`` must be one every rung has already judged.

    ``finish`` IS ``fastpath._finish`` — the factored tail both ladders emit
    through (slot ordering, the split-pair refusal, the certification lookup, the
    composition block, the record). It is passed rather than imported so the two
    ladders cannot drift on it, and so a probe can drive this ladder with its own
    tail.
    """
    from . import fastpath  # noqa: PLC0415

    record["table"] = TABLE

    # (4aM) THE PRIOR HOLD, RELEASED BEFORE ANYTHING ELSE. A re-freeze of a Fields
    # some earlier plan held must start from host-authoritative, unsealed arrays --
    # see :func:`_release_prior_hold`. First, so that every refusal below also
    # leaves the array path nothing sealed; and it never raises, like every rung.
    try:
        released_prior = _release_prior_hold(fields, "a Metal freeze")
    except Exception as exc:  # noqa: BLE001 - a rung refuses, it does not raise
        fastpath._refuse(record,  # noqa: SLF001
                         f"releasing the prior hold on this Fields raised {exc!r}; "
                         "a hold that could not be released cannot be re-armed")
        return None
    if released_prior is not None:
        record.setdefault("residency", {})["released_prior_hold"] = released_prior

    from .metal_kernels import device, launch  # noqa: PLC0415

    environment = record.setdefault("environment", {})

    # (4M) THE TOOLCHAIN, AVAILABLE. torch supplies the launch path, so the three
    # availability reads below are refusals. Which environment the kernels run in
    # is judged after them.
    toolchain = metal_toolchain()
    environment.update(environment_block(grid, toolchain))
    if not toolchain["importable"]:
        fastpath._refuse(record,  # noqa: SLF001
                         "torch is not importable on this host "
                         f"({toolchain.get('import_error')}); the Metal kernels "
                         "launch through torch.mps.compile_shader")
        return None
    if not toolchain.get("built"):
        fastpath._refuse(record, "this torch build was not built with MPS")  # noqa: SLF001
        return None
    if not toolchain.get("available"):
        fastpath._refuse(record, "no MPS device is available on this host")  # noqa: SLF001
        return None
    if not toolchain.get("compile_shader"):
        fastpath._refuse(record,  # noqa: SLF001
                         "torch.mps.compile_shader is missing; hand-written Metal "
                         "sources cannot be compiled on this build")
        return None
    # (4M) THE ENVIRONMENT: GPU architecture, torch and Metal frontend, each judged
    # against the live runs every cited weld recorded (:func:`environment_verdict`):
    # the architecture by whether each weld has a run for it, torch and the frontend
    # against those runs. Fast math is certified only off. BY DEFAULT AN UNCERTIFIED ENVIRONMENT RUNS: a fact
    # the welds contradict is recorded under ``uncertified.admitted`` (copied to
    # ``uncertified.served`` when the table serves), so the record says
    # ``certified: False`` and the run prints one NOTE line naming it.
    # ``certified_only()`` (``UNCERTIFIED_SWITCH=0``) refuses instead, and refuses a
    # fact it could not judge as well: a run restricted to certified environments
    # does not run on one it cannot identify.
    architecture = toolchain.get("architecture")
    # THE NUMBER OF CITED WELDS, not the sum of the rows: a weld with runs for two
    # architectures is one row per architecture.
    cited = len(cited_environments())
    strict = certified_only()
    switch = fastpath.UNCERTIFIED_SWITCH
    # WHETHER THIS TABLE MAY RUN AN UNCERTIFIED ENVIRONMENT, as the record's switch
    # block states it for the NVIDIA tables (``1`` only); here, anything but ``0``.
    record.setdefault("uncertified", {})["allowed"] = not strict
    for fact, what, verdict_key, read in (
            ("architecture", "GPU architecture", "device_certified",
             toolchain.get("architecture")),
            ("torch", "torch", "torch_certified", toolchain.get("version")),
            ("metal_frontend", "Metal frontend", "frontend_certified",
             toolchain.get("metal_frontend")),
            ("fast_math", FAST_MATH, "fast_math_certified", toolchain.get("fast_math"))):
        verdict = environment.get(verdict_key)
        if verdict is True:
            continue
        # WHAT IS CERTIFIED, as the NOTE line and a refusal name it: only values every
        # cited weld certifies (:func:`certified_values`), so a partial round never
        # lists the GPU it has just found not certified. torch and the frontend are
        # certified PER ARCHITECTURE, so theirs are read off this GPU's runs.
        values = (["unset"] if fact == "fast_math" else
                  certified_values(fact, architecture))
        if verdict is False:
            if strict:
                cause = (f"{FAST_MATH}={read} compiles the Metal kernels in fast-math "
                         "mode, which no Metal weld ran" if fact == "fast_math" else
                         f"{what} {read} is not the one every Metal weld this table "
                         f"cites ran on (certified: {values or 'none'})")
                fastpath._refuse(record,  # noqa: SLF001
                                 f"{cause}; {switch}=0 restricts the Metal kernels to "
                                 "certified environments")
                return None
            because = (SUPPORTED_BECAUSE if environment.get("device_supported")
                       else UNCERTIFIED_BECAUSE)
            fastpath._admit_uncertified(  # noqa: SLF001
                record, TABLE, what, read, values, because=because.format(switch=switch))
            continue
        if strict:
            if read is None:
                cause = (f"the {what} could not be read on this host"
                         + (f" ({toolchain.get('architecture_error')})"
                            if fact == "architecture"
                            and toolchain.get("architecture_error") else ""))
            else:
                silent = sum(1 for _gate, judged in environment_judgements(
                    fact, read, architecture) if judged is None)
                cause = (f"{what} {read} cannot be certified: {silent} of the {cited} "
                         + ("Metal welds this table cites hold no per-architecture "
                            "run record to judge it by" if fact == "architecture" else
                            f"Metal welds this table cites record no {what} in a live "
                            f"run for {architecture or 'this GPU architecture'}"))
            fastpath._refuse(record,  # noqa: SLF001
                             f"{cause}; {switch}=0 restricts the Metal kernels to "
                             "certified environments, and an environment that cannot "
                             "be judged is not one")
            return None

    record["subnormal"] = fastpath._subnormal_block()  # noqa: SLF001
    # THE BLOCK IS THE TRITON CONSTANT'S until ``fastpath._subnormal_block`` grows
    # the table argument, so this table's own answer is recorded beside it rather
    # than left to be read wrong. WAITS ON ``fastpath.TABLE_SUBNORMAL_POLICY``.
    record["subnormal"]["certification_policy"] = certification_policy()
    record["subnormal"]["governed_executors"] = list(GOVERNED_EXECUTORS)
    record["run_shape"] = fastpath._run_shape(fields, pml, grid)  # noqa: SLF001

    # (4cM) THE EXPANSION LICENCES, before the composer consumes them and four
    # rungs before rung 8bM installs anything.
    probes, drops = _load_probes(record)

    # (5M) THE COMPOSER. Never raises and never returns None; a configuration
    # nothing covers is a plan that replaces nothing, which is the array path.
    opted_in = fastpath.requested_fused_arms()
    vetoed = fastpath.fused_arms_vetoed()
    envelope_reasons = fused_release_reasons_metal(record["run_shape"])
    released = released_fused_arms_metal(record["run_shape"])
    admitted = () if vetoed else tuple(sorted(set(opted_in) | set(released)))
    fusion = record.setdefault("fusion", {})
    fusion["driver_route_gate"] = METAL_DRIVER_ROUTE_GATE
    fusion["released_by_the_gate"] = sorted(METAL_RELEASED_FUSED_ARMS)
    fusion["released_here"] = list(released)
    fusion["admitted"] = list(admitted)
    fusion["outside_the_released_envelope"] = list(envelope_reasons)
    per_arm = {arm: list(fused_release_arm_reasons_metal(record["run_shape"], arm))
               for arm in sorted(METAL_RELEASED_FUSED_ARMS)}
    per_arm = {arm: why for arm, why in per_arm.items() if why}
    if per_arm and not envelope_reasons:
        fusion["outside_this_arms_own_cases"] = per_arm

    # THE RESIDENCY IS BUILT HERE AND OWNED BY THE PLAN. Every mirror the
    # composition binds is registered against this object, and the bracket
    # installed below is what keeps the HOST authoritative between launches.
    residency = device.Residency()
    with _composition_window(drops) as refusals:
        probes.update(refusals)
        # THE TWO-PASS SCOUT, AND THE SECOND PASS IS NOT REDUNDANT. The residency
        # verdict needs the SYNCED set — which array-path passes this caller
        # brackets with an explicit sync — and that cannot be known until the
        # composer has said which slots are its own. The first call answers that on
        # a throwaway residency; the second is the composition that actually runs.
        #
        # THE SYNCED SET IS DERIVED THROUGH ``replaces_sub_steps`` AND THAT READ IS
        # LOAD-BEARING (the rule ``gate_metal_whole_step.covered_passes`` states): a
        # fill plan occupies ONE slot and may replace TWO passes — ``fill_B`` and
        # ``fill_folded_far_ghosts_B``, because the driver runs them either side of
        # the wall pass. "Live minus the slots the composer filled" would declare
        # the far pass array-path, bracket a pass that never ran on the host, and
        # report a composition weaker than the one it executed.
        scout = launch.plan_step(fields, pml, residency=device.Residency(),
                                 sources=sources, fuse=bool(admitted),
                                 fuse_labels=admitted, **probes)
        covered = set()
        for slot, product in scout.plans.items():
            covered.add(slot)
            covered.update(getattr(launch.declaring_plan(product),
                                   "replaces_sub_steps", ()) or ())
        live = launch.live_sub_steps(fields, pml, sources) or ()
        synced = tuple(name for name in live if name not in covered)
        step_plan = launch.plan_step(fields, pml, residency=residency,
                                     sources=sources, synced=synced,
                                     fuse=bool(admitted), fuse_labels=admitted,
                                     **probes)
    selected = dict(getattr(step_plan, "selected", {}) or {})
    filled = set(getattr(step_plan, "plans", {}) or {})
    record["composition_inputs"] = {"live": list(live), "synced": list(synced),
                                    "scouted_passes": sorted(covered)}

    # (8M) NO FUSION EXCEPT BY NAME. Every Metal fused label refuses the WHOLE plan
    # unless the driver-route gate released that exact label for this exact
    # configuration, or this process opted into that exact label. The composition is
    # a cross-sub-step substitution the driver's consults cannot see into, and what
    # licenses one is a gate that drove it through those consults.
    #
    # SINCE THE COMPOSER IS HANDED ``fuse_labels`` THIS IS A BACKSTOP with no live
    # path: an un-admitted product is never installed, so its seam keeps the
    # separate certified arms rather than being refused whole with them. The clause
    # stays because "no composer can reach it today" is a fact about the composer.
    labels = set(fused_labels())
    fused_slots = sorted(slot for slot in filled if selected.get(slot) in labels)
    ungated = sorted({selected[slot] for slot in fused_slots
                      if selected[slot] not in admitted})
    if ungated:
        fastpath._record_slots(record, step_plan, (), {}, {})  # noqa: SLF001
        outside = [arm for arm in ungated if arm in METAL_RELEASED_FUSED_ARMS]
        detail = (f"released by {METAL_DRIVER_ROUTE_GATE}: "
                  f"{', '.join(sorted(METAL_RELEASED_FUSED_ARMS))}")
        if outside and envelope_reasons:
            detail += (f" — but {', '.join(outside)} is released only for the "
                       "configuration that gate drove, and this one is outside it: "
                       + "; ".join(envelope_reasons))
        route = (f"{fastpath.FUSE_ARMS_SWITCH}={fastpath.FUSE_ARMS_VETO} vetoed "
                 "every fused arm, this one included" if vetoed
                 else f"set {fastpath.FUSE_ARMS_SWITCH} to drive one anyway")
        fastpath._refuse(record,  # noqa: SLF001
                         "a fused cross-sub-step product won "
                         f"{', '.join(fused_slots)}; it has not been driven through "
                         "the driver seam on this configuration. Not admitted: "
                         f"{', '.join(ungated)} ({detail}; {route})")
        return None
    if fused_slots:
        fusion["driven"] = {slot: selected[slot] for slot in fused_slots}
        fusion["released_on_cases"] = {
            arm: list(METAL_RELEASED_FUSED_ARMS[arm])
            for arm in sorted({selected[slot] for slot in fused_slots})
            if arm in METAL_RELEASED_FUSED_ARMS}
        driven_arms = sorted({selected[slot] for slot in fused_slots})
        beyond: List[str] = list(envelope_reasons)
        for arm in driven_arms:
            if arm in METAL_RELEASED_FUSED_ARMS:
                beyond.extend(
                    f"{arm}: {reason}" for reason
                    in fused_release_arm_reasons_metal(record["run_shape"], arm))
        if beyond:
            fusion["driven_outside_the_released_envelope"] = {
                "arms": driven_arms, "reasons": beyond,
                "reached_by": fastpath.FUSE_ARMS_SWITCH}

    # (7M) THE NULL DROP, before completeness is judged, and READ OFF THE PLAN
    # rather than matched against a label. ``performs_device_work`` is the
    # composer's own declaration — a null plan runs on neither executor, binds no
    # buffer and writes no host array — and it is read THROUGH the wrapper, because
    # after a pair is installed two slots hold objects that declare nothing of
    # their own and name the launch in ``absorbed_by`` instead. Matching on the
    # arm's label would make a rename a silent correctness change.
    dropped_null = {
        slot: selected.get(slot, "unknown") for slot in sorted(filled)
        if not getattr(launch.declaring_plan(step_plan.plans[slot]),
                       "performs_device_work", True)}
    dispatchable = filled - set(dropped_null)

    # (7aM) THE PENDING RUNG. A fused label is asked about its CONSTITUENTS as well
    # as about itself: a fused product writes one label into both slots it absorbs,
    # so the arms it implements disappear from ``selected`` and a rung reading the
    # label alone would pass over a pending one. A fused label this module cannot
    # name the constituents of is refused here by name — a product no rung below can
    # judge is exactly the product that must not dispatch.
    constituents = fused_arm_constituents()
    selected_labels = {selected.get(slot, "unknown") for slot in dispatchable}
    unmapped_fused = sorted(label for label in selected_labels
                            if label in labels and label not in constituents)
    if unmapped_fused:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
        fastpath._refuse(record,  # noqa: SLF001
                         f"{', '.join(unmapped_fused)} won a slot, and this module "
                         "cannot name the arms it substitutes; the pending-gate "
                         "rung and every arm-level record below it would be "
                         "reporting on an absorb nobody declared "
                         "(metal_dispatch.fused_arm_constituents)")
        return None
    substituted = set(selected_labels)
    for label in selected_labels:
        substituted |= set(constituents.get(label, ()))
    pending = sorted(substituted & set(METAL_PENDING_DEVICE_GATE_ARMS))
    if pending:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
        fastpath._refuse(record, "; ".join(  # noqa: SLF001
            METAL_PENDING_DEVICE_GATE_ARMS[arm] for arm in pending))
        return None

    # (6M) COMPLETENESS. A slot the composer fills that the driver has no consult
    # for at all refuses everything: running the rest would be a composition no gate
    # certified.
    unrunnable = sorted(dispatchable - set(fastpath.DRIVER_SLOTS))
    if unrunnable:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in unrunnable}
        fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
        fastpath._refuse(record, "the composer filled "  # noqa: SLF001
                         f"{', '.join(unrunnable)}, which the driver seam has no "
                         "consult for; running the rest would be a composition no "
                         "gate certified")
        return None

    # (6bM) THE FOLD, read from the GRID rather than inferred from what the composer
    # did — a rule that sniffed for arms named "folded" would decide the fold from
    # the same product selection whose gaps produced the hole, and would answer "not
    # folded" for exactly the case that matters. A folded run must have BOTH fill
    # slots carrying a kernel: a fold whose fills are on the array path while its
    # curls are on kernels is a mixed composition nothing measured, and it is
    # refused whole. The wording carries the phrases the route gate's classifier
    # greps for ("mirror-folded", "fill sub-steps").
    fold = fastpath._fold_description(grid)  # noqa: SLF001
    uncovered_fills = sorted(set(fastpath.FAR_FILL_PASSES) - dispatchable)
    if fold is not None and dispatchable and uncovered_fills:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
        fastpath._refuse(record,  # noqa: SLF001
                         f"the configuration is mirror-folded ({fold}); the driver "
                         "runs fill_symmetry_bc_B/D on every step and no kernel "
                         f"covers the fill sub-steps {', '.join(uncovered_fills)}, "
                         "so dispatching the rest would be a folded composition no "
                         "gate ran")
        return None

    # (8aM) THE RESIDENCY MODE AND, WHEN IT IS HELD, THE HOLD'S ADMISSION -- after
    # every structural refusal above (their wording is what the route gate's
    # classifier reads) and BEFORE rung 8bM, the first rung that changes the
    # process. A misspelled mode read any later escaped the ladder as "internal
    # error" with the subnormal policy already installed; here it is a named
    # refusal on an untouched process. Every refusal lands on the array path.
    try:
        mode = _residency_mode()
    except ValueError as exc:
        fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
        record.setdefault("residency", {})["mode_refused"] = str(exc)
        fastpath._refuse(record, str(exc))  # noqa: SLF001
        return None
    admission: Optional[Dict[str, Any]] = None
    if mode == RESIDENCY_HELD and dispatchable:
        refusals, admission = _held_admission(residency, fields, synced, released_prior)
        if refusals:
            fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
            block = record.setdefault("residency", {})
            block["mode"] = mode
            block["admission"] = admission
            fastpath._refuse(record, "; ".join(refusals))  # noqa: SLF001
            return None

    # (8bM) THE SUBNORMAL POLICY GATE, last rung before anything would compile and
    # first rung that changes the process. It must FOLLOW every structural refusal,
    # because installing a policy moves this process's arithmetic and a
    # configuration that was never going to dispatch has no business having its
    # numerics moved.
    #
    # WAITS ON ``fastpath._subnormal_gate(record, xp, table)`` — the per-table
    # argument that makes ``required`` this table's ``flush`` and ``governed`` this
    # table's ``("host", "mps")``. Until that lands nothing reaches this line,
    # because nothing reaches this module.
    policy_licence: Optional[Tuple[Any, str, int]] = None
    if dispatchable:
        refusal = fastpath._subnormal_gate(record, xp, TABLE)  # noqa: SLF001
        if refusal is not None:
            fastpath._record_slots(record, step_plan, (), dropped_null, {})  # noqa: SLF001
            fastpath._refuse(record, refusal)  # noqa: SLF001
            return None
        from . import subnormal_policy as _policy  # noqa: PLC0415

        policy_licence = (_policy, certification_policy(), _policy.policy_epoch())

    # (9M) THE WARM PASS, RECORDED RATHER THAN ATTEMPTED. On the Triton table the
    # warm pass is what turns a compile failure into a plan-time event; on this one
    # every kernel is compiled inside ``plan_step``'s builders through
    # ``device.compile_source``, so a source that will not compile has ALREADY
    # refused its slot by the time this rung is reached. ``fastpath.warm_plan``
    # would find no settable backing attribute on a Metal ``KernelPlan``
    # (``plans.py`` __slots__ is ``_args``/``_functions``/``launches``) and return a
    # harmless "no settable attribute" reason for every slot, which would read as a
    # gap rather than as the answer it is.
    unwarmed = {slot: "compiled at plan build (metal_kernels.device.compile_source); "
                      "a source that will not compile refuses its slot in plan_step"
                for slot in sorted(dispatchable)}

    # THE RESIDENCY BRACKET, INSTALLED AT PLAN TIME. From here the plans in
    # ``step_plan.plans`` carry their own sync_in/sync_out, so the driver's consult
    # sites, ``FastPathPlan.dispatch`` and ``synchronize_magnetic_fields`` need no
    # residency branch at all: the invariant "the host arrays are authoritative
    # between launches" is a property of the objects rather than of the caller.
    launch.wrap_for_residency(step_plan, residency)
    held_names, hold_refused, barriered, polarization_block = _arm_residency_hold(
        residency, fields, synced, step_plan, mode=mode)
    record["residency"] = {
        "invariant": ("device-authoritative for the held set between launches, "
                      "host-authoritative for every other mirror"
                      if held_names else "host-authoritative between launches"),
        "mode": mode,
        # What the freeze released before it built this residency (None: nothing
        # was held on this Fields), and the hold's admission verdict (None: the
        # mode is shipped). Recorded here rather than on the ``Residency``, which
        # declares ``__slots__``: an attribute there would raise inside the freeze.
        "released_prior_hold": released_prior,
        "admission": admission,
        "mirrors": len(residency.names),
        "mirror_names": list(residency.names),
        "synced_passes": list(synced),
        # COUNTED PER CASE FROM HERE ON. The bracket's cost is per COPY -- 0.217 ms
        # on this host, size-free at campaign widths -- and until now no artifact
        # recorded how many copies a case performed, so every calibration had to be
        # reconstructed from wall-clock deltas. ``Residency`` already counted them;
        # writing them out makes every future campaign a calibration set for free.
        "syncs_in": residency.syncs_in,
        "syncs_out": residency.syncs_out,
        "copies_in": residency.copies_in,
        "copies_out": residency.copies_out,
        "held": list(held_names),
        "hoisted": sorted(residency.hoisted),
        "hold_refused": list(hold_refused),
        "plan_owned_not_held": len(residency.names) - len(held_names)
                               - len(residency.hoisted),
        "read_barrier": list(barriered),
        "polarizations": polarization_block,
    }

    plan = finish(record=record, step_plan=step_plan, dispatchable=dispatchable,
                  dropped_null=dropped_null, unwarmed=unwarmed, selected=selected,
                  table=TABLE, residency=residency, policy_licence=policy_licence)
    # THE TAIL CAN STILL REFUSE AFTER THE HOLD WAS ARMED -- "no slot is left
    # carrying a kernel" and the split-pair refusal both come after this ladder
    # sealed the mirrors and installed the barriers, because the record the tail
    # emits carries the residency block. No plan exists to release that hold, so
    # it is released here: measured on ``pml_2d`` with a forced split-pair refusal,
    # the run otherwise continued on the array path with 3 of 30 engine arrays
    # read-only and 27 of 30 still owned by the dead plan's residency until the
    # next Metal freeze of the same ``Fields``. A shipped residency holds nothing.
    if plan is None and (held_names or residency.hoisted):
        moved = release_residency_hold(residency, fields)
        record["residency"]["released_after_refusal"] = list(moved)
    return plan


# ---------------------------------------------------------------------------
# Self-checks the tests and the board read
# ---------------------------------------------------------------------------

def unresolved_certification_rows() -> Tuple[str, ...]:
    """Rows of :data:`ARM_CERTIFICATION` whose gate key resolves to nothing.

    THE ANTI-FORGERY CHECK, and it is the twin of the Triton suite's
    ``test_every_recorded_certification_points_at_a_readable_record``: a typed row
    citing a weld that does not exist is exactly the record forgery this campaign
    forbids, and it is invisible in an artifact — the arm names a gate, the gate
    names nothing, and only a reader who opens the ledger can tell.
    """
    ledger = fingerprints()
    return tuple(sorted(
        f"{arm}: {gate}" for arm, (_family, gate) in ARM_CERTIFICATION.items()
        if not isinstance(ledger.get(gate), Mapping)))


def release_rows_without_a_weld() -> Tuple[str, ...]:
    """Released labels that are not a registry weld, or carry no certification row.

    The other half of the anti-forgery check: :data:`METAL_RELEASED_FUSED_ARMS` is
    typed, so every label in it must be one the composer can actually write
    (:func:`fused_labels`), must have an absorb declaration the ladder can see
    through (:func:`fused_arm_constituents`), and must name a certification that
    resolves.
    """
    labels = set(fused_labels())
    constituents = fused_arm_constituents()
    problems: List[str] = []
    for arm in sorted(METAL_RELEASED_FUSED_ARMS):
        if arm not in labels:
            problems.append(f"{arm}: no registry weld writes this label")
        if arm not in constituents:
            problems.append(f"{arm}: no FUSED_PAIR_ARMS absorb declaration")
        if arm not in ARM_CERTIFICATION:
            problems.append(f"{arm}: no ARM_CERTIFICATION row")
        if arm not in METAL_FUSED_RELEASE_ARM_AXES:
            problems.append(f"{arm}: no METAL_FUSED_RELEASE_ARM_AXES row")
    return tuple(problems)

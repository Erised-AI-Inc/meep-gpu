"""The dispatch seam: which KERNEL TABLE replaces array sub-steps, and which of them.

THERE ARE THREE TABLES, and the backend rung is where that is decided. A NumPy
engine with an MPS device takes the METAL table, whose rungs and release rows live
in :mod:`meep_gpu.metal_dispatch` for the reason that module's docstring gives —
every gate artifact pins the modules it imports, so a Metal release row in THIS
file would re-cut the Triton record every time it moved. A CuPy engine has TWO
candidates rather than one: the Triton table, whose ladder and release rows stay in
this file, and the hand-CUDA table, whose rows, release and merge rule live in
:mod:`meep_gpu.fastpath_cuda` for the same cost reason. Both compose, in the order
:data:`BACKEND_PRECEDENCE` rules and :data:`BACKEND_PREFERENCE_SWITCH` may reorder,
and :func:`meep_gpu.fastpath_cuda.merge_tables` folds the second table's whole
fused units into the first's plan. What every table shares is everything below the
composer: the driver's slot order, the whole-or-nothing rule for a fused pair, the
subnormal gate, the certification lookup and the artifact — one implementation
each, in :func:`_finish` and the functions it calls.

THE ARRAY PATH IS THE UNIVERSAL FALLBACK AND THE ORACLE. ``stepping.py`` handles
100% of configurations by construction, is validated against CPU MEEP to the
floors in ``the test coverage notes``, and is bit-identical between NumPy and CuPy. The
kernels under ``triton_kernels/`` and ``metal_kernels/`` are an accelerated SUBSET,
gated on sha256 bit-identity against that path. So the contract this module
implements is not "cover everything". It is:

    dispatch only what is certified, fall back correctly for everything else,
    and make which of the two happened legible.

An uncovered slot, an uncovered grid feature, a missing Triton, a missing GPU —
every one of them reaches the array path with a NAMED reason recorded in the run
artifact, never an error and never a silent difference.

PLAN-TIME OR NEVER. Every refusal is decided here, at the driver's configuration
freeze. After the freeze there is NO runtime fallback: an exception out of a
dispatched ``plan.run()`` is FATAL and propagates. That is deliberate and it is
the opposite of defensive. A Triton launch that raises mid-write leaves its
target arrays PARTLY updated, and running the array call on top of that
double-applies the sub-step — a converged, smooth, wrong answer, which is
exactly what this design exists to prevent. "Fall back" is the wrong answer
there; "raise loudly" is the right one.

The two BEFORE-THE-LAUNCH guards in :meth:`FastPathPlan.dispatch` are not an
exception to that and are the reason it is worded around ``run()``: a stale
``Fields``, and a subnormal policy that is no longer the one the freeze gated on,
are both answered False with nothing written, which is the array path. The second
exists because the freeze's guarantee is a statement about the freeze and
``uninstall_subnormal_policy`` is public; measured, a plan kept dispatching
Triton kernels after it had been called, with CuPy's ``-ftz=true`` compiler seam
back in place — the shipped split, arrived at from inside a certified plan.

DISPATCH IS ON BY DEFAULT (:data:`DISPATCH_BY_DEFAULT` is ``True``), and
``MEEP_GPU_DISPATCH=0`` is how a run opts out; ``MEEP_GPU_DISPATCH=1`` asks for
dispatch explicitly, and any other value is refused by name
(:data:`DISPATCH_ENABLE_VALUES`) with the run on the array path. What a run
dispatched, and why each sub-step it did not was refused, is reported by
``driver.fast_path_report()``; the manual's capability-coverage guide gives what
each kernel table serves. The driver-route gate that byte-compares two drivers
across complete ``step()`` calls RAN on 2026-08-15: it passed under a uniform
subnormal policy (eight cases byte-identical over 16,440 complete steps) and
FAILED as shipped, because nothing in the package installed a policy and the two
executors ran on opposite ones. That split is what :func:`_subnormal_gate`
closes — dispatch installs the certified policy or refuses by name — and
:data:`DISPATCH_BY_DEFAULT` states the licence the default rests on and what that
licence does not cover.

WITHIN A DISPATCHING RUN, TWENTY-SEVEN CROSS-SUB-STEP TRITON PRODUCTS NOW DISPATCH, and
the hand-CUDA table brings its own
(:data:`meep_gpu.fastpath_cuda.CUDA_RELEASED_FUSED_ARMS`, released against that
table's own ledger and its own route gate, and reached only where the precedence
rung puts that table first).
:data:`RELEASED_FUSED_ARMS` names the Triton twenty-seven and the release is decided
in two halves — the CUDA rows are decided by the identical two halves in the sibling
module, through the SAME :func:`_axis_reasons`:
:data:`FUSED_RELEASE_ENVELOPE` is the shared half, which successive rounds have
emptied as the arms stopped agreeing on one axis after another, and
:data:`FUSED_RELEASE_ARM_AXES` is what each arm's OWN cases
drove, which is what lets the FOLDED pairs require a fold while the unfolded ones
refuse one, and what moved ``cylindrical``, then the six phase-B axes, then
``pml_active`` off the shared envelope — each the moment one arm was released on a
case the others were not. Every other fused label,
and every one of these twenty-seven outside its own axes, still refuses the whole
plan by name at rung (8). A fused pair takes BOTH slots of the sub-step pair it replaces and
performs the driver's in-seam fills in-launch; the driver then re-runs those
array passes on top, and that composition is what the gate measured
byte-identical rather than what this module argues about.

THE COMPOSER IS TOLD WHICH LABELS THIS LADDER WILL RUN, not just that fusion is
wanted (``plan_step``'s ``fuse_labels``). It has to be: a fused product occupies
BOTH slots of its seam, so a label the ladder cannot admit is one clause (8) can
only reject WHOLE, taking the separate certified arms underneath it with it.
Measured on 2026-09-02, before this round: a conductive 2-D PML grid that had
been dispatching ``fused pair B`` and four arms went entirely to the array path
once ``conductive_fused_electric_pair`` was wired into the composer, because its
label has no ledger entry and cannot be admitted. Handing the admitted set down
means an un-admitted product is never installed and its seam keeps the plans it
had, which is what the envelope paragraph in :func:`_decide` already promised at
the granularity of the whole run.

AND THE RELEASE CAME WITH A VETO, ``MEEP_GPU_FUSE_ARMS=0``
(:data:`FUSE_ARMS_VETO`), because it took a reachable configuration away. Dispatch
with the separate arms and no fusion used to be what the switch unset meant; after
the release that is the fused composition, so without the veto there is no way to
run the sub-step kernels without the pairs — no way to bisect a fusion defect, and
no way to count the launches fusion saves. That last one is not an argument: the
driver-route gate's substitution leg fused on both sides against the released tree
and reported a drop of 0.0 per step, and the gate refused to release until the
veto gave it back its baseline.

Package rule (backends.py): no module here imports ``cupy`` at module level, and
this one additionally decides the backend rung WITHOUT importing cupy, Triton or
torch at all — ``test_package_boundary``, ``test_fastpath`` and
``test_metal_dispatch`` each pin a half of it. Every device read below the rung is
function-local: :func:`metal_hardware_present`'s ``import torch``, the Metal
ladder's own module, and rung 4's ``import triton``.

NEITHER OTHER TABLE'S RELEASE ROWS ARE IN THIS FILE, and the cost argument is the
whole reason. :data:`RELEASED_FUSED_ARMS`, :data:`ARM_CERTIFICATION`,
:data:`FUSED_RELEASE_ENVELOPE` and :data:`FUSED_RELEASE_ARM_AXES` below are the
TRITON table's and stay that way; the Metal twins are in
:mod:`meep_gpu.metal_dispatch` and the hand-CUDA twins in
:mod:`meep_gpu.fastpath_cuda`, and the same helpers decide all three —
``_run_shape``, :func:`_axis_reasons` and :func:`fused_release_arm_reasons` are
imported there rather than re-implemented, so there is one definition of "does this
arm's release cover this shape" and three tables it is applied to. The two siblings
are priced differently on purpose: a Metal release row moving costs the Metal
artifacts that imported the module, and a CUDA release row or a merge-rule fix
moving costs the two route re-gates and NO Metal re-cut, because a NumPy step never
imports ``fastpath_cuda`` and so never puts it in a Metal manifest.

:func:`arm_is_fused` ANSWERS TWO OF THE THREE, and by two different mechanisms. Its
Triton half is a spelling (a prefix and a label set) and is unchanged; its CUDA half
is a NAMESPACE lookup, which is what the ``cuda:`` prefix on a merged label buys —
measured 2026-09-10, not one of the 38 labels the hand-CUDA composer writes matches
a Triton prefix, so covering both by spelling would have been guessing. The Metal
table does not come through here at all: it reads its fused set off its own
registry's weld flag, a declaration rather than a spelling.

THE KERNEL PACKAGES' OWN DOCSTRINGS SAID DISPATCH IS NOT WIRED, and the four that
said it loudest have been corrected: ``metal_kernels/launch.py``,
``metal_kernels/__init__.py``, ``cuda_kernels/arms.py`` and
``cuda_kernels/fused_pairs.py`` each stated that this module "returns ``None`` on
every branch", and the two CUDA ones additionally said that no dispatcher reaches
that package at all. Both halves were true when they were written and neither is
now. What correcting them cost is not the same on the two sides and the difference
is measured rather than assumed: the two Metal files are sha256-welded into gate
records, so editing a comment there turns a weld test red until the gate that
pinned it is re-run, and their corrections rode the re-gate the campaign named by
:data:`DRIVER_ROUTE_FUSED_GATE` was owed anyway (this file moved, and 32 Metal
products pin it). The two CUDA files are pinned by NO ledger entry — 0 of the 45,
counted — so their correction cost one coverage-census re-cut and no device time.

A RESIDUE REMAINS AND IS NAMED RATHER THAN QUIETLY LEFT: roughly two dozen further
Metal and Triton family modules carry the same sentence in their own docstrings.
Each is welded to a family gate this campaign did not re-run, so correcting one
would cost that family's device gate for a comment. They are wrong in the same
way and are listed by a grep for "on every branch"; the recorded shape under
``fingerprints.json['driver_dispatch']`` contradicts them where a reader of this
seam will find it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Set, Tuple)

import numpy

# Value-only, dependency-free: no CuPy, no Triton, no family module. Safe at
# module scope, which matters because the five Triton family modules must stay
# out of ``sys.modules`` after a bare package import.
from .expansion_refusal import declaring_run_policy, refusing_expansion_probe

# ---------------------------------------------------------------------------
# Environment: one veto, one enable, two diagnostics
# ---------------------------------------------------------------------------

#: The kill switch, for benchmarking and bisection: ``MEEP_GPU_FUSED=0``
#: forces the array path everywhere, regardless of everything below it. A VETO
#: ONLY — no value of it can enable dispatch, which is why the enable is a
#: separate variable: ``1`` or unset means "not vetoed", which leaves dispatch to
#: the enable, never "dispatch". Every other value is refused by name at rung 1
#: with the run on the array path (:data:`FUSED_KILL_SWITCH_VALUES`).
FUSED_KILL_SWITCH = "MEEP_GPU_FUSED"

#: The enable: ``1`` enables dispatch, ``0`` disables it, unset takes
#: :data:`DISPATCH_BY_DEFAULT` (``True``). Every other value — empty, whitespace,
#: ``true``, ``false``, ``off`` — is refused by name (:data:`DISPATCH_ENABLE_VALUES`).
DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"

#: ``0`` skips the plan-time warm pass (:func:`warm_plan`), so the
#: empty-grid compile mechanism can be measured against its absence on a device
#: without editing code. Every other value (including unset) runs it.
#:
#: IT CHANGES WHICH SLOTS DISPATCH, not just when they compile. A slot whose
#: kernel will not compile is unfilled by the warm pass and falls to the array
#: path; with the pass off nothing unfills, so the dispatch set is WIDER and the
#: compile failure lands mid-step instead. The artifact records that under
#: ``warm_pass.changes_the_dispatch_set_when_disabled`` so a run made with it off
#: cannot be read as the same run made with it on.
WARM_SWITCH = "MEEP_GPU_WARM"

#: ``0`` stops dispatch INSTALLING the subnormal policy, so a caller who wants to
#: own that decision can. It does NOT relax the rule: with the install off,
#: dispatch refuses unless the caller has already installed the certified policy
#: on all three executors. Every other value (including unset) installs.
#:
#: THE OPT-OUT CANNOT REACH A SPLIT, which is the whole reason it is spelled this
#: way. "Do not install" and "do not care" are different requests, and only the
#: first is offered: the failure this variable is next to is two executors under
#: two policies, and a switch that let a run proceed into it would be a switch for
#: turning the measurement off.
SUBNORMAL_INSTALL_SWITCH = "MEEP_GPU_SUBNORMAL_INSTALL"

#: THE FUSION OPT-IN: the route to a cross-sub-step fused product that no gate has
#: released, or to a released one on a configuration outside what its gate measured.
#: A comma-separated list of ARM LABELS (``fused pair B``, ``fused pair D``,
#: ``fused pair B (folded)``, ``dispersive fused pair``, ``fused ADE state``).
#:
#: IT IS NO LONGER THE ONLY ROUTE, and that is the 2026-08-29 change. Clause (8)
#: now admits the union of this switch and :data:`RELEASED_FUSED_ARMS` — the arms
#: the driver-route gate actually drove — restricted to the configurations
#: :data:`FUSED_RELEASE_ENVELOPE` says it drove them on. Unset, on any
#: configuration outside that envelope, the answer is exactly what it has always
#: been: the composer is asked with ``fuse=False, fuse_ade=False`` and clause (8)
#: refuses any fused label that reaches it anyway.
#:
#: WHY A SWITCH AND AN ALLOW-LIST CONSTANT, AND WHAT EACH IS FOR. Clause (8)'s
#: predicate is a statement about whether a GATE HAS RUN, and until 2026-08-29 no
#: gate could run: ``_decide`` passed ``fuse=False`` as a literal, so the driver
#: seam was unreachable from a fused arm and the refusal was unfalsifiable. The
#: chicken-and-egg was broken the way :data:`DISPATCH_ENABLE` broke the same one
#: for dispatch itself — an OPT-IN whose default was the old behaviour byte for
#: byte, so the gate could ask for the composition it had to measure without the
#: shipped answer moving. The gate then ran and released three arms, and RELEASING
#: is what the constant is: a measurement's verdict, written down once, reviewable
#: in a diff. The switch keeps its job — reaching an arm or a configuration the
#: constant does not cover, which is how the NEXT gate measures the next one.
#:
#: SETTING IT ON A CONFIGURATION OUTSIDE THE ENVELOPE IS NOT REFUSED, and the
#: record says so: the named reasons are carried by
#: ``fusion.outside_the_released_envelope`` when the SHARED half refuses (empty
#: since the target round emptied :data:`FUSED_RELEASE_ENVELOPE`, so in practice
#: only on an unreadable shape) and by ``fusion.outside_this_arms_own_cases`` when
#: the per-arm half does, which is now every ordinary refusal;
#: ``fusion.driven_outside_the_released_envelope`` marks the run and reads BOTH
#: halves, so an opt-in driving an arm past its own axes is stamped either way. That
#: is the same shape ``MEEP_GPU_DISPATCH=1`` has — an explicit request, recorded
#: as one — and it is what a gate needs to measure a shape nothing has measured.
#:
#: IT IS NOT A KILL SWITCH'S TWIN. Setting it cannot make anything dispatch that
#: the rest of the ladder refuses — the fold rung (6b), completeness (6), the
#: subnormal gate (8b) and the warm pass all still run, in that order, after it.
#: All it does is stop clause (8) refusing the labels it NAMES, and clause (8)
#: still refuses every fused label neither it nor the release names, by name.
FUSE_ARMS_SWITCH = "MEEP_GPU_FUSE_ARMS"

#: THE FUSION VETO: ``MEEP_GPU_FUSE_ARMS=0`` admits NO fused arm, however many
#: :data:`RELEASED_FUSED_ARMS` covers, and leaves the rest of dispatch running. It
#: is the third value of the same switch, the way ``0`` is the third value of
#: :data:`DISPATCH_ENABLE`, and it is a VETO — no value of it can admit an arm.
#:
#: WHY IT HAD TO EXIST THE MOMENT THE RELEASE DID, measured rather than foreseen.
#: Before 2026-08-29 there were three reachable configurations: the array path
#: (``MEEP_GPU_FUSED=0``), dispatch with separate arms (the switch unset), and
#: dispatch with fusion (the switch set). The release collapsed the middle one:
#: with the switch unset, a released arm on an in-envelope configuration now
#: fuses, so "dispatch, no fusion" became unreachable and the run that isolates a
#: fusion defect from a sub-step defect could not be asked for at all.
#:
#: THAT IS NOT A HYPOTHETICAL COST, it is the one that stopped a gate. The
#: driver-route gate's substitution proof — the only claim about fusion no byte
#: comparison can reach — counts device launches per step against a THIRD leg that
#: dispatches the SAME SLOTS with fusion off. Run against the released tree on
#: the GPU host GPU 6, that leg fused too: pml_2d measured 2.0 launches/step on both
#: sides, ``launch_drop_per_step 0.0``, verdict ``NO-DROP``, and the gate refused
#: to release. A release that cannot be re-measured on the tree that ships it is
#: not a release; this switch is what keeps the baseline reachable.
#:
#: THE VETO WINS WHEREVER IT APPEARS, including beside arm labels
#: (``0,fused pair B``). A veto a caller can override by naming an arm is not a
#: veto, and the ambiguity is better answered by the safe branch than by a new
#: error path. ``fusion.vetoed`` in the record says it was in force, and
#: ``fusion.released_here`` still reports what the release WOULD have admitted, so
#: the two facts stay separable in the artifact.
FUSE_ARMS_VETO = "0"

#: Where the per-freeze dispatch artifact is appended, one JSON object per line.
#: Unset keeps the last record in memory, served by ``driver.fast_path_report()``.
DISPATCH_LOG = "MEEP_GPU_DISPATCH_LOG"

#: THE OPT-IN FOR A DEVICE OR TOOLCHAIN NO RECORDED GATE RAN ON. ``1`` lets the
#: kernels dispatch on an identity that was READ and is not in the certified set: an
#: NVIDIA compute capability or a Triton version (rungs 4 and 4b), a torch version
#: or a Metal frontend (rung 4M). ``0`` and unset keep the refusal to the array
#: path, and every other value is refused by name (:data:`UNCERTIFIED_VALUES`).
#:
#: IT ADMITS AN IDENTITY AND NOTHING ELSE. Every rung below the identity rungs
#: still runs, in order, and an identity that could not be READ is untouched by it:
#: that one was never refused. A run it admits carries NO CERTIFICATION: the record
#: says ``certified: False`` with the identity read under ``uncertified.served``,
#: and one line per process names what is certified.
UNCERTIFIED_SWITCH = "MEEP_GPU_ALLOW_UNCERTIFIED"

# ---------------------------------------------------------------------------
# Which KERNEL TABLE: the hardware picks the candidate set, a preference selects
# within it
# ---------------------------------------------------------------------------

#: THE EXPLICIT PREFERENCE, and it SELECTS WITHIN the candidate set rather than
#: widening it. A run that asked for a table this hardware cannot run is refused BY
#: NAME at the backend rung, never answered with a different table: an artifact
#: describing a table the caller did not ask for is worse than a refusal, because
#: the caller has no way to see that the request was dropped.
KERNEL_TABLE_SWITCH = "MEEP_GPU_KERNEL_TABLE"

#: THE ORDER A CuPy ENGINE'S CANDIDATES ARE TAKEN IN, and the constant is the
#: ruling rather than a preference: Triton is the release-gated incumbent, and the
#: hand-CUDA table fills its refusals. Measured-fastest-per-arm replaces the order
#: once an idle-box timing round exists; until then the incumbent goes first, and
#: flipping it is a one-constant edit priced as a fastpath edit (this file is bound
#: by BOTH ``driver_dispatch`` records, so an edit here re-cuts them and re-drifts
#: every Metal artifact that imported this module).
NVIDIA_TABLE_PRECEDENCE: Tuple[str, ...] = ("triton", "cuda")

#: THE SAME TUPLE UNDER THE NAME THE LADDER'S PRECEDENCE RUNG READS. Rung 3 asks the
#: HARDWARE which tables exist; rung 4d orders the ones that survived their own
#: candidacy checks, and it is that rung the campaign's arbitration legs drive. One
#: definition, two readers, so a flip cannot move one and not the other.
BACKEND_PRECEDENCE: Tuple[str, ...] = NVIDIA_TABLE_PRECEDENCE

#: THE PREFERENCE THAT ORDERS THE NVIDIA TABLES AT RUNG 4d. Distinct from
#: :data:`KERNEL_TABLE_SWITCH`, which answers a different question one rung earlier:
#: that one picks WHICH HARDWARE'S table (a NumPy host with an MPS device against a
#: CuPy host), this one picks which of the two NVIDIA tables composes FIRST and
#: therefore holds every slot both admit. A value naming a table that is not a
#: candidate here is refused BY NAME — never answered with the other table.
BACKEND_PREFERENCE_SWITCH = "MEEP_GPU_BACKEND_PREFERENCE"

#: The table a NumPy engine with an MPS device takes. Its ladder and its release
#: rows live in :mod:`meep_gpu.metal_dispatch`, NOT here — see that module's
#: docstring for why a Metal release row in this file would re-drift the Triton
#: record every time it moved.
METAL_TABLE = "metal"

#: The hand-CUDA table's name, and the prefix its labels carry once they cross into
#: this file. Its rows, its release and its merge rule live in
#: :mod:`meep_gpu.fastpath_cuda` for the reason the Metal rows live in
#: ``metal_dispatch``: that module is in no Metal manifest, so a merge-rule or
#: release-row fix costs two route re-gates and no Metal re-cut.
CUDA_TABLE = "cuda"
CUDA_LABEL_PREFIX = "cuda:"


def _bare(label: Any) -> str:
    """A label with its table namespace taken off. Bare labels pass through."""
    text = str(label)
    return (text[len(CUDA_LABEL_PREFIX):]
            if text.startswith(CUDA_LABEL_PREFIX) else text)


def _table_of(label: Any) -> Optional[str]:
    """Which table wrote this label, or ``None`` when the label carries no namespace.

    ``None`` RATHER THAN ``"triton"``, and the distinction is load-bearing. A bare
    label means "whatever table the caller was composing", which on a CuPy host is
    Triton and on a NumPy host is Metal — two different ledgers. Answering
    ``"triton"`` here would quote the Triton ledger beside a Metal gate, which is the
    record-forgery shape :func:`_certification_for` exists to make visible.
    """
    return (CUDA_TABLE if str(label).startswith(CUDA_LABEL_PREFIX) else None)


def backend_precedence(candidates: Sequence[str]) -> Any:
    """The order the candidate NVIDIA tables compose in, or a NAMED refusal string.

    Reads :data:`BACKEND_PREFERENCE_SWITCH` FRESH, like every other switch in this
    file: a process that sets it between two freezes gets what it set for the second.

    * unset -> :data:`BACKEND_PRECEDENCE` filtered to the candidates;
    * a value naming a candidate -> that table first, the rest in constant order;
    * anything else -> a REFUSAL STRING naming the switch, the token and the
      candidate set.

    THE REFUSAL IS A RETURN VALUE, NOT A RAISE. ``plan_fast_path``'s wrapper turns an
    exception into "internal error", which is exactly the wording that makes a
    deliberate refusal look like a defect; the ladder's contract is that every rung
    answers with a named reason and none of them raises.
    """
    wanted = (os.environ.get(BACKEND_PREFERENCE_SWITCH) or "").strip()
    order = tuple(name for name in BACKEND_PRECEDENCE if name in candidates)
    if not wanted:
        return order
    if wanted not in candidates:
        return (f"{BACKEND_PREFERENCE_SWITCH}={wanted} names no candidate table on "
                f"this host; candidates are "
                f"{', '.join(candidates) or 'none'}. The preference orders WHAT THE "
                "HARDWARE AND THE CERTIFICATIONS ADMIT and never widens it")
    return (wanted,) + tuple(name for name in order if name != wanted)


def metal_hardware_present() -> bool:
    """Is there an MPS device this process could compile a Metal shader for?

    THE SAME THREE READS ``metal_kernels/coverage._metal_backend_reasons`` MAKES,
    so the backend rung and the coverage clause cannot disagree about whether the
    hardware is there: torch built with MPS, an MPS device available, and
    ``torch.mps.compile_shader`` present (the entry point every hand-written Metal
    source is compiled through). Never raises — a host with no torch answers False,
    which is the array path.

    The ``import torch`` is function-local on purpose. This module decides the
    NumPy case without importing a device library, and ``test_package_boundary``
    measures exactly that kind of needless pull-in; a module-level import here
    would drag torch into every process that touches the fast path.
    """
    try:
        import torch  # noqa: PLC0415

        return bool(torch.backends.mps.is_built()
                    and torch.backends.mps.is_available()
                    and hasattr(torch.mps, "compile_shader"))
    except Exception:  # noqa: BLE001 - no torch, or a build that cannot answer
        return False


def candidate_tables(grid: Any) -> Tuple[str, ...]:
    """Which kernel tables could POSSIBLY run on this host, in precedence order.

    THE RUNG IS NOT "WHICH TABLE DO WE LIKE". It is answered from two facts the
    caller cannot argue with — the engine's array module and the device that is
    actually present — and only then does :data:`KERNEL_TABLE_SWITCH` choose among
    what came back. Splitting it that way is what makes a preference naming an
    absent table a NAMED REFUSAL instead of a silent fallback.

    * a CuPy engine -> :data:`NVIDIA_TABLE_PRECEDENCE`;
    * a NumPy engine with an MPS device -> ``(METAL_TABLE,)``;
    * anything else -> ``()``, and the backend rung refuses with the compound
      reason that quotes both halves.

    WHOSE NUMPY ENGINE REACHES THIS RUNG: from :class:`~meep_gpu.driver.FdtdDriver`,
    only a driver built with ``prefer_gpu=True`` on an Apple GPU. A driver built with
    ``prefer_gpu=False`` is the NumPy reference and never calls the planner (its
    freeze publishes :func:`reference_record`), so the answer here is about the
    engine and the hardware, and the caller's intent was settled before it.

    An engine that merely NAMES itself something else — a stub backend, a future
    array module — is not a candidate for either table, which is the fail-closed
    direction: the array path is always correct.
    """
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") == "cupy":
        return NVIDIA_TABLE_PRECEDENCE
    if xp is numpy or getattr(xp, "__name__", "") == "numpy":
        return (METAL_TABLE,) if metal_hardware_present() else ()
    return ()


#: THE NAMES THIS PACKAGE USED TO READ, and what replaced each. Renamed 2026-08-22: the
#: package ships standalone as a MEEP accelerator, so its public switch surface must not
#: carry the name of the tree it happens to live in. Seven of its switches were already
#: ``MEEP_GPU_*``; these five were the inconsistent minority. ``*_FDTD_TRITON``
#: carried a second leak besides the project name -- it is the general dispatch enable,
#: and only one of the three backends is named Triton.
#:
#: THIS IS A REFUSAL, NOT AN ALIAS. The rename-outright rule forbids a back-compatibility
#: spelling, and nothing here accepts one. But the failure a silent rename produces is
#: the worst one available: a run with the old name set loses dispatch and still
#: completes, so the measurement changes and nothing says so. :func:`_refuse_legacy_names`
#: raises instead, naming the replacement.
LEGACY_SWITCH_NAMES = {
    "TRIDENT_FDTD_FUSED": FUSED_KILL_SWITCH,
    "TRIDENT_FDTD_TRITON": DISPATCH_ENABLE,
    "TRIDENT_FDTD_WARM": WARM_SWITCH,
    "TRIDENT_FDTD_SUBNORMAL_INSTALL": SUBNORMAL_INSTALL_SWITCH,
    "TRIDENT_FDTD_DISPATCH_LOG": DISPATCH_LOG,
}


def _refuse_legacy_names() -> None:
    """Raise if a retired switch name is set, rather than ignoring it silently."""
    stale = sorted(name for name in LEGACY_SWITCH_NAMES if os.environ.get(name) is not None)
    if not stale:
        return
    lines = ", ".join(f"{name} -> {LEGACY_SWITCH_NAMES[name]}" for name in stale)
    raise RuntimeError(
        f"retired environment switch set: {lines}. This package renamed its switches to "
        f"the MEEP_GPU_ prefix on 2026-08-22 so it can ship standalone; the old spelling "
        f"is refused rather than ignored, because ignoring it would silently change what "
        f"a run dispatches.")

#: The complex families' expansion licence. Named here only so the artifact can
#: report whether it was set — four of the nine certified families refuse without
#: it, and that refusal must be visible rather than show up as families silently
#: missing from a coverage count.
COMPLEX_PROBE_ENV = "MEEP_GPU_COMPLEX_EXPANSION_PROBE"

#: The slots ``FdtdDriver`` can actually RUN a plan from — the seven sub-step calls
#: :meth:`FastPathPlan.dispatch` sits in front of, reached from nine call sites
#: (``step`` runs all seven; ``synchronize_magnetic_fields`` runs ``step_B`` and
#: ``fill_B`` again). FOUR FURTHER sites consult the same method under a PASS name
#: rather than a slot name: three under the folded-far pass names
#: (:data:`FAR_FILL_OWNERS`, the fill sub-step's second half) and one under
#: :data:`SYNC_UPDATE_H_PASS`, which is the magnetic half-step's own name for
#: ``update_H`` and the site a plan may DECLINE (:data:`SYNC_PASS_OWNERS`).
#:
#: ``fill_B``/``fill_D`` JOINED THIS TUPLE ON 2026-09-02, and what let them is the
#: three things the previous text priced: the two driver consults, retiring the
#: blanket fold rung with them, and a gate that runs the composition through the
#: real seam. ALL THREE ARE NOW DONE — the third is
#: :data:`DRIVER_ROUTE_FUSED_GATE`'s ``folded_2d`` and ``folded_dispersive_2d``
#: cases, which drive both fill slots in-launch — and the two folded pairs are in
#: :data:`RELEASED_FUSED_ARMS` as a result. Every OTHER folded fused label still
#: refuses by name at clause (8) unless :data:`FUSE_ARMS_SWITCH` opts into it.
#:
#: THE FILL SUB-STEP IS TWO DRIVER PASSES, not one, and the seam has to say so.
#: ``launch.STEP_ORDER``'s ``fill_B``/``fill_D`` stand for the combined
#: near-symmetry and folded-far passes, and the driver runs ``zero_metal_*``
#: BETWEEN them (``driver.step``). The two fill arms disagree about whether that
#: matters, so the adapter reads the PLAN rather than the arm label:
#: ``symmetry.MirrorGhostFillPlan`` fuses near and far into one launch per axis and
#: is correct to (±1 * float32 is exact), while
#: ``folded_complex.FoldedMirrorGhostFillComplexPlan`` splits them into
#: ``run_near``/``run_far`` because complex multiplication is not associative and
#: its own docstring forbids fusing across the wall clear. :meth:`dispatch` runs
#: the near half (or the fused whole) and the far-pass consult runs whatever
#: is left, so neither plan is asked to do the thing it says it must not.
#:
#: WHAT THE CONSULTS DO NOT DO IS SKIP A FUSED PRODUCT'S FILLS. A folded fused pair
#: DECLARES ``fill_*``/``zero_metal_*``/``fill_folded_far_ghosts_*`` in its
#: ``REPLACES``, and it performs them IN-LAUNCH — against the PRE-injection field,
#: because ``_install_fused_pair`` writes only the curl and constitutive slots and
#: the driver injects between those two consults. The driver's post-injection
#: passes are therefore load-bearing rather than redundant: ``deposit_repair``'s
#: image closure recomputes the constitutive at the cells those passes image a
#: deposit into, and it reads the field AFTER they have run. So a fill slot is
#: served from the fill slot's own plan and never from a fused product's
#: declaration — see :meth:`FastPathPlan._dispatch_far_fill` and
#: ``test_a_fused_product_that_declares_the_fills_still_gets_the_drivers_passes``.
#:
#: THE VOCABULARY LIVES HERE, NOT IN ``triton_kernels``, and the direction is the
#: point: dispatch depends on the kernel package and the kernel package must not
#: know a dispatcher exists (``test_triton_kernels`` enforces both halves). The
#: package's own files are additionally sha256-welded to the device gates that
#: certified them, so a constant added there is a device-gate event; a constant
#: added here is not, and it earns its keep only if something MEASURES that it
#: still matches what the composer writes — which is what
#: ``test_dispatch_contract`` does, by reading a real composition rather than by
#: sharing a symbol.
DRIVER_SLOTS: Tuple[str, ...] = (
    "step_B", "fill_B", "update_H", "step_D", "fill_D", "update_E", "update_P",
)

#: The array pass each fill slot's SECOND half stands in for. Named here so the
#: driver's two far-ghost consults and this module's vocabulary are one fact, and so
#: a reader can see that ``zero_metal_*`` is NOT in it: the wall clear runs
#: unconditionally, behind no consult, exactly as ``deposit_repair`` states.
FAR_FILL_PASSES: Mapping[str, str] = {
    "fill_B": "fill_folded_far_ghosts_B",
    "fill_D": "fill_folded_far_ghosts_D",
}

#: The inverse, which is what :meth:`FastPathPlan.dispatch` is CONSULTED with.
#:
#: THE FAR PASS IS ASKED FOR BY ITS OWN DRIVER NAME rather than through a second
#: verb, and that choice is about everything else that implements this seam. The
#: driver's consult protocol is one method — the parity probes
#: (``probe_triton_folded_deposit_closure``, ``probe_metal_complex_driver_step``,
#: ``probe_metal_special_kz_driver_step``) install their own adapters at
#: ``driver._fast_path``, and a FROZEN copy of one ships inside a gate's result
#: directory where nothing may edit it. A second required method would have raised
#: ``AttributeError`` in each of them; a slot NAME they do not carry answers False
#: through the adapter they already have, and False is the array path, which is
#: correct everywhere. The names are the driver's own pass names, the same ones a
#: fused product's ``REPLACES`` declares, so nothing new is invented to spell them.
FAR_FILL_OWNERS: Mapping[str, str] = {
    pass_name: slot for slot, pass_name in FAR_FILL_PASSES.items()
}

#: The sub-steps ``driver.synchronize_magnetic_fields`` runs — the magnetic half of
#: ``step``, repeated for a flux or a field energy and then UNDONE. It is a subset of
#: :data:`DRIVER_SLOTS` and it is the boundary the channel below is about: the
#: half-step backs up ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY`` (magnetic names only)
#: and ``restore_magnetic_fields`` puts exactly those back, so ANY electric state a
#: plan advances inside this window is advanced permanently and silently.
#: ``D``, ``fu_D`` and ``f_cond_D`` are in neither list — checked, not assumed:
#: ``test_sync_pass_channel`` reads both tuples off the driver and asserts it.
SYNC_PATH_SLOTS: Tuple[str, ...] = ("step_B", "fill_B", "update_H")

#: THE SECOND BY-NAME CHANNEL, and it exists for the opposite reason to the first.
#:
#: :data:`FAR_FILL_OWNERS` gives a pass its own name so a plan can OWN a site the
#: slot vocabulary could not spell. This one gives a site its own name so a plan can
#: DECLINE it. ``synchronize_magnetic_fields`` consults ``update_H`` — the same slot
#: ``step`` consults — but inside a window that can only put magnetic arrays back. A
#: product spanning ``update_H`` and ``step_D`` is correct in ``step`` and a silent
#: wrong answer here: it advances ``D``/``fu_D`` inside the half-step, the restore
#: cannot reach them, and every ``flux_in_box`` and ``field_energy_in_box`` call
#: leaves the run's electric state one half-step ahead of itself.
#:
#: So the sync site asks for ``update_H_synchronize`` and :meth:`FastPathPlan.dispatch`
#: answers it only where the plan installed at ``update_H`` stays inside
#: :data:`SYNC_PATH_SLOTS`. A pair that escapes them is refused and the driver runs
#: the array ``update_H``, which is correct everywhere.
#:
#: A NAME AND NOT A SECOND METHOD, for exactly the reason :data:`FAR_FILL_OWNERS`
#: gives: the ``DispatchAdapter`` shims in ``probe_triton_folded_deposit_closure``,
#: ``probe_metal_complex_driver_step``, ``probe_metal_special_kz_driver_step`` and a
#: FROZEN copy inside a gate's result directory implement this seam as one method
#: over the names they carry. A name they do not carry answers False, which is the
#: array path; a second required method would raise ``AttributeError`` in each of
#: them, and one of those copies is inside a record nothing may edit.
#:
#: WHAT THIS DOES NOT COVER, stated rather than left to be found: the sync site's
#: other three consults (``step_B``, ``fill_B`` and the folded-far pass) still carry
#: their plain names, and a product LEADING at one of them and spanning past
#: ``update_H`` would reach the same hazard by the same route. No such product
#: exists — every shipped seam is inside one field type, and
#: ``test_sync_pass_channel`` measures that premise over the corpus rather than
#: asserting it. The four-slot ``step_B..update_E`` weld is the shape that breaks it,
#: and it owes a name here for each site it leads, in the same change.
#: The name the driver's sync site spells, held here so the site and the channel are
#: one fact rather than two string literals that agree today.
SYNC_UPDATE_H_PASS: str = "update_H_synchronize"

SYNC_PASS_OWNERS: Mapping[str, str] = {SYNC_UPDATE_H_PASS: "update_H"}

#: The arm whose product launches nothing (``plan_null_constitutive``). Pinned
#: against a real composition rather than shared with the composer.
NULL_ARM_LABEL: str = "no-PML null"

#: The arm labels a FUSED product writes into ``TritonStepPlan.selected``. Every
#: one of them refuses the WHOLE plan, so a future flip of ``fuse``/``fuse_ade``
#: to default-on cannot silently reach a driver seam no gate drove. ``fused pair``
#: is a PREFIX (``fused pair B``, ``fused pair D``); the other two are whole labels.
FUSED_ARM_PREFIXES: Tuple[str, ...] = ("fused pair",)
FUSED_ARM_LABELS: Tuple[str, ...] = ("dispersive fused pair", "fused ADE state")


def arm_is_fused(label: Any) -> bool:
    """Whether ``label`` names a cross-sub-step fused product on EITHER NVIDIA table.

    THE TRITON HALF IS ONE TABLE'S VOCABULARY AND IT IS UNCHANGED. Measured
    2026-09-10: not one of the 47 labels the Metal registry writes matches either
    the prefix or the whole labels above (``fused magnetic B/H pair``, ``folded
    fused B/H pair``, ...), and not one of the 38 the hand-CUDA composer writes does
    either. That is the honest answer rather than a gap: the three composers name
    their products differently and always did, so a predicate that tried to cover
    all of them by SPELLING would be guessing at two.

    THE CUDA HALF IS ANSWERED BY THE NAMESPACE, WHICH IS WHY THE NAMESPACE EXISTS.
    A ``"cuda:"`` label is looked up in the typed
    ``fastpath_cuda.CUDA_FUSED_LABELS``, which is pinned to ``fused_pairs.py``'s AST
    by test; that lookup needs no ``cuda_kernels`` import, so this function stays
    reachable from a NumPy step that must not pay for the CUDA package.

    THE METAL LADDER DOES NOT COME THROUGH HERE AT ALL: it reads its fused set off
    its own registry's ``is_weld`` flag (``metal_dispatch.fused_labels``), which is a
    DECLARATION rather than a spelling and cannot drift with a rename.
    """
    text = str(label)
    if _table_of(text) == CUDA_TABLE:
        from . import fastpath_cuda  # noqa: PLC0415

        return fastpath_cuda.is_fused(text)
    return (text in FUSED_ARM_LABELS
            or any(text.startswith(prefix) for prefix in FUSED_ARM_PREFIXES))


#: What an unset :data:`DISPATCH_ENABLE` means. TRUE since 2026-09-27: a driver built
#: with ``prefer_gpu=True`` that asks for nothing gets every certified kernel this
#: host can run, and ``MEEP_GPU_DISPATCH=0`` is how it asks for the array path
#: instead. A driver built with ``prefer_gpu=False`` is the NumPy reference: it never
#: reaches this ladder, so neither this constant nor any value of the enable turns
#: its kernels on (:func:`reference_record`).
#:
#: THE LICENCE, NOT YET RECORDED, is the measurement this constant's previous text
#: named as missing: ``gate_dispatch_end_to_end`` run on the shipping bytes with the
#: harness installing NO policy and the enable left UNSET, released across its nine
#: cases — every case PASS-DISPATCHED or PASS-FELL-BACK, no checkpoint divergent. That
#: run is to be transcribed into ``triton_kernels/fingerprints.json['driver_dispatch']
#: ['runs'][<compute capability>]['dispatch_by_default_licence']`` -- one licence per
#: architecture, on that architecture's primary table (:func:`primary_table`) --
#: with the digest of this file it executed; that block is not in this release, and
#: ``test_dispatch_contract`` binds this constant to it, so ``True`` here fails there
#: (a certification test, expected to fail until the run is made). That is the same
#: one-change weld the 2026-08-15 tripwire made for ``False``.
#:
#: WHAT CAME BEFORE, kept because it explains the shape of the licence. The
#: 2026-08-15 ship leg diverged on six of nine cases because the CuPy and Triton
#: executors ran on opposite subnormal policies; the same day's re-run under the
#: shipped policy gate closed that (0 divergent, 7 PASS-DISPATCHED, 2
#: PASS-FELL-BACK, 16,440 complete steps) on a tree that predates the fused release.
#: The driver-route gate (:data:`DRIVER_ROUTE_FUSED_GATE`) then drove the seven real
#: consults with the harness installing no policy. What the default waited on after
#: that was the end-to-end ship leg on bytes where that leg FUSES, because this enable
#: governs every configuration and not only the fused route.
#:
#: WHAT THE LICENCE DOES NOT COVER, stated so a default run is not read as more than
#: it is:
#:
#: * ONE NVIDIA COMPOSITION. The end-to-end gate runs the NVIDIA tables with Triton
#:   first. A host with no validated Triton composes the hand-CUDA table alone; the
#:   CUDA route campaign's ``cuda_alone`` leg drives that composition, NON-BLOCKING,
#:   and the CUDA record's ``cuda_alone_leg`` says whether it ran and released. This
#:   gate does not drive it. On the Metal table the analogous evidence is the Metal
#:   route campaign's legs, run under held residency (that table's default), not this
#:   gate.
#: * THE CERTIFIED SURFACE ONLY. Triton or hand-CUDA on the compute capability their
#:   ledgers name, and Metal on the torch and frontend pair its ledger names. Every
#:   other device and toolchain is refused by name at rung 4 and takes the array
#:   path, unless the run sets :data:`UNCERTIFIED_SWITCH` to ``1``, which dispatches
#:   on it with ``certified: False`` in the record. A device identity that cannot be
#:   READ is not a refusal (rung 4b), so such a host dispatches on an architecture
#:   nothing verified.
#: * COMPLEX STORAGE WITHOUT THE EXPANSION LICENCE, which is a statement about
#:   SLOTS and not about runs. On the NVIDIA tables the complex families refuse by
#:   name unless :data:`COMPLEX_PROBE_ENV` supplies an expansion licence, and the
#:   package ships none — but that refuses those families' slots, not the run. The
#:   ladder decides each slot on its own arms, so a default complex run takes the
#:   array path throughout only where no arm outside the licensed families admits a
#:   slot; where one does, the run is a MIXED composition, some slots on kernels and
#:   the licence-gated ones on the array path, which the record discloses as such.
#:   The route campaigns drive their licence-dependent cases on the expansion-probe
#:   legs only, so what a default complex NVIDIA run composes has not been measured
#:   as a leg of its own. On the Metal table the held-residency admission rung
#:   (``metal_dispatch``, 8aM) refuses BY NAME, to the array path, any composition
#:   that leaves a sub-step outside the door-wired seam passes on the array path, so a
#:   default complex Metal run is held throughout or on the array path, never mixed.
#: * SPEED AGAINST EVERY ALTERNATIVE. On the measured NVIDIA 2-D rows fused dispatch
#:   beats the array path but loses to the certified singles below about 2.4M cells;
#:   the measured-cost preference veto that would arbitrate between dispatched
#:   options is not wired into this ladder. On Metal, held residency (the default
#:   since 2026-09-27) crosses the host array path at 28,246-48,027 cells (Phase 0,
#:   harness-only) and is SLOWER below it -- 1.51x-2.34x at the route campaign's
#:   step-weighted 14,123 cells -- so a small 2-D run on an Apple host is faster
#:   built with ``prefer_gpu=False`` (the host reference), or with
#:   ``MEEP_GPU_DISPATCH=0``. ``MEEP_GPU_METAL_RESIDENCY=shipped`` copies every mirror
#:   both ways on every launch and is slower than the array path at every measured
#:   size.
#:
#: WHAT A DEFAULT RUN NOW PAYS that an opted-out run does not: plan-time compilation,
#: the table's certified subnormal policy installed process-wide at rung 8b (the
#: array path included), and PLAN-TIME OR NEVER — an exception out of a dispatched
#: ``plan.run()`` after the freeze is fatal, as the module docstring states.
DISPATCH_BY_DEFAULT = True

#: The enable's accepted values and what each means. EXACT MATCH, with no stripping
#: and no synonyms: every other value — empty, whitespace-only, ``true``, ``false``,
#: ``off``, a value with a stray space — is refused BY NAME at rung 2 and the run
#: takes the array path. Before 2026-09-27 every value but ``0`` enabled, so
#: ``MEEP_GPU_DISPATCH=false`` dispatched; with dispatch on by default, the value a
#: user types to turn it off is exactly the one a permissive reader gets wrong, and
#: the failure is silent — the run completes and measures something else.
DISPATCH_ENABLE_VALUES: Mapping[str, bool] = {"1": True, "0": False}


class DispatchEnableValueError(RuntimeError):
    """An unrecognised :data:`DISPATCH_ENABLE` value, refused by name.

    A ``RuntimeError`` so rung 2 catches it where it already catches the retired-name
    rule (:func:`_refuse_legacy_names`), and a subclass of its own so the record can
    say which of the two rules answered.
    """


#: The kill switch's accepted values: ``0`` vetoes dispatch, ``1`` does not (and
#: neither does unset). EXACT MATCH, as :data:`DISPATCH_ENABLE_VALUES` is and for the
#: same reason, word for word: with dispatch on by default this is the second "off"
#: switch a user reaches for, and before 2026-09-27 every value that did not strip to
#: ``0`` — ``false``, ``off``, an empty string — silently left dispatch on. Every other
#: value is refused BY NAME at rung 1 and the run takes the array path, announced; the
#: direction is fail-closed because a user who typed anything here meant "off".
FUSED_KILL_SWITCH_VALUES: Mapping[str, bool] = {"1": True, "0": False}


def fused_dispatch_enabled() -> bool:
    """The kill switch, read fresh on every plan build: ``False`` is the array path.

    NON-RAISING, because ``_base_record`` reports it on every record before rung 1
    runs. It reads ``False`` for ``0`` AND for an unrecognised value — the run takes
    the array path either way — and :func:`_kill_switch_value_refusal` is what tells
    the two apart, for rung 1's reason and for the stderr line.
    """
    raw = os.environ.get(FUSED_KILL_SWITCH)
    if raw is None:
        return True
    return FUSED_KILL_SWITCH_VALUES.get(raw, False)


def _kill_switch_value_refusal(raw: Optional[str]) -> Optional[str]:
    """The named refusal for an unrecognised kill-switch value; ``None`` when unset or accepted.

    Non-raising and handed the value, like :func:`_enable_value_refusal`, so rung 1
    and the stderr announcement ask the same question of the value a record carries.
    """
    if raw is None or raw in FUSED_KILL_SWITCH_VALUES:
        return None
    return (f"{FUSED_KILL_SWITCH}={raw!r} is not an accepted value: set "
            f"{FUSED_KILL_SWITCH}=0 to force the array path, or "
            f"{FUSED_KILL_SWITCH}=1 or leave it unset to leave dispatch to "
            f"{DISPATCH_ENABLE}. The value is refused rather than interpreted, so "
            "this run takes the array path")


def _enable_value_refusal(raw: Optional[str]) -> Optional[str]:
    """The named refusal for an unrecognised enable value; ``None`` when unset or accepted.

    Non-raising and handed the value rather than reading it, so the record reader and
    the stderr announcement can ask the same question of the value a record carries.
    """
    if raw is None or raw in DISPATCH_ENABLE_VALUES:
        return None
    default = "dispatch" if DISPATCH_BY_DEFAULT else "the array path"
    return (f"{DISPATCH_ENABLE}={raw!r} is not an accepted value: set "
            f"{DISPATCH_ENABLE}=1 for dispatch, {DISPATCH_ENABLE}=0 for the array "
            f"path, or leave it unset for the default ({default}). The value is "
            "refused rather than interpreted, so this run takes the array path")


#: The uncertified opt-in's accepted values. EXACT MATCH, as
#: :data:`DISPATCH_ENABLE_VALUES` is: a value that is neither is refused BY NAME at
#: rung 3b and the run takes the array path, on a certified host as well, because a
#: value that was typed and not understood must not be read as either answer.
UNCERTIFIED_VALUES: Mapping[str, bool] = {"1": True, "0": False}

#: What an identity refusal says about the way past it. APPENDED to the refusal,
#: never put in front of it: the readers of these messages match their heads.
UNCERTIFIED_HINT = (f"; set {UNCERTIFIED_SWITCH}=1 to dispatch the kernels on it "
                    "without certification")


def _uncertified_value_refusal(raw: Optional[str]) -> Optional[str]:
    """The named refusal for an unrecognised opt-in value; ``None`` when unset or accepted."""
    if raw is None or raw in UNCERTIFIED_VALUES:
        return None
    return (f"{UNCERTIFIED_SWITCH}={raw!r} is not an accepted value: set "
            f"{UNCERTIFIED_SWITCH}=1 to dispatch the kernels on a device or "
            f"toolchain that is not certified, or {UNCERTIFIED_SWITCH}=0 or leave "
            "it unset to refuse one to the array path. The value is refused rather "
            "than interpreted, so this run takes the array path")


def uncertified_allowed() -> bool:
    """The uncertified opt-in, read fresh on every plan build. Never raises.

    ``True`` for ``1`` only. An unrecognised value reads ``False`` here, which is
    what the run gets: rung 3b refuses it by name.
    """
    raw = os.environ.get(UNCERTIFIED_SWITCH)
    if raw is None:
        return False
    return UNCERTIFIED_VALUES.get(raw, False)


def _admit_uncertified(record: Dict[str, Any], table: str, what: str, read: Any,
                       certified: Sequence[Any]) -> None:
    """Record one identity the opt-in admitted: which table, what was read, what is certified."""
    record.setdefault("uncertified", {}).setdefault("admitted", []).append(
        {"table": table, "what": what, "read": read, "certified": list(certified)})


def _uncertified_note(record: Mapping[str, Any]) -> str:
    """The one line an opted-in process prints: what was read and what is certified.

    Built from every identity the opt-in ADMITTED and not from the tables that
    served this freeze, and an identity two tables share is named once: the line is
    a statement about the host, so every freeze of the process repeats it and
    :func:`_announce_line` prints it once.
    """
    named: List[str] = []
    for entry in (record.get("uncertified") or {}).get("admitted") or ():
        certified = ", ".join(str(value) for value in entry["certified"]) or "none"
        text = f"{entry['what']} {entry['read']} (certified: {certified})"
        if text not in named:
            named.append(text)
    return ("meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: "
            + "; ".join(named)
            + f". They were dispatched because {UNCERTIFIED_SWITCH}=1; compare the "
            "results with a prefer_gpu=False run of the same simulation before "
            "relying on them")


def _certified_verdict(record: Mapping[str, Any],
                       tables: Sequence[str]) -> Optional[bool]:
    """Are the kernels certified on the identity that served? THREE-VALUED.

    ``False`` when the opt-in admitted the identity of a table that SERVED, ``True``
    when every identity read of every table that served is a certified one, and
    ``None`` when one of them could not be read, which is not a refusal and not a
    certification either.
    """
    block = record.get("uncertified") or {}
    served = [entry for entry in block.get("admitted") or ()
              if entry.get("table") in tables]
    if isinstance(block, dict):
        block["served"] = served
    if served:
        return False
    environment = record.get("environment") or {}
    by_table = environment.get("device_certified_by_table") or {}
    reads: List[Any] = []
    for table in tables:
        if table == METAL_TABLE:
            reads += [environment.get("torch_certified"),
                      environment.get("frontend_certified")]
            continue
        reads.append(by_table.get(table))
        if table == "triton":
            reads.append(environment.get("triton_certified"))
    if reads and all(value is True for value in reads):
        return True
    return None


def _enable_as_set() -> bool:
    """What the enable says, and NOTHING else — no retired-name check, so it cannot raise.

    Split out of :func:`dispatch_enabled` because ``_base_record`` reports the
    enable's effective value on EVERY record, and ``plan_fast_path`` builds that
    record before entering the wrapper that makes the ladder unable to raise. Asking
    the raising reader there put :func:`_refuse_legacy_names` ahead of rung 1 and
    outside the wrapper at once, so a retired name set alongside ``MEEP_GPU_FUSED=0``
    came out of ``plan_fast_path`` as a ``RuntimeError`` — against the kill switch's
    "highest precedence, read before anything else" AND against "never raises".
    Reporting a value is not the place to enforce a rule.

    AN UNRECOGNISED VALUE READS ``False`` here, which is what the run gets: rung 2
    refuses it by name. The refusal itself is :func:`dispatch_enabled`'s.
    """
    raw = os.environ.get(DISPATCH_ENABLE)
    if raw is None:
        return DISPATCH_BY_DEFAULT
    return DISPATCH_ENABLE_VALUES.get(raw, False)


def dispatch_enabled() -> bool:
    """The enable, read fresh on every plan build.

    RAISES on a retired switch name, and then on an unrecognised enable value
    (:class:`DispatchEnableValueError`), deliberately — see
    :func:`_refuse_legacy_names` and :data:`DISPATCH_ENABLE_VALUES`. The ladder turns
    either into a named refusal at rung 2; neither is softened here.
    """
    _refuse_legacy_names()   # every plan build, so a stale name cannot run quietly
    refusal = _enable_value_refusal(os.environ.get(DISPATCH_ENABLE))
    if refusal is not None:
        raise DispatchEnableValueError(refusal)
    return _enable_as_set()


def warm_pass_enabled() -> bool:
    return os.environ.get(WARM_SWITCH, "").strip() != "0"


#: The one label that rides ``fuse_ade`` rather than ``fuse``. Named here because
#: the two composer arguments are separate and a switch that drove only ``fuse``
#: would silently do nothing when asked for this arm.
FUSED_ADE_STATE_LABEL = "fused ADE state"


def requested_fused_arms() -> Tuple[str, ...]:
    """The fused arm labels this process opted into, sorted and de-duplicated.

    Empty is the shipped answer. Read fresh on every plan build, like every other
    switch here, so a gate can flip it between two legs of one process — which is
    exactly how the driver-route gate runs its A/B.

    :data:`FUSE_ARMS_VETO` is DROPPED here rather than returned as a label. It is a
    value of the switch, not a member of the list it carries, and letting it
    through would put ``'0'`` into ``fusion.opted_in`` and into clause (8)'s
    "not admitted" text as though a user had asked for an arm by that name.
    """
    raw = os.environ.get(FUSE_ARMS_SWITCH, "")
    return tuple(sorted({part.strip() for part in raw.split(",")
                         if part.strip() and part.strip() != FUSE_ARMS_VETO}))


def fused_arms_vetoed() -> bool:
    """Whether this process vetoed the fused route. Read fresh on every plan build.

    TRUE WHEREVER THE TOKEN APPEARS, beside arm labels included — see
    :data:`FUSE_ARMS_VETO` for why the ambiguous value answers to the safe branch
    instead of raising. This function can only ever REMOVE arms from what clause
    (8) admits; there is no value of the switch it reads that adds one.
    """
    raw = os.environ.get(FUSE_ARMS_SWITCH, "")
    return any(part.strip() == FUSE_ARMS_VETO for part in raw.split(","))


def subnormal_install_enabled() -> bool:
    """Whether dispatch may INSTALL the policy itself. Read fresh on every plan build."""
    return os.environ.get(SUBNORMAL_INSTALL_SWITCH, "").strip() != "0"


def _launch_grid_of(plan: Any) -> Optional[Tuple[int, ...]]:
    """The single-axis launch grid a plan will actually subscript, or None.

    Read through BOTH spellings — the stored ``_grid`` and the derived
    ``launch_grid`` property — because the empty-grid mechanism has to VERIFY
    that the attribute it emptied is the one the launch reads, rather than
    assume it. A plan whose grid is derived (the folded off-diagonal product
    computes it from ``n_elem`` and ``block``) answers here after the backing
    attribute moves, which is what makes that case measurable instead of a guess.

    A WRAPPER IS READ THROUGH TO THE PLAN THAT LAUNCHES, and only in that
    direction. ``deposit_repair.LeadingRepairPlan`` (deposit_repair.py:668-696)
    occupies the leading slot of a repair-carrying fused pair, holds the pair in
    ``inner`` and launches it inside its own ``run``; it spells no grid of its
    own, so this used to answer None there. MEASURED on 2026-08-29, the
    driver-route gate's first fused run on an RTX A6000: ``step_D`` carried
    ``fused pair D`` with ``dispatches: 13`` and ``programs_per_dispatch: null``
    — the one route whose non-vacuity signal is NEW reported the counter's
    "unknown", so "a kernel launched" and "a kernel computed over data" could not
    be told apart on it, which is the exact distinction
    :attr:`FastPathPlan.launch_counters` exists to draw (a warm launch at
    ``grid=(0,)`` counts in every other counter).

    THE ABSORBED SLOT IS DELIBERATELY NOT READ THROUGH. ``NoopPlan`` and
    ``TrailingRepairPlan`` both name the plan that did the work in
    ``absorbed_by``, and following THAT would report the pair's program count for
    a slot whose consult launched nothing — a claim about a launch that did not
    happen there. None on those slots is the honest answer and means unknown,
    which is what the property's own docstring already says.
    """
    for attribute in ("_grid", "launch_grid"):
        try:
            grid = getattr(plan, attribute)
        except Exception:  # noqa: BLE001 - a property that raises is simply not readable
            continue
        if isinstance(grid, tuple) and len(grid) == 1:
            return grid
    inner = getattr(plan, "inner", None)
    if inner is not None and inner is not plan:
        return _launch_grid_of(inner)
    return None


def _plan_launches_of(plan: Any) -> Optional[int]:
    """How many launches the PLAN ITSELF counted, or None when it counts none.

    THE SECOND, INDEPENDENT WITNESS that a dispatched slot ran a kernel. Everything
    else :attr:`FastPathPlan.launch_counters` reports is this module's own
    bookkeeping — ``dispatches`` is incremented by :meth:`FastPathPlan._ran`, one
    line after the call — so a defect in the seam's accounting would report itself
    as success. ``launches`` is kept by the object that performs the launch and is
    written nowhere in this file.

    THE SAME TWO READING RULES AS :func:`_launch_grid_of`, and for the same reasons.
    A WRAPPER IS READ THROUGH ``inner`` to the plan that launches, so a
    deposit-repair or residency-bracketed leading slot answers for the launch it
    performs. THE ABSORBED SLOT IS DELIBERATELY NOT READ THROUGH ``absorbed_by``:
    following it would report the pair's launch count for a slot whose consult
    launched nothing, which is a claim about a launch that did not happen there.

    ``None`` means the plan spells no counter this module can read — every Triton
    plan except the two scratch-output off-diagonal welds, whose shared base keeps
    ``launches`` as a count of ``run`` calls and deliberately not of ``warm`` — and
    it is UNKNOWN, not zero.
    """
    entries = plan if isinstance(plan, (list, tuple)) else (plan,)
    total = 0
    counted = False
    for entry in entries:
        current = entry
        value: Optional[int] = None
        for _ in range(_ABSORB_CHAIN_LIMIT):
            try:
                found = getattr(current, "launches")
            except Exception:  # noqa: BLE001 - a property that raises is not readable
                found = None
            if isinstance(found, int) and not isinstance(found, bool):
                value = found
                break
            inner = getattr(current, "inner", None)
            if inner is None or inner is current:
                break
            current = inner
        if value is None:
            return None
        counted = True
        total += value
    return total if counted else None


def _warm_with_empty_grid(plan: Any, invoke: Any) -> Optional[str]:
    """Launch ``plan`` with an EMPTY grid so Triton compiles and enqueues nothing.

    Returns ``None`` when the plan was warmed, or a REASON when no attribute this
    function can set empties that plan's launch grid. Raises when the launch
    itself failed (the kernel will not compile) or when the plan could not be put
    back — both of which unfill the slot at the caller, where nothing has been
    written yet.

    TWO backing attributes, tried in order, because the certified products spell
    their grid two ways and one of them is READ-ONLY: ``PmlCurlPlan`` and its
    siblings store ``_grid``; ``FoldedOffdiagConstitutivePlan`` exposes
    ``launch_grid`` as a property with no setter and derives it from ``n_elem``,
    which IS settable and is additionally the element count the kernel masks on.
    Setting the derived source rather than the derived value is why this reads the
    grid back and checks it is ``(0,)`` before launching: an attribute that does
    not empty the grid is a reason, never a launch.
    """
    for attribute, empty in (("_grid", (0,)), ("n_elem", 0)):
        try:
            original = getattr(plan, attribute)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(original, type(empty)) or isinstance(original, bool):
            continue
        if isinstance(original, tuple) and len(original) != 1:
            continue
        try:
            setattr(plan, attribute, empty)
        except Exception:  # noqa: BLE001 - a read-only attribute is not a failure, it is the next candidate
            continue
        if _launch_grid_of(plan) != (0,):
            setattr(plan, attribute, original)  # It was settable a line ago.
            continue
        error: Optional[BaseException] = None
        try:
            invoke()
        except BaseException as exc:  # noqa: BLE001 - re-raised below, after the restore
            error = exc
        try:
            setattr(plan, attribute, original)
        except Exception as restore_error:  # noqa: BLE001
            raise RuntimeError(
                f"the warm pass emptied {attribute!r} on {type(plan).__name__} and could "
                f"not put it back ({restore_error!r}); the slot must not dispatch"
            ) from (error if error is not None else restore_error)
        if error is not None:
            raise error
        return None
    return (f"{type(plan).__name__} exposes no settable attribute that empties its "
            "launch grid, so the empty-grid warm mechanism does not apply to it")


def _warm_polarization_plans(entries: Any, drive: Any) -> Optional[str]:
    """Warm every per-susceptibility ADE plan, ROTATION AND ALL, then put it back.

    ``update_P`` is the one slot the warm pass used to skip, and skipping it made a
    recoverable condition fatal: a kernel that will not compile there raises in the
    MIDDLE of step 1, after ``step_B``, ``update_H``, ``step_D`` and ``update_E``
    have already advanced the fields — exactly the half-applied step the plan-time
    rule exists to prevent, for the one slot the rule did not cover.

    The obstacle was real and is handled rather than avoided: ``AdeUpdatePPlan.run``
    ROTATES three buffer references per component (``P`` takes the scratch, the
    scratch takes ``P_prev``, ``P_prev`` takes the old ``P``), so a zero-program
    launch still permutes them and would leave ``P`` naming an unwritten array. The
    rotation is a permutation of REFERENCES and the launch writes nothing, so
    snapshotting the three names per component and restoring them afterwards
    returns the state exactly — no array content is involved either way.
    """
    for entry in entries:
        warm = getattr(entry, "warm", None)
        if callable(warm):
            warm()
            continue
        if drive is None:
            return ("update_P holds the per-susceptibility plan list, whose run "
                    "rotates the polarization buffers; warming it needs the "
                    "drive-field reader and this Fields exposes none")
        state = getattr(entry, "state", None)
        components = tuple(getattr(entry, "components", ()) or ())
        if state is None or not components:
            return (f"{type(entry).__name__} exposes no polarization state to snapshot, "
                    "so its buffer rotation cannot be undone after a warm launch")
        try:
            saved_p = {name: state.P[name] for name in components}
            saved_prev = {name: state.P_prev[name] for name in components}
            saved_scratch = state._scratch
        except Exception as exc:  # noqa: BLE001
            return (f"the polarization buffers are not readable for a snapshot ({exc!r}), "
                    "so the rotation a warm launch performs could not be undone")
        try:
            skipped = _warm_with_empty_grid(entry, lambda: entry.run(drive))
        finally:
            for name in components:
                state.P[name] = saved_p[name]
                state.P_prev[name] = saved_prev[name]
            state._scratch = saved_scratch
        if skipped is not None:
            return skipped
    return None


def warm_plan(plan: Any, drive: Any = None) -> Optional[str]:
    """Compile a plan's kernel WITHOUT executing it, at plan time.

    Triton JIT-compiles at the first launch — ``PmlCurlPlan.run`` does the import
    and the ``kernel[grid](...)`` subscript inline — so without this a compile
    failure lands in the middle of step 1, after earlier sub-steps of that step
    have already written the device. A dispatcher that must not fall back at
    runtime (a half-launched sub-step re-run on the array path double-applies it)
    needs compilation to be a PLAN-TIME event, and this is it.

    The mechanism is the plan's own launch with an EMPTY grid: Triton compiles on
    the host at the subscript call and enqueues zero programs, so nothing is read
    and nothing is written. See :func:`_warm_with_empty_grid` for how the grid is
    emptied through whichever attribute backs it, and
    :func:`_warm_polarization_plans` for the ``update_P`` list, whose entries take
    ``drive`` and rotate their buffers.

    ``update_P`` IS A LIST ONLY ON THE TRITON TABLE. The other two composers fill it
    with ONE plan — a Metal ``MetalAdeUpdatePPlan``, a third-backend ``NoopPlan`` or
    trailing half — and a single object takes the ORDINARY path below: ``warm()`` if
    it has one, else the empty-grid mechanism, else a recorded reason. That is the
    right answer rather than a special case, because the rotation the list branch
    exists to undo is the Triton ADE plan's own; nothing to snapshot means nothing to
    restore. The ``isinstance`` test is what keeps the two apart, and it is on the
    SHAPE rather than on a table name for the same reason the dispatch branch is.

    Returns ``None`` when the plan was warmed, or a REASON when this mechanism
    does not apply to it — a plan with no settable backing attribute, or a
    polarization list with no drive-field reader to launch through. That is not a
    failure: the caller keeps the slot and records that it was not warmed. A warm
    that was ATTEMPTED and FAILED raises, and the caller unfills the slot; nothing
    was written, so that is safe.

    NOT VERIFIED ON HARDWARE. Whether Triton 3.1 compiles-without-executing at
    ``grid=(0,)`` is a measurement this repository has not taken. The dispatch
    artifact records the outcome per slot, and ``MEEP_GPU_WARM=0`` skips the
    pass, so the two can be compared on a device without editing code.
    """
    warm = getattr(plan, "warm", None)
    if callable(warm):
        # A ``warm()`` MAY ANSWER WITH A REASON INSTEAD OF A COMPILE, and that is the
        # same three-way contract this function has always had, moved one level down.
        # The hand-CUDA products compile through their family module's
        # ``_get_kernel``; a family that exposes none has no compile to perform ahead
        # of its first launch, and RAISING there would unfill a working slot over the
        # absence of a warm mechanism. ``None`` still means warmed, which is what
        # every plan with a ``warm()`` answered before this.
        return warm() or None
    if isinstance(plan, (list, tuple)):
        return _warm_polarization_plans(plan, drive)
    return _warm_with_empty_grid(plan, plan.run)


# ---------------------------------------------------------------------------
# Which certification each dispatched arm rides on
# ---------------------------------------------------------------------------

#: The nine certified families' re-run record, as a key in ``fingerprints.json``.
#:
#: IT USED TO BE A PATH, and that was the defect: the path resolves into
#: ``parity/meep_gpu/results/``, which ``.gitignore`` excludes, so a clone has no
#: such file, ``_fingerprints().get(<path>)`` was always ``None``, and all nine of
#: the families certified this round reached a user's artifact carrying a family
#: name, a dead path and nothing else — no ``recorded_utc``, no host, no run id —
#: while the eight legacy families carried full provenance. The facts are now
#: TRANSCRIBED into the tracked record under this key (the results directory is
#: still named there, as the source they came from), so the artifact reports a
#: lookup that resolves in a fresh clone.
FAMILY_RECERT_GATE = "family_recert_2026-08-14"

#: Arm label -> the family it belongs to and the gate that certified it. The arm
#: label is what ``TritonStepPlan.selected`` writes, and it is the ONLY seam that
#: distinguishes "ordinary" from "real beta run" from "BFAST run" on ``update_H``:
#: all three build a ``ConstitutivePlan``, so a record naming the class would be
#: reporting an ambiguity as a decision.
#:
#: ``gate`` is either a key in ``triton_kernels/fingerprints.json`` — the artifact
#: then quotes that record's own ``recorded_utc``/``host``/``purpose`` — or the
#: path above, for the families whose provenance lives in their results
#: directory. Either way the artifact reports a LOOKUP, not a claim.
ARM_CERTIFICATION: Mapping[str, Tuple[str, str]] = {
    # Curl arms.
    "PML": ("pml_curl", "bit_identity_gate"),
    "conductive PML": ("conductivity", "conductivity_composition_gate"),
    "no-PML": ("no_pml", "no_pml_composition_gate"),
    "folded PML": ("symmetry", "symmetry_composition_gate"),
    "cylindrical PML": ("cylindrical", "cylindrical_composition_gate"),
    "complex PML": ("complex", FAMILY_RECERT_GATE),
    "real beta PML": ("special_kz", FAMILY_RECERT_GATE),
    "complex beta PML": ("special_kz", FAMILY_RECERT_GATE),
    "BFAST PML": ("bfast", FAMILY_RECERT_GATE),
    "folded complex PML": ("folded_complex", FAMILY_RECERT_GATE),
    "folded real beta PML": ("folded_complex", FAMILY_RECERT_GATE),
    "folded complex beta PML": ("folded_complex", FAMILY_RECERT_GATE),
    "cylindrical complex PML": ("cylindrical_complex", FAMILY_RECERT_GATE),
    # Fill arms — not dispatchable this round (see DRIVER_SLOTS), named so the
    # artifact can still say which product would have served them.
    "mirror fill": ("symmetry", "symmetry_composition_gate"),
    "folded complex fill": ("folded_complex", FAMILY_RECERT_GATE),
    # Constitutive arms.
    "ordinary": ("constitutive", "bit_identity_gate"),
    "dispersive": ("dispersive_composition", "dispersive_composition_gate"),
    "folded": ("symmetry", "symmetry_composition_gate"),
    "cylindrical": ("cylindrical", "cylindrical_composition_gate"),
    "complex": ("complex", FAMILY_RECERT_GATE),
    "real beta run": ("special_kz", FAMILY_RECERT_GATE),
    "complex beta run": ("special_kz", FAMILY_RECERT_GATE),
    "BFAST run": ("bfast", FAMILY_RECERT_GATE),
    "nonlinear": ("nonlinear", FAMILY_RECERT_GATE),
    "off-diagonal": ("offdiag", FAMILY_RECERT_GATE),
    "folded complex": ("folded_complex", FAMILY_RECERT_GATE),
    "folded beta run": ("folded_complex", FAMILY_RECERT_GATE),
    "cylindrical complex": ("cylindrical_complex", FAMILY_RECERT_GATE),
    "folded off-diagonal": ("folded_offdiag", FAMILY_RECERT_GATE),
    "no-PML null": ("no_pml_constitutive", FAMILY_RECERT_GATE),
    # Polarization arm.
    "ADE update_P": ("ade_update_p", "dispersive_composition_gate"),
    # The residual-group ADMISSIONS (2026-08-16). Every one of these plans a body
    # a gate above ALREADY certified — the arm is a wider ADMISSION of the same
    # kernel, not a new product — so each is mapped to the family whose gate cut
    # those bytes, and the entry says which body it is. The identity on the newly
    # admitted CONFIGURATION is the closure round's device leg
    # (results/residual_closure_2026-08-15/device/newpred/), which is a separate
    # measurement from the family gate named here and is why these arms could be
    # added without a new kernel.
    "nonlinear run PML": ("pml_curl", "bit_identity_gate"),
    "nonlinear run": ("constitutive", "bit_identity_gate"),
    "folded dispersive": ("dispersive_composition", "dispersive_composition_gate"),
    "folded complex off-diagonal PML": ("folded_complex", FAMILY_RECERT_GATE),
    "folded complex off-diagonal": ("complex", FAMILY_RECERT_GATE),
    "no-PML ADE update_P": ("ade_update_p", "dispersive_composition_gate"),
    # New no-absorber products, each bound to its own 2026-08-17 A6000 byte gate.
    # None borrows certification from a superficially similar body: the complex
    # arm has a distinct no-layer binding, the conductive arm has its four-case
    # tail and one-sided ownership rule, and stored E has a live rotating-pole
    # binding.
    "complex no-PML curl": ("complex_no_pml_curl",
                            "triton_complex_no_pml_curl_device_gate"),
    "conductive no-PML": ("no_pml_conductive",
                           "triton_no_pml_conductive_device_gate"),
    "no-PML stored E": ("no_pml_stored_e",
                         "triton_no_pml_stored_e_device_gate"),
    # THE COMPLEX NO-ABSORBER SIBLINGS, moved here from PENDING_DEVICE_GATE_ARMS on
    # 2026-09-14 by the release decision recorded beside that map. Each lookup resolves
    # TODAY and is not a promise about a run that has yet to happen: all three keys
    # are in triton_kernels/fingerprints.json with status PASS and source_drift
    # None, recorded 2026-09-13T07:53:3xZ on the GPU host (RTX A6000, cc 8.6, CuPy
    # 13.5.1, Triton 3.1.0), seeded by seed_triton_welds.py from the fleet
    # artifacts named in each record's own `records` field. The family names are
    # each gate's own, read off the ledger rather than transcribed from a plan.
    "complex ADE update_P": ("complex_ade",
                             "triton_complex_ade_device_gate"),
    "complex conductive no-PML curl": ("complex_no_pml_conductive",
                                       "triton_complex_no_pml_conductive_device_gate"),
    "complex no-PML stored E": ("complex_no_pml_stored_e",
                                "triton_complex_no_pml_stored_e_device_gate"),
    # THE UPDATE_E ARM THE SAME RULING WAS EXTENDED TO ON 2026-09-15, recorded beside
    # PENDING_DEVICE_GATE_ARMS with what stands behind it. NOT the update_H admission
    # `folded complex off-diagonal` mapped above to the complex family: the labels
    # differ in word order alone. The family name is the ledger key's own directory
    # (rebind_triton_welds.CAMPAIGN_DIRS). Unlike its three siblings this lookup was
    # NOT clean when it was ruled -- launch.py pins drifted, policy "see artifact" --
    # and the ruling conditioned the row on the key being re-gated and rebound in the
    # same batch; test_the_complex_folded_offdiag_arm_is_certified_on_a_rebound_weld
    # is what holds the ledger to that.
    "complex folded off-diagonal": ("complex_offdiag",
                                    "triton_complex_offdiag_device_gate"),
    # THE CROSS-SUB-STEP FUSED PRODUCTS, mapped 2026-08-29 with the fusion
    # opt-in. A fused arm was ``("unmapped", "unmapped")`` until now, which was
    # harmless while clause (8) refused every one of them and is NOT harmless the
    # moment one can dispatch: an artifact whose dispatched arm names no gate is
    # the over-claim this map exists to prevent.
    #
    # THE GATE NAMED IS THE ONE THAT CUT THOSE BYTES, caveats included, and one
    # of the caveats is why the mapping matters. ``triton_fused_electric_device_gate``
    # carries ``subnormal_policy: "NOT INSTALLED - see _mixed_policy"`` — the CuPy
    # oracle it was compared against was flushing while Triton kept — so a run
    # dispatching ``fused pair B`` now says so in its own record rather than
    # leaving a reader to find it in ``fingerprints.json``. See
    # ``test_triton_weld_contract.POLICY_UNNAMED_BUDGET``.
    "fused pair B": ("fused_electric", "triton_fused_electric_device_gate"),
    "fused pair D": ("fused_electric", "triton_fused_electric_device_gate"),
    "fused pair B (folded)": ("folded_fused_magnetic_pair",
                              "triton_folded_fused_magnetic_pair_device_gate"),
    "fused pair D (folded)": ("folded_fused_pair",
                              "triton_folded_fused_pair_device_gate"),
    "dispersive fused pair": ("dispersive_fused_pair",
                              "triton_dispersive_fused_pair_device_gate"),
    "fused ADE state": ("fused_ade_state", "triton_fused_ade_state_device_gate"),
    # THE INSTALLER WAVE, 2026-09-02 — three of the twenty-five products
    # ``triton_kernels/launch.CERTIFIED_FUSED_PRODUCTS`` routed in the same edit.
    # THREE, not twenty-five, and the split is the whole content of this comment.
    #
    # A ROW HERE IS A PROMISE THAT THE LOOKUP RESOLVES. Every gate named in this
    # map is a key in ``triton_kernels/fingerprints.json``, and
    # ``test_dispatch_contract.test_every_recorded_certification_points_at_a_readable_record``
    # holds it to that with no exemption — an exemption for gates that lived only
    # in the gitignored results tree is precisely the defect
    # :data:`FAMILY_RECERT_GATE` records having been fixed once, where nine
    # families reached a user's artifact carrying a family name, a dead path and
    # nothing else. So the twenty pending fused labels whose released gate has NOT
    # been transcribed into the ledger get no row here (twenty-three until
    # 2026-09-11, when the three rows below were cut); they are named in
    # :data:`PENDING_DEVICE_GATE_ARMS` instead, with the artifact that released
    # them, and the pending rung refuses them before anything warms. That is
    # wiring without releasing spelled at the level where it is enforced rather
    # than asserted, and it prices the phase that follows: twenty ledger
    # entries have to be CUT BY A GATE RUN — no in-tree tool creates one, and
    # writing one by hand is exactly the record forgery this campaign forbids.
    "fused pair B (complex)": ("complex_fused_magnetic_pair",
                               "triton_complex_fused_magnetic_pair_device_gate"),
    "fused pair D (complex)": ("complex_fused_electric_pair",
                               "triton_complex_fused_electric_pair_device_gate"),
    "fused pair B (cylindrical)": (
        "cylindrical_real_fused_magnetic_pair",
        "triton_cylindrical_real_fused_magnetic_pair_device_gate"),
    # THE 2026-09-11 RELEASE WAVE — the three D-seam products whose arms this round
    # drives through the seam. Every gate named below is a key that
    # ``parity/meep_gpu/seed_triton_welds.py`` CUTS from the fleet artifacts of
    # ``dispatch_fused_route_2026-09-11_realarms``'s campaign, after the campaign,
    # from the run's own recorded digests. It is not typed and it cannot be: no
    # in-tree tool invents a ledger entry, and writing one by hand is the record
    # forgery this campaign forbids.
    #
    # SO THESE THREE ROWS ARE DELIBERATELY UNRESOLVED UNTIL THE SEED RUNS, and the
    # window is named rather than avoided. The ordering rule says every source edit
    # lands BEFORE the campaign, or the run records a file that does not ship; this
    # file is bound by ``driver_dispatch``, by all ten H->D gate artifacts and by
    # all 42 Metal gate artifacts, so it MUST be edited first.
    # ``test_every_recorded_certification_points_at_a_readable_record`` is therefore
    # red between this edit and the seed, by design and by name in the campaign log.
    "fused pair D (conductive)": (
        "conductive_fused_electric_pair",
        "triton_conductive_fused_electric_pair_device_gate"),
    "fused pair D (cylindrical)": (
        "cylindrical_real_fused_electric_pair",
        "triton_cylindrical_real_fused_electric_pair_device_gate"),
    # THE THIRD ROW OF THAT WAVE, and it arrives on the re-attribution round rather
    # than with its two siblings because the arm was held back on a divergence the
    # investigation has since attributed elsewhere (see
    # :data:`PENDING_DEVICE_GATE_ARMS`). The lookup resolves TODAY and is not a
    # promise about a run that has yet to happen:
    # ``triton_folded_dispersive_fused_pair_device_gate`` is a key in
    # ``triton_kernels/fingerprints.json`` with status PASS, recorded
    # 2026-09-13T07:53:41Z on the GPU host (RTX A6000, cc 8.6, CuPy 13.5.1, Triton
    # 3.1.0), seeded by ``seed_triton_welds.py`` from
    # ``results/triton_fleet_2026-09-13_batch_B/folded_dispersive_fused_pair/``,
    # whose own record reads the policy as "keep (installed by the gate before its
    # first device compile)" -- which is the ordering the shipped lift path does NOT
    # take and the reason the array comparison leg, not this kernel, moved.
    "fused pair D (folded dispersive)": (
        "folded_dispersive_fused_pair",
        "triton_folded_dispersive_fused_pair_device_gate"),
    # --- PHASE B, 2026-09-13: seeded by seed_triton_welds.py from the 2026-09-12
    # identity fleet, after the probes learned to record the device identity and the
    # provenance stamp the weld contract requires. ---
    "fused pair B (real beta)": (
        "beta_fused_magnetic_pair", "triton_beta_fused_magnetic_pair_device_gate"),
    "fused pair D (real beta)": (
        "beta_fused_electric_pair", "triton_beta_fused_electric_pair_device_gate"),
    "fused pair B (folded real beta)": (
        "folded_beta_fused_magnetic_pair",
        "triton_folded_beta_fused_magnetic_pair_device_gate"),
    "fused pair D (folded real beta)": (
        "folded_beta_fused_electric_pair",
        "triton_folded_beta_fused_electric_pair_device_gate"),
    "fused pair B (complex beta)": (
        "complex_beta_fused_magnetic_pair",
        "triton_complex_beta_fused_magnetic_pair_device_gate"),
    "fused pair D (complex beta)": (
        "complex_beta_fused_electric_pair",
        "triton_complex_beta_fused_electric_pair_device_gate"),
    "fused pair B (folded complex beta)": (
        "folded_beta_complex_fused_magnetic_pair",
        "triton_folded_beta_complex_fused_magnetic_pair_device_gate"),
    "fused pair D (folded complex beta)": (
        "folded_beta_complex_fused_pair",
        "triton_folded_beta_complex_fused_pair_device_gate"),
    "fused pair B (BFAST)": (
        "bfast_fused_magnetic_pair", "triton_bfast_fused_magnetic_pair_device_gate"),
    "fused pair D (BFAST)": (
        "bfast_fused_electric_pair", "triton_bfast_fused_electric_pair_device_gate"),
    "fused pair B (nonlinear)": (
        "nonlinear_fused_magnetic_pair",
        "triton_nonlinear_fused_magnetic_pair_device_gate"),
    "fused pair B (folded complex)": (
        "folded_complex_fused_magnetic_pair",
        "triton_folded_complex_fused_magnetic_pair_device_gate"),
    "fused pair D (folded complex)": (
        "folded_complex_fused_pair", "triton_folded_complex_fused_pair_device_gate"),
    "fused pair B (cylindrical complex)": (
        "cylindrical_fused_magnetic_pair",
        "triton_cylindrical_fused_magnetic_pair_device_gate"),
    "fused pair D (cylindrical complex)": (
        "cylindrical_fused_electric_pair",
        "triton_cylindrical_fused_electric_pair_device_gate"),
    # --- THE TARGET ROUND, 2026-09-13: ONE ROW, and it is a lookup that already
    # resolves rather than one the campaign still owes. The gate below is a key in
    # ``triton_kernels/fingerprints.json`` TODAY, status PASS, recorded
    # 2026-09-13T07:53:43Z on the GPU host (RTX A6000, cc 8.6, CuPy 13.5.1, Triton
    # 3.1.0, GPU 1), seeded by ``seed_triton_welds.py`` from
    # ``results/triton_fleet_2026-09-13_batch_C/no_pml_fused_electric_pair/`` -- the
    # family name here is that record's own directory, read off the ledger rather
    # than transcribed from a plan. So this row is NOT in the deliberately-red
    # window the 2026-09-11 wave above describes: the arm's
    # :data:`PENDING_DEVICE_GATE_ARMS` entry said "no fingerprints.json entry has
    # been cut for it", which stopped being true before this edit, and deleting it
    # is a correction rather than a release. What the arm still owes is its two
    # ROUTE cases, which is a statement about :data:`RELEASED_FUSED_ARMS` and is
    # made there.
    "fused pair D (no-PML stored E)": (
        "no_pml_fused_electric_pair",
        "triton_no_pml_fused_electric_pair_device_gate"),
    # --- 2026-09-15: THE COMPLEX CONDUCTIVE NO-PML PAIR, and this row is
    # DELIBERATELY UNRESOLVED UNTIL THE SEED RUNS, in the window the 2026-09-11 wave
    # above names. The key does not exist in ``triton_kernels/fingerprints.json``
    # when this edit lands. It is CUT by ``parity/meep_gpu/seed_triton_welds.py
    # --only complex_conductive_fused_pair`` from the pair gate re-run through
    # ``drive_triton_weld_gates.py`` AFTER this edit (the gate records this file), and
    # the family name is that driver's gate directory, from which
    # ``seed_triton_welds.ledger_key`` derives the key. No existing artifact could
    # be seeded instead: read 2026-09-15, every Triton gate.json this product has
    # (2026-08-30 to ``triton_fleet_2026-09-14_target_A``) records ``environment``
    # without ``device`` and a hand-written ``subnormal_policy`` with no
    # ``requested``/``resolved``, so ``probe_triton_complex_conductive_fused_pair.py``
    # gains the identity and policy stamps in the same batch.
    # ``test_every_recorded_certification_points_at_a_readable_record`` is red
    # between this edit and the seed, by design and by name.
    "fused pair D (complex conductive no-PML)": (
        "complex_conductive_fused_pair",
        "triton_complex_conductive_fused_pair_device_gate"),
    # --- THE SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-17 -- and these two rows ARE
    # in the deliberately-red window the 2026-09-11 wave above names, for its
    # ordering reason. Each key is the one ``seed_triton_welds.ledger_key`` derives
    # from a FAMILY-NAMED campaign directory, and the run that cuts it is
    # ``gate_triton_offdiag_stencil_welds.py`` re-run on the bytes this edit ships.
    # No earlier stencil artifact can seed it: the 2026-09-13 batch records no
    # ``environment`` block, writes directories named ``plain``/``folded`` (which
    # would mint ``triton_plain_device_gate``), and pins a ``launch.py`` and a
    # ``fastpath.py`` this edit moves. Until that seed runs,
    # ``test_every_recorded_certification_points_at_a_readable_record`` is red on
    # these two labels by name.
    "fused pair D (off-diagonal)": (
        "offdiag_fused_electric_pair",
        "triton_offdiag_fused_electric_pair_device_gate"),
    "fused pair D (folded off-diagonal)": (
        "folded_offdiag_fused_electric_pair",
        "triton_folded_offdiag_fused_electric_pair_device_gate"),
}

#: The two arms a FUSED label's kernel implements, so the rungs that decide on an
#: ARM can still see through a label that replaced two of them.
#:
#: WHY THIS EXISTS AT ALL, and it is a hole the installer wave opened rather than a
#: tidy-up. Rung (7a) refuses a plan whose ``selected`` names an arm in
#: :data:`PENDING_DEVICE_GATE_ARMS`; it reads one label per slot. A fused product
#: absorbs BOTH slots and writes ITS OWN label into both, so the constituent arms
#: vanish from ``selected`` and the rung passes over them — and
#: ``complex_conductive_fused_pair`` is a real instance: it absorbs
#: ``complex conductive no-PML curl`` and ``complex no-PML stored E``, and BOTH are
#: pending. Without this map, wiring that product would have quietly removed a
#: standing refusal by covering the arms it names.
#:
#: IT IS A NARROWING AND NEVER A WIDENING. Every existing released arm's
#: constituents are non-pending (``PML``/``ordinary``, ``folded PML``/``folded``,
#: ``PML``/``dispersive``), so no behaviour on the shipped release changes; what the
#: map can do is refuse MORE, which is the only direction a wiring round is allowed
#: to move a gate.
#:
#: THE VOCABULARY LIVES HERE AND THE COMPOSER IS WHAT IT IS CHECKED AGAINST, exactly
#: as :data:`DRIVER_SLOTS` is: ``test_triton_certified_fused_products`` compares this
#: map with ``launch.FUSED_PAIR_ARMS``, ``launch.FOLDED_FUSED_PAIR_ARMS``,
#: ``launch.CERTIFIED_FUSED_PAIR_ARMS`` and ``CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` —
#: reading the composer's tables rather than sharing a symbol with them — and
#: requires the map to be TOTAL over the labels those tables can write, because a
#: label missing from it is refused by name at the rung below.
FUSED_ARM_CONSTITUENTS: Mapping[str, Tuple[str, ...]] = {
    "fused pair B": ("PML", "ordinary"),
    "fused pair D": ("PML", "ordinary"),
    "dispersive fused pair": ("PML", "dispersive"),
    "fused pair B (folded)": ("folded PML", "folded"),
    "fused pair D (folded)": ("folded PML", "folded"),
    "fused pair B (complex)": ("complex PML", "complex"),
    "fused pair D (complex)": ("complex PML", "complex"),
    "fused pair B (cylindrical)": ("cylindrical PML", "cylindrical"),
    "fused pair D (cylindrical)": ("cylindrical PML", "cylindrical"),
    "fused pair B (cylindrical complex)": ("cylindrical complex PML",
                                           "cylindrical complex"),
    "fused pair D (cylindrical complex)": ("cylindrical complex PML",
                                           "cylindrical complex"),
    # THE H->D SEAM, 2026-09-07: (update_H arm, step_D arm) of the two cylindrical
    # products; both INSTALLABLE = False, so the composer never writes either label,
    # and both stay in PENDING_DEVICE_GATE_ARMS until a ledger entry exists.
    "fused pair H->D (cylindrical)": ("cylindrical", "cylindrical PML"),
    "fused pair H->D (cylindrical complex)": ("cylindrical complex",
                                              "cylindrical complex PML"),
    "fused pair H->D (complex)": ("complex", "complex PML"),
    # Both cells of the one launch, because the rung asks about the arms the
    # product COULD have absorbed and the answer must not depend on which cell it
    # took on this grid.
    "fused pair B (folded complex)": ("folded complex PML", "folded complex",
                                      "folded complex off-diagonal PML",
                                      "folded complex off-diagonal"),
    "fused pair D (folded complex)": ("folded complex PML", "folded complex"),
    "fused pair B (folded complex beta)": ("folded complex beta PML",
                                           "folded complex"),
    "fused pair D (folded complex beta)": ("folded complex beta PML",
                                           "folded complex"),
    "fused pair B (folded real beta)": ("folded real beta PML",
                                        "folded beta run"),
    "fused pair D (folded real beta)": ("folded real beta PML",
                                        "folded beta run"),
    "fused pair B (complex beta)": ("complex beta PML", "complex beta run"),
    "fused pair D (complex beta)": ("complex beta PML", "complex beta run"),
    "fused pair B (real beta)": ("real beta PML", "real beta run"),
    "fused pair D (real beta)": ("real beta PML", "real beta run"),
    "fused pair B (BFAST)": ("BFAST PML", "BFAST run"),
    "fused pair D (BFAST)": ("BFAST PML", "BFAST run"),
    "fused pair B (nonlinear)": ("nonlinear run PML", "nonlinear run"),
    "fused pair D (folded dispersive)": ("folded PML", "folded dispersive"),
    "fused pair D (conductive)": ("conductive PML", "ordinary"),
    "fused pair D (complex conductive no-PML)": ("complex conductive no-PML curl",
                                                 "complex no-PML stored E"),
    "fused pair D (no-PML stored E)": ("conductive no-PML", "no-PML stored E",
                                       "no-PML"),
    # The two scratch-output off-diagonal welds, 2026-09-15: each absorbs exactly its
    # ``launch.CERTIFIED_FUSED_PAIR_ARMS`` row and declares no second cell. Both
    # constituents of each are certified arms (``PML``/``off-diagonal`` and
    # ``folded PML``/``folded off-diagonal``), so the pending rung sees nothing here.
    "fused pair D (off-diagonal)": ("PML", "off-diagonal"),
    "fused pair D (folded off-diagonal)": ("folded PML", "folded off-diagonal"),
    # ``fused ADE state`` is NOT a pair — it occupies ``update_P`` ALONE — and its
    # row names ONE arm for that reason. The map's subject is "the arms this label
    # stands in place of", not "the second slot it took": ``plan_step``'s
    # ``fuse_ade`` block chooses between ``plan_fused_ade_state`` and
    # ``plan_ade_update_p`` for the same slot, so the arm it displaces is
    # ``ADE update_P`` and nothing else. Leaving it out would refuse it at the
    # unnamed-absorb clause below, which is a behaviour change this wiring round has
    # no evidence for.
    "fused ADE state": ("ADE update_P",),
}

#: Planner-visible arms whose release certification is not yet complete.  A
#: dedicated family gate is necessary but not sufficient: the central planner
#: and driver seam must also be recertified against the bytes that contain the
#: arm.  Keeping these labels here makes an opted-in run refuse the whole plan
#: before warming instead of borrowing a sibling family's provenance.
PENDING_DEVICE_GATE_ARMS: Mapping[str, str] = {
    # ``fused pair D (folded dispersive)`` LEFT THIS MAP ON THE RE-ATTRIBUTION ROUND,
    # and it left because the hold was a MISATTRIBUTION rather than because a ledger
    # entry was cut. Its row read "synchronize_magnetic_fields diverges from the
    # array path on this shape with fusion OFF as well as on, so the second consult
    # site cannot license it". The device investigation that went looking for that
    # divergence did not find it and found what produced the signature instead.
    #
    # WHAT WAS MEASURED, driving the real route gate on the GPU host with the arm
    # admitted in-process at 7 of 7 slots over the full 1600-step ladder, three
    # times in fresh processes: PASS-FUSED every time, ``first_divergent_checkpoint``
    # null, 0 of 41 arrays differing over 200,080 words on BOTH ``fused_vs_array``
    # and ``unfused_vs_array``, the second consult site PASS with
    # ``state_entering_the_site`` {true, true}, and substitution 7.00 against 9.00
    # launches per step. Not a degenerate pass: the armed nulls FIRE on the same
    # runs (absorbed-consult-withheld, in-seam-pass-mutated-B and
    # in-seam-pass-mutated-D all diverged as required, ENVELOPE-HOLDS). Across every
    # artifact on both machines there are 1,606 recorded second-consult verdicts and
    # ZERO failures, of which 86 are ``folded_dispersive_2d`` rows over 23 runs and
    # all three backends, every one site=PASS with both legs byte-identical.
    #
    # AND THERE IS NO MECHANISM: this arm spans ``step_D``/``update_E`` while
    # ``synchronize_magnetic_fields`` consults only ``step_B``, ``fill_B``,
    # ``fill_folded_far_ghosts_B`` and ``update_H_synchronize``. Installing an
    # ELECTRIC product cannot move the magnetic half-step, which is why "with fusion
    # off as well as on" never had an explanation.
    #
    # WHAT ACTUALLY PRODUCED THE SIGNATURE, and it is a real defect that keeps its
    # own owner: the subnormal policy installs LAZILY, at the first dispatch's plan
    # freeze, so every CuPy kernel compiled before that carries CuPy's own
    # ``-ftz=true``. The ARRAY comparison leg then flushes subnormals while both
    # dispatch legs keep them, which is exactly why the two dispatch legs differed
    # from the array leg by IDENTICAL amounts. Controlled A/B on the same case list
    # with the same arm and one flag: the shipped ordering diverges at checkpoint
    # 48, and installing the policy first PASSES. The engine-side fix (installing
    # the policy before the first CuPy compile in ``from_meep.lift_simulation``,
    # plus a pre-install compile counter in ``subnormal_policy``) is a SEPARATE
    # round; nothing in this file's seam or gate code moved for it.
    "complex no-PML off-diagonal":
        "its family byte gate passed on the device, but the central planner/driver "
        "recertification and tracked fingerprint entry are pending",
    "folded off-diagonal dispersive":
        "its repaired family byte gate passed on the device, but the central "
        "planner/driver recertification and tracked fingerprint entry are pending",
    # THREE ARMS LEFT THIS MAP ON 2026-09-14 — `complex ADE update_P`,
    # `complex conductive no-PML curl` and `complex no-PML stored E` — and they left
    # on a RELEASE DECISION rather than on a new run, which is recorded here because
    # the map's rule above says a family gate is "necessary but not sufficient".
    #
    # WHAT WAS STALE. Their reasons read "its dedicated CUDA byte gate has not run"
    # and "tracked fingerprint entry are pending". Both had stopped being true: the
    # three family gates ran and RELEASED on 2026-09-13 and sit in
    # triton_kernels/fingerprints.json today as triton_complex_ade_device_gate,
    # triton_complex_no_pml_conductive_device_gate and
    # triton_complex_no_pml_stored_e_device_gate — status PASS, source_drift None,
    # recorded 2026-09-13T07:53:3xZ on the GPU host (RTX A6000, cc 8.6), seeded by
    # seed_triton_welds.py from results/triton_fleet_2026-09-13_batch_{A,D}/.
    #
    # WHAT WAS RULED. The sufficiency half — "the central planner and driver seam
    # must also be recertified against the bytes that contain the arm" — was ruled
    # SATISFIED on 2026-09-14: the driver half is re-cut from the route
    # run by recut_driver_dispatch_record.py, and the only planner record
    # (planner_integration_recert_2026-08-14) is hand-authored, no in-tree tool
    # writes one, it PREDATES these gates, and its own `dispatch` field is stale
    # ("STILL DISABLED ... returns None on every branch"), which dispatch no longer
    # is. The ruling is the licence; this comment is the record of it, so a reader
    # who finds these three in ARM_CERTIFICATION can see it was decided rather than
    # drifted into.
    #
    # A FOURTH LEFT ON 2026-09-15, ON THE SAME RULING EXTENDED BY NAME:
    # `complex folded off-diagonal`, the update_E arm. It is NOT `folded complex
    # off-diagonal`, the update_H admission ARM_CERTIFICATION has mapped to the
    # complex family since 2026-08-16. The two labels differ in word order alone,
    # and this entry concerns the first.
    #
    # WHAT STANDS BEHIND IT. Its family gate is triton_complex_offdiag_device_gate,
    # status PASS, recorded 2026-09-13T07:53:36Z from
    # results/triton_fleet_2026-09-13_batch_D/complex_offdiag/ (07:03:52Z to
    # 07:04:06Z): five product rows IDENTICAL over 8 cycles -- folded_3d 110,592
    # words, folded_twofold (XY mirror, metallic walls) 8,112, folded_bloch (Y
    # mirror, in-plane k_point) 16,128, no_pml_3d 281,250, no_pml_2d 11,250 -- five
    # mutations DIVERGENT (708 to 93,750 words) and every refusal predicate agreeing.
    # The twofold and Bloch tails are the shapes of the three corpus rows the arm
    # serves. planner_integration_recert_2026-08-14 never covered this family, which
    # is the ground the 2026-09-14 ruling had already set aside for its siblings.
    #
    # WHAT THE ROUTE MEASURED WHILE IT WAS HERE. On folded_complex_offdiag_2d and
    # folded_complex_nobloch_offdiag_2d the composer builds this arm for update_E
    # beside `fused pair B (folded complex)` on step_B/update_H, and this map refused
    # the WHOLE plan. dispatch_fused_route_2026-09-15_weldgrid drove both as
    # `dispatch` and read DID-NOT-FUSE / NO-DROP, every slot on the array path;
    # dispatch_fused_route_2026-09-15_offdiag drove both as `no-fused-arm` and read
    # PASS-NO-FUSED-ARM with refusing_rung `pending-device-gate`, byte-identical to
    # the array path at 1600 and 1000 steps (287,680 and 380,538 words).
    #
    # WHAT THE LIFT IS CONDITIONED ON, because unlike the three above this entry was
    # not clean when it was ruled. Its launch.py pins had drifted (source de606ebe
    # recorded against b7701503 in the tree, code a6b3a46c against af61e4f0) and its
    # subnormal_policy read "see artifact", the one name in
    # test_triton_weld_contract.POLICY_UNNAMED_BUDGET. So gate_triton_complex_offdiag.py
    # is re-run under keep on the bytes that ship -- AFTER this edit, because the gate
    # imports this file -- writing the environment block and policy stamp it lacked,
    # and rebind_triton_welds.py rebinds the key in the same batch, deriving the
    # policy from that stamp (its POLICY_UNNAMED rule). Until that rebind,
    # `test_the_complex_folded_offdiag_arm_is_certified_on_a_rebound_weld` is red by
    # design.
    #
    # `test_the_residual_closure_arms_are_explicitly_fail_closed_until_recertified`
    # pins the two that REMAIN, and its disjointness clause still holds for them.
    # ---------------------------------------------------------------------
    # THE INSTALLER WAVE'S UNTRANSCRIBED LABELS: the twenty that remain after
    # 2026-09-11, of the twenty-three the 2026-09-02 wave wired.
    # ---------------------------------------------------------------------
    # Every one of these is a fused LABEL rather than a sub-step arm, and every one
    # of them has a device gate that RELEASED — the difference from the six rows
    # above, whose gates have not run at all. What is missing is the other half of
    # a certification: a tracked ledger entry. Their gates live under
    # ``parity/meep_gpu/results/``, which ``.gitignore`` excludes, so
    # ``triton_kernels/fingerprints.json`` holds no key for them and
    # :func:`_certification_for` could only report a family name beside a path a
    # clone cannot open. The fusion board binds each of these artifacts to the live
    # module bytes by ``source_sha256`` on every cut (its
    # ``release_binding_per_product`` block), so the evidence exists and is checked
    # — it is simply not in the ledger a dispatching artifact can quote.
    #
    # SO THEY ARE WIRED AND REFUSED, which is this round's whole intent: the
    # composer can SELECT them, which is the precondition a release campaign
    # measures against, and the ladder stops them here — before the warm pass,
    # before any kernel compiles, and without borrowing a sibling family's
    # provenance. Clause (8) refuses them too (none is in
    # :data:`RELEASED_FUSED_ARMS`); this rung is what still refuses them when a run
    # opts into a label by name through :data:`FUSE_ARMS_SWITCH`.
    #
    # THE TWO SCRATCH-OUTPUT WELDS ARE NOT HERE, and since 2026-09-15 for the other
    # reason. Until then they were not wired: their plan rotates the engine's own D
    # volumes and ``warm_plan`` would have driven that rotation at plan time. Their
    # shared plan base now has a ``warm`` that never rotates, ``launch.py`` routes
    # both, and their labels sit in :data:`ARM_CERTIFICATION` keyed to the ledger
    # entries ``seed_triton_welds.py`` cuts from the stencil gate re-run on the
    # routing bytes -- the named window the 2026-09-11 wave also ran through, not a
    # pending hold.
    #
    # WHAT LIFTS A ROW: a gate run that cuts the ledger entry, and since 2026-09-11
    # the tool that cuts it is named — ``parity/meep_gpu/seed_triton_welds.py``,
    # which reads a fleet campaign's artifacts and writes the key from the run's own
    # recorded digests, refusing any artifact that does not carry them. No in-tree
    # tool INVENTS one — ``parity/meep_gpu/rebind_triton_welds.py`` rebinds keys that
    # already exist and refuses to create one — and writing one by hand is the
    # record forgery this campaign forbids. Each reason names the artifact so the
    # run that has to be repeated is identified rather than searched for.
    #
    # THREE ROWS LEFT ON 2026-09-11 by exactly that route: ``fused pair D
    # (cylindrical)``, ``fused pair D (folded dispersive)`` and ``fused pair D
    # (conductive)`` were seeded from the ``_realarms`` fleet and moved to
    # :data:`ARM_CERTIFICATION`. The twenty below keep their artifact paths, which
    # are still true.
    "fused pair H->D (cylindrical)":
        "its gate released (results/triton_cylindrical_real_fused_hd_pair_2026-09-07/"
        "keep/gate.json) but no fingerprints.json entry has been cut for it, and the "
        "product declares INSTALLABLE = False (the composer prices its 7 -> 3 launch "
        "saving as a loss)",
    "fused pair H->D (cylindrical complex)":
        "its gate released (results/triton_cylindrical_fused_hd_pair_2026-09-07/"
        "keep/gate.json) but no fingerprints.json entry has been cut for it, and the "
        "product declares INSTALLABLE = False",
    "fused pair H->D (complex)":
        "its gate released (results/triton_complex_fused_hd_pair_2026-09-07/"
        "keep/gate.json) but no fingerprints.json entry has been cut for it, and the "
        "product declares INSTALLABLE = False -- the composer prices this span as a "
        "LOSS on every row of its cell, because on a complex row BOTH neighbouring "
        "pairs install (`fused pair B (complex)` holds step_B+update_H and `fused "
        "pair D (complex)` holds step_D+update_E) and a two-slot H->D span turns two "
        "pairs into one",
    # ``fused pair D (complex conductive no-PML)`` LEFT THIS MAP ON 2026-09-15, by the
    # route this map names as the only one that lifts a row: its
    # :data:`ARM_CERTIFICATION` row landed in the same batch as its
    # :data:`RELEASED_FUSED_ARMS` row, and the key that row names is cut by
    # ``seed_triton_welds.py`` from the pair gate re-run after the batch. Its reason
    # here ended "seeding it wants the pair gate re-run through
    # drive_triton_weld_gates.py, which stamps the environment identity
    # seed_triton_welds.py refuses to invent", and the second half was wrong: the
    # driver copies a gate's environment into ``gates.jsonl`` as
    # ``host_environment``, while the seeder reads the artifact's OWN
    # ``environment``, and this gate recorded no ``device`` there on any of its
    # twelve Triton runs (read 2026-09-15). A re-run alone would have been refused
    # the same way; what it needed was the probe stamping the identity itself,
    # which ``probe_triton_complex_conductive_fused_pair.py`` does from this batch
    # on. Rung (7a)'s see-through stays clear: both constituents
    # :data:`FUSED_ARM_CONSTITUENTS` names for it left this map on 2026-09-14.
    # ``fused pair D (no-PML stored E)`` LEFT THIS MAP IN THE TARGET ROUND, and it
    # left for the reason the map itself names as the only one that lifts a row: a
    # gate run cut the ledger entry. Its text here read "its gate released
    # (results/triton_regate_2026-09-02_residue/probe_triton_no_pml_fused_electric_pair/
    # gate.json) but no fingerprints.json entry has been cut for it", and
    # ``seed_triton_welds.py`` cut ``triton_no_pml_fused_electric_pair_device_gate``
    # from the 2026-09-13 fleet on the GPU host, status PASS. Its
    # :data:`ARM_CERTIFICATION` row is above. Rung (7a)'s see-through clears with
    # it: :data:`FUSED_ARM_CONSTITUENTS` names ``conductive no-PML``, ``no-PML
    # stored E`` and ``no-PML``, and none of the three is in this map.
}

#: THE DRIVER-ROUTE GATE FOR THE FUSED ROUTE, and the artifact that released it.
#: ``parity/meep_gpu/gate_dispatch_fused_route.py``, run 2026-08-30 on the GPU host
#: GPU 6 (RTX A6000, sm_86, Triton 3.1.0, CuPy 13.5.1, MEEP 1.33.0), four legs,
#: all four ``release.released == true``.
#:
#: WHY IT IS NAMED IN PRODUCT CODE. :data:`RELEASED_FUSED_ARMS` below is the first
#: constant in this file that ADMITS rather than refuses, so a reader has to be
#: able to get from the admission to the measurement without being told where to
#: look, and a dispatching run's own artifact has to cite it. The same rule
#: :data:`ARM_CERTIFICATION` follows for a sub-step family, one level up.
#:
#: WHY IT NAMES ``_bind`` AND NOT THE TWO RUNS BEFORE IT. All three released and
#: all three measured a real route; what separates them is WHICH BYTES they ran.
#: ``_seamgate`` reached every arm through ``MEEP_GPU_FUSE_ARMS``, because at the
#: time it ran no arm was released and the switch was the only way in — evidence
#: about a path no user takes. ``_veto`` drove the three in-envelope cases with the
#: switch UNSET, with this constant's own table doing the admitting, and added two
#: measurements ``_seamgate`` could not: the ENVELOPE declining a configuration on
#: a device (``pml_3d_diagonal``, both directions) and a substitution proof against
#: a baseline that is genuinely unfused (see :data:`FUSE_ARMS_VETO`, without which
#: the baseline fused too and the drop was 0.0). But ``_veto`` ran this file at
#: 29ac1216, and what shipped afterwards was 87f740ee and then d11454ba — so the
#: citation named a run that had never executed the code it was cited from, and
#: three different digests for one file were in play at once
#: (``fingerprints.json`` recorded a fourth, 06cdbf37). ``_bind`` is that run
#: repeated against the bytes that ship, and it adds the case named below.
#:
#: THE ARTIFACT NO LONGER PREDATES THE LINE THAT NAMES IT, and closing that gap is
#: what this round is. The name below, the case list in :data:`RELEASED_FUSED_ARMS`
#: and every other edit landed BEFORE the campaign started, so the four legs each
#: recorded this file at the digest it ships at, and
#: ``fingerprints.json``'s ``driver_dispatch.source_sha256`` is cut from that
#: recording rather than typed. ``test_dispatch_contract`` checks all three against
#: each other — record, artifact and tree — so a self-naming citation that goes
#: stale again fails there instead of being argued about in prose.
#:
#: WHY IT MOVED OFF ``_bind``, 2026-09-02, and it is the same rule applied to a
#: different file. ``driver.py``'s ``_inject_electric_through_conductivity`` now
#: rescales SPARSELY at the deposit indices the sources publish, where it used to
#: run three whole-volume passes; the passes canonicalised ``-0.0`` to ``+0.0``
#: across the whole component and, on CuPy, flushed every non-deposit subnormal
#: word, which is what forced the fused electric pairs to refuse conductive rows.
#: ``_bind``'s four legs ran the OLD ``driver.py`` (cd437accdd37), so the citation
#: named a run that had not executed the bytes that ship — the exact defect the
#: paragraph above was written to close, one file over. ``_bind`` is NOT deleted:
#: it stays as the record of what the old driver did. This name is the same
#: four-leg protocol repeated against the new bytes, with every source edit again
#: landed before the campaign started. What the re-run settles is whether the
#: sparse rescale moved anything the gate certifies; the artifact it names is the
#: answer, and ``test_dispatch_contract`` is what holds this line to it.
#:
#: WHY IT MOVED AGAIN, 2026-09-02, and this time the run is LARGER rather than
#: repeated. ``_sparserescale`` drove six cases and released three arms; this one
#: drives nine and releases five. What is new in it, and what each addition is for:
#:
#:   * ``folded_2d`` and ``folded_dispersive_2d`` DISPATCH rather than refuse. The
#:     fill consults (:data:`DRIVER_SLOTS`) retired the completeness rung for the
#:     fills, so a folded composition reaches the seam for the first time — six of
#:     seven slots, both fill slots carrying the mirror-fill kernel, and the two
#:     folded pairs absorbing four of them. Every earlier campaign recorded the
#:     folded cases as refusals, and a folded case that does not run a fill
#:     in-launch measures nothing about this composition.
#:   * ``pml_1d`` is the 1-D shape the previous envelope refused on four corpus
#:     seam-instances. The composer selects the same two ordinary pairs on it that
#:     it selects in 2-D.
#:   * ``conductive_2d`` had REGRESSED between the two runs: the 2026-09-02 product
#:     wave wired ``conductive_fused_electric_pair``, whose label cannot be
#:     admitted, and clause (8) then rejected the whole plan rather than one seam.
#:     ``plan_step``'s ``fuse_labels`` offer is what restores it, and this case is
#:     what measures that it is restored.
#:   * ``pml_3d_diagonal`` stays the ENVELOPE control and is deliberately NOT
#:     released. After ``dimensions`` widened to admit 1-D it is the only reachable
#:     shape left on which a RELEASED arm is selected and the release declines it,
#:     so releasing 3-D would have cost the one device measurement that the
#:     envelope can say no at all. One corpus seam-instance, weighed against that,
#:     and named here so the trade is legible rather than implied.
#: WHY IT MOVED AGAIN, 2026-09-04, and this time nothing about the release
#: changed — the FILES the release is cited from did. Three of them moved in one
#: round: ``driver.py`` and this file gained the by-name magnetic-half-step
#: consult (:data:`SYNC_UPDATE_H_PASS`), and ``triton_kernels/launch.py`` gained
#: the H_to_D seam row and the neighbouring-seam arbitration rule. All four legs
#: of ``_foldedrelease`` ran the pre-edit bytes, so the citation named a run that
#: had not executed the code citing it — the same defect this line has been moved
#: for twice before, a third file over. This name is the same four-leg protocol
#: repeated against the bytes that ship, with every source edit of the round
#: landed BEFORE the campaign started; ``fingerprints.json``'s
#: ``driver_dispatch.source_sha256`` is re-cut from that recording by
#: ``parity/meep_gpu/recut_driver_dispatch_record.py``, which refuses unless leg,
#: leg and tree all agree. ``_foldedrelease`` is NOT deleted: it stays as the
#: record of what the pre-channel driver did, and the two runs' verdicts are
#: compared case by case rather than assumed equal.
#: RE-RUN 2026-09-07 as `dispatch_fused_route_2026-09-07_cylfinal`, and repointed
#: here, for the reason the paragraph above gives: the cylindrical H_to_D wiring
#: round routed both CuPy-side products through their composers, which edited this
#: very file (fastpath.py a5934d2bba7a -> b74a4c1ac90a) and triton_kernels/launch.py.
#: The 09-04 record bound the bytes its own flush leg executed, so leaving the
#: citation here would have the record binding one fastpath.py and the CITED gate
#: having run another -- which is exactly what test_the_cited_driver_route_gate_ran_
#: the_bytes_the_record_binds refuses. All four legs of the new run released
#: (shipped, harness_keep, flush, shipped_expansion_probe) and its substitution
#: counts reproduce the 09-04 run's case by case. `_syncchannel` is NOT deleted: it
#: stays as the record of what the pre-routing driver did, and neither is
#: `_cylwire`, the first run of this round -- it measured the bytes as they were
#: BEFORE this constant was repointed, which is the ordering trap the paragraph
#: above names: every source edit of a round must land BEFORE its campaign starts,
#: and editing this line is itself such an edit.
#:
#: WHY IT MOVED, 2026-09-11, and the first thing to say is what this line was NOT
#: naming. ``fingerprints.json``'s ``driver_dispatch`` record binds
#: ``dispatch_fused_route_2026-09-08_wired`` — the run cut after the H_to_D wiring
#: round — while this constant still read ``_cylfinal``, so two record welds have
#: been red since 09-08 (``test_the_recorded_dispatch_shape_matches_the_wiring_it_
#: describes``, which compares the two names, and ``test_the_cited_driver_route_
#: gate_ran_the_bytes_the_record_binds``, whose cited directory holds a different
#: ``fastpath.py`` digest from the one the record binds). The ledger's authority
#: has been the 09-08 run throughout; this line was the stale half.
#:
#: WHAT ``_realarms`` ADDS, and it is a release rather than a repeat. Four more
#: fused arms are driven through the seam on THREE cases: ``conductive_2d`` and
#: ``folded_dispersive_2d``, which have dispatched in every campaign since 09-02
#: and grow from one pair to two now that the D-seam product beside them is
#: admitted, and ``cylindrical_m0`` — a NEW route-gate case, the end-to-end gate's
#: Dcyl m = 0 shape with a flux monitor — which is the first cylindrical
#: configuration any release gate has driven and the reason ``cylindrical`` leaves
#: the shared envelope for :data:`FUSED_RELEASE_ARM_AXES` below.
#:
#: THE ORDERING RULE IS UNCHANGED AND IS WHAT THIS PARAGRAPH IS FOR. Every source
#: edit of this round — this name, the four :data:`RELEASED_FUSED_ARMS` rows, the
#: envelope move, the three :data:`ARM_CERTIFICATION` rows, the three
#: :data:`PENDING_DEVICE_GATE_ARMS` deletions, the route gate's new case and its
#: DRIVE rows — lands BEFORE the campaign starts, so all four legs record this
#: file at the digest it ships at, and ``fingerprints.json``'s
#: ``driver_dispatch.source_sha256`` is re-cut FROM that recording by
#: ``parity/meep_gpu/recut_driver_dispatch_record.py``, which refuses unless leg,
#: leg and tree agree. Between the edit and the recut the two welds above stay
#: red, joined by the three ARM_CERTIFICATION lookups whose ledger entries do not
#: exist yet — that window is named in the campaign log rather than papered over,
#: because the alternative is writing a ledger entry by hand, which is the record
#: forgery this campaign forbids. ``_cylfinal`` and ``_wired`` are NOT deleted:
#: they stay as the record of what the pre-release route did, and their verdicts
#: are compared case by case against this run rather than assumed equal.
#: REPOINTED 2026-09-15 to a FRESH stamp. The ``2026-09-14_weldgrid`` run drove only
#: 9 of the 36 DRIVE cases (its driver passed a hardcoded ``--cases`` subset), so its
#: recut refused all 44 release rows whose case it never ran -- and not one of the 44
#: was a composer finding. That directory is kept rather than overwritten: it is still
#: the on-device record that ``no_pml_dispersive_2d`` passes fused on three legs.
#: REPOINTED AGAIN to ``_2026-09-15_offdiag``. The ``_2026-09-15_weldgrid`` run drove
#: all 36 cases and released three legs; ``shipped_expansion_probe`` refused on
#: ``folded_complex_offdiag_2d`` and ``folded_complex_nobloch_offdiag_2d``, which a
#: premature widening of ``fused pair B (folded complex)`` had marked ``dispatch``
#: while the ``update_E`` arm they need is pending. The widening is reverted, the two
#: cases are ``no-fused-arm`` controls, and that run is kept as the measurement.
#: REPOINTED 2026-09-15 to ``_2026-09-15_gapclose``, before that campaign starts, for
#: ONE batch carrying two changes to this table. (1) It releases ``fused pair D
#: (complex conductive no-PML)`` on ``complex_no_pml_3d`` (:data:`RELEASED_FUSED_ARMS`,
#: :data:`FUSED_RELEASE_ARM_AXES`, :data:`ARM_CERTIFICATION`, the
#: :data:`PENDING_DEVICE_GATE_ARMS` deletion) and restores that case's Triton DRIVE
#: row. (2) With the 2026-09-15 ruling that lifted ``complex folded off-diagonal`` out
#: of :data:`PENDING_DEVICE_GATE_ARMS`, it widens ``fused pair B (folded complex)``
#: back onto the two off-diagonal folds, which are DRIVE rows again. Every source edit
#: of the batch lands before the new campaign starts, so every leg records this file
#: at the digest it ships at, and the ``triton_complex_offdiag_device_gate`` re-gate
#: runs on those bytes too. ``_2026-09-15_offdiag`` is NOT deleted: all four of its
#: legs released, and it stays the record of the table before both changes, including
#: both off-diagonal folds as refusal controls (PASS-NO-FUSED-ARM, refusing_rung
#: ``pending-device-gate``) on the bytes before the lift.
#:
#: REPOINTED 2026-09-17 to ``_2026-09-17_allpaths``, again before the campaign runs,
#: for the batch that routes the two SCRATCH-OUTPUT OFF-DIAGONAL WELDS. That batch
#: adds ``fused pair D (off-diagonal)`` and ``fused pair D (folded off-diagonal)`` to
#: :data:`ARM_CERTIFICATION` and to the certified-product table in
#: ``triton_kernels/launch.py``, and gives :class:`ScratchWeldPairPlan` a ``warm()``
#: that launches once at grid ``(0,)`` without rotating. ``_2026-09-15_gapclose`` is
#: NOT deleted: it stays the record of the table with both welds present, released and
#: DELIBERATELY UNWIRED, which is the control this batch overturns. The stencil gate
#: is re-run on the bytes this edit ships before the route campaign starts, so the
#: ledger entries these two labels name are cut from the same tree the route records.
#:
#: REPOINTED 2026-09-27 to ``_2026-09-27_flip``, before that campaign starts, for the
#: edit that turns :data:`DISPATCH_BY_DEFAULT` on and makes the enable refuse every
#: value but ``0`` and ``1`` by name. Neither changes which arm a plan admits, but
#: both change this file's bytes, and the record's recut refuses a route run whose
#: legs executed other bytes than the tree ships. ``_2026-09-23_sparse`` is NOT
#: deleted: it stays the record of the route on the opt-in tree.
#:
#: REPOINTED 2026-09-30 to ``_2026-09-30_091`` for release 0.9.1, whose certification
#: round re-runs every table's route on the released files; ``_2026-09-27_flip`` stays
#: the record of the 0.9.0 route.
DRIVER_ROUTE_FUSED_GATE = "dispatch_fused_route_2026-09-30_091"

#: THE FUSED ARMS THAT HAVE BEEN DRIVEN THROUGH THE DRIVER SEAM, and the gate
#: cases each one was driven on. This is the per-arm allow-list clause (8) reads,
#: and it is the whole of what the fused route ships enabled.
#:
#: WHAT "DRIVEN THROUGH THE SEAM" MEANT FOR EACH ROW, measured, not argued: the
#: arm's kernel LAUNCHED from :meth:`FastPathPlan.dispatch` at the leading consult
#: inside ``driver.step()``, on a driver lifted by ``lift_simulation`` from a real
#: ``mp.Simulation``; the absorbed consult answered True off the sentinel or the
#: repair; the whole run was byte-compared as uint32 against the same driver with
#: ``MEEP_GPU_FUSED=0`` at ten checkpoints and at the end, plus flux spectra, with
#: ``first_divergent_checkpoint`` null everywhere; and device launches per complete
#: step DROPPED by exactly one per pair against a third leg that dispatched the
#: same slot set with fusion off, counted twice over on every case.
#: ``synchronize_magnetic_fields`` — the second consult
#: site, which any flux or energy monitor reaches — was exercised in the same runs.
#: The per-case drops are in the artifact (``cases[*].substitution``) rather than
#: transcribed here, because they moved when the case list grew.
#:
#: BOTH INSTALLED PAIR SHAPES ARE NOW DRIVEN, which ``magnetic_seam_2d`` is here
#: for and is the one row of this table the ``_veto`` run could not have written.
#: A fused pair's absorbed slot holds either a ``launch.NoopPlan`` sentinel or
#: ``deposit_repair.TrailingRepairPlan``, and which one depends on whether the
#: driver injects a source inside that seam. Every case before this one carried an
#: ELECTRIC source, so the repair only ever landed on the D seam: measured on the
#: ``_veto`` campaign, the withheld control's ``leading_repair_slots_left_running``
#: reads ``{"step_D": 12}`` on all three legs and ``step_B`` never appears, while
#: the shipped composition runs the B-seam repair on 11 corpus seam-instances
#: (``parity/meep_gpu/dispatch_reachability.served_in_dispatch``: 38 repair pairs,
#: 27 on D and 11 on B). ``magnetic_seam_2d`` is ``pml_2d`` with an ``mp.Hz``
#: source instead of ``mp.Ez``, which puts the deposit on the other seam and
#: nothing else: measured, the same control reads ``{"step_B": 12}`` there, both
#: pairs dispatch, 2400 complete steps over a ten-rung ladder are byte-identical
#: to the array path across 32 arrays and 768,000 words at every rung, and the
#: launch drop is 4.0 -> 2.0 per step on both counters.
#:
#: AND THE ROUTE MEASURED IS THE ONE THAT SHIPS. All seven dispatching cases ran with
#: ``MEEP_GPU_FUSE_ARMS`` UNSET, so what admitted each arm was THIS TABLE and not a
#: switch — asserted per case rather than inferred from the environment, by a leg
#: that reads the record back and requires ``fusion.value`` absent, ``opted_in``
#: empty, and ``released_here`` covering every arm the composer drove. The
#: substitution baseline is reached the only way that is now possible, through
#: :data:`FUSE_ARMS_VETO`, and the gate asserts the baseline's OWN record says
#: ``vetoed: true`` with nothing driven — because on the first attempt at this run
#: the baseline fused too and the drop was 0.0 per step on every case.
#:
#: THE ARM LABEL IS NOT THE WHOLE PREDICATE, which is why
#: :data:`FUSED_RELEASE_ENVELOPE` and :data:`FUSED_RELEASE_ARM_AXES` exist beside
#: it. ``fused pair B`` is the ``("PML", "ordinary")`` pair
#: (``triton_kernels/launch.py`` FUSED_PAIR_ARMS), and that arm pair is selected on
#: a 3-D grid too — a shape this gate drives only as the envelope's refusal
#: control. Releasing on the label alone would turn dispatch on for a configuration
#: no gate certified, which is the one outcome this file exists to prevent.
#: THE 2026-09-02 ROUND ADDS THE TWO FOLDED ARMS AND ONE 1-D CASE, and both are
#: consequences of the fill consults rather than of a new predicate. Until those
#: landed, a folded run that filled ``fill_B``/``fill_D`` died at the completeness
#: rung before any fused label could be judged, so ``fused pair B (folded)`` had
#: been SELECTED by three campaigns and driven by none. Measured on the GPU host GPU 4
#: before this table was written: ``folded_2d`` now composes 6 of 7 slots with both
#: folded pairs driven and both fill slots carrying the mirror-fill kernel, and
#: ``folded_dispersive_2d`` composes the folded magnetic pair beside a live
#: ``update_P``. ``pml_1d`` is the 1-D shape the previous envelope refused: the
#: composer selects the same two ordinary pairs on it that it selects in 2-D.
#:
#: THE 2026-09-11 ROUND ADDS FOUR ARMS AND ONE CASE, and what is new about it is
#: that TWO of the three cases were already dispatching. On ``conductive_2d`` and
#: ``folded_dispersive_2d`` the D-seam product was SELECTED by the composer in
#: every campaign since 09-02 and withheld by ``plan_step``'s ``fuse_labels``
#: offer alone — the artifacts record it under ``fusion.not_installed`` as
#: "offered to run" — so those two cases go from ONE pair to TWO with no change to
#: what they lift: ``conductive_2d`` now drives ``fused pair D (conductive)``
#: beside ``fused pair B``, and ``folded_dispersive_2d`` drives ``fused pair D
#: (folded dispersive)`` beside ``fused pair B (folded)``. Neither case is a mixed
#: step any more, and their substitution counts move from 4 -> 3 launches per step
#: to 4 -> 2.
#:
#: ``cylindrical_m0`` IS THE ONE NEW CASE, and it is new evidence rather than a
#: wider read of old evidence: no release gate had ever driven a Dcyl grid, which
#: is exactly why ``cylindrical`` sat in the shared envelope at False. It is the
#: end-to-end gate's ``case_cylindrical`` — cell (4, 0, 4), PML 0.8, an ``mp.Ez``
#: Gaussian at r = 0.6, ``dimensions=mp.CYLINDRICAL``, ``m=0`` — with one flux
#: monitor added, so the observable comparison and the second consult site
#: (``synchronize_magnetic_fields``) are non-vacuous. Measured on a NumPy lift
#: before this table was written: the composer admits exactly the two
#: ``cylindrical_real`` products there and nothing else fused, the D seam carrying
#: the deposit repair for the Ez source and the B seam the ``NoopPlan`` sentinel.
#:
#: THE TRITON TABLE'S, AND ONLY ITS. The Metal table's release lives in
#: ``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` beside its own envelope, per-arm
#: axes, certifications and pending map, and the SAME helpers decide both:
#: :func:`_run_shape`, :func:`_axis_reasons` and :func:`fused_release_arm_reasons`
#: are imported there rather than re-implemented. The split is a cost decision and
#: is recorded as one — this file is bound by the Triton ``driver_dispatch`` record
#: and by 42 gate-bound Metal artifacts, so a Metal release row HERE would re-cut
#: the Triton record every time the Metal envelope widened, for a change Triton has
#: no stake in.
RELEASED_FUSED_ARMS: Mapping[str, Tuple[str, ...]] = {
    # --- THE TARGET ROUND, 2026-09-13. EVERY CASE THIS ROUND ADDS WAS LIFTED
    # OFF-DEVICE AND NONE OF THEM HAS BEEN DRIVEN YET, and stating that once here is
    # the honest form of what the rows below do. Each new name was constructed on
    # stock MEEP 1.33.0, handed to ``lift_simulation(prefer_gpu=False)``, and its run
    # shape read with :func:`_run_shape`; the axis value each one licenses in
    # :data:`FUSED_RELEASE_ARM_AXES` is that READ value. What is still owed is the
    # byte comparison: the route campaign that succeeds
    # ``dispatch_fused_route_2026-09-13_phaseB`` drives every one of them through
    # ``run_case``, and :data:`DRIVER_ROUTE_FUSED_GATE` still names phase B, so a
    # reader comparing this table against the gate it cites can see that these
    # cases' evidence is OWED rather than banked. That is the opposite of the two
    # blocks above, where the case ran first and the row was typed after.
    #
    # A NAME HERE IS PROVENANCE, NOT A GATE INPUT. The release decides on the AXES
    # (:func:`fused_release_arm_reasons`); this table says which case drove them and
    # :func:`released_fused_arms` only requires the list to be non-empty. So a name
    # that never runs is a FALSE RECORD rather than a loosened predicate, which is
    # why nothing optional is listed: ``folded_complex_kz2d_3d`` is a second 3-D
    # corner the Metal roster requires and the Triton one does not, so it is
    # deliberately absent here rather than listed on the chance it rides.
    #
    # ``pml_3d``, ``offdiag_2d`` AND ``offdiag_magnetic_2d`` ARE CO-REQUISITE WITH A
    # COMPOSER EDIT, not a row on their own: they evidence ``fused pair B``'s widened
    # ``off_diagonal_epsilon``, and the Triton composer refuses that pair on an
    # off-diagonal grid until ``triton_kernels/launch.py``'s
    # ``specialized_family_owns_the_grid`` is narrowed per-pair in this same batch.
    # If that narrowing does not land, this row admits a composition the composer
    # will not build and the three names are evidence of nothing; see the arm's own
    # ``off_diagonal_epsilon`` reason below, which says the same thing where a
    # refusal quotes it.
    "fused pair B": ("pml_2d", "magnetic_seam_2d", "conductive_2d", "dispersive_2d",
                     "pml_1d", "pml_3d_diagonal", "offdiag_2d",
                     "offdiag_magnetic_2d", "pml_3d"),
    "fused pair D": ("pml_2d", "magnetic_seam_2d", "pml_1d", "pml_3d_diagonal"),
    "dispersive fused pair": ("dispersive_2d",),
    # NARROWED 2026-09-11 AND RESTORED ON THE RE-ATTRIBUTION ROUND, and both moves
    # are about the CASE rather than the arm. `folded_dispersive_2d` was this arm's
    # second case until 2026-09-11, when the route gate's second consult site was
    # read as diverging from the array path on that shape with fusion off as well as
    # on; a case that diverges in its own control cannot be release evidence for any
    # arm, so the case was withdrawn and this arm's ``susceptibilities`` row was
    # pinned at 0 to match what its remaining cases carried.
    #
    # THAT WITHDRAWAL IS RETIRED. The divergence was measured, on the real route
    # gate over three fresh processes, to be an artefact of the LAZY subnormal-policy
    # install rather than anything the seam does -- the array comparison leg was
    # flushing subnormals that both dispatch legs kept -- and the seam itself read
    # clean (:data:`PENDING_DEVICE_GATE_ARMS` carries the measurements). So the case
    # is live evidence again, it returns here, and it is what licenses the
    # ``susceptibilities`` row below widening from 0 to {0, 1}: one pole from this
    # case, zero from ``folded_2d``.
    #
    # AND ITS ONE POLE IS BANKED ON THE RELEASE ROUTE rather than owed, which is
    # what makes the widening a restoration rather than a new claim. Read out of
    # ``results/dispatch_fused_route_2026-09-08_wired/shipped/cases.jsonl``: on
    # ``folded_dispersive_2d`` the record is PASS-FUSED over 1599 steps with
    # ``fusion.opted_in`` EMPTY and ``fusion.driven`` ``{step_B: fused pair B
    # (folded), update_H: fused pair B (folded)}`` — the switch unset, this table
    # doing the admitting — and that run's ``second_consult_site.verdict`` is PASS,
    # on the very shape and site the 2026-09-11 hold was written about. Every
    # campaign in that window that DROVE the case carries it at the same verdict,
    # which is the honest form of the claim and not a universal: the 2026-09-02
    # ``_sparserescale`` run drove a different case list and never lifted this
    # shape at all, and a ``flush`` leg records it PASS-POLICY-REFUSED by name
    # rather than PASS-FUSED, because that policy refuses before any arithmetic.
    # ``folded_offdiag_magnetic_2d`` is the off-diagonal fold and
    # ``folded_3d`` the 3-D fold; both were lifted 2026-09-13, neither has run, and
    # the route campaign that succeeds ``dispatch_fused_route_2026-09-13_phaseB`` is
    # what drives them.
    "fused pair B (folded)": ("folded_2d", "folded_dispersive_2d",
                              "folded_offdiag_magnetic_2d", "folded_3d"),
    "fused pair D (folded)": ("folded_2d", "folded_3d"),
    "fused pair D (conductive)": ("conductive_2d",),
    # THE RE-ATTRIBUTION ROUND'S ONE RELEASE, and the row states honestly which half
    # of its evidence is banked and which the campaign owes. What is BANKED: the
    # family weld (``triton_folded_dispersive_fused_pair_device_gate``, PASS) and
    # three fresh-process drives of this gate's own ``run_case`` on
    # ``folded_dispersive_2d`` with the arm admitted in-process -- PASS-FUSED on all
    # three, ``first_divergent_checkpoint`` null, 0 of 41 arrays differing over
    # 200,080 words on both the fused and the unfused comparison, the second consult
    # site PASS, substitution 7.00 against 9.00 launches per step, and the armed
    # nulls firing. What is OWED: every one of those drives reached the arm by
    # OPT-IN (``MEEP_GPU_FUSE_ARMS``), because no release row existed to admit it, so
    # the release-route control read NOT-APPLICABLE on all three. This row is what
    # makes the shipped route reachable, and the route campaign that succeeds
    # ``dispatch_fused_route_2026-09-13_phaseB`` is the run that drives it with the
    # switch UNSET. Until that lands the arm's release evidence is a switch-reached
    # measurement, which is weaker than every other row in this table and is said so
    # here rather than left for a reader to infer from the gate name.
    "fused pair D (folded dispersive)": ("folded_dispersive_2d",),
    "fused pair B (cylindrical)": ("cylindrical_m0",),
    "fused pair D (cylindrical)": ("cylindrical_m0",),
    # THE SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-15, each on the route case that
    # already carries its shape. Both cases have run on the shipped leg with the D
    # seam UNFUSED -- ``dispatch_fused_route_2026-09-15_offdiag`` reads
    # ``offdiag_magnetic_2d`` PASS-FUSED EXACT at 4.0 -> 3.0 launches per step over
    # 2399 steps and ``folded_offdiag_magnetic_2d`` at 6.0 -> 5.0 over 1999 -- so the B
    # seam, the fills and the run shapes are banked, and what these rows add is the D
    # seam on them. What is OWED is the drive with the weld installed (expected 4.0
    # -> 2.0 and 6.0 -> 4.0), taken by the route campaign at
    # :data:`DRIVER_ROUTE_FUSED_GATE` after the stencil gate re-runs on these bytes
    # and its ledger entries are seeded. The product evidence is that gate's: on
    # ``results/triton_offdiag_stencil_welds_2026-09-13_batch`` both families read S1
    # 60/60 IDENTICAL against the array path and the certified singles, S2 IDENTICAL
    # at BLOCK 64/128/256/512/1024, keep policy, and every mutation as declared (7
    # plain, 9 folded, one predicted null each).
    "fused pair D (off-diagonal)": ("offdiag_magnetic_2d",),
    "fused pair D (folded off-diagonal)": ("folded_offdiag_magnetic_2d",),
    # --- PHASE B, 2026-09-13. Every row below was driven through this gate's own
    # ``run_case`` on the GPU host (2026-09-13, one RTX A6000, 96-step ladders, every
    # case PASS-FUSED, substitution EXACT, second consult site PASS) with the rows
    # patched in-process before they were written down; the welds each cites were
    # seeded from the 2026-09-12 identity fleet (``seed_triton_welds.py``). ---
    "fused pair B (real beta)": ("special_kz_2d",),
    "fused pair D (real beta)": ("special_kz_2d",),
    "fused pair B (folded real beta)": ("folded_special_kz_2d",),
    "fused pair D (folded real beta)": ("folded_special_kz_2d",),
    # THE TARGET ROUND'S ``bloch`` CASES. ``complex_beta_2d`` and
    # ``folded_complex_beta_2d`` drove a z-only k_point, which lifts as a beta and
    # reads ``bloch False``; ``complex_beta_bloch_2d`` and
    # ``folded_complex_beta_bloch_2d`` carry the SAME beta 0.4 beside an in-plane
    # k_x = 0.25 and lift ``bloch True``, so the pair of cases is what drives both
    # values of the one axis these four rows open.
    "fused pair B (complex beta)": ("complex_beta_2d", "complex_beta_bloch_2d"),
    "fused pair D (complex beta)": ("complex_beta_2d", "complex_beta_bloch_2d"),
    "fused pair B (folded complex beta)": ("folded_complex_beta_2d",
                                           "folded_complex_beta_bloch_2d"),
    "fused pair D (folded complex beta)": ("folded_complex_beta_2d",
                                           "folded_complex_beta_bloch_2d"),
    "fused pair B (BFAST)": ("bfast_1d",),
    "fused pair D (BFAST)": ("bfast_1d",),
    "fused pair B (nonlinear)": ("nonlinear_1d",),
    # THE TARGET ROUND'S ``dimensions`` CASES on the complex pairs, and there are
    # THREE because the corpus rows the widening serves are all (1, 1, N) cells:
    # ``complex_1d`` is the declared 1-D cell, ``complex_3d_thinline`` the declared
    # 3-D one-cell-wide line, and ``complex_3d`` a genuine (40, 40, 40) complex
    # block that no board instance needs. The third is here so the widened row is
    # not evidenced by degenerate grids alone -- the corpus DOES carry real 3-D
    # complex rows, and the row would admit them at runtime.
    "fused pair B (complex)": ("bloch_2d", "complex_nobloch_2d", "complex_1d",
                               "complex_3d_thinline", "complex_3d"),
    "fused pair D (complex)": ("bloch_2d", "complex_nobloch_2d", "complex_1d",
                               "complex_3d_thinline", "complex_3d"),
    # THE TARGET ROUND ON THE FOLDED COMPLEX PAIRS. ``folded_complex_3d`` is the
    # 3-D fold (the triangular-lattice-oblique cell) and drives BOTH arms.
    #
    # THE TWO OFF-DIAGONAL FOLDS ARE THE MAGNETIC PAIR'S AGAIN FROM 2026-09-15, and
    # they return with the arm that kept them out. On ``folded_complex_offdiag_2d``
    # (bloch True) and ``folded_complex_nobloch_offdiag_2d`` (bloch False) the
    # composer selects this pair for step_B/update_H and the single arm ``complex
    # folded off-diagonal`` for update_E. While that arm sat in
    # :data:`PENDING_DEVICE_GATE_ARMS` the ladder refused the WHOLE plan:
    # ``dispatch_fused_route_2026-09-15_weldgrid`` read DID-NOT-FUSE / NO-DROP with
    # every slot on the array path, and ``dispatch_fused_route_2026-09-15_offdiag``
    # drove both as ``no-fused-arm`` controls and read PASS-NO-FUSED-ARM with the
    # refusal classified ``pending-device-gate``. The 2026-09-15 ruling lifted the
    # arm (the record is beside that map) on condition that its weld is re-gated and
    # rebound in the same batch, and both cases are DRIVE rows again. The electric
    # twin ``fused pair D (folded complex)`` is refused by name on the row product
    # of an off-diagonal grid, so its row does not move. The two off-diagonal
    # corners are NOT credited here until the route campaign at
    # :data:`DRIVER_ROUTE_FUSED_GATE` reads PASS-FUSED on both.
    "fused pair B (folded complex)": ("folded_complex_2d", "folded_complex_3d",
                                      "folded_complex_offdiag_2d",
                                      "folded_complex_nobloch_offdiag_2d"),
    "fused pair D (folded complex)": ("folded_complex_2d", "folded_complex_3d"),
    "fused pair B (cylindrical complex)": ("cylindrical_m1", "cylindrical_m0_complex"),
    "fused pair D (cylindrical complex)": ("cylindrical_m1", "cylindrical_m0_complex"),
    # --- THE TARGET ROUND'S ONE NEW ARM. ``fused pair D (no-PML stored E)`` was
    # WIRED and REFUSED until now: the composer could select it, clause (8) refused
    # it because it was in no released set, and rung (7a) refused it because
    # :data:`PENDING_DEVICE_GATE_ARMS` said no ledger entry had been cut. Both of
    # those are answered in this edit -- ``triton_no_pml_fused_electric_pair_device_gate``
    # is in ``triton_kernels/fingerprints.json`` with status PASS, recorded
    # 2026-09-13T07:53:43Z on the GPU host (RTX A6000, cc 8.6, CuPy 13.5.1, Triton
    # 3.1.0), seeded by ``seed_triton_welds.py`` from
    # ``results/triton_fleet_2026-09-13_batch_C/no_pml_fused_electric_pair/`` -- so
    # the pending row is deleted and the :data:`ARM_CERTIFICATION` row is added
    # beside this one.
    #
    # NEITHER OF ITS TWO CASES IS THE EXISTING ``no_pml_2d``, and that is the point:
    # that case carries no poles and no conductivity, so it evidences neither half
    # of the arm's row. ``absorber_1d`` is a 1-D ``mp.Absorber`` cell over a
    # five-pole medium (conductivity True, pml_active False) and
    # ``no_pml_dispersive_2d`` a (120, 120, 1) two-pole cell with no boundary layers
    # at all (conductivity False, pml_active False); between them they drive
    # ``dimensions`` 1 and 2 and both values of ``conductivity``, which is what
    # lets that axis stay unpinned honestly.
    #
    # THE SECOND CASE WAS ``material_dispersion_0d`` UNTIL 2026-09-14, a zero-extent
    # (1, 1, 1) cell carrying the same two axis values. It was replaced rather than
    # narrowed away: on a one-cell grid every non-vacuity control the driver-route
    # gate owns is vacuous -- decisively the in-seam mutation, whose wall plane IS
    # the single cell and also the source's deposit cell that ``deposit_repair.apply``
    # recomputes (cells MUTATED BUT NOT REPAIRED: ZERO of 1, against 399 of 400 on
    # ``absorber_1d``) -- so the 2026-09-14 Triton shipped leg refused on that
    # control alone with zero failing cases. The replacement drives the SAME axis
    # values on a cell with real extent, so this row is unchanged by the swap.
    "fused pair D (no-PML stored E)": ("absorber_1d", "no_pml_dispersive_2d"),
    # --- 2026-09-15: THE COMPLEX NO-ABSORBER D PAIR, ON ONE CASE, AND THE DRIVE IS
    # OWED. ``complex_no_pml_3d`` has not run on the Triton route: its Triton DRIVE
    # row was withdrawn on 2026-09-14 and is restored in this batch, and the campaign
    # :data:`DRIVER_ROUTE_FUSED_GATE` names is the run that drives it with the switch
    # unset. What is BANKED, read 2026-09-15 rather than argued:
    #
    #   * the pair gate released on every keep run it has had, eleven Triton
    #     artifacts from ``triton_fuse_2026-08-30_cells`` to
    #     ``triton_fleet_2026-09-14_target_A``, though none could be seeded (see
    #     :data:`ARM_CERTIFICATION`); the re-run after this batch cuts the entry;
    #   * the case's arm set on THIS table, measured on a device leg rather than a
    #     lift: the unfused comparison of
    #     ``dispatch_fused_route_cuda_2026-09-15_weldgrid/cuda_shipped_expansion_probe``
    #     dispatched the Triton table alone (``opted_in`` empty) and selected
    #     ``complex conductive no-PML curl`` for step_D and ``complex no-PML stored
    #     E`` for update_E, this label's two constituents, at 6.0 launches a step on
    #     run_shape dimensions 3, grid (23, 21, 27), complex_storage True, bloch
    #     True, beta 0.0, conductivity True, off_diagonal_epsilon False,
    #     susceptibilities 1, pml_active False;
    #   * the rows it serves: four ``TestLoadDump`` 3-D rows (``chunk_layout_file_3d``,
    #     ``chunk_layout_sim_3d``, ``structure_3d``, ``structure_sharded_3d``), whose
    #     D->E seam-instances the Triton board already counts as served by
    #     ``complex_conductive_fused_pair`` and holds out of dispatch on this label's
    #     release alone.
    "fused pair D (complex conductive no-PML)": ("complex_no_pml_3d",),
}

#: THE CONFIGURATION AXES THE DRIVER-ROUTE GATE MEASURED, each pinned to the value
#: it measured, keyed by the name :func:`_run_shape` reports it under — so the
#: shape the artifact publishes and the shape the release decision was made on are
#: THE SAME READS rather than two spellings that can drift apart. That is the rule
#: :func:`_run_shape` states for the coverage predicates; this table extends it to
#: the release.
#:
#: EVERY VALUE HERE WAS READ OFF A LIFTED CASE, not transcribed from the gate's
#: prose: all eight of the 2026-09-11 dispatching cases reported
#: ``complex_storage false``, ``bloch false``, ``beta 0.0``, ``bfast false``,
#: ``nonlinearity false``, ``off_diagonal_epsilon false`` and ``pml_active true``,
#: which is what made those seven rows shared. The axes they DISAGREED on —
#: ``dimensions`` (1 on pml_1d, 2 on the rest), ``folded`` (present on two cases,
#: absent on six), ``cylindrical`` (true on cylindrical_m0 alone), ``conductivity``
#: (true on conductive_2d) and ``susceptibilities`` (1 on dispersive_2d and
#: folded_dispersive_2d, 0 elsewhere) — are per-arm rows in
#: :data:`FUSED_RELEASE_ARM_AXES`, because which VALUE was driven depends on which
#: arm was driven on it.
#:
#: AND THE TABLE IS NOW EMPTY, which is where that rule ends rather than a gap in
#: it. Six of the seven rows left on 2026-09-13 (phase B); the target round moved
#: the seventh, ``pml_active``, for the same reason and the shared half of the
#: release is now the empty conjunction. An empty envelope refuses nothing and
#: asserts nothing: :func:`fused_release_reasons` still fails closed on an
#: unreadable shape, and every axis any arm was driven on is stated on that arm's
#: own row, where it can carry the arm's value rather than the intersection of
#: twenty-seven of them. The structure is kept rather than deleted because the
#: decision is still made in two halves — ``_run_shape`` reads once,
#: :func:`_axis_reasons` decides twice — and because the next axis every released
#: arm genuinely shares belongs here rather than copied twenty-seven times.
#:
#: WHY ``cylindrical`` IS A ROW OF ITS OWN AND NOT IMPLIED BY ``dimensions``, here
#: until 2026-09-11 and per-arm since. Measured: a Dcyl m=0 lift reports
#: ``dimensions 2`` with ``grid_shape`` ``[80, 1, 80]``. A release keyed on
#: dimensionality alone would have admitted it.
#:
#: WHAT IS NOT PINNED, AND WHY. The grid EXTENT is not: extent is a launch-grid
#: argument rather than a codegen input, and the underlying sub-step families were
#: byte-gated across the corpus extents. The STEP BUDGET is not: each case ran its
#: own budget over a ten-rung checkpoint ladder with no divergence at any rung.
#: Both are stated in the record (``fusion.released_on_cases``,
#: ``run_shape.grid_shape``) so a reader can compare what they ran against what was
#: measured; neither is a refusal.
#:
#: ``None`` as the required value means the axis must be ABSENT from the shape —
#: which is how :func:`_run_shape` spells "not folded", since it drops None values.
#: A ``frozenset`` means "any of these", and it is used where the gate drove more
#: than one value of an axis; every other required value is an equality.
#:
#: WHAT LEFT THIS TABLE ON 2026-09-02 AND WHY. ``folded`` is now a PER-ARM axis
#: (:data:`FUSED_RELEASE_ARM_AXES`): the two folded pairs REQUIRE a fold and the
#: three unfolded arms require its absence, and one shared row could only have said
#: one of those. Keeping it here at ``None`` would have refused the folded arms on
#: the only shape they run; widening it here would have admitted the ORDINARY pairs
#: on a folded grid, which no gate has ever driven and which is this file's named
#: failure mode. The axes below are the ones every released arm was driven on with
#: the SAME value, which is the only thing a shared table can honestly say.
#:
#: WHAT LEFT ON 2026-09-11, on that same rule. ``cylindrical`` is now a PER-ARM
#: axis: the two Dcyl arms REQUIRE a cylindrical grid and the five Cartesian arms
#: refuse one, and a single shared row cannot say both. It stayed here for as long
#: as it did because no case drove it either way; ``cylindrical_m0`` is what made
#: the row have two answers. The move costs no live row — every arm that was
#: admitted under the shared False keeps a per-arm ``("cylindrical", False)`` row
#: saying the same thing about the same shapes.
#:
#: ``m`` IS DELIBERATELY NOT PINNED ANYWHERE, and the reason is a reader rather
#: than a predicate: the shape ``parity/meep_gpu/dispatch_reachability`` derives
#: from the census carries no ``m`` key at all, so a pinned ``m`` would read "m did
#: not read" and refuse every board row — a refusal about the census reader, not
#: about the run. The ``cylindrical_real`` coverage predicates refuse ``m != 0``
#: themselves, which is where that question belongs.
FUSED_RELEASE_ENVELOPE: Tuple[Tuple[str, Any, str], ...] = (
    # WHAT LEFT ON 2026-09-13 (phase B), on the same rule as the two moves above:
    # ``complex_storage``, ``bloch``, ``beta``, ``bfast``, ``nonlinearity`` and
    # ``off_diagonal_epsilon`` are PER-ARM axes now, because the arms stopped
    # agreeing on them -- the complex, beta, BFAST and nonlinear products REQUIRE
    # the value the eight 2026-09-11 arms refuse. Every previously released arm
    # carries the six rows explicitly, pinned to exactly what the shared table said
    # for it, so the move refuses no live row.
    #
    # AND ``pml_active`` LEFT IN THE TARGET ROUND, which empties the table. The
    # trigger is the same one every earlier move had: ``fused pair D (no-PML stored
    # E)`` is released on ``absorber_1d`` and ``no_pml_dispersive_2d``, two cells
    # with no split-field PML at all, so the shared row would have had to say both
    # True and False. It is now on every arm explicitly -- True on the twenty-five
    # that carried it under the shared row, False on the new one, and True again on
    # the folded dispersive pair the re-attribution round added, twenty-seven rows
    # in all -- and the move was MEASURED to refuse nothing: recounted over the
    # 2026-09-13 Triton
    # board's own 530 served seam-instances with the per-arm rows in place and the
    # shared row gone, ``served_in_dispatch`` reads 236, exactly what it reads with
    # the shared row, and the refused-by-envelope bucket is unchanged at 77.
    #
    # WHAT AN EMPTY SHARED TABLE DOES TO THE RECORD, because a reader of an
    # artifact will notice it before reading this file: ``fusion.
    # outside_the_released_envelope`` is fed by :func:`fused_release_reasons`, so
    # from this round it is EMPTY on every run whose shape read -- it fires only on
    # the unreadable-shape branch, which is a refusal about the reader rather than
    # about the configuration. The refusals a reader wants have not gone anywhere:
    # ``fusion.outside_this_arms_own_cases`` is reported precisely WHEN the shared
    # half is silent, so every axis that refused an arm is still named, per arm,
    # beside a ``released_here`` that is short of the full release.
    #
    # WHAT THE OLD ``off_diagonal_epsilon`` ROW'S CLAUSE SAID, and why it is not
    # repeated per-arm verbatim: it rested on the Triton composer's
    # ``specialized_family_owns_the_grid`` guard refusing every pair on an
    # off-diagonal grid. The target round narrows that guard per-pair -- the B
    # seam's kernel takes no inverse-epsilon pointer, so an off-diagonal chi1inv
    # cannot reach its arithmetic, while the D seam's does and stays refused -- so
    # the blanket sentence is false after this batch and each arm's own row now says
    # which half of the split it is on.
)

#: THE AXES ONE ARM WAS DRIVEN ON, beyond the shared table above. This is the half
#: of the release that used to be PROVENANCE ONLY, and 2026-09-02 turned it into a
#: predicate because the shared table could no longer carry both answers.
#:
#: THE RULE FOR A ROW HERE, applied uniformly: an axis is pinned for an arm when
#: that arm's OWN case list drove exactly one of its values. An axis the arm's cases
#: drove both values of is left out — that is what ``fused pair B``'s missing
#: ``conductivity`` and ``susceptibilities`` rows mean, since ``conductive_2d`` and
#: ``dispersive_2d`` are in its list and ``pml_2d`` is too.
#:
#: WHAT THIS CLOSED, and it is the gap the previous text stated and declined to
#: close: ``fused pair D`` was admitted on conductive and dispersive shapes its own
#: two cases never drove, because the shared table left both axes unpinned on the
#: ground that the gate measured both values ACROSS ITS CASE SET. Measured against
#: the 2026-09-02 board's credited instances, the narrowing costs NOTHING: all 26
#: ``fused pair D`` instances are ``conductivity False`` and ``susceptibilities 0``,
#: so the axes below refuse zero live rows and retire a latent over-admission. The
#: same is true of every ``conductivity`` row here — 0 instances of any of the four
#: other arms sits on a conductive grid.
#:
#: ``susceptibilities`` IS PINNED ON TWO ARMS AND NOT ON THE OTHER THREE, and the
#: split is about WHICH ARM THE COMPOSER SELECTS rather than about what a kernel
#: reads. On a dispersive grid the D/E seam goes to ``dispersive fused pair`` (and,
#: on a fold, to the folded dispersive product), so ``fused pair D`` and ``fused
#: pair D (folded)`` are simply not the arms that run there — measured on the GPU host,
#: not argued — and a release that admitted them would be admitting an arm no
#: composer can select. The magnetic pairs are the mirror of that: ``fused pair B``
#: and ``fused pair B (folded)`` ARE what the composer selects on a dispersive grid,
#: which is what ``dispersive_2d`` and ``folded_dispersive_2d`` drive.
#:
#: ``dispersive fused pair`` IS THE ONE ROW LEFT DELIBERATELY UNPINNED ON THIS AXIS,
#: and it is an admitted asymmetry rather than an oversight. Its own case drove
#: ``susceptibilities 1``, so the rule above would pin it — but the rule's purpose is
#: to stop an arm being admitted where its gate did not run it, and this axis is not
#: expressible as an equality without ALSO refusing the two-pole rows this arm is
#: the composer's selection on. Admitting it on a susceptibility-free shape costs
#: nothing measurable — ``dispersive_fused_pair_coverage`` refuses one — but that is
#: an argument about the composer, not a measurement of this table, and it is
#: recorded here as such.
#: ``cylindrical`` IS THE 2026-09-11 ADDITION, and it is on every row here rather
#: than on the new ones alone. Widening a shared axis is only honest if it moves
#: NO current release's rows, so the five Cartesian arms each carry an explicit
#: ``("cylindrical", False)`` saying exactly what the shared table said for them,
#: and the two Dcyl arms carry ``True``. Measured against the 09-04 board's
#: credited instances the five rows refuse zero: no instance of any Cartesian arm
#: sits on a cylindrical grid, because ``launch.py``'s ``has_cylindrical`` branch
#: refuses the ordinary and folded pairs there by name.
#:
#: ``pml_active`` IS THE TARGET ROUND'S ADDITION, on exactly the same rule and with
#: the same control: it is on every arm rather than on the new one alone, True on
#: the twenty-five that were admitted under the old shared row and False on
#: ``fused pair D (no-PML stored E)`` (the re-attribution round's folded dispersive
#: pair carries True too, making twenty-seven rows), and the move was recounted
#: over the Triton
#: board's own 530 served seam-instances at 236 before and 236 after. See
#: :data:`FUSED_RELEASE_ENVELOPE`, which the move empties.
#:
#: WHAT A TWO-AXIS WIDENING DOES AND DOES NOT SAY, because the target round widens
#: two axes on two arms and this table has no corner structure to spell a
#: conjunction. A row is a per-axis equality or membership test, so widening
#: ``dimensions`` and ``off_diagonal_epsilon`` on one arm admits the CORNER where
#: both are new even though no case drove it — unlike the hand-CUDA and Metal
#: tables, which carry ``*_OFF_DIAGONAL_CORNERS`` for exactly this, and this file
#: has no analogue. The corners that opens are named where they open:
#: ``fused pair B (folded)`` gains (3-D x off-diagonal) with ``folded_3d`` diagonal
#: and ``folded_offdiag_magnetic_2d`` 2-D; ``fused pair B (folded complex)``, whose
#: ``off_diagonal_epsilon`` and ``bloch`` rows widened again on the 2026-09-15
#: ruling, gains (diagonal x unphased), (3-D x off-diagonal) and (3-D x unphased) --
#: its four cases drive (2-D and 3-D x diagonal x phased), (2-D x off-diagonal x
#: phased) and (2-D x off-diagonal x unphased) -- and on the diagonal unphased 2-D
#: fold it is released beside ``fused pair B (folded complex beta)``; and
#: the complex pairs gain (1-D or 3-D x unphased), ``complex_nobloch_2d`` being the
#: only unphased case and 2-D. ``fused pair B`` is the one arm where the corner IS
#: driven: ``pml_3d`` is 3-D AND off-diagonal in the same lift. None of these
#: corners carries a board instance, so none is credited; they are stated because
#: what the row ADMITS at runtime is wider than what any case drove.
FUSED_RELEASE_ARM_AXES: Mapping[str, Tuple[Tuple[str, Any, str], ...]] = {
    "fused pair B": (
        ("dimensions", frozenset({1, 2, 3}),
         "the gate ran pml_1d and four 2-D cases; pml_3d_diagonal lifted "
         "off-device 2026-09-13 to dimensions 3 on a (40, 40, 40) diagonal PML "
         "cell -- extent is not a pinned axis, and the corpus row it stands for, "
         "examples:differential_cross_section.py, is the same shape at 120 cubed "
         "-- and is driven on the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB, where it stops being this "
         "table's envelope control and becomes a DRIVE row"),
        ("folded", None, "no folded case drove the ORDINARY pair; the folded grid "
                         "has its own pair, released separately below"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair, "
         "released separately below"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        # THE ONE ROW IN THIS TABLE THAT DEPENDS ON A COMPOSER EDIT, and it is said
        # in the reason rather than in a comment a refusal cannot quote. The
        # sentence this row used to carry -- "the Triton composer's
        # specialized_family_owns_the_grid guard refuses the pair there" -- was
        # TRUE, and the target round narrows that guard per-pair so it stops being
        # true for the MAGNETIC pair alone: launch.plan_fused_pair passes the
        # inverse-epsilon list only when the pair's constitutive side is "E", pair
        # "B" is "H", and kernels.fused_curl_constitutive_B's signature carries no
        # inverse-epsilon pointer at all, so an off-diagonal chi1inv is structurally
        # absent from this seam's arithmetic. The ELECTRIC pair's kernel does take
        # one and stays refused by name on the same rows -- that pairing, recorded
        # in the route artifact, is what makes the narrowing a measurement rather
        # than an argument.
        ("off_diagonal_epsilon", frozenset({False, True}),
         "offdiag_2d (Ez source, single-axis isolation against pml_2d), "
         "offdiag_magnetic_2d (Hz source, so the in-plane E the off-diagonal rows "
         "couple is live and the ELECTRIC pair genuinely contends for the D seam) "
         "and pml_3d (a 48-cubed sphere, off-diagonal AND 3-D in one lift) all "
         "lifted off-device 2026-09-13 to off_diagonal_epsilon True. NONE HAS BEEN "
         "DRIVEN, and none CAN be until triton_kernels/launch.py's "
         "specialized_family_owns_the_grid is narrowed per-pair in this same batch "
         "-- that guard is what refuses the pair on these grids today, so this row "
         "and that narrowing stand or fall together; the route campaign that "
         "succeeds dispatch_fused_route_2026-09-13_phaseB is what drives all three"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; every case that drove "
         "this arm carries a split-field PML, which is what the shared row said "
         "about it"),
    ),
    "fused pair D": (
        ("dimensions", frozenset({1, 2, 3}),
         "the gate ran pml_1d and pml_2d/magnetic_seam_2d; pml_3d_diagonal lifted "
         "off-device 2026-09-13 to dimensions 3 and is driven on the route "
         "campaign that succeeds dispatch_fused_route_2026-09-13_phaseB"),
        ("folded", None, "no folded case drove the ORDINARY pair"),
        ("conductivity", False,
         "its two 2-D cases and pml_1d are all lossless; conductive_2d drives "
         "fused pair B and leaves the D seam to the conductive-PML arms"),
        ("susceptibilities", 0,
         "no case that drove this arm carried a susceptibility; dispersive_2d's "
         "D seam goes to the dispersive fused pair"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair, "
         "released separately below"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        # NOT the sentence this row carried until the target round. It named
        # specialized_family_owns_the_grid, which the batch narrows per-pair, so a
        # blanket "refuses the pair there" would be false the moment that lands.
        # What is true for the ELECTRIC seam before and after is the arithmetic:
        # kernels.fused_curl_constitutive_D takes an inverse-epsilon pointer and
        # plan_fused_pair fills it (the pair's constitutive side is "E"), so an
        # off-diagonal chi1inv reaches this kernel and no gate has run it there.
        ("off_diagonal_epsilon", False,
         "no off-diagonal case drove it, and this is the seam that reads chi1inv: "
         "the Triton composer refuses the ELECTRIC pair by name on an off-diagonal "
         "grid, before the target round's per-pair narrowing of "
         "specialized_family_owns_the_grid and after it"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; every case that drove "
         "this arm carries a split-field PML, which is what the shared row said "
         "about it"),
    ),
    "dispersive fused pair": (
        ("dimensions", 2, "dispersive_2d is a 2-D Cartesian grid"),
        ("folded", None, "no folded case drove it"),
        ("conductivity", False, "dispersive_2d is lossless"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair, "
         "released separately below"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        ("off_diagonal_epsilon", False,
         "no off-diagonal case drove it, and this is the seam that reads chi1inv: "
         "the Triton composer refuses the dispersive ELECTRIC pair by name on an "
         "off-diagonal grid, before the target round's per-pair narrowing of "
         "specialized_family_owns_the_grid and after it"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; every case that drove "
         "this arm carries a split-field PML, which is what the shared row said "
         "about it"),
    ),
    "fused pair B (folded)": (
        ("dimensions", frozenset({2, 3}),
         "folded_2d is a 2-D grid; folded_3d lifted off-device 2026-09-13 to "
         "dimensions 3 on a (40, 21, 40) mirror-on-Y cell whose geometry is an "
         "axis-aligned BLOCK -- a curved surface there would have answered "
         "off_diagonal_epsilon True and the case would have stopped being about "
         "dimensions. It is driven on the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB"),
        ("folded", "required",
         "this arm IS the folded product; on an unfolded grid the ordinary pair "
         "is what the composer selects"),
        ("conductivity", False, "neither folded case carries a conductivity"),
        # ADDED IN THE TARGET ROUND AS A NARROWING AND WIDENED BY ONE VALUE ON THE
        # RE-ATTRIBUTION ROUND. This arm had no susceptibilities row at all while its
        # case list carried folded_dispersive_2d, the one pole it was driven on; when
        # that case's evidence was withdrawn on 2026-09-11 the arm was left admitting
        # any pole count on the strength of a case that no longer counted, so the row
        # went in at 0 -- measured then as -4 on the Triton board. The withdrawal is
        # retired (the divergence was the lazy subnormal-policy install, not the
        # seam), so the case is back in the arm's list and the row carries the two
        # values its own cases drive.
        #
        # IT IS DELIBERATELY NOT WIDENED TO 5, and the trade is the point rather than
        # an oversight. The four instances a five-pole admission would credit are the
        # 2-D folded TestLoadDump rows, and the route gate's Triton ENVELOPE WITNESS
        # is ``folded_dispersive5_2d`` -- the same fold at five poles -- which
        # witnesses the release DECLINING a configuration only while this row refuses
        # it. Measured on a NumPy lift: with the row at {0, 1} that case's
        # released set is empty and this arm's refusal names the axis. A release
        # that cannot be shown saying no is worth less than four credited rows.
        ("susceptibilities", frozenset({0, 1}),
         "folded_2d carries zero poles and folded_dispersive_2d one Lorentz pole, "
         "which are the two values this arm's own cases drive -- the second banked "
         "on the release route, not owed: dispatch_fused_route_2026-09-08_wired's "
         "shipped leg drove this arm on that case at step_B/update_H with "
         "fusion.opted_in empty, PASS-FUSED over 1599 steps and the second consult "
         "site PASS. Five poles is refused because folded_dispersive5_2d, the same "
         "fold at five, is the route gate's envelope witness and witnesses nothing "
         "if this row admits it"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair, "
         "released separately below"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        # THE CHEAP HALF OF THE OFF-DIAGONAL WORK, and it needs no composer change
        # at all: launch.py's `if fuse and has_fold` branch runs BEFORE the
        # specialized-family guard is consulted, so on a fold that guard is never
        # evaluated and the folded pairs are built by _install_folded_fused_pairs.
        # The sentence this row used to carry named that guard, which was wrong
        # about folds before this round and is corrected here rather than widened
        # around.
        ("off_diagonal_epsilon", frozenset({False, True}),
         "folded_offdiag_magnetic_2d lifted off-device 2026-09-13 to a 2-D "
         "mirror-on-Y grid with off_diagonal_epsilon True -- a cylinder on the "
         "fold plane, whose subpixel averaging writes the off-diagonal chi1inv "
         "rows -- and an Hz source, so the deposit stays on the B seam. It has NOT "
         "been driven; the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB is what drives it. The magnetic "
         "seam does not read chi1inv (its kernel takes no inverse-epsilon "
         "pointer), which is why the ELECTRIC folded pair below stays refused on "
         "the same grids; their D seam is the scratch-output weld's, released as "
         "fused pair D (folded off-diagonal) on 2026-09-15"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; every case that drove "
         "this arm carries a split-field PML, which is what the shared row said "
         "about it"),
    ),
    "fused pair D (folded)": (
        ("dimensions", frozenset({2, 3}),
         "folded_2d is a 2-D grid; folded_3d lifted off-device 2026-09-13 to "
         "dimensions 3 on a (40, 21, 40) mirror-on-Y cell and is driven on the "
         "route campaign that succeeds dispatch_fused_route_2026-09-13_phaseB"),
        ("folded", "required", "this arm IS the folded product"),
        ("conductivity", False, "folded_2d is lossless"),
        ("susceptibilities", 0,
         "folded_dispersive_2d gives its D seam to the folded dispersive product, "
         "so no case drove THIS arm on a susceptibility"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair, "
         "released separately below"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        ("off_diagonal_epsilon", False,
         "no off-diagonal case drove THIS arm: on an off-diagonal fold the D seam's "
         "constitutive arm is 'folded off-diagonal', and folded_offdiag_magnetic_2d "
         "gives that seam to the scratch-output weld released as fused pair D "
         "(folded off-diagonal) on 2026-09-15"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; every case that drove "
         "this arm carries a split-field PML, which is what the shared row said "
         "about it"),
    ),
    # THE 2026-09-11 ARMS. The first three rows of each are the axes its own case
    # drove; the fourth and fifth say which of the other three families' shapes it
    # is NOT released on, so a reader can see the four D-seam products separating
    # each other rather than inferring it from the labels.
    #
    # ``susceptibilities`` IS PINNED AT 1 ON ``fused pair D (folded dispersive)``,
    # AND THE EXEMPTION THIS TABLE GRANTS ``dispersive fused pair`` IS DELIBERATELY
    # NOT EXTENDED TO IT. The exemption's argument fits: the arm's E half REQUIRES a
    # susceptibility to exist at all, its one case drove exactly one Lorentz pole,
    # and the corpus rows this product serves -- tests:TestLoadDump's four 2-D
    # chunk-layout/structure folds, grid (250, 126, 1) -- carry FIVE, so the pin
    # credits this arm ZERO board instances today. It is taken anyway, and the reason
    # is a measurement rather than a preference: the route gate's Triton ENVELOPE
    # WITNESS is ``folded_dispersive5_2d``, that same fold at five poles, and
    # direction 1 of the envelope leg requires the release to admit NOTHING there.
    # Measured on a NumPy lift of the case, with this row at 1: ``released_fused_arms``
    # is empty and this arm's refusal names the axis; with the axis unpinned the arm
    # IS admitted, the witness stops witnessing, and the board reads four more
    # instances. The four instances are the cheaper thing to give up. The exemption
    # stands where it was granted -- it is an argument about the composer, not a
    # measurement of this table -- and widening this row needs a five-pole ROUTE case
    # and an envelope witness that does not sit on the same axis.
    "fused pair D (folded dispersive)": (
        ("dimensions", 2, "folded_dispersive_2d lifts a 2-D (80, 61, 1) grid"),
        ("folded", "required",
         "this arm IS the folded dispersive product; on an unfolded dispersive grid "
         "the D seam goes to `dispersive fused pair`"),
        ("susceptibilities", 1,
         "folded_dispersive_2d carries exactly one Lorentz pole, which is the only "
         "pole count any route case has driven this arm on; the five-pole fold is "
         "the envelope witness and is refused here for that reason"),
        ("conductivity", False, "folded_dispersive_2d is lossless"),
        ("cylindrical", False,
         "its case is a Cartesian fold; no Dcyl case drove this arm"),
        ("complex_storage", False, "folded_dispersive_2d stores real fields"),
        ("bloch", False, "its case carries no Bloch phase"),
        ("beta", 0, "its case is not a special-kz run"),
        ("bfast", False, "its case is not a BFAST run"),
        ("nonlinearity", False, "its case is not nonlinear"),
        ("off_diagonal_epsilon", False,
         "its case has a diagonal epsilon, and this is the seam that reads chi1inv: "
         "on an off-diagonal fold the D seam's constitutive arm is 'folded "
         "off-diagonal dispersive', whose products are not released"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; folded_dispersive_2d "
         "carries a split-field PML, which is what the shared row said about this "
         "arm"),
    ),
    # THE SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-15. Every value is READ off the
    # run shape the route case each row names recorded on
    # ``dispatch_fused_route_2026-09-15_offdiag`` -- 2-D, real storage, lossless, no
    # susceptibility, no Bloch phase, beta 0, no BFAST, linear, a split-field PML,
    # an off-diagonal chi1inv, and a Y mirror on the folded one -- and those are the
    # shapes of the census rows the two rows serve (9 unfolded, 9 folded on the
    # 2026-09-11 census). Where a row admits a shape its route case did not drive,
    # the reason on that axis names the corner.
    "fused pair D (off-diagonal)": (
        ("dimensions", 2,
         "offdiag_magnetic_2d lifts a (200, 120, 1) grid; the product gate also ran "
         "a walled 3-D fixture (walled_3d, S1 IDENTICAL), but no route case has "
         "driven this weld at 3-D, so 3-D is not admitted"),
        ("folded", None,
         "offdiag_magnetic_2d carries no mirror; on a fold the curl half's "
         "pml_curl_coverage refuses and the folded weld is the product"),
        ("conductivity", False,
         "offdiag_magnetic_2d is lossless, and the curl half is the lossless PML "
         "curl arm"),
        ("susceptibilities", 0,
         "offdiag_magnetic_2d carries no susceptibility, and the constitutive half "
         "refuses a registered polarization"),
        ("cylindrical", False, "a Cartesian grid"),
        ("complex_storage", False, "offdiag_magnetic_2d stores real fields"),
        ("bloch", False, "offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "offdiag_magnetic_2d is not a special-kz run"),
        ("bfast", False, "offdiag_magnetic_2d is not a BFAST run"),
        ("nonlinearity", False, "offdiag_magnetic_2d is linear"),
        ("off_diagonal_epsilon", True,
         "this arm IS the off-diagonal weld: its constitutive half requires a live "
         "off-diagonal row, so a diagonal grid is the ordinary fused pair D's. "
         "ADMITTED WITHOUT A DRIVE OF ITS OWN: an off-diagonal run whose electric "
         "source deposits INSIDE the D seam reads identically on every axis here -- "
         "offdiag_2d (Ez source) and offdiag_magnetic_2d (Hz) recorded the same run "
         "shape -- and the product's predicate is what refuses it, by name "
         "(CARRIES_DEPOSIT_REPAIR is False); offdiag_2d is the route control that "
         "measures that refusal"),
        ("pml_active", True,
         "offdiag_magnetic_2d carries a split-field PML, and the curl half is the "
         "absorber curl arm"),
    ),
    "fused pair D (folded off-diagonal)": (
        ("dimensions", 2,
         "folded_offdiag_magnetic_2d lifts a (160, 61, 1) grid; the product gate also "
         "ran a 3-D X fold (fold_x_walled_3d, S1 IDENTICAL), but no route case has "
         "driven this weld at 3-D, so 3-D is not admitted"),
        ("folded", "required",
         "this arm IS the folded weld. ADMITTED WITHOUT A ROUTE DRIVE: the axis does "
         "not say WHICH planes fold, folded_offdiag_magnetic_2d folds on Y alone, and "
         "7 of the 9 census rows this label serves fold on X AND Y. That corner rests "
         "on the product gate's fold_xy_mixed_walled fixture (S1 IDENTICAL against "
         "the array path and the certified singles, "
         "results/triton_offdiag_stencil_welds_2026-09-13_batch/folded)"),
        ("conductivity", False, "folded_offdiag_magnetic_2d is lossless"),
        ("susceptibilities", 0,
         "folded_offdiag_magnetic_2d carries no susceptibility; with a pole the D "
         "seam's constitutive arm is 'folded off-diagonal dispersive', which this "
         "weld does not implement"),
        ("cylindrical", False, "a Cartesian fold"),
        ("complex_storage", False, "folded_offdiag_magnetic_2d stores real fields"),
        ("bloch", False, "folded_offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "folded_offdiag_magnetic_2d is not a special-kz run"),
        ("bfast", False, "folded_offdiag_magnetic_2d is not a BFAST run"),
        ("nonlinearity", False, "folded_offdiag_magnetic_2d is linear"),
        ("off_diagonal_epsilon", True,
         "this arm IS the folded off-diagonal weld; a diagonal fold is fused pair D "
         "(folded)'s. The unfolded weld's source-kind corner applies unchanged: an "
         "in-seam electric deposit reads identically on every axis here and is "
         "refused by the product's predicate by name"),
        ("pml_active", True,
         "folded_offdiag_magnetic_2d carries a split-field PML"),
    ),
    "fused pair D (conductive)": (
        ("dimensions", 2, "conductive_2d is a 2-D grid"),
        ("folded", None, "conductive_2d carries no mirror; on a fold the folded "
                         "products are what the composer selects"),
        ("conductivity", True,
         "conductive_2d carries a D_conductivity block; this arm IS the conductive "
         "product, and on a lossless grid the ordinary fused pair D is what runs"),
        ("susceptibilities", 0, "conductive_2d carries no susceptibility"),
        ("cylindrical", False, "its case is a Cartesian grid"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        ("off_diagonal_epsilon", False,
         "no off-diagonal case drove it, and this is the seam that reads chi1inv: "
         "the Triton composer refuses the ELECTRIC pair by name on an off-diagonal "
         "grid, before the target round's per-pair narrowing of "
         "specialized_family_owns_the_grid and after it"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; conductive_2d carries "
         "a split-field PML beside its D_conductivity block, which is what the "
         "shared row said about this arm"),
    ),
    "fused pair B (cylindrical)": (
        ("dimensions", 2,
         "a Dcyl lift reports dimensions 2 with grid_shape (80, 1, 80)"),
        ("cylindrical", True,
         "cylindrical_m0 is the e2e Dcyl m = 0 shape with a flux monitor; this arm "
         "IS the cylindrical product and the ordinary pair is refused there by "
         "launch.py's has_cylindrical branch"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "cylindrical_m0 is lossless"),
        ("susceptibilities", 0, "cylindrical_m0 carries no susceptibility"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        ("off_diagonal_epsilon", False,
         "cylindrical_m0 has a diagonal epsilon, and on a Dcyl grid launch.py's "
         "has_cylindrical branch refuses every pair by name before the "
         "off-diagonal question is reached at all"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; cylindrical_m0 "
         "carries a split-field PML, which is what the shared row said about this "
         "arm"),
    ),
    "fused pair D (cylindrical)": (
        ("dimensions", 2,
         "a Dcyl lift reports dimensions 2 with grid_shape (80, 1, 80)"),
        ("cylindrical", True,
         "cylindrical_m0 is the e2e Dcyl m = 0 shape with a flux monitor; this arm "
         "IS the cylindrical product and the ordinary pair is refused there by "
         "launch.py's has_cylindrical branch"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "cylindrical_m0 is lossless"),
        ("susceptibilities", 0, "cylindrical_m0 carries no susceptibility"),
        ("complex_storage", False, "every case that drove it stores real fields"),
        ("bloch", False, "no case that drove it carries a Bloch phase"),
        ("beta", 0, "no case that drove it is a special-kz run"),
        ("bfast", False, "no case that drove it is a BFAST run"),
        ("nonlinearity", False, "no case that drove it is nonlinear"),
        ("off_diagonal_epsilon", False,
         "cylindrical_m0 has a diagonal epsilon, and on a Dcyl grid launch.py's "
         "has_cylindrical branch refuses every pair by name before the "
         "off-diagonal question is reached at all"),
        ("pml_active", True,
         "moved off the shared envelope in the target round; cylindrical_m0 "
         "carries a split-field PML, which is what the shared row said about this "
         "arm"),
    ),
    # --- PHASE B, 2026-09-13. Every row READ off the lifted run shape of the case that
    # drove it, on the GPU host, before it was typed. ``beta`` is deliberately absent from
    # the four beta products' rows -- the arm IS the beta product and its predicate
    # requires beta != 0 -- and ``bloch`` from the complex pair's, which bloch_2d
    # (True) and complex_nobloch_2d (False) both drove. ``m`` is not pinned anywhere
    # (see the shared-table note): the census shape carries no m. ---
    "fused pair B (real beta)": (
        ("dimensions", 2, "special_kz_2d is a 2-D grid"),
        ("folded", None, "special_kz_2d carries no mirror; the folded beta grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "special_kz_2d (kz_2d=\"real/imag\") stores real fields; the complex beta grid has its own pair"),
        ("bloch", False, "a z-only k_point lifts as beta, not as a Bloch phase"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (folded real beta)": (
        ("dimensions", 2, "folded_special_kz_2d is a 2-D grid"),
        ("folded", "required", "this arm IS the folded beta product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "folded_special_kz_2d stores real fields"),
        ("bloch", False, "a z-only k_point lifts as beta, not as a Bloch phase"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (complex beta)": (
        ("dimensions", 2, "complex_beta_2d is a 2-D grid"),
        ("folded", None, "complex_beta_2d carries no mirror; the folded complex beta grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS the complex beta product; the real beta grid has its own pair"),
        # WIDENED IN THE TARGET ROUND. The sentence below describes the CASE and
        # not the corpus rows it was refusing: complex_beta_2d's k_point is z-only,
        # so it lifts beta 0.4 and bloch False, while TestSpecialKz.test_special_kz
        # and the binary-grating special-kz rows carry a z beta AND an in-plane
        # phase at once. complex_beta_bloch_2d is that shape.
        ("bloch", frozenset({False, True}),
         "complex_beta_2d's z-only k_point lifts as beta and reads bloch False; "
         "complex_beta_bloch_2d lifted off-device 2026-09-13 to the SAME beta 0.4 "
         "beside an in-plane k_x = 0.25 and reads bloch True, so both values are "
         "driven by a case rather than one of them inferred. It has not been "
         "driven yet -- the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB is what drives it"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (folded complex beta)": (
        ("dimensions", 2, "folded_complex_beta_2d is a 2-D grid"),
        ("folded", "required", "this arm IS the folded complex beta product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS a complex product"),
        ("bloch", frozenset({False, True}),
         "folded_complex_beta_2d's z-only k_point lifts as beta and reads bloch "
         "False; folded_complex_beta_bloch_2d lifted off-device 2026-09-13 to the "
         "same beta 0.4 beside an in-plane k_x = 0.25 on a (120, 62, 1) "
         "mirror-on-Y cell and reads bloch True. It has not been driven yet -- the "
         "route campaign that succeeds dispatch_fused_route_2026-09-13_phaseB is "
         "what drives it"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (BFAST)": (
        ("dimensions", 3, "bfast_1d is refl-angular's shape: a z-only cell DECLARED 3-D, which MEEP builds in 3-D with one cell in x and y (the lift reads 3)"),
        ("folded", None, "no folded BFAST case has been driven"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "BFAST is the REAL-storage broadband-angle product"),
        ("bloch", False, "bfast_1d's k_point is zero; BFAST carries the angle itself"),
        ("beta", 0, "bfast_1d carries no special-kz beta"),
        ("bfast", True, "this arm IS the BFAST product; on an ordinary grid the ordinary pair is what the composer selects"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (complex)": (
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d and complex_nobloch_2d are 2-D grids; three cases lifted "
         "off-device 2026-09-13 give the other two values -- complex_1d, a cell "
         "DECLARED dimensions=1 that reads (1, 1, 480); complex_3d_thinline, a "
         "(0, 0, 11) cell DECLARED 3 that MEEP builds one cell wide in x and y and "
         "the reader keeps at 3, which is the shape all seven served corpus rows "
         "carry; and complex_3d, a genuine (40, 40, 40) complex block that no "
         "board instance needs and that is here so the 3-D value is not evidenced "
         "by degenerate (1, 1, N) grids alone. None has been driven; the route "
         "campaign that succeeds dispatch_fused_route_2026-09-13_phaseB drives all "
         "three"),
        ("folded", None, "neither case carries a mirror; the folded complex grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS the complex product; on a real grid the ordinary pair is what the composer selects"),
        ("beta", 0, "neither case carries a special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    # THE ARM THE TARGET ROUND MOVED ON dimensions, and the two axes it moved, gave
    # back, and takes again. folded_complex_3d moves dimensions against
    # folded_complex_2d (same cell class, mirror and PML) and dispatches. The round
    # also widened off_diagonal_epsilon and bloch, on the premise that an
    # off-diagonal fold needs no composer change. dispatch_fused_route_2026-09-15_weldgrid
    # measured otherwise: on folded_complex_offdiag_2d and
    # folded_complex_nobloch_offdiag_2d the composer also selects `complex folded
    # off-diagonal` for update_E, that arm was in PENDING_DEVICE_GATE_ARMS, and the
    # ladder refused the WHOLE plan -- every slot on the array path, zero Triton
    # launches -- not the update_E slot alone. Both axes went back, and
    # dispatch_fused_route_2026-09-15_offdiag drove both cases as `no-fused-arm`
    # controls (PASS-NO-FUSED-ARM, refusing_rung pending-device-gate).
    #
    # THE 2026-09-15 RULING LIFTED THAT ARM, so both axes widen again, and the lift
    # and the widening are ONE change: the board's served_in_dispatch reads this row
    # and never the pending rung, so a widening without the lift reads +3 on the
    # board and refuses at runtime, which is the weldgrid run exactly. The cases
    # that drive each value are named per axis below; the corners no case drove are
    # named in the axis texts and in the corner paragraph above this table.
    "fused pair B (folded complex)": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d is a 2-D grid; folded_complex_3d lifted off-device "
         "2026-09-13 to dimensions 3 on the (8, 21, 126) triangular-lattice "
         "oblique cell -- the corpus row's own shape -- and is driven on the route "
         "campaign that succeeds dispatch_fused_route_2026-09-13_phaseB"),
        ("folded", "required", "this arm IS the folded complex product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", frozenset({False, True}),
         "folded_complex_2d and folded_complex_3d have a diagonal epsilon; "
         "folded_complex_offdiag_2d (bloch True) and "
         "folded_complex_nobloch_offdiag_2d (bloch False) are 2-D folds with "
         "off-diagonal chi1inv rows, where update_E is the single arm `complex folded "
         "off-diagonal`, released from the pending list on 2026-09-15. NOT DRIVEN: an "
         "off-diagonal fold at 3-D; no census row sits there"),
        ("complex_storage", True, "this arm IS a complex product"),
        ("bloch", frozenset({False, True}),
         "folded_complex_2d, folded_complex_3d and folded_complex_offdiag_2d carry "
         "a Bloch phase; folded_complex_nobloch_offdiag_2d (two mirrors, "
         "force_complex_fields, no k_point) reads bloch False. NOT DRIVEN: a "
         "diagonal fold at bloch False and a 3-D fold at bloch False; no census "
         "row sits in either. `fused pair B (folded complex beta)` (bloch {False, "
         "True}, no beta row) is released on every diagonal 2-D fold this row "
         "admits, as it already was at bloch True, so the widening extends that "
         "overlap to the bloch-False diagonal fold; which label serves is the "
         "composer's selection rather than this table's"),
        ("beta", 0, "folded_complex_2d carries no special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (cylindrical complex)": (
        ("dimensions", 2, "a Dcyl lift reports dimensions 2 with grid_shape (80, 1, 80)"),
        ("cylindrical", True, "this arm IS the Dcyl complex product"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical cases are lossless"),
        ("susceptibilities", 0, "the cylindrical cases carry no susceptibility"),
        ("off_diagonal_epsilon", False, "the cylindrical cases have a diagonal epsilon"),
        ("complex_storage", True, "this arm steps complex64 word pairs only (cylindrical_m1 and cylindrical_m0_complex drove m = 1 and m = 0 under forced complex storage); the real m = 0 Dcyl shape has its own pair"),
        ("bloch", False, "no cylindrical case carries a Bloch phase"),
        ("beta", 0, "no cylindrical case carries a special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (real beta)": (
        ("dimensions", 2, "special_kz_2d is a 2-D grid"),
        ("folded", None, "special_kz_2d carries no mirror; the folded beta grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "special_kz_2d (kz_2d=\"real/imag\") stores real fields; the complex beta grid has its own pair"),
        ("bloch", False, "a z-only k_point lifts as beta, not as a Bloch phase"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (folded real beta)": (
        ("dimensions", 2, "folded_special_kz_2d is a 2-D grid"),
        ("folded", "required", "this arm IS the folded beta product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "folded_special_kz_2d stores real fields"),
        ("bloch", False, "a z-only k_point lifts as beta, not as a Bloch phase"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (complex beta)": (
        ("dimensions", 2, "complex_beta_2d is a 2-D grid"),
        ("folded", None, "complex_beta_2d carries no mirror; the folded complex beta grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS the complex beta product; the real beta grid has its own pair"),
        # WIDENED IN THE TARGET ROUND. The sentence below describes the CASE and
        # not the corpus rows it was refusing: complex_beta_2d's k_point is z-only,
        # so it lifts beta 0.4 and bloch False, while TestSpecialKz.test_special_kz
        # and the binary-grating special-kz rows carry a z beta AND an in-plane
        # phase at once. complex_beta_bloch_2d is that shape.
        ("bloch", frozenset({False, True}),
         "complex_beta_2d's z-only k_point lifts as beta and reads bloch False; "
         "complex_beta_bloch_2d lifted off-device 2026-09-13 to the SAME beta 0.4 "
         "beside an in-plane k_x = 0.25 and reads bloch True, so both values are "
         "driven by a case rather than one of them inferred. It has not been "
         "driven yet -- the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB is what drives it"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (folded complex beta)": (
        ("dimensions", 2, "folded_complex_beta_2d is a 2-D grid"),
        ("folded", "required", "this arm IS the folded complex beta product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS a complex product"),
        ("bloch", frozenset({False, True}),
         "folded_complex_beta_2d's z-only k_point lifts as beta and reads bloch "
         "False; folded_complex_beta_bloch_2d lifted off-device 2026-09-13 to the "
         "same beta 0.4 beside an in-plane k_x = 0.25 on a (120, 62, 1) "
         "mirror-on-Y cell and reads bloch True. It has not been driven yet -- the "
         "route campaign that succeeds dispatch_fused_route_2026-09-13_phaseB is "
         "what drives it"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (BFAST)": (
        ("dimensions", 3, "bfast_1d is refl-angular's shape: a z-only cell DECLARED 3-D, which MEEP builds in 3-D with one cell in x and y (the lift reads 3)"),
        ("folded", None, "no folded BFAST case has been driven"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "BFAST is the REAL-storage broadband-angle product"),
        ("bloch", False, "bfast_1d's k_point is zero; BFAST carries the angle itself"),
        ("beta", 0, "bfast_1d carries no special-kz beta"),
        ("bfast", True, "this arm IS the BFAST product; on an ordinary grid the ordinary pair is what the composer selects"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (complex)": (
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d and complex_nobloch_2d are 2-D grids; three cases lifted "
         "off-device 2026-09-13 give the other two values -- complex_1d, a cell "
         "DECLARED dimensions=1 that reads (1, 1, 480); complex_3d_thinline, a "
         "(0, 0, 11) cell DECLARED 3 that MEEP builds one cell wide in x and y and "
         "the reader keeps at 3, which is the shape all seven served corpus rows "
         "carry; and complex_3d, a genuine (40, 40, 40) complex block that no "
         "board instance needs and that is here so the 3-D value is not evidenced "
         "by degenerate (1, 1, N) grids alone. None has been driven; the route "
         "campaign that succeeds dispatch_fused_route_2026-09-13_phaseB drives all "
         "three"),
        ("folded", None, "neither case carries a mirror; the folded complex grid has its own pair"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS the complex product; on a real grid the ordinary pair is what the composer selects"),
        ("beta", 0, "neither case carries a special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (folded complex)": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d is a 2-D grid; folded_complex_3d lifted off-device "
         "2026-09-13 to dimensions 3 on the (8, 21, 126) triangular-lattice "
         "oblique cell and is driven on the route campaign that succeeds "
         "dispatch_fused_route_2026-09-13_phaseB. The off-diagonal folds the "
         "MAGNETIC arm gains above are NOT this arm's: on an off-diagonal fold the "
         "D seam goes to the folded off-diagonal products, so this row stays "
         "diagonal"),
        ("folded", "required", "this arm IS the folded complex product"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", True, "this arm IS a complex product"),
        ("bloch", True, "folded_complex_2d carries a Bloch phase; no unphased folded complex case has been driven"),
        ("beta", 0, "folded_complex_2d carries no special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair D (cylindrical complex)": (
        ("dimensions", 2, "a Dcyl lift reports dimensions 2 with grid_shape (80, 1, 80)"),
        ("cylindrical", True, "this arm IS the Dcyl complex product"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical cases are lossless"),
        ("susceptibilities", 0, "the cylindrical cases carry no susceptibility"),
        ("off_diagonal_epsilon", False, "the cylindrical cases have a diagonal epsilon"),
        ("complex_storage", True, "this arm steps complex64 word pairs only (cylindrical_m1 and cylindrical_m0_complex drove m = 1 and m = 0 under forced complex storage); the real m = 0 Dcyl shape has its own pair"),
        ("bloch", False, "no cylindrical case carries a Bloch phase"),
        ("beta", 0, "no cylindrical case carries a special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", False, "linear"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    "fused pair B (nonlinear)": (
        ("dimensions", 1, "nonlinear_1d is 3rd-harm-1d's 1-D shape"),
        ("folded", None, "nonlinear_1d carries no mirror"),
        ("cylindrical", False, "a Cartesian grid"),
        ("conductivity", False, "its case is lossless"),
        ("susceptibilities", 0, "its case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "its case has a diagonal epsilon"),
        ("complex_storage", False, "nonlinear_1d stores real fields"),
        ("bloch", False, "nonlinear_1d carries no Bloch phase"),
        ("beta", 0, "nonlinear_1d carries no special-kz beta"),
        ("bfast", False, "not a BFAST run"),
        ("nonlinearity", True, "this arm IS the nonlinear B product; on a linear grid the ordinary pair is what the composer selects"),
        ("pml_active", True, "moved off the shared envelope in the target round; its case carries a split-field PML"),
    ),
    # --- THE TARGET ROUND'S ONE NEW ARM, and the row that made ``pml_active`` a
    # per-arm axis. ``fused pair D (no-PML stored E)`` runs where there is no
    # split-field PML at all, so the shared envelope row would have had to hold
    # True and False at once; every other arm now carries its own True and this one
    # carries False. See :data:`RELEASED_FUSED_ARMS` for its two cases and the
    # ledger entry that lifts its pending row.
    #
    # ``conductivity`` AND ``susceptibilities`` ARE DELIBERATELY ABSENT. The two
    # cases drive both values of the first (absorber_1d True, no_pml_dispersive_2d
    # False) and two different counts of the second (5 and 2), which is the
    # condition this table states for leaving an axis out -- an axis is pinned when
    # the arm's own case list drove exactly ONE of its values. The second case was
    # ``material_dispersion_0d`` until 2026-09-14 and carried the SAME two values on
    # a zero-extent cell; the replacement keeps both axes unpinned for the same
    # reason, on a cell whose controls can actually fire.
    #
    # AND THAT LEAVES AN ADMITTED ASYMMETRY, recorded here as one, under the same
    # exemption this table grants ``dispersive fused pair`` -- and grants to that arm
    # alone, since the re-attribution round declined to extend it to ``fused pair D
    # (folded dispersive)`` and pinned that row at one pole instead, for the envelope
    # witness's sake. MEASURED on the shipped predicate: a ZERO-pole,
    # lossless, diagonal, no-absorber 1-D or 2-D grid is admitted by this row --
    # ``released_fused_arms`` returns exactly this arm there -- and on such a grid
    # there is no stored E for the product's electric half to update, so it is an
    # admission the composer has nothing to select against. That is an argument
    # about the composer rather than a measurement of this table, which is why the
    # axis stays unpinned rather than being narrowed to a value no case drove. The
    # case that would settle it is ``no_pml_2d``, the existing no-absorber control
    # whose Triton row expects NO fused arm -- and it does not settle it today,
    # because it lifts ``off_diagonal_epsilon True`` (a cylinder writes the
    # off-diagonal rows) and this row refuses it on THAT axis instead. So the
    # zero-pole corner is unmeasured in both directions and is named here rather
    # than left for a reader to find.
    "fused pair D (no-PML stored E)": (
        ("dimensions", frozenset({1, 2}),
         "absorber_1d lifted off-device 2026-09-13 to dimensions 1 on a (1, 1, 400) "
         "declared-1-D cell and no_pml_dispersive_2d lifted off-device 2026-09-14 to "
         "dimensions 2 on a (120, 120, 1) cell; no case drove 3. The 2 was carried "
         "by material_dispersion_0d's zero-extent (1, 1, 1) cell until 2026-09-14, "
         "which MEEP also infers as 2 -- the replacement moved the evidence onto a "
         "grid whose controls can diverge, not onto a different axis value"),
        ("folded", None, "neither case carries a mirror"),
        ("cylindrical", False, "both cases are Cartesian"),
        ("complex_storage", False, "both cases store real fields"),
        ("bloch", False, "neither case carries a Bloch phase"),
        ("beta", 0, "neither case is a special-kz run"),
        ("bfast", False, "neither case is a BFAST run"),
        ("nonlinearity", False, "both cases are linear"),
        ("off_diagonal_epsilon", False, "both cases have a diagonal epsilon"),
        ("pml_active", False,
         "this arm IS the no-absorber electric pair: absorber_1d's boundary is an "
         "mp.Absorber, a scalar conductivity rather than a split-field PML, and "
         "no_pml_dispersive_2d declares no boundary layers at all, so both lift "
         "pml_active False. On a PML grid the ordinary fused pair D is what the "
         "composer selects"),
    ),
    # --- 2026-09-15: THE COMPLEX NO-ABSORBER D PAIR. EVERY AXIS IS PINNED, because
    # the arm has ONE case and an axis is left open here only where its own cases
    # drove more than one value. Each value is the ``run_shape`` the Triton-only
    # unfused leg of ``dispatch_fused_route_cuda_2026-09-15_weldgrid``
    # (``cuda_shipped_expansion_probe``) recorded on ``complex_no_pml_3d``, which is
    # the same read the route builder's 2026-09-13 lift gave. The row mirrors the
    # hand-CUDA table's ``cuda:complex no-absorber three-slot weld`` row, which rests
    # on the same case.
    #
    # WHAT IS ADMITTED WITHOUT BEING DRIVEN, named because a row can admit more than
    # its case: only the grid EXTENT, which this table never pins (see the note
    # above :data:`FUSED_RELEASE_ENVELOPE`). The case is (23, 21, 27); the four
    # TestLoadDump rows it serves read (35, 32, 41) through
    # ``dispatch_reachability.run_shape_from_census`` with the other twelve values
    # equal to the case's (read 2026-09-15 on the 2026-09-11 Triton census). With
    # every other axis an equality there is no second corner.
    "fused pair D (complex conductive no-PML)": (
        ("dimensions", 3,
         "complex_no_pml_3d lifted to dimensions 3, grid (23, 21, 27); it is the only "
         "route case this arm has, so a 1-D or 2-D complex no-absorber conductive "
         "grid stays refused until a case drives one"),
        ("folded", None, "the case carries no mirror"),
        ("cylindrical", False, "the case is Cartesian"),
        ("complex_storage", True,
         "this arm IS the complex no-absorber D pair; on real storage the real "
         "no-absorber pair, fused pair D (no-PML stored E), is what the composer "
         "selects"),
        ("bloch", True,
         "the case carries an oblique k_point (0.4, -1.3, 0.7) and lifted bloch True; "
         "pinned because ONE value was driven, not because it looks incidental"),
        ("beta", 0, "the case carries no special-kz beta"),
        ("bfast", False, "the case is not a BFAST run"),
        ("nonlinearity", False, "the case is linear"),
        ("conductivity", True,
         "the case's mp.Absorber is a scalar conductivity, and no case drives the "
         "False side"),
        ("off_diagonal_epsilon", False,
         "the case carries a diagonal epsilon; complex_no_pml_offdiag is the "
         "off-diagonal shape of this span, and its update_E arm, complex no-PML "
         "off-diagonal, is still pending"),
        ("susceptibilities", 1,
         "the case carries ONE Lorentz pole. The pair gate's product cases run 1, 2 "
         "and 4 poles (triton_fleet_2026-09-14_target_A), but no route case has "
         "driven more than one pole through the seam on this arm, so a multi-pole "
         "complex no-absorber grid stays refused until one does"),
        ("pml_active", False,
         "this arm IS the complex no-absorber electric pair: the case's boundary is "
         "an mp.Absorber rather than a split-field PML, so it lifted pml_active "
         "False. On a split-field PML grid the released complex D pair is fused "
         "pair D (complex), whose row pins pml_active True"),
    ),
}


def fused_release_reasons(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """Why this configuration is OUTSIDE what the driver-route gate measured.

    Empty means the release applies here. Every reason NAMES the axis, the value
    read, and what the gate ran instead — a refusal that said only "outside the
    envelope" would be untraceable to the measurement that drew it.

    FAILS CLOSED ON AN UNREAD FACT, which is the opposite of the device rung's
    rule and deliberately so. Rung (4b) does not refuse a compute capability it
    could not read, because refusing on an unread fact asserts one; here the
    decision is an ADMISSION, and admitting on an unread fact asserts one.
    """
    if "unreadable" in shape:
        return (f"the run shape did not read ({shape['unreadable']}); admitting a "
                "released arm on a configuration this record cannot describe would "
                "be asserting the configuration",)
    return _axis_reasons(shape, FUSED_RELEASE_ENVELOPE)


def _axis_reasons(shape: Mapping[str, Any],
                  axes: Sequence[Tuple[str, Any, str]]) -> Tuple[str, ...]:
    """Why ``shape`` is outside ``axes``, one reason per axis, naming the read value.

    FOUR KINDS OF REQUIRED VALUE, and each spells a different question:

    * ``None`` — the axis must be ABSENT (how :func:`_run_shape` says "not folded");
    * ``"required"`` — the axis must be PRESENT, whatever it says (how a folded
      product says "this arm only exists on a fold"); the fold description is free
      text, so its VALUE is not a thing an envelope can pin;
    * a ``frozenset`` — any of these values, for an axis the gate drove more than
      one value of;
    * anything else — equality.

    AN AXIS THAT DID NOT READ IS A REFUSAL, not a pass, for every kind but the
    first. :func:`fused_release_reasons` fails closed on an unread fact because the
    decision it feeds is an ADMISSION, and that rule has to hold per axis too.
    """
    reasons: List[str] = []
    for axis, required, measured in axes:
        present = axis in shape
        value = shape.get(axis)
        if required is None:
            if present:
                reasons.append(f"{axis}={value!r}; {measured}")
        elif required == "required":
            if not present:
                reasons.append(f"{axis} is absent; {measured}")
        elif not present:
            reasons.append(f"{axis} did not read; {measured}")
        elif isinstance(required, frozenset):
            if value not in required:
                reasons.append(f"{axis}={value!r}, not one of "
                               f"{sorted(required)!r}; {measured}")
        elif value != required:
            reasons.append(f"{axis}={value!r}, not {required!r}; {measured}")
    return tuple(reasons)


def fused_release_arm_reasons(shape: Mapping[str, Any],
                              arm: str) -> Tuple[str, ...]:
    """Why THIS arm is outside what its own cases drove, beyond the shared envelope.

    Empty means the arm's own axes hold here — which is not by itself an admission;
    :func:`released_fused_arms` requires both this and
    :func:`fused_release_reasons`.

    AN ARM WITH NO ROW FAILS CLOSED. A released label that nobody wrote per-arm axes
    for is refused by name rather than admitted on the shared table alone, because
    the shared table is deliberately only the axes every arm agrees on: an arm
    missing from :data:`FUSED_RELEASE_ARM_AXES` has no statement about the axes that
    separate the arms from each other, and admitting on silence is the shape of
    every over-claim this file exists to prevent.
    """
    axes = FUSED_RELEASE_ARM_AXES.get(arm)
    if axes is None:
        return (f"{arm} has no per-arm axis row (FUSED_RELEASE_ARM_AXES), so the "
                "configurations its own gate cases drove have never been written "
                "down; the shared envelope alone cannot admit it",)
    return _axis_reasons(shape, axes)


def released_fused_arms(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """The fused arm labels admitted here: the shared envelope AND the arm's own axes.

    THE PER-ARM CASE LIST STOPPED BEING PROVENANCE ONLY ON 2026-09-02, and this
    paragraph is what used to state the gap it closed. :data:`RELEASED_FUSED_ARMS`
    maps each arm to the gate cases that drove IT; until this round the function
    consulted only the key, so any shape the shared envelope admitted got EVERY
    released arm — ``fused pair D``, driven on two clean-D cases, was admitted on a
    conductive or dispersive shape as well, because the shared table left both axes
    unpinned on the ground that the gate had measured both values across its case
    SET rather than per arm.

    IT IS NOW DECIDED IN TWO PARTS. :data:`FUSED_RELEASE_ENVELOPE` is the shared
    half — the axes every released arm was driven on with the same value — and
    :data:`FUSED_RELEASE_ARM_AXES` is the per-arm half, which is what lets the two
    folded arms REQUIRE a fold while the three unfolded ones refuse it, and what
    narrows ``fused pair D`` to the lossless susceptibility-free shapes its own
    cases ran. Measured against the 2026-09-02 board's credited instances, the
    narrowing costs zero live rows and the folded release adds 98.
    """
    if fused_release_reasons(shape):
        return ()
    return tuple(sorted(arm for arm in RELEASED_FUSED_ARMS
                        if not fused_release_arm_reasons(shape, arm)))


#: The step budget the CAMPAIGN states, reported once at the top of the record.
#: NOT extrapolated, and NOT a per-family claim: the 24000-step leg ran on
#: 2d_dispersive and 2d_pml, which is the pml_curl and dispersive material only.
#: Each family's OWN budget — bfast 80/80 complete steps, complex 78/78,
#: special_kz 192/192 — is read out of that family's record by
#: :func:`_certification_for` and reported beside its arm, because telling a user
#: dispatching BFAST that their certification covers 24000 steps on two cases that
#: family never ran is exactly the over-claim this artifact exists to prevent.
CERTIFICATION_BUDGET = (
    "Per family, from that family's own gate record (see families[*].step_budget). "
    "The campaign-wide legs: 8-64 complete steps per composition case (60 on the "
    "whole-step bit-identity legs); 24000/24000 steps bit-identical between the "
    "Triton and array paths on 2d_dispersive and 2d_pml — the pml_curl and "
    "dispersive families' cases — under a uniform subnormal policy"
)

#: Every one of the nine families was cut under the "keep" policy — and, since the
#: driver-route gate, this is also the policy DISPATCH INSTALLS. One constant
#: governs both, so a re-certification under another policy moves the gate with it.
CERTIFICATION_SUBNORMAL_POLICY = "keep"

CERTIFICATION_SUBNORMAL_NOTE = (
    "GATED ON, NOT MERELY RECORDED. This note used to say the shipped flush "
    "resolution was covered transitively and that dispatch does not compare the "
    "run's policy against the certification's. That was WRONG, and the "
    "driver-route gate measured it wrong: the transitive argument holds only for a "
    "process that INSTALLED a policy, and the shipped path installs none, so CuPy "
    "flushes by its own unconditional -ftz=true while Triton keeps by its own "
    "default. Read directly out of the emitted code on the same sub-step: 19 of 19 "
    "CuPy f32 PTX instructions carry .ftz against 0 of 186 for Triton, and the "
    "generated Triton PTX is byte-identical to its keep-policy build while the "
    "generated CuPy PTX is byte-identical to its flush-policy build. End to end "
    "that split diverged 6 of 7 dispatching cases inside the 8-64-step window the "
    "families are certified over. Dispatch therefore refuses unless host, CuPy and "
    "Triton are all attained under this policy, and installs it itself by default."
)

#: THE POLICY EACH TABLE CERTIFIES UNDER. Not an exemption from the one-policy rule
#: — it is the same rule with the table's own value, and the value has to be
#: ATTAINABLE ON EVERY EXECUTOR THAT TABLE DRIVES or the rule is decoration.
#:
#: WHY THE METAL ROW IS ``flush`` AND THE TRITON ROW IS NOT. MPS flushes float32
#: denormals natively and exposes no lever (``metal_kernels/subnormal.ATTAINABLE``
#: is ``(FLUSH,)``), so ``keep`` is unattainable on that table's device half; the
#: arm64 host is driven to flush by ``fesetenv`` on
#: ``_FE_DFL_DISABLE_DENORMS_ENV``, measured 2026-09-10 as attained and reversible.
#: On the Triton table the argument runs the other way and is recorded at
#: :data:`CERTIFICATION_SUBNORMAL_NOTE`: ``flush`` is unattainable in a process that
#: has already imported CuPy, and ``keep`` is what all nine families were cut under.
#:
#: WHAT A SPLIT WOULD HAVE COST, measured rather than argued: under a KEEPING host
#: the released Metal pairs are byte-identical at step 12 and diverge at step 24
#: (``pml_2d`` 12 arrays / 108 words on ``fields.Ez``), with every differing element
#: inside a few ULP of the subnormal band. A Metal dispatch under host-keep would be
#: a certified twelve-step composition and a documented divergence forever after.
#:
#: :data:`CERTIFICATION_SUBNORMAL_POLICY` keeps its value and every existing reader
#: (the route gate, the conftest, the tests) is unchanged: this map is beside it,
#: not in front of it.
#: WHY THE CUDA ROW IS THE TRITON VALUE AND NOT A THIRD ONE. Every entry in
#: ``cuda_kernels/fingerprints.json`` records TWO canonical legs —
#: ``meep_x86_flush`` and ``ieee_keep_ftz_stripped`` — and the keep leg is the
#: dispatching one: the hand-written kernels compile through
#: ``compile_cache.compile_policy_token`` with CuPy's ``-ftz=true`` stripped, which
#: is the same arithmetic the Triton families were cut under. So the two NVIDIA
#: tables can be composed into ONE step under ONE policy, which is what makes the
#: merge sound at all — a merged plan under two policies would be the executor split
#: :data:`CERTIFICATION_SUBNORMAL_NOTE` records, one layer up.
TABLE_SUBNORMAL_POLICY: Mapping[str, str] = {
    "triton": CERTIFICATION_SUBNORMAL_POLICY,
    CUDA_TABLE: CERTIFICATION_SUBNORMAL_POLICY,
    METAL_TABLE: "flush",
}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def _pair_identity(entry: Any) -> int:
    """The identity that EVERY slot of one fused pair agrees on, in either shape.

    A fused product occupies two ``STEP_ORDER`` slots, and every backend's installer
    writes the same two shapes — ``triton_kernels/launch.py:1660-1668``,
    ``metal_kernels/launch.py:737-745`` and
    ``cuda_kernels/fused_pairs.py:_install_fused_pair``. That third path is now spelt
    out where it used to be cited without one: until Phase 2 this module imported
    ``triton_kernels`` and nothing else, and the boundary test refused to find the
    hand-CUDA package NAMED here at all. It composes that table as its second table
    now, so the honest boundary is narrower and the test says so — one function-local
    import inside :func:`_decide`, below the backend rung, and no engine module may
    name a FAMILY at all:

    * NO DEPOSIT — the curl slot holds the pair plan itself and the constitutive
      slot holds ``NoopPlan``, whose ``absorbed_by`` names the pair plan;
    * A DEPOSIT IN THE SEAM — the curl slot holds ``LeadingRepairPlan``, whose
      ``absorbed_by`` names its ``inner`` pair plan, and the constitutive slot holds
      ``TrailingRepairPlan``, whose ``absorbed_by`` names THAT SAME inner plan
      (``deposit_repair.py:683``, ``:711``).

    So "does this slot carry an ``absorbed_by``" tells the two shapes apart rather
    than the two slots of a pair: in the repair shape BOTH slots carry one. Reading
    ``absorbed_by`` as an IDENTITY instead — the object that does the work, which is
    the entry itself when it names nobody — collapses to the same integer for both
    slots of both shapes, and to a slot-unique one for every unpaired plan
    (``update_P`` included, where the slot holds a LIST that names nobody).

    That distinction is not academic: reading it the other way is what left the 38
    deposit-repair pairs of the released composition outside the mid-step licence
    guard while its own comment said they were covered.

    THE WALK IS TRANSITIVE, AND ON THE SHIPPED TRITON SHAPES THAT CHANGES NOTHING.
    Every Triton wrapper names the pair plan itself and the pair plan names nobody,
    so one hop and the full walk answer the same integer on all three installed
    shapes (``LeadingWithdrawPlan``/``NoopPlan``, bare pair/``NoopPlan``,
    ``LeadingRepairPlan``/``TrailingRepairPlan``) — verified against the composer
    rather than assumed. What the walk is FOR is the third backend's triple: a
    ``_TripleHalfPlan`` names the triple's LEADING half, which names the triple, so
    a single hop puts the three slots of one product into TWO identities and
    ``_split_pairs``, ``_pair_of`` and the mid-step commitment clause each treat the
    third slot as unpaired — the same defect the deposit-repair shape had, one level
    deeper. Following the chain to its terminal object collapses all three.

    BOUNDED, and by two things rather than one: an object that names itself, and a
    cycle. Neither is reachable from any shipped installer; both are cheaper to
    answer here than to reason about, because this runs at plan time on a mapping
    the composer built and a non-terminating walk here would hang the freeze.
    """
    current = entry
    seen = {id(current)}
    for _ in range(_ABSORB_CHAIN_LIMIT):
        inner = getattr(current, "absorbed_by", None)
        if inner is None or inner is current or id(inner) in seen:
            break
        seen.add(id(inner))
        current = inner
    return id(current)


#: How far :func:`_pair_identity` follows ``absorbed_by`` before it stops. The
#: deepest shipped chain is two (a triple's trailing half -> its leading half ->
#: the triple); the limit is a guard against a malformed plan, not a budget.
_ABSORB_CHAIN_LIMIT = 8


def _split_pairs(plans: Mapping[str, Any], slots: Sequence[str],
                 order: Sequence[str]) -> Tuple[str, ...]:
    """Fused pairs that would dispatch only PART of themselves — named, one per pair.

    THE ONE COMPOSITION THE SEAM CANNOT GUARD, so it is refused at PLAN TIME rather
    than admitted. :meth:`FastPathPlan.dispatch` reasons about a pair through the
    slots that actually dispatch, and every safety it offers — the mid-step licence
    clause, its step-scoped commitment — assumes the pair arrives whole. A pair whose
    slots do NOT all survive to :attr:`FastPathPlan.slots` breaks that on every step,
    not on a lapse:

    * the LEADING slot dropped — the driver runs the array curl, then the surviving
      absorbed slot dispatches its sentinel, which does nothing, and the
      constitutive sub-step never runs on either path;
    * the ABSORBED slot dropped — the fused launch performs both halves and the
      driver then runs the array constitutive call ON TOP, which accumulates.

    Nothing in the shipped composer produces this today: ``_install_fused_pair``
    writes both slots together and the null-arm drop is keyed on a label no fused
    product carries. The reachable route is the WARM PASS, which unfills per slot
    (``warm_failures``), so one half of a pair failing to compile would leave the
    other standing. That is a refusal, not a fallback: refusing the whole plan puts
    the entire step on the array path, which is correct everywhere.

    A pair NONE of whose slots dispatches is not this shape and is not refused. It is
    the ordinary unfill — both halves on the array path, which is the answer
    ``test_a_slot_whose_kernel_will_not_compile_unfills_and_the_rest_dispatches``
    pins for every other slot — and refusing the plan over it would take the rest of
    a working dispatch set down with an unrelated compile failure.
    """
    dispatching = set(slots)
    position = {name: index for index, name in enumerate(order)}
    grouped: Dict[int, List[str]] = {}
    for name in sorted(plans, key=lambda n: position.get(n, len(position))):
        grouped.setdefault(_pair_identity(plans[name]), []).append(name)
    refusals = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        missing = [name for name in members if name not in dispatching]
        if not missing or len(missing) == len(members):
            continue
        refusals.append(
            f"the fused product occupying {', '.join(members)} would dispatch only "
            f"{', '.join(n for n in members if n in dispatching)}: "
            f"{', '.join(missing)} did not survive to the dispatch set, and a pair "
            f"that runs half of itself either skips a sub-step or double-applies one")
    return tuple(refusals)


@dataclass(frozen=True)
class FastPathPlan:
    """One frozen configuration's DISPATCH SET, built once per configuration freeze.

    Built by :func:`plan_fast_path` at the driver's configuration freeze (the
    first ``step()`` after construction or after ``invalidate_fast_path()``) and
    never mutated afterwards: a material mutation invalidates the whole plan and
    the next step re-plans from scratch. That is what keeps the cached device
    views inside ``step_plan`` honest — nothing here can outlive the arrays it
    points at, and ``driver.invalidate_fast_path`` is the one funnel every
    mutator already calls.

    Attributes:
        fields: The ``Fields`` the plan was built from, held for the identity
            guard in :meth:`dispatch`. Not a reference the plan reads through;
            an ``is`` comparison and nothing more.
        step_plan: The ``TritonStepPlan`` the composer produced.
        slots: The sub-steps that will actually DISPATCH, in step order. Every
            one is kernel-bearing: null arms are dropped and fused arms refuse
            the whole plan before this is built.
        arms: ``slot -> arm label`` for the dispatching slots.
        dropped_null: ``slot -> arm label`` for slots removed because their
            selected arm launches nothing.
        unwarmed: ``slot -> reason`` for slots the plan-time warm pass could not
            compile ahead of step 1. Dispatched anyway; the reason is recorded.
        record: The dispatch artifact for this freeze (see :func:`plan_fast_path`).
        launches: ``slot -> how many times a kernel was actually RUN from this
            plan``. The one field here that is not decided at the freeze — see
            :attr:`launch_counters`.
    """

    fields: Any
    step_plan: Any
    slots: Tuple[str, ...]
    arms: Mapping[str, str]
    dropped_null: Mapping[str, str]
    unwarmed: Mapping[str, str]
    record: Mapping[str, Any]
    #: Mutable inside a frozen plan on purpose: the dataclass freezes the
    #: DECISION, and this counts what the decision then did. Nothing reads it to
    #: decide anything.
    launches: Dict[str, int] = field(default_factory=dict, repr=False)
    #: ``(subnormal_policy module, required policy, epoch at the freeze)`` — the
    #: licence :meth:`dispatch` re-checks on every consult. ``None`` only for a
    #: plan built without the gate having run, which no dispatching plan is.
    policy_licence: Optional[Tuple[Any, str, int]] = field(default=None, repr=False)
    #: WHICH KERNEL TABLE composed this plan — ``"triton"`` or ``"metal"``. Read by
    #: :meth:`dispatch` for the one protocol the two tables spell differently
    #: (``update_P``), and reported so an artifact naming a dispatched arm also names
    #: the composer that produced it. The default keeps every existing constructor
    #: call — the gates, the probes and the tests build this object directly.
    table: str = "triton"
    #: The Metal ``device.Residency`` the composition registered its mirrors
    #: against, or ``None`` on a table that has no device-side mirror at all. HELD,
    #: NOT CONSULTED: the sync bracket is installed at PLAN time by
    #: ``metal_kernels.launch.wrap_for_residency``, so by the time the driver reaches
    #: a consult the plan in the slot already carries its own copies. What this field
    #: is for is the record — ``residency_syncs`` in :meth:`report` is the counter a
    #: gate reads back to prove the bracket ran.
    residency: Any = field(default=None, repr=False)
    #: ``slot -> which kernel table composed the plan in it``, on a plan whose slots
    #: came from more than one composer. RECORD-ONLY: nothing in this class decides
    #: anything from it, and an empty mapping means "every slot came from
    #: :attr:`table`", which is every single-table plan including all three gates'
    #: and every test's. What it is for is the artifact and the route gate's
    #: evidence clause — a merged step that credits a launch to the wrong table
    #: would be an un-catchable mis-attribution, because both tables launch real
    #: kernels over the same arrays.
    backends: Mapping[str, str] = field(default_factory=dict, repr=False)
    #: Slot -> how many consults were answered False because the policy licence
    #: no longer held. Non-zero means the run left the certified configuration
    #: mid-flight and the array path took over; the count is the evidence.
    licence_lapses: Dict[str, int] = field(default_factory=dict, repr=False)
    #: Slot -> how many consults on the MAGNETIC SYNCHRONIZATION path were answered
    #: False because the plan there spans past :data:`SYNC_PATH_SLOTS` — see
    #: :meth:`_sync_path_holds`. Non-zero means a flux or field-energy call took the
    #: array ``update_H`` while ``step`` keeps taking the kernel, which is correct and
    #: is also a launch the run is not getting; the count is what says so.
    sync_refusals: Dict[str, int] = field(default_factory=dict, repr=False)
    #: The LEADING SLOT of every fused pair whose launch has run IN THE CURRENT STEP.
    #: Mutable inside a frozen plan for the same reason ``launches`` is: it records
    #: what the decision DID, not the decision. Emptied at every step roll-over in
    #: :meth:`dispatch`, so a commitment can never outlive the step that made it —
    #: see the roll-over comment there for what a leaked one did.
    _committed: Set[str] = field(default_factory=set, repr=False)
    #: Index into :attr:`slots` of the previous consult this plan SERVED, ``-1``
    #: before the first. A one-element list rather than a plain int because the
    #: dataclass is frozen; like ``launches`` it is state, not decision.
    _last_position: List[int] = field(default_factory=lambda: [-1], repr=False)
    #: ``slot -> the slots of the fused pair it belongs to``, in :attr:`slots` order,
    #: built once by :meth:`_pair_of`. THIS SEAM IS THE PER-TIMESTEP LOOP, so the
    #: grouping is derived from ``step_plan.plans`` — frozen for the plan's life —
    #: rather than rebuilt on each of the five consults of every step.
    _pair_slots: Dict[str, Tuple[str, ...]] = field(default_factory=dict, repr=False)

    @property
    def replaces(self) -> Tuple[str, ...]:  # The sub-steps this plan takes over.
        return self.slots

    @property
    def dispatches_a_kernel(self) -> bool:
        """Will any KERNEL launch? What ``driver.active_step_path`` answers with.

        Not ``bool(step_plan.replaces)``: the null family's product replaces
        ``update_H``/``update_E`` with a plan that does what the array path does,
        namely nothing, so a NumPy host with an inert layer would read "fused" off
        a composition that launches no kernel at all. Null arms are dropped before
        :attr:`slots` is built, so this is the true answer.
        """
        return bool(self.slots)

    def dispatch(self, slot: str, fields: Any) -> bool:
        """Run ``slot`` from this plan, or answer False so the caller runs the array call.

        The driver's consult is ``if fast is None or not fast.dispatch(slot,
        fields): <array call>`` — ten identical two-line sites — so False here
        must mean "nothing was run", and True must mean "the whole sub-step was".
        There is no third answer and no partial one.

        THE TWO FILL SLOTS ARE THE ONE PLACE THAT LOOKS LIKE AN EXCEPTION AND IS
        NOT. ``fill_B``/``fill_D`` name a sub-step the driver runs as TWO array
        passes with ``zero_metal_*`` between them, so the seam has two consult
        sites for it. Consulted with the SLOT name this stands in front of
        ``fill_symmetry_bc_*``; consulted with the PASS name
        ``fill_folded_far_ghosts_*`` it stands in front of that pass and routes to
        :meth:`_dispatch_far_fill`. Each answer is total for the pass its site
        guards, and neither site is ever told "half".

        The ``fields`` argument is an IDENTITY GUARD, not a parameter: a plan
        holds cached device views, and a ``Fields`` that was rebound without
        going through ``invalidate_fast_path`` would have this plan writing the
        previous allocation. Cheap, and it fails to the array path.

        THE SUBNORMAL POLICY IS THE SECOND GUARD, and the same shape: everything
        rung 8b established is a statement about the freeze, so the licence is
        re-checked here rather than assumed — see
        :meth:`_policy_still_licenses_this`.

        THE ONE PIECE OF STATE THIS METHOD KEEPS is a fused pair's commitment, and
        it lives for exactly one step: the consult positions are the driver's own
        order, so a position that does not advance opens a new step and clears what
        the last one left. Both of the shapes a pair can take are covered — see
        :func:`_pair_identity` — and a pair that would dispatch only half of itself
        never reaches here, because :func:`_split_pairs` refuses it at plan time.

        AN EXCEPTION OUT OF ``run()`` PROPAGATES, deliberately (see the module
        docstring): a launch that raised mid-write leaves the targets partly
        updated, and the array call on top would double-apply the sub-step.
        """
        if fields is not self.fields:
            return False
        owner = FAR_FILL_OWNERS.get(slot)
        if owner is not None:
            # THE FAR HALF OF A FILL, which is a different question from the five
            # whole sub-steps and is answered away from their bookkeeping: it is the
            # SAME slot consulted a second time, so running it through the position
            # roll-over below would read as "a new step opened" and clear a fused
            # pair's commitment in the middle of the step that made it.
            return self._dispatch_far_fill(owner)
        sync_owner = SYNC_PASS_OWNERS.get(slot)
        if sync_owner is not None:
            # THE MAGNETIC HALF-STEP'S OWN NAME FOR A SLOT IT SHARES WITH ``step``.
            # Answered off the SAME plan and through the SAME body below — the
            # commitment bookkeeping, the pair clause and the launch counters are the
            # ones ``step`` would have got, because this site IS ``step``'s magnetic
            # half repeated. All this branch does is decide whether the plan may be
            # asked at all; see :meth:`_sync_path_holds`.
            if sync_owner not in self.slots or not self._sync_path_holds(sync_owner):
                return False
            slot = sync_owner
        if slot not in self.slots:
            return False
        plan = self.step_plan.plans[slot]

        # A COMMITMENT LASTS ONE STEP AND NOT A CONSULT LONGER. The driver walks its
        # consults in ``slots`` order exactly once per ``step()`` (driver.py:3186),
        # so a consult whose position does not ADVANCE opens a new step, and anything
        # still standing was left behind by a step that reached a leading consult but
        # never its absorbed one. That is reachable without anything exotic — an
        # exception between the two sites (a source ``inject``, a symmetry fill, a
        # ``DepositNotRepairable``), or a caller driving the seam itself.
        #
        # A LEAKED COMMITMENT SKIPS A SUB-STEP ENTIRELY, which is worse than the
        # double-apply the clause below exists to stop: the next step's leading
        # consult refuses on a genuinely lapsed licence and the driver runs the ARRAY
        # curl, then the absorbed consult answers True off the stale entry and runs
        # only the sentinel, so the constitutive half never happens on either path.
        # Reproduced against this method, with an arming control, before this reset.
        position = self.slots.index(slot)
        if position <= self._last_position[0]:
            self._committed.clear()
        self._last_position[0] = position

        # A FUSED PAIR'S SLOTS ANSWER TOGETHER, and the licence cannot separate
        # them. A pair owns the curl slot and the constitutive slot; the launch at the
        # FIRST consult performs BOTH halves, and the later consult holds a sentinel
        # (``NoopPlan``) or the deposit repair (``TrailingRepairPlan``). If the policy
        # licence lapsed in between, the plain rule would answer False here and the
        # driver would run the array ``update_E``/``update_H`` ON TOP of the half
        # already computed. ``_apply_constitutive_pml`` ACCUMULATES rather than
        # assigns, so that is not a redundant recomputation but a WRONG ANSWER: the
        # field gains an extra ``(kps - kms) * source``, non-zero wherever the PML is.
        # Measured on a stub plan before this clause existed.
        #
        # Refusing the FIRST consult is always safe -- nothing has run. Refusing a
        # later one after the first committed is not, so once the leading plan has run
        # the rest of the pair dispatches regardless of the licence. The lapse is still
        # counted and announced by the leading slot's own check; what it must not do is
        # un-run a launch.
        #
        # BOTH INSTALLED SHAPES ARE COVERED, which is the correction of 2026-08-29:
        # the clause used to read ``absorbed_by`` as "I am the absorbed slot", and in
        # the deposit-repair shape the LEADING slot carries one too
        # (``deposit_repair.py:683``). So every repair-shaped pair — 38 of the 55 the
        # released composition actually runs, 27 on the D seam and 11 on the B seam —
        # took the plain rule and was open to exactly the double-apply above.
        # :meth:`_pair_of` groups a pair's slots by :func:`_pair_identity`, which
        # answers the same integer for every slot of a pair in EITHER shape; the
        # pair's LEADING slot is simply its first in ``slots``, which is
        # ``STEP_ORDER`` and therefore the driver's own consult order.
        pair = self._pair_of(slot)
        committed = len(pair) > 1 and slot != pair[0] and pair[0] in self._committed
        if not committed and not self._policy_still_licenses_this(slot):
            return False
        if slot == "update_P":
            # THE ONE ASYMMETRIC SLOT, AND THE TWO TABLES SPELL IT DIFFERENTLY.
            #
            # On the Triton table it holds the LIST of per-susceptibility plans, and
            # each takes ``fields.drive_field`` — MEEP's ``w``, not a stored-E
            # reader. The two agree exactly outside the absorber, so taking E
            # instead passes every no-PML case and is wrong only under PML.
            #
            # ON THE METAL TABLE IT HOLDS ONE PLAN, whose ``run()`` takes a contract
            # it resolved at plan build and no argument here. MEASURED 2026-09-10:
            # iterating the slot and calling ``entry.run(drive)`` raised
            # ``KeyError("this ADE plan holds no (<bound method Fields.drive_field
            # ...>, True, 'float32') variant")`` on ``dispersive_2d`` and
            # ``folded_dispersive_2d`` — on the DISPATCH path, where this module's
            # own docstring says exceptions PROPAGATE, so it is a raised error in the
            # middle of a step rather than a refusal. With the branch below both
            # cases are byte-identical to 192 steps.
            #
            # THE BRANCH IS ON THE SHAPE, NOT ON THE TABLE NAME, for the same reason
            # the split-fill branch below reads ``run_near`` rather than a label: the
            # object states its own contract, and after
            # ``wrap_for_residency`` the Metal slot holds a ``SyncedPlan``, which is
            # a single object either way. ``self.table`` decides only what to HAND
            # the entries, which is the half a shape cannot answer.
            #
            # THE HAND-CUDA TABLE TAKES THE DRIVE FIELD TOO, and its slot holds a
            # single object rather than a list: an E->P pair leaves a ``NoopPlan``
            # there and a three-slot weld leaves the triple's TRAILING half, both of
            # which accept ``*args`` and both of which advanced (or deliberately
            # declined to advance) the polarizations inside their own group. So the
            # branch that matters is METAL's — the one table whose ``update_P`` plan
            # resolved its contract at plan build and takes no argument here.
            entries = plan if isinstance(plan, (list, tuple)) else (plan,)
            if self.table == METAL_TABLE:
                for entry in entries:
                    entry.run()
            else:
                drive = fields.drive_field
                for entry in entries:
                    entry.run(drive)
            self._ran(slot, pair)
            return True
        if slot in FAR_FILL_PASSES and hasattr(plan, "run_near"):
            # THE SPLIT FILL, and the branch is on the PLAN, not on the arm label.
            # ``FoldedMirrorGhostFillComplexPlan`` states in its own docstring that a
            # composed plan must call ``run_near``/``run_far`` separately because the
            # driver runs ``zero_metal_*`` between the two passes and complex
            # multiplication is not associative; ``MirrorGhostFillPlan`` has no
            # ``run_near`` and fuses the two, which it is correct to do (±1 * float32
            # is exact). Reading the capability rather than ``selected[slot]`` means a
            # third fill arm gets the contract it declares rather than the contract a
            # label table remembered for it.
            plan.run_near()
            self._ran(slot, pair)
            return True
        plan.run()
        self._ran(slot, pair)
        return True

    def _dispatch_far_fill(self, slot: str) -> bool:
        """Did ``slot``'s FOLDED-FAR pass run? The second half of a fill sub-step.

        Reached from :meth:`dispatch` when the driver consults the far pass by its
        own name — ``if fast is None or not
        fast.dispatch("fill_folded_far_ghosts_B", fields):
        fill_folded_far_ghosts_B(fields)`` — a site that sits AFTER ``zero_metal_B``,
        the pass ``deposit_repair`` relies on running unconditionally behind no
        consult, which is why the fill sub-step needs two consult sites rather than
        one wider one. The identity guard has already run in :meth:`dispatch`.

        THE CONTRACT IS TOTAL FOR THE PASS IT NAMES, like :meth:`dispatch`: True
        means ``fill_folded_far_ghosts_*`` will not run on the array path because a
        kernel already performed it, False means nothing did. There are two ways to
        answer True and they are not the same event:

        * a SPLIT plan (``run_near``/``run_far``) launches its far pass here, in the
          driver's own slot for it, so nothing is reordered at all;
        * a FUSED plan (``MirrorGhostFillPlan``) already wrote the far planes inside
          the near launch, so this answers True having launched nothing — the same
          shape ``update_H`` takes off a ``NoopPlan`` after a fused pair's leading
          launch, and counted the same way (``launches`` counts launch PASSES, so a
          fused fill reads 1 per step and a split fill reads 2).

        THAT SECOND SHAPE MOVES THE FAR PASS FROM AFTER ``zero_metal_*`` TO BEFORE
        IT, and the two orders agree for a reason that is structural rather than
        incidental: ``_zero_metal`` writes stored plane 0 of every axis that is
        metallic AND NOT mirrored, while ``_fill_folded_far_ghosts`` writes the last
        stored plane of a folded PERIODIC axis and reads another plane of the same
        axis — so the two passes only ever meet inside the wall plane, where the
        clear is the last writer in one order and the fill's own SOURCE row is
        already zero in the other. Both land on zero. Stated here as the argument;
        ``test_the_far_ghost_pass_and_the_wall_clear_commute`` measures it on the
        real array path, and the driver-route gate measures it on device.

        ANSWERING FALSE IS ALWAYS SAFE HERE, which is why no commitment machinery
        guards it and why a lapsed policy licence simply falls back: the far fill is
        an idempotent plane write from another plane of the same array, so the array
        pass running on top of a kernel that already did it writes the same bytes.
        That is not true of the constitutive slots, and the asymmetry is the reason
        this method does not go through :meth:`dispatch`'s pair clause.
        """
        if slot not in self.slots:
            return False
        if not self._policy_still_licenses_this(slot):
            return False
        plan = self.step_plan.plans[slot]
        runner = getattr(plan, "run_far", None)
        if runner is None:
            return True  # The near launch carried it; nothing left to run.
        runner()
        self.launches[slot] = self.launches.get(slot, 0) + 1
        return True

    def _sync_path_holds(self, slot: str) -> bool:
        """May ``slot``'s plan run inside ``synchronize_magnetic_fields``?

        Reached only from the by-name channel :data:`SYNC_PASS_OWNERS` opens, and the
        question it answers is not about ``slot`` — it is about the PAIR ``slot``
        belongs to. A fused product occupies every slot of its span and performs all
        of them in one launch, so consulting one of its slots inside the magnetic
        half-step runs the whole span, electric halves included.

        THE RULE IS CONTAINMENT AND IT IS DERIVED, not a list of refused labels: the
        half-step backs up and restores magnetic names only, so a plan may run here
        exactly when every slot it spans is one the half-step itself runs
        (:data:`SYNC_PATH_SLOTS`). That admits every shipped product — each spans one
        field type — and refuses an ``update_H``/``step_D`` weld by its span rather
        than by its name, which is what makes it a rule and not a patch.

        REFUSING IS ALWAYS SAFE HERE, and it is the whole point: False sends the
        driver to the array ``update_H``, which advances the magnetic constitutive
        and nothing else. The count is kept because a refusal on this path means a
        run is paying for a fused product it cannot use half the time, which is a
        thing a gate should be able to read rather than infer.
        """
        outside = tuple(name for name in self._pair_of(slot)
                        if name not in SYNC_PATH_SLOTS)
        if not outside:
            return True
        self.sync_refusals[slot] = self.sync_refusals.get(slot, 0) + 1
        return False

    def _pair_of(self, slot: str) -> Tuple[str, ...]:
        """The slots of the fused pair ``slot`` belongs to — itself alone if unpaired.

        In :attr:`slots` order, which is ``STEP_ORDER`` and therefore the driver's own
        consult order, so ``[0]`` is the LEADING slot and every later entry is one the
        leading launch already performed.

        Built once per plan and cached: this is the per-timestep loop, and
        ``step_plan.plans`` cannot change under a frozen plan — a material mutation
        invalidates the whole plan and the next step re-plans.
        """
        cached = self._pair_slots.get(slot)
        if cached is not None:
            return cached
        plans = self.step_plan.plans
        keys = {name: _pair_identity(plans[name]) for name in self.slots}
        for name in self.slots:
            self._pair_slots[name] = tuple(
                other for other in self.slots if keys[other] == keys[name])
        return self._pair_slots[slot]

    def _ran(self, slot: str, pair: Tuple[str, ...]) -> None:
        """Book one dispatched sub-step, and the commitment if it led a pair.

        Shared by the two return paths of :meth:`dispatch` so ``update_P``'s early
        return cannot be the one shape whose bookkeeping is skipped — the asymmetry
        that slot already carries is the argument it takes, not its accounting.

        NOTHING IS DISCARDED HERE. The commitment is released by the step roll-over
        rather than by the consult that consumes it, so a product spanning more than
        two slots keeps every absorbed slot covered instead of only the next one.
        """
        if len(pair) > 1 and slot == pair[0]:
            self._committed.add(slot)
        self.launches[slot] = self.launches.get(slot, 0) + 1

    def _policy_still_licenses_this(self, slot: str) -> bool:
        """Is the subnormal policy the freeze gated on STILL the one in force?

        THE FREEZE IS NOT THE WHOLE RUN. Everything the gate established — one
        policy across host, CuPy and Triton — is a statement about the moment the
        plan was built, and ``uninstall_subnormal_policy`` is a public function
        that puts CuPy's compiler seam back. Measured before this check existed:
        one process froze a dispatching plan with the policy installed, called
        ``uninstall_subnormal_policy()``, and ``dispatch('step_B', fields)`` kept
        answering True — so already-compiled Triton kernels went on KEEPING while
        every new CuPy compile carried CuPy's own ``-ftz=true`` again. That is the
        shipped split, arrived at from inside a certified plan.

        One integer comparison plus two attribute-free calls, against a kernel
        launch. It answers False rather than raising, which is the array path —
        correct everywhere, and the only answer this seam's contract allows
        besides "the whole sub-step ran".
        """
        licence = self.policy_licence
        if licence is None:
            return True
        policy_module, required, epoch = licence
        if (policy_module.policy_epoch() == epoch
                and policy_module.policy_is_installed()
                and policy_module.get_subnormal_policy() == required):
            return True
        self.licence_lapses[slot] = self.licence_lapses.get(slot, 0) + 1
        now = (policy_module.get_subnormal_policy()
               if policy_module.policy_is_installed() else None)
        _announce_line(
            f"meep_gpu: step path array from here; the {required!r} float32 "
            f"subnormal policy this plan was gated on is no longer in force "
            f"(now {now!r}) — every dispatched sub-step falls back")
        return False

    @property
    def launch_counters(self) -> Dict[str, Dict[str, Any]]:
        """Per slot: how many times a kernel RAN, and how many programs each run asks for.

        THE ONLY THING HERE THAT IS A MEASUREMENT RATHER THAN A DECISION, and it
        exists because every other signal this module publishes is a statement
        about what was PLANNED. ``step_path`` says "fused", ``slots`` says which
        five, ``families`` says on whose certification — and a run in which the
        driver never reached a consult, or reached it with a stale ``Fields`` and
        took the identity guard's False, publishes exactly the same three. The
        vacuous pass this project has caught repeatedly is a leg that fell back and
        therefore matched; a dispatch count of zero on a slot the record calls
        dispatched is that leg, named.

        ``programs`` is the second half and it is not decoration. The plan-time warm
        pass EMPTIES a launch grid on purpose (:func:`_warm_with_empty_grid`) so
        Triton compiles and enqueues nothing, so "a launch happened" and "a launch
        did arithmetic on device" are genuinely different claims here. A slot with
        ``dispatches > 0`` and ``programs == 0`` launched a kernel over no data.
        ``None`` means the plan spells its grid in neither form this module reads
        (``update_P`` holds a LIST of per-susceptibility plans and is reported as
        the list's total), which is unknown, not zero.

        ``programs_per_dispatch`` IS NOT THE WITNESS ON EVERY TABLE, which is why
        ``plan_launches`` is beside it. A Metal ``KernelPlan`` spells no launch grid
        at all (``metal_kernels/plans.py`` __slots__ is ``_args``/``_functions``/
        ``launches``), so on that table the program count is ``None`` BY
        CONSTRUCTION and a gate that read it as its non-vacuity signal would be
        reading "unknown" as evidence. What that table's plan does carry is its own
        launch counter, and that is what ``plan_launches`` reports: the plan's
        count, read from the object that launches, independent of this module's
        ``dispatches`` bookkeeping. Two counters that must agree beat one counter
        that cannot be wrong.

        ``backend`` NAMES WHICH TABLE'S KERNEL THE COUNT IS ABOUT, and on a merged
        NVIDIA step that is not a detail: both tables launch real kernels over the
        same arrays, so a route gate proving "the fused product ran" has to be able
        to say WHOSE, and a mis-attribution here would be invisible to every byte
        comparison. It falls back to :attr:`table` on a single-table plan, which is
        every plan the three family gates and the tests build.
        """
        counters: Dict[str, Dict[str, Any]] = {}
        for slot in self.slots:
            plan = self.step_plan.plans.get(slot)
            entries = plan if isinstance(plan, (list, tuple)) else [plan]
            programs: Optional[int] = 0
            for entry in entries:
                grid = _launch_grid_of(entry)
                if grid is None:
                    programs = None
                    break
                programs += int(grid[0])
            counters[slot] = {
                "arm": self.arms.get(slot),
                "backend": self.backends.get(slot, self.table),
                "dispatches": int(self.launches.get(slot, 0)),
                "programs_per_dispatch": programs,
                "plan_launches": _plan_launches_of(plan),
            }
        return counters

    def report(self) -> Mapping[str, Any]:
        """This freeze's dispatch artifact — the answer to "why did I get what I got".

        A COPY carrying live :attr:`launch_counters`, not the frozen record itself.
        The record is emitted (and appended to ``MEEP_GPU_DISPATCH_LOG``) at the
        freeze, when every counter is zero by construction, so the counts have to be
        read from the plan at the moment the question is asked rather than off the
        line the log already wrote.
        """
        report = dict(self.record, launch_counters=self.launch_counters,
                      licence_lapses=dict(self.licence_lapses),
                      sync_refusals=dict(self.sync_refusals),
                      table=self.table)
        # THE SYNC COUNTERS ARE A MEASUREMENT LIKE THE LAUNCH ONES, and they belong
        # beside them for the same reason: the record says a residency bracket was
        # INSTALLED, and only a count says it RAN. A dispatching Metal plan whose
        # syncs are zero is a composition that read the device's stale mirrors, which
        # no byte comparison of the final state can distinguish from a fallback.
        residency = self.residency
        if residency is not None:
            report["residency_syncs"] = {
                "in": getattr(residency, "syncs_in", None),
                "out": getattr(residency, "syncs_out", None),
            }
        return report

    def describe(self) -> str:  # One log/test line saying what the plan dispatches.
        covered = ", ".join(self.slots) if self.slots else "nothing"
        arms = ", ".join(sorted(set(self.arms.values()))) or "none"
        return f"FastPathPlan(dispatches=[{covered}], arms=[{arms}])"


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

#: The last record written, for ``driver.fast_path_report()`` when no log path is
#: configured. One process-wide slot: a freeze overwrites it, and the file log is
#: the append-only history.
_LAST_RECORD: Optional[Mapping[str, Any]] = None

#: Which one-line stderr announcements this process has already made, so a sweep
#: that builds a thousand drivers says it once rather than a thousand times.
_ANNOUNCED: set = set()


def last_dispatch_report() -> Optional[Mapping[str, Any]]:
    """The most recent configuration freeze's dispatch artifact, or None."""
    return _LAST_RECORD


def reset_dispatch_announcements() -> None:
    """Forget which stderr lines were printed. For tests and long-lived hosts."""
    _ANNOUNCED.clear()


#: Which ledger each table's certifications are read out of. One mapping rather
#: than a branch per reader, so a third table is a row.
TABLE_LEDGERS: Mapping[str, str] = {
    "triton": "triton_kernels",
    CUDA_TABLE: "cuda_kernels",
    METAL_TABLE: "metal_kernels",
}


def _fingerprints(table: str = "triton") -> Mapping[str, Any]:
    """That table's ``fingerprints.json``, read once per process. Never raises.

    ONE READER, TWO LEDGERS. A record that quotes a gate has to quote it out of the
    ledger the gate was written to, and the two are genuinely different files with
    different keys — a Metal arm's gate key resolves to nothing in the Triton ledger
    and would be reported as an unmapped arm, which is the record-forgery shape this
    lookup exists to make visible rather than to produce.
    """
    ledger = TABLE_LEDGERS.get(table)
    if ledger is None:
        return {"_unreadable": f"no ledger is recorded for the {table!r} table"}
    if ledger not in _FINGERPRINTS:
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ledger, "fingerprints.json")
            with open(path, "r", encoding="utf-8") as handle:
                _FINGERPRINTS[ledger] = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - an unreadable record is not a fatal one
            _FINGERPRINTS[ledger] = {"_unreadable": repr(exc)}
    return _FINGERPRINTS[ledger]


_FINGERPRINTS: Dict[str, Mapping[str, Any]] = {}


def validated_triton_versions() -> Tuple[str, ...]:
    """The Triton versions the recorded gates ran on. Empty when unreadable."""
    versions = _fingerprints().get("validated_triton_versions") or ()
    try:
        return tuple(str(v) for v in versions)
    except Exception:  # noqa: BLE001
        return ()


#: The sweep that re-ran the bound gates against the current planner bytes. Its
#: census carries a per-family exit code and its ``blocked_and_why`` /
#: ``not_re_run_and_why`` name the gates whose re-run did NOT come back clean —
#: facts an artifact that quotes those gates has to carry, because a reader has no
#: other way to learn that the provenance they are being shown is qualified.
RECERT_SWEEP_KEY = "planner_integration_recert_2026-08-14"


def _arm_certification_map(table: str) -> Mapping[str, Tuple[str, str]]:
    """That table's ``arm label -> (family, ledger key)`` map.

    THE METAL MAP IS NOT IN THIS FILE and the import is function-local for the
    reason :mod:`meep_gpu.metal_dispatch` states in its own docstring: a Metal
    release-row edit must not re-drift the Triton ``driver_dispatch`` record, which
    pins exactly ``driver.py``, ``fastpath.py`` and ``fields.py``. What this module
    owns is the LOOKUP; what that module owns is the ROWS.
    """
    if table == METAL_TABLE:
        from . import metal_dispatch  # noqa: PLC0415

        return metal_dispatch.ARM_CERTIFICATION
    if table == CUDA_TABLE:
        from . import fastpath_cuda  # noqa: PLC0415

        return fastpath_cuda.CUDA_ARM_CERTIFICATION
    return ARM_CERTIFICATION


def _table_for_label(arm: Any, table: str) -> str:
    """Which ledger answers for THIS arm, given the table the plan was composed on.

    A NAMESPACED LABEL ANSWERS FOR ITSELF and a bare one takes the composing table's
    answer. That asymmetry is the whole of what a merged plan needs: under
    ``MEEP_GPU_BACKEND_PREFERENCE=cuda`` the primary table is ``cuda`` and the plan
    still holds Triton arms in the slots the CUDA products did not take, so keying
    the lookup on the PLAN's table would quote the CUDA ledger for a Triton arm — a
    family name beside a key that resolves to nothing, which is exactly the
    record-forgery shape :func:`_certification_for`'s miss branch exists to make
    visible rather than to produce.
    """
    return _table_of(arm) or table


def _certification_for(arm: str, table: str = "triton", *,
                       capability: Optional[str] = None) -> Dict[str, Any]:
    """What certified this arm: family, gate, the gate's recorded facts, AND its caveats.

    Three things beyond the gate's own ``recorded_utc``/``host``/``purpose``, each
    closing a way this record could have read cleaner than the evidence behind it:

    * the FAMILY's own step budget and pass counts, so a family is not credited
      with a campaign leg it never ran;
    * the RE-RUN status from :data:`RECERT_SWEEP_KEY` — two dispatchable arms
      ('conductive PML', 'no-PML') cite gates whose re-run against the current
      planner bytes exited rc=1 and is recorded as blocked, and one ('PML',
      'ordinary') cites a gate the sweep did not re-run at all;
    * the DIGESTS the gate recorded for its probe and adapter, because the
      not-re-run gate's are known to have drifted from the live files and an
      artifact that drops the fields cannot show it.
    """
    table = _table_for_label(arm, table)
    if table == CUDA_TABLE:
        # THE CUDA ENTRY IS BUILT BY THE MODULE THAT OWNS THE ROWS, in the same
        # shape this function returns. Delegating rather than branching inside is
        # what keeps a CUDA release-row edit off this file — see
        # :mod:`meep_gpu.fastpath_cuda` for what that costs and what it saves.
        from . import fastpath_cuda  # noqa: PLC0415

        return fastpath_cuda.certification_for(arm, capability=capability)
    family, gate = _arm_certification_map(table).get(arm, ("unmapped", "unmapped"))
    ledger = TABLE_LEDGERS.get(table, "triton_kernels")
    entry: Dict[str, Any] = {
        "family": family,
        "gate": gate,
        # THE POLICY IS THE TABLE'S, and it has to be, because the two tables
        # certify under opposite values: an entry that reported ``keep`` beside a
        # Metal gate cut under ``flush`` would be a false claim about the bytes.
        "certification_policy": TABLE_SUBNORMAL_POLICY.get(
            table, CERTIFICATION_SUBNORMAL_POLICY),
        "step_budget": None,
    }
    record = _fingerprints(table).get(gate)
    if not isinstance(record, Mapping):
        # SAY THAT THE LOOKUP FAILED rather than emitting a family name beside a
        # key that resolves to nothing. That asymmetry — some arms carrying
        # recorded_utc/host/purpose and others carrying nothing, with no field to
        # tell them apart — is the defect :data:`FAMILY_RECERT_GATE` records having
        # been fixed once, when nine families reached a user's artifact with a dead
        # path and no way to see that it was dead.
        #
        # EVERY ROW IN :data:`ARM_CERTIFICATION` RESOLVES, and
        # ``test_every_recorded_certification_points_at_a_readable_record`` holds
        # it there, so this branch is reached only by an arm with NO row — the
        # ``("unmapped", "unmapped")`` default, which is what a wired-but-pending
        # fused label gets. Naming the miss is what keeps that legible instead of
        # letting a record read as though a gate had been consulted.
        entry["gate_record"] = (
            f"no record under {gate!r} in {ledger}/fingerprints.json; this "
            "arm names no certification this module can quote. A fused label in "
            "PENDING_DEVICE_GATE_ARMS is the expected case, and that map carries "
            "the gate artifact its release was cut from")
    if isinstance(record, Mapping):
        # THE RUN THIS DEVICE'S ARCHITECTURE WAS CERTIFIED BY, not whichever run the
        # entry happens to hold: the ledger keeps one run per compute capability, and
        # quoting another architecture's host and artifact beside this plan would be
        # the label/evidence mismatch the container exists to close. A capability with
        # no live run here quotes no run fields and says so; ``capabilities_live`` is
        # always reported, so a reader can see what the entry does have.
        live = live_capabilities(record)
        entry["capabilities_live"] = list(live)
        run: Mapping[str, Any] = {}
        if capability is None:
            entry["run_record"] = (
                "this host's compute capability could not be read, so no run of this "
                "gate is quoted; the capabilities it has live runs for are above")
        elif capability in live:
            run = record[RUNS][capability]
            entry["capability"] = capability
        else:
            entry["run_record"] = (
                f"this gate has no live run on compute capability {capability}: its "
                f"live capabilities are {list(live)}. A plan that dispatched here was "
                f"admitted by the opt-in, not by a record")
        for key in ("recorded_utc", "host", "purpose", "records", "step_budget",
                    "status", "subnormal_policy"):
            if key in run:
                entry[key] = run[key]
            elif key in record:
                entry[key] = record[key]
        for key in ("probe_sha256", "adapter_sha256"):
            if key in record:
                entry.setdefault("recorded_digests", {})[key] = record[key]
        detail = run.get("families") if "families" in run else record.get("families")
        if isinstance(detail, Mapping) and isinstance(detail.get(family), Mapping):
            family_record = detail[family]
            for key in ("run_id", "rc", "verdict", "step_budget",
                        "certified_under_subnormal_policy", "source_sha256_at_recert"):
                if key in family_record:
                    entry[key] = family_record[key]
    if entry.get("step_budget") is None:
        entry["step_budget"] = (
            f"not stated per family by {gate}; see that record's legs and the "
            "campaign-wide certification_budget at the top of this artifact")

    # THE RE-CERT SWEEP IS THE TRITON LEDGER'S, and it is read out of the arm's OWN
    # ledger rather than out of that one: a Metal ledger carries no such key, so the
    # block below is skipped for that table by the lookup itself instead of by a
    # branch. Quoting the Triton sweep's verdict beside a Metal gate would be
    # crediting one campaign with another's re-run.
    sweep = _fingerprints(table).get(RECERT_SWEEP_KEY)
    if isinstance(sweep, Mapping):
        census = sweep.get("gpu_placement_census")
        if isinstance(census, Mapping) and isinstance(census.get(family), Mapping):
            rc = census[family].get("rc")
            re_run: Dict[str, Any] = {"sweep": RECERT_SWEEP_KEY, "rc": rc}
            if rc != 0:
                blocked = sweep.get("blocked_and_why")
                if isinstance(blocked, Mapping) and gate in blocked:
                    re_run["blocked_and_why"] = blocked[gate]
                re_run["read_this_as"] = (
                    "the gate this arm cites did NOT re-run clean against the current "
                    "planner bytes; the recorded cause and the legs that did reproduce "
                    "are in that gate's own record")
            entry["last_re_run"] = re_run
        not_re_run = sweep.get("not_re_run_and_why")
        if isinstance(not_re_run, Mapping) and gate in not_re_run:
            entry["not_re_run"] = not_re_run[gate]
    return entry


def _subnormal_block(table: str = "triton") -> Dict[str, Any]:
    """The resolved policy verbatim, plus which policy the families were cut under.

    READ BEFORE THE GATE RUNS, which is why it is worth keeping separate from it:
    the stamp taken here records what this process would resolve to on its own,
    and :func:`_subnormal_gate` may then drive it somewhere else. A stamp taken
    only afterwards could not tell a run that already kept from one dispatch moved.

    PER TABLE, because the two certify under opposite values and this block is what
    a reader compares a run's own resolution AGAINST. Recording the Triton constant
    on a Metal record would say the run was measured against ``keep`` when every
    Metal weld was cut under ``flush`` — see :data:`TABLE_SUBNORMAL_POLICY`.
    """
    block: Dict[str, Any] = {
        "certification_policy": TABLE_SUBNORMAL_POLICY.get(
            table, CERTIFICATION_SUBNORMAL_POLICY),
        "governed_executors": list(TABLE_GOVERNED_EXECUTORS.get(
            table, GOVERNED_EXECUTORS)),
        "note": CERTIFICATION_SUBNORMAL_NOTE,
    }
    try:
        from .subnormal_policy import policy_stamp  # noqa: PLC0415

        block["stamp"] = policy_stamp()
    except Exception as exc:  # noqa: BLE001 - a stamp that will not read is not a refusal
        block["stamp_error"] = repr(exc)
    return block


#: The executors that must all be attained under one policy before a kernel may
#: run beside an array call. Not a subset: the host is in it because the array
#: path's NumPy/CuPy reductions and the FPU-sensitive host arithmetic sit in the
#: same step as the kernels, and a keeping host beside a flushing CuPy is the same
#: defect one layer down (measured on the bfast composition: uniform keep is
#: 80/80 bit-identical, host-keeps-Triton-flushes diverges at step 1).
GOVERNED_EXECUTORS: Tuple[str, str, str] = ("host", "cupy", "triton")

#: THE SAME ARGUMENT, PER TABLE. Each row names every executor that runs float32
#: arithmetic inside one step of that table's composition, host included, and the
#: rule below drives ALL of them to that table's :data:`TABLE_SUBNORMAL_POLICY`
#: value or refuses the plan by name.
#:
#: THE METAL ROW HAS NO CuPy AND NO TRITON IN IT, and that is a statement about the
#: engine rather than a shortcut: that table composes over a NUMPY engine, so
#: neither device library is in the step at all, and probing for them would either
#: govern an executor that runs nothing (an install leg reporting "nothing to
#: govern") or, worse, let an unrelated CuPy that happens to be importable decide a
#: refusal. ``mps`` replaces them and is the executor ``metal_kernels/subnormal``
#: answers for.
#: THE CUDA ROW IS THE TRITON ROW, and that is a statement about the STEP rather
#: than a shortcut. A merged plan can hold Triton arms and hand-CUDA products in the
#: same step over the same CuPy arrays, so every executor either table drives is in
#: every such step: the host, CuPy, and Triton whenever it is importable (the probe
#: in :func:`_governed_executors` is what decides that, not this row).
TABLE_GOVERNED_EXECUTORS: Mapping[str, Tuple[str, ...]] = {
    "triton": GOVERNED_EXECUTORS,
    CUDA_TABLE: GOVERNED_EXECUTORS,
    METAL_TABLE: ("host", "mps"),
}

#: ``subnormal_policy.policy_epoch()`` as it stood right after DISPATCH installed
#: the policy, or ``None``. Compared against the live epoch rather than kept as a
#: bare flag so it cannot outlive the install it describes: an uninstall bumps the
#: epoch, and this stops matching without anything having to notice.
_POLICY_INSTALLED_BY_DISPATCH: Optional[int] = None


def _importable(name: str) -> bool:
    """``find_spec``, with a RAISE read as PRESENT — the fail-CLOSED direction.

    ``importlib.util.find_spec`` RAISES ``ValueError`` on a module object whose
    ``__spec__`` is ``None``, and a module that is in ``sys.modules`` well enough
    to be found and interrogated is a module that RUNS. Reading that raise as
    "absent" is what the previous revision did, and it is the wrong direction: see
    :func:`_governed_executors`.
    """
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - see the docstring: a raise means present
        return True


def _governed_executors(xp: Any, table: str = "triton") -> Tuple[str, ...]:
    """The executors this freeze must drive to ONE policy, from facts it established.

    NOT RE-DERIVED, and that is the whole of this function. By the time the gate
    runs, rung 3 has required ``xp.__name__ == "cupy"`` and rung 4 has IMPORTED
    Triton and matched its version against the recorded validated list — so both
    device executors are present and about to run, and a second opinion that
    disagrees with those two rungs is wrong by construction.

    ON THE METAL TABLE THERE IS NOTHING LEFT TO DERIVE, so nothing is: the row in
    :data:`TABLE_GOVERNED_EXECUTORS` is returned whole. Rung 4M has already required
    torch, an MPS build, an available MPS device and ``compile_shader`` — that IS
    the presence of the ``mps`` executor, established by refusals rather than by a
    probe — and the host is unconditional on every table. Probing for CuPy or Triton
    here would be the fail-open this docstring's next paragraph records the cost of,
    in the other direction: an importable CuPy that this composition never touches
    would be governed, and a refusal to attain a policy on it would refuse a plan
    that has no CuPy in it.

    THE PREVIOUS REVISION ASKED ``importlib.util.find_spec`` AND FAILED OPEN.
    ``find_spec`` raises ``ValueError`` on a module whose ``__spec__`` is ``None``,
    that raise was read as "absent", and an executor dropped from ``governed`` can
    never trip the ``ungoverned`` refusal below — because that list is built OVER
    ``governed``. Measured on this laptop through the shipped entry point: a Triton
    that rung 4 had just imported and certified ("triton 3.1.0 certified" on
    stderr) produced ``governed_executors == ['host']``,
    ``executor_report('triton') == {}``, ``refused_because is None`` and a
    DISPATCHED verdict — dispatch proceeding with Triton on its own default, which
    is the exact split the gate exists to prevent. The same hole swallowed the
    CuPy leg for a backend object accepted at rung 3.

    THE DEFAULT IS THEREFORE TO GOVERN. An executor that turns out to run nothing
    costs one install leg reporting "nothing to govern" (both installers answer
    that case themselves, attained and not installed); an executor wrongly left
    OUT costs the split, silently, on every step of the run.
    """
    declared = TABLE_GOVERNED_EXECUTORS.get(table, GOVERNED_EXECUTORS)
    if table != "triton":
        return tuple(declared)
    governed: List[str] = ["host"]
    # The backend object rung 3 accepted IS the CuPy this run computes with — the
    # installer is handed that same object (``cupy=xp``) rather than importing a
    # second one, so presence and identity are one fact here.
    if getattr(xp, "__name__", "") == "cupy" or _importable("cupy"):
        governed.append("cupy")
    # Rung 4's ``import triton`` put it here. ``sys.modules`` is the fact; the
    # spec lookup is only the fallback for an executor no rung reached.
    if sys.modules.get("triton") is not None or _importable("triton"):
        governed.append("triton")
    return tuple(name for name in declared if name in governed)


def _subnormal_gate(record: Dict[str, Any], xp: Any,
                    table: str = "triton") -> Optional[str]:
    """Put every executor this table drives on ONE policy before anything compiles.

    THE RULE: dispatch never runs two executors under disagreeing float32
    subnormal policies. It is ONE RULE with a per-table value, not one rule per
    table: :data:`TABLE_SUBNORMAL_POLICY` names what each table certified under and
    :data:`TABLE_GOVERNED_EXECUTORS` names who must attain it, and everything below
    verifies the same things about both. The paragraphs that follow are the Triton
    table's argument, recorded where it was measured; the Metal table's is at
    :data:`TABLE_SUBNORMAL_POLICY` and comes out the other way for the same reason
    — the value has to be attainable on every executor in the step.

    As shipped the Triton table's two executors DO disagree — CuPy appends ``-ftz=true``
    to every NVRTC compile (``cupy/cuda/compiler.py:552``) and Triton emits no ftz
    attribute at all — and nothing in the package called
    ``install_subnormal_policy``, so the split was the DEFAULT rather than an edge
    case. It is measured at instruction level (19/19 CuPy f32 PTX instructions
    carrying ``.ftz`` against 0/186 for Triton on the same sub-step) and end to
    end (6 of 7 dispatching cases diverging inside the certified step window).

    THE POLICY IS THE CERTIFICATION'S, NOT THE RUN'S, and that is a deliberate
    choice against the more obvious one:

    * the nine families and the driver-route gate's eight byte-identical cases
      were ALL cut under :data:`CERTIFICATION_SUBNORMAL_POLICY`; installing the
      run's own resolution would dispatch kernels under a policy no gate ran;
    * the run's resolution is ``flush`` on x86 (that is what MEEP does there), and
      ``flush`` is measurably UNATTAINABLE in a process that has already imported
      CuPy: its CUB reduction accelerators are fixed at import and keep
      subnormals, so flush needs ``CUPY_ACCELERATORS=''`` set before that import.
      Choosing the run's resolution would therefore refuse dispatch on exactly the
      hosts dispatch exists for;
    * ``keep`` is IEEE-754 and attainable on every host, and it needs no
      environment set before an import — only a policy-private ``CUPY_CACHE_DIR``,
      because CuPy's cache key is computed above the seam the strip installs at.

    WHAT IT COSTS, recorded rather than hidden: on a host whose own resolution is
    ``flush`` this moves the whole run — the array path included — onto IEEE-754.
    That is a real change to the run's arithmetic, it is what makes the dispatched
    bytes attributable to a gate, and ``overrode_run_resolution`` plus the stderr
    line say so on every such run.

    Returns ``None`` when dispatch may proceed, else the refusal reason. Never
    raises: an install that will not come up is a named refusal and the array
    path, like every other rung.

    A REFUSAL PUTS ``CUPY_CACHE_DIR`` BACK, and that undo is owned here because
    the repoint is done here. Measured: with the install refused, the variable was
    left pointing at a freshly created keep-policy cache directory for the whole
    life of the process — while the strip was NOT installed, so every subsequent
    CuPy compile carried CuPy's own ``-ftz=true`` straight into the directory the
    keep policy claims as private. CuPy's cache key is computed ABOVE the strip
    seam, so the directory NAME is the only separator there is: the next process
    that derives the same deterministic path is served those flushed binaries
    under the keep policy's name, with no counter to notice (a disk-cache hit
    makes no NVRTC call, so ``ftz_removed`` simply stays low).
    """
    from . import subnormal_policy as policy_module  # noqa: PLC0415

    refusal = _drive_one_policy(record, xp, policy_module, table)
    gate = record.setdefault("subnormal", {}).setdefault("gate", {})
    if refusal is not None and not policy_module.policy_is_installed():
        # ONLY when nothing was installed. A refusal that comes AFTER a successful
        # install must LEAVE the pointer moved: the strip is live from then on, so
        # this process's CuPy compiles are stripped and belong in the policy's
        # directory. Putting the pointer back there would be the poisoning, not the
        # fix — it would send stripped binaries into the caller's shared cache.
        _restore_cupy_cache_dir(gate)
    return refusal


def _restore_cupy_cache_dir(gate: Dict[str, Any]) -> None:
    """Undo :func:`point_cupy_cache_at_keep_policy` after a refusal that installed nothing."""
    cache = gate.get("cupy_cache_dir")
    if not isinstance(cache, dict) or not cache.get("changed"):
        return
    if os.environ.get("CUPY_CACHE_DIR") != cache.get("after"):
        # The repoint refused before setting the variable (unwritable, or a
        # directory of unknown provenance). There is nothing to put back, and
        # recording a restore that did not happen is the kind of artifact claim
        # this round exists to stop.
        return
    before = cache.get("before")
    if before is None:
        os.environ.pop("CUPY_CACHE_DIR", None)
    else:
        os.environ["CUPY_CACHE_DIR"] = before
    cache["restored_to"] = before
    cache["restored_because"] = (
        "the configuration was refused with no policy installed, so this process "
        "compiles CuPy kernels with CuPy's own '-ftz=true' — those binaries must "
        "not land in the keep policy's private cache directory, whose name is the "
        "only thing separating it (CuPy's cache key is computed above the strip "
        "seam)")


def _drive_one_policy(record: Dict[str, Any], xp: Any, policy_module: Any,
                      table: str = "triton") -> Optional[str]:
    """The gate proper. :func:`_subnormal_gate` wraps it to own the cache-dir undo.

    ``table`` decides two things and nothing else: WHICH policy is required
    (:data:`TABLE_SUBNORMAL_POLICY`) and WHICH executors must attain it
    (:data:`TABLE_GOVERNED_EXECUTORS`). Every verification below — attained per
    executor, the policy in force equalling the required one, the host FPU read
    taken fresh at the freeze — is the same on both tables, because the rule they
    enforce is the same rule.
    """
    global _POLICY_INSTALLED_BY_DISPATCH

    required = TABLE_SUBNORMAL_POLICY.get(table, CERTIFICATION_SUBNORMAL_POLICY)
    declared = TABLE_GOVERNED_EXECUTORS.get(table, GOVERNED_EXECUTORS)
    block = record.setdefault("subnormal", {})
    governed = _governed_executors(xp, table)
    gate: Dict[str, Any] = {
        "rule": "dispatch never runs two executors under disagreeing float32 "
                "subnormal policies",
        "required_policy": required,
        "why_this_policy": (
            "the policy every dispatchable family was certified under, and the one "
            "the driver-route gate's eight byte-identical cases ran under"
            if table == "triton" else
            f"the policy every {table} weld was cut under, and the only one "
            f"attainable on every executor this table drives "
            f"({', '.join(declared)})"),
        "table": table,
        "governed_executors": list(governed),
        "not_installed_in_this_process": [name for name in declared
                                          if name not in governed],
        "variable": SUBNORMAL_INSTALL_SWITCH,
        "install_permitted": subnormal_install_enabled(),
        "installed_by": None,
        "attained": {},
    }
    block["gate"] = gate
    stamp_before = block.get("stamp") or {}
    would_resolve_to = stamp_before.get("resolved")
    gate["run_would_resolve_to"] = would_resolve_to

    if policy_module.policy_is_installed():
        installed = policy_module.get_subnormal_policy()
        # WHICH FREEZE, not just "not this one". The driver re-plans after every
        # ``invalidate_fast_path()``, so on a long run the SECOND and later freezes
        # of a process dispatch itself put on a policy used to be recorded as
        # "something other than this freeze" — the artifact crediting the caller
        # for a policy dispatch installed, on the majority of the records a run
        # emits. ``probe_dispatch_policy_gate`` reads this same string as its
        # evidence that the CALLER installed it. The EPOCH is what makes the
        # answer survive an uninstall: a flag alone would still read True for a
        # policy this process later dropped and a caller then re-installed.
        ours = (_POLICY_INSTALLED_BY_DISPATCH is not None
                and _POLICY_INSTALLED_BY_DISPATCH == policy_module.policy_epoch())
        gate["installed_by"] = (
            "dispatch, at an earlier configuration freeze in this process"
            if ours else "something other than this freeze, before it planned")
        gate["caller_installed_policy"] = installed
        if installed != required:
            return (f"this process already installed the {installed!r} float32 "
                    f"subnormal policy and every dispatchable family was certified "
                    f"under {required!r}. The policy cannot be swapped in place — "
                    f"binaries compiled under {installed!r} are still reachable, and "
                    f"CuPy's cache key is computed above the seam the strip installs "
                    f"at — so dispatch takes the array path. Install {required!r} "
                    f"before the first kernel compiles, in a fresh process")
    elif not gate["install_permitted"]:
        return (f"{SUBNORMAL_INSTALL_SWITCH}=0 asked dispatch not to install a "
                f"subnormal policy, and none is installed. As shipped the two "
                f"executors disagree — CuPy appends '-ftz=true' to every NVRTC "
                f"compile and Triton emits no ftz attribute — which is measured to "
                f"diverge inside the certified step window. Call "
                f"meep_gpu.install_subnormal_policy({required!r}) before the first "
                f"kernel compiles, or unset {SUBNORMAL_INSTALL_SWITCH} and let "
                f"dispatch install it")
    else:
        if "cupy" in governed:
            # Only when there is a CuPy to cache FOR. The directory is CuPy's own,
            # and minting one in a process that has no CuPy would be a side effect
            # with nothing on the other end of it.
            cache = policy_module.point_cupy_cache_at_keep_policy(cupy=xp)
            gate["cupy_cache_dir"] = cache
            if not cache.get("writable"):
                return (f"the {required!r} policy needs a private CUPY_CACHE_DIR "
                        f"(CuPy's cache key is computed above the strip seam) and "
                        f"{cache['after']!r} could not be created or written: "
                        f"{cache.get('error', 'not writable')}")
            # PROVENANCE, because the token in the directory NAME is all that
            # separates this cache from a flush-policy one, and a name is not
            # evidence about the binaries inside it. An unmarked directory that
            # already holds entries is of unknown origin: it may hold CuPy's own
            # '-ftz=true' binaries, which a keep run would then be served without
            # a single NVRTC call to notice.
            provenance = cache.get("provenance") or {}
            if provenance.get("reasons"):
                return (f"the {required!r} policy's CuPy cache directory "
                        f"{cache['after']!r} is of unknown provenance: "
                        + "; ".join(provenance["reasons"]))
        try:
            # ``cupy`` is deliberately left for the installer to resolve. Handing it
            # ``xp`` was tried and reverted: on a real host ``import cupy`` is the
            # same singleton object rung 3 accepted, so the two are indistinguishable
            # where it matters, and the difference only shows against a backend
            # object that merely NAMES itself cupy — where the installer's own
            # "CuPy is not importable, so it compiles nothing" is the true answer
            # and forcing the stub through the compiler seam would refuse a
            # configuration in which there is nothing to govern.
            policy_module.install_subnormal_policy(required, strict=True,
                                                   executors=governed)
        except Exception as exc:  # noqa: BLE001 - every install failure is a refusal
            gate["install_error"] = repr(exc)
            return (f"installing the {required!r} float32 subnormal policy for "
                    f"dispatch did not come up: {exc}")
        gate["installed_by"] = "dispatch, at the configuration freeze"
        _POLICY_INSTALLED_BY_DISPATCH = policy_module.policy_epoch()

    # VERIFY, whoever installed, AND VERIFY THE STATE RATHER THAN THE REPORT.
    # "install_subnormal_policy returned" and "every executor obeys it" are
    # different claims, and only the second one licenses a kernel: an install can
    # report a policy while one leg was skipped or refused, and a caller may have
    # driven a subset (``executors=("host","cupy")`` leaves Triton on its own
    # default, which is the split with an extra step). The reads below are the
    # policy module's own state — ``policy_is_installed``, ``get_subnormal_policy``
    # and the per-executor reports — not ``policy_stamp()``'s serialization of it,
    # because a stamp is an ARTIFACT and this round exists because an artifact
    # said a policy was covered when no executor had been driven to it.
    executors = {name: policy_module.executor_report(name) for name in governed}
    gate["attained"] = {name: bool(report.get("attained"))
                        for name, report in executors.items()}
    stamp = policy_module.policy_stamp()
    block["stamp"] = stamp
    gate["stamp_after_install"] = {
        "policy": stamp.get("policy"), "resolved": stamp.get("resolved"),
        "installed": stamp.get("installed"), "unattained": stamp.get("unattained"),
        "ftz_removed": stamp.get("ftz_removed"),
        "triton_ptx_audited": (stamp.get("triton") or {}).get("ptx_audited"),
    }
    ungoverned = [name for name, report in executors.items() if not report]
    if ungoverned:
        return (f"the {required!r} subnormal policy was never driven onto "
                f"{', '.join(ungoverned)}: that executor is running on its own "
                f"default while the others carry the policy, which is the split "
                f"this gate exists to prevent")
    in_force = (policy_module.get_subnormal_policy()
                if policy_module.policy_is_installed() else None)
    gate["policy_in_force"] = in_force
    failed = [name for name in governed if not gate["attained"][name]]
    if failed or in_force != required:
        detail = "; ".join(
            f"{name}: " + "; ".join(executors[name].get("reasons")
                                    or ["(no reason recorded)"])
            for name in failed) or (
            "no policy is installed in this process" if in_force is None else
            f"the policy in force is {in_force!r}")
        return (f"the {required!r} float32 subnormal policy was not attained on "
                f"{', '.join(failed) or 'this process'} — {detail}")

    # THE HOST'S FPU, READ NOW, not taken from the record ``install_host_policy``
    # wrote when it ran. Those are different claims the moment anything moves the
    # FPU after the install — ``mp.set_zero_subnormals`` is public MEEP API and a
    # second library setting MXCSR needs no API at all — and the comment above
    # says to verify the STATE rather than the REPORT, which for two of the three
    # executors is all that is available and for this one is not. It costs two
    # float64 multiplies (``backends.subnormals_flushed`` is the same measurement
    # ``install_host_policy`` makes its own verdict from).
    from . import backends  # noqa: PLC0415 - no cupy import; see the package rule

    host_flushing = backends.subnormals_flushed()
    gate["host_flushing_now"] = host_flushing
    if host_flushing != (required == policy_module.FLUSH):
        return (f"the host FPU is flushing={host_flushing!r} at the freeze, which "
                f"is not what the {required!r} policy needs — the install reported "
                f"the host attained, so something moved the FPU after it "
                f"(mp.set_zero_subnormals is public MEEP API, and a second library "
                f"setting MXCSR needs none). A keeping CuPy and Triton beside a "
                f"flushing host is the same split one layer down")

    gate["overrode_run_resolution"] = bool(
        would_resolve_to is not None and would_resolve_to != required)
    if gate["overrode_run_resolution"]:
        gate["consequence"] = (
            f"this run resolved {would_resolve_to!r} on its own (that is what MEEP "
            f"does on this host) and dispatch installed {required!r} on every "
            f"executor, so the ARRAY path's arithmetic moved too. Kill-switch the "
            f"dispatch ({FUSED_KILL_SWITCH}=0) to get the run's own resolution back")
    # The epoch this plan's kernels are licensed under. ``FastPathPlan.dispatch``
    # compares one integer against it on every consult, because everything above
    # this line is a statement about the freeze and a policy can be uninstalled
    # afterwards — see :meth:`FastPathPlan.dispatch`.
    gate["policy_epoch"] = policy_module.policy_epoch()
    return None


# ---------------------------------------------------------------------------
# Per-capability certification records
# ---------------------------------------------------------------------------

#: The key, inside a ledger entry, holding one record per environment the entry's
#: gate ran on. For the NVIDIA tables the environment is a compute capability,
#: spelled by :func:`_normalized_capability`.
RUNS = "runs"

#: The entry fields that name the BYTES a run certified. A run record carries the
#: digest of these (:func:`bound_digest`) and is live only while they are unchanged.
BOUND_FIELDS: Tuple[str, ...] = ("source_sha256", "source_sha256_at_recert", "probe",
                                 "probe_sha256", "adapter", "adapter_sha256")

#: The fields that describe ONE RUN of a weld's gate. They live in that run's record
#: under :data:`RUNS`, never beside the digests.
RUN_FIELDS: Tuple[str, ...] = (
    "artifact_sha256", "log_sha256", "host", "records", "recorded_utc",
    "verdict_read_from", "subnormal_policy", "campaign", "gate_started_utc", "elapsed",
    "_this_recut", "device_policy", "cupy_version", "triton_version", "families",
    "step_budget", "_digests_taken_from_the_checkout")

#: The route-campaign fields of a table's ``driver_dispatch`` record, kept per
#: capability for the same reason.
DISPATCH_RUN_FIELDS: Tuple[str, ...] = (
    "status", "records", "legs", "recorded_utc", "verdict_read_from",
    "subnormal_policy", "arbitration", "cuda_alone_leg", "released_fused_arms",
    "dispatch_by_default_licence")


class CapabilityRecordError(ValueError):
    """A ledger entry whose per-capability records cannot be read or written as asked."""


class CapabilityStale(CapabilityRecordError):
    """A write that would leave other capabilities' records bound to bytes that moved."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(sorted(names))
        super().__init__(
            f"this write changes the bytes the entry binds, which would leave the "
            f"records for {list(self.names)} certifying bytes that no longer ship; "
            f"re-run those capabilities on these bytes, or supersede them by name")


#: What a capability key may be: ``8.6``, ``9.0``, ``12.0``. Checked rather than
#: inferred from :func:`_normalized_capability`, which leaves anything it does not
#: recognise alone — so ``sm_86`` normalises to itself and would otherwise key a record.
_CAPABILITY_KEY = re.compile(r"^[1-9][0-9]*\.[0-9]$")


def _require_capability(value: Any) -> str:
    """``value`` as a capability key, or a named refusal."""
    normalised = _normalized_capability(value)
    if not _CAPABILITY_KEY.match(normalised or ""):
        raise CapabilityRecordError(
            f"{value!r} is not a normalised compute capability (major.minor, as "
            f"{_CAPABILITY_KEY.pattern} spells it)")
    return normalised


def bound_digest(entry: Mapping[str, Any]) -> str:
    """sha256 over the entry's :data:`BOUND_FIELDS`, canonically serialised."""
    bound = {field: entry[field] for field in BOUND_FIELDS if field in entry}
    if not bound:
        raise CapabilityRecordError(
            f"the entry binds no bytes: none of {list(BOUND_FIELDS)} is present")
    text = json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def live_capabilities(entry: Any) -> Tuple[str, ...]:
    """The capabilities whose run record still binds this entry's bytes."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get(RUNS), Mapping):
        return ()
    try:
        current = bound_digest(entry)
    except CapabilityRecordError:
        return ()
    return tuple(sorted(
        capability for capability, run in entry[RUNS].items()
        if isinstance(run, Mapping) and run.get("bound_sha256") == current))


def retired_shape_reasons(entry: Any, run_fields: Sequence[str] = RUN_FIELDS) -> List[str]:
    """Why this entry is not in the per-capability shape. Empty means it is.

    THE SHAPE IS REFUSED BY NAME rather than read both ways. A reader that accepted a
    run field beside the digests would make "which bytes did this run certify" a
    question with two answers, which is the whole defect the container closes.
    """
    reasons: List[str] = []
    if not isinstance(entry, Mapping):
        return [f"not a mapping: {type(entry).__name__}"]
    stranded = sorted(field for field in run_fields if field in entry)
    if stranded:
        reasons.append(f"run fields beside the digests: {stranded}")
    runs = entry.get(RUNS)
    if runs is None:
        reasons.append(f"no {RUNS!r} record")
    elif not isinstance(runs, Mapping):
        reasons.append(f"{RUNS!r} is {type(runs).__name__}, not a mapping")
    else:
        for capability in sorted(runs):
            if not _CAPABILITY_KEY.match(str(capability)):
                reasons.append(f"{RUNS}[{capability!r}] is not a normalised capability")
            elif not isinstance(runs[capability], Mapping):
                reasons.append(f"{RUNS}[{capability!r}] is not a mapping")
            elif "bound_sha256" not in runs[capability]:
                reasons.append(f"{RUNS}[{capability!r}] records no bound_sha256")
    if not any(field in entry for field in BOUND_FIELDS):
        reasons.append(f"binds no bytes: none of {list(BOUND_FIELDS)}")
    return reasons


def capability_report(ledger: Mapping[str, Any],
                      keys: Sequence[str]) -> Dict[str, Any]:
    """Which capabilities every one of ``keys`` has live evidence for, and what blocks.

    THE INTERSECTION IS THE ANSWER, not the union: a table dispatches an arm from any
    of its certified families, so a capability one family never ran on is a capability
    the table cannot claim. A partial round therefore admits nothing, and the keys that
    hold it back are named rather than left to be worked out from the ledger.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    admitted: Optional[set] = None
    if not isinstance(ledger, Mapping) or "_unreadable" in ledger:
        return {"admitted": None, "by_key": by_key,
                "unreadable": (ledger or {}).get("_unreadable")
                if isinstance(ledger, Mapping) else repr(ledger)}
    for key in sorted(set(keys)):
        entry = ledger.get(key)
        live = live_capabilities(entry)
        problem = None
        if not isinstance(entry, Mapping):
            problem = f"no entry under {key!r}"
        else:
            reasons = retired_shape_reasons(entry)
            if reasons:
                problem = "; ".join(reasons)
            elif not live:
                problem = ("every recorded run binds bytes that have since moved; "
                           "re-run this gate")
        runs = entry.get(RUNS) if isinstance(entry, Mapping) else None
        by_key[key] = {
            "live": list(live),
            "stale": sorted(set(runs) - set(live)) if isinstance(runs, Mapping) else [],
            "problem": problem,
        }
        admitted = set(live) if admitted is None else (admitted & set(live))
    return {"admitted": tuple(sorted(admitted or ())), "by_key": by_key}


def capability_admission(table: str = "triton") -> Dict[str, Any]:
    """:func:`capability_report` for the keys that table's arms cite."""
    keys = {gate for _family, gate in _arm_certification_map(table).values()}
    return capability_report(_fingerprints(table), sorted(keys))


def table_capabilities(ledger: Mapping[str, Any],
                       keys: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """The admitted tuple alone. ``None`` is unreadable; ``()`` refuses every device."""
    return capability_report(ledger, keys)["admitted"]


def route_campaign(stamp: str, capability: str) -> str:
    """The route campaign directory for one capability: ``<stamp>_cc86``.

    One stamp per release, one campaign per capability under it, so a reader can see
    which architecture a route record describes from the directory name alone.
    """
    normalised = _require_capability(capability)
    if normalised != capability:
        raise CapabilityRecordError(
            f"{capability!r} is not spelled as this ledger keys a capability "
            f"({normalised!r} is)")
    return f"{stamp}_cc{normalised.replace('.', '')}"


def primary_table(capability: str) -> Optional[str]:
    """Which NVIDIA table carries the dispatch-by-default licence for a capability.

    The first table in :data:`NVIDIA_TABLE_PRECEDENCE` that admits it, because that is
    the table whose plan composes first and therefore the one the licence describes.
    """
    for table in NVIDIA_TABLE_PRECEDENCE:
        admitted = capability_admission(table)["admitted"]
        if admitted and capability in admitted:
            return table
    return None


def bind_capability(entry: Dict[str, Any], *, bound_before: Optional[str],
                    capability: str, run: Mapping[str, Any],
                    run_fields: Sequence[str] = RUN_FIELDS,
                    supersede: Sequence[str] = ()) -> Tuple[str, ...]:
    """Write one capability's run record, and refuse to strand the others.

    THE WRITE IS THE PLACE THE STALENESS RULE IS ENFORCED, because it is the only
    moment both the old and the new bound are in hand. A rebind on UNCHANGED bytes
    adds a capability beside the ones already there, which is what makes a second
    architecture additive. A rebind on bytes that MOVED invalidates every other
    capability's evidence, and that is a decision a person has to make: those
    capabilities are named in the refusal and have to be re-run, or superseded by name.

    Returns the capabilities this write superseded.
    """
    normalised = _require_capability(capability)
    if normalised != capability:
        raise CapabilityRecordError(
            f"{capability!r} is not spelled as this ledger keys a capability "
            f"({normalised!r} is)")
    foreign = sorted(set(run) - set(run_fields))
    if foreign:
        raise CapabilityRecordError(
            f"{foreign} are not run fields; a field that describes the BYTES belongs "
            f"beside the digests, not inside {RUNS}[{capability!r}]")
    current = bound_digest(entry)
    runs = entry.setdefault(RUNS, {})
    if not isinstance(runs, Mapping):
        raise CapabilityRecordError(f"{RUNS!r} is {type(runs).__name__}, not a mapping")
    staled: Tuple[str, ...] = ()
    if bound_before is not None and current != bound_before:
        staled = tuple(sorted(
            name for name, record in runs.items()
            if name != capability and isinstance(record, Mapping)
            and record.get("bound_sha256") == bound_before))
        unnamed = sorted(set(staled) - set(supersede))
        if unnamed:
            raise CapabilityStale(unnamed)
    runs[capability] = {key: run[key] for key in sorted(run)}
    runs[capability]["bound_sha256"] = current
    entry[RUNS] = {name: runs[name] for name in sorted(runs)}
    return staled


def validated_compute_capabilities() -> Tuple[str, ...]:
    """The GPU architectures this table's cited gates ALL ran on. DERIVED, never typed.

    It used to be one hand-typed key in the ledger, which is the defect this replaces:
    no tool wrote it, so a certification round on another architecture changed every
    weld's evidence and left the declaration saying what it had always said. Now it is
    the intersection of the capabilities each cited weld has a live run for
    (:func:`capability_report`), so it can only say what some run actually measured, and
    a round that certifies one more architecture widens it by writing records.

    THIS FUNCTION RETURNS A TUPLE AND COLLAPSES THE TWO EMPTIES, because it answers
    "what does the record say" for the dispatch report, where both read as nothing. The
    DECISION does not go through it: rung 4b asks :func:`capability_admission`, whose
    ``admitted`` is ``None`` for a ledger that could not be read (unknown, not refused)
    and ``()`` for a readable one whose cited welds share no live capability, which
    refuses every device until a round repairs it.
    """
    admitted = capability_admission("triton")["admitted"]
    return () if admitted is None else admitted


def _normalized_capability(value: Any) -> str:
    """``(8, 6)``, ``"86"``, ``"8.6"`` -> ``"8.6"``. One spelling, so a comparison means something.

    CuPy hands the same number out three ways — ``Device().compute_capability`` is
    the string ``"86"``, ``getDeviceProperties()`` gives ``major``/``minor`` ints,
    and every recorded gate wrote ``"8.6"`` — so the gate would be decided by
    which reader happened to answer first without this.
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{int(value[0])}.{int(value[1])}"
    text = str(value).strip()
    if text.isdigit() and len(text) >= 2:
        return f"{text[:-1]}.{text[-1]}"
    return text


def _device_identity(xp: Any) -> Dict[str, Any]:
    """WHICH GPU this run would launch on, read the way the gates read it. Never raises.

    Triton owns the PTX/SASS the bit-identity certifications are statements about,
    and it generates that code FOR AN ARCHITECTURE — which is the same argument
    :data:`validated_triton_versions` rests on, applied to the other half of the
    codegen input. Every gate in ``fingerprints.json`` records
    ``the GPU host, NVIDIA RTX A6000, compute capability 8.6``; before this block the
    run artifact recorded no device at all, so a dispatched run could not afterwards
    say which device produced it, and an sm_90 host cleared every rung.

    The reads are the gate harnesses' own (``cp.cuda.runtime.getDeviceProperties``
    on ``getDevice()``, plus the driver/runtime versions), so the artifact and the
    certification are describing the device the same way.
    """
    block: Dict[str, Any] = {}
    try:
        runtime = xp.cuda.runtime
        index = int(runtime.getDevice())
        properties = runtime.getDeviceProperties(index)
        name = properties["name"]
        block["index"] = index
        block["name"] = name.decode() if isinstance(name, bytes) else str(name)
        block["compute_capability"] = _normalized_capability(
            (properties["major"], properties["minor"]))
    except Exception as exc:  # noqa: BLE001 - an unreadable device is recorded, not a crash
        block["unreadable"] = repr(exc)
        return block
    for key, reader in (("cuda_runtime", "runtimeGetVersion"),
                        ("cuda_driver", "driverGetVersion")):
        try:
            block[key] = int(getattr(xp.cuda.runtime, reader)())
        except Exception:  # noqa: BLE001
            pass
    return block


def _complex_probe_policy_reasons(probe: Any) -> List[str]:
    """Why this expansion-probe artifact may not be consumed by THIS dispatch.

    Delegates to the library's own rule so there is one definition of what a
    licence is conditional on, and pins it to :data:`CERTIFICATION_SUBNORMAL_POLICY`
    — the policy dispatch requires and installs — rather than to whatever is
    resolved at this rung, because at this rung nothing is installed yet.
    """
    try:
        from .triton_kernels.complex_fields import (  # noqa: PLC0415
            expansion_policy_reasons)
    except Exception as exc:  # noqa: BLE001 - an unreadable rule is not a licence
        return [f"the complex-expansion policy rule is not importable ({exc!r}), "
                f"so this artifact cannot be shown to have been cut under "
                f"{CERTIFICATION_SUBNORMAL_POLICY!r}"]
    try:
        return list(expansion_policy_reasons(probe, CERTIFICATION_SUBNORMAL_POLICY))
    except Exception as exc:  # noqa: BLE001 - a malformed artifact is a refusal
        return [f"the expansion probe artifact could not be judged against the "
                f"{CERTIFICATION_SUBNORMAL_POLICY!r} policy ({exc!r})"]


@contextmanager
def _composition_window(probe: Any, refusal_reasons: Sequence[str]) -> Iterator[Any]:
    """The window in which every EXPANSION constexpr is bound, with what this
    dispatch knows about it in force. Yields the value ``plan_step`` receives.

    TWO FACTS THE RUNGS BELOW CANNOT READ FOR THEMSELVES, and one bug each.

    THE REFUSAL. When rung (4c) refused the artifact, the value yielded here is a
    refusal rather than ``None``, and the same refusal is recorded against the
    environment variable for the duration. Passing it binds the rungs it reaches;
    recording it binds the rungs it does not — and the rung that broke this
    invariant was one that read the variable rather than the argument. Refusal
    and absence are DIFFERENT values because they deserve opposite outcomes:
    nothing offered may fall back to the environment, something offered and
    refused may not.

    THE POLICY. Rung (8b) installs :data:`CERTIFICATION_SUBNORMAL_POLICY` and
    refuses any process that installed anything else, so which arithmetic this
    dispatch will use is a fact it already owns — but it is not INSTALLED yet at
    this rung, and nothing below can read an intention. Declaring it is what lets
    the library's policy clause fail closed without refusing every honest run.

    Restored on the way out, including on an exception, so neither fact leaks
    into unrelated later work in this process.
    """
    with ExitStack() as scope:
        scope.enter_context(declaring_run_policy(CERTIFICATION_SUBNORMAL_POLICY))
        if not refusal_reasons:
            yield probe
            return
        # Imported here rather than spelled again: the variable a refusal is
        # recorded against must be the one the reader consults, and one constant
        # is how that stays true.
        from .triton_kernels.complex_fields import (  # noqa: PLC0415
            PROBE_PATH_ENVIRONMENT)
        yield scope.enter_context(
            refusing_expansion_probe(PROBE_PATH_ENVIRONMENT, refusal_reasons))


def _probe_licence_block(probe: Any) -> Dict[str, Any]:
    """What licence this artifact carries, recorded so the run can be AUDITED.

    The dispatch record used to say only whether a probe env var was set and
    whether a record resolved. That is not enough to answer, after the fact,
    which licence a run consumed: not the arm, not the basis (measured vs a row
    of ENVIRONMENT_DEFAULTS), not the policy it was cut under, and not the
    environment it CLAIMS — which is what ``environment_default`` keys on, so a
    probe.json copied from another machine would license dispatch here and leave
    no trace of having done so. All four go in.
    """
    out: Dict[str, Any] = {}
    if not isinstance(probe, dict):
        return out
    stamp = probe.get("subnormal_policy")
    if isinstance(stamp, dict):
        out["policy"] = stamp.get("policy")
        out["policy_resolved"] = stamp.get("resolved")
    claimed = probe.get("environment")
    if isinstance(claimed, dict):
        out["claimed_environment"] = {key: claimed.get(key) for key in
                                      ("backend", "machine", "cupy_version",
                                       "device_name", "compute_capability")}
    try:
        from .triton_kernels.complex_fields import (  # noqa: PLC0415
            expansion_license)
        verdict = expansion_license(probe)
    except Exception as exc:  # noqa: BLE001
        out["licence_error"] = repr(exc)
        return out
    out["arm"] = verdict.get("arm")
    out["basis"] = verdict.get("basis")
    out["non_discriminating"] = list(verdict.get("non_discriminating") or ())
    out["refusals"] = list(verdict.get("refusals") or ())
    if verdict.get("environment_default"):
        out["environment_default_artifact"] = verdict["environment_default"].get("artifact")
    return out


def _probe_capability_reasons(probe: Any, live: Any) -> List[str]:
    """Why this expansion probe does not describe THIS architecture. Empty means it may.

    Three-valued like every other identity rung: a probe that records no capability,
    or a host whose device will not say, yields no reason — the mismatch is reported in
    the record by :func:`_probe_environment_mismatch` instead. Only two READ values
    that differ drop the artifact.
    """
    claimed = probe.get("environment") if isinstance(probe, dict) else None
    if not isinstance(claimed, dict) or live is None:
        return []
    stated = claimed.get("compute_capability") or claimed.get("cc")
    if not stated:
        return []
    stated = _normalized_capability(stated)
    here = _normalized_capability(live)
    if stated == here:
        return []
    return [f"the expansion probe was measured on compute capability {stated} and this "
            f"host is {here}; which arm a platform's compiler takes is a per-"
            f"architecture measurement, so this artifact licenses nothing here"]


def _probe_environment_mismatch(probe: Any, block: Mapping[str, Any]) -> List[str]:
    """Where the artifact's CLAIMED environment disagrees with this live host.

    ``environment_default`` matches its table against the record's own
    ``environment`` block and nothing cross-checks that block against the machine
    the record is being READ on. Triton version and compute capability are already
    checked against the live host at rungs (4) and (4b); the machine and the CuPy
    version — the other half of ENVIRONMENT_DEFAULT_KEYS — were not.

    Reported, not refused, and deliberately: this is a mismatch between two
    self-descriptions, one of which (``machine``) may legitimately be absent, and
    the policy check above is what actually holds the line. A named mismatch in
    the record is what makes a wrong-host artifact visible in the audit.
    """
    out: List[str] = []
    claimed = probe.get("environment") if isinstance(probe, dict) else None
    if not isinstance(claimed, dict):
        return out
    live_backend = block.get("backend")
    if claimed.get("backend") not in (None, live_backend):
        out.append(f"artifact claims backend {claimed.get('backend')!r}, this host "
                   f"runs {live_backend!r}")
    live_machine = platform.machine()
    if claimed.get("machine") not in (None, live_machine):
        out.append(f"artifact claims machine {claimed.get('machine')!r}, this host "
                   f"is {live_machine!r}")
    live_cupy = block.get("backend_version")
    if claimed.get("cupy_version") not in (None, live_cupy):
        out.append(f"artifact claims cupy_version {claimed.get('cupy_version')!r}, "
                   f"this host runs {live_cupy!r}")
    return out


def _environment_block(grid: Any, probe: Any, probe_error: Optional[str],
                       table: str = "triton") -> Dict[str, Any]:
    """The toolchain, the device and the expansion licence, as READ not assumed.

    ONE ENTRY POINT, TWO TOOLCHAINS. The Triton half is below; the Metal half is
    ``metal_dispatch.environment_block``, which reads torch, the Metal frontend and
    the four expansion-probe states and answers ``torch_certified`` /
    ``frontend_certified`` THREE-VALUED — a version it could not read is recorded as
    unknown and not refused, which is rung 4b's rule on the other table. The
    delegation keeps the reader's question ("what was this run's toolchain") one
    question, while leaving the answer with the module that owns that table.
    """
    if table == METAL_TABLE:
        from . import metal_dispatch  # noqa: PLC0415

        return dict(metal_dispatch.environment_block(
            grid, metal_dispatch.metal_toolchain()))
    block: Dict[str, Any] = {
        "complex_expansion_probe_env_set": bool(os.environ.get(COMPLEX_PROBE_ENV)),
        "complex_expansion_probe_resolved": probe is not None,
    }
    if probe_error is not None:
        block["complex_expansion_probe_error"] = probe_error
    licence = _probe_licence_block(probe)
    if licence:
        block["complex_expansion_licence"] = licence
    try:
        import triton  # noqa: PLC0415

        version = str(getattr(triton, "__version__", "unknown"))
        block["triton"] = version
        block["triton_certified"] = version in validated_triton_versions()
    except Exception as exc:  # noqa: BLE001
        block["triton"] = None
        block["triton_certified"] = False
        block["triton_import_error"] = repr(exc)
    xp = getattr(grid, "xp", None)
    block["backend"] = getattr(xp, "__name__", None)
    block["backend_version"] = str(getattr(xp, "__version__", "")) or None
    device = _device_identity(xp)
    block["device"] = device
    block["device_certified"] = _device_is_validated(
        device, capability_admission("triton")["admitted"])
    block["validated_compute_capabilities"] = list(validated_compute_capabilities())
    # THE DEVICE QUESTION IS PER TABLE, because the two NVIDIA tables were certified
    # by different campaigns and neither list is derived from the other: Triton's is
    # the one key its ledger carries, and the hand-CUDA table's is DERIVED from the
    # certification blocks its own welds cite. ``device_certified`` above stays the
    # Triton answer — every existing reader means that by it — and the per-table map
    # is what rung 4b consults.
    from . import fastpath_cuda  # noqa: PLC0415

    # THE THREE-VALUED ANSWER COMES FROM THE REPORT, not from the public tuple: that
    # tuple cannot tell an unreadable ledger (unknown) from a readable one whose welds
    # share no live capability (refuse), and rung 4b needs them apart.
    cuda_admitted = capability_admission(CUDA_TABLE)["admitted"]
    block["validated_compute_capabilities_by_table"] = {
        "triton": list(validated_compute_capabilities()),
        CUDA_TABLE: list(fastpath_cuda.validated_compute_capabilities()),
    }
    block["device_certified_by_table"] = {
        "triton": block["device_certified"],
        CUDA_TABLE: _device_is_validated(device, cuda_admitted),
    }
    mismatch = _probe_environment_mismatch(probe, block)
    if mismatch:
        block["complex_expansion_probe_environment_mismatch"] = mismatch
    return block


def _device_is_validated(device: Mapping[str, Any],
                         validated: Optional[Sequence[str]]) -> Optional[bool]:
    """Is this device's architecture one a recorded gate ran on? None when unreadable.

    THREE-VALUED on purpose, and the third value is the honest one: a host whose
    device identity cannot be read (every stubbed backend in the test suite, and any
    CuPy whose runtime API moves) is neither certified nor refused, it is UNKNOWN,
    and the artifact says so rather than picking whichever answer is convenient.

    THE LIST IS A REQUIRED ARGUMENT, with no default: it used to default to the Triton
    table's, which made ``None`` mean two things at once -- "ask the Triton ledger" and
    "the ledger could not be read" -- and those now have opposite consequences. Both
    tables' lists are DERIVED from their cited welds' live
    per-capability runs (:func:`capability_report`), and the two empties mean different
    things, so they are spelled differently:

    * ``None`` — the ledger could not be read at all. UNKNOWN, as above.
    * ``()`` — the ledger reads, and its cited welds share no live capability. That
      REFUSES every device, because there is no architecture the table can claim. It
      used to read as unknown, which meant one stale weld admitted every device.
    """
    capability = device.get("compute_capability")
    if capability is None or validated is None:
        return None
    return _normalized_capability(capability) in validated


def _read(obj: Any, name: str, *args: Any, default: Any = None) -> Any:
    """Read an attribute that may be a property or a method. Never raises.

    The same defensive idiom ``coverage._call`` uses, and for the same reason: the
    record has to describe whatever object it was handed — including a stand-in in
    a test and a ``Grid`` from a future revision — without a missing attribute
    turning a report into an exception.
    """
    try:
        value = getattr(obj, name)
    except Exception:  # noqa: BLE001
        return default
    if callable(value):
        try:
            return value(*args)
        except Exception:  # noqa: BLE001
            return default
    return value


def _fold_description(grid: Any) -> Optional[str]:
    """Which axes a mirror plane folds, or None when the run is not folded.

    Read from the GRID, deliberately, and not from the arm labels the composer
    wrote: a rule that sniffs for arms named "folded ..." decides the fold from the
    same product selection whose gaps produced the hole in the first place, and it
    answers "not folded" for exactly the case that matters — the one where a folded
    product refused.
    """
    axes = [name for axis, name in enumerate("XYZ")
            if bool(_read(grid, "is_mirrored", axis, default=False))]
    if axes:
        return "mirror plane on " + ", ".join(axes)
    if bool(_read(grid, "has_symmetry", default=False)):
        return "a mirror plane is active on an axis this record could not name"
    return None


def _run_shape(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """The configuration axes THIS run carries, read from the objects. Never raises.

    Why the artifact needs it: an arm's certification names a gate, and a gate
    composed the shapes it composed. ``ADE update_P`` cites the dispersive
    composition gate and its predicate carries no geometry clause at all, so it
    dispatches ALONE on a cylindrical m=0 grid and on a chi2/chi3 grid — shapes
    that gate never ran. Reading update_P's own arithmetic says there is nothing
    for the geometry to change (``stepping.update_P`` is a loop over polarizations
    with no coordinate branch, and the plan transcribes the same rotation), so this
    is not a refusal — but a reader comparing "what the gate composed" against
    "what I ran" could not do it from the artifact at all, and now can.
    """
    # Every name here is the one the COVERAGE PREDICATES read (coverage.py), so the
    # shape the artifact reports and the shape the admission decision was made on
    # are the same reads rather than two spellings that can drift apart.
    shape: Dict[str, Any] = {}
    try:
        shape["dimensions"] = _read(grid, "dimensions")
        shape["grid_shape"] = tuple(_read(grid, "shape", default=()) or ())
        shape["cylindrical"] = bool(_read(grid, "cylindrical", default=False))
        if shape["cylindrical"]:
            shape["m"] = _read(grid, "m")
        shape["folded"] = _fold_description(grid)
        shape["complex_storage"] = bool(_read(fields, "force_complex_fields", default=False))
        shape["bloch"] = bool(_read(grid, "has_bloch", default=False))
        shape["k_point"] = _read(grid, "k_point")
        shape["beta"] = _read(grid, "beta")
        shape["bfast"] = bool(_read(grid, "bfast_active", default=False))
        shape["conductivity"] = bool(_read(fields, "has_conductivity", default=False))
        shape["nonlinearity"] = bool(_read(fields, "has_nonlinearity", default=False))
        shape["off_diagonal_epsilon"] = bool(
            _read(fields, "has_offdiagonal_epsilon", default=False))
        polarizations = _read(fields, "polarizations", default=()) or ()
        shape["susceptibilities"] = len(polarizations)
        shape["pml_active"] = bool(_read(pml, "is_active", default=False))
    except Exception as exc:  # noqa: BLE001 - a shape that will not read is not a refusal
        shape["unreadable"] = repr(exc)
    return {key: value for key, value in shape.items() if value is not None}


def _composition_block(record: Mapping[str, Any], slots: Tuple[str, ...],
                       arms: Mapping[str, str],
                       table: str = "triton") -> Dict[str, Any]:
    """Which sub-steps a KERNEL ran and which the ARRAY path ran, said plainly.

    A mixed step is the normal, designed outcome — Triton is an accelerated subset
    — and each dispatched sub-step is bit-identity gated against the array sub-step
    it replaces, which is what makes the mix sound. What is NOT certified is the
    mix itself, and two measured cases show why saying so matters:

    * a complex beta run dispatches the ``special_kz`` CONSTITUTIVE arms while that
      same family's curl arm refuses (its probe clause is the base pattern set),
      so a family that was certified as a unit dispatches half of itself and the
      artifact cites the family record for both halves;
    * ``update_P`` dispatches alone on grids where nothing else is covered.

    Neither has a demonstrated divergence — the array sub-steps are the oracle and
    still run — so the honest answer is disclosure, not a refusal that would cost
    coverage no measurement asks for.
    """
    array_slots = [slot for slot in DRIVER_SLOTS if slot not in slots]
    block: Dict[str, Any] = {
        "table": table,
        "dispatched_slots": list(slots),
        "array_slots": array_slots,
        "mixed": bool(array_slots),
        "families_dispatched": sorted({
            record["families"][arm]["family"] for arm in set(arms.values())
            if arm in record["families"]}),
    }
    if table == METAL_TABLE:
        # WHICH RESIDENCY MODE THE MIX WAS COMPOSED UNDER, because on this table
        # that is part of what "mixed" means. The array sub-steps write the HOST
        # arrays and the dispatched ones read device mirrors, so the answer to "is
        # the mix sound" depends on when the two are copied. The Metal ladder
        # records the mode it ran under, and which mirrors it held, in the record's
        # own ``residency`` block; this points there rather than restating it,
        # because a restated mode is a second copy of a fact that already went stale
        # here once (it described the per-launch bracket after held residency had
        # shipped).
        block["residency_mode"] = (
            "recorded under residency.mode, with the mirrors held on device "
            "between launches under residency.held")
    if array_slots:
        block["read_this_as"] = (
            "each dispatched sub-step is bit-identity gated against the array "
            "sub-step it replaces, and the slots listed under array_slots ran the "
            "array path with the reasons in slots[*].reason. The MIX itself was not "
            "composed by any gate: the gates named under families ran their own "
            "compositions, on their own shapes. Compare run_shape against the gate "
            "record before treating a dispatched family's certification as covering "
            "this configuration.")
    return block


#: The one reason a reference driver's record carries (:func:`reference_record`).
#: HOST-NEUTRAL, because the reference reads nothing about the host: it cannot say
#: "this host's GPU" without probing for one, and on a host with none the advice to
#: build with ``prefer_gpu=True`` would be advice to raise.
REFERENCE_REASON = (
    "reference driver: built with prefer_gpu=False, the NumPy reference, which "
    "consults no kernel table whatever MEEP_GPU_DISPATCH says; kernels are "
    "dispatched only by a driver built with prefer_gpu=True, which needs a GPU on "
    "the host (a CUDA device through CuPy, or an Apple GPU) and raises without one")


def _base_record() -> Dict[str, Any]:
    return {
        "step_path": "array",
        "decision": "refused",
        "refused_because": None,
        # A DRIVER BUILT WITH prefer_gpu=False never reaches the ladder; its freeze
        # publishes :func:`reference_record`, which sets this True. Every other
        # record, dispatched or refused, is a GPU driver's (or a direct caller's).
        "reference_driver": False,
        # WHICH ARRAY MODULE THE ENGINE RUNS ON, read off the grid before any rung
        # answers (:func:`plan_fast_path`), so a record refused at rung 1 or 2 --
        # where the ``environment`` block is deliberately not read -- still says
        # where its array path runs. ``"numpy"`` is the host CPU; ``None`` is a record
        # built with no grid in hand.
        "engine_backend": None,
        # WHICH KERNEL TABLE, on EVERY record and not only the ones that reached
        # one. ``None`` is the honest pre-backend-rung answer and it is a different
        # fact from "triton": a reader comparing two refused records has to be able
        # to tell a run refused before any table was chosen from a run whose chosen
        # table then refused it, and an absent key collapses those two.
        "table": None,
        # WHICH TABLES WERE EVEN ASKED, and why each answered as it did. Present on
        # EVERY record for the reason ``table`` is: a reader comparing two refused
        # runs has to be able to tell "refused before any table was consulted" from
        # "both tables were asked and both declined", and an absent key collapses
        # those. Filled at rung 4, one entry per table, each with its own reason.
        "tables": {"not_read": "refused before any table was consulted"},
        # WHICH TABLE COMPOSED FIRST when more than one was a candidate, and under
        # what rule. Filled at rung 4d.
        "arbitration": {"not_reached": "refused before the precedence rung"},
        "slots": {},
        "dropped_null": {},
        "built_not_dispatched": {},
        "families": {},
        "certification_budget": CERTIFICATION_BUDGET,
        "kill_switch": {
            "variable": FUSED_KILL_SWITCH,
            "value": os.environ.get(FUSED_KILL_SWITCH),
            "vetoes": not fused_dispatch_enabled(),
        },
        "enable": {
            "variable": DISPATCH_ENABLE,
            "value": os.environ.get(DISPATCH_ENABLE),
            "default": DISPATCH_BY_DEFAULT,
            # The NON-RAISING reader: this record is built outside the wrapper and
            # before rung 1, so enforcing anything here breaks two of the ladder's
            # own contracts. See :func:`_enable_as_set`.
            "effective": _enable_as_set(),
        },
        # WHETHER THE KERNELS ARE CERTIFIED ON THE IDENTITY THAT SERVED. ``None``
        # until a plan dispatches, and after it when an identity could not be read;
        # ``False`` only for a run the uncertified opt-in admitted, whose identity
        # is then under ``uncertified.served`` (:func:`_certified_verdict`).
        "certified": None,
        "uncertified": {
            "variable": UNCERTIFIED_SWITCH,
            "value": os.environ.get(UNCERTIFIED_SWITCH),
            "allowed": uncertified_allowed(),
            "admitted": [],
        },
        # THE FUSION OPT-IN, on EVERY record and not only the ones that used it.
        # A reader comparing two runs has to be able to tell "this run asked for
        # nothing" from "this run asked and was refused", and an absent key
        # collapses those two into one.
        # ``released``/``admitted``/``outside_the_released_envelope`` are the
        # RUN-DEPENDENT half and are filled at the composer rung, where the run
        # shape has been read; they say "not reached" until then rather than
        # being absent, for the same reason the block itself is always present.
        "fusion": {
            "variable": FUSE_ARMS_SWITCH,
            "value": os.environ.get(FUSE_ARMS_SWITCH),
            "opted_in": list(requested_fused_arms()),
            # THE VETO IS A SEPARATE FACT FROM AN EMPTY LIST, and the record has to
            # keep them apart: "this run asked for no extra arm" and "this run
            # refused the released ones" both leave ``opted_in`` empty and mean
            # opposite things about what dispatched.
            "vetoed": fused_arms_vetoed(),
            "driver_route_gate": DRIVER_ROUTE_FUSED_GATE,
            "released_by_the_gate": sorted(RELEASED_FUSED_ARMS),
            "released_here": "the run shape was not reached",
            "released_here_by_table": "the run shape was not reached",
            "admitted": "the run shape was not reached",
            "admitted_by_table": "the run shape was not reached",
            "outside_the_released_envelope": "the run shape was not reached",
            "driven": {},
        },
        "warm_pass": {
            "variable": WARM_SWITCH,
            "enabled": warm_pass_enabled(),
            # Not a timing switch. With the warm pass ON, a slot whose kernel will
            # not compile is unfilled at plan time and its family falls to the
            # array path; with it OFF nothing is unfilled, so the DISPATCH SET is
            # wider and a compile failure lands mid-step instead. Recorded here so
            # a run made with it off cannot be read as the same run made with it on.
            "changes_the_dispatch_set_when_disabled": not warm_pass_enabled(),
        },
        # Filled in once the ladder reaches a rung where a kernel could still run.
        # Reading the policy stamp on a NumPy host would drag the policy machinery
        # into a step that cannot dispatch, and ``test_package_boundary`` measures
        # exactly that kind of needless pull-in. ``gate`` arrives later still, at
        # rung 8b, and only for a configuration that got that far: it is the record
        # of a policy INSTALL, and a refused configuration installs nothing.
        "subnormal": {"not_read": "the configuration was refused before any "
                                  "kernel could run"},
    }


def _emit(record: Mapping[str, Any]) -> None:
    """Publish one freeze's artifact: memory always, the log file when configured.

    Appended UNBUFFERED, one JSON object per line, per the progress-reporting rule — a long
    run's dispatch state has to be readable with ``tail`` on the machine that owns
    the job, without attaching to it and without waiting for it to finish.
    """
    global _LAST_RECORD
    _LAST_RECORD = record
    path = os.environ.get(DISPATCH_LOG, "").strip()
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
        except Exception:  # noqa: BLE001 - an unwritable log never refuses a step
            pass
    _announce(record)


def _certified_mark(value: Any) -> str:
    """THREE-VALUED, in one spelling. ``None`` is unknown and is not ``False``.

    The device read has always had three answers (:func:`_device_is_validated`) and
    both toolchain reads now do — a torch or Metal-frontend version this host would
    not report is RECORDED as unread rather than refused, which is rung 4b's rule.
    Collapsing unknown into UNCERTIFIED on the stderr line would say the run was
    measured and failed where it was never measured at all.
    """
    return {True: "certified", False: "UNCERTIFIED"}.get(value, "uncertified-unknown")


def _announce(record: Mapping[str, Any]) -> None:
    """One line to stderr, so a user who never opens the artifact still knows."""
    if record.get("reference_driver"):
        # THE REFERENCE CHOSE THE ARRAY PATH and is quiet, unless the environment
        # asked for dispatch: that request does not apply here, and a run that set it
        # and got the reference did not choose that, so it is told once per process.
        value = (record.get("enable") or {}).get("value")
        if value is not None and value != "0":
            line = (f"meep_gpu: step path array (NumPy reference, "
                    f"prefer_gpu=False); {DISPATCH_ENABLE}={value!r} does not "
                    "apply to a reference driver; kernels are dispatched only by a "
                    "driver built with prefer_gpu=True, which needs a GPU on the "
                    "host")
            if _enable_value_refusal(value) is not None:
                # THE ADVICE MUST NOT SEND THE VALUE ALONG: on a GPU driver rung 2
                # refuses it by name, and the run would be on the array path again.
                line += (f"; NOTE {DISPATCH_ENABLE}={value!r} is not an accepted "
                         "value either, and a prefer_gpu=True driver refuses it by "
                         "name (unset or 1 dispatches, 0 disables)")
            _announce_line(line)
        return
    if record["decision"] == "dispatched":
        slots = record["slots"]
        running = [name for name, entry in slots.items() if entry["state"] == "dispatched"]
        arms = sorted({slots[name]["arm"] for name in running})
        subnormal = record["subnormal"].get("stamp") or {}
        resolved = subnormal.get("resolved", "unknown")
        requested = subnormal.get("requested", "unknown")
        # WHETHER A POLICY WAS INSTALLED, not just what it resolved to. The stamp
        # reports 'none_installed' with resolved='keep' when match_meep could not be
        # MEASURED and fell back — a fallback, not a measurement of MEEP — and a line
        # that prints only the resolution states the fallback as a fact.
        installed = subnormal.get("installed")
        if installed is False or subnormal.get("policy") == "none_installed":
            policy = f"policy {resolved} NOT INSTALLED (requested {requested})"
        else:
            policy = f"policy {resolved} (requested {requested})"
        # WHO installed it, and what it cost. A dispatched run whose host would
        # have flushed is now keeping — the ARRAY path included — and a user who
        # reads one line about this run has to be told that here, not only in the
        # artifact, because it is the one change dispatch makes that survives
        # outside the kernels.
        gate = record["subnormal"].get("gate") or {}
        if gate.get("installed_by"):
            policy += f", installed by {gate['installed_by'].split(',')[0]}"
        if gate.get("overrode_run_resolution"):
            policy += (f" OVERRIDING this run's own "
                       f"{gate.get('run_would_resolve_to')} resolution")
        environment = record["environment"]
        # THE TOOLCHAIN HALF IS THE TABLE'S OWN. Printing "triton None UNCERTIFIED"
        # beside a dispatched Metal composition would be reporting the absence of a
        # compiler that table never asks for as though it were a caveat about the
        # run — a line a user reads once and misreads. Each table names what it
        # actually compiled with, and the certification mark is that table's answer.
        table = record.get("table") or "triton"
        # A MERGED STEP SAYS SO ON THE ONE LINE A USER READS. "table triton" beside
        # a step whose D seam is a hand-CUDA product would name the composer that
        # did not produce half of it, and the artifact is the only other place that
        # fact lives.
        dispatched_tables = ((record.get("composition") or {})
                             .get("tables_dispatched") or [table])
        if len(dispatched_tables) > 1:
            table_text = "tables " + "+".join(dispatched_tables)
        else:
            table_text = f"table {dispatched_tables[0]}"
        if table == METAL_TABLE:
            toolchain = (f"torch {environment.get('torch')} "
                         f"{_certified_mark(environment.get('torch_certified'))}, "
                         f"metal frontend {environment.get('metal_frontend')} "
                         f"{_certified_mark(environment.get('frontend_certified'))}")
        elif table == CUDA_TABLE:
            # THE HAND-CUDA TABLE COMPILES THROUGH NVRTC, not Triton, so naming a
            # Triton version beside it would credit a compiler that produced none of
            # the bytes. What it does share with the Triton table is the CuPy the
            # kernels are launched from, which is what the welds record.
            toolchain = (f"cupy {environment.get('backend_version')}")
            if "triton" in dispatched_tables:
                toolchain += (f", triton {environment.get('triton')} "
                              f"{_certified_mark(environment.get('triton_certified'))}")
        else:
            toolchain = (f"triton {environment.get('triton')} "
                         f"{_certified_mark(environment.get('triton_certified'))}")
        device = environment.get("device") or {}
        device_text = device.get("name") or f"device {device.get('unreadable', 'unknown')}"
        device_mark = _certified_mark(environment.get("device_certified"))
        line = (f"meep_gpu: step path fused; {len(running)}/{len(slots)} slots "
                f"({','.join(running)}) via {','.join(arms)}; "
                f"{table_text}; "
                f"{policy}; "
                f"{toolchain}; "
                f"{device_text} {device_mark}")
        if record.get("certified") is False:
            # THE OPT-IN'S OWN LINE, after the run's own and once per process.
            _announce_line(line)
            line = _uncertified_note(record)
    else:
        # EVERY REFUSAL THE RUN DID NOT CHOOSE IS SAID, once per distinct line per
        # process (:func:`_announce_line`): a GPU driver asked for kernels, so a
        # freeze that gives it the array path says so and why, and every later
        # driver in the process repeats the same line, so it is printed once. The
        # two refusals a run CHOSE stay quiet — ``MEEP_GPU_DISPATCH=0`` and
        # ``MEEP_GPU_FUSED=0`` — because an opted-out run does not need telling.
        #
        # NO DRIVER BUILT THROUGH THE PUBLIC API REACHES THIS BRANCH ON A HOST WITH
        # NO GPU: ``prefer_gpu=True`` raises there at construction, and
        # ``prefer_gpu=False`` is the reference, which never plans and is answered by
        # the branch at the top. The empty-set refusal ("backend is not CuPy ... no
        # MPS device") is still announced for a direct caller of
        # :func:`plan_fast_path`, and for a GPU driver whose device went away
        # between construction and the freeze.
        #
        # AN UNRECOGNISED VALUE OF EITHER IS NOT A CHOICE. It reads as "off" only
        # because its rung refused it, and a run whose ``MEEP_GPU_DISPATCH=true`` or
        # ``MEEP_GPU_FUSED=off`` was refused has to be told, or the refusal changes
        # what it measures without saying so. That holds when ``MEEP_GPU_FUSED=0``
        # answers first as well: rung 1 never reads the enable, so a bad enable value
        # would otherwise ride unseen until the kill switch is lifted.
        enable, kill = record["enable"], record["kill_switch"]
        enable_refusal = _enable_value_refusal(enable["value"])
        kill_refusal = _kill_switch_value_refusal(kill["value"])
        chose_the_array_path = kill["vetoes"] or not enable["effective"]
        if chose_the_array_path and enable_refusal is None and kill_refusal is None:
            return
        # WHERE THE ARRAY PATH RUNS: a NumPy engine is the host CPU. On an Apple
        # GPU driver that is the whole run, so the line a user reads says so -- at
        # every rung, which is why the engine is read off the record's own
        # ``engine_backend`` and not only off the ``environment`` block, which rungs
        # 1 and 2 answer before reading.
        backend = (record.get("engine_backend")
                   or (record.get("environment") or {}).get("backend"))
        where = " on the host CPU" if backend == "numpy" else ""
        if kill["vetoes"] and kill_refusal is None:
            line = f"meep_gpu: step path array{where} ({FUSED_KILL_SWITCH}=0)"
        else:
            line = (f"meep_gpu: step path array{where}; dispatch refused: "
                    f"{record['refused_because']}")
        if enable_refusal is not None and record["refused_because"] != enable_refusal:
            line += (f"; NOTE {DISPATCH_ENABLE}={enable['value']!r} is not an "
                     "accepted value, and rung 2 refuses it by name once the kill "
                     "switch lets a run reach it")
        # The policy this refusal did NOT undo. See :func:`_refuse`.
        outlived = ((record.get("subnormal") or {}).get("gate")
                    or {}).get("policy_outlived_the_refusal")
        if outlived:
            line += (f"; NOTE the {outlived['policy']!r} float32 subnormal policy "
                     f"was installed before this refusal and STAYS IN FORCE for "
                     f"this process, the array path included")
    _announce_line(line)


def _announce_line(line: str) -> None:
    """Print a line to stderr once per DISTINCT line in this process.

    KEYED ON THE LINE, not on the outcome. Keyed on the outcome ("refused"), a
    process whose first freeze refused for an unrelated reason — a NumPy grid, an
    uncertified Triton — printed that one and then swallowed EVERY later refusal,
    including the subnormal policy's. Measured: a first freeze printing "backend
    is not CuPy" left a subsequent policy refusal on the array path with nothing
    on stderr at all, visible only to a reader who opens the artifact, which is
    precisely the reader this line exists for. The set stays small because the
    ladder has a bounded number of refusal texts.
    """
    if line in _ANNOUNCED:
        return
    _ANNOUNCED.add(line)
    print(line, file=sys.stderr, flush=True)


def _refuse(record: Dict[str, Any], reason: str) -> None:
    record["decision"] = "refused"
    record["step_path"] = "array"
    record["refused_because"] = reason
    # A REFUSAL AFTER THE GATE PASSED IS NOT "nothing was installed". Rung 8b sits
    # before the warm pass, so a plan whose every slot then fails to compile is
    # refused at rung 9 with the policy already installed and CUPY_CACHE_DIR
    # already moved — the process keeps the certification's policy for its whole
    # life, the ARRAY path's arithmetic included. On a host whose own resolution
    # is 'flush' that silently moves the entire run onto IEEE-754 while the line a
    # user reads says "step path array". It is recorded and announced instead.
    #
    # NOT ROLLED BACK, deliberately: the warm pass has by then compiled CuPy
    # and/or Triton kernels under the installed policy and both executors memoize
    # them in process, so uninstalling the seams would leave keep-compiled binaries
    # live in a process reporting no policy — the split again, from the other side.
    gate = (record.get("subnormal") or {}).get("gate") or {}
    if gate.get("policy_in_force"):
        gate["policy_outlived_the_refusal"] = {
            "policy": gate["policy_in_force"],
            "installed_by": gate.get("installed_by"),
            "overrode_run_resolution": bool(gate.get("overrode_run_resolution")),
            "consequence": (
                "this refusal came from a rung BELOW the policy gate, so the "
                "policy stays installed for the rest of the process and the array "
                "path runs under it. Kill-switch the dispatch "
                f"({FUSED_KILL_SWITCH}=0) to get this run's own resolution back"),
            "not_rolled_back_because": (
                "the warm pass compiles under the installed policy and both device "
                "executors memoize in process; dropping the seams would leave those "
                "binaries live under no policy at all"),
        }
    _emit(record)


def reference_record() -> Mapping[str, Any]:
    """Publish the freeze of a driver built with ``prefer_gpu=False``: no table, by design.

    The NumPy reference never consults the ladder, so no environment value can turn
    its kernels on and nothing is imported or installed for it. The record still
    goes through :func:`_emit`, so ``last_dispatch_report()`` and
    ``MEEP_GPU_DISPATCH_LOG`` carry one line per freeze, and the ``kill_switch`` /
    ``enable`` blocks still report what the environment said beside the reason --
    ``effective`` and ``vetoes`` included, which describe the VALUES and decide
    nothing on this driver. The retired ``*_FDTD_*`` names, the table
    preference and the hardware are not read at all (``environment.not_read``).
    ``decision`` stays ``"refused"`` (two-valued everywhere it is read);
    ``reference_driver`` is what separates this record from a refusal.
    """
    record = _base_record()
    record["reference_driver"] = True
    record["engine_backend"] = "numpy"
    record["environment"] = {"not_read": "reference driver"}
    record["tables"] = {"not_read": "reference driver: no table consulted"}
    record["arbitration"] = {"not_reached": "reference driver"}
    # WHAT THE HOST'S FLOATING-POINT STATE IS, read not assumed: a reference driver
    # installs nothing, but a GPU driver's freeze earlier in this process (or MEEP's
    # own initialization) may have left the thread flushing subnormals.
    from .backends import subnormals_flushed  # noqa: PLC0415

    record["subnormal"] = {"not_read": "reference driver: no policy installed",
                           "host_flushes_subnormals": bool(subnormals_flushed())}
    _refuse(record, REFERENCE_REASON)
    return record


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def plan_fast_path(fields: Any, pml: Any, grid: Any,
                   sources: Any = None) -> Optional[FastPathPlan]:
    """Decide whether Triton kernels may serve this frozen configuration.

    Returns a :class:`FastPathPlan` when at least one sub-step will dispatch a
    certified kernel, else ``None`` — the array path, everywhere, which is always
    correct. The caller (``FdtdDriver.step``) builds this once per configuration
    freeze and re-plans after any ``invalidate_fast_path()`` — for a driver built
    with ``prefer_gpu=True`` only. A ``prefer_gpu=False`` driver is the NumPy
    reference: its freeze never calls this function and publishes
    :func:`reference_record` instead, whatever the environment says.

    ``sources`` IS THE RUN'S SOURCE LIST, and it reaches only the fused-pair predicates,
    which are the only ones that ask. They cannot read it off ``Fields`` and refuse
    outright when it is not declared, so leaving it ``None`` is not a neutral default —
    it is the refusal. It was inert while the composer was asked with a literal
    ``fuse=False``; since the release it is LOAD-BEARING on every configuration
    :data:`FUSED_RELEASE_ENVELOPE` covers — a driver that stopped threading it
    would silently lose the three released arms to a by-name refusal rather than
    to a measurement.

    THE LADDER, in order, every rung degrading to the array path with a named
    reason and none of them raising:

    1. ``MEEP_GPU_FUSED=0`` — the kill switch, highest precedence, a veto
       that can never enable; ``1`` or unset passes, and any value outside
       :data:`FUSED_KILL_SWITCH_VALUES` is refused by name.
    2. The enable (:data:`DISPATCH_BY_DEFAULT`, ``True``, when unset; ``0``
       disables; any value outside :data:`DISPATCH_ENABLE_VALUES` is refused by
       name).
    3. THE BACKEND, AND WITH IT THE KERNEL TABLE. The hardware picks the CANDIDATE
       SET (:func:`candidate_tables`) — a CuPy engine gets
       :data:`NVIDIA_TABLE_PRECEDENCE`, a NumPy engine with an MPS device gets
       ``("metal",)``, anything else gets nothing — and :data:`KERNEL_TABLE_SWITCH`
       then selects WITHIN it, refusing BY NAME a table this host cannot run rather
       than falling back to one the caller did not ask for. An empty candidate set
       is the whole-plan refusal, and it keeps the substring "backend is not CuPy"
       that three independent classifiers read. Decided WITHOUT importing cupy or
       Triton; torch is imported only to read the MPS device for a NumPy engine.
       From ``FdtdDriver`` a NumPy engine reaches this rung only as an Apple GPU
       driver (``prefer_gpu=True``); on a host with no device the array path is
       all there is and there is nothing to launch on.
       ON THE METAL TABLE THE LADDER CONTINUES ELSEWHERE — rungs 4M-9M are
       ``metal_dispatch.decide``, handed :func:`_finish` so every table emits
       through one tail — and rungs 4 to 9 below are the NVIDIA tables', BOTH of
       them: a CuPy engine reaches them with two candidates, not one.
    4. THE TABLES' OWN CANDIDACY, ASKED AND ANSWERED PER TABLE. Triton: absent, or
       a version no recorded gate ran on. Hand-CUDA:
       :func:`meep_gpu.fastpath_cuda.cuda_candidate`, which asks the same kind of
       question of the toolchain its own certifications name. A failure DROPS THAT
       TABLE BY NAME and the ladder continues; the plan is refused only when the
       candidate set empties, and that refusal quotes every table's reason so a
       reader never has to guess which half failed. This stopped being a whole-plan
       refusal the moment there were two candidates: an unvalidated Triton
       invalidates the Triton certifications wholesale and says nothing whatever
       about the hand-CUDA table.
    4b. AN UNCERTIFIED GPU ARCHITECTURE, also per table. Same argument as 4 on the
       other half of the codegen input: a readable compute capability no recorded
       gate ran on is refused by name. The two validated lists come from different
       campaigns and neither implies the other, so a device one table was certified
       on and the other was not is a real state rather than a contradiction. An
       UNREADABLE capability is recorded as unknown and refuses neither — see
       :func:`_device_is_validated`. :data:`UNCERTIFIED_SWITCH` set to ``1`` admits
       what rungs 4 and 4b would refuse on identity, recorded ``certified: False``;
       any value of it but ``1`` and ``0`` is refused by name at rung 3b.
    4d. PRECEDENCE, which is to say WHICH CANDIDATE COMPOSES FIRST and therefore
       holds every slot both of them admit. :data:`BACKEND_PRECEDENCE` is the
       ruling — Triton first as the release-gated incumbent, hand-CUDA filling its
       refusals — and :data:`BACKEND_PREFERENCE_SWITCH` reorders within the
       candidate set. A preference naming a table that is not a candidate here is a
       NAMED refusal, never a silent fallback to the other one, for the same reason
       rung 3's is.
    5. EVERY CANDIDATE COMPOSES, IN EFFECTIVE ORDER, AND THE RESULTS MERGE BY WHOLE
       UNITS (:func:`meep_gpu.fastpath_cuda.merge_tables`): a secondary table's
       fused unit is adopted only if EVERY slot it spans is free in the primary's
       plan, and a refused unit is recorded by name against the label that held it.
       Under each composer, its own fail-closed contract still holds: a raising
       predicate is a refusal, a builder that raises or returns None leaves the slot
       unfilled, and two or more admitters leave it UNSELECTED. All three arrive
       here as an unfilled slot with a reason, and an unfilled slot is the array
       path.
    6. COMPLETENESS: a slot the composer FILLED that the driver consults cannot
       RUN refuses the WHOLE plan. ``fill_B``/``fill_D`` USED TO BE THAT SLOT and
       are not any more — they joined :data:`DRIVER_SLOTS` with the two driver
       consults on 2026-09-02 — so this rung now guards only a slot a future
       composer fills that no consult stands in front of. Running the remainder
       would be a composition no gate ever certified, and this project ranks that
       worse than a slower right answer.
    6b. THE FOLD, NARROWED to exactly what the consults did not discharge. It used
       to refuse EVERY folded configuration, because rule 6 fires only on slots the
       composer FILLED and a fold whose two fill arms both REFUSE (a complex-storage
       mirror run: the mirror-fill arm does not carry complex64 storage and the
       folded-complex fill arm wants an expansion-probe pattern the base artifact
       lacks) left ``fill_B``/``fill_D`` empty, passed completeness, and dispatched
       four folded kernel slots with the ghost fills on the array path — dispatch
       ENABLED BY a sibling product's refusal, and the exact composition
       ``fields.py`` twice states this function keeps off the kernels.
       WHAT SURVIVES IS THAT HOLE AND ONLY IT: a folded run must have BOTH fill
       slots carrying a kernel, or the whole plan is refused. The fold is still read
       from the GRID rather than sniffed off arm labels, so the refusal does not
       depend on which arms won; what changed is that a fold whose fills ARE covered
       is now a composition the seam can reproduce, because there is a consult in
       front of each of them.
       THIS IS THE NARROWING THE 2026-08-29 CENSUS LICENSED and the 2026-08-29
       measurement declined to take: it found no folded row whose live array-path
       fills a winning fused arm's ``REPLACES`` fails to span, but the rung was
       UNREACHED then (both folded cases died at rule 6 first) so narrowing it would
       have changed no runtime behaviour. The consults change that — rule 6 no
       longer catches the folded cases — so the narrowing now decides something, and
       it decides it in the fail-closed direction.
    7. THE NULL DROP, the one sanctioned exception to 6: a slot whose selected arm
       launches nothing is removed before completeness is judged. Sound because
       both sides of that slot are empty — the null plan's ``run`` does nothing,
       and the array call returns before its first statement.
    8. NO FUSION EXCEPT BY NAME, COUNTED PER TABLE. Each composer is asked to fuse
       only the arms ITS OWN release table names, on the configurations ITS OWN
       envelope and per-arm axes pin, plus whatever :data:`FUSE_ARMS_SWITCH` asks
       for; any other fused arm that appears anyway refuses the whole plan by name,
       and the refusal cites the table's own gate rather than the other one's. Nine
       Triton arms are released (:data:`RELEASED_FUSED_ARMS`) and twelve hand-CUDA
       ones (:data:`meep_gpu.fastpath_cuda.CUDA_RELEASED_FUSED_ARMS`); every other
       fused label of either table, and every one of these outside its own axes,
       still refuses.
    8b. THE SUBNORMAL POLICY (:func:`_subnormal_gate`) — every executor this TABLE
       drives (:data:`TABLE_GOVERNED_EXECUTORS`: host, CuPy and Triton here; host
       and MPS on the Metal table) driven to that table's one policy
       (:data:`TABLE_SUBNORMAL_POLICY`), or the whole plan refused by name. It sits
       here and
       not earlier because installing moves the process's arithmetic and a
       configuration refused above would have paid that for nothing; it sits here
       and not later because a policy installed after a compile does not reach the
       binary that already exists, and rung 9 compiles. WHICH executors it governs
       is read from what rungs 3 and 4 established, never re-derived — see
       :func:`_governed_executors` for the fail-open that cost.
    9. The warm pass — Triton compilation made a PLAN-TIME event. A slot whose
       kernel will not compile unfills here, where nothing has been written yet.

    Never raises. The whole body is wrapped, and an escape is recorded as
    ``internal error`` and answered with ``None``.

    THE ESCAPE KEEPS THE RECORD THE LADDER HAD BUILT. It used to mint a FRESH
    ``_base_record()``, whose ``subnormal`` block hardcodes "the configuration was
    refused before any kernel could run" — false for any escape from below rung
    8b, where a policy IS installed and ``CUPY_CACHE_DIR`` HAS moved. That was the
    one path on which the artifact made a false negative claim about coverage;
    the record is now built here and handed down, so whatever the ladder had
    established survives the escape.
    """
    record = _base_record()
    # An attribute read, importing nothing, so rung 1 still answers from a cold
    # process; it is the same name rung 3 writes into ``environment.backend``.
    record["engine_backend"] = getattr(getattr(grid, "xp", None), "__name__", None)
    try:
        return _decide(record, fields, pml, grid, sources)
    except Exception as exc:  # noqa: BLE001 - see the closing paragraph above
        record.setdefault("environment",
                          {"decided_before_environment_was_read": True})
        _refuse(record, f"internal error: plan_fast_path raised {exc!r}")
        return None


def _decide(record: Dict[str, Any], fields: Any, pml: Any,
            grid: Any, sources: Any = None) -> Optional[FastPathPlan]:

    # (1) The kill switch. Read before anything else and answered before anything
    # is imported, so bisection works from a cold process.
    #
    # AN UNRECOGNISED VALUE IS ANSWERED BEFORE ``0``. The non-raising reader reads
    # both as "vetoed", so asking it first would report "kill switch
    # MEEP_GPU_FUSED=0" for a run whose value was ``yes`` — the refusal would be
    # right about the path and wrong about why.
    kill_refusal = _kill_switch_value_refusal(os.environ.get(FUSED_KILL_SWITCH))
    if kill_refusal is not None:
        record["environment"] = {
            "not_read": "refused at an unrecognised kill-switch value"}
        _refuse(record, kill_refusal)
        return None
    if not fused_dispatch_enabled():
        record["environment"] = {"not_read": "refused at the kill switch"}
        _refuse(record, f"kill switch {FUSED_KILL_SWITCH}=0")
        return None

    # (2) The enable. It is also where a RETIRED switch name and an UNRECOGNISED
    # enable value are refused: `dispatch_enabled` raises on either, deliberately
    # (the retired name first), and the ladder's contract is that every rung answers
    # with a NAMED reason and none of them raise. The message is the raiser's own,
    # so neither rule is softened by being caught — it is only stopped from
    # arriving as "internal error".
    try:
        enabled = dispatch_enabled()
    except RuntimeError as exc:
        record["environment"] = {"not_read": (
            "refused at an unrecognised enable value"
            if isinstance(exc, DispatchEnableValueError)
            else "refused at a retired switch name")}
        _refuse(record, str(exc))
        return None
    if not enabled:
        # ONE REFUSAL, NAMING THE VARIABLE AND THE VALUE THAT DISABLED IT. With the
        # default on, ``0`` is the only value that reaches here: unset dispatches and
        # every other value was refused by name above. The value is read rather than
        # assumed so the text stays true if the default is ever turned off again.
        record["environment"] = {"not_read": "refused at the enable"}
        _refuse(record, f"dispatch disabled by {DISPATCH_ENABLE}="
                        f"{os.environ.get(DISPATCH_ENABLE, '<unset>')}")
        return None

    # (3) THE BACKEND, AND WITH IT THE KERNEL TABLE. Decided without importing cupy;
    # torch is imported only to read the MPS device for a NumPy engine, and from
    # FdtdDriver the only NumPy engine that gets here is an Apple GPU driver's
    # (``prefer_gpu=True``) — a ``prefer_gpu=False`` driver is the NumPy reference and
    # never calls this planner. On a host with neither engine's device the array path
    # is all there is, and there is nothing to launch a kernel on.
    #
    # THE HARDWARE PICKS THE CANDIDATE SET AND A PREFERENCE SELECTS WITHIN IT
    # (:func:`candidate_tables`). Splitting the rung that way is what makes
    # ``MEEP_GPU_KERNEL_TABLE`` naming an absent table a NAMED REFUSAL rather than a
    # silent fallback to whatever this host does have: a run that asked for Metal on
    # an NVIDIA box and quietly got Triton would publish an artifact describing a
    # table its caller never asked for, and nothing in it would say so.
    #
    # THE SUBSTRING "backend is not CuPy" SURVIVES VERBATIM in the empty-set
    # refusal. Two independent readers classify by it — ``test_dispatch_contract``
    # and ``stress_cuda_scale.py:1211``, and the Metal route gate's own classifier
    # table (``gate_dispatch_metal_route.py:452``) — and a classifier that stops
    # recognising the rung it watches reports REFUSAL-CHANGED rather than the
    # narrowing, which is a failure this seam has already paid for once.
    xp = getattr(grid, "xp", None)
    backend = getattr(xp, "__name__", None)
    tables = candidate_tables(grid)
    preference = os.environ.get(KERNEL_TABLE_SWITCH)
    record["environment"] = {
        "backend": backend,
        "candidate_tables": list(tables),
        "kernel_table_switch": KERNEL_TABLE_SWITCH,
        "kernel_table_preference": preference,
    }
    wanted = (preference or "").strip()
    if wanted and wanted not in tables:
        # THE PREFERENCE IS ANSWERED BEFORE THE EMPTY SET, so a caller who NAMED a
        # table is told about the table they named rather than about the hardware in
        # general. Both are refusals and both reach the array path; only one of them
        # tells the caller their request was seen.
        _refuse(record, f"{KERNEL_TABLE_SWITCH}={wanted} names no kernel table this "
                        f"host can run; the candidate set is "
                        f"{', '.join(tables) or 'empty on this host'}. The "
                        "preference selects WITHIN what the hardware admits and "
                        "never widens it")
        return None
    if not tables:
        _refuse(record, "backend is not CuPy; the array path is the reference "
                        "here, and no MPS device is available on this host "
                        f"(this engine's array module is {backend!r})")
        return None
    # (3b) THE UNCERTIFIED OPT-IN'S VALUE, answered before any table's identity is
    # read and on every table, the Metal one included. An unrecognised value is
    # refused here whether or not this host's identity is certified: read only
    # where an identity had failed, it would ride unseen on a certified host and
    # answer for the first time on the host it was typed for.
    opt_in_refusal = _uncertified_value_refusal(os.environ.get(UNCERTIFIED_SWITCH))
    if opt_in_refusal is not None:
        _refuse(record, opt_in_refusal)
        return None
    # THE PREFERENCE NARROWS THE SET; IT DOES NOT COLLAPSE IT. On a CuPy host the
    # hardware admits BOTH NVIDIA tables and rung 4d decides which composes first,
    # so this rung's answer is a LIST. Naming one here narrows that list to the one
    # named, which is the honest reading of "selects within": a run that asked for
    # the hand-CUDA table alone must not silently get a step half-composed by Triton.
    candidates = [wanted] if wanted else list(tables)
    table = candidates[0]
    record["table"] = table
    record["environment"]["table"] = table
    if table == METAL_TABLE:
        # THE METAL LADDER IS RUNGS 4M-9M AND IT LIVES IN ITS OWN MODULE, imported
        # HERE and never at module level: a NumPy step that opted out must not pay
        # for a table it will not use, and ``test_package_boundary`` measures
        # exactly that. ``_finish`` is handed down rather than imported there so the
        # two ladders cannot drift on slot ordering, pair splitting, the
        # certification lookup or emission.
        from .metal_dispatch import decide as _decide_metal  # noqa: PLC0415

        def _finish_metal(**kwargs: Any) -> Optional[FastPathPlan]:
            """:func:`_finish` with ``fields`` bound by the caller that has it.

            THE IDENTITY GUARD IS THE CALLER'S HALF TO SUPPLY.
            :meth:`FastPathPlan.dispatch` opens with ``if fields is not
            self.fields: return False``, so a plan built without it answers False at
            every consult — measured: the Metal composition dispatches four slots and
            the plan then launches nothing, syncs nothing, and publishes a record
            that says "fused". Fail-closed and loud (the launch counters stay at
            zero, which is the vacuity the gates' witnesses catch), but wrong.

            BINDING IT HERE RATHER THAN ASKING THE LADDER FOR IT is the placement
            that cannot be forgotten: ``plan_fast_path`` was handed this ``Fields``
            and every ladder it hands off to was handed the same one, so there is
            nothing for a second ladder to get right. ``setdefault`` leaves the
            override open — a ladder that passes ``fields`` explicitly wins — so this
            is a floor rather than a ceiling.
            """
            kwargs.setdefault("fields", fields)
            return _finish(**kwargs)

        return _decide_metal(record, fields, pml, grid, sources, xp,
                             finish=_finish_metal)

    # (4) THE TABLES' OWN CANDIDACY, ASKED PER TABLE AND ANSWERED PER TABLE.
    #
    # THIS RUNG STOPPED BEING A WHOLE-PLAN REFUSAL WHEN THE SECOND NVIDIA TABLE
    # LANDED, and the change is the point rather than a loosening. A version no gate
    # ran on is not a small difference — Triton owns the code generation the
    # bit-identity certifications are statements about, so an unvalidated compiler
    # invalidates them wholesale — but that is a statement about the TRITON TABLE,
    # and answering it by refusing the plan would take the hand-CUDA table down with
    # it on a host where Triton is simply absent. Each table is dropped BY NAME with
    # its own reason; the plan is refused only when the candidate set empties, and
    # that refusal quotes both reasons so a reader never has to guess which half
    # failed.
    from . import fastpath_cuda  # noqa: PLC0415

    tables_block: Dict[str, Dict[str, Any]] = {}
    record["tables"] = tables_block
    triton_reason: Optional[str] = None
    triton_version: Optional[str] = None
    if "triton" not in candidates:
        triton_reason = (f"{KERNEL_TABLE_SWITCH}={wanted} selected another table; "
                         "this one was not consulted")
    else:
        try:
            import triton  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - a broken install is as absent as a missing one
            record["environment"].update({"triton": None,
                                          "triton_import_error": repr(exc)})
            triton_reason = f"Triton is not importable on this host ({exc!r})"
        else:
            triton_version = str(getattr(triton, "__version__", "unknown"))
            validated = validated_triton_versions()
            if triton_version not in validated:
                record["environment"].update(
                    {"triton": triton_version, "triton_certified": False,
                     "validated_triton_versions": list(validated)})
                if uncertified_allowed():
                    # ADMITTED BY THE OPT-IN, and recorded as read: the table stays
                    # a candidate and ``triton_certified`` stays False.
                    _admit_uncertified(record, "triton", "Triton", triton_version,
                                       validated)
                else:
                    triton_reason = (
                        f"Triton {triton_version} is not in the recorded "
                        f"validated_triton_versions {list(validated)}"
                        + UNCERTIFIED_HINT)
    tables_block["triton"] = {"candidate": triton_reason is None,
                              "refused_because": triton_reason}
    if CUDA_TABLE not in candidates:
        tables_block[CUDA_TABLE] = {
            "candidate": False,
            "refused_because": (f"{KERNEL_TABLE_SWITCH}={wanted} selected another "
                                "table; this one was not consulted")}
    else:
        tables_block[CUDA_TABLE] = fastpath_cuda.cuda_candidate(record, xp)
    candidates = [name for name in candidates
                  if tables_block.get(name, {}).get("candidate")]
    if not candidates:
        _refuse(record, "no kernel table is a candidate on this host: "
                        + "; ".join(
                            f"{name}: {tables_block[name]['refused_because']}"
                            for name in sorted(tables_block)))
        return None

    record["subnormal"] = _subnormal_block(candidates[0])

    # ``launch`` is NOT imported here any more: the only thing this ladder read off
    # it was ``STEP_ORDER``, and that read moved into :func:`_finish` with the tail
    # every table shares.
    #
    # THE PACKAGE IS IMPORTED EVEN WHERE ITS COMPOSER IS NOT CONSULTED, and that is
    # deliberate rather than a leak. ``triton_kernels.launch`` imports Triton lazily
    # per builder and the module itself imports cleanly with none installed, which is
    # what makes ``STEP_ORDER`` — the ONE definition of the driver's slot order, the
    # thing both other composers re-import — readable on a Triton-less host at all.
    # The expansion probe is likewise ONE artifact that licenses every table's
    # complex arms; reading a JSON record out of the package that owns the licence
    # functions is not composing with it. What IS conditional is the call to
    # ``plan_step`` below.
    from . import triton_kernels  # noqa: PLC0415

    probe = None
    probe_error = None
    try:
        probe = triton_kernels.load_expansion_probe()
    except Exception as exc:  # noqa: BLE001 - a missing licence is a refusal, not a crash
        probe_error = repr(exc)
    # UPDATED RATHER THAN REPLACED: rung 3 wrote the candidate set, the switch and
    # the chosen table into this block, and those are facts about the decision that
    # a fuller environment read has no business dropping.
    record["environment"].update(_environment_block(grid, probe, probe_error, table))
    record["run_shape"] = _run_shape(fields, pml, grid)

    # (4c) The complex-expansion licence's POLICY, against the one this dispatch
    # will run under. Dropping a probe artifact that does not match is the only
    # honest answer, and it has to happen HERE — before ``plan_step`` consumes
    # the record at the composer rung below, and four rungs before
    # ``_subnormal_gate`` installs anything.
    #
    # WHY THE ORDER IS THE BUG AND NOT AN INCONVENIENCE. On shipped x86 the
    # artifact this platform naturally cuts is stamped 'flush' (CuPy appends
    # -ftz=true to every NVRTC compile), while CERTIFICATION_SUBNORMAL_POLICY is
    # 'keep' and the gate below refuses any process that installed anything else.
    # So the natural artifact and the only permitted run policy disagree BY
    # CONSTRUCTION. Until 2026-08-15 nothing compared them: the flush-cut licence
    # was consumed with zero refusals by a run whose fields would be stepped under
    # keep — the policy under which the three zero-imaginary patterns that license
    # `_mul_field_left` / `_mul_coefficient_left`, i.e. every real-coefficient
    # multiply in both complex kernels, are NOT bit-identical between the arms.
    # ``POLICY_CONDITIONAL_LICENCE`` said this did not transfer; nothing enforced
    # it at either seam that consumes a record (launch.py:2018-2049,
    # fastpath.py:1701/1722).
    #
    # The probe is DROPPED rather than the whole plan refused: a refused record
    # makes every complex predicate refuse by name, which is the array path for
    # the complex arms and no change at all for a run that has no complex storage.
    #
    # HOW THE DROP IS SPELLED, AND WHY IT USED NOT TO WORK. Until 2026-08-16 this
    # rung expressed the refusal by assigning ``None`` to its OWN LOCAL NAME.
    # That withholds the caller's copy; it does not disqualify the artifact — and
    # ``None`` is also how "no artifact was offered" is spelled, which every rung
    # below reads as a licence to open the file itself. Measured on the shipped
    # path: this clause refused a flush-cut artifact against the required 'keep',
    # and ``plan_step`` re-read the same environment variable and bound complex
    # arms on 4 of 7 slots over 8004 launches. The drop was advisory where it had
    # to bind.
    #
    # The refusal is now a VALUE the rungs below cannot mistake for absence, and
    # it is recorded against the environment variable for the composition window
    # so a rung that reads the variable directly gets the refusal back instead of
    # the artifact. Both halves are installed at the composer call below.
    probe_refusal_reasons: List[str] = []
    if probe is not None:
        probe_refusal_reasons = _complex_probe_policy_reasons(probe)
        # (4c) THE SAME DROP FOR THE ARCHITECTURE, for the same reason the policy is
        # checked: the expansion licence is a measurement of how one device's compiler
        # orders a multiply-add, and ``environment_default`` keys its table on the
        # artifact's CLAIMED backend/machine/CuPy — none of which distinguishes one
        # NVIDIA architecture from another. So a probe cut on 8.6 and carried onto a
        # 9.0 host would license FMA_V1 there with nothing measured on 9.0. Dropped
        # only when BOTH capabilities were read and differ: a probe that records none,
        # or a host whose device will not identify itself, is reported rather than
        # refused, because refusing on an unread fact asserts one.
        probe_refusal_reasons += _probe_capability_reasons(
            probe, (record["environment"].get("device") or {}).get("compute_capability"))
        if probe_refusal_reasons:
            record["environment"]["complex_expansion_probe_dropped"] = probe_refusal_reasons
            record["environment"]["complex_expansion_probe_resolved"] = False

    # (4b) The DEVICE, ASKED PER TABLE. Both NVIDIA tables generate or assemble code
    # FOR AN ARCHITECTURE — Triton's PTX and the hand-written kernels' NVRTC compile
    # alike — so a compute capability no gate ran on invalidates a bit-identity claim
    # the same way an unvalidated Triton does. Refused only when the identity was
    # actually READ: the per-table answer is None on a host whose device will not
    # identify itself, and refusing on an unread fact would be asserting one.
    #
    # A FALSE DROPS THAT TABLE, NOT THE PLAN. The two lists come from different
    # campaigns and neither implies the other, so a device one table was certified on
    # and the other was not is a real state rather than a contradiction; the whole
    # plan is refused only when the drop empties the candidate set.
    certified_by_table = record["environment"].get("device_certified_by_table") or {}
    capability = (record["environment"].get("device") or {}).get("compute_capability")
    for name in list(candidates):
        if certified_by_table.get(name) is False:
            recorded = (record["environment"]
                        .get("validated_compute_capabilities_by_table", {})
                        .get(name, []))
            if uncertified_allowed():
                # ADMITTED BY THE OPT-IN: the table stays a candidate, and its
                # ``device_certified_by_table`` answer stays False.
                _admit_uncertified(record, name, "GPU compute capability",
                                   capability, recorded)
                continue
            # NAME WHAT WOULD HAVE TO BE RE-RUN. The table's list is the intersection
            # of its cited welds' live capabilities, so "not certified here" is always
            # some set of welds with no live run on this architecture; a reader who is
            # certifying a new card needs those names, not just the verdict.
            blocking = [key for key, state in sorted(
                capability_admission(name)["by_key"].items())
                if capability is None
                or _normalized_capability(capability) not in state["live"]]
            tables_block[name] = {
                "candidate": False,
                "refused_because": (
                    f"GPU compute capability {capability} is not in the recorded "
                    f"validated_compute_capabilities {list(recorded)} for the "
                    f"{name} table" + UNCERTIFIED_HINT),
                "welds_without_a_live_run_here": blocking[:5],
                "welds_without_a_live_run_here_count": len(blocking)}
            candidates.remove(name)
    if not candidates:
        _refuse(record, "no kernel table is a candidate on this host: "
                        + "; ".join(
                            f"{name}: {tables_block[name]['refused_because']}"
                            for name in sorted(tables_block)))
        return None

    # (4d) PRECEDENCE. Which candidate composes FIRST, and therefore which table
    # holds every slot both of them admit. The ruling is a constant
    # (:data:`BACKEND_PRECEDENCE`) and the switch orders within it; a switch naming a
    # table that is not a candidate here is a NAMED refusal rather than a silent
    # fallback, for the same reason rung 3's is.
    order = backend_precedence(candidates)
    if isinstance(order, str):
        record["arbitration"] = {
            "precedence": list(BACKEND_PRECEDENCE),
            "preference_switch": BACKEND_PREFERENCE_SWITCH,
            "preference_value": os.environ.get(BACKEND_PREFERENCE_SWITCH),
            "candidates": list(candidates),
            "refused_because": order,
        }
        _refuse(record, order)
        return None
    table = order[0]
    record["table"] = table
    record["environment"]["table"] = table
    record["arbitration"] = {
        "precedence": list(BACKEND_PRECEDENCE),
        "preference_switch": BACKEND_PREFERENCE_SWITCH,
        "preference_value": os.environ.get(BACKEND_PREFERENCE_SWITCH),
        "effective_order": list(order),
        "candidates": list(candidates),
        "why": ("Triton is the release-gated incumbent and the hand-CUDA table "
                "fills its refusals; measured-fastest-per-arm replaces this order "
                "once an idle-box timing round exists (plan section 5.2)"),
        "yield_pending_primary_slots": fastpath_cuda.YIELD_PENDING_PRIMARY_SLOTS,
    }

    # (5) The composer. Never raises and never returns None; a configuration
    # nothing covers is a plan that replaces nothing, which is the array path.
    # fuse/fuse_ade are asked for exactly the arms a gate drove through this seam
    # on a configuration like this one, plus whatever the opt-in adds.
    #
    # TWO THINGS THIS DISPATCH KNOWS AND THE RUNGS BELOW CANNOT READ, stated here
    # for the duration of the composition — which is the window in which every
    # EXPANSION constexpr is bound:
    #
    #   THE REFUSAL. ``refusing_expansion_probe`` yields the value handed to
    #   ``plan_step`` AND records it against the environment variable, so the two
    #   routes THIS DISPATCH CONTROLS — the argument and the file — carry the same
    #   verdict. Recording it is not belt-and-braces: passing a value only binds
    #   the rungs the value reaches, and the whole defect was a rung that read
    #   the file.
    #
    #   THERE ARE THREE ROUTES, NOT TWO, and the third is covered by neither
    #   half. The VALUE is caught by the by-name clause every funnel opens with;
    #   the VARIABLE is caught by the reader, which consults the standing refusal
    #   in its own function body. A RAW DICT handed straight to a plan builder
    #   arrives by neither and is caught by neither: it never passes the reader,
    #   so the standing refusal is not consulted, and it is a ``dict`` rather
    #   than a ``RefusedExpansionProbe``, so the by-name clause does not fire.
    #   MEASURED 2026-08-16 through the shipped composer: with a standing refusal
    #   in force and ``keep`` declared, a raw keep-cut dict scored zero reasons at
    #   all five gating seams and bound FMA_V1 — the refusal was simply not in
    #   the path. What still stops a raw dict is the POLICY clause, which judges
    #   the record itself rather than how it arrived: the same leg with a
    #   flush-cut dict refused at all five. So the coverage is "any dict is
    #   judged on its merits, and the standing refusal adds nothing for a route
    #   that bypasses the reader" — not "a refused artifact cannot come back".
    #   No in-tree caller passes a raw dict to a plan builder; every route
    #   through ``fastpath`` goes via the argument or the reader.
    #
    #   THE POLICY. Rung 8b installs CERTIFICATION_SUBNORMAL_POLICY below and
    #   refuses any process that installed anything else, so it is a fact this
    #   dispatch owns — but it is not INSTALLED yet, and nothing under
    #   ``plan_step`` can read an intention. Without the declaration the library's
    #   policy clause would be asked a question it cannot answer at every rung of
    #   every dispatch, and that clause now fails closed (see
    #   ``complex_fields.expansion_policy_reasons``): declaring is what makes an
    #   honest run judgeable instead of refused.
    #
    #   THE FUSION OPT-IN AND THE RELEASE. ``fuse``/``fuse_ade`` were literals
    #   here until 2026-08-29, which made clause (8) below unfalsifiable: no fused
    #   product could reach the seam, so the sentence "no fused arm has been driven
    #   through the driver seam" could never stop being true. They are now read
    #   from the union of :data:`FUSE_ARMS_SWITCH` and the arms the driver-route
    #   gate RELEASED for this configuration.
    #
    #   THE ENVELOPE GATES THE REQUEST, NOT ONLY THE ANSWER, and the order is
    #   load-bearing. Asking the composer for fusion on a configuration the
    #   release does not cover and refusing the result at clause (8) would turn a
    #   run that dispatches its separate arms TODAY into a whole-plan refusal —
    #   the fused arm wins the slot, and clause (8) refuses the plan, not just the
    #   fusion. Deciding the release BEFORE the call means an unreleased
    #   configuration asks for, and gets, the composition it gets today, byte for
    #   byte. Clause (8) stays as the backstop for a fused label that arrives
    #   anyway.
    #
    #   THE VETO SUBTRACTS, AND IT IS APPLIED TO THE ADMISSION AND NOT TO THE
    #   RELEASE. ``released_here`` goes on reporting what the release covers for
    #   this configuration — a fact about the shape, which a switch does not
    #   change — while ``admitted`` empties, so an artifact from a vetoed run says
    #   "the release applied here and this run declined it" rather than "the
    #   release did not apply". :data:`FUSE_ARMS_VETO` is what keeps the
    #   dispatch-without-fusion baseline reachable at all.
    #
    #   THE ADMISSION IS PER TABLE, because the release is. The two NVIDIA tables
    #   were driven through this seam by two different campaigns on two different
    #   case sets, and a shared admitted set would let one table's measurement admit
    #   the other's product. ``released_here`` and ``admitted`` keep their meaning as
    #   the UNION a reader of the artifact sees; the per-table halves sit beside them
    #   under ``released_here_by_table`` / ``admitted_by_table``, which is what each
    #   composer is actually handed.
    opted_in = requested_fused_arms()
    vetoed = fused_arms_vetoed()
    envelope_reasons = fused_release_reasons(record["run_shape"])
    released = released_fused_arms(record["run_shape"])
    cuda_released = fastpath_cuda.released_fused_arms(record["run_shape"])
    cuda_envelope_reasons = fastpath_cuda.fused_release_reasons(record["run_shape"])
    triton_opted = tuple(label for label in opted_in
                         if _table_of(label) != CUDA_TABLE)
    cuda_opted = tuple(label for label in opted_in
                       if _table_of(label) == CUDA_TABLE)
    admitted_by_table: Dict[str, Tuple[str, ...]] = {
        "triton": () if vetoed else tuple(sorted(set(triton_opted) | set(released))),
        CUDA_TABLE: (() if vetoed
                     else tuple(sorted(set(cuda_opted) | set(cuda_released)))),
    }
    admitted = tuple(sorted(set(admitted_by_table["triton"])
                            | set(admitted_by_table[CUDA_TABLE])))
    released_by_table = {"triton": list(released), CUDA_TABLE: list(cuda_released)}
    # THE FLAT KEYS REPORT THE PRIMARY TABLE AND THE MAPS REPORT BOTH, which is the
    # only reading that is true on every record. A UNION under ``released_here``
    # would say a Triton-primary run's release covers products its composer cannot
    # install; the primary's own view is what "the release applied here" has always
    # meant, and it is byte-identical to what this key said before the second table
    # existed on every run that has one.
    record["fusion"]["released_here"] = released_by_table[table]
    record["fusion"]["released_here_by_table"] = released_by_table
    record["fusion"]["admitted"] = list(admitted_by_table[table])
    record["fusion"]["admitted_by_table"] = {
        name: list(values) for name, values in admitted_by_table.items()}
    record["fusion"]["outside_the_released_envelope"] = list(
        cuda_envelope_reasons if table == CUDA_TABLE else envelope_reasons)
    record["fusion"]["outside_the_released_envelope_by_table"] = {
        "triton": list(envelope_reasons), CUDA_TABLE: list(cuda_envelope_reasons)}
    # THE PER-ARM HALF, REPORTED WHENEVER IT REFUSES SOMETHING THE SHARED TABLE
    # DID NOT. Without it an artifact from, say, a conductive 2-D grid would carry
    # an EMPTY ``outside_the_released_envelope`` beside a ``released_here`` missing
    # two arms, and a reader would have no way to tell a narrowed release from a
    # bug. Only the refusing arms appear: a row per admitted arm would be a list of
    # empty tuples on every dispatching run.
    per_arm = {arm: list(fused_release_arm_reasons(record["run_shape"], arm))
               for arm in sorted(RELEASED_FUSED_ARMS)}
    per_arm = {arm: why for arm, why in per_arm.items() if why}
    if per_arm and not envelope_reasons:
        record["fusion"]["outside_this_arms_own_cases"] = per_arm
    cuda_per_arm = {
        arm: list(fastpath_cuda.fused_release_arm_reasons(record["run_shape"], arm))
        for arm in sorted(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)}
    cuda_per_arm = {arm: why for arm, why in cuda_per_arm.items() if why}
    if cuda_per_arm and not cuda_envelope_reasons:
        record["fusion"].setdefault("outside_this_arms_own_cases_by_table", {})[
            CUDA_TABLE] = cuda_per_arm
    #   THE OFFER IS PER LABEL, NOT PER RUN, and 2026-09-02 is when that stopped
    #   being a distinction without a difference. ``fuse`` was the whole of what
    #   this call could say, so the composer installed EVERY product whose
    #   predicate admitted — including labels this ladder cannot run. A fused
    #   product occupies BOTH slots of its seam, so clause (8) below could only
    #   reject it WHOLE, taking the separate certified arms underneath it down as
    #   well. Measured on the device: a conductive 2-D PML grid that dispatched
    #   ``fused pair B`` plus four arms went entirely to the array path once
    #   ``conductive_fused_electric_pair`` was wired, because its label has no
    #   ledger entry and cannot be admitted. ``fuse_labels`` hands the composer the
    #   admitted set, so an un-admitted product is never installed and its seam
    #   keeps the plans it had — which is the sentence the paragraph above already
    #   made about the envelope, applied at the granularity the release decides at.
    #
    #   EVERY CANDIDATE IS COMPOSED, IN EFFECTIVE ORDER, INSIDE ONE WINDOW. The
    #   window is what binds the EXPANSION constexprs and the declared policy, and
    #   both tables' complex products read them — so composing the second one outside
    #   it would license its arms under facts this dispatch had stopped asserting.
    #   The merge then walks whole units: see
    #   :func:`meep_gpu.fastpath_cuda.merge_tables` for which units a secondary table
    #   may take and why a pair is never split in either direction.
    composed: List[Tuple[str, Any]] = []
    with _composition_window(probe, probe_refusal_reasons) as offered_probe:
        for name in order:
            if name == "triton":
                triton_admitted = admitted_by_table["triton"]
                composed.append(("triton", triton_kernels.plan_step(
                    fields, pml,
                    fuse=bool(set(triton_admitted) - {FUSED_ADE_STATE_LABEL}),
                    fuse_ade=FUSED_ADE_STATE_LABEL in triton_admitted,
                    fuse_labels=triton_admitted,
                    sources=sources, probe=offered_probe)))
                continue
            # THE ONE ``cuda_kernels`` IMPORT IN THIS FILE, function-local and BELOW
            # the backend rung: a NumPy step never reaches this line, which is what
            # ``test_package_boundary`` measures, and the AST test in
            # ``test_triton_kernels`` pins that this is the only one and that it sits
            # after the rung-3 check rather than at module scope.
            from .cuda_kernels import arms as _cuda_arms  # noqa: PLC0415

            cuda_admitted = admitted_by_table[CUDA_TABLE]
            composed.append((CUDA_TABLE, _cuda_arms.plan_step(
                fields, pml, grid,
                licenses=fastpath_cuda.licences_from_probe(offered_probe),
                subnormal_policy=TABLE_SUBNORMAL_POLICY[CUDA_TABLE],
                sources=sources,
                fuse=bool(cuda_admitted),
                fuse_labels=[_bare(label) for label in cuda_admitted])))
    step_plan = fastpath_cuda.merge_tables(
        composed,
        pending_of={"triton": PENDING_DEVICE_GATE_ARMS,
                    CUDA_TABLE: fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS},
        record=record)
    selected = dict(getattr(step_plan, "selected", {}) or {})
    reasons = dict(getattr(step_plan, "reasons", {}) or {})
    filled = set(getattr(step_plan, "plans", {}) or {})

    # (8) NO FUSION EXCEPT BY NAME. Every fused label refuses the whole plan
    # unless a gate RELEASED that exact label for this exact configuration, or
    # this process opted into that exact label. The composition is a
    # cross-sub-step substitution the driver's five consults cannot see into, and
    # what licenses one is a gate that drove it through those consults, which is
    # a fact about a gate run and not about the plan in hand.
    #
    # THE RELEASE IS NARROW ON PURPOSE AND THE REFUSAL SURVIVES IT. Nine arms have
    # been driven through the seam (:data:`RELEASED_FUSED_ARMS`), on the
    # configurations :data:`FUSED_RELEASE_ENVELOPE` and
    # :data:`FUSED_RELEASE_ARM_AXES` pin between them. Twenty further fused labels
    # are selectable by the shipped composer and none is admitted here: seventeen
    # are in :data:`PENDING_DEVICE_GATE_ARMS` for want of a ledger entry, three
    # (the H->D products) are in it and additionally declare ``INSTALLABLE = False``
    # in their own modules, and ``fused ADE state`` has never been driven through a
    # driver seam at all.
    # Since 2026-09-02 they do not usually reach this clause either — the composer
    # is handed the admitted set and does not install them — so this stays the
    # backstop it was written as rather than the thing that stops them.
    #
    # THE REFUSAL IS PER LABEL, not per opt-in: asking for ``fused pair B`` does
    # not admit ``fused pair D`` riding along in the same step, because "the two
    # halves of the step were each measured" is not "the step was measured".
    fused_slots = sorted(slot for slot in filled
                         if arm_is_fused(selected.get(slot, "")))
    ungated = sorted({selected[slot] for slot in fused_slots
                      if selected[slot] not in admitted})
    if ungated:
        _record_slots(record, step_plan, (), {}, {})
        # NAME WHAT IS REFUSED AND WHY IT IS NOT COVERED. Two different things
        # land here and a reader has to be able to tell them apart: a label no
        # gate has driven at all, and a released label on a configuration its
        # gate did not run. The envelope reasons are what distinguish them, so
        # they are quoted rather than summarised.
        #
        # AND A THIRD THING LANDS HERE NOW: a label of the OTHER table. Each label
        # is judged against its OWN table's release and cites its own gate, because
        # the two were driven by two campaigns and quoting one beside the other's
        # label would name a gate that never ran that product.
        detail_parts: List[str] = []
        for name, releases, gate, reasons_here in (
                ("triton", RELEASED_FUSED_ARMS, DRIVER_ROUTE_FUSED_GATE,
                 envelope_reasons),
                (CUDA_TABLE, fastpath_cuda.CUDA_RELEASED_FUSED_ARMS,
                 fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE,
                 cuda_envelope_reasons)):
            mine = [arm for arm in ungated
                    if (_table_of(arm) or "triton") == name]
            if not mine:
                continue
            part = (f"released by {gate}: {', '.join(sorted(releases))}")
            inside = [arm for arm in mine if arm in releases]
            if inside and reasons_here:
                part += (f" — but {', '.join(inside)} is released only for the "
                         "configuration that gate drove, and this one is outside "
                         "it: " + "; ".join(reasons_here))
            detail_parts.append((name, part))
        # THE SINGLE-TABLE SENTENCE IS THE ONE IT ALWAYS WAS, down to the
        # punctuation: the gate's classifier and this file's own tests parse it, and
        # a table name prepended to a refusal that names one table would be a
        # changed refusal reported as a narrowing. The prefix appears only when the
        # refusal genuinely spans two tables, where the reader needs it.
        if len(detail_parts) > 1:
            detail = " | ".join(f"table {name}: {part}"
                                for name, part in detail_parts)
        else:
            detail = detail_parts[0][1] if detail_parts else ""
        # A VETOED RUN IS TOLD IT VETOED, not told to set a switch it has already
        # set. This branch is defence in depth rather than a path a composer can
        # take today — the veto empties ``admitted``, so ``fuse`` is False and no
        # fused label is selected — and it is worded for the day one arrives anyway.
        route = (f"{FUSE_ARMS_SWITCH}={FUSE_ARMS_VETO} vetoed every fused arm, "
                 "this one included" if vetoed
                 else f"set {FUSE_ARMS_SWITCH} to drive one anyway")
        _refuse(record, "a fused cross-sub-step product won "
                        f"{', '.join(fused_slots)}; it has not been driven through "
                        "the driver seam on this configuration. Not admitted: "
                        f"{', '.join(ungated)} ({detail}; {route})")
        return None
    if fused_slots:
        # DRIVEN, and the record has to say by what: a reader who sees
        # ``step_path: fused`` with a fused arm in two slots needs to know whether
        # they are looking at the released composition or at an opt-in reaching
        # past it. ``ARM_CERTIFICATION`` then names each arm's own byte gate
        # beside it, one level down.
        record["fusion"]["driven"] = {slot: selected[slot] for slot in fused_slots}
        driven_releases: Dict[str, Tuple[str, ...]] = dict(RELEASED_FUSED_ARMS)
        driven_releases.update(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)
        record["fusion"]["released_on_cases"] = {
            arm: list(driven_releases[arm])
            for arm in sorted({selected[slot] for slot in fused_slots})
            if arm in driven_releases}
        # THE STAMP READS BOTH HALVES OF THE RELEASE, since 2026-09-02. It used to
        # fire on the shared envelope alone, which was total while the envelope was;
        # once ``dimensions`` and ``folded`` moved to :data:`FUSED_RELEASE_ARM_AXES`
        # an opt-in could drive a released arm on a shape its own cases never ran
        # and carry NO stamp — an artifact that reads as the released composition.
        # A driven arm is outside the release when either half refuses it.
        driven_arms = sorted({selected[slot] for slot in fused_slots})
        beyond: List[str] = []
        if any(_table_of(arm) != CUDA_TABLE for arm in driven_arms):
            beyond.extend(envelope_reasons)
        if any(_table_of(arm) == CUDA_TABLE for arm in driven_arms):
            beyond.extend(cuda_envelope_reasons)
        for arm in driven_arms:
            if _table_of(arm) == CUDA_TABLE:
                if arm in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS:
                    beyond.extend(f"{arm}: {reason}" for reason in
                                  fastpath_cuda.fused_release_arm_reasons(
                                      record["run_shape"], arm))
            elif arm in RELEASED_FUSED_ARMS:
                beyond.extend(f"{arm}: {reason}" for reason
                              in fused_release_arm_reasons(record["run_shape"], arm))
        if beyond:
            record["fusion"]["driven_outside_the_released_envelope"] = {
                "arms": driven_arms,
                "reasons": beyond,
                "reached_by": FUSE_ARMS_SWITCH,
            }

    # (7) The null drop, before completeness is judged.
    #
    # THE LABEL IS COMPARED BARE, because both NVIDIA composers spell their null
    # constitutive arm the same way — measured on ``no_pml_2d``, where each selects
    # ``"no-PML null"`` at ``update_H`` — and the CUDA one arrives here as
    # ``"cuda:no-PML null"``. A namespaced null that reached the dispatch set would
    # launch nothing while the record called the step fused, which is the exact
    # vacuity this rung exists to drop.
    dropped_null = {slot: selected[slot] for slot in sorted(filled)
                    if _bare(selected.get(slot, "")) == NULL_ARM_LABEL}
    dispatchable = filled - set(dropped_null)

    # An arm may be planner-visible before its dedicated CUDA gate is available.
    # That is useful for composition coverage, but dispatching it would turn a
    # planned validation into a user-visible numerical claim.  Refuse the WHOLE
    # step: running only siblings is an unmeasured mixed composition.
    #
    # A FUSED LABEL IS ASKED ABOUT ITS CONSTITUENTS, not just about itself, and the
    # 2026-09-02 installer wave is what made that necessary. A fused product writes
    # ONE label into BOTH slots it absorbs, so the arms it implements disappear from
    # ``selected`` — and ``complex_conductive_fused_pair`` absorbs two arms that are
    # both pending. Reading the label alone would have let a wiring edit lift a
    # standing refusal without anything measuring that it had.
    # :data:`FUSED_ARM_CONSTITUENTS` is the see-through, and a fused label MISSING
    # from it is treated as covering nothing rather than as covering the empty set:
    # it is refused here by name, because a fused product whose constituent arms
    # this file cannot name is exactly the product no rung below can judge.
    #
    # BOTH MAPS ARE CONSULTED, ROUTED BY THE LABEL'S OWN TABLE. A merged step can
    # hold a Triton arm and a hand-CUDA product side by side, and each one's absorb
    # declaration and pending reason live with its own table's rows. Routing by the
    # PLAN's table instead would ask the Triton map about a CUDA label, find nothing,
    # and refuse a released product as unmapped — a refusal about the lookup rather
    # than about the evidence.
    constituents: Dict[str, Tuple[str, ...]] = dict(FUSED_ARM_CONSTITUENTS)
    constituents.update(fastpath_cuda.CUDA_FUSED_ARM_CONSTITUENTS)
    pending_map: Dict[str, str] = dict(PENDING_DEVICE_GATE_ARMS)
    pending_map.update(fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS)
    selected_labels = {selected.get(slot, "unknown") for slot in dispatchable}
    unmapped_fused = sorted(label for label in selected_labels
                            if arm_is_fused(label)
                            and label not in constituents)
    if unmapped_fused:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        _record_slots(record, step_plan, (), dropped_null, {})
        _refuse(record, f"{', '.join(unmapped_fused)} won a slot, and this module "
                        "cannot name the arms it substitutes; the pending-gate rung "
                        "and every arm-level record below it would be reporting on "
                        "an absorb nobody declared (FUSED_ARM_CONSTITUENTS)")
        return None
    substituted = set(selected_labels)
    for label in selected_labels:
        substituted |= set(constituents.get(label, ()))
    pending = sorted(substituted & set(pending_map))
    if pending:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        _record_slots(record, step_plan, (), dropped_null, {})
        _refuse(record, "; ".join(pending_map[arm] for arm in pending))
        return None

    # (6) Completeness. A filled slot the seam cannot run refuses everything.
    #
    # THE FILLS LEFT THIS RUNG ON 2026-09-02. It used to be the rung that blocked
    # every folded configuration: the driver-route gate drove both folded cases with
    # the fused route opted in, ``fused pair B (folded)`` was selected and admitted
    # at clause (8), and then the mirror-fill arm won ``fill_B``/``fill_D`` and this
    # rung refused the plan — in both probe configurations, on both shapes. That was
    # never narrowable by a predicate, and the 2026-08-29 note said so: what retires
    # it is a CONSULT, not a rule. Those consults now exist — ``driver.step``'s
    # ``fill_B``/``fill_D`` sites and the far-ghost sites it consults under
    # :data:`FAR_FILL_OWNERS` — the fills are in :data:`DRIVER_SLOTS`, and this rung
    # is back to what it was for: a slot a composer fills that the driver has no
    # call site for at all.
    unrunnable = sorted(dispatchable - set(DRIVER_SLOTS))
    if unrunnable:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in unrunnable}
        _record_slots(record, step_plan, (), dropped_null, {})
        _refuse(record, "the composer filled "
                        f"{', '.join(unrunnable)}, which the driver seam has no "
                        "consult for; running the rest would be a composition no "
                        "gate certified")
        return None

    # (6b) The FOLD, read from the grid rather than inferred from what the composer
    # did, and NARROWED on 2026-09-02 to the one hole rule 6 never covered.
    #
    # THE HOLE, unchanged and still the reason this rung exists. Rule 6 fires on a
    # filled fill slot; a mirror run whose fill arms both REFUSE fills nothing, so
    # rule 6 passes it and the folded curl and constitutive kernels dispatch with the
    # ghost fills left on the array path. Nothing about that composition is certified
    # — every folded gate ran the fill KERNELS in the same step — and it is dispatch
    # ENABLED BY a sibling product's refusal, which is the fail-open direction.
    #
    # WHAT THE CONSULTS DISCHARGED AND WHAT THEY DID NOT. Before them this rung had
    # to refuse every fold, because the seam could not run a fill slot at all. Now it
    # can, so the refusal is narrowed to exactly the census's own wording: a folded
    # run must have BOTH fill slots carrying a kernel. A fold whose fills are covered
    # is a composition the seam reproduces pass for pass; a fold whose fills are on
    # the array path while its curls are on kernels is still the mixed composition
    # nothing measured, and is still refused whole.
    #
    # WHAT THE NARROWING LICENSED, AND WHAT IT DID NOT. The narrowing let a folded
    # configuration REACH the rungs below; the driver-route gate re-run named by
    # :data:`DRIVER_ROUTE_FUSED_GATE` is what then released TWO folded labels —
    # ``fused pair B (folded)`` and ``fused pair D (folded)`` — on the axes
    # :data:`FUSED_RELEASE_ARM_AXES` pins for them. Every OTHER folded fused label
    # (complex, beta, off-diagonal, dispersive) is still refused by name at clause
    # (8) unless :data:`FUSE_ARMS_SWITCH` names it.
    fold = _fold_description(grid)
    uncovered_fills = sorted(set(FAR_FILL_PASSES) - dispatchable)
    if fold is not None and dispatchable and uncovered_fills:
        record["built_not_dispatched"] = {
            slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
        _record_slots(record, step_plan, (), dropped_null, {})
        # THE WORDING CARRIES THE GATE'S OWN NEEDLES. ``gate_dispatch_fused_route``
        # classifies which rung answered by substring ("mirror-folded", "fill
        # sub-steps"), and a classifier that stops recognising the rung it watches
        # reports REFUSAL-CHANGED rather than the narrowing — the same failure the
        # gate's clause-(8) needle already paid for once. Both phrases are kept.
        _refuse(record, f"the configuration is mirror-folded ({fold}); the driver "
                        "runs fill_symmetry_bc_B/D on every step and no kernel "
                        f"covers the fill sub-steps {', '.join(uncovered_fills)}, so "
                        "dispatching the rest would be a folded composition no gate "
                        "ran")
        return None

    # (8b) THE SUBNORMAL POLICY GATE. Last rung before anything compiles, and that
    # position is the whole mechanism: a policy installed after a compile is not
    # seen by the binary that already exists, so this must precede the warm pass —
    # and it must FOLLOW every structural refusal, because installing a policy
    # changes the process's arithmetic and a configuration that was never going to
    # dispatch has no business having its numerics moved. Gated on ``dispatchable``
    # for the same reason.
    policy_licence: Optional[Tuple[Any, str, int]] = None
    if dispatchable:
        refusal = _subnormal_gate(record, xp, table)
        if refusal is not None:
            _record_slots(record, step_plan, (), dropped_null, {})
            _refuse(record, refusal)
            return None
        from . import subnormal_policy as _policy  # noqa: PLC0415

        policy_licence = (_policy, TABLE_SUBNORMAL_POLICY[table],
                          _policy.policy_epoch())

    # (9) The warm pass: compilation becomes a plan-time event, where nothing has
    # been written and unfilling a slot is free.
    unwarmed: Dict[str, str] = {}
    warm_failures: Dict[str, str] = {}
    if warm_pass_enabled():
        # ``update_P``'s entries take the drive-field reader and rotate their
        # buffers; the warm pass snapshots and restores that rotation rather than
        # skipping the slot, because the slot it used to skip is the LAST one in the
        # step — a kernel that would not compile there raised after four sub-steps
        # had already advanced the fields, turning a plan-time refusal into a
        # half-applied step.
        drive = getattr(fields, "drive_field", None)
        for slot in sorted(dispatchable):
            try:
                skipped = warm_plan(step_plan.plans[slot], drive=drive)
            except Exception as exc:  # noqa: BLE001 - a kernel that will not compile unfills
                warm_failures[slot] = f"the warm pass failed: {exc!r}"
                continue
            if skipped is not None:
                unwarmed[slot] = skipped
    else:
        for slot in sorted(dispatchable):
            unwarmed[slot] = f"the warm pass is disabled by {WARM_SWITCH}=0"
    dispatchable -= set(warm_failures)
    for slot, reason in warm_failures.items():
        reasons[slot] = (reason,)
    step_plan.reasons.update({slot: reasons[slot] for slot in warm_failures})

    # Completeness again, now that the warm pass has moved the fill set. It cannot
    # fail here — unfilling only shrinks a subset — and it is re-applied anyway so
    # the invariant has one spelling rather than one spelling and one argument.
    still_unrunnable = sorted(dispatchable - set(DRIVER_SLOTS))
    if still_unrunnable:  # pragma: no cover - unreachable; the invariant, not a branch
        _record_slots(record, step_plan, (), dropped_null, unwarmed)
        _refuse(record, f"after the warm pass the composer still filled "
                        f"{', '.join(still_unrunnable)}, which the seam cannot run")
        return None

    # (6b) AGAIN, and unlike completeness this one CAN fail here. Rung 6b is judged
    # above, before anything compiles; the warm pass then unfills per slot, and a
    # fill slot that lost its kernel there would leave exactly the composition 6b
    # exists to refuse — folded curl and constitutive kernels dispatching with the
    # ghost fills back on the array path, this time enabled by a compile failure
    # rather than by a sibling product's refusal. No shipped fill plan can reach it
    # today (neither ``MirrorGhostFillPlan`` nor
    # ``FoldedMirrorGhostFillComplexPlan`` exposes a backing attribute
    # :func:`_warm_with_empty_grid` can empty, so both are RECORDED unwarmed rather
    # than unfilled), and that is a fact about two classes rather than an invariant,
    # so it is re-applied instead of argued.
    if fold is not None and dispatchable:
        lost_fills = sorted(set(FAR_FILL_PASSES) - dispatchable)
        if lost_fills:
            record["built_not_dispatched"] = {
                slot: selected.get(slot, "unknown") for slot in sorted(dispatchable)}
            _record_slots(record, step_plan, (), dropped_null, unwarmed)
            _refuse(record, f"the configuration is mirror-folded ({fold}); the warm "
                            "pass unfilled the fill sub-steps "
                            f"{', '.join(lost_fills)}, so dispatching the rest would "
                            "be a folded composition no gate ran")
            return None

    return _finish(record=record, step_plan=step_plan, dispatchable=dispatchable,
                   dropped_null=dropped_null, unwarmed=unwarmed, selected=selected,
                   table=table, residency=None, policy_licence=policy_licence,
                   fields=fields)


def _finish(*, record: Dict[str, Any], step_plan: Any, dispatchable: Set[str],
            dropped_null: Mapping[str, str], unwarmed: Mapping[str, str],
            selected: Mapping[str, str], table: str = "triton",
            residency: Any = None,
            policy_licence: Optional[Tuple[Any, str, int]] = None,
            fields: Any = None) -> Optional[FastPathPlan]:
    """The tail BOTH ladders emit through: order, split, certify, compose, publish.

    Factored out of :func:`_decide` when the second kernel table landed, and the
    factoring is the point rather than a tidy-up. Everything here is a statement
    about the DISPATCH SEAM and not about a composer — the driver's own slot order,
    the whole-or-nothing rule for a fused pair, which certification each arm rides
    on, what the artifact says and where it is written — so a second ladder that
    re-implemented it would be a second chance to get the seam's contract wrong, and
    the two would drift silently the first time either side moved. There is one
    implementation and the table it ran for is an argument to it.

    KEYWORD-ONLY, because the argument list is long enough that a positional call
    site would be unreadable and a re-ordering would be a silent defect. The Metal
    ladder calls it by keyword with the nine names below;
    :mod:`meep_gpu.metal_dispatch` names them in its own docstring as the join.

    ``fields`` IS THE IDENTITY GUARD'S OTHER HALF, not decoration.
    :meth:`FastPathPlan.dispatch` opens with ``if fields is not self.fields: return
    False`` — the guard that stops a plan built for one ``Fields`` from serving
    another after a mutation — so a ladder that does not pass it builds a plan whose
    every consult answers False. That is FAIL-CLOSED (the array path, which is
    always correct) and it is loud rather than silent: the dispatch counters stay at
    zero on slots the record calls dispatched, which is exactly the vacuity the
    gates' launch witnesses exist to catch.

    ``STEP_ORDER`` IS READ FROM ``triton_kernels.launch`` ON BOTH TABLES, and that
    is not a leak: ``metal_kernels.launch`` re-imports the same tuple, so there is
    one definition of the driver's slot order and both composers are ordered against
    it. ``triton_kernels.launch`` imports Triton lazily per builder and the module
    itself imports cleanly with no Triton installed, which is what makes this
    reachable from a NumPy host at all.
    """
    from .triton_kernels import launch as _launch  # noqa: PLC0415

    slots = tuple(name for name in _launch.STEP_ORDER if name in dispatchable)
    if not slots:
        _record_slots(record, step_plan, (), dropped_null, unwarmed)
        _refuse(record, "no slot is left carrying a kernel: every product was "
                        "refused, dropped as a no-op, or failed to compile")
        return None

    # A FUSED PAIR DISPATCHES WHOLE OR NOT AT ALL — see :func:`_split_pairs` for the
    # two ways half a pair goes wrong and why the answer is a refusal rather than a
    # per-slot fallback.
    split = _split_pairs(step_plan.plans, slots, _launch.STEP_ORDER)
    if split:
        _record_slots(record, step_plan, (), dropped_null, unwarmed)
        _refuse(record, "; ".join(split))
        return None

    arms = {slot: selected.get(slot, "unknown") for slot in slots}
    # WHICH TABLE EACH SLOT CAME FROM IS READ OFF THE PLAN, not passed in: only the
    # composer's own merge knows it, and a ninth keyword would be a second place for
    # the answer to be wrong. A single-table plan declares none, and every reader
    # falls back to ``table`` — which is what every gate, probe and test builds.
    backends = dict(getattr(step_plan, "backends", {}) or {})
    # THE DEVICE'S OWN ARCHITECTURE, so each family quotes the run that certified it
    # HERE rather than whichever run its ledger entry holds.
    _device_capability = ((record.get("environment") or {}).get("device")
                          or {}).get("compute_capability")
    _record_slots(record, step_plan, slots, dropped_null, unwarmed)
    record["decision"] = "dispatched"
    record["step_path"] = "fused"
    record["table"] = table
    record["families"] = {}
    for arm in sorted(set(arms.values())):
        dispatched = [slot for slot in slots if arms[slot] == arm]
        # THE ARM'S OWN TABLE, READ OFF THE MERGED PLAN, not the primary one. A bare
        # label carries no namespace, so under `MEEP_GPU_BACKEND_PREFERENCE=cuda` --
        # where the primary table is CUDA and the plan still holds Triton arms in the
        # slots the CUDA products did not take -- keying the lookup on the PRIMARY
        # table quotes the CUDA ledger for a Triton arm and returns family 'unmapped'
        # beside a gate 'unmapped', losing recorded_utc/host/records. That is the
        # record-forgery shape `_certification_for`'s miss branch exists to make
        # visible rather than to produce, and the datum that prevents it is already
        # here: the composer's merge wrote `backends[slot]` for every adopted slot,
        # including the primary's. A slot the merge did not name falls back to the
        # primary table, which is what every single-table plan is.
        owning = {backends.get(slot, table) for slot in dispatched} or {table}
        arm_table = _table_for_label(
            arm, sorted(owning)[0] if len(owning) == 1 else table)
        entry = _certification_for(
            arm, arm_table,
            capability=_normalized_capability(_device_capability)
            if _device_capability else None)
        entry["dispatched_slots"] = dispatched
        entry["table"] = arm_table
        record["families"][arm] = entry
    record["composition"] = _composition_block(record, slots, arms, table)
    record["composition"]["tables_dispatched"] = sorted(
        {backends.get(slot, table) for slot in slots})
    record["certified"] = _certified_verdict(
        record, record["composition"]["tables_dispatched"])
    plan = FastPathPlan(fields=fields, step_plan=step_plan, slots=slots, arms=arms,
                        dropped_null=dropped_null, unwarmed=unwarmed, record=record,
                        policy_licence=policy_licence, table=table,
                        residency=residency, backends=backends)
    _emit(record)
    return plan


def _record_slots(record: Dict[str, Any], step_plan: Any, slots: Tuple[str, ...],
                  dropped_null: Mapping[str, str], unwarmed: Mapping[str, str]) -> None:
    """Per-slot state, arm and refusal text — the whole of "where did each slot go".

    Reported over the composer's full ``STEP_ORDER``, which since the fill consults
    landed is also :data:`DRIVER_SLOTS` — the loop is kept keyed on ``STEP_ORDER``
    anyway, because the invariant it protects is "every slot the COMPOSER can fill
    is reported", and a composer slot the driver cannot run is exactly the case rung
    6 exists for. An array slot ALWAYS carries a named reason: the composer's
    contract is that an unselected slot never gets a falsy reason tuple, and a
    reader testing "is there anything to report" must never read "nothing covers
    this" as "nothing to say".
    """
    from .triton_kernels import launch as _launch  # noqa: PLC0415

    selected = dict(getattr(step_plan, "selected", {}) or {})
    reasons = dict(getattr(step_plan, "reasons", {}) or {})
    backends = dict(getattr(step_plan, "backends", {}) or {})
    entries: Dict[str, Any] = {}
    for slot in _launch.STEP_ORDER:
        entry: Dict[str, Any] = {
            "state": "dispatched" if slot in slots else "array",
            "arm": selected.get(slot),
        }
        # WHICH TABLE FILLED IT, on a merged step only. Absent on a single-table
        # plan rather than repeated per slot: a key that says the same thing seven
        # times is one a reader stops reading, and the record already names the
        # table once at the top.
        if slot in backends:
            entry["backend"] = backends[slot]
        if slot in dropped_null:
            entry["dropped_as_null"] = dropped_null[slot]
            entry["reason"] = ("the selected arm launches nothing; the array call "
                               "returns before its first statement")
        elif slot in unwarmed:
            entry["unwarmed"] = unwarmed[slot]
        if entry["state"] == "array" and "reason" not in entry:
            refusals: List[str] = list(reasons.get(slot, ()))
            if not refusals and slot not in selected:
                refusals = ["no consulted product admitted this slot"]
            elif not refusals:
                refusals = [f"the {selected.get(slot)!r} product was built but the "
                            "driver seam has no consult for this slot"]
            entry["reason"] = "; ".join(refusals)
        entries[slot] = entry
    record["slots"] = entries
    # WHY EACH SEAM DID NOT FUSE, which no slot entry can carry. The composer keys
    # its fusion refusals ``fused_pair_<product>`` — not a ``STEP_ORDER`` slot — so
    # until 2026-09-02 they were composed, reported to nobody, and dropped. That is
    # the one question a reader of a dispatching-but-unfused run has, and it is the
    # only place the label OFFER's refusal is visible at all: a product the run
    # cannot admit leaves its seam on the separate arms, and without this the
    # artifact shows a fused-free step with no statement about why.
    fusion_refusals = {name[len("fused_pair_"):]: "; ".join(value)
                       for name, value in sorted(reasons.items())
                       if name.startswith("fused_pair_") and value}
    if fusion_refusals:
        record.setdefault("fusion", {})["not_installed"] = fusion_refusals
    record["dropped_null"] = dict(dropped_null)

"""A Metal gate artifact's recorded sources must still BE the tree's sources.

WHY THIS EXISTS. 29 of 30 ``gate_metal_*.py`` already record the digests of the
sources they ran — that half is done and done well. What was missing is anything
that READS them: ``test_the_checked_in_fingerprints_match_the_tree`` compares
``fingerprints.json`` against the tree, not an ARTIFACT against the tree, so a
gate could release, its source could change afterwards, and the artifact would go
on asserting a result about bytes that no longer ship.

That is not hypothetical. Measured 2026-08-19 by hand, not by any test: the
complex-PML dispersive whole-step artifact (58 cases, 43,865,856 uint32
comparisons, zero divergences) records 73 sources, and
``complex_dispersive_update_e.py`` — the E/ADE leg of the very composition being
certified — was edited two minutes AFTER the run. 72/73 matched; the one that did
not was the one that mattered. A self-recording artifact nothing reads is a log,
not a guarantee.

DECLARED DRIFT, NOT AN EXEMPTION. Known-stale entries are named in
:data:`KNOWN_DRIFT` with the reason. Anything NOT named there fails. Clearing an
entry means re-running that gate, never editing this list to match the tree.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu" / "results"

#: Artifacts under verification. Adding one here is the deliberate act that puts
#: it under this test; an artifact nobody lists is evidence nobody checks.
CURRENT_EVIDENCE = (
    # RE-CUT 2026-08-29 (repair), fresh directories, same rule. CAUSE: a kernel panic
    # killed the dispatch round's own re-gate campaign after 5 of its 44 gates, leaving
    # every Metal weld pinning a meep_gpu/fastpath.py the dispatch edits had already
    # moved. Nothing was half-written -- the gate that died left a zero-byte run.log and
    # no artifact -- so the repair is simply the campaign finished: 44 of 44 released,
    # all 44 welds rebound, and both artifacts below carry ZERO drift (43 and 91
    # recorded digests respectively).

    # RE-CUT 2026-08-19T21:45 into a FRESH directory. The previous re-cut went
    # stale the moment gate_provenance.py gained three verdict spellings: that file
    # is imported by every gate, so editing the RECORDER invalidates every artifact
    # that recorded it. Re-running is the fix; a KNOWN_DRIFT entry would only hide
    # that the evidence no longer describes the tree.
    #
    # A fresh directory is required rather than a convenience: each artifact
    # directory carries its own source_sha256.txt manifest, and the gate REFUSES to
    # release against a manifest it disagrees with. Re-running in place would have
    # to overwrite that manifest, which is the check the manifest exists to make.
    # RE-CUT 2026-08-23, fresh directory, same rule. The in-seam deposit round edited
    # driver.py, fastpath.py, triton_kernels/coverage.py and triton_kernels/launch.py --
    # the source-injection wiring -- and delegated the source-presence clause in eleven
    # metal_kernels fused-pair modules, so both artifacts below stopped describing the
    # tree. The `direct` artifact is no longer exempt: it does not import registry.py,
    # but it does import the driver and the composer. Re-run rather than declared;
    # gate_metal_dispersive_update_e reported PASS over 43 digests with zero
    # disagreements, and now carries FOUR product rows where the T2145 artifact carried
    # three -- the family gained a row in the interim, which is coverage the old
    # evidence did not have.
    # RE-CUT 2026-08-26, fresh directory, same rule and this round's own cause: every
    # fused-pair module gained a ``replaces=REPLACES`` argument to its ``arms.register``
    # call, and ``arms.py`` gained the ``replaces`` slot and the ``is_weld`` property
    # that argument feeds. Both are reached through registry.py, so the 08-23 artifacts
    # stopped describing the tree the moment those bytes moved. THE RE-RUN CARRIES ZERO
    # DRIFT over 43 recorded digests and released=True. What says the edit moved no
    # byte is not the re-run alone but two independent facts beside it: the arm table is
    # UNCHANGED at 98 registered and 82 wired -- a declaration on an UNWIRED arm changes
    # no dispatch -- and ``metal_kernels/fingerprints.json``'s ``kernel_source_sha256``
    # is byte-identical across the edit, so not one character of what the MPS compiler
    # received changed. The three sub-step gates whose disjointness sweeps this round
    # re-partitioned (bfast, complex, special_kz) all report released=True.
    # RE-CUT 2026-08-27, fresh directory, same rule and this round's own cause: the
    # cross-sub-step ROUTING wired the in-seam deposit repair through
    # metal_kernels/launch.py (_install_fused_pair, mirroring Triton's), which every
    # Metal gate imports, so the 08-26 artifact stopped describing the tree the moment
    # that landed. Re-run rather than declared: the fresh artifact carries ZERO drift
    # over its 43 recorded digests with released=True.
    # RE-CUT 2026-08-28 (deposit), fresh directory, same rule and this round's own
    # cause: two Metal fused families were given an absorb row in
    # metal_kernels/launch.py's FUSED_PAIR_ARMS and flipped
    # CARRIES_DEPOSIT_REPAIR to True in the same edit
    # (complex_fused_magnetic_pair, fused_dispersive_pair). launch.py is reached
    # through registry.py, which this gate imports, so the 08-27 artifact stopped
    # describing the tree the moment the table moved. Re-run rather than declared:
    # the fresh artifact carries ZERO drift over its 43 recorded digests with
    # released=True, and the round re-gated 43 of 43 Metal families green.
    # RE-CUT 2026-08-28 (folded carry), fresh directory, same rule and this round's
    # own cause: the two FOLDED Metal fused families flipped CARRIES_DEPOSIT_REPAIR
    # once `deposit_repair.repair_cells` extended the repair's saved and restored set
    # to the cells the post-injection fills image a deposit into, and
    # metal_kernels/launch.py's absorb-table comment moved with them. All three files
    # are reached through registry.py, which this gate imports, so the earlier 08-28
    # artifact stopped describing the tree the moment they moved.
    # WHAT SAYS THE EDIT MOVED NO DISPATCHED BYTE, beside the re-run: `fingerprints.json`'s
    # `kernel_source_sha256` is byte-identical across the whole round (only host-file
    # digests moved), so not one character of what the MPS compiler received changed,
    # and both folded families still register wired=False. Re-run rather than declared:
    # the fresh artifact carries ZERO drift over its 43 recorded digests, verdict PASS
    # with released=True. The `_carry2` stamp is the SECOND full re-cut of the round --
    # the first (`_carry`) certified bytes that a later comment edit moved, and a gate
    # certifies the bytes it ran against and not the ones that follow it.
    # RE-CUT 2026-08-28 (guard), fresh directory, same rule and this round's own
    # cause: `deposit_repair.repairable` used to FAIL OPEN on an inactive absorber --
    # it took no PML layer at all, so it answered (True, ()) on a grid where
    # `stepping._pml_is_active` sends both constitutive halves down the plain path and
    # the repair's split-field arithmetic inverts a recurrence that never ran (measured:
    # 23 words of Ex/Ey/Ez and f_w_E* differing from the driver's own order on a 14x14
    # grid at thickness 0). The guard now takes the layer and refuses BY NAME an absent
    # one, an inactive one, and any coefficient whose layout `apply` cannot broadcast the
    # way the sub-step read it. Every Metal gate imports the package, so this artifact
    # stopped describing the tree the moment those bytes moved. WHAT SAYS THE EDIT MOVED
    # NO DISPATCHED BYTE, beside the re-run: no shipped predicate hands the guard a
    # layer, so the added clauses are unreachable from coverage -- measured, not argued,
    # by cutting the fusion matrix twice with the fix live and with the pre-fix guard
    # restored in-process and finding the ledger and census BYTE-IDENTICAL, served 242
    # both ways. The fresh artifact carries ZERO drift over its 43 recorded digests,
    # verdict PASS with released=True. TWO SUPERSEDED ARTIFACTS SIT BESIDE IT and are
    # named rather than tidied away: `..._failclosed` is a FILE rather than a directory,
    # a mis-invocation of `--out` whose manifest is malformed and which records
    # released=False; `..._failclosed2` is a correct PASS that certified bytes a later
    # edit moved (the guard gained its rank clause afterwards), and a gate certifies the
    # bytes it ran against and not the ones that follow it. Neither is bound by anything.
    # RE-CUT 2026-08-28 (arity), fresh directory, same rule and this round's own
    # cause -- and unlike the six re-cuts above it, the cause is a NEW PRODUCT rather
    # than a shared file drifting. The D->E (PML, ordinary) cell -- 26 of 26 rows, the
    # largest unserved cell on the Metal board -- was built as
    # `metal_kernels/fused_electric_pair.py`, which added its import line and list
    # entry to `registry.py` and an absorb row to `metal_kernels/launch.py`'s
    # FUSED_PAIR_ARMS. This gate imports both through `registry.py`, so the `_guard`
    # artifact stopped describing the tree the moment they landed.
    # WHAT SAYS THE EDIT MOVED NO DISPATCHED BYTE, beside the re-run:
    # `fingerprints.json`'s `kernel_source_sha256` is byte-identical across the whole
    # round -- only three host digests moved (`launch.py`, `registry.py`, and the new
    # module itself) -- so not one character of what the MPS compiler received changed
    # for any family that already shipped; and the new family registers wired=False,
    # so `plan_step` cannot select it and `fuse` is still False at the one shipped
    # call site. Re-run rather than declared, over the whole round: 44 of 44 gates
    # exit 0 with released=True.
    # RE-CUT 2026-08-29 (fused route), fresh directory, same rule and this round's
    # own cause: the fusion opt-in (`fastpath.FUSE_ARMS_SWITCH`) and the fused arms'
    # `ARM_CERTIFICATION` rows landed in `meep_gpu/fastpath.py`, along with the
    # source-list argument `driver.py` now hands `plan_fast_path`. Every Metal gate
    # imports the package, so the `_arity` artifact stopped describing the tree the
    # moment those bytes moved -- and it said so by name: `fastpath.py` was the one
    # undeclared digest of the 43.
    # WHAT SAYS THE EDIT MOVED NO DISPATCHED BYTE, beside the re-run: the opt-in's
    # unset default asks the composer for exactly the composition the old literals
    # asked for (`fuse=False, fuse_ade=False`), and `ARM_CERTIFICATION` is a lookup
    # table read only when an artifact is written. The re-run carries ZERO drift over
    # its 43 recorded digests with released=True.
    # RE-CUT AGAIN 2026-08-29 (release), fresh directory, same rule and the same
    # cause one round on: `fastpath.py` moved again when the fused route was
    # RELEASED -- RELEASED_FUSED_ARMS, FUSED_RELEASE_ENVELOPE and the clause (8)
    # rewrite -- and every Metal gate imports the package, so the `_fusedroute`
    # artifact stopped describing the tree the moment those bytes moved. It said
    # so by name: `fastpath.py` was again the one undeclared digest of the 43.
    # WHAT SAYS THE EDIT MOVED NO METAL BYTE, beside the re-run: the release
    # reaches only `fastpath.plan_fast_path`, which imports `triton_kernels` and
    # nothing else, so no Metal product is reachable from it at all -- measured,
    # not asserted, by `parity/meep_gpu/dispatch_reachability.py`.
    # RE-CUT AGAIN 2026-08-29 (fusion veto, then the gate name), fresh directory,
    # same rule and the same cause a third and fourth time: `fastpath.py` moved when the release was given its
    # OPT-OUT -- `FUSE_ARMS_VETO`, `fused_arms_vetoed()`, and the one line in
    # `_decide` that empties `admitted` when it is set. The veto had to exist
    # because the release removed a reachable configuration: "dispatch, do not
    # fuse" used to be what an unset switch meant, and the driver-route gate's
    # substitution baseline is exactly that run. Measured on the GPU host GPU 6 against
    # the released tree, that baseline fused too -- pml_2d at 2.0 launches/step on
    # both legs, `launch_drop_per_step 0.0` -- and the gate refused to release.
    # WHAT SAYS THE EDIT MOVED NO METAL BYTE, beside the re-run: the veto is read
    # only inside `fastpath.plan_fast_path`, which imports `triton_kernels` and no
    # other kernel package, so no Metal product is reachable from it at all --
    # measured off the parse tree by `parity/meep_gpu/dispatch_reachability.py`,
    # not asserted. The re-run carries ZERO drift over its 43 recorded digests
    # with released=True.
    # THE FOURTH ROUND (`_gatename`) is a ONE-STRING edit: the veto's gate ran and
    # released, and `DRIVER_ROUTE_FUSED_GATE` was repointed from the run that
    # reached the arms through the opt-in to the one that reached them through the
    # release. Its whole executable content is that constant -- proved by reverting
    # it and reproducing the as-run `code_digest` exactly -- and it still owes a
    # full round, because a raw sha256 pin does not care what the bytes mean.
    # RE-CUT 2026-08-30 (bind), fresh directory, same rule and the same cause a
    # fifth time -- but this round the drift was not a new feature, it was the
    # RECORD being wrong about which bytes exist. `fastpath.py` shipped at
    # `87f740ee`, which is neither HEAD nor anything on disk, and these three
    # welds plus this artifact all pinned it; the driver-route gate the release
    # cites had run a fourth digest again (`29ac1216`). The `_repair` artifact said
    # so by name: `meep_gpu/fastpath.py` was its one drifted digest of 43.
    # WHAT SAYS THE EDIT MOVED NO METAL BYTE, beside the re-run: this round's
    # `fastpath.py` edits are the gate-name constant, the `RELEASED_FUSED_ARMS`
    # case lists and their prose, all read only inside `fastpath.plan_fast_path`,
    # which imports `triton_kernels` and no other kernel package -- so no Metal
    # product is reachable from them at all, measured off the parse tree by
    # `parity/meep_gpu/dispatch_reachability.py`. The re-run carries ZERO drift
    # over its 43 recorded digests with released=True.
    # RE-CUT 2026-08-30 (deposit carry), fresh directory, same rule and this round's
    # own cause: the THREE TRITON fused families flipped CARRIES_DEPOSIT_REPAIR once
    # their device gates were re-run against the flipped bytes, and
    # `triton_kernels/launch.py`'s absorb-table prose moved with them -- the two
    # comment blocks that stated the fold boundary rested on the flag being False.
    # This gate pins `meep_gpu/triton_kernels/launch.py`, so the `_bind` artifact
    # stopped describing the tree the moment that prose moved.
    #
    # WHAT SAYS THE EDIT MOVED NO METAL BYTE, beside the re-run: the whole change on
    # the Metal side of this artifact is a COMMENT in a file no Metal product
    # executes -- `metal_kernels` imports nothing from `triton_kernels.launch` -- and
    # `metal_kernels/fingerprints.json`'s `kernel_source_sha256` is byte-identical
    # across the round, so not one character of what the MPS compiler received
    # changed. Re-run rather than declared, because a comment-only edit is still an
    # edit and this file's KNOWN_DRIFT is empty on purpose.
    # ZERO drift over its 43 recorded digests, VERDICT PASS, released=True.
    #
    # RE-CUT 2026-08-30 (arms rows), fresh directory, same rule and this round's own
    # cause -- and this time it is a Metal byte, not a comment in another track. The
    # round landed nine new fused products, and every one of them needs a row in
    # `metal_kernels/launch.py`'s FUSED_PAIR_ARMS before the seam loop will even ask
    # its predicate. That table went 6 rows -> 15. `launch.py` is reached through
    # `registry.py`, which this gate imports, so the `_carry` artifact stopped
    # describing the tree the moment those rows landed. Two family modules moved with
    # it -- `folded_complex_fused_magnetic_pair.py` and
    # `folded_beta_complex_fused_magnetic_pair.py` -- whose CARRIES_DEPOSIT_REPAIR
    # flipped False -> True in the same edit, for the reason the flag exists: until
    # the row existed nothing could bracket their launch, so False was the only true
    # value. Three paths, 39 of 44 welds, one cause.
    # Re-run rather than declared, and KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-08-31 (the coefficient-pack round), fresh directory, same rule and
    # the same shape of cause one more time. The round landed the two Dcyl D->E fused
    # pairs -- the first products on this board recovered from the BINDING CEILING
    # rather than from a clause -- and each needs a row in `metal_kernels/launch.py`'s
    # FUSED_PAIR_ARMS before the seam loop will ask its predicate, so that table went
    # 15 rows -> 17. `registry.py` gained the two imports. This gate reaches both
    # through its own import set, so the `_armsrows` artifact stopped describing the
    # tree the moment those rows landed. Re-run rather than declared.
    #
    # RE-CUT 2026-08-31 AGAIN (the plain-repair round), fresh directory, same rule and
    # the fourth instance of the same shape of cause. This round moved
    # `meep_gpu/deposit_repair.py` -- it gained `PLAIN_PATH`, the second repair, which
    # inverts `update_E`'s plain overwrite -- and `meep_gpu/triton_kernels/launch.py`,
    # whose `_install_fused_pair` gained the `repair_paths` argument that lets a
    # product declare WHICH repair its two slots install. This gate reaches both
    # through its own import set, so the `_pack` artifact stopped describing the tree
    # the moment those two files moved. Re-run rather than declared, and KNOWN_DRIFT
    # stays empty.
    #
    # RE-CUT 2026-09-01 (the last-cells round), fresh directory, same rule and the
    # fifth instance of the same shape of cause. The round landed the two
    # no-absorber stored-E D->E pairs -- the first METAL products on
    # `deposit_repair.PLAIN_PATH` -- so `metal_kernels/launch.py`'s FUSED_PAIR_ARMS
    # went 17 rows -> 19, its `_install_fused_pair` gained the `repair_paths`
    # threading the Triton installer already had, and `registry.py` gained the two
    # imports. This gate reaches all three through its own import set, so the
    # `_plainrepair` artifact stopped describing the tree the moment those rows
    # landed. Re-run rather than declared (the whole-board re-cut at
    # `metal_regate_2026-09-01_lastcells`, every gate green).
    #
    # RE-CUT 2026-09-01 AGAIN (the residue round), fresh directory, same rule and
    # the sixth instance of the same shape of cause. The round landed the four
    # residue welds -- the two coefficient_pack-recovered CANNOT-BIND cells
    # (`bfast_fused_electric_pair`, `conductive_fused_electric_pair`) and the two
    # special_kz complex-beta pairs -- so `metal_kernels/launch.py`'s
    # FUSED_PAIR_ARMS went 19 rows -> 23, `registry.py` gained the four imports,
    # and `no_pml_conductive_fused_electric_pair.py`'s conductive-injection clause
    # was lifted onto the driver's sparse rescale. This gate reaches all of them
    # through its own import set, so the `_lastcells` artifact stopped describing
    # the tree the moment those rows landed. Re-run rather than declared (the
    # whole-board re-cut at stamp `2026-09-01_residue`, every gate green).
    # RE-CUT 2026-09-01 (stencil), fresh directories, same rule. CAUSE: the four
    # off-diagonal SCRATCH-OUTPUT stencil welds landed, and with them four new rows
    # in metal_kernels/launch.py's FUSED_PAIR_ARMS and four new imports and
    # FAMILY_MODULES entries in metal_kernels/registry.py. Both files are reached by
    # every Metal gate, so the 09-01 residue artifacts stopped describing the tree
    # the moment those bytes moved. RE-RUN RATHER THAN DECLARED: the whole
    # `recut_metal_gates.sh` campaign was re-cut on this Mac (53 processes: three
    # expansion probes and 50 gates), 52 returned green on the first pass and the
    # 53rd — gate_metal_complex — was refused by the campaign source manifest
    # because a source file was edited WHILE the campaign was running. That refusal
    # is the manifest working: the edit was a comment, it was reverted to the byte,
    # and the re-run released. All 50 welds were then rebound to the fresh round.
    # RE-CUT 2026-09-02 (the fastpath repoint), fresh directory, same rule and the
    # seventh instance of the same shape of cause. `driver.py`'s sparse conductive
    # rescale meant the four-leg driver-route campaign had to run again, and the
    # record the release cites is found BY NAME through
    # `fastpath.DRIVER_ROUTE_FUSED_GATE` -- so closing that drift required moving
    # that constant off `dispatch_fused_route_2026-08-30_bind`, which edited
    # `meep_gpu/fastpath.py`. This gate reaches fastpath.py through its own import
    # set, so the `_stencil` artifact stopped describing the tree the moment the
    # constant moved. Re-run rather than declared: the fresh artifact carries 43
    # recorded digests and ZERO drift.
    #
    # RE-CUT 2026-09-02 AGAIN (the folded-release round), and the cause is the
    # widest yet: `fastpath.py` gained the two-half release
    # (FUSED_RELEASE_ARM_AXES beside FUSED_RELEASE_ENVELOPE) and the label OFFER it
    # hands the composer, `triton_kernels/launch.py` and its package `__init__`
    # gained the `fuse_labels` argument that carries it, and this package's own
    # `launch.py`/`__init__.py` docstrings were corrected where they still said
    # dispatch returns None on every branch. This gate imports all of them. Re-run
    # rather than declared — the whole-board re-cut at stamp
    # `2026-09-02_foldedrelease`, 54 of 54 green, VERDICT: ALL GREEN — and
    # KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-09-04 (the cylindrical m = 0 complex round), same rule. The
    # cylindrical complex family gained its `M_ZERO` arm (`cylindrical_complex.py`,
    # both complex cylindrical fused pairs, the real family's storage refusal
    # wording, one `launch.py` docstring sentence), and every Metal gate reaches
    # those modules through `registry.py`. Re-run rather than declared — the
    # whole-board re-cut at stamp `2026-09-04_m0complex` — and KNOWN_DRIFT stays
    # empty.
    #
    # RE-CUT 2026-09-04 AGAIN (the H_to_D Phase 0 round), same rule and four
    # subjects at once: `meep_gpu/driver.py` and `meep_gpu/fastpath.py` gained the
    # by-name magnetic-half-step consult channel, and BOTH composers'
    # `launch.py` gained the H_to_D seam row and the neighbouring-seam
    # arbitration rule. This gate imports all four. Re-run rather than declared —
    # the whole-board re-cut at stamp `2026-09-04_p0regate`, 54 of 54 green,
    # VERDICT: ALL GREEN — and KNOWN_DRIFT stays empty.
    #
    # RE-CUT ONCE MORE the same day, at `2026-09-04_syncchannel`, because ONE
    # constant moved after the `_p0regate` fleet had run:
    # `fastpath.DRIVER_ROUTE_FUSED_GATE` was repointed at the driver-route
    # campaign cut from the Phase 0 bytes, and every Metal gate imports
    # `fastpath.py`. That is the whole difference between the two fleets, and
    # `_p0regate` is kept rather than deleted: it is the record of the
    # pre-repoint bytes. 54 of 54 green, VERDICT: ALL GREEN, KNOWN_DRIFT empty.
    # RE-CUT ONCE MORE 2026-09-04, at `2026-09-04_arbfix`, same rule again: the
    # composer's earlier-neighbour arbitration half was REMOVED from
    # `metal_kernels/launch.py` after the `_syncchannel` fleet had run (it reversed
    # the released E->P trade ruling by refusing the polarization pair), and this
    # gate imports that module. Re-run rather than declared — the whole fleet at
    # `2026-09-04_arbfix`, 54 of 54 green — and KNOWN_DRIFT stays empty.
    # RE-CUT ONCE MORE 2026-09-05, at `2026-09-05_hdweld`, same rule a third time:
    # `metal_kernels/registry.py` gained the import and the FAMILY_MODULES row for
    # `fused_hd_pair` — the H->D weld — after `_arbfix` had run, and every Metal gate
    # reaches that module through `arms.ensure_registered()`. The product it registers
    # is UNWIRED and declares INSTALLABLE = False, so no gate's composition changed;
    # what changed is the bytes, and the bytes are what an artifact binds. Re-run
    # rather than declared — the whole fleet at `2026-09-05_hdweld`, 54 of 54 green,
    # VERDICT: ALL GREEN — and KNOWN_DRIFT stays empty.
    # RE-CUT A FIFTH TIME 2026-09-05, at `2026-09-05_hdwithdraw`, and this one is a
    # PROSE change with the same price. `metal_kernels/fused_hd_pair.py` gained
    # `WELD_OWED` — the declaration that its gate has run and refused, which is what
    # lets `test_metal_weld_contract::test_every_metal_family_is_welded` be green on
    # honest grounds instead of red — and corrected the tail of `INSTALLABLE_REASON`,
    # which used to say the family's own gate certifies its arithmetic. Every Metal
    # gate reaches that module through `arms.ensure_registered()`, so every artifact
    # that recorded it stopped describing the tree the moment those constants moved:
    # the drift is the price of touching a file the fleet imports, and it is paid by
    # re-running, never by declaring. The whole fleet at `2026-09-05_hdwithdraw`,
    # 54 of 55 exits zero — the one non-zero is `gate_metal_fused_hd_pair` itself,
    # which is the withdrawal's own subject and must not release. 50 welds rebound at
    # zero drift; the board re-cut on the same census with ZERO instances moved
    # (358 / 597 served by predicate, unchanged). KNOWN_DRIFT stays empty.
    # RE-CUT A SIXTH TIME 2026-09-06, at `2026-09-06_hdland`, and this one is the
    # release the fifth withdrew. `metal_kernels/fused_hd_pair.py` emptied `WELD_OWED`
    # and rewrote the tail of `INSTALLABLE_REASON`, and every Metal gate reaches that
    # module through `arms.ensure_registered()`, so the whole fleet was owed a re-run
    # by the same rule the fifth paid: the drift is the price of touching a file the
    # fleet imports. 54 of 54 green, VERDICT: ALL GREEN, `gate_metal_fused_hd_pair`
    # included this time — VERDICT PASS, released, 52 legs, 115 recorded sources. 50
    # welds rebound at zero drift and `metal_fused_hd_pair_device_gate` minted as the
    # fleet's 51st. KNOWN_DRIFT stays empty.
    # RE-CUT A SEVENTH TIME 2026-09-06, at `2026-09-06_hdfinal`. CAUSE: after the sixth
    # cut released, the docstring and the tail of `INSTALLABLE_REASON` in
    # `metal_kernels/fused_hd_pair.py` were rewritten to state the four measured facts
    # behind "executes nowhere today", and every Metal gate imports that module through
    # `arms.ensure_registered()`, so the fleet was owed a re-run by the rule the fifth
    # and sixth paid: prose in a module the fleet imports is still bytes the fleet
    # pins. 55 of 55 green, VERDICT: ALL GREEN, `gate_metal_fused_hd_pair` VERDICT PASS
    # and released; 51 welds rebound at zero drift, 0 skipped; fingerprints.json
    # regenerated last with all 52 top-level keys intact. The module's own text still
    # names the sixth cut as what released it -- that stays true and is not edited,
    # because editing it would owe an eighth. KNOWN_DRIFT stays empty.
    # RE-CUT A NINTH TIME 2026-09-07, at `2026-09-07_wire`. CAUSE: the cylindrical
    # H->D wiring round routed the two CuPy-side products through their composers,
    # which edited `meep_gpu/fastpath.py` and `meep_gpu/triton_kernels/launch.py`.
    # Neither is a Metal file, and that is the point: a Metal gate PROCESS imports
    # both through the driver and the arm registry, so its artifact records their
    # bytes, and three Metal welds plus both artifacts below stopped describing the
    # tree the moment those two files moved. Measured before the re-run: 12 stale
    # digests of 717 checked across 3 entries, every one naming one of those two
    # files. Re-running is the fix; a KNOWN_DRIFT entry would only hide it.
    # RE-CUT A TENTH TIME 2026-09-07, at `2026-09-07_final`. CAUSE: `meep_gpu/fastpath.py`
    # moved once more, for a reason worth stating because it is a rule rather than an
    # accident. `fastpath.DRIVER_ROUTE_FUSED_GATE` is a string constant naming the
    # driver-route campaign the dispatch record is cut from, and it lives IN the file that
    # campaign pins, so repointing it to this round's run necessarily moved the file
    # (b74a4c1ac90a -> 67b754de3a96). Three Metal welds and both artifacts below pin it:
    # 6 stale digests of 717 checked across 3 entries, measured before this run. Neither
    # file is a Metal file; a Metal gate PROCESS imports both. The file is final at
    # 67b754de3a96 and every source edit of the round landed BEFORE this campaign started,
    # which is the ordering the route gate's own note prescribes and the one an earlier
    # attempt in this round got wrong.
    # RE-CUT AN ELEVENTH TIME 2026-09-07, at `2026-09-07_wired`. CAUSE: two H_to_D
    # products were REGISTERED -- `metal_kernels/folded_fused_hd_pair.py` (75 board
    # instances) and `metal_kernels/complex_fused_hd_pair.py` (17). Registering a family
    # is not a local edit: it lands an import and a list entry in `registry.py`, an
    # absorb row in `metal_kernels/launch.py`, a name in `metal_kernels/__init__.py` and
    # an emptied `WELD_OWED` in each product module -- and every Metal gate PROCESS
    # imports the first three, so both artifacts below stopped describing the tree the
    # moment the wiring landed. Measured before this run: 6 stale digests of 161 checked
    # across the 2 entries here, naming `registry.py`, `launch.py`, `__init__.py` and
    # `fused_hd_pair.py`, and 56 of 399 pinned weld paths drifted beside them. Re-running
    # is the fix; a KNOWN_DRIFT entry would only hide it. Every source edit of the round
    # landed BEFORE the campaign started -- including the two edits the campaign itself
    # forced, the complex gate's own pre-wire pin (it asserted the family was NOT
    # registered, which the wiring inverts) and the folded gate's absorb-row note.
    # 60 of 60 legs green.
    # RE-CUT 2026-09-08, at `2026-09-08_regate`, and this round's cause is the H->D
    # wiring merge: it moved `meep_gpu/fastpath.py`, `metal_kernels/launch.py` and
    # `triton_kernels/launch.py`, all three of which this gate's process imports, so
    # the `_wired` artifact stopped describing the tree. Measured on it before the
    # re-run: 3 stale digests of 44. The fresh artifact records the same 44 and
    # carries ZERO drift, `release.released` true. Re-run rather than declared --
    # KNOWN_DRIFT stays empty.
    # RE-CUT 2026-09-11, at `2026-09-11_dispatch`, and this round's cause is the
    # dispatch batch: it moved `meep_gpu/fastpath.py`, `metal_kernels/{arms,launch,
    # subnormal}.py`, `subnormal_policy.py` and `parity/meep_gpu/metal_gate_runner.py`,
    # all of which this gate's process imports, so the `_close` artifact stopped
    # describing the tree. Measured on it before the re-run: 6 stale digests. Re-run
    # rather than declared -- the whole Metal fleet was re-gated at this stamp and
    # KNOWN_DRIFT stays empty.
    # RE-CUT ONCE MORE 2026-09-13, at `2026-09-13_batch`, same cause as the entry
    # above: the Triton dispatch release (Phase B) moved `meep_gpu/fastpath.py` --
    # fifteen new arms, their `ARM_CERTIFICATION` rows and the enlarged released
    # set -- and this gate's process imports `fastpath.py`, so the `_dispatch`
    # artifact stopped describing the tree (its recorded `fastpath.py` digest was
    # the one that drifted). No Metal byte moved: the release reaches only the
    # Triton route. The whole Metal fleet was re-gated at this batch stamp; the
    # fresh artifact records 44 digests and carries ZERO drift, released=True.
    # Re-run rather than declared, and KNOWN_DRIFT stays empty.
    # RE-CUT AGAIN 2026-09-21, at `2026-09-20_restrict`: the component-restricted
    # deposit repair moved this artifact's import closure and the fleet re-gated it.
    # RE-CUT 2026-09-19, at `2026-09-19_witness`. CAUSE: the `_batch` artifact had
    # drifted on FOUR recorded digests of 44, dated by diffing the recorded source
    # tables of every twin cut since rather than assumed. `meep_gpu/fastpath.py` moved
    # in every round from `2026-09-14_target` through `2026-09-17_allpaths`;
    # `meep_gpu/triton_kernels/launch.py` moved at `_14_target` and again at
    # `_17_allpaths`; `meep_gpu/metal_kernels/launch.py` moved only at `_17_allpaths`;
    # and `meep_gpu/deposit_repair.py` moved on 2026-09-19 (mtime 01:20 local), after
    # `_17_allpaths` had run. Every round from `_14_target` on re-cut the fleet and
    # repointed the Metal board's RELEASE_BINDING in
    # `parity/meep_gpu/build_fusion_matrix.py`, but none repointed this tuple, so the
    # `_batch` pin has been stale since `_14_target` moved `fastpath.py`. The
    # `_17_allpaths` twin is NOT pinned here: measured against the tree it still drifts
    # on `deposit_repair.py`, one digest of 44. The witness fleet
    # (`metal_campaign_2026-09-19_witness/fleet/recut_summary.txt`, 66 of 66 processes
    # exit 0, VERDICT: ALL GREEN) re-cut this gate: VERDICT PASS, released=True, 44
    # recorded digests and ZERO drift. `metal_kernels/fingerprints.json:5940` already
    # reads its verdict from this same artifact (`verdict_read_from`), so the pin now
    # names the twin the ledger binds. Re-run rather than declared, and KNOWN_DRIFT
    # stays empty.
    #
    # EDITING THIS FILE REFUSES NOTHING, recorded so the next reader need not
    # rediscover it: two records carry this file's own digest --
    # `h_to_d_seam_2026-09-04/subject_digests_tests.json:64` (a2aaf794...) and
    # `h_to_d_withdraw_order_2026-09-04/subject_digests.json:64` (a284bbc5...) -- and
    # both were already historical before this edit (the file was 723034e4...).
    # `parity/meep_gpu/h_to_d_seam.py:226` reads only their `manifest_sha256`, never a
    # per-file digest, so neither can refuse.
    # RE-CUT AGAIN 2026-09-24 at `_2026-09-24_round3`: the night round's held-step fixes
    # (conditional drain, Metal door kernels, the device-side deposit subtraction in
    # `device.py`/`host_writes.py`/`sources.py`) moved this gate's import closure; the
    # round-3 fleet re-ran it (ALL GREEN), released=True with ZERO of its 90 recorded
    # digests disagreeing with the tree, checked before this line moved.
    # RE-CUT AGAIN 2026-09-25 at `_2026-09-25_night`: the citation re-point and the batched wall
    # clears moved this gate's import closure; the `_2026-09-25_night` fleet re-ran it (ALL GREEN),
    # released=True with ZERO recorded digests disagreeing with the tree, checked before
    # this line moved.
    # RE-CUT AGAIN 2026-10-06 at `_2026-10-06_092_g13s`. CAUSE: 891243a (the supported,
    # uncertified NVIDIA default) and 4c19311 (the NVIDIA route stamps) moved
    # `meep_gpu/fastpath.py` (4c04d6d8 -> 4506b154 -> 9f4abde4) after the `_2026-10-02_arch3`
    # artifact pinned here had run, and after its `_2026-10-04_g13s` twin, which the ledger
    # reads this gate's verdict from; each recorded 45 digests and drifted on that one. This
    # gate's ledger weld does not pin `fastpath.py`, so the rounds that scoped their re-gates
    # from the ledger did not re-run it. The re-run on 06435a9 released (verdict PASS, 4
    # product rows and 3 mutation rows) with ZERO of its 45 recorded digests disagreeing with
    # the tree. Re-run rather than declared, and KNOWN_DRIFT stays empty.
    "metal_dispersive_update_e_2026-10-06_092_g13s/gate.json",
    # RE-CUT AGAIN 2026-08-20T00:15, fresh directory, for the same reason and by
    # the same rule: the fused dispersive pair family added one import line and one
    # list entry to `registry.py`, which the whole-step gate imports, so the
    # T2145 artifact stopped describing the tree the moment the family landed. The
    # re-run reproduced the T2145 measurement EXACTLY — 60 cases, 12 complete steps
    # each, 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused,
    # zero divergences — which is the evidence that registering an UNWIRED arm
    # changed no selection and no byte. The `direct` artifact above does not import
    # `registry.py` and is untouched, so it keeps its own path.
    #
    # RE-CUT A THIRD TIME 2026-08-20T02:57, fresh directory, same rule and now a
    # SECOND instance of the same cause: the fused MAGNETIC pair family added one
    # import line and one list entry to `registry.py`. The T0015 artifact was
    # ALREADY stale before that landed — `fused_dispersive_pair.py` had drifted
    # from it when that module's binding-ceiling docstring was corrected — so this
    # re-run clears both drifts at once. It reproduced the measurement EXACTLY
    # again, to the same six figures, which is now twice that registering an
    # UNWIRED arm has been shown to change no selection and no byte rather than
    # once. Superseded eighteen minutes later by T0315 — see below.
    #
    # RE-CUT AGAIN 2026-08-20T03:15, and the reason is the rule working rather
    # than a mistake being papered over: `fused_magnetic_pair.py` was edited AFTER
    # T0257 ran (its packed-struct device is now read from `residency.device`
    # instead of sniffed off a bound tensor), so T0257 stopped describing the tree
    # the moment that landed. THAT IS EXACTLY WHAT THIS TEST IS FOR. The re-run
    # reproduced the measurement to the same six figures a third time, which is
    # also the evidence that the edit was byte-neutral.
    #
    # RE-CUT A FOURTH TIME 2026-08-20T06:55, fresh directory, same rule and now
    # a THIRD instance of the same cause — with TWO families behind it rather
    # than one, which is worth recording rather than smoothing over. The folded
    # fused MAGNETIC pair and `complex_fused_magnetic_pair` each added one import
    # line and one list entry to `registry.py`, which the whole-step gate
    # imports, so T0315 stopped describing the tree the moment the first of them
    # landed. Diffing the two artifacts' source tables is how that was
    # established rather than assumed: T0315 records 77 imported sources and
    # T0655 records 79, the two new paths are exactly those families' modules,
    # and `registry.py` is the ONLY digest that changed.
    #
    # The re-run reproduced the measurement EXACTLY a fourth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences — with the arm table TWO rows wider (87
    # registered, 82 wired, against T0315's 85/82). That the WIRED count is
    # unchanged while the registered count moved is the measurement: an unwired
    # arm is enumerable and cannot be selected.
    #
    # RE-CUT A FIFTH TIME 2026-08-20T23:30, fresh directory, same rule and now with
    # FIVE families behind it in one re-cut rather than two — the three
    # BELOW-THE-CUT magnetic welds (nonlinear, real-beta, BFAST) plus
    # `cylindrical_fused_magnetic_pair` and `fused_ade_chain`, each of which added
    # one import line and one list entry to `registry.py`. Established by diffing
    # the two artifacts' source tables rather than assumed: T0655 records 79
    # imported sources and T2330 records 84, the five new paths are exactly those
    # families' modules, and `registry.py` is again the ONLY digest that changed.
    #
    # The re-run reproduced the measurement EXACTLY a fifth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences. Five more unwired arms, not one selection
    # and not one byte different.
    #
    # RE-CUT A SIXTH TIME 2026-08-20T2330b, fresh directory, same rule — and this
    # one has TWO causes, only one of them its own, which is worth saying
    # plainly rather than smoothing into a single sentence.
    #
    # Established by DIFFING the two artifacts' source tables, not assumed:
    # T2330 records 84 imported sources and T2330b records 85. ONE path is new
    # (`cylindrical_real_fused_magnetic_pair.py`, a family that landed after T2330
    # ran and is not this re-cut's) and TWO digests changed —
    # `cylindrical_fused_magnetic_pair.py`, whose module docstring gained its
    # DEVICE STATUS block once its own byte gate released, and `registry.py`,
    # which the new family's import line moved. Nothing else in the table moved.
    #
    # The re-run reproduced the measurement EXACTLY a sixth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences, byte-for-byte the same totals dict. A
    # sixth family registered UNWIRED, and a docstring on a seventh: not one
    # selection and not one byte different.
    #
    # RE-CUT A SEVENTH TIME 2026-08-20T2200, fresh directory, and this one has a
    # DIFFERENT cause from the six before it — worth saying rather than folding
    # into "another family landed". No family landed and no import line moved:
    # ONE ALREADY-REGISTERED MODULE'S BYTES CHANGED.
    #
    # Established by DIFFING the two artifacts' source tables, not assumed: T2330b
    # and T2200 both record 85 imported sources, NOTHING was added, NOTHING was
    # removed, and exactly ONE digest changed — `fused_ade_chain.py`, edited after
    # T2330b ran. Three edits, none of which moves an emitted byte: its
    # launch-time alias check now also requires the pole OUTPUTS to be distinct
    # from each other; a dead exported helper was removed; and its three arms were
    # split across three FAMILY NAMES, because the arm table holds at most one row
    # per (family, slot) and three rows from one family would have left `update_E`
    # permanently UNSELECTED.
    #
    # That last one is the interesting part and the reason this re-cut is evidence
    # rather than bookkeeping: the split CHANGES THE ARM TABLE, and the whole-step
    # arbiter is the thing that would see it if it changed a selection. It did
    # not. The re-run reproduced the measurement EXACTLY a seventh time — 60 cases,
    # 12 complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences, byte-for-byte the same totals dict.
    #
    # RE-CUT AN EIGHTH TIME 2026-08-21T0115, fresh directory, and this one is back
    # to the ordinary cause: a family landed. `folded_complex_fused_magnetic_pair`
    # — the folded Bloch `step_B` welded into `update_H`, the intersection of the
    # two shipped B/H pairs — added one import line and one `FAMILY_MODULES` entry
    # to `registry.py`, which the whole-step gate imports, so T2200 stopped
    # describing the tree the moment the family registered.
    #
    # ONE MORE DIGEST MOVED AND IT IS NOT THIS FAMILY'S:
    # `cylindrical_real_fused_magnetic_pair.py` was edited after T2200 ran, by
    # a separate change made meanwhile. Recorded here rather than left implicit,
    # because a re-cut that folds in another change's uncertified bytes should say
    # so: this artifact asserts that the ARBITER saw no selection change, not that
    # that family's bytes were measured — its own device gate is what would say
    # that, and at the time of this re-cut it had released but was not yet welded.
    #
    # The re-run reproduced the measurement EXACTLY an eighth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences, byte-for-byte the same totals dict. A
    # seventh family registered UNWIRED: not one selection and not one byte
    # different.
    #
    # RE-CUT AGAIN 2026-08-21T0300, fresh directory, and the cause is worth naming
    # rather than folding into the entry above: the T0115 re-cut recorded the new
    # family's module at a digest that moved TWO HOURS LATER, when its "what it is
    # worth" section was rewritten from the ranked-gap table's reachable CEILING (4)
    # to the number the re-priced matrix actually returned (+2, 83 -> 85 on the named
    # baseline). A docstring is not arithmetic, and none of the 60 cases moved — but
    # the artifact's claim is about BYTES, and an evidence entry that quietly tolerated
    # "only a comment changed" would be the exemption this whole test exists to refuse.
    # Re-running is the fix; a KNOWN_DRIFT entry would only hide it.
    #
    # The re-run reproduced the measurement EXACTLY a ninth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal words
    # censused, zero divergences, byte-for-byte the same totals dict.
    # RE-CUT AGAIN 2026-08-21T0545, fresh directory, and this one has TWO causes
    # with only one of them its own — the shape the eighth entry above
    # already had, and worth stating plainly rather than folding into one sentence.
    #
    # THIS RE-CUT'S OWN CAUSE is the ordinary one: TWO families landed together, both
    # closing cells the 2026-08-20 fusion matrix listed as `FITS — NOT BUILT` and
    # both driven by the SAME four corpus rows. `complex_fused_ade_chain` (the
    # complex64 sibling of the E->P chain) and `complex_conductive_fused_pair` (the
    # complex conductive `step_D` welded into `update_E`) each added one import line
    # and one `FAMILY_MODULES` entry to `registry.py`, which the whole-step gate
    # imports, so T0300 stopped describing the tree the moment the first registered.
    #
    # THE OTHER CAUSE IS NOT THIS RE-CUT'S and is recorded rather than left
    # implicit: `folded_fused_magnetic_pair.py` was edited by a separate change
    # made meanwhile — the `fill_folded_far_ghosts_B` carry — after T0300 ran. Unlike the
    # eighth entry's case, those bytes ARE certified at re-cut time: their own
    # device gate is welded as `metal_folded_fused_magnetic_pair_device_gate`
    # (PASS, recorded 2026-08-20T22:40:00Z, records
    # `results/metal_folded_far_carry_2026-08-20T2/`). This artifact still asserts
    # only what the ARBITER saw, which is that no selection and no byte moved.
    #
    # Established by DIFFING the two artifacts' source tables, not assumed: T0300
    # records 86 imported sources and T0545 records 88, the two new paths are
    # exactly the two new families' modules, and the only two digests that changed
    # are `registry.py` and `folded_fused_magnetic_pair.py`. Nothing else moved.
    #
    # The re-run reproduced the measurement EXACTLY a ninth time — 60 cases, 12
    # complete steps each, 44,243,856 uint32 comparisons, 74,276,352 subnormal words
    # censused, zero divergences — with the arm table TWO rows wider and the WIRED
    # count UNCHANGED: 98 registered against T0300's 96, both at 82 wired. That the
    # registered count moved while the wired count did not is the measurement, and
    # it is the ninth time it has been made: an unwired arm is enumerable and cannot
    # be selected.
    # RE-CUT 2026-08-21T09:35, fresh directory, same rule and a new cause: the
    # fusion round edited complex_no_pml_conductive.py AFTER the whole-step gate
    # ran, and that module is reached through registry.py, so the T0545 artifact
    # stopped describing the tree. THE RE-RUN REPRODUCED THE MEASUREMENT EXACTLY —
    # 60 cases, 12 cycles each, 44,243,856 uint32 comparisons, 74,276,352 subnormal
    # words censused, zero divergences, every armed mutation still CAUGHT — which
    # is the evidence that the post-gate edit moved no byte. Note this gate's
    # --out is a DIRECTORY, so the artifact lands one level deeper than the others.
    # RE-CUT 2026-08-21, fresh directory, same rule and the round's own cause: the
    # far carry landed in folded_fused_pair.py, folded_complex_fused_magnetic_pair.py
    # and folded_fused_magnetic_pair.py, all three reached through registry.py, so the
    # earlier artifact stopped describing the tree the moment those bytes moved. THE
    # RE-RUN REPRODUCED THE MEASUREMENT EXACTLY A TENTH TIME — 60 cases, 12 cycles
    # each, 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused, zero
    # divergences — and the arm table is UNCHANGED at 98 registered, 82 wired. That
    # matters more here than usual: this round added a fifth pass to TWO families'
    # REPLACES tuples, and the wired count not moving is what says a declaration on an
    # UNWIRED arm changes no dispatch and no byte.
    # RE-CUT 2026-08-26, fresh directory, same rule and this round's own cause: every
    # fused-pair module gained a ``replaces=REPLACES`` argument to its ``arms.register``
    # call, and ``arms.py`` gained the ``replaces`` slot and the ``is_weld`` property
    # that argument feeds. Both are reached through registry.py, so the 08-23 artifacts
    # stopped describing the tree the moment those bytes moved. THE RE-RUN CARRIES ZERO
    # DRIFT over 89 recorded digests and released=True. What says the edit moved no
    # byte is not the re-run alone but two independent facts beside it: the arm table is
    # UNCHANGED at 98 registered and 82 wired -- a declaration on an UNWIRED arm changes
    # no dispatch -- and ``metal_kernels/fingerprints.json``'s ``kernel_source_sha256``
    # is byte-identical across the edit, so not one character of what the MPS compiler
    # received changed. The three sub-step gates whose disjointness sweeps this round
    # re-partitioned (bfast, complex, special_kz) all report released=True.
    # RE-CUT 2026-08-27, fresh directory, same rule and this round's own cause:
    # registering folded_beta_complex_fused_magnetic_pair moved
    # meep_gpu/metal_kernels/registry.py, which the whole-step gate imports, so the
    # 08-26 artifact stopped describing the tree the moment the family landed. The
    # arbiter was RE-RUN rather than the drift declared, and the fresh artifact carries
    # ZERO drift over its 90 recorded digests with released=True.
    # RE-CUT 2026-08-27 (routing), fresh directory, same rule. TWO causes, and the
    # second is the interesting one rather than bookkeeping.
    #
    # THE ORDINARY CAUSE: the routing round moved metal_kernels/launch.py plus
    # triton_kernels/launch.py and triton_kernels/coverage.py -- the deposit-repair
    # wiring -- and moved four fused-pair modules with it. All are reached through
    # registry.py, which this gate imports.
    #
    # THE SECOND CAUSE IS A GATE DEFECT THIS ROUND FOUND AND FIXED, recorded here
    # because it changes what an earlier artifact was worth. cylindrical_real's
    # prefix_wall_row_dropped mutation had been carried as a known flake (CAUGHT 7/8,
    # then 5/8, 8/8 when run alone). It was not a flake. The 2026-08-19 re-arm replaced
    # an out-of-bounds read with src[(nxi - 1) * nyz + base] and called that "in bounds
    # and deterministic"; in bounds it was, deterministic it was not, and the claim was
    # asserted rather than measured. That row is the WALL ROW -- the one the kernel
    # never writes -- living in PLAN-OWNED scratch reused across launches, so it held
    # whatever the previous tenant left, which is why it passed run-alone and failed in
    # a full sweep. The needle is now a literal 1.0f: same defect class, no read of
    # undefined memory. Measured CAUGHT 8/8 on five consecutive full runs.
    #
    # What says the real kernel was never implicated: the UNMUTATED cylindrical_real
    # legs were run ten consecutive times and are bit-stable 10/10, and 35 of the 36
    # mutations reported identical catch counts across runs. The nondeterminism lived
    # in the mutant, not in the path the welds pin.
    #
    # The re-run carries ZERO drift over its 90 recorded digests with released=True,
    # and the round re-gated 43 of 43 Metal families green.
    # RE-CUT 2026-08-28 (deposit), fresh directory, same rule. The round added two
    # rows to metal_kernels/launch.py's FUSED_PAIR_ARMS and flipped
    # CARRIES_DEPOSIT_REPAIR on the two families those rows name, so the composer
    # this gate imports through registry.py moved. WHAT SAYS THE EDIT MOVED NO
    # DISPATCHED BYTE, beside the re-run itself: both families still register
    # wired=False, so plan_step cannot select either, and `fuse` is still False at
    # the one shipped call site — the two new rows only bound which families the
    # opt-in seam loop may reach. The re-run carries ZERO drift over its 90 recorded
    # digests with released=True, and 43 of 43 Metal families re-gated green.
    # RE-CUT 2026-08-28 (folded carry), fresh directory, same rule. The round flipped
    # CARRIES_DEPOSIT_REPAIR on the two FOLDED families and edited the absorb table's
    # comment in metal_kernels/launch.py, all reached through registry.py. The re-run
    # reproduced the measurement EXACTLY -- 60 cases, 12 complete steps each,
    # 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused, zero
    # divergences -- over 90 recorded digests with released=True, which is the evidence
    # that a flag and three comments changed no selection and no byte.
    # RE-CUT 2026-08-28 (guard), fresh directory, same rule. `deposit_repair`'s
    # fail-open guard was closed (see the entry above), and every Metal gate imports it.
    # The re-run reproduced the measurement EXACTLY -- 60 cases, 12 complete steps each,
    # 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused, zero
    # divergences, `totals` and `legs` byte-equal to the `_carry2` artifact -- over 90
    # recorded digests with released=True. That equality is the evidence that closing the
    # guard changed nothing the whole step computes: the new clauses fire only when a
    # caller hands the guard a PML layer, and every configuration this gate steps carries
    # an ACTIVE one, which the clauses admit. (`..._failclosed2` beside it is the same
    # measurement against the bytes before the guard's rank clause landed; superseded,
    # bound by nothing, and kept rather than tidied away.)
    # RE-CUT 2026-08-28 (arity), fresh directory, same rule. The round built the
    # D->E (PML, ordinary) fused pair (`fused_electric_pair.py`), which reaches this
    # gate through `registry.py` along with the absorb row it needed in `launch.py`.
    # The re-run reproduced the measurement EXACTLY -- 60 cases, 12 complete steps
    # each, 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused, zero
    # divergences -- with released=True. That equality is what says a product
    # registered UNWIRED changed no selection and no byte: the arbiter steps the same
    # sixty configurations it stepped before the family existed and gets the same
    # words back, which is the same measurement this entry has now recorded on every
    # family added to the table.
    # RE-CUT 2026-08-29 (fused route), fresh directory, for the same cause as the
    # entry above: `fastpath.py` gained the fusion opt-in and the fused arms'
    # certification rows, and this gate imports the package. ZERO drift over its 91
    # recorded digests with released=True.
    # RE-CUT 2026-08-29 (release), fresh directory, for the cause the entry above
    # states: `fastpath.py` gained the fused-route release and this gate imports
    # the package.
    # RE-CUT AGAIN 2026-08-29 (fusion veto, then the gate name), fresh directory,
    # same cause twice more: `fastpath.py` gained `FUSE_ARMS_VETO` and the clause
    # that reads it, and then `DRIVER_ROUTE_FUSED_GATE` was repointed at the run
    # that drove the SHIPPED route. See the entry above for both. ZERO drift over
    # its 91 recorded digests, 60 cases, no divergences, released=True.
    # RE-CUT 2026-08-30 (bind), fresh directory, for the cause the entry above
    # states: the record, not the tree, was what had drifted -- `fastpath.py` was
    # pinned here at `87f740ee`, a digest that exists neither at HEAD nor on disk.
    # 60 cases, no divergences, ZERO drift over its 91 recorded digests,
    # released=True.
    # RE-CUT 2026-08-30 (deposit carry), fresh directory, same rule and this round's
    # own cause: the THREE TRITON fused families flipped CARRIES_DEPOSIT_REPAIR once
    # their device gates were re-run against the flipped bytes, and
    # `triton_kernels/launch.py`'s absorb-table prose moved with them -- the two
    # comment blocks that stated the fold boundary rested on the flag being False.
    # This gate pins `meep_gpu/triton_kernels/launch.py`, so the `_bind` artifact
    # stopped describing the tree the moment that prose moved.
    #
    # WHAT SAYS THE EDIT MOVED NO METAL BYTE, beside the re-run: the whole change on
    # the Metal side of this artifact is a COMMENT in a file no Metal product
    # executes -- `metal_kernels` imports nothing from `triton_kernels.launch` -- and
    # `metal_kernels/fingerprints.json`'s `kernel_source_sha256` is byte-identical
    # across the round, so not one character of what the MPS compiler received
    # changed. Re-run rather than declared, because a comment-only edit is still an
    # edit and this file's KNOWN_DRIFT is empty on purpose.
    # 60 cases, no divergences, ZERO drift over its 91 recorded digests,
    # released=True.
    #
    # RE-CUT 2026-08-30 (arms rows), fresh directory, for the cause recorded on the
    # artifact above: FUSED_PAIR_ARMS went 6 rows -> 15 with this round's nine new
    # products, and this gate imports `metal_kernels/launch.py` directly -- it plans
    # and runs COMPLETE driver steps, so the composer is not merely on its import
    # path, it is what builds the plans it compares. The `_carry` artifact stopped
    # describing the tree the moment the table moved. Re-run rather than declared.
    #
    # RE-CUT 2026-08-31 (the coefficient-pack round), for the cause recorded on the
    # dispersive artifact above: FUSED_PAIR_ARMS went 15 rows -> 17 with the two Dcyl
    # D->E pairs, and this gate imports `metal_kernels/launch.py` DIRECTLY -- it plans
    # and runs COMPLETE driver steps, so the composer is what builds the plans it
    # compares. Re-run rather than declared, and KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-08-31 AGAIN (the plain-repair round), for the cause recorded on the
    # dispersive artifact above: `meep_gpu/deposit_repair.py` gained the SECOND repair
    # and `meep_gpu/triton_kernels/launch.py`'s `_install_fused_pair` gained the
    # argument that names which repair a product carries. This gate plans and runs
    # COMPLETE driver steps, so both files are on the path that builds the plans it
    # compares. Re-run rather than declared, and KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-09-01 (the last-cells round), for the cause recorded on the
    # dispersive artifact above: FUSED_PAIR_ARMS went 17 rows -> 19 with the two
    # no-absorber stored-E D->E pairs, the METAL `_install_fused_pair` gained the
    # `repair_paths` threading, and `registry.py` gained the two imports. This gate
    # imports `metal_kernels/launch.py` DIRECTLY -- it plans and runs COMPLETE
    # driver steps, so the composer is what builds the plans it compares. Re-run
    # rather than declared (the whole-board re-cut at
    # `metal_regate_2026-09-01_lastcells`, every gate green), and KNOWN_DRIFT
    # stays empty.
    #
    # RE-CUT 2026-09-01 AGAIN (the residue round), for the cause recorded on the
    # dispersive artifact above: FUSED_PAIR_ARMS went 19 rows -> 23 with the four
    # residue welds, `registry.py` gained the four imports, the conductive
    # no-PML pair's injection clause was lifted onto `driver.py`'s sparse
    # per-deposit-cell rescale, and this gate imports `metal_kernels/launch.py`
    # and `meep_gpu/driver.py` DIRECTLY -- it plans and runs COMPLETE driver
    # steps. Re-run rather than declared (the whole-board re-cut at stamp
    # `2026-09-01_residue`, every gate green), and KNOWN_DRIFT stays empty.
    # RE-CUT 2026-09-02 (the fastpath repoint), fresh directory, same cause as the
    # dispersive_update_e entry above: `fastpath.DRIVER_ROUTE_FUSED_GATE` had to
    # move so the dispatch record could be cut from a run of the sparse-rescale
    # `driver.py`, and this gate imports fastpath.py. Re-run rather than declared:
    # the fresh artifact carries 112 recorded digests and ZERO drift.
    #
    # RE-CUT 2026-09-02 AGAIN (the folded-release round), same cause as the
    # dispersive_update_e entry above: the release predicate, the composer's
    # `fuse_labels` offer, and this package's corrected dispatch docstrings all
    # moved, and this gate plans and runs COMPLETE driver steps through every one
    # of them. Re-run rather than declared (`2026-09-02_foldedrelease`, 54 of 54
    # green), and KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-09-04 (the cylindrical m = 0 complex round), same cause as the
    # dispersive_update_e entry above: the cylindrical complex family's `M_ZERO`
    # arm moved four `metal_kernels` modules this gate plans through. Re-run rather
    # than declared (`2026-09-04_m0complex`), and KNOWN_DRIFT stays empty.
    #
    # RE-CUT 2026-09-04 AGAIN (the H_to_D Phase 0 round), same cause as the
    # dispersive_update_e entry above: the driver's by-name sync consult, the
    # planner constant behind it, and both composers' H_to_D seam row all moved,
    # and this gate plans and runs COMPLETE driver steps through every one of
    # them. Re-run rather than declared (`2026-09-04_p0regate`, 54 of 54 green),
    # and KNOWN_DRIFT stays empty.
    #
    # RE-CUT ONCE MORE the same day, same cause as the dispersive_update_e entry
    # above: `fastpath.DRIVER_ROUTE_FUSED_GATE` was repointed after the
    # `_p0regate` fleet had run, and this gate plans COMPLETE driver steps
    # through `fastpath.py`. `2026-09-04_syncchannel`, 54 of 54 green, and
    # KNOWN_DRIFT stays empty.
    # RE-CUT ONCE MORE 2026-09-04, at `2026-09-04_arbfix`, same cause: the
    # composer's earlier-neighbour arbitration half was removed from
    # `metal_kernels/launch.py` after `_syncchannel` had run, and this gate plans
    # COMPLETE driver steps through the composer. 54 of 54 green, KNOWN_DRIFT empty.
    # RE-CUT ONCE MORE 2026-09-05, at `2026-09-05_hdweld`, same cause as the
    # dispersive_update_e entry above: `registry.py` gained the `fused_hd_pair` row
    # after `_arbfix` had run, and this gate plans COMPLETE driver steps through the
    # composer that reads it. The re-run is the evidence the new UNWIRED arm changed
    # no selection: 54 of 54 green, KNOWN_DRIFT empty.
    # RE-CUT ONCE MORE 2026-09-08, at `2026-09-08_regate`, same cause as the
    # dispersive_update_e entry above: the H->D wiring merge moved `fastpath.py`,
    # `metal_kernels/{launch,registry,fused_hd_pair}.py` and `triton_kernels/launch.py`
    # after `_wired` had run, and this gate plans COMPLETE driver steps through the
    # composer that reads them. Measured on `_wired` before the re-run: 5 stale
    # digests of 119. The fresh artifact records 124 -- the wiring is coverage the old
    # evidence did not have -- and carries ZERO drift. KNOWN_DRIFT stays empty.
    # RE-CUT ONCE MORE 2026-09-11, at `2026-09-11_dispatch`, same cause as the
    # dispersive_update_e entry above: the dispatch batch moved `fastpath.py`, the
    # `metal_kernels` files and `metal_gate_runner.py` after `_close` had run, and
    # this gate plans COMPLETE driver steps through the composer that reads them.
    # Re-run rather than declared, and KNOWN_DRIFT stays empty.
    # RE-CUT ONCE MORE 2026-09-13, at `2026-09-13_batch`, same cause as the
    # dispersive_update_e entry above: the Triton dispatch release (Phase B) moved
    # `meep_gpu/fastpath.py`, which this gate plans COMPLETE driver steps through,
    # after `_dispatch` had run, so its recorded `fastpath.py` digest drifted. No
    # Metal byte moved -- the release reaches only the Triton route. The whole
    # Metal fleet was re-gated at this batch stamp; the fresh artifact records 124
    # digests and carries ZERO drift, released=True. Re-run rather than declared,
    # and KNOWN_DRIFT stays empty.
    # RE-CUT AGAIN 2026-09-21, at `2026-09-20_restrict`, same cause as above.
    # RE-CUT 2026-09-19, at `2026-09-19_witness`, same causes as the
    # dispersive_update_e entry above plus two of this gate's own. The `_batch`
    # artifact had drifted on SIX recorded digests of 124: the four named there, and
    # `metal_kernels/cylindrical_real_fused_hd_pair.py` and
    # `metal_kernels/nonlinear_fused_magnetic_pair.py`, which moved only in the
    # `_17_allpaths` round and which this gate records because it plans COMPLETE
    # driver steps through the composer that imports them. Dated by diffing the
    # recorded tables of every twin since (`_14_target`, `_15_gapclose`,
    # `_15_offdiag`, `_17_allpaths`; this gate was not re-cut in either weldgrid
    # round). The `_17_allpaths` twin is NOT pinned: it still drifts on
    # `deposit_repair.py`, one digest of 124, which moved on 2026-09-19 after it
    # ran. The witness fleet re-cut this gate
    # (`metal_campaign_2026-09-19_witness/fleet/gate_metal_whole_step.log`): 60
    # cases, 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused,
    # divergences none -- the same figures as every earlier cut -- released=True,
    # 124 recorded digests and ZERO drift. `metal_kernels/fingerprints.json:7029`
    # already reads its verdict from this same artifact. Re-run rather than
    # declared, and KNOWN_DRIFT stays empty.
    # RE-CUT AGAIN 2026-09-24 at `_2026-09-24_round3`, same cause: the round-3 fleet
    # re-ran the whole-step gate (released=True in its fleet log), 250 recorded digests,
    # ZERO drift against the tree.
    # RE-CUT AGAIN 2026-09-25 at `_2026-09-25_night`, same cause: released=True in the `_2026-09-25_night`
    # fleet, ZERO recorded digests disagreeing with the tree.
    # RE-CUT AGAIN 2026-10-06 at `_2026-10-06_092_g13s`, same cause as the
    # dispersive_update_e entry above: 891243a and 4c19311 moved `meep_gpu/fastpath.py`
    # after the `_2026-10-04_g13s` artifact had run, one recorded digest of 125. The re-run on
    # 06435a9: 60 cases, 44,243,856 uint32 comparisons, 74,276,352 subnormal words censused,
    # divergences none -- the same figures as every earlier cut -- released=True, 125
    # recorded digests and ZERO drift. KNOWN_DRIFT stays empty.
    "metal_whole_step_2026-10-06_092_g13s/whole_step.json",
)

#: path-substring -> why it is known stale. MUST shrink, never grow silently.
#: EMPTY, and it should stay that way. The 2026-08-18 entry
#: (complex_dispersive_update_e.py, edited two minutes after its runs) was cleared
#: on 2026-08-19 by RE-RUNNING both gates on this machine's MPS device, not by
#: widening this list. The re-run reproduced 43,865,856 uint32 comparisons exactly,
#: with zero divergences, and every recorded source now matches the tree.
KNOWN_DRIFT: dict = {}


def _recorded_digests(artifact: pathlib.Path):
    """(path, sha) pairs from the artifact's json, or its sidecar digest list."""
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    # imported_source_sha256 FIRST: that is the shared key both tracks now write
    # through gate_provenance. The older per-gate spellings stay readable so
    # artifacts cut before the unification are still verifiable.
    table = (doc.get("imported_source_sha256")
             or doc.get("source_sha256") or doc.get("provenance") or {})
    if isinstance(table, dict) and table:
        return [(p, s) for p, s in table.items() if isinstance(s, str)]
    sidecar = artifact.with_name(artifact.stem + "_source_sha256.txt")
    if sidecar.exists():
        pairs = []
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[1], parts[0]))
        return pairs
    return []


@pytest.mark.parametrize("relative", CURRENT_EVIDENCE)
def test_a_metal_artifacts_recorded_sources_still_match_the_tree(relative):
    artifact = RESULTS / relative
    if not artifact.exists():
        pytest.skip(f"{relative} not present in this checkout")

    pairs = _recorded_digests(artifact)
    assert pairs, (
        f"{relative} records no source digests at all — it cannot support a claim "
        f"about which bytes produced it")

    undeclared, declared = [], []
    for path, sha in pairs:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() == sha:
            continue
        # THE RAW BYTES, AND KNOWN_DRIFT IS THE ONE ALLOWANCE — named, per file, and
        # recorded. A ``weld_survives_edit`` fall-through stood here and was removed on
        # 2026-08-29: it sat BEFORE the declaration check, so a drift it cleared skipped
        # KNOWN_DRIFT altogether, which is the one mechanism in this file that makes a
        # surviving drift visible to a reader. Measured before removing it -- these
        # artifacts carry no device_sha256/code_sha256 map, so the helper returned False
        # for every path and the branch admitted nothing; the two artifacts that fail
        # today (meep_gpu/fastpath.py) fail identically with and without it. What it did
        # carry was the readiness to start admitting, silently, the moment either block
        # was added to an artifact.
        if any(marker in path for marker in KNOWN_DRIFT):
            declared.append(path)
        else:
            undeclared.append(path)

    assert not undeclared, (
        f"{relative} records sources that have since changed and are NOT declared "
        f"in KNOWN_DRIFT: {sorted(undeclared)}. The artifact's result does not "
        f"cover the bytes that ship. Re-run the gate — do not add the file to "
        f"KNOWN_DRIFT to make this pass unless you are recording, not hiding, it.")
    # Non-vacuity: this test must actually be comparing something.
    assert len(pairs) >= 20, f"{relative}: only {len(pairs)} digests — reader broken?"


def test_the_known_drift_list_is_still_describing_real_drift():
    """A cleared entry must be REMOVED, so the list cannot rot into fiction."""
    stale = []
    for marker in KNOWN_DRIFT:
        still = False
        for relative in CURRENT_EVIDENCE:
            artifact = RESULTS / relative
            if not artifact.exists():
                continue
            for path, sha in _recorded_digests(artifact):
                if marker in path and pathlib.Path(path).exists():
                    # THE SAME SPELLING OF "DRIFTED" AS THE TEST ABOVE, which is the
                    # whole point: this test says an entry must go when it no longer
                    # drifts, and that one says a drift must be declared. Give them two
                    # different notions of drift and a path can be required to be
                    # declared and required to be undeclared at the same time. Both
                    # briefly consulted ``weld_survives_edit``; both now read the bytes.
                    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() != sha:
                        still = True
        if not still:
            stale.append(marker)
    assert not stale, (
        f"KNOWN_DRIFT names {stale}, which no longer drift — the gate was re-run "
        f"or the file reverted. Delete the entry; a debt list that outlives its "
        f"debt stops being read.")

"""Every device gate must record the source bytes IT imported.

WHY THIS IS A TEST AND NOT A CONVENTION. Measured 2026-08-17: a weld recorded
``8ec99ec2...`` for ``no_pml_stored_e.py`` while the gate had actually executed
``e24660a2...``. The digest had been transcribed by hand from one of the GPU host's
67 staged trees, chosen by eye, and the wrong one was picked. The weld test caught
it only because the mis-transcribed value happened to differ from the local file —
where a wrong tree HAPPENS to hold the local bytes, the same error is silent and
permanent, because that test compares recorded against local, not against what ran.

So the transcription step is being removed gate by gate, and this file is the
ratchet: the wired set may only grow.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"

#: ALL THREE TRACKS, ONE RULE. Triton, Metal and CUDA gates answer "which source
#: produced this artifact?" through the same helper and the same key, so a single
#: verifier reads any of them and no track can drift to a weaker standard.
TRACKS = {
    "triton": {"pattern": "gate_triton_*.py"},
    "metal":  {"pattern": "gate_metal_*.py"},
    "cuda":   {"pattern": "gate_cuda_*.py"},
}

#: The call spellings that MEAN "stamp here". ``_stamp_provenance`` is the local
#: wrapper most gates bind; ``gate_provenance.stamp`` is the helper itself, which
#: three CUDA gates call directly. Both were live in the tree on 2026-08-20 and a
#: marker that knew only the first read those three as recording nothing.
HELPER_MODULE = "gate_provenance"


def _stamp_spellings(text):
    """Every way THIS file spells a call to the provenance stamp.

    Returned as literal call prefixes (``"kit.stamp("``), so a window check is a
    substring test against the spellings that file actually uses — not against a
    single blessed name that the next gate is free to not use.
    """
    import ast as _ast
    spellings = {"_stamp_provenance("}
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return spellings
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for a in node.names:
                if a.name == HELPER_MODULE:
                    spellings.add(f"{a.asname or a.name}.stamp(")
        elif isinstance(node, _ast.ImportFrom) and node.module == HELPER_MODULE:
            for a in node.names:
                if a.name == "stamp":
                    spellings.add(f"{a.asname or a.name}(")
    return spellings


def _stamps_before_writing(text):
    """Does this module define a ``save()`` that stamps BEFORE it serialises?"""
    if "def save(" not in text:
        return False
    body = text[text.index("def save("):]
    nxt = body.find("\ndef ", 1)
    if nxt != -1:
        body = body[:nxt]
    if "json.dump(" not in body:
        return False
    hits = [body.index(s) for s in _stamp_spellings(text) if s in body]
    return bool(hits) and min(hits) < body.index("json.dump(")


def _stamping_writers():
    """DERIVED, not listed. A gate that binds one of these delegates safely.

    MEASURED 2026-08-19, and the reason this keys on WRITERS rather than on an
    import: gate_cuda_offdiag imports gate_triton_complex AND binds
    ``save = probe.save`` from probe_fused_kernel_bit_identity. An import-based
    check passed it while it wrote every verdict with no provenance at all.

    MEASURED 2026-08-20, and the reason the set is derived rather than written
    down: the hand-listed version named three writers while the tree held more,
    so gate_cuda_no_pml_null_constitutive — which binds ``save = ref.save`` from
    a Triton gate that stamps correctly — was reported as unwired. A list of
    writers goes stale exactly as silently as a list of gates does.
    """
    return {path.name for path in sorted(PARITY.glob("*.py"))
            if _stamps_before_writing(path.read_text(encoding="utf-8"))}


STAMPING_WRITERS = _stamping_writers()

#: The derivation must never come back empty or near-empty: these three were the
#: hand-maintained set, and any of them dropping out means the scan broke, not
#: that the tree changed.
REQUIRED_WRITERS = {"gate_triton_complex.py", "metal_gate_kit.py",
                    "probe_fused_kernel_bit_identity.py"}

#: DECLARED DEBT per track: gates recording nothing, directly or by delegation.
#: All ZERO. The numbers may only go DOWN.
UNWIRED_BUDGET = {"triton": 0, "metal": 0, "cuda": 0}


def _sources(track):
    return sorted(PARITY.glob(TRACKS[track]["pattern"]))


def _write_sites(text):
    """Line indices that actually SERIALISE a payload.

    Bare ``json.dumps(...)`` returns a string and writes nothing — it appears in
    log lines, JSONL progress rows and ``json.loads(json.dumps(x))`` deep copies.
    Chasing those with an exclusion list was wrong three times on 2026-08-19; the
    rule is that a write is ``json.dump(`` (to a handle) or
    ``write_text(json.dumps(`` (to a path), and nothing else.
    """
    lines = text.splitlines()
    return lines, [n for n, line in enumerate(lines)
                   if "json.dump(" in line or "write_text(json.dumps(" in line]


#: Module names (no .py) whose save() stamps.
_STAMPING_MODULES = {name[:-3] for name in STAMPING_WRITERS}


def _delegates_safely(text):
    """Does this gate's ``save`` resolve to a writer that stamps?

    AST, not regex, because the three tracks bind the writer three different ways
    and each spelling defeated a textual marker on 2026-08-19:

      * ``import gate_triton_complex as gate`` then ``save = gate.save``
        — an alias, so "gate_triton_complex.save" never appears;
      * ``from metal_gate_kit import (..., save, ...)``
        — a from-import, so "kit.save" never appears;
      * ``import probe_fused_kernel_bit_identity as probe`` then
        ``save = probe.save`` — which an IMPORT-based check passed while the CUDA
        gate wrote every verdict with no provenance at all.

    Resolving the alias table and then the binding answers the real question:
    whichever object this file finally calls ``save``, does IT stamp?
    """
    import ast as _ast
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return False

    alias_to_module = {}
    from_imported_save = False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for a in node.names:
                alias_to_module[a.asname or a.name] = a.name
        elif isinstance(node, _ast.ImportFrom):
            if node.module in _STAMPING_MODULES:
                if any(a.name == "save" for a in node.names):
                    from_imported_save = True
    if from_imported_save:
        return True

    # A DIRECT CALL through the alias: ``kit.save(payload, path)``. Ten Metal
    # gates use exactly this and bind nothing, so binding-only detection missed
    # every one of them.
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute) \
                and node.func.attr == "save" \
                and isinstance(node.func.value, _ast.Name):
            if alias_to_module.get(node.func.value.id) in _STAMPING_MODULES:
                return True

    def _module_of(value):
        if isinstance(value, _ast.Attribute) and value.attr == "save" \
                and isinstance(value.value, _ast.Name):
            return alias_to_module.get(value.value.id, value.value.id)
        return None

    # The LAST binding of the name ``save`` wins. Both spellings seen in tree:
    #   save = kit.save
    #   log, save, differing = kit.log, kit.save, kit.differing   <- tuple form,
    # which single-target detection missed on all ten Metal delegators.
    bound = None
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, _ast.Name) and target.id == "save":
                bound = _module_of(node.value) or bound
            elif isinstance(target, _ast.Tuple) and isinstance(node.value, _ast.Tuple):
                for slot, value in zip(target.elts, node.value.elts):
                    if isinstance(slot, _ast.Name) and slot.id == "save":
                        bound = _module_of(value) or bound
    return bound in _STAMPING_MODULES


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_every_gate_records_provenance_directly_or_by_delegation(track):
    gates = _sources(track)
    assert gates, f"{track}: no gates discovered — glob broken?"
    unwired = []
    for path in gates:
        text = path.read_text(encoding="utf-8")
        if any(s in text for s in _stamp_spellings(text)) or _delegates_safely(text):
            continue
        lines, writes = _write_sites(text)
        if not writes and "save(" not in text:
            continue                       # writes nothing at all
        unwired.append(path.name)
    assert len(unwired) <= UNWIRED_BUDGET[track], (
        f"{track}: {len(unwired)} gates record no provenance, budget is "
        f"{UNWIRED_BUDGET[track]}: {sorted(unwired)}")


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_every_payload_write_is_preceded_by_a_stamp(track):
    """EVERY write site, not merely the first.

    Measured 2026-08-19: the first mechanical wiring anchored on the first
    ``json.dump(...)`` per file. In gate_triton_complex_no_pml_curl that
    occurrence sits inside the REFUSED early-exit branch, so a run that actually
    released wrote its artifact from a different branch and recorded NO provenance
    at all — an empty record beside RELEASED=True.
    """
    offenders = []
    for path in _sources(track):
        text = path.read_text(encoding="utf-8")
        lines, writes = _write_sites(text)
        spellings = _stamp_spellings(text)
        for n in writes:
            window = "\n".join(lines[max(0, n - 3):n])
            if not any(s in window for s in spellings):
                offenders.append(f"{path.name}:{n + 1}")
    assert not offenders, (
        f"{track}: payload writes not preceded by a stamp: {offenders}")


@pytest.mark.parametrize("writer", sorted(STAMPING_WRITERS))
def test_every_shared_writer_actually_stamps(writer):
    """Delegation only counts if the thing delegated to does the work."""
    text = (PARITY / writer).read_text(encoding="utf-8")
    assert _stamps_before_writing(text), (
        f"{writer}.save no longer stamps before it serialises; every gate "
        f"delegating to it silently stopped recording what it ran")


def test_the_derived_writer_set_did_not_collapse():
    """A derivation that returns nothing would pass every delegation check.

    The set is scanned rather than listed so it cannot go stale — but the failure
    mode a list does not have is returning EMPTY, which would make
    ``_delegates_safely`` answer False everywhere and, worse, make
    ``test_every_shared_writer_actually_stamps`` vacuous by parametrising over
    nothing. These three are the writers the hand-maintained set named.
    """
    missing = REQUIRED_WRITERS - STAMPING_WRITERS
    assert not missing, (
        f"the scan no longer finds {sorted(missing)} — the derivation is broken, "
        f"not the tree. Found: {sorted(STAMPING_WRITERS)}")


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "gate_provenance_under_test", PARITY / "gate_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provenance_keys_are_repo_relative_and_cover_the_kernel_tree():
    helper = _load_helper()
    import meep_gpu.triton_kernels.launch  # noqa: F401  (must be imported to be seen)
    recorded = helper.provenance()
    assert "meep_gpu/triton_kernels/launch.py" in recorded, (
        "an imported kernel module is missing; REPO_ROOT depth is the usual cause "
        "— two levels lands on parity/ and hides the whole kernel tree")
    assert all(not k.startswith("/") for k in recorded), "keys must be repo-relative"
    assert all(len(v) == 64 for v in recorded.values()), "values must be sha256 hex"


def test_the_digest_is_of_the_file_that_was_imported():
    """Not a path built from cwd: the sha must match the module's own __file__."""
    import hashlib
    helper = _load_helper()
    import meep_gpu.triton_kernels.launch as launch
    recorded = helper.provenance()["meep_gpu/triton_kernels/launch.py"]
    with open(launch.__file__, "rb") as handle:
        assert recorded == hashlib.sha256(handle.read()).hexdigest()


def test_an_identical_restamp_is_allowed_but_a_changed_one_is_refused():
    """A gate may serialise one payload twice; two WRITERS may not disagree."""
    helper = _load_helper()
    payload = {"gate": "x"}
    helper.stamp(payload)
    first = dict(payload[helper.IMPORTED_KEY])
    helper.stamp(payload)                      # identical re-stamp: allowed
    assert payload[helper.IMPORTED_KEY] == first
    payload[helper.IMPORTED_KEY] = {"meep_gpu/triton_kernels/launch.py": "0" * 64}
    with pytest.raises(RuntimeError, match="changed DURING this run"):
        helper.stamp(payload)


def test_a_growing_import_set_is_merged_not_refused():
    """Later legs import more modules; that is growth, not disagreement.

    Measured 2026-08-19: whole-dict equality read this as two writers conflicting
    and killed two working gates before they wrote a verdict.
    """
    helper = _load_helper()
    payload = {"gate": "x"}
    helper.stamp(payload)
    payload[helper.IMPORTED_KEY].pop(next(iter(payload[helper.IMPORTED_KEY])))
    shrunk = len(payload[helper.IMPORTED_KEY])
    helper.stamp(payload)                       # must merge the missing one back
    assert len(payload[helper.IMPORTED_KEY]) > shrunk


def test_a_gates_own_source_sha256_is_never_clobbered():
    """The two facts must coexist: a curated binding list and what was imported.

    Measured 2026-08-19: sharing the name killed two gates outright — they build
    their own source_sha256, the stamp raised, and the process died before
    writing a verdict, producing ``release: null`` on a gate that was working.
    """
    helper = _load_helper()
    payload = {"gate": "x", "source_sha256": {"curated/file.py": "a" * 64}}
    helper.stamp(payload)
    assert payload["source_sha256"] == {"curated/file.py": "a" * 64}
    assert helper.IMPORTED_KEY in payload and payload[helper.IMPORTED_KEY]


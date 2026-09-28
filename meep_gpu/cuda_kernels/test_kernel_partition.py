"""The certified/uncertified partition, over the WHOLE package rather than per family.

WHAT THIS FILE IS FOR. Every family module in this package carries two module-level
names -- ``CERTIFIED_KERNELS`` and ``UNCERTIFIED_KERNELS`` -- and a per-module test
beside it checks that its own kernels land in exactly one of them. Those per-module
checks are the ones that know the family: how many kernels it ships, which gate a
dead name owes, what its emitter can produce. What none of them can see is a module
that carries NEITHER name, because a test that reads one file cannot notice a file
that has no test.

MEASURED 2026-08-28, before this file existed: EIGHT family modules shipped kernels
and declared neither name -- ``bfast_curl``, ``conductive_kernels``, ``no_pml_curl``,
``special_kz_curl``, ``complex_beta_kernels``, ``complex_folded_kernels``,
``folded_offdiag_kernels`` and ``dispersive_offdiag_update_e``, eighteen kernels
between them. Six of the eight had RELEASED device verdicts sitting in
``parity/meep_gpu/results/`` that ``certification.json`` did not carry, and the other
two had none and said so nowhere a merge-bar reader would look. Nothing failed,
because there was nothing to fail: the partition was enforced file by file and those
files were not in the list.

So this walks the DIRECTORY. It reads the two names off the syntax tree without
importing anything -- most of these modules ``import cupy`` at scope and this has to
run on a laptop -- and it asks four questions that only a package-wide reader can
ask:

1. Is every kernel written into this package's device text described by exactly ONE
   module's two sets? (Not "its own module's": ``offdiag_emitter`` writes the kernel
   ``offdiag_constitutive_kernels`` owns, and the ownership is the point.)
2. Is every CERTIFIED name backed by a block in ``certification.json`` that claims
   it -- so "certified" is a verdict and never a word someone typed?
3. Is every UNCERTIFIED name absent from that record -- so a name cannot be dead in
   one place and covered in another?
4. Does the record claim a kernel no module certifies -- an orphaned verdict, the
   failure that put six Triton welds in a ledger nothing recomputed?

THE ANNOTATION TRAP, AND WHY THIS READER DOES NOT FALL INTO IT. The per-module
readers match ``ast.Assign`` only, which is why every family module's comment tells
the next author to spell ``UNCERTIFIED_KERNELS`` WITHOUT a type annotation: an
annotated assignment is an ``ast.AnnAssign``, invisible to them, and an invisible
partition is an unenforced one. Two modules annotate anyway
(``cylindrical_kernels.py`` and ``cylindrical_complex_kernels.py``, both
``UNCERTIFIED_KERNELS: Tuple[str, ...] = ()``), so a walk that copied the per-module
reader would silently skip them. This one reads BOTH forms. That is strictly
stronger and it is the only spelling difference this file has an opinion about.

WHAT THIS FILE DOES NOT REPLACE. The per-module partition tests, all of which assert
something this reader cannot derive -- the exact kernel set a family ships, that a
particular family's dead set is EMPTY, which gate a dead name owes, or a set
produced by calling an emitter over its row masks. Those clauses are the reason a
name landing in the wrong half is caught with a family-specific message; this file
is the reason a whole family cannot be missing from the conversation.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

HERE = pathlib.Path(__file__).parent
RECORD = HERE / "certification.json"

#: The kernel entry points a module writes into its own device text. STRICT: the
#: name must be a C identifier, so an emitter's ``void {name}(`` template does not
#: read as a kernel called ``{name}``.
KERNEL_DECLARATION = re.compile(r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_]\w*)\s*\(')

#: The same declaration with the name unconstrained. Where the two disagree the
#: module emits at least one name this reader cannot see, and the "described but not
#: shipped" direction is not asserted for it -- see :func:`device_text_is_complete`.
LOOSE_DECLARATION = re.compile(r'__global__\s+void\s+([^(\s]*)\s*\(')

#: ``complex_emitter``'s substitution sentinel. It is a literal identifier in the
#: template text and it is not a kernel; every emitted source replaces it.
PLACEHOLDERS = frozenset({"__NAME__"})

#: The two names this file walks for.
PARTITION_NAMES = ("CERTIFIED_KERNELS", "UNCERTIFIED_KERNELS")

#: BLINDNESS FLOORS, measured 2026-08-28 on the tree that introduced this file. They
#: are not budgets and they do not encode what the package ought to contain: they are
#: the smallest numbers at which the discovery below is still finding the work. A
#: reader that stops matching -- a renamed constant, a changed declaration spelling,
#: a walk that skips a subdirectory -- reports zero findings and passes every clause
#: vacuously, which is exactly how a partition stops being enforced without anyone
#: seeing a red test. Lower one only with the deletion that justifies it.
MODULES_WITH_A_PARTITION_FLOOR = 21
DECLARED_KERNELS_FLOOR = 29
DESCRIBED_KERNELS_FLOOR = 52


def family_modules() -> list[pathlib.Path]:
    """Every non-test Python module in this package, sorted."""
    return sorted(path for path in HERE.glob("*.py")
                  if not path.name.startswith("test_"))


def partition_sets(source: str) -> dict[str, object]:
    """``{name: value}`` for whichever of the two names the module assigns.

    BOTH SYNTAX FORMS, deliberately -- see the annotation trap in the module
    docstring. Evaluated with ``ast.literal_eval`` rather than by importing, because
    importing most of these modules needs CuPy and this must run on a laptop; a
    module that spells one of the two as a name reference rather than a literal is
    reported as such rather than skipped.
    """
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else (node.targets[0]
                        if isinstance(node, ast.Assign) and len(node.targets) == 1
                        else None))
        if not isinstance(target, ast.Name) or target.id not in PARTITION_NAMES:
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except ValueError as error:  # a name reference, a call, an f-string
            raise AssertionError(
                f"{target.id} is not a literal and cannot be read without importing "
                f"the module, which needs CuPy: {error}") from None
    return found


def declared_kernels(source: str) -> set[str]:
    """The kernel entry points this module's own device text declares."""
    return set(KERNEL_DECLARATION.findall(source)) - PLACEHOLDERS


def device_text_is_complete(source: str) -> bool:
    """Whether every kernel this module emits is visible as a literal declaration.

    False for an emitter that interpolates the name (``void {name}(``), because then
    the strict scan under-reports and "a described name that is not shipped" would be
    a false positive rather than a drift.
    """
    strict = declared_kernels(source)
    loose = set(LOOSE_DECLARATION.findall(source)) - PLACEHOLDERS
    return bool(strict) and strict == loose


def survey() -> dict[str, dict]:
    """One pass over the package: sets, declarations and completeness per module."""
    out: dict[str, dict] = {}
    for path in family_modules():
        source = path.read_text(encoding="utf-8")
        out[path.name] = {
            "sets": partition_sets(source),
            "declared": declared_kernels(source),
            "complete": device_text_is_complete(source),
        }
    return out


def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def record_claims() -> dict[str, list[str]]:
    """``{kernel name: [block names that certify it]}``.

    The root of the record certifies ``step_curl_kernels.py``'s pair through a
    TOP-LEVEL ``certified_kernels`` list; every later family carries its own block
    with the same key. Both are read, because reading only the nested one reports the
    certified curl pair as unbacked and reading only the root reports every other
    family as unbacked.
    """
    entry = record()
    claims: dict[str, list[str]] = {}
    for name in entry.get("certified_kernels", ()):
        claims.setdefault(name, []).append("<root>")
    for block, value in entry.items():
        if not isinstance(value, dict):
            continue
        for name in value.get("certified_kernels", ()):
            claims.setdefault(name, []).append(block)
    return claims


# --------------------------------------------------------------------------
# Discovery, and the control that it can fail.
# --------------------------------------------------------------------------

def test_the_walk_is_finding_the_package():
    """A reader that has stopped matching passes every clause below vacuously."""
    found = survey()
    with_sets = {name for name, facts in found.items() if facts["sets"]}
    declared = set().union(*(facts["declared"] for facts in found.values()))
    described = set()
    for facts in found.values():
        for value in facts["sets"].values():
            described |= set(value)

    assert len(with_sets) >= MODULES_WITH_A_PARTITION_FLOOR, (
        f"only {len(with_sets)} modules carry a partition and the floor is "
        f"{MODULES_WITH_A_PARTITION_FLOOR}: {sorted(with_sets)}")
    assert len(declared) >= DECLARED_KERNELS_FLOOR, (
        f"the declaration scan found {len(declared)} kernels and the floor is "
        f"{DECLARED_KERNELS_FLOOR}; the spelling it matches has probably moved")
    assert len(described) >= DESCRIBED_KERNELS_FLOOR, (
        f"the two sets name {len(described)} kernels and the floor is "
        f"{DESCRIBED_KERNELS_FLOOR}")


def test_a_module_that_ships_a_kernel_and_declares_neither_set_is_caught():
    """The blindness control, and the failure this file was written for.

    A synthetic module with a kernel declaration and no partition is exactly the
    shape the eight modules had on 2026-08-27. It must come back as declaring a
    kernel and describing nothing, or the walk cannot see the thing it exists for.
    """
    synthetic = ('_code = r\'\'\'\n'
                 'extern "C" __global__ void synthetic_kernel(float* f) {}\n'
                 '\'\'\'\n')
    assert declared_kernels(synthetic) == {"synthetic_kernel"}
    assert partition_sets(synthetic) == {}
    assert device_text_is_complete(synthetic)


def test_the_reader_sees_an_annotated_assignment_too():
    """The trap the per-module readers fall into, controlled here.

    ``ast.Assign``-only readers miss ``UNCERTIFIED_KERNELS: Tuple[str, ...] = ()``,
    which two modules in this package really do spell that way. A walk blind to the
    annotated form would skip them and report a clean partition over a set it never
    looked at.
    """
    annotated = ("from typing import Tuple\n"
                 "CERTIFIED_KERNELS: Tuple[str, ...] = ('a',)\n"
                 "UNCERTIFIED_KERNELS: Tuple[str, ...] = ()\n")
    assert partition_sets(annotated) == {"CERTIFIED_KERNELS": ("a",),
                                         "UNCERTIFIED_KERNELS": ()}
    plain = "CERTIFIED_KERNELS = ('a',)\nUNCERTIFIED_KERNELS = {}\n"
    assert partition_sets(plain) == {"CERTIFIED_KERNELS": ("a",),
                                     "UNCERTIFIED_KERNELS": {}}


# --------------------------------------------------------------------------
# The partition itself.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", sorted(
    name for name, facts in survey().items() if facts["sets"]))
def test_each_module_partitions_the_kernels_it_declares(module):
    """Disjoint, exhaustive over what the module writes, and closed where it can be.

    Two directions, and only the second is conditional. A kernel written into the
    device text and named in neither set would ship, be wireable and be described by
    nothing -- that is asserted everywhere. A name in the sets that the device text
    does not declare is a drift, and it is asserted only where the text is complete;
    an emitter that interpolates its kernel name declares nothing this reader can
    see, and asserting the second direction there would fail on a correct file.
    """
    facts = survey()[module]
    sets = facts["sets"]
    assert set(sets) == set(PARTITION_NAMES), (
        f"{module} declares {sorted(sets)}; a module that carries one of the two "
        f"must carry both, or half the partition is unstated")

    certified = set(sets["CERTIFIED_KERNELS"])
    dead = set(sets["UNCERTIFIED_KERNELS"])
    assert all(isinstance(name, str) and name for name in certified | dead), module
    assert not certified & dead, (
        f"{module}: {sorted(certified & dead)} are in BOTH halves")

    declared = facts["declared"]
    assert declared <= certified | dead, {
        "module": module,
        "declared and described by neither": sorted(declared - certified - dead)}
    if facts["complete"]:
        assert certified | dead == declared, {
            "module": module,
            "described but not declared": sorted((certified | dead) - declared),
            "declared but not described": sorted(declared - certified - dead)}


def test_every_kernel_in_the_package_has_exactly_one_owning_module():
    """One family owns a name, and every declared name is owned by some family.

    The cross-module clause no per-module test can make. Two families describing one
    name would make ``certification.json`` ambiguous about which body a verdict saw,
    which is the reason ``folded_offdiag_kernels.py`` deliberately refuses to reuse
    ``offdiag_emitter.KERNEL_NAME``. The second half catches the opposite: an
    emitter's kernel whose owning family forgot to name it -- ``offdiag_emitter``
    writes ``update_E_pml_real_offdiag`` and ``offdiag_constitutive_kernels`` is the
    module that has to describe it.
    """
    found = survey()
    owner: dict[str, str] = {}
    collisions = []
    for module, facts in sorted(found.items()):
        for value in facts["sets"].values():
            for name in value:
                if name in owner:
                    collisions.append((name, owner[name], module))
                owner[name] = module
    assert not collisions, f"kernels described by two modules: {collisions}"

    declared = set().union(*(facts["declared"] for facts in found.values()))
    assert declared <= set(owner), {
        "declared in the package and described by no module":
            sorted(declared - set(owner))}


# --------------------------------------------------------------------------
# "Certified" must be a verdict, and "dead" must stay dead.
# --------------------------------------------------------------------------

def test_every_certified_kernel_is_claimed_by_a_record_block():
    """The clause that stops "certified" from being a word someone typed."""
    claims = record_claims()
    unbacked = {}
    for module, facts in sorted(survey().items()):
        for name in facts["sets"].get("CERTIFIED_KERNELS", ()):
            if name not in claims:
                unbacked[name] = module
    assert not unbacked, (
        f"{unbacked} are in CERTIFIED_KERNELS and no certification.json block names "
        f"them. A gate verdict lives in the record or it does not exist.")


def test_the_record_claims_no_kernel_that_no_module_certifies():
    """The orphan direction, and it is the one that has actually gone wrong here.

    Six Triton welds shipped as records nothing recomputed because the test named its
    gates by hand. A block claiming a kernel this package does not certify is the
    same failure wearing the other face: a verdict about a body that may no longer
    ship, read by anyone counting coverage.
    """
    described = set()
    for facts in survey().values():
        described |= set(facts["sets"].get("CERTIFIED_KERNELS", ()))
    orphans = {name: blocks for name, blocks in record_claims().items()
               if name not in described}
    assert not orphans, (
        f"certification.json certifies {sorted(orphans)}, which no module names in "
        f"CERTIFIED_KERNELS: {orphans}")


def test_no_uncertified_kernel_is_named_anywhere_in_the_record():
    """A name cannot be dead in the source and covered in the record.

    Deliberately a scan of the whole document rather than of ``certified_kernels``
    lists: a dead kernel mentioned in a block's prose reads as covered to anyone
    grepping the record, which is how a name gets wired.
    """
    text = json.dumps(record())
    named = {}
    for module, facts in sorted(survey().items()):
        for name in facts["sets"].get("UNCERTIFIED_KERNELS", ()):
            if f'"{name}"' in text:
                named[name] = module
    assert not named, (
        f"{named} have no gate verdict and certification.json names them")


def test_every_dead_kernel_names_the_gate_it_owes():
    """A dead name with no reason beside it is a kernel nobody can un-strand.

    The per-module tests spell the expected gate for their own family; this asks the
    weaker package-wide question -- that the value is a non-empty string pointing at
    a gate or probe under ``parity/`` -- so a family added without one is caught even
    before it has a test of its own.
    """
    for module, facts in sorted(survey().items()):
        dead = facts["sets"].get("UNCERTIFIED_KERNELS", {})
        if not isinstance(dead, dict):
            assert not dead, (
                f"{module} spells UNCERTIFIED_KERNELS as {type(dead).__name__} and "
                f"it is non-empty, so no name in it says which gate it owes")
            continue
        for name, owes in dead.items():
            assert isinstance(owes, str) and owes.strip(), (
                f"{module}: {name} carries {owes!r} instead of the gate it owes")
            assert "gate_" in owes or "probe_" in owes, (
                f"{module}: {name} owes {owes!r}, which names no gate or probe")

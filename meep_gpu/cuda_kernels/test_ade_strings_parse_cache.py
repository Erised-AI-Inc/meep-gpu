"""The ADE device-string reader: read every call, parsed once per distinct text.

``fused_polarization_pair._certified_ade_strings`` is asked twice per emission of
every fused polarization kernel on the CUDA track (``_verified_ade_bodies`` and
``_ade_prologue``, shared by ``three_slot_dispersive_weld`` and
``dispersive_offdiag_fused_polarization_pair``), and until 2026-09-19 each ask
re-ran ``ast.parse`` over ``ade_kernels.py``. The parse is now memoized on the
file's WHOLE TEXT. What this file pins is everything that memo could get silently
wrong:

1. **A hit is a copy.** Equal to the memo, never the memo object, so a caller that
   edits what it was handed cannot edit the next caller's strings -- and a hit
   does not parse.
2. **The key is the text.** A size-preserving edit (the stale-``.pyc`` shape) is
   parsed anew and yields different strings, and an edit reaching the reader
   reaches the certified-body check, which refuses it.
3. **A raise is not memoized.** A text missing a required name, or carrying one
   that no longer folds, raises on every call and leaves the memo as it was.
4. **The file is still read on every call.** Only the parse is skipped.

Device-free: nothing here imports CuPy or compiles.
"""

from __future__ import annotations

import ast
import builtins

import pytest

from . import fused_polarization_pair as family

#: The volume recurrence's store line, anchored on its leading newline so the
#: module docstring's deeper-indented copy of the same arithmetic does not match.
#: It appears once in each of the two kernel strings.
_STORE_LINE = ("\n    p_out[idx] = ((p * c_now) + (c_prev * q)) + "
               "(c_drive * (s * w));\n")
#: The same line regrouped -- the SAME LENGTH, so a size- or stat-keyed memo would
#: not see the edit.
_REGROUPED = ("\n    p_out[idx] = ((p * c_now) + (c_prev * q)) + "
              "((c_drive * s) * w);\n")


class _CountingAst:
    """The ``ast`` module with ``parse`` counted; every other name delegated."""

    def __init__(self) -> None:
        self.parses = 0

    def parse(self, *args, **kwargs):
        self.parses += 1
        return ast.parse(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(ast, name)


@pytest.fixture
def cold(monkeypatch):
    """An empty memo and a parse counter, both restored after the test."""
    counter = _CountingAst()
    monkeypatch.setattr(family, "_ADE_STRINGS_BY_TEXT", {})
    monkeypatch.setattr(family, "ast", counter)
    return counter


def _real_text() -> str:
    return family._read_ade_kernels_text()


def test_a_memo_hit_is_an_equal_dict_that_is_a_distinct_object(cold):
    text = _real_text()
    first = family._parse_ade_strings(text)
    second = family._parse_ade_strings(text)
    assert cold.parses == 1, "the second call on the same text parsed again"
    assert set(first) == set(family._ADE_STRING_NAMES)
    assert first == second
    assert first is not second
    stored = family._ADE_STRINGS_BY_TEXT[text]
    assert first is not stored and second is not stored

    # Mutating what a caller was handed changes nothing the next caller sees.
    name = "_update_P_pml_real_kernel_code"
    original = second[name]
    first[name] = "mutated"
    del first["_ADE_UPDATE_P_PROLOGUE"]
    third = family._parse_ade_strings(text)
    assert third[name] == original
    assert "_ADE_UPDATE_P_PROLOGUE" in third
    assert third == second
    assert cold.parses == 1


def test_a_different_text_of_the_same_length_is_parsed_anew(cold):
    text = _real_text()
    assert text.count(_STORE_LINE) == 2, (
        "ade_kernels.py no longer carries the volume and uniform store lines this "
        "test edits; re-anchor it")
    altered = text.replace(_STORE_LINE, _REGROUPED, 1)
    assert len(altered) == len(text) and altered != text

    real = family._parse_ade_strings(text)
    edited = family._parse_ade_strings(altered)
    assert cold.parses == 2
    assert len(family._ADE_STRINGS_BY_TEXT) == 2
    assert edited != real
    # Exactly the string the edit landed in moved; the other two did not.
    assert edited["_update_P_pml_real_kernel_code"] != real[
        "_update_P_pml_real_kernel_code"]
    assert "((c_drive * s) * w)" in edited["_update_P_pml_real_kernel_code"]
    for name in ("_ADE_UPDATE_P_PROLOGUE", "_update_P_pml_real_uniform_kernel_code"):
        assert edited[name] == real[name]
    # And the real text is still served from its own entry.
    assert family._parse_ade_strings(text) == real
    assert cold.parses == 2


def test_an_edit_reaching_the_reader_reaches_the_certified_body_check(cold,
                                                                      monkeypatch):
    """The path callers take: reader -> parse -> ``_verified_ade_bodies``. A
    regrouped recurrence read off the disk must be refused, not served from a memo
    filled by the unedited file a moment earlier."""
    text = _real_text()
    family._verified_ade_bodies()  # warm the memo on the real text
    altered = text.replace(_STORE_LINE, _REGROUPED, 1)
    monkeypatch.setattr(family, "_read_ade_kernels_text", lambda: altered)
    assert "((c_drive * s) * w)" in family._certified_ade_strings()[
        "_update_P_pml_real_kernel_code"]
    with pytest.raises(AssertionError, match="no longer carries the recurrence"):
        family._verified_ade_bodies()


def test_a_text_missing_a_required_name_raises_on_every_call(cold):
    text = _real_text()
    anchor = "\n_update_P_pml_real_uniform_kernel_code = "
    assert text.count(anchor) == 1
    missing = text.replace(anchor, "\n_update_P_pml_real_uniform_kernel_codX = ")
    for _ in range(2):
        with pytest.raises(AssertionError,
                           match=r"no longer assigns \['_update_P_pml_real_uniform"):
            family._parse_ade_strings(missing)
    assert cold.parses == 2, "a raise was memoized"
    assert family._ADE_STRINGS_BY_TEXT == {}


def test_a_text_that_no_longer_folds_raises_on_every_call(cold):
    text = _real_text()
    anchor = "\n_update_P_pml_real_kernel_code = _ADE_UPDATE_P_PROLOGUE + "
    assert text.count(anchor) == 1
    unfoldable = text.replace(
        anchor, "\n_update_P_pml_real_kernel_code = str(_ADE_UPDATE_P_PROLOGUE) + ")
    for _ in range(2):
        with pytest.raises(AssertionError, match="no longer a foldable string"):
            family._parse_ade_strings(unfoldable)
    assert cold.parses == 2
    assert family._ADE_STRINGS_BY_TEXT == {}


def test_the_reader_opens_ade_kernels_on_every_call(cold, monkeypatch):
    real_open = builtins.open
    opened = []

    def counting_open(file, *args, **kwargs):
        if str(file).endswith("ade_kernels.py"):
            opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    results = [family._certified_ade_strings() for _ in range(3)]
    assert len(opened) == 3, opened
    assert cold.parses == 1
    assert results[0] == results[1] == results[2]
    assert len({id(result) for result in results}) == 3


def test_a_cold_and_a_warm_memo_emit_the_same_kernel_text(cold):
    """Byte identity of the emitted source across the memo's two states, on both
    arms of the E->P weld at a mixed sigma specialization."""
    for arm in ("pml", "no_pml"):
        family._ADE_STRINGS_BY_TEXT.clear()
        cold_text = family.fused_polarization_pair_source(arm, 1, 3,
                                                          (True, False, True))
        warm_text = family.fused_polarization_pair_source(arm, 1, 3,
                                                          (True, False, True))
        assert cold_text == warm_text
    assert cold.parses == 2  # one per cleared memo, none on the warm emissions

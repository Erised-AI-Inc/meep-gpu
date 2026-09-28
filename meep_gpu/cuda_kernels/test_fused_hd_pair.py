"""Host tests for the CUDA H->D weld: the lift, the declarations and the composition.

WHAT CAN BE TESTED WITHOUT A DEVICE, AND WHAT CANNOT. The predicate, the declared edit
tables, the seam facts and the composer's refusal are all decidable here and their
failure mode is a SILENT WRONG ANSWER, so they are tested at the merge bar. The
EMITTER is not: it splices the certified device strings, which live in modules that
import CuPy at scope, so :func:`~.fused_hd_pair.kernel_source` raises by name on a host
without one and the emitter tests declare themselves skipped rather than passing
quietly. Those are what ``parity/meep_gpu/gate_cuda_fused_hd_pair.py``'s
``transcription`` leg measures on the validation host.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from .. import withdraw_hoist
from . import fused_hd_pair as family
from . import fused_pairs

_API_ROOT = Path(__file__).resolve().parents[2]
_SEAM_RECORD = (_API_ROOT / "parity" / "meep_gpu" / "results"
                / "h_to_d_seam_2026-09-04" / "h_to_d_seam.jsonl")

#: The cell this product serves and the extra arm's, in the census's own spelling.
_CELL = ("cuda_constitutive/ordinary", "cuda_curl/PML")
_EXTRA_CELL = ("cuda_nonlinear/nonlinear", "cuda_curl/PML")

#: THE CERTIFIED DEVICE TEXT IS A GENUINELY ABSENT RESOURCE ON A HOST WITHOUT CuPy,
#: and it is declared as one rather than skipped quietly: both certified modules import
#: CuPy at scope, so on a laptop there is no text to splice and the emitter's own
#: guard raises by name. On the validation host these run, and the gate's
#: ``transcription`` leg measures the same properties over the whole emitted string.
_HAS_CERTIFIED_TEXT = not (family.step_curl_kernels is None
                           or family.constitutive_kernels is None)


def _needs_certified_text() -> None:
    if not _HAS_CERTIFIED_TEXT:
        pytest.skip("[requires_resource][cuda-certified-device-text] "
                    "step_curl_kernels / constitutive_kernels import CuPy at module "
                    "scope, so the certified strings this weld splices are not "
                    "loadable here")


_emitter = pytest.mark.requires_resource("cuda-certified-device-text")


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_seam_is_the_withdraw_hoists_and_not_the_deposit_repairs():
    """The seam holds a WITHDRAWAL, not an injection, and the flags say so.

    Getting this backwards is not a crash: ``deposit_repair._in_seam_indexed``
    classifies anything that is not ``"B"`` as the ELECTRIC injection list, so a
    product that named a field letter here would have its launch bracketed with a
    repair for a deposit that is not in its seam.
    """
    assert family.SEAM == withdraw_hoist.SEAM
    assert family.REPLACES == withdraw_hoist.SEAM_SPAN
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert fused_pairs.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)


def test_the_span_is_the_one_the_hoist_was_measured_for():
    """``withdraw_hoist`` accepts this product's span and refuses a wider or narrower one."""
    assert withdraw_hoist._span_reasons(family.REPLACES) == ()  # noqa: SLF001
    assert withdraw_hoist._span_reasons(("update_H",))  # noqa: SLF001
    assert withdraw_hoist._span_reasons(  # noqa: SLF001
        ("step_B", "update_H", "step_D"))


def test_the_product_declares_itself_uninstallable_with_a_measured_reason():
    """``INSTALLABLE`` False, and the composer reports the reason on every run."""
    assert family.INSTALLABLE is False
    reason = fused_pairs._declared_uninstallable(  # noqa: SLF001
        family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY])
    assert reason is not None
    assert family.FAMILY in reason
    # The reason must be a MEASUREMENT about the composition, not a preference.
    for phrase in ("4 - (installed pairs)", "TIE", "LOSS", "four-slot"):
        assert phrase in reason, phrase


def test_the_release_licence_disclaims_execution():
    """A released product on this backend executes NOWHERE, and the module says so."""
    text = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    assert "PREDICATE ADMISSION" in text
    assert "INSTALLABLE = False" in text
    assert "no timing" in text.lower()


def test_the_kernel_is_in_exactly_one_certification_set():
    """CERTIFIED, and claimed by a record block.

    ``test_kernel_partition.py`` reads both names off the syntax tree and refuses a
    certified name no block claims; this asserts the partition from the other side, so
    a name that moved without its record is a failure here too.
    """
    assert family.KERNEL_NAME in family.CERTIFIED_KERNELS
    assert family.KERNEL_NAME not in family.UNCERTIFIED_KERNELS
    assert not family.UNCERTIFIED_KERNELS


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------

def test_the_composer_carries_the_absorb_rows_this_product_needs():
    """A product with no absorb declaration is refused BY NAME rather than installed."""
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == ("ordinary", "PML")
    assert fused_pairs.FUSED_PAIR_EXTRA_ARMS[family.FAMILY] == (("nonlinear", "PML"),)
    assert fused_pairs.FUSED_PRODUCTS[family.FAMILY]["curl_slot"] == family.SLOT
    assert fused_pairs.span_of(
        family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY]) == family.REPLACES


def test_the_h_to_d_seam_is_decided_after_the_seam_that_holds_update_H():
    """The seam ORDER is what leaves the released B->H pair its slot.

    ``install_fused_pairs`` walks ``FUSED_PAIR_SEAMS`` in insertion order and
    ``_spans_may_absorb`` reads the LIVE ``selected``, so a seam decided after
    ``step_B`` sees ``update_H`` already taken. Asserted rather than assumed: a
    reordered table would silently hand this product the released pair's slot on every
    row, which is the displacement the arbitration guard exists to prevent.
    """
    order = list(fused_pairs.FUSED_PAIR_SEAMS)
    assert order.index("step_B") < order.index("update_H")
    assert order.index("step_D") < order.index("update_H")


def test_the_update_H_seam_candidates_are_partitioned_by_the_cells_they_absorb():
    """SEVEN candidates since the H->D tails wired, and the ambiguity rule still
    cannot fire -- on four axes rather than one.

    THE GRID SPLITS THE SEVEN: the five Cartesian products refuse a Dcyl grid through
    their certified curls' own clauses; both cylindrical products REQUIRE one. THE
    STORAGE SPLITS EACH GROUP: this product's halves refuse complex64 by name and the
    complex one's require it, the two cylindrical products partition the same way, and
    the two beta products split from each other on the same boolean. BETA splits the
    beta pair from every other candidate (each of the others refuses ``grid.beta != 0``
    by name), and CONDUCTIVITY / ``grid.bfast_active`` split the conductive-BFAST
    product's two variants from the rest and from each other. So no configuration
    reaches two candidates.

    ASSERTED FROM THE ABSORB TABLE, not from this docstring, and as the PARTITION
    PROPERTY rather than a transcribed list: every (constitutive arm, curl arm) cell any
    of these products may absorb -- the primary ``FUSED_PAIR_ARMS`` row plus every
    ``FUSED_PAIR_EXTRA_ARMS`` row -- is required to be distinct across the whole seam, so
    an eighth product landing here with a label pair that collides is a failure in this
    test rather than a silent double admission at install time. The membership list is
    pinned beside it because a product silently LEAVING the seam is the other direction
    this guards.
    """
    candidates = sorted(name for name, product in fused_pairs.FUSED_PRODUCTS.items()
                        if product["curl_slot"] == "update_H")
    assert candidates == sorted([family.FAMILY, "cuda_complex_fused_hd_pair",
                                 "cuda_cylindrical_real_fused_hd_pair",
                                 "cuda_cylindrical_fused_hd_pair",
                                 "cuda_special_kz_fused_hd_pair",
                                 "cuda_complex_beta_fused_hd_pair",
                                 "cuda_conductive_bfast_fused_hd_pair"])
    assert {name: fused_pairs.FUSED_PAIR_ARMS[name] for name in candidates} == {
        "cuda_fused_hd_pair": ("ordinary", "PML"),
        "cuda_complex_fused_hd_pair": ("complex", "complex"),
        "cuda_cylindrical_real_fused_hd_pair": ("ordinary", "cylindrical"),
        "cuda_cylindrical_fused_hd_pair": ("complex", "cylindrical complex"),
        "cuda_special_kz_fused_hd_pair": ("real beta", "real beta"),
        "cuda_complex_beta_fused_hd_pair": ("complex beta", "complex beta"),
        "cuda_conductive_bfast_fused_hd_pair": ("ordinary", "conductive"),
    }
    # THE PROPERTY, over every cell rather than only the primary rows: two of these
    # products carry a second cell (``cuda_fused_hd_pair``'s nonlinear one and the
    # conductive product's BFAST one), and a collision there admits two candidates just
    # as a collision between primary rows would.
    cells = []
    for name in candidates:
        cells.append(fused_pairs.FUSED_PAIR_ARMS[name])
        cells.extend(fused_pairs.FUSED_PAIR_EXTRA_ARMS.get(name, ()))
    assert len(cells) == len(set(cells)), sorted(
        cell for cell in set(cells) if cells.count(cell) > 1)


def test_the_installer_routes_this_seam_to_the_withdraw_hoist():
    """The seam name reaches ``_install_fused_pair``'s hoist branch, read off the source.

    Not driven (building a plan needs a device); read from the installer's own text, so
    a branch that stopped keying on ``withdraw_hoist.SEAM`` fails here.
    """
    source = textwrap.dedent(inspect.getsource(fused_pairs._install_fused_pair))  # noqa: SLF001
    tree = ast.parse(source)
    compared = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    assert any(
        isinstance(node.left, ast.Name) and node.left.id == "pair_name"
        and any(isinstance(c, ast.Attribute) and c.attr == "SEAM" for c in node.comparators)
        for node in compared), source[:400]
    assert "LeadingWithdrawPlan" in source


# ---------------------------------------------------------------------------
# The cell, read from the artifact rather than transcribed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SEAM_RECORD.is_file(),
                    reason=f"{_SEAM_RECORD} is not in this checkout")
def test_the_cell_this_product_claims_is_the_one_the_seam_record_measured():
    """127 rows in the primary cell, 2 in the nonlinear one, 5 refused by the withdraw.

    READ FROM THE ARTIFACT, never transcribed: the module docstring quotes these
    numbers and a record that moved under them would leave the prose describing a
    different cell.
    """
    rows = [json.loads(line) for line in
            _SEAM_RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    primary = [row for row in rows
               if (row["arms"]["cuda"]["update_H"],
                   row["arms"]["cuda"]["step_D"]) == _CELL]
    extra = [row for row in rows
             if (row["arms"]["cuda"]["update_H"],
                 row["arms"]["cuda"]["step_D"]) == _EXTRA_CELL]
    withdraws = [row for row in primary
                 if row["h_to_d_seam"]["withdraw_in_seam"]]
    assert len(primary) == 127, len(primary)
    assert len(extra) == 2, len(extra)
    assert len(withdraws) == 5, sorted(row["label"] for row in withdraws)
    assert len(primary) - len(withdraws) == 122
    for text in ("127 rows", "122", "5", "2 rows"):
        assert text in family.__doc__


@pytest.mark.skipif(not _SEAM_RECORD.is_file(),
                    reason=f"{_SEAM_RECORD} is not in this checkout")
def test_the_nonlinear_widening_is_the_same_kernel_and_not_a_second_one():
    """Both cells record the SAME ``update_H`` kernel, which is what the extra arm rests on.

    On this backend the widening is a MODULE IDENTITY rather than a text comparison:
    ``nonlinear_constitutive`` ships no ``update_H`` kernel at all. This asserts the
    census's own agreement from the other side, and asserts the module's declaration
    directly, so a future nonlinear ``update_H`` body would fail here rather than being
    silently absorbed by the extra arm.
    """
    from . import nonlinear_constitutive  # noqa: PLC0415

    rows = [json.loads(line) for line in
            _SEAM_RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    kernels = {(row["arms"]["cuda"]["update_H"], row["arms"]["cuda"]["step_D"]):
               row["arms"]["cuda"].get("update_H_kernel")
               for row in rows
               if (row["arms"]["cuda"]["update_H"],
                   row["arms"]["cuda"]["step_D"]) in (_CELL, _EXTRA_CELL)}
    assert kernels[_CELL] == kernels[_EXTRA_CELL] == "fused_update_H_pml_real"
    # THE MODULE'S OWN SIDE OF IT: the nonlinear family certifies exactly one kernel
    # and it is the E one.
    assert nonlinear_constitutive.CERTIFIED_KERNELS == ("update_E_pml_real_nonlinear",)
    source = Path(nonlinear_constitutive.__file__).read_text(encoding="utf-8")
    assert "_update_H_pml_real_nonlinear_kernel_code" not in source


# ---------------------------------------------------------------------------
# The declared edit tables
# ---------------------------------------------------------------------------

def test_every_declared_edit_carries_a_reason():
    """DATA, not prose: a table entry with no ``why`` is an undocumented edit."""
    for table in (family.CONSTITUTIVE_LIFT_EDITS, family.CURL_LIFT_EDITS):
        assert table
        for entry in table:
            assert set(entry) == {"line", "became", "why"}, entry
            assert entry["why"].strip()


def test_the_tap_counts_are_declared_and_not_inferred():
    assert family.HALO_TAPS == 6
    assert family.OWN_LOAD_EDITS == 6


# ---------------------------------------------------------------------------
# The emitter (device host only)
# ---------------------------------------------------------------------------

@_emitter
def test_the_curl_prelude_is_carried_verbatim():
    _needs_certified_text()
    from . import step_curl_kernels  # noqa: PLC0415

    assert family.curl_prelude() == step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001


@_emitter
def test_the_emitted_source_is_pure_ascii_and_encodes():
    _needs_certified_text()
    source = family.kernel_source()
    assert source.isascii()
    source.encode("ascii")


@_emitter
def test_no_stale_magnetic_read_survives_below_the_weld():
    _needs_certified_text()
    source = family.kernel_source()
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    for name in family.H_TARGETS:
        assert f"{name}[" not in below, name
    assert "shift_dn(" not in below


@_emitter
def test_pml_apply_is_defined_exactly_once():
    """The certified curl prelude's helper must not arrive twice.

    ``shift_dn_recompute_source`` lifts the certified ``shift_dn`` out of the prelude
    and everything past its closing brace is ``pml_apply``, which
    :func:`~.fused_hd_pair.curl_prelude` already emits. A second copy is a redefinition
    NVRTC refuses; this pins the drop so a prelude edit cannot smuggle one back.
    """
    _needs_certified_text()
    source = family.kernel_source()
    assert source.count("__device__ __forceinline__ void pml_apply(") == 1


@_emitter
def test_a_changed_index_expression_raises_rather_than_redirecting_a_tap():
    """The parse discipline, driven: plant a different spelling and require a failure."""
    _needs_certified_text()
    from . import step_curl_kernels  # noqa: PLC0415

    saved = step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001
    try:
        step_curl_kernels._REAL_PML_PRELUDE = saved.replace(  # noqa: SLF001
            "return g[idx - stride];", "return g[idx - 1 * stride];", 1)
        emitted = family.shift_dn_recompute_source()
        # The parse CARRIES the new expression rather than the old one -- that is the
        # point of parsing it -- so the tap follows the emitter instead of a table.
        assert "resolve_H(comp, idx - 1 * stride, weld)" in emitted
        step_curl_kernels._REAL_PML_PRELUDE = saved.replace(  # noqa: SLF001
            "    if (ia > 0) return g[idx - stride];\n", "", 1)
        with pytest.raises(AssertionError):
            family.shift_dn_recompute_source()
    finally:
        step_curl_kernels._REAL_PML_PRELUDE = saved  # noqa: SLF001


@_emitter
def test_a_lost_magnetic_tap_raises_rather_than_emitting_a_partial_redirect():
    """A certified curl that stopped making six shifted reads is a named failure."""
    _needs_certified_text()
    from . import step_curl_kernels  # noqa: PLC0415

    saved = step_curl_kernels._step_D_pml_real_kernel_code  # noqa: SLF001
    try:
        step_curl_kernels._step_D_pml_real_kernel_code = saved.replace(  # noqa: SLF001
            "        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);\n",
            "        float sf = 0.0f;\n", 1)
        with pytest.raises(AssertionError):
            family.welded_curl_tail()
    finally:
        step_curl_kernels._step_D_pml_real_kernel_code = saved  # noqa: SLF001

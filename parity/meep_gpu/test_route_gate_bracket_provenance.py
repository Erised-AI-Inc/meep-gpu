"""Both driver-route gates record the deposit-repair bracket's bytes, off-device.

WHY THIS FILE EXISTS. The byte-parity evidence behind ``served_in_dispatch`` on a
bracketed seam is taken with ``deposit_repair.LeadingRepairPlan`` /
``TrailingRepairPlan`` wrapped around the fused launch, and until 2026-09-19 neither
route gate named ``meep_gpu/deposit_repair.py`` in the curated
``provenance.source_sha256`` block that ``recut_driver_dispatch_record.py`` compares.
Read that day off the three 2026-09-17 ``allpaths`` campaigns: all 14 legs carried it
only in the top-level ``imported_source_sha256`` (``c1d55704``), and the curated block
(39 keys per fused-route leg, 11 per Metal leg) did not. This pins the curated block,
host-only, so the bracket cannot quietly leave it again.

WHAT A PASS DOES NOT LICENSE. It says the block RECORDS the bracket. No
``driver_dispatch`` record binds it: the recut binds the key set of the record it cuts
(``recut_driver_dispatch_record.py:310``), so whether a record should is the record
release decision, and binding it needs a campaign run with this block.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_fused_route as fused_route  # noqa: E402
import gate_dispatch_metal_route as metal_route  # noqa: E402
import recut_driver_dispatch_record as recut  # noqa: E402

PACKAGE = HERE.parent.parent / "meep_gpu"


def _live(name: str) -> str:
    return hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()


@pytest.fixture
def no_device(monkeypatch):
    """``e2e._provenance`` touches no GPU: CuPy unimportable, ``nvidia-smi`` not run.

    Both are already inside ``try`` in that function, so this changes only what the
    device fields SAY, never the digest map under test.
    """
    monkeypatch.setitem(sys.modules, "cupy", None)

    def _refuse(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi withheld by the test")

    monkeypatch.setattr(e2e, "subprocess", types.SimpleNamespace(run=_refuse))


def test_the_fused_route_block_records_the_bracket_at_the_live_digest(no_device):
    record = fused_route._provenance(0)  # noqa: SLF001
    digests = record["source_sha256"]
    assert digests.get("deposit_repair.py") == _live("deposit_repair.py"), (
        "the fused-route gate's curated block does not carry the live "
        "deposit_repair.py digest")
    # ONE SPELLING PER MAP: the package-relative key, never the repo-relative one.
    assert "meep_gpu/deposit_repair.py" not in digests
    # The composer the bracket installs through, digested by the end-to-end list.
    assert digests.get("cuda_kernels/fused_pairs.py") == _live(
        "cuda_kernels/fused_pairs.py")


def test_the_fused_route_block_adds_only_the_bracket_to_the_end_to_end_list(no_device):
    base = e2e._provenance(0)["source_sha256"]  # noqa: SLF001
    extended = fused_route._provenance(0)["source_sha256"]  # noqa: SLF001
    assert set(extended) - set(base) <= set(fused_route.BRACKET_SOURCES), (
        sorted(set(extended) - set(base)))
    assert set(base) <= set(extended), sorted(set(base) - set(extended))


def test_a_disagreeing_digest_for_a_bracket_file_refuses(monkeypatch):
    """A file rewritten between the two reads is a refusal, not a silent pick."""
    stale = {"source_sha256": {"deposit_repair.py": "0" * 64}}
    monkeypatch.setattr(e2e, "_provenance", lambda _gpu_id: stale)
    with pytest.raises(RuntimeError, match="deposit_repair.py"):
        fused_route._provenance(0)  # noqa: SLF001


def test_the_metal_route_block_records_the_bracket_at_the_live_digest():
    digests = metal_route._provenance()["source_sha256"]  # noqa: SLF001
    assert digests.get("deposit_repair.py") == _live("deposit_repair.py"), (
        "the Metal route gate's curated block does not carry the live "
        "deposit_repair.py digest")
    assert "meep_gpu/deposit_repair.py" not in digests


@pytest.mark.parametrize("backend", sorted(recut.BACKENDS))
def test_a_record_binding_the_bracket_would_find_the_recorded_key(backend):
    """Either spelling a ledger might bind resolves to the key the gates record."""
    for bound in ("meep_gpu/deposit_repair.py", "deposit_repair.py"):
        assert recut._leg_key(backend, bound) == "deposit_repair.py"  # noqa: SLF001

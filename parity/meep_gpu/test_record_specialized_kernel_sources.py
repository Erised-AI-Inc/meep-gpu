"""``record_specialized_kernel_sources.py`` derives an index digest only from welds that ran it.

The rule: a module's ``specialized_kernel_sources`` digest moves to the module's live
sha256 only when its standalone device weld AND its composition weld each have a live
per-capability record and each pin those live bytes. These tests drive the tool on a
scratch tree with a synthetic ledger, so each half of the rule is exercised on its own
and every refusal also checks that the ledger's bytes did not move. The last test
reads the shipped tree.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import record_specialized_kernel_sources as w2  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

OLD = "0" * 64
#: Two modules of the table, the second with its kernel indented under a guard as the
#: shipped conductivity.py and cylindrical_triton.py define theirs.
MODULES = {
    "no_pml.py": ("plain_curl_step", "def plain_curl_step(\n    x,\n):\n    return x\n"),
    "conductivity.py": ("conductive_pml_curl_step",
                        "if True:\n    def conductive_pml_curl_step(x):\n        return x\n"),
}


def bound(entry: dict) -> dict:
    fastpath.bind_capability(entry, bound_before=None, capability="8.6",
                             run={"host": "a test run", "records": "runs/",
                                  "artifact_sha256": "c" * 64})
    return entry


def make_tree(tmp_path, *, edit=None, indent=2) -> pathlib.Path:
    """A tree whose welds certify each module's live bytes; ``edit(ledger, live)`` breaks one."""
    root = tmp_path / "tree"
    kernels = root / "meep_gpu" / "triton_kernels"
    kernels.mkdir(parents=True)
    (root / "parity" / "meep_gpu").mkdir(parents=True)
    ledger = {w2.INDEX_KEY: {}}
    live = {}
    for module, (kernel, source) in MODULES.items():
        (kernels / module).write_text(source, encoding="utf-8")
        live[module] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        standalone, composition = w2.WELDS_OF_MODULE[module]
        ledger[standalone] = {"source_sha256": {
            f"meep_gpu/triton_kernels/{module}": live[module],
            f"parity/meep_gpu/gate_{module}": "e" * 64}}
        ledger[composition] = {"probe": f"parity/meep_gpu/composition_{module}",
                               "probe_sha256": "f" * 64,
                               "source_sha256": {module: live[module], "launch.py": "1" * 64}}
        ledger[w2.INDEX_KEY][module] = {"kernel": kernel, "sha256": OLD}
    if edit is not None:
        edit(ledger, live)
    for key, entry in ledger.items():
        if key != w2.INDEX_KEY and "runs" not in entry:
            bound(entry)
    text = json.dumps(ledger, indent=indent, sort_keys=True) + "\n"
    (root / w2.LEDGER_RELATIVE).write_text(text, encoding="utf-8")
    return root


def ledger_bytes(root: pathlib.Path) -> bytes:
    return (root / w2.LEDGER_RELATIVE).read_bytes()


def live_of(root: pathlib.Path, module: str) -> str:
    return hashlib.sha256((root / w2.MODULE_DIR / module).read_bytes()).hexdigest()


def test_the_report_derives_the_digest_and_writes_nothing(tmp_path):
    root = make_tree(tmp_path)
    before = ledger_bytes(root)
    report = w2.record(root, modules=["no_pml.py"])
    (verdict,) = report["modules"]
    assert verdict["before"] == OLD and verdict["after"] == live_of(root, "no_pml.py")
    assert verdict["live_capabilities"] == {"triton_no_pml_device_gate": ["8.6"],
                                            "no_pml_composition_gate": ["8.6"]}
    assert report["changed"] == ["specialized_kernel_sources.no_pml.py.sha256"]
    assert report["written"] is False and ledger_bytes(root) == before


def test_the_write_moves_only_the_requested_leaf_and_leaves_every_record_live(tmp_path):
    root = make_tree(tmp_path)
    before = json.loads(ledger_bytes(root))
    report = w2.record(root, modules=["no_pml.py"], write=True)
    after = json.loads(ledger_bytes(root))
    assert report["written"] is True
    assert after[w2.INDEX_KEY]["no_pml.py"] == {"kernel": "plain_curl_step",
                                                "sha256": live_of(root, "no_pml.py")}
    assert after[w2.INDEX_KEY]["conductivity.py"]["sha256"] == OLD
    assert w2.changed_leaves(before, after) == [(w2.INDEX_KEY, "no_pml.py", "sha256")]
    for key, entry in after.items():
        if key != w2.INDEX_KEY:
            assert fastpath.live_capabilities(entry) == ("8.6",), key
    assert ledger_bytes(root).decode("utf-8") == w2.canonical(after)


def test_a_current_digest_is_not_rewritten(tmp_path):
    root = make_tree(tmp_path)
    w2.record(root, modules=["no_pml.py"], write=True)
    before = ledger_bytes(root)
    report = w2.record(root, modules=["no_pml.py"], write=True)
    assert report["changed"] == [] and report["written"] is False
    assert ledger_bytes(root) == before


def pin_old(key, spelling):
    def edit(ledger, live):
        ledger[key]["source_sha256"][spelling] = OLD
    return edit


def unbound(key):
    """The weld's bytes moved after its run: the run record no longer binds them."""
    def edit(ledger, live):
        bound(ledger[key])
        ledger[key]["source_sha256"]["parity/meep_gpu/moved_after_the_run.py"] = "9" * 64
    return edit


def both_spellings(key, second):
    def edit(ledger, live):
        ledger[key]["source_sha256"]["no_pml.py"] = second(live["no_pml.py"])
    return edit


def drop(key):
    def edit(ledger, live):
        del ledger[key]
    return edit


def unpinned(key, spelling):
    def edit(ledger, live):
        del ledger[key]["source_sha256"][spelling]
    return edit


def index_edit(change):
    def edit(ledger, live):
        change(ledger[w2.INDEX_KEY])
    return edit


@pytest.mark.parametrize("edit, named", [
    (pin_old("triton_no_pml_device_gate", "meep_gpu/triton_kernels/no_pml.py"),
     "standalone weld triton_no_pml_device_gate pins meep_gpu/triton_kernels/no_pml.py"),
    (pin_old("no_pml_composition_gate", "no_pml.py"),
     "composition weld no_pml_composition_gate pins no_pml.py"),
    (unbound("triton_no_pml_device_gate"), "standalone weld .* has no live"),
    (unbound("no_pml_composition_gate"), "composition weld .* has no live"),
    (unpinned("no_pml_composition_gate", "no_pml.py"),
     "composition weld no_pml_composition_gate does not pin no_pml.py"),
    (both_spellings("triton_no_pml_device_gate", lambda live: OLD),
     "pins no_pml.py at 000000000000"),
    (drop("triton_no_pml_device_gate"), "standalone weld .* is not in the ledger"),
    (index_edit(lambda index: index["no_pml.py"].update(kernel="renamed_kernel")),
     "no longer defines the indexed kernel 'renamed_kernel'"),
    (index_edit(lambda index: index["no_pml.py"].update(note="a field")),
     "does not reshape it"),
    (index_edit(lambda index: index.pop("no_pml.py")), "this tool does not add one"),
], ids=["standalone_pins_old_bytes", "composition_pins_old_bytes", "standalone_not_live",
        "composition_not_live", "composition_does_not_pin", "spellings_disagree",
        "standalone_absent", "kernel_not_defined", "index_entry_reshaped", "index_entry_absent"])
def test_a_module_its_welds_do_not_certify_is_refused(tmp_path, edit, named):
    root = make_tree(tmp_path, edit=edit)
    before = ledger_bytes(root)
    with pytest.raises(w2.Refusal, match=named):
        w2.record(root, modules=["no_pml.py"], write=True)
    assert ledger_bytes(root) == before


def test_both_spellings_agreeing_is_accepted(tmp_path):
    root = make_tree(tmp_path, edit=both_spellings("triton_no_pml_device_gate",
                                                   lambda live: live))
    assert w2.record(root, modules=["no_pml.py"], write=True)["written"] is True


def test_a_request_is_all_or_nothing(tmp_path):
    root = make_tree(tmp_path, edit=pin_old("triton_conductivity_device_gate",
                                            "meep_gpu/triton_kernels/conductivity.py"))
    before = ledger_bytes(root)
    with pytest.raises(w2.Refusal, match="conductivity.py: the standalone weld"):
        w2.record(root, modules=["no_pml.py", "conductivity.py"], write=True)
    assert ledger_bytes(root) == before
    assert w2.record(root, modules=["no_pml.py"], write=True)["written"] is True


def test_a_module_without_a_row_is_refused_by_name(tmp_path):
    root = make_tree(tmp_path)
    before = ledger_bytes(root)
    with pytest.raises(w2.Refusal, match="dispersive_update_e.py has no row"):
        w2.record(root, modules=["dispersive_update_e.py"], write=True)
    assert ledger_bytes(root) == before


def test_a_ledger_in_another_serialisation_is_refused(tmp_path):
    root = make_tree(tmp_path, indent=1)
    before = ledger_bytes(root)
    with pytest.raises(w2.Refusal, match="does not round-trip"):
        w2.record(root, modules=["no_pml.py"], write=True)
    assert ledger_bytes(root) == before


@pytest.mark.parametrize("module", sorted(w2.WELDS_OF_MODULE))
def test_every_row_names_welds_and_an_index_entry_the_shipped_ledger_has(module):
    """Read only, and independent of drift: the row's keys exist and the welds pin the module."""
    ledger = json.loads((w2._API / w2.LEDGER_RELATIVE).read_text(encoding="utf-8"))
    path = w2._API / w2.MODULE_DIR / module
    assert path.is_file()
    item = ledger[w2.INDEX_KEY][module]
    assert set(item) == set(w2.INDEX_FIELDS)
    assert f"def {item['kernel']}(" in path.read_text(encoding="utf-8")
    for key in w2.WELDS_OF_MODULE[module]:
        assert w2.module_pins(ledger[key], module), f"{key} does not pin {module}"

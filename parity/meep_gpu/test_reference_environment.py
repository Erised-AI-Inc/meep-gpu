"""The certification reference locks and the drift report that reads them.

``environments/locks/`` holds, per platform, an explicit conda lock and a hashed pip
requirements file; the reference build scripts install exactly those files, and
``tools/compare_reference_environment.py`` (also step 7 of ``tools/check_install.py``)
tells a user whether an environment is the reference, package by package. These tests
are pure: no MEEP, no GPU, no conda, no network.

* the committed lock files: every line is a comment, ``@EXPLICIT`` or an https URL with
  an MD5 (conda), or a comment, an option or ``name==version --hash=sha256:...`` (pip);
  no local path or e-mail address in any file of the directory, nor a home-directory
  path or e-mail address in the build scripts, the tools and the documents that
  describe them; the numerics set names only packages its lock holds;
* lock parsing, including what it refuses;
* the comparison over synthetic environments: same, different, missing, another file
  of the same build, the documented hdf5 exception (that build and no other), a pip-only
  install, a PyPI package replaced by a conda build, a pip install over a conda build,
  a wheel's build tag, the host floor, the marker of a solved build;
* MEEP judged against the build the reference names (version, precision, MPI);
* the verdict line, and ``--require-reference`` as an exit status, through the tool's
  command line on a synthetic prefix and through check_install's step 7, which also
  fails on a MEEP that is not the reference's;
* the build scripts install the lock files that exist and assert the lock before they
  configure MEEP, and their step 1b, run against a stand-in conda, accepts the lock and
  refuses one changed line or a conda that fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "environments" / "locks"
TOOL_PATH = ROOT / "tools" / "compare_reference_environment.py"
CHECK_PATH = ROOT / "tools" / "check_install.py"
SCRIPTS = {
    "osx-arm64": ROOT / "parity" / "meep_gpu" / "build_meep_133_macos.sh",
    "linux-64": ROOT / "parity" / "meep_gpu" / "build_meep_133_linux.sh",
}


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load("reference_environment_under_test", TOOL_PATH)
check_install = _load("check_install_under_test", CHECK_PATH)

NUMERICS = json.loads((LOCK_DIR / tool.NUMERICS_FILE).read_text(encoding="utf-8"))
PLATFORMS = sorted(NUMERICS["platforms"])

# Host and account names are checked before a commit by a search outside this
# repository: a list of them here, even as digests, would disclose them.
#: Home-directory and scratch prefixes, spelled in pieces so this file carries none.
LOCAL_PATH = re.compile("|".join(
    [re.escape("/" + part + "/") for part in ("Users", "home", "private", "tmp")]
    + [re.escape("/var/" + "folders/"), r"[A-Za-z]:\\"]))
#: The same without the scratch prefixes, for files that name a default build
#: directory under ``/tmp``.
HOME_PATH = re.compile("|".join(
    [re.escape("/" + part + "/") for part in ("Users", "home")]
    + [re.escape("/private/" + part + "/") for part in ("var", "tmp")]
    + [re.escape("/var/" + "folders/"), r"[A-Za-z]:\\Users"]))
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
CONDA_URL = re.compile(r"^https://conda\.anaconda\.org/conda-forge/(noarch|osx-arm64|linux-64)/"
                       r"[A-Za-z0-9_.+-]+\.(conda|tar\.bz2)#[0-9a-f]{32}$")
PIP_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9._+!-]+ --hash=sha256:[0-9a-f]{64}$")
PIP_OPTIONS = {"--index-url https://pypi.org/simple", "--only-binary :all:", "--require-hashes"}


# --------------------------------------------------------------------------
# The committed files
# --------------------------------------------------------------------------

def test_every_file_in_the_lock_directory_is_free_of_private_details():
    files = sorted(path for path in LOCK_DIR.iterdir() if path.is_file())
    assert len(files) >= 6, [path.name for path in files]
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in (LOCAL_PATH, EMAIL):
            for match in pattern.finditer(text):
                findings.append(f"{path.name}: {match.group(0)!r}")
    assert not findings, findings


#: The files around the locks that a user reads or runs.
DESCRIBING_FILES = [
    "parity/meep_gpu/build_meep_133_macos.sh", "parity/meep_gpu/build_meep_133_linux.sh",
    "parity/meep_gpu/build_meep_133_native.sh", "tools/compare_reference_environment.py",
    "tools/check_install.py", "parity/meep_gpu/test_reference_environment.py",
    "INSTALL.md", "CHANGELOG.md", "AGENTS.md", "parity/meep_gpu/README.md",
    "docs/development/certification.md", "docs/getting-started/installation.md",
    ".gitattributes",
]


@pytest.mark.parametrize("relative", DESCRIBING_FILES)
def test_the_scripts_tools_and_documents_carry_no_home_path_or_e_mail(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    findings = [match.group(0) for pattern in (HOME_PATH, EMAIL)
                for match in pattern.finditer(text)]
    assert not findings, findings


def test_the_lock_files_are_checked_out_with_unix_line_endings():
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "environments/locks/* text eol=lf" in rules


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_the_conda_lock_is_comments_explicit_and_hashed_urls_only(platform_tag):
    spec = NUMERICS["platforms"][platform_tag]
    lines = (LOCK_DIR / spec["conda_lock"]).read_text(encoding="utf-8").splitlines()
    bad = [line for line in lines
           if not (line.startswith("#") or line == "@EXPLICIT" or CONDA_URL.match(line))]
    assert not bad, bad[:5]
    assert lines.count("@EXPLICIT") == 1
    assert f"# platform: {platform_tag}" in lines
    urls = [line for line in lines if line.startswith("https://")]
    subdirs = {url.split("/")[4] for url in urls}
    assert subdirs <= {"noarch", platform_tag}, subdirs
    parsed = tool.parse_conda_lock("\n".join(lines))
    assert len(parsed) == len(urls) > 100, (len(parsed), len(urls))


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_the_pip_lock_is_comments_options_and_hashed_pins_only(platform_tag):
    spec = NUMERICS["platforms"][platform_tag]
    lines = (LOCK_DIR / spec["pip_lock"]).read_text(encoding="utf-8").splitlines()
    bad = [line for line in lines if not (line.startswith("#") or line in PIP_OPTIONS
                                          or PIP_PIN.match(line))]
    assert not bad, bad[:5]
    assert PIP_OPTIONS <= set(lines)
    pins = tool.parse_pip_lock("\n".join(lines))
    assert pins and all(len(entry["sha256"]) == 1 for entry in pins.values())


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_the_numerics_set_names_only_packages_its_lock_holds(platform_tag):
    reference = tool.load_reference(platform_tag, str(LOCK_DIR))
    conda_names = [name for name, _group in tool.judged_names(reference["spec"])]
    absent = [name for name in conda_names if name not in reference["conda"]]
    absent += [name for name in reference["spec"]["pypi"]
               if tool.normalize(name) not in reference["pip"]]
    assert not absent, absent
    for name in reference["spec"].get("by_design") or {}:
        assert name in conda_names, name
    assert reference["spec"]["host_floor"], platform_tag


def test_the_mac_lock_keeps_the_mpi_hdf5_and_the_certified_torch_wheel():
    reference = tool.load_reference("osx-arm64", str(LOCK_DIR))
    assert reference["conda"]["hdf5"]["build"].startswith("mpi_openmpi_")
    assert reference["conda"]["h5py"]["build"].startswith("mpi_openmpi_")
    assert reference["pip"]["torch"]["version"] == "2.10.0"
    assert reference["conda"]["libopenblas"]["build"].startswith("openmp_")
    linux = tool.load_reference("linux-64", str(LOCK_DIR))
    assert linux["conda"]["hdf5"]["build"].startswith("mpi_openmpi_")
    assert linux["pip"]["triton"]["version"] == "3.1.0"
    assert linux["conda"]["sysroot_linux-64"]["version"] == "2.28"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def url(name, version, build, md5="0" * 32, subdir="osx-arm64", ext=".conda"):
    return f"https://conda.anaconda.org/conda-forge/{subdir}/{name}-{version}-{build}{ext}#{md5}"


def test_parse_conda_lock_reads_names_versions_builds_and_md5():
    text = "\n".join([
        "# a comment", "", "@EXPLICIT",
        url("clang_impl_osx-arm64", "22.1.0", "default_h17d1ed9_0", "a" * 32),
        url("cached_property", "1.5.2", "pyha770c72_1", "b" * 32, "noarch", ".tar.bz2"),
        url("libgfortran5", "15.2.0", "hdae7583_18", "c" * 32),
    ])
    parsed = tool.parse_conda_lock(text)
    assert parsed["clang_impl_osx-arm64"]["version"] == "22.1.0"
    assert parsed["clang_impl_osx-arm64"]["build"] == "default_h17d1ed9_0"
    assert parsed["cached_property"]["fn"] == "cached_property-1.5.2-pyha770c72_1.tar.bz2"
    assert parsed["libgfortran5"]["md5"] == "c" * 32


@pytest.mark.parametrize("text, words", [
    ("# no marker\n" + url("a", "1", "0"), "no @EXPLICIT"),
    ("@EXPLICIT\n" + url("a", "1", "0").split("#")[0], "MD5"),
    ("@EXPLICIT\n" + url("a", "1", "0").replace("https://", "http://"), "https URL"),
    ("@EXPLICIT\n" + url("a", "1", "0") + "\n" + url("a", "2", "0"), "listed twice"),
    ("@EXPLICIT\nnumpy==2.4.3", "https URL"),
])
def test_parse_conda_lock_refuses_what_is_not_an_explicit_lock(text, words):
    with pytest.raises(ValueError, match=words):
        tool.parse_conda_lock(text)


def test_parse_pip_lock_skips_options_and_normalizes_names():
    text = ("# comment\n--require-hashes\n--index-url https://pypi.org/simple\n"
            "Typing_Extensions==4.15.0 --hash=sha256:" + "d" * 64 + "\n")
    parsed = tool.parse_pip_lock(text)
    assert parsed == {"typing-extensions": {"name": "Typing_Extensions", "version": "4.15.0",
                                            "sha256": ["d" * 64]}}
    with pytest.raises(ValueError):
        tool.parse_pip_lock("torch>=2.10\n")


# --------------------------------------------------------------------------
# The comparison, on synthetic environments
# --------------------------------------------------------------------------

LOCKED = {
    "fftw": ("3.3.10", "nompi_haf1500d_112"),
    "hdf5": ("1.14.6", "mpi_openmpi_h8451b09_6"),
    "libopenblas": ("0.3.30", "openmp_ha158390_4"),
    "numpy": ("2.4.3", "py312h84a4f5f_0"),
    "python": ("3.12.13", "h8561d8f_0_cpython"),
    "make": ("4.4.1", "h84a0fba_3"),
}
NUMERICS_SPEC = {
    "conda_lock": "ref.conda.txt", "pip_lock": "ref.pip.txt",
    "host_floor": {"macos": "11.0"},
    "conda": {"meep_links": ["fftw", "hdf5", "libopenblas"],
              "python_numerics": ["python", "numpy"]},
    "pypi": ["torch"],
    "by_design": {"hdf5": {"rule": "certified_build", "version": "1.14.6",
                           "build": "nompi_had3affe_106", "md5": "1" * 32,
                           "why": "documented exception"}},
}


def md5_of(name):
    return hashlib.md5(name.encode()).hexdigest()


def write_lock_dir(directory: pathlib.Path, spec=NUMERICS_SPEC, platform_tag="osx-arm64"):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# synthetic", f"# platform: {platform_tag}", "@EXPLICIT"]
    lines += [url(name, version, build, md5_of(name)) for name, (version, build) in LOCKED.items()]
    (directory / "ref.conda.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "ref.pip.txt").write_text(
        "--require-hashes\ntorch==2.10.0 --hash=sha256:" + "e" * 64 + "\n", encoding="utf-8")
    (directory / tool.NUMERICS_FILE).write_text(
        json.dumps({"schema": 1, "platforms": {platform_tag: spec}}), encoding="utf-8")
    return directory


@pytest.fixture()
def reference(tmp_path):
    return tool.load_reference("osx-arm64", str(write_lock_dir(tmp_path / "locks")))


def conda_record(name, version=None, build=None, md5=None):
    version = version or LOCKED[name][0]
    build = build or LOCKED[name][1]
    return {"name": name, "version": version, "build": build, "build_number": None,
            "md5": md5 if md5 is not None else md5_of(name), "url": url(name, version, build)}


def environment(conda="all", distributions=None, python=None, host=None):
    if conda == "all":
        conda = {name: conda_record(name) for name in LOCKED}
    if distributions is None:
        distributions = {"torch": {"name": "torch", "version": "2.10.0", "installer": "pip"},
                         "numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"}}
    return {"conda": conda, "distributions": distributions, "python": python,
            "host": host or {"macos": "26.2", "glibc": None}, "user_site": None,
            "where": "<synthetic>", "conda_prefix": "<synthetic>" if conda is not None else None}


def statuses(result):
    return {row["name"]: row["status"] for row in result["rows"]}


def test_an_environment_holding_every_locked_build_matches(reference):
    result = tool.compare(reference, environment())
    assert set(statuses(result).values()) == {tool.SAME}
    assert result["kind"] == tool.MATCHES and tool.passes(result)
    assert result["verdict"] == ("matches the certification reference: 6 of 6 "
                                 "numerics-relevant packages are the reference's builds")
    assert result["total"] == 6


def test_another_version_is_different_and_named_in_the_verdict(reference):
    env = environment()
    env["conda"]["fftw"] = conda_record("fftw", "3.3.11", "nompi_haf1500d_100", "f" * 32)
    result = tool.compare(reference, env)
    row = next(row for row in result["rows"] if row["name"] == "fftw")
    assert row["status"] == tool.DIFFERENT
    assert row["reference"] == "3.3.10=nompi_haf1500d_112"
    assert row["found"] == "3.3.11=nompi_haf1500d_100"
    assert result["kind"] == tool.DIFFERS and not tool.passes(result)
    assert result["verdict"] == ("differs from the certification reference in 1 of 6 "
                                 "numerics-relevant packages: fftw")


def test_a_missing_package_and_another_file_of_the_same_build(reference):
    env = environment()
    del env["conda"]["libopenblas"]
    env["conda"]["numpy"] = conda_record("numpy", md5="9" * 32)
    result = tool.compare(reference, env)
    assert statuses(result)["libopenblas"] == tool.MISSING
    assert statuses(result)["numpy"] == tool.DIFFERENT
    assert "another file" in next(r for r in result["rows"] if r["name"] == "numpy")["note"]
    assert result["verdict"].startswith("differs from the certification reference in 2 of 6 "
                                        "numerics-relevant packages: libopenblas, numpy")


def test_the_certified_nompi_hdf5_differs_by_design(reference):
    env = environment()
    env["conda"]["hdf5"] = dict(conda_record("hdf5", build="nompi_had3affe_106", md5="1" * 32),
                                build_number=106)
    result = tool.compare(reference, env)
    assert statuses(result)["hdf5"] == tool.BY_DESIGN
    assert tool.passes(result)
    assert result["verdict"].endswith("5 of 6 numerics-relevant packages are the reference's "
                                      "builds and 1 differs from it by design (hdf5)")


@pytest.mark.parametrize("version, build, md5", [
    ("1.14.6", "nompi_had3affe_105", "1" * 32),      # another build
    ("1.14.6", "nompi_hffffff0_106", "1" * 32),      # another nompi build, same number
    ("1.14.6", "mpi_mpich_h1234567_106", "1" * 32),  # not the nompi build
    ("1.14.5", "nompi_had3affe_106", "1" * 32),      # another version
    ("1.14.6", "nompi_had3affe_106", "3" * 32),      # same build, another file
])
def test_only_the_documented_build_is_by_design(reference, version, build, md5):
    env = environment()
    env["conda"]["hdf5"] = conda_record("hdf5", version=version, build=build, md5=md5)
    assert statuses(tool.compare(reference, env))["hdf5"] == tool.DIFFERENT


def test_the_documented_build_without_an_md5_in_conda_meta_is_by_design(reference):
    env = environment()
    env["conda"]["hdf5"] = dict(conda_record("hdf5", build="nompi_had3affe_106"), md5=None)
    assert statuses(tool.compare(reference, env))["hdf5"] == tool.BY_DESIGN


def test_without_the_rule_the_sibling_is_drift(tmp_path):
    spec = dict(NUMERICS_SPEC, by_design={})
    ref = tool.load_reference("osx-arm64", str(write_lock_dir(tmp_path / "locks", spec)))
    env = environment()
    env["conda"]["hdf5"] = dict(conda_record("hdf5", build="nompi_had3affe_106"), build_number=106)
    assert statuses(tool.compare(ref, env))["hdf5"] == tool.DIFFERENT


def test_a_pip_only_install_reports_what_it_can_read_and_says_what_it_cannot(reference):
    env = environment(conda=None, python="3.12.13", distributions={
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip"}})
    result = tool.compare(reference, env)
    assert statuses(result) == {"fftw": tool.UNREAD, "hdf5": tool.UNREAD,
                                "libopenblas": tool.UNREAD, "python": tool.UNREAD,
                                "numpy": tool.MISSING, "torch": tool.SAME}
    assert result["kind"] == tool.DIFFERS
    assert result["verdict"].endswith("1 of 6 numerics-relevant packages: numpy; "
                                      "4 more could not be read")


def test_pip_only_with_nothing_contradicting_is_unconfirmed_not_matched(reference):
    spec = dict(NUMERICS_SPEC, conda={"meep_links": ["fftw"]})
    env = environment(conda=None, python="3.12.13", distributions={
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip"}})
    ref = dict(reference, spec=spec)
    result = tool.compare(ref, env)
    assert result["kind"] == tool.UNCONFIRMED and not tool.passes(result)
    assert result["verdict"].startswith(
        "cannot be confirmed to match the certification reference: 1 of 2 "
        "numerics-relevant packages could not be read (this is not a conda environment")


def test_a_pypi_numpy_in_place_of_the_conda_build_is_different(reference):
    env = environment(conda=None, python="3.12.13", distributions={
        "numpy": {"name": "numpy", "version": "2.4.3", "installer": "pip"},
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip"}})
    row = next(row for row in tool.compare(reference, env)["rows"] if row["name"] == "numpy")
    assert row["status"] == tool.DIFFERENT
    assert "installed by pip" in row["found"]


def test_a_pip_install_over_the_conda_build_is_different(reference):
    env = environment()
    env["distributions"]["numpy"] = {"name": "numpy", "version": "2.5.0", "installer": "pip"}
    row = next(row for row in tool.compare(reference, env)["rows"] if row["name"] == "numpy")
    assert row["status"] == tool.DIFFERENT
    assert "over the conda build" in row["found"]


def test_an_unreadable_installer_over_the_conda_build_is_not_taken_for_pip(reference):
    env = environment()
    env["distributions"]["numpy"] = {"name": "numpy", "version": "2.4.3", "installer": None}
    assert statuses(tool.compare(reference, env))["numpy"] == tool.SAME


@pytest.mark.parametrize("found, status", [
    ({"name": "torch", "version": "2.10.0", "installer": "conda"}, "different"),
    ({"name": "torch", "version": "2.14.0", "installer": "pip"}, "different"),
    ({"name": "torch", "version": "2.10.0", "installer": "uv"}, "same"),
    (None, "missing"),
])
def test_the_pypi_package_is_judged_by_version_and_origin(reference, found, status):
    distributions = {"numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"}}
    if found:
        distributions["torch"] = found
    result = tool.compare(reference, environment(distributions=distributions))
    assert statuses(result)["torch"] == status


def test_a_python_of_another_version_outside_conda_is_different(reference):
    env = environment(conda=None, python="3.11.9", distributions={})
    assert statuses(tool.compare(reference, env))["python"] == tool.DIFFERENT


def test_a_host_below_the_floor_differs_and_an_unread_host_is_a_note(reference):
    below = tool.compare(reference, environment(host={"macos": "10.15.7"}))
    assert below["kind"] == tool.DIFFERS
    assert below["verdict"] == ("differs from the certification reference: macOS 10.15.7 "
                                "is below the lock's floor 11.0")
    unread = tool.compare(reference, environment(host={"macos": None}))
    assert unread["kind"] == tool.MATCHES
    assert any("macOS version not read" in note for note in unread["notes"])


def test_the_certified_mac_environment_as_the_lock_records_it_matches():
    """The lock's own packages, hdf5 swapped for the certified nompi build, torch from PyPI."""
    ref = tool.load_reference("osx-arm64", str(LOCK_DIR))
    conda = {name: dict(entry, build_number=None) for name, entry in ref["conda"].items()}
    conda["hdf5"] = dict(conda["hdf5"], build="nompi_had3affe_106", build_number=106,
                         md5="2d1270d283403c542680e969bea70355")
    env = environment(conda=conda, distributions={
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip", "build_tag": "2"},
        "numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"}})
    result = tool.compare(ref, env)
    assert result["total"] == 29
    assert result["counts"][tool.SAME] == 28 and result["counts"][tool.BY_DESIGN] == 1
    assert tool.passes(result)
    assert result["meep_reference"] == {"version": "1.33.0", "single_precision": True,
                                        "mpi": True}


@pytest.mark.parametrize("build_tag, status", [
    ("2", "same"),
    ("", "different"),      # PyPI's earlier, untagged wheel of the same version
    ("3", "different"),
    (None, "not read"),     # no WHEEL file to read
])
def test_a_wheel_build_tag_the_spec_names_is_judged(reference, build_tag, status):
    spec = dict(NUMERICS_SPEC, pypi_build_tags={"torch": "2"})
    ref = dict(reference, spec=spec)
    env = environment(distributions={
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip",
                  "build_tag": build_tag},
        "numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"}})
    row = next(row for row in tool.compare(ref, env)["rows"] if row["name"] == "torch")
    assert row["status"] == status, row
    assert row["reference"] == "2.10.0, build 2 (PyPI wheel)"
    if status == "different":
        assert "another wheel of the same version" in row["note"]


def test_without_a_named_build_tag_the_wheel_is_judged_by_version(reference):
    env = environment(distributions={
        "torch": {"name": "torch", "version": "2.10.0", "installer": "pip", "build_tag": ""},
        "numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"}})
    assert statuses(tool.compare(reference, env))["torch"] == tool.SAME


def test_the_marker_of_a_solved_build_is_a_note(reference):
    env = dict(environment(), solved_marker="<prefix>/" + tool.SOLVED_MARKER)
    result = tool.compare(reference, env)
    assert any("MEEP_REFERENCE_SOLVE=1" in note for note in result["notes"])
    assert result["kind"] == tool.MATCHES


REFERENCE_MEEP = {"version": "1.33.0", "single_precision": True, "mpi": True}


@pytest.mark.parametrize("facts, ok", [
    ({"version": "1.33.0", "single_precision": True, "mpi": True}, True),
    ({"version": "1.33.0", "single_precision": False, "mpi": True}, False),
    ({"version": "1.33.0", "single_precision": True, "mpi": False}, False),
    ({"version": "1.34.0", "single_precision": True, "mpi": True}, False),
    ({"version": "1.33.0", "single_precision": None, "mpi": True}, False),
    (None, False),
])
def test_judge_meep_requires_the_version_precision_and_mpi_build(facts, ok):
    judged = tool.judge_meep(REFERENCE_MEEP, facts)
    assert judged["ok"] is ok
    assert judged["reference"] == "1.33.0, single precision, MPI build"
    if ok:
        assert judged["line"].endswith(": the reference's build")
    else:
        assert "NOT the reference's build" in judged["line"]


def test_judge_meep_without_an_expectation_is_not_a_match():
    judged = tool.judge_meep(None, REFERENCE_MEEP)
    assert not judged["ok"] and "not compared" in judged["line"]


def test_format_report_lists_every_package_and_ends_with_the_verdict(reference):
    result = tool.compare(reference, environment())
    lines = tool.format_report(result)
    assert lines[-1] == f"   Verdict: this environment {result['verdict']}."
    for name in LOCKED:
        if name != "make":
            assert any(line.strip().startswith(name + " ") for line in lines), name
    assert not any(line.strip().startswith("make ") for line in lines)


# --------------------------------------------------------------------------
# Reading an environment, and never raising
# --------------------------------------------------------------------------

def make_prefix(directory: pathlib.Path, conda: dict, distributions: dict, wheel_tags=None):
    meta = directory / "conda-meta"
    meta.mkdir(parents=True)
    for name, record in conda.items():
        (meta / f"{name}-{record['version']}-{record['build']}.json").write_text(
            json.dumps(dict(record, files=[])), encoding="utf-8")
    (meta / "history").write_text("", encoding="utf-8")
    (meta / "broken.json").write_text("{not json", encoding="utf-8")
    site = directory / "lib" / "python3.12" / "site-packages"
    for name, (version, installer) in distributions.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
        (info / "INSTALLER").write_text(installer + "\n", encoding="utf-8")
        if name in (wheel_tags or {}):
            build = f"Build: {wheel_tags[name]}\n" if wheel_tags[name] else ""
            (info / "WHEEL").write_text(f"Wheel-Version: 1.0\nTag: py3-none-any\n{build}",
                                        encoding="utf-8")
    return directory


def test_read_environment_reads_conda_meta_and_the_prefix_distributions(tmp_path):
    prefix = make_prefix(tmp_path / "env", {"fftw": conda_record("fftw")},
                         {"torch": ("2.10.0", "pip"), "numpy": ("2.4.3", "conda"),
                          "jinja2": ("3.1.6", "pip")},
                         wheel_tags={"torch": "2", "jinja2": ""})
    env = tool.read_environment(str(prefix))
    assert set(env["conda"]) == {"fftw"}
    assert env["conda"]["fftw"]["build"] == "nompi_haf1500d_112"
    assert env["distributions"]["torch"] == {"name": "torch", "version": "2.10.0",
                                             "installer": "pip", "build_tag": "2"}
    assert env["distributions"]["jinja2"]["build_tag"] == ""
    assert env["distributions"]["numpy"]["build_tag"] is None
    assert env["conda_prefix"] == str(prefix)
    assert env["solved_marker"] is None
    marker = prefix / tool.SOLVED_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("solved\n", encoding="utf-8")
    assert tool.read_environment(str(prefix))["solved_marker"] == str(marker)


def test_a_prefix_without_conda_meta_is_read_as_not_conda(tmp_path):
    (tmp_path / "plain").mkdir()
    env = tool.read_environment(str(tmp_path / "plain"))
    assert env["conda"] is None and env["conda_prefix"] is None


@pytest.mark.parametrize("case", ["no lock directory", "unknown platform", "bad environment",
                                  "malformed lock"])
def test_report_never_raises(tmp_path, case):
    if case == "no lock directory":
        result = tool.report("osx-arm64", lock_dir=str(tmp_path / "absent"))
    elif case == "unknown platform":
        result = tool.report("linux-64", lock_dir=str(write_lock_dir(tmp_path / "locks")))
    elif case == "bad environment":
        result = tool.report("osx-arm64", lock_dir=str(write_lock_dir(tmp_path / "locks")),
                             environment={"conda": 5})
    else:
        locks = write_lock_dir(tmp_path / "locks")
        (locks / "ref.conda.txt").write_text("not a lock\n", encoding="utf-8")
        result = tool.report("osx-arm64", lock_dir=str(locks))
    assert result["kind"] == tool.NOT_COMPARED
    assert result["verdict"].startswith("not compared with the certification reference")
    assert not tool.passes(result)
    assert tool.format_report(result) == [f"   {result['verdict']}"]


def run_tool(*arguments):
    return subprocess.run([sys.executable, str(TOOL_PATH), *arguments], capture_output=True,
                          text=True, timeout=120, check=False)


def test_the_command_line_exit_status_follows_require_reference(tmp_path):
    locks = write_lock_dir(tmp_path / "locks")
    same = {name: conda_record(name) for name in LOCKED}
    good = make_prefix(tmp_path / "good", same,
                       {"torch": ("2.10.0", "pip"), "numpy": ("2.4.3", "conda")})
    drifted = dict(same, fftw=conda_record("fftw", "3.3.11", "nompi_haf1500d_100"))
    bad = make_prefix(tmp_path / "bad", drifted,
                      {"torch": ("2.10.0", "pip"), "numpy": ("2.4.3", "conda")})
    common = ("--platform", "osx-arm64", "--lock-dir", str(locks))
    informational = run_tool("--prefix", str(bad), *common)
    assert informational.returncode == 0, informational.stdout + informational.stderr
    assert "Verdict: this environment differs from the certification reference in 1 of 6" \
        in informational.stdout
    required = run_tool("--prefix", str(bad), *common, "--require-reference")
    assert required.returncode == 1, required.stdout
    assert required.stdout.rstrip().endswith("does not match the certification reference.")
    matched = run_tool("--prefix", str(good), *common, "--require-reference")
    assert matched.returncode == 0, matched.stdout
    as_json = json.loads(run_tool("--prefix", str(good), *common, "--json").stdout)
    assert as_json["kind"] == "matches" and as_json["total"] == 6


# --------------------------------------------------------------------------
# check_install, step 7
# --------------------------------------------------------------------------

@pytest.fixture()
def reference_with_meep(tmp_path):
    spec = dict(NUMERICS_SPEC, meep=REFERENCE_MEEP)
    return tool.load_reference("osx-arm64", str(write_lock_dir(tmp_path / "locks", spec)))


def test_check_install_step_7_is_informational_without_the_flag(reference_with_meep):
    differs = tool.compare(reference_with_meep, environment(host={"macos": "10.0"}))
    double = dict(REFERENCE_MEEP, single_precision=False)
    lines, failure = check_install.reference_section(False, tool=tool, result=differs,
                                                     meep=double)
    assert failure is None
    assert lines[0].startswith("   MEEP       1.33.0, double precision, MPI build: NOT")
    assert lines[-1].startswith("   Verdict: this environment differs")


def test_check_install_require_reference_fails_on_anything_but_a_match(reference_with_meep):
    differs = tool.compare(reference_with_meep, environment(host={"macos": "10.0"}))
    _, failure = check_install.reference_section(True, tool=tool, result=differs,
                                                 meep=REFERENCE_MEEP)
    assert isinstance(failure, check_install.Failure)
    assert failure.section == check_install.SECTION_REFERENCE
    assert "--require-reference" in failure.reason and "differs" in failure.reason
    assert "MEEP" not in failure.reason
    matches = tool.compare(reference_with_meep, environment())
    lines, failure = check_install.reference_section(True, tool=tool, result=matches,
                                                     meep=REFERENCE_MEEP)
    assert failure is None
    assert lines[0] == "   MEEP       1.33.0, single precision, MPI build: the reference's build"


@pytest.mark.parametrize("meep", [
    dict(REFERENCE_MEEP, single_precision=False),   # conda-forge's double-precision pymeep
    dict(REFERENCE_MEEP, mpi=False),
    dict(REFERENCE_MEEP, version="1.34.0"),
    None,                                           # not read
])
def test_check_install_require_reference_fails_on_a_meep_that_is_not_the_reference(
        reference_with_meep, meep):
    matches = tool.compare(reference_with_meep, environment())
    assert tool.passes(matches)
    _, failure = check_install.reference_section(True, tool=tool, result=matches, meep=meep)
    assert isinstance(failure, check_install.Failure)
    assert failure.section == check_install.SECTION_REFERENCE
    assert failure.reason.startswith("--require-reference was given and its MEEP is ")
    assert "not the reference's 1.33.0, single precision, MPI build" in failure.reason


def test_check_install_step_7_never_raises():
    class Broken:
        def report(self):
            raise RuntimeError("unreadable")

    lines, failure = check_install.reference_section(False, tool=Broken())
    assert failure is None and "not compared" in lines[0] and "unreadable" in lines[0]
    _, failure = check_install.reference_section(True, tool=Broken(), meep=REFERENCE_MEEP)
    assert isinstance(failure, check_install.Failure)
    assert "its MEEP was not compared" in failure.reason and "unreadable" in failure.reason


def test_check_install_loads_the_tool_by_path_without_touching_the_import_path():
    before = list(sys.path)
    module = check_install.load_reference_tool()
    assert sys.path == before
    assert os.path.samefile(module.__file__, TOOL_PATH)
    assert "meep_gpu_reference_environment" not in sys.modules


def test_install_md_has_the_section_a_failure_names():
    text = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    assert f"### {check_install.SECTION_REFERENCE}\n" in text


def test_check_install_accepts_require_reference():
    done = subprocess.run([sys.executable, str(CHECK_PATH), "--help"], capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 0 and "--require-reference" in done.stdout


# --------------------------------------------------------------------------
# The build scripts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_build_script_installs_and_asserts_the_lock_before_building(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    spec = NUMERICS["platforms"][platform_tag]
    for key in ("conda_lock", "pip_lock"):
        assert f"environments/locks/{spec[key]}" in text, spec[key]
        assert (LOCK_DIR / spec[key]).is_file()
    assert '--no-default-packages --file "$LOCK_CONDA"' in text
    assert 'conda" list --explicit --md5 -p "$ENV_PREFIX"' in text
    assert '-m pip install --no-deps -r "$LOCK_PIP"' in text
    assert '-m pip check' in text
    assert "tools/compare_reference_environment.py\" --require-reference" in text
    assert text.index('say "1b.') < text.index('say "2.') < text.index("./configure")
    assert "MEEP_REFERENCE_SOLVE" in text and "NOT the certification reference" in text
    for variable in ("MEEP_ENV_NAME", "MEEP_BUILD_ROOT", "MEEP_BUILD_JOBS"):
        assert variable in text, variable


def step_1b(platform_tag: str) -> str:
    """The build script's step 1b, from its ``say "1b.`` line to its ``say "2.`` line."""
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    return text[text.index('say "1b.'):text.index('say "2.')]


def run_step_1b(tmp_path: pathlib.Path, platform_tag: str, listing, exit_status=0):
    """Step 1b against a stand-in conda whose ``list`` prints ``listing`` (lines)."""
    spec = NUMERICS["platforms"][platform_tag]
    stub = tmp_path / "conda" / "bin" / "conda"
    stub.parent.mkdir(parents=True)
    (tmp_path / "listing.txt").write_text("".join(line + "\n" for line in listing),
                                          encoding="utf-8")
    stub.write_text(f'#!/bin/sh\ncat "{tmp_path / "listing.txt"}"\nexit {exit_status}\n',
                    encoding="utf-8")
    stub.chmod(0o755)
    prefix = tmp_path / "env"
    prefix.mkdir()
    program = "\n".join([
        "set -euo pipefail",
        "say() { printf '\\n=== %s ===\\n' \"$1\"; }",
        f'CONDA_ROOT="{tmp_path / "conda"}"', f'ENV_PREFIX="{prefix}"',
        f'LOCK_CONDA="{LOCK_DIR / spec["conda_lock"]}"', f'LOCK_PIP="{LOCK_DIR / spec["pip_lock"]}"',
        "SOLVE=0", 'NOT_REFERENCE="not the reference"',
        step_1b(platform_tag)])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    return done, prefix / "share" / "meep-gpu-build"


def lock_urls(platform_tag: str):
    spec = NUMERICS["platforms"][platform_tag]
    lines = (LOCK_DIR / spec["conda_lock"]).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith("https://")]


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_step_1b_accepts_a_prefix_holding_exactly_the_lock(tmp_path, platform_tag):
    urls = lock_urls(platform_tag)
    listing = [f"# platform: {platform_tag}", "@EXPLICIT"] + list(reversed(urls))
    done, record = run_step_1b(tmp_path, platform_tag, listing)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"{len(urls)} of {len(urls)} conda packages are the lock's artifacts" in done.stdout
    spec = NUMERICS["platforms"][platform_tag]
    assert (record / spec["conda_lock"]).is_file() and (record / spec["pip_lock"]).is_file()


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
@pytest.mark.parametrize("change", ["another md5", "one missing", "one added", "conda fails"])
def test_step_1b_refuses_a_prefix_that_is_not_the_lock(tmp_path, platform_tag, change):
    urls = lock_urls(platform_tag)
    exit_status = 0
    if change == "another md5":
        urls[3] = urls[3][:-32] + "0" * 32
    elif change == "one missing":
        del urls[5]
    elif change == "one added":
        urls.append(url("extra", "1.0", "h0_0", "f" * 32, subdir=platform_tag))
    else:
        urls, exit_status = [], 1
    done, record = run_step_1b(tmp_path, platform_tag, ["@EXPLICIT"] + urls, exit_status)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "REFUSING" in done.stdout and "nothing was built on it" in done.stdout
    assert not (record / NUMERICS["platforms"][platform_tag]["conda_lock"]).exists()

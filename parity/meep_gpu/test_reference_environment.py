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
  refuses one changed line or a conda that fails; they clear the calling environment and
  refuse the retired knobs by name;
* the corpus-only packages (``corpus_only``): pinned in every pip lock or held by every
  conda lock and judged by no numerics set; reported apart and never counted in the
  verdict, so an environment without them is judged on its numerics exactly as one with
  them; the stand-in held only with its pinned files; ffmpeg held only as the
  imageio-ffmpeg pin's executable; ``--require-pip-lock`` requires the pip lock's pins and
  the stand-in, exactly; the build scripts install them and check that, and step 7b links
  the pin's ffmpeg;
* the lock re-cut (``environments/locks/recut_locks.py``): the committed locks are what
  their inputs make, an addition moves no locked package, and a solve that moves, drops or
  adds one is refused by name; ``additions.json`` names only what the locks hold;
* the build record helper (``parity/meep_gpu/build_meep_133_record.py``): the corpus kept
  and listed, the tarball unpacked inside its directory, the load commands, the ffmpeg
  link, the programs on the reference PATH and the record it writes;
* the stand-in's builder (``parity/meep_gpu/rounds/1.0/build_pymiescatt_standin.py``): its
  patch parser and applier (no context, no fuzz, line endings kept), and its refusals of
  an sdist of another hash and of members outside the sdist's directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tarfile
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "environments" / "locks"
TOOL_PATH = ROOT / "tools" / "compare_reference_environment.py"
CHECK_PATH = ROOT / "tools" / "check_install.py"
BUILDER_PATH = ROOT / "parity" / "meep_gpu" / "rounds" / "1.0" / "build_pymiescatt_standin.py"
RECUT_PATH = LOCK_DIR / "recut_locks.py"
RECORD_PATH = ROOT / "parity" / "meep_gpu" / "build_meep_133_record.py"
SCRIPTS = {
    "osx-arm64": ROOT / "parity" / "meep_gpu" / "build_meep_133_macos.sh",
    "linux-64": ROOT / "parity" / "meep_gpu" / "build_meep_133_linux.sh",
}
#: How each script's step 7 installs the pip lock. The Linux script installs it one pin
#: at a time from the parts the build record helper splits it into, so pip holds one wheel
#: at a time (its lock carries 3.1 GB of wheels, built on a nearly full shared disk).
PIP_INSTALL = {
    "osx-arm64": 'py -m pip install --no-deps -r "$LOCK_PIP"',
    "linux-64": ('helper pip-lock-parts "$LOCK_PIP" "$BUILD_ROOT/pip-lock-parts"\n'
                 'for part in "$BUILD_ROOT"/pip-lock-parts/*.txt; do\n'
                 '  py -m pip install --no-deps -r "$part"\n'
                 'done\n'),
}


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load("reference_environment_under_test", TOOL_PATH)
check_install = _load("check_install_under_test", CHECK_PATH)
builder = _load("pymiescatt_standin_builder_under_test", BUILDER_PATH)
recut = _load("recut_locks_under_test", RECUT_PATH)
record = _load("build_meep_133_record_under_test", RECORD_PATH)

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
    "parity/meep_gpu/build_meep_133_record.py", "tools/fma_census.py",
    "parity/meep_gpu/meep_identity_smoke.py", "parity/meep_gpu/meep_identity_smoke.json",
    "parity/meep_gpu/test_identity_reference.py",
    "parity/meep_gpu/rounds/1.0/build_pymiescatt_standin.py",
    "parity/meep_gpu/rounds/1.0/pymiescatt_standin.patch",
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


CORPUS_ONLY = NUMERICS["corpus_only"]


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_every_pip_lock_pins_the_corpus_packages_and_no_numerics_set_names_them(platform_tag):
    reference = tool.load_reference(platform_tag, str(LOCK_DIR))
    judged = {name for name, _group in tool.judged_names(reference["spec"])}
    judged |= {tool.normalize(name) for name in reference["spec"]["pypi"]}
    for name, entry in CORPUS_ONLY["pypi"].items():
        pin = reference["pip"][tool.normalize(name)]
        assert len(pin["sha256"]) == 1, (platform_tag, name)
        assert entry["rows"], name
        assert name not in judged and tool.normalize(name) not in judged, name
    for name, entry in CORPUS_ONLY["conda"].items():
        assert name in reference["conda"], (platform_tag, name)
        assert entry["rows"], name
        assert name not in judged, name
    standin = CORPUS_ONLY["standin"]
    assert tool.normalize(standin["name"]) not in reference["pip"]
    assert standin["name"] not in judged
    assert reference["corpus"] is not None and reference["corpus"]["standin"] == standin


def test_the_corpus_pins_are_the_wheels_the_corpus_legs_measured():
    """shapely 2.1.2 and nlopt 2.11.0, the wheels of 2026-10-10's corpus legs, per platform."""
    mac = tool.load_reference("osx-arm64", str(LOCK_DIR))["pip"]
    linux = tool.load_reference("linux-64", str(LOCK_DIR))["pip"]
    assert (mac["shapely"]["version"], mac["nlopt"]["version"]) == ("2.1.2", "2.11.0")
    assert (linux["shapely"]["version"], linux["nlopt"]["version"]) == ("2.1.2", "2.11.0")
    assert mac["shapely"]["sha256"] == [
        "ba4d1333cc0bc94381d6d4308d2e4e008e0bd128bdcff5573199742ee3634359"]
    assert mac["nlopt"]["sha256"] == [
        "251b51a6b4ee761945211c9213c97d2c6a6d444138fa15547dd77b83f6bbe1e6"]
    assert linux["shapely"]["sha256"] == [
        "e38a190442aacc67ff9f75ce60aec04893041f16f97d242209106d502486a142"]
    assert linux["nlopt"]["sha256"] == [
        "fd42cc951ca0015a4b6dea52ac5dcba652adf1897e97be279a4b6d2c4e0e29e5"]


def test_the_stand_in_is_pinned_by_its_sdist_its_patch_and_its_files():
    standin = CORPUS_ONLY["standin"]
    assert standin["name"] == "PyMieScatt" and standin["version"] == "1.8.1.1"
    assert standin["sdist"]["url"].startswith("https://files.pythonhosted.org/packages/")
    assert standin["sdist"]["url"].endswith(f"/PyMieScatt-{standin['version']}.tar.gz")
    assert re.fullmatch(r"[0-9a-f]{64}", standin["sdist"]["sha256"])
    assert (ROOT / standin["builder"]).resolve() == BUILDER_PATH.resolve()
    assert (ROOT / standin["patch"]).is_file()
    assert sorted(standin["files"]) == [f"PyMieScatt/{name}" for name in
                                        ("CoreShell.py", "Inverse.py", "Mie.py", "__init__.py",
                                         "_version.py")]
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in standin["files"].values())
    patch = builder.parse_patch((ROOT / standin["patch"]).read_text(encoding="utf-8"))
    assert sorted(patch) == ["PyMieScatt/Inverse.py", "PyMieScatt/Mie.py"]
    for hunks in patch.values():
        assert [(removed, added) for _start, removed, added in hunks] == [
            (["from scipy.integrate import trapz"], ["from numpy import trapezoid as trapz"])]
    ffmpeg = CORPUS_ONLY["pypi"]["imageio-ffmpeg"]
    assert ffmpeg["rows"] == ["t.test_visualization.TestVisualization.test_animation_output"]
    assert ffmpeg["links"] == {"ffmpeg": "imageio_ffmpeg/binaries"}
    assert CORPUS_ONLY["conda"]["gdspy"]["rows"] == ["ex.ring_gds"]
    assert "host_tools" not in CORPUS_ONLY


def test_a_numerics_file_with_the_retired_host_tools_entry_is_refused_by_name(tmp_path):
    locks = write_lock_dir(tmp_path / "locks")
    numerics = json.loads((locks / tool.NUMERICS_FILE).read_text(encoding="utf-8"))
    numerics["corpus_only"] = {"host_tools": {"ffmpeg": {"rows": ["t.x"]}}}
    (locks / tool.NUMERICS_FILE).write_text(json.dumps(numerics), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_only.host_tools is retired"):
        tool.load_reference("osx-arm64", str(locks))
    result = tool.report("osx-arm64", lock_dir=str(locks), environment=environment())
    assert result["kind"] == tool.NOT_COMPARED and "host_tools is retired" in result["verdict"]


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_libgdsii_is_judged_with_what_meep_links_and_meep_is_built_with_it(platform_tag):
    spec = NUMERICS["platforms"][platform_tag]
    assert "libgdsii" in spec["conda"]["meep_links"]
    assert spec["meep"] == {"version": "1.33.0", "single_precision": True, "mpi": True,
                            "gdsii": True}
    reference = tool.load_reference(platform_tag, str(LOCK_DIR))
    assert reference["conda"]["libgdsii"]["version"] == "0.21"
    total = len(tool.judged_names(reference["spec"])) + len(reference["spec"]["pypi"])
    assert total == {"osx-arm64": 30, "linux-64": 37}[platform_tag]


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
    assert result["total"] == 30
    assert result["counts"][tool.SAME] == 29 and result["counts"][tool.BY_DESIGN] == 1
    assert tool.passes(result)
    assert result["meep_reference"] == {"version": "1.33.0", "single_precision": True,
                                        "mpi": True, "gdsii": True}


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


@pytest.mark.parametrize("gdsii, ok, words", [
    (True, True, "1.33.0, single precision, MPI build, libGDSII: the reference's build"),
    (False, False, "NOT the reference's build (1.33.0, single precision, MPI build, libGDSII)"),
    (None, True, "the reference's build (libGDSII not read)"),
])
def test_judge_meep_judges_libgdsii_when_the_facts_carry_it(gdsii, ok, words):
    facts = dict(REFERENCE_MEEP, **({} if gdsii is None else {"gdsii": gdsii}))
    judged = tool.judge_meep(dict(REFERENCE_MEEP, gdsii=True), facts)
    assert judged["ok"] is ok
    assert words in judged["line"], judged["line"]


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
# The corpus-only packages: reported apart, never in the verdict
# --------------------------------------------------------------------------

STANDIN_FILES = {"PyMieScatt/Mie.py": "a" * 64, "PyMieScatt/__init__.py": "b" * 64}
CORPUS = {
    "pypi": {"shapely": {"rows": ["ex.mie_scattering"]}, "nlopt": {"rows": ["ex.waveguide_crossing"]},
             "imageio-ffmpeg": {"rows": ["t.test_animation_output"],
                                "links": {"ffmpeg": "imageio_ffmpeg/binaries"}}},
    "conda": {"make": {"rows": ["ex.ring_gds"]}},
    "standin": {"name": "PyMieScatt", "version": "1.8.1.1", "rows": ["ex.mie_scattering"],
                "builder": "build_pymiescatt_standin.py", "files": STANDIN_FILES},
}
CORPUS_PINS = {"shapely": ("2.1.2", "1"), "nlopt": ("2.11.0", "2"), "imageio-ffmpeg": ("0.6.0", "3")}
#: Where the synthetic imageio-ffmpeg distribution's executables are.
BINARIES = "/env/lib/python3.12/site-packages/imageio_ffmpeg/binaries"


@pytest.fixture()
def corpus_reference(tmp_path):
    locks = write_lock_dir(tmp_path / "locks")
    with open(locks / "ref.pip.txt", "a", encoding="utf-8") as handle:
        for name, (version, digit) in CORPUS_PINS.items():
            handle.write(f"{name}=={version} --hash=sha256:{digit * 64}\n")
    numerics = json.loads((locks / tool.NUMERICS_FILE).read_text(encoding="utf-8"))
    numerics["corpus_only"] = CORPUS
    (locks / tool.NUMERICS_FILE).write_text(json.dumps(numerics), encoding="utf-8")
    return tool.load_reference("osx-arm64", str(locks))


def corpus_distributions(standin_files=STANDIN_FILES, **changes):
    found = {"torch": {"name": "torch", "version": "2.10.0", "installer": "pip"},
             "numpy": {"name": "numpy", "version": "2.4.3", "installer": "conda"},
             "shapely": {"name": "shapely", "version": "2.1.2", "installer": "pip"},
             "nlopt": {"name": "nlopt", "version": "2.11.0", "installer": "pip"},
             "imageio-ffmpeg": {"name": "imageio-ffmpeg", "version": "0.6.0", "installer": "pip",
                                "located": {"imageio_ffmpeg/binaries": BINARIES}},
             "pymiescatt": {"name": "PyMieScatt", "version": "1.8.1.1", "installer": "pip",
                            "files_sha256": dict(standin_files)}}
    for name, value in changes.items():
        if value is None:
            found.pop(name)
        else:
            found[name] = value
    return found


def corpus_environment(links=None, **changes):
    env = environment(distributions=corpus_distributions(**changes))
    env["links"] = {"ffmpeg": BINARIES + "/ffmpeg-macos-aarch64-v7.1"} if links is None else links
    return env


def corpus_statuses(result):
    return {row["name"]: row["status"] for row in result["corpus"]["rows"]}


def test_the_corpus_packages_never_enter_the_numerics_rows_or_the_verdict(corpus_reference,
                                                                           reference):
    without = environment()           # none of the corpus packages, no program looked up
    del without["conda"]["make"]
    held = corpus_environment()
    plain = tool.compare(reference, environment())
    for env in (without, held):
        result = tool.compare(corpus_reference, env)
        assert result["rows"] == plain["rows"] and result["total"] == plain["total"] == 6
        assert result["verdict"] == plain["verdict"] and tool.passes(result)
        assert not {row["name"] for row in result["rows"]} & {
            "shapely", "nlopt", "imageio-ffmpeg", "PyMieScatt", "ffmpeg"}
    assert corpus_statuses(tool.compare(corpus_reference, without)) == {
        "shapely": tool.ABSENT, "nlopt": tool.ABSENT, "imageio-ffmpeg": tool.ABSENT,
        "make": tool.ABSENT, "PyMieScatt": tool.ABSENT, "ffmpeg": tool.UNREAD}
    result = tool.compare(corpus_reference, held)
    assert set(corpus_statuses(result).values()) == {tool.HELD}
    assert (result["corpus"]["held"], result["corpus"]["total"]) == (6, 6)


def test_a_numerics_difference_is_judged_as_strictly_with_or_without_the_corpus(corpus_reference,
                                                                                 reference):
    for env in (environment(), corpus_environment()):
        env["conda"]["fftw"] = conda_record("fftw", "3.3.11", "nompi_haf1500d_100", "f" * 32)
        for ref in (reference, corpus_reference):
            result = tool.compare(ref, env)
            assert result["kind"] == tool.DIFFERS
            assert result["verdict"] == ("differs from the certification reference in 1 of 6 "
                                         "numerics-relevant packages: fftw")


@pytest.mark.parametrize("change, name, status", [
    ({"shapely": {"name": "shapely", "version": "2.0.7", "installer": "pip"}}, "shapely", "another build"),
    ({"nlopt": {"name": "nlopt", "version": "2.11.0", "installer": "conda"}}, "nlopt", "another build"),
    ({"nlopt": None}, "nlopt", "absent"),
    ({"pymiescatt": {"name": "PyMieScatt", "version": "1.8.1", "installer": "pip",
                     "files_sha256": STANDIN_FILES}}, "PyMieScatt", "another build"),
    ({"standin_files": dict(STANDIN_FILES, **{"PyMieScatt/Mie.py": "c" * 64})},
     "PyMieScatt", "another build"),
    ({"standin_files": {"PyMieScatt/Mie.py": "a" * 64}}, "PyMieScatt", "another build"),
])
def test_each_corpus_package_is_held_only_as_pinned(corpus_reference, change, name, status):
    result = tool.compare(corpus_reference, corpus_environment(**change))
    assert corpus_statuses(result)[name] == status
    assert tool.passes(result)
    row = next(row for row in result["corpus"]["rows"] if row["name"] == name)
    if name == "PyMieScatt" and "standin_files" in change:
        assert "not the stand-in's files" in row["note"]
        assert "PyMieScatt/Mie.py" in row["note"] or "PyMieScatt/__init__.py" in row["note"]


@pytest.mark.parametrize("links, status", [
    ({"ffmpeg": BINARIES + "/ffmpeg-macos-aarch64-v7.1"}, "held"),
    ({"ffmpeg": "/usr/bin/ffmpeg"}, "another build"),        # a host's own ffmpeg
    ({"ffmpeg": BINARIES + "-elsewhere/ffmpeg"}, "another build"),
    ({"ffmpeg": None}, "absent"), ({}, "not read")])
def test_ffmpeg_is_held_only_as_the_pinned_wheels_executable(corpus_reference, links, status):
    result = tool.compare(corpus_reference, corpus_environment(links=links))
    assert corpus_statuses(result)["ffmpeg"] == status
    assert tool.passes(result)
    row = next(row for row in result["corpus"]["rows"] if row["name"] == "ffmpeg")
    if status == "absent":
        assert "t.test_animation_output" in row["note"] and "<prefix>/bin" in row["note"]
    if status == "another build":
        assert "a host's own ffmpeg is not the reference's" in row["note"]


def test_ffmpeg_without_its_distribution_is_not_held(corpus_reference):
    result = tool.compare(corpus_reference, corpus_environment(**{"imageio-ffmpeg": None}))
    assert corpus_statuses(result)["ffmpeg"] == "another build"


@pytest.mark.parametrize("record, distribution, status", [
    ("same", None, "held"),
    ("other build", None, "another build"),
    ("same", {"name": "make", "version": "4.4.1", "installer": "pip"}, "another build"),
    (None, {"name": "make", "version": "4.4.1", "installer": "pip"}, "another build"),
    (None, None, "absent"),
])
def test_a_conda_corpus_package_is_held_only_as_the_locked_build(corpus_reference, record,
                                                                 distribution, status):
    env = corpus_environment()
    if record is None:
        del env["conda"]["make"]
    elif record == "other build":
        env["conda"]["make"] = conda_record("make", build="h0_0")
    if distribution:
        env["distributions"]["make"] = distribution
    assert corpus_statuses(tool.compare(corpus_reference, env))["make"] == status


def test_format_report_prints_the_corpus_section_before_the_verdict(corpus_reference):
    result = tool.compare(corpus_reference, corpus_environment(nlopt=None))
    lines = tool.format_report(result)
    assert lines[-1] == f"   Verdict: this environment {result['verdict']}."
    start = next(i for i, line in enumerate(lines) if line.startswith("   corpus     5 of 6 held"))
    assert "not part of the verdict" in lines[start]
    of = next(i for i, line in enumerate(lines) if line.startswith("   of 6:"))
    assert of < start < len(lines) - 1
    section = lines[start:]
    assert any(line.split()[:1] == ["nlopt"] and line.rstrip().endswith("absent") for line in section)
    assert any("refuses ex.waveguide_crossing by name" in line for line in section)
    assert tool.format_corpus(None) == [] and tool.format_corpus({"rows": []}) == []


def test_the_pip_lock_check_requires_exactly_the_pins_and_the_stand_in(corpus_reference):
    exact = tool.pip_lock_exact(corpus_reference, corpus_environment())
    assert exact["ok"] and (exact["held"], exact["expected"]) == (5, 5)
    assert exact["line"] == ("the PyPI packages are exactly the pip lock's 4 pins and the "
                             "PyMieScatt stand-in (5 of 5)")
    cases = {
        "missing": ({"nlopt": None}, "nlopt 2.11.0 is missing"),
        "version": ({"shapely": {"name": "shapely", "version": "2.0.7", "installer": "pip"}},
                    "shapely is 2.0.7, the lock pins 2.1.2"),
        "conda": ({"torch": {"name": "torch", "version": "2.10.0", "installer": "conda"}},
                  "torch is 2.10.0 installed by conda, not the pinned wheel"),
        "stand-in files": ({"standin_files": dict(STANDIN_FILES, **{"PyMieScatt/Mie.py": "c" * 64})},
                           "PyMieScatt 1.8.1.1 is not the stand-in: PyMieScatt/Mie.py differ"),
        "stand-in missing": ({"pymiescatt": None}, "PyMieScatt 1.8.1.1 is missing"),
        "extra": ({"jax": {"name": "jax", "version": "0.4.30", "installer": "pip"}},
                  "jax 0.4.30 is not in the lock"),
    }
    for label, (change, words) in cases.items():
        result = tool.pip_lock_exact(corpus_reference, corpus_environment(**change))
        assert not result["ok"], label
        assert words in result["line"], (label, result["line"])
    # a conda-installed distribution the lock does not pin is the conda part's, not extra
    conda_extra = corpus_environment(matplotlib={"name": "matplotlib", "version": "3.10.8",
                                                 "installer": "conda"},
                                     wheel={"name": "wheel", "version": "0.46.3", "installer": None})
    assert tool.pip_lock_exact(corpus_reference, conda_extra)["ok"]


def link_ffmpeg_into(prefix: pathlib.Path) -> pathlib.Path:
    """The wheel's executable in the prefix's site-packages, linked into <prefix>/bin."""
    binary = (prefix / "lib" / "python3.12" / "site-packages" / "imageio_ffmpeg" / "binaries"
              / "ffmpeg-test")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\necho 'ffmpeg version test'\n", encoding="utf-8")
    binary.chmod(0o755)
    (prefix / "bin").mkdir(exist_ok=True)
    (prefix / "bin" / "ffmpeg").symlink_to(os.path.relpath(binary, prefix / "bin"))
    return binary


def test_read_environment_hashes_the_named_distribution_and_resolves_the_links(tmp_path):
    prefix = make_prefix(tmp_path / "env", {"fftw": conda_record("fftw")},
                         {"PyMieScatt": ("1.8.1.1", "pip"), "torch": ("2.10.0", "pip"),
                          "imageio-ffmpeg": ("0.6.0", "pip")},
                         files={"PyMieScatt/Mie.py": b"mie\r\n"})
    binary = link_ffmpeg_into(prefix)
    env = tool.read_environment(str(prefix), {"pymiescatt": ["PyMieScatt/Mie.py",
                                                             "PyMieScatt/absent.py"]},
                                {"ffmpeg": ("imageio-ffmpeg", "imageio_ffmpeg/binaries"),
                                 "no-such-tool-here": ("imageio-ffmpeg", "imageio_ffmpeg")})
    assert env["distributions"]["pymiescatt"]["files_sha256"] == {
        "PyMieScatt/Mie.py": hashlib.sha256(b"mie\r\n").hexdigest(), "PyMieScatt/absent.py": None}
    assert "files_sha256" not in env["distributions"]["torch"]
    assert env["links"] == {"ffmpeg": os.path.realpath(binary), "no-such-tool-here": None}
    assert env["distributions"]["imageio-ffmpeg"]["located"] == {
        "imageio_ffmpeg/binaries": os.path.realpath(binary.parent),
        "imageio_ffmpeg": os.path.realpath(binary.parent.parent)}


def test_the_command_line_exit_status_follows_require_pip_lock(tmp_path):
    locks = write_lock_dir(tmp_path / "locks")
    with open(locks / "ref.pip.txt", "a", encoding="utf-8") as handle:
        for name, (version, digit) in CORPUS_PINS.items():
            handle.write(f"{name}=={version} --hash=sha256:{digit * 64}\n")
    content = {"PyMieScatt/Mie.py": b"patched mie\r\n", "PyMieScatt/__init__.py": b"init\r\n"}
    corpus = dict(CORPUS, standin=dict(CORPUS["standin"], files={
        relative: hashlib.sha256(data).hexdigest() for relative, data in content.items()}))
    numerics = json.loads((locks / tool.NUMERICS_FILE).read_text(encoding="utf-8"))
    numerics["corpus_only"] = corpus
    (locks / tool.NUMERICS_FILE).write_text(json.dumps(numerics), encoding="utf-8")
    same = {name: conda_record(name) for name in LOCKED}
    dists = {"torch": ("2.10.0", "pip"), "numpy": ("2.4.3", "conda"), "shapely": ("2.1.2", "pip"),
             "nlopt": ("2.11.0", "pip"), "PyMieScatt": ("1.8.1.1", "pip")}
    dists["imageio-ffmpeg"] = ("0.6.0", "pip")
    good = make_prefix(tmp_path / "good", same, dists, files=content)
    link_ffmpeg_into(good)
    stock = make_prefix(tmp_path / "stock", same, dists,
                        files=dict(content, **{"PyMieScatt/Mie.py": b"stock mie\r\n"}))
    bare = make_prefix(tmp_path / "bare", same, {"torch": ("2.10.0", "pip"),
                                                 "numpy": ("2.4.3", "conda")})
    common = ("--platform", "osx-arm64", "--lock-dir", str(locks))
    ok = run_tool("--prefix", str(good), *common, "--require-pip-lock", "--require-reference")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "pip lock   ref.pip.txt: the PyPI packages are exactly the pip lock's 4 pins and the " \
        "PyMieScatt stand-in (5 of 5)." in ok.stdout
    # the prefix's own bin comes first on the reference PATH, so a host's /usr/bin/ffmpeg
    # does not change the count; the conda corpus package (make) is held as locked.
    assert "corpus     6 of 6 held" in ok.stdout, ok.stdout
    for prefix, words in ((stock, "is not the stand-in: PyMieScatt/Mie.py differ"),
                          (bare, "nlopt 2.11.0 is missing")):
        failed = run_tool("--prefix", str(prefix), *common, "--require-pip-lock")
        assert failed.returncode == 1, failed.stdout
        assert words in failed.stdout
        assert failed.stdout.rstrip().endswith("exactly the pip lock's and the corpus stand-in.")
        matched = run_tool("--prefix", str(prefix), *common, "--require-reference")
        assert matched.returncode == 0, matched.stdout    # the numerics verdict ignores them
        assert "Verdict: this environment matches the certification reference" in matched.stdout


# --------------------------------------------------------------------------
# Reading an environment, and never raising
# --------------------------------------------------------------------------

def make_prefix(directory: pathlib.Path, conda: dict, distributions: dict, wheel_tags=None,
                files=None):
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
    for relative, data in (files or {}).items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
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
# The stand-in's builder
# --------------------------------------------------------------------------

def test_the_patch_applies_at_its_line_numbers_and_keeps_the_line_endings():
    data = b"one\r\ntwo\r\nthree\r\nfour\r\n"
    patch = builder.parse_patch("prose\n\n--- a/p.py\n+++ b/p.py\n@@ -2 +2 @@\n-two\n+TWO\n"
                                "@@ -4 +4,2 @@\n-four\n+FOUR\n+five\n")
    assert builder.apply_hunks(data, patch["p.py"], "p.py") == \
        b"one\r\nTWO\r\nthree\r\nFOUR\r\nfive\r\n"
    assert builder.apply_hunks(b"one\ntwo\n", builder.parse_patch(
        "--- a/p.py\n+++ b/p.py\n@@ -2 +2 @@\n-two\n+TWO\n")["p.py"]) == b"one\nTWO\n"


@pytest.mark.parametrize("text, words", [
    ("--- a/p.py\n+++ b/p.py\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n", "context line"),
    ("--- a/p.py\n+++ b/p.py\n@@ -2,2 +2 @@\n-two\n+TWO\n", "its header says"),
    ("--- a/p.py\n+++ b/q.py\n@@ -2 +2 @@\n-two\n+TWO\n", "a/<path> and b/<path>"),
    ("--- a/p.py\n@@ -2 +2 @@\n-two\n+TWO\n", "without '+++'"),
    ("just prose\n", "changes no file"),
])
def test_the_patch_parser_refuses_anything_but_bare_hunks(text, words):
    with pytest.raises(builder.Refused, match=re.escape(words)):
        builder.parse_patch(text)


def test_a_removed_line_must_be_exactly_where_the_patch_says():
    patch = builder.parse_patch("--- a/p.py\n+++ b/p.py\n@@ -2 +2 @@\n-two\n+TWO\n")["p.py"]
    for data in (b"one\nthree\ntwo\n", b"one\ntwo \n", b"one\n"):
        with pytest.raises(builder.Refused, match="the patch removes"):
            builder.apply_hunks(data, patch, "p.py")


def _sdist(path: pathlib.Path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("members, words", [
    ({"PyMieScatt-1.8.1.1/../evil.py": b"x"}, "outside its directory"),
    ({"/abs.py": b"x"}, "outside its directory"),
    ({"PyMieScatt-1.8.1.1/link": None}, "not a file or directory"),
    ({"a/x.py": b"x", "b/y.py": b"y"}, "2 top directories"),
])
def test_unpacking_refuses_members_outside_one_directory(tmp_path, members, words):
    sdist = _sdist(tmp_path / "s.tar.gz", members)
    with pytest.raises(builder.Refused, match=words):
        builder.unpack(str(sdist), str(tmp_path / "out"))


def test_an_sdist_of_another_hash_is_refused_and_nothing_is_built(tmp_path, capsys):
    sdist = _sdist(tmp_path / "PyMieScatt-1.8.1.1.tar.gz", {"PyMieScatt-1.8.1.1/setup.py": b"x"})
    status = builder.main(["--python", sys.executable, "--sdist", str(sdist),
                           "--work", str(tmp_path / "work"), "--lock-dir", str(LOCK_DIR)])
    out = capsys.readouterr().out
    assert status == 1
    assert "REFUSING: the sdist's SHA-256 is" in out and CORPUS_ONLY["standin"]["sdist"]["sha256"] in out
    assert not (tmp_path / "work" / "dist").exists()


def test_the_package_files_are_checked_against_the_pins(tmp_path):
    spec = {"name": "Pkg", "files": {"Pkg/a.py": hashlib.sha256(b"a").hexdigest()}}
    (tmp_path / "Pkg").mkdir()
    (tmp_path / "Pkg" / "a.py").write_bytes(b"a")
    builder.check_package_files(str(tmp_path), spec, "test")
    (tmp_path / "Pkg" / "b.py").write_bytes(b"b")
    with pytest.raises(builder.Refused, match="the package holds"):
        builder.check_package_files(str(tmp_path), spec, "test")
    (tmp_path / "Pkg" / "b.py").unlink()
    (tmp_path / "Pkg" / "a.py").write_bytes(b"A")
    with pytest.raises(builder.Refused, match="has SHA-256"):
        builder.check_package_files(str(tmp_path), spec, "test")


def test_the_builder_reads_its_pins_from_the_lock_directory():
    assert builder.load_spec(str(LOCK_DIR)) == CORPUS_ONLY["standin"]
    assert os.path.samefile(builder.LOCK_DIR, LOCK_DIR)


# --------------------------------------------------------------------------
# The build scripts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_build_script_installs_and_asserts_the_lock_before_building(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    spec = NUMERICS["platforms"][platform_tag]
    for key in ("conda_lock", "pip_lock"):
        assert f"environments/locks/{spec[key]}" in text or f'$LOCK_DIR/{spec[key]}' in text, spec[key]
        assert (LOCK_DIR / spec[key]).is_file()
    assert '--no-default-packages --file "$LOCK_CONDA"' in text
    assert 'conda" list --explicit --md5 -p "$ENV_PREFIX"' in text
    assert PIP_INSTALL[platform_tag] in text
    assert 'py -m pip check' in text
    assert 'tools/compare_reference_environment.py" --require-reference' in text
    assert text.index('say "1b.') < text.index('say "2.') < text.index("./configure")
    for variable in ("MEEP_ENV_NAME", "MEEP_ENV_PREFIX", "CONDA_PKGS_DIRS", "MEEP_BUILD_ROOT",
                     "MEEP_BUILD_JOBS", "PIP_CONFIG_FILE=/dev/null", "PYTHONNOUSERSITE=1",
                     "PIP_NO_CACHE_DIR=1"):
        assert variable in text, variable
    assert "MEEP_SIGMA_PATCH" in text.split("RETIRED=(", 1)[1].split(")", 1)[0]
    assert "patch -p1" not in text and "meep-sigma-reader.patch" not in text


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_build_keeps_the_corpus_before_it_builds_and_records_it_last(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    two = text[text.index('say "2.'):text.index('say "3.')]
    assert two.index("fetch_tarball ") < two.index("helper extract") < two.index("helper keep-corpus")
    assert 'CORPUS_ROOT="$RECORD_DIR/meep-${MEEP_VERSION}/python"' in text
    assert "ac_cv_lib_GDSII_libGDSIIExists=yes" in text[text.index('say "4.'):text.index('say "5.')]
    assert 'CORPUS_SIBLINGS="$RECORD_DIR/corpus_siblings_sha256.txt"' in text
    assert ('helper keep-corpus "$BUILD_ROOT/meep-${MEEP_VERSION}/python" "$CORPUS_ROOT" '
            '"$CORPUS_LIST" \\\n  "$CORPUS_SIBLINGS"') in two
    last = text[text.rindex('say "12.'):text.rindex('say "DONE"')]
    assert last.index("helper tools") < last.index("helper verify-corpus") < last.index("helper manifest")
    assert '--corpus-root "share/meep-gpu-build/meep-${MEEP_VERSION}/python"' in last
    assert 'helper verify-corpus "$CORPUS_ROOT" "$CORPUS_LIST" "$CORPUS_SIBLINGS"' in last
    assert '--corpus-siblings "$CORPUS_SIBLINGS"' in last


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
        f'RECORD_DIR="{prefix / "share" / "meep-gpu-build"}"', f'LOCK_DIR="{LOCK_DIR}"',
        f'LOCK_CONDA="{LOCK_DIR / spec["conda_lock"]}"', f'LOCK_PIP="{LOCK_DIR / spec["pip_lock"]}"',
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
    done, record_dir = run_step_1b(tmp_path, platform_tag, listing)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"{len(urls)} of {len(urls)} conda packages are the lock's artifacts" in done.stdout
    spec = NUMERICS["platforms"][platform_tag]
    for name in (spec["conda_lock"], spec["pip_lock"], "numerics-relevant.json", "additions.json"):
        assert (record_dir / name).is_file(), name


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
    done, record_dir = run_step_1b(tmp_path, platform_tag, ["@EXPLICIT"] + urls, exit_status)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "REFUSING" in done.stdout and "nothing was built on it" in done.stdout
    assert not (record_dir / NUMERICS["platforms"][platform_tag]["conda_lock"]).exists()


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_step_7_installs_the_pip_lock_and_the_stand_in_and_requires_exactly_them(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    seven = text[text.index('say "7.'):text.index('say "7b.')]
    order = [PIP_INSTALL[platform_tag], "rounds/1.0/build_pymiescatt_standin.py",
             '--python "$ENV_PREFIX/bin/python" --install', "py -m pip check",
             'tools/compare_reference_environment.py" \\\n    --prefix "$ENV_PREFIX" --require-pip-lock']
    positions = [seven.index(needle) for needle in order]
    assert positions == sorted(positions), positions
    assert text.index('say "7b.') < text.index('say "8.')


def pip_lock_lines(platform_tag: str):
    """The pip lock's option lines and its pins, as the lock spells them."""
    lock = LOCK_DIR / NUMERICS["platforms"][platform_tag]["pip_lock"]
    lines = lock.read_text(encoding="utf-8").splitlines()
    return (lock, [line for line in lines if line.startswith("--")],
            [line for line in lines if PIP_PIN.match(line)])


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_the_pip_lock_parts_are_the_lock_one_pin_each_in_its_order(tmp_path, platform_tag):
    lock, options, pins = pip_lock_lines(platform_tag)
    assert set(options) == PIP_OPTIONS and pins
    parts = record.pip_lock_parts(str(lock), str(tmp_path / "parts"))
    assert len(parts) == len(pins)
    # the shell glob's order is the lock's order
    assert sorted(os.listdir(tmp_path / "parts")) == [os.path.basename(part) for part in parts]
    held = []
    for part in parts:
        lines = pathlib.Path(part).read_text(encoding="utf-8").splitlines()
        assert lines[:-1] == options, part
        held.append(lines[-1])
    assert held == pins


@pytest.mark.parametrize("change", ["a pin over two lines", "a pin without its hash",
                                    "no --require-hashes", "no pin", "an option after a pin",
                                    "the parts directory exists"])
def test_the_pip_lock_parts_refuse_a_lock_they_cannot_split_exactly(tmp_path, change):
    _lock, options, pins = pip_lock_lines("linux-64")
    lines = options + pins
    dest = tmp_path / "parts"
    if change == "a pin over two lines":
        name, digest = lines[5].split(" ", 1)
        lines[5:6] = [name + " \\", "    " + digest]
    elif change == "a pin without its hash":
        lines[5] = lines[5].split(" ", 1)[0]
    elif change == "no --require-hashes":
        lines.remove("--require-hashes")
    elif change == "no pin":
        lines = options
    elif change == "an option after a pin":
        lines.append("--extra-index-url https://example.invalid/simple")
    else:
        dest.mkdir()
    lock = tmp_path / "lock.txt"
    lock.write_text("# a comment\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(record.Refused):
        record.pip_lock_parts(str(lock), str(dest))
    assert not dest.exists() or change == "the parts directory exists"
    assert change != "the parts directory exists" or not list(dest.iterdir())


def test_linux_step_7_installs_every_pin_once_in_the_lock_order(tmp_path):
    """The install loop of the Linux script's step 7, with pip stood in for by a recorder."""
    text = SCRIPTS["linux-64"].read_text(encoding="utf-8")
    assert text.count(PIP_INSTALL["linux-64"]) == 1
    lock, _options, pins = pip_lock_lines("linux-64")
    calls = tmp_path / "calls.txt"
    program = "\n".join([
        "set -euo pipefail",
        f'LOCK_PIP="{lock}"', f'BUILD_ROOT="{tmp_path / "build"}"', 'mkdir -p "$BUILD_ROOT"',
        f'helper() {{ "{sys.executable}" -s -I "{RECORD_PATH}" "$@"; }}',
        f'py() {{ printf "%s|%s\\n" "$*" "$(tail -n 1 "${{@: -1}}")" >> "{calls}"; }}',
        PIP_INSTALL["linux-64"]])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [line.split("|", 1) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert [pin for _argv, pin in rows] == pins
    assert all(argv.startswith("-m pip install --no-deps -r ") for argv, _pin in rows)
    assert len({argv for argv, _pin in rows}) == len(pins)


def test_site_packages_reads_the_linux_python_directory_once_beside_its_link(tmp_path):
    """The Linux environment's Python ships lib/python3.1 -> python3.10: the helper reads
    python3.10 alone, so step 7b finds the wheel's one ffmpeg executable once, not twice."""
    prefix = tmp_path / "env"
    binary = (prefix / "lib" / "python3.10" / "site-packages" / "imageio_ffmpeg" / "binaries"
              / "ffmpeg-test")
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'ffmpeg version test'\n", encoding="utf-8")
    binary.chmod(0o755)
    (prefix / "bin").mkdir()
    (prefix / "lib" / "python3.1").symlink_to("python3.10")
    assert record.site_packages(str(prefix)) == [
        str(prefix / "lib" / "python3.10" / "site-packages")]
    facts = record.link_ffmpeg(str(prefix))
    assert facts["target"] == "lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-test"
    assert os.path.realpath(prefix / "bin" / "ffmpeg") == os.path.realpath(binary)


def test_the_linux_script_finds_libmeep_and_meep_once_beside_a_linked_python_directory(tmp_path):
    """Step 6's image list, run against the installed layout of a Linux build: libmeep and
    _meep with their version links, and lib/python3.1 -> python3.10."""
    text = SCRIPTS["linux-64"].read_text(encoding="utf-8")
    function = text[text.index("meep_images() {"):text.index('if [ "$REFRESH" = 1 ]; then')]
    prefix = tmp_path / "env"
    lib = prefix / "lib"
    meep = lib / "python3.10" / "site-packages" / "meep"
    meep.mkdir(parents=True)
    for name in ("libmeep.so.37.0.0", "libmeep.a", "libmeep.la"):
        (lib / name).write_bytes(b"x")
    for name in ("libmeep.so.37", "libmeep.so"):
        (lib / name).symlink_to("libmeep.so.37.0.0")
    for name in ("_meep.so.37.0.0", "_meep.a", "_meep.la"):
        (meep / name).write_bytes(b"x")
    for name in ("_meep.so.37", "_meep.so"):
        (meep / name).symlink_to("_meep.so.37.0.0")
    (lib / "python3.1").symlink_to("python3.10")
    program = "\n".join(["set -euo pipefail", f'ENV_PREFIX="{prefix}"', function, "meep_images"])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [str(lib / "libmeep.so.37.0.0"),
                                        str(meep / "_meep.so.37.0.0")]


def test_the_linux_script_reads_the_gdsii_layers_as_plain_integers(tmp_path):
    """Step 8's GDSII check against a stand-in MEEP returning what MEEP 1.33.0 returns: an
    int32 array, whose elements NumPy 2 prints as np.int32(0) inside a list."""
    numpy = pytest.importorskip("numpy")
    assert int(numpy.__version__.split(".")[0]) >= 2
    text = SCRIPTS["linux-64"].read_text(encoding="utf-8")
    eight = text[text.index('say "8.'):text.index('say "9.')].splitlines()
    read = next(line for line in eight if line.startswith('GDSII="$('))
    judge = next(line for line in eight if line.startswith('[ "$GDSII" = '))
    fake = tmp_path / "site" / "meep"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "import numpy\n"
        "def verbosity(level):\n    pass\n"
        "def get_GDSII_layers(name):\n"
        "    assert name == 'coupler.gds'\n"
        "    return numpy.array([0, 1, 2, 3, 4, 5, 31, 32], dtype=numpy.int32)\n",
        encoding="utf-8")
    (tmp_path / "corpus" / "examples").mkdir(parents=True)
    program = "\n".join([
        "set -euo pipefail", f'CORPUS_ROOT="{tmp_path / "corpus"}"',
        f'py() {{ PYTHONPATH="{tmp_path / "site"}" "{sys.executable}" -s "$@"; }}',
        read, judge, 'echo "judged: $GDSII"'])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "judged: LAYERS [0, 1, 2, 3, 4, 5, 31, 32]" in done.stdout


#: Stand-ins for step 8's first assertions on Linux. CuPy's NVRTC version call raises what
#: CuPy 13.5.1 raises when every device is hidden (CUDA_VISIBLE_DEVICES empty); ctypes
#: stands for NVRTC itself: CDLL records the library it is asked for, and nvrtcVersion
#: answers 11.8 with the status in FAKE_NVRTC_STATUS ("missing": the library is absent).
LINUX_STEP_8_STAND_INS = {
    "meep/__init__.py": "__version__ = '1.33.0'\n"
                        "def is_single_precision():\n    return True\n"
                        "def get_realnum_size():\n    return 4\n"
                        "def with_mpi():\n    return True\n",
    "meep_gpu/__init__.py": "",
    "meep_gpu/backends.py": "def subnormals_flushed():\n    return True\n",
    "numpy/__init__.py": "__version__ = '2.2.6'\n",
    "triton/__init__.py": "__version__ = '3.1.0'\n",
    "cupy/__init__.py": "__version__ = '13.5.1'\nfrom cupy import cuda\n",
    "cupy/cuda/__init__.py": "from cupy.cuda import nvrtc\n",
    "cupy/cuda/nvrtc.py": "def getVersion():\n"
                          "    raise RuntimeError('cudaErrorNoDevice: no CUDA-capable device is "
                          "detected')\n",
    "ctypes/__init__.py": "import os\n"
                          "class c_int:\n    def __init__(self, value=0):\n"
                          "        self.value = value\n"
                          "def byref(value):\n    return value\n"
                          "class CDLL:\n    def __init__(self, name):\n"
                          "        with open(os.environ['FAKE_NVRTC_ASKED'], 'a') as handle:\n"
                          "            handle.write(name + '\\n')\n"
                          "        if os.environ['FAKE_NVRTC_STATUS'] == 'missing':\n"
                          "            raise OSError(name + ': cannot open shared object file')\n"
                          "    def nvrtcVersion(self, major, minor):\n"
                          "        major.value, minor.value = 11, 8\n"
                          "        return int(os.environ['FAKE_NVRTC_STATUS'])\n",
}


@pytest.mark.parametrize("status, verdict", [("0", "11.8 ok"),
                                             ("7", "nvrtcVersion status 7 EXPECTED 11.8"),
                                             ("missing", "not loaded (")])
def test_the_linux_script_reads_nvrtcs_version_with_every_device_hidden(tmp_path, status,
                                                                       verdict):
    """Step 8's first assertions, run as the script runs them, against stand-ins: NVRTC's
    version comes from NVRTC's own call on <prefix>/lib/libnvrtc.so.11.2 (the library CuPy's
    CUDA 11 build opens), never from CuPy's call, which asks the CUDA runtime first and with
    every device hidden fails (MEASURED on the GPU host: cudaErrorNoDevice); a status other
    than 0, or no library, refuses the build by name."""
    text = SCRIPTS["linux-64"].read_text(encoding="utf-8")
    eight = text[text.index('say "8.'):text.index('say "9.')]
    block = eight[eight.index('PYTHONPATH="$REPO_ROOT" py - <<\'PYEOF\''):]
    block = block[:block.index("\nPYEOF\n") + len("\nPYEOF\n")]
    site = tmp_path / "site"
    for name, source in LINUX_STEP_8_STAND_INS.items():
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_text(source, encoding="utf-8")
    asked = tmp_path / "asked.txt"
    program = "\n".join([
        "set -euo pipefail", f'REPO_ROOT="{tmp_path / "repo"}"',
        f'py() {{ PYTHONPATH="{site}" FAKE_NVRTC_STATUS={status} FAKE_NVRTC_ASKED="{asked}" '
        f'"{sys.executable}" -s "$@"; }}',
        block, 'echo "step 8 assertions passed"'])
    # from a directory of its own, as the script runs it (from the MEEP source tree): the
    # child's first sys.path entry is its working directory
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False, cwd=tmp_path)
    output = done.stdout + done.stderr
    row = next((line for line in done.stdout.splitlines() if "NVRTC (cudatoolkit 11.8)" in line),
               None)
    assert row is not None, output
    assert " ".join(row.split()[3:]).startswith(verdict), output
    prefix = subprocess.run([sys.executable, "-s", "-c", "import sys; print(sys.prefix)"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert asked.read_text(encoding="utf-8").splitlines() == [
        os.path.join(prefix, "lib", "libnvrtc.so.11.2")], output
    assert "cudaErrorNoDevice" not in output
    if status == "0":
        assert done.returncode == 0, output
        assert "reference environment OK" in done.stdout
        assert "step 8 assertions passed" in done.stdout
    else:
        assert done.returncode == 1, output
        assert "FAILED: ['NVRTC (cudatoolkit 11.8)']" in done.stdout
        assert "step 8 assertions passed" not in done.stdout


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_step_7b_links_the_pinned_ffmpeg_and_holds_meeps_load_commands(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    seven_b = text[text.index('say "7b.'):text.index('say "7c.' if platform_tag == "linux-64"
                                                     else 'say "8.')]
    steps = ['helper link-ffmpeg "$ENV_PREFIX" "$RECORD_DIR/ffmpeg.json"',
             'helper load-commands "$RECORD_DIR/load_commands.json"',
             'helper same-load-commands "$RECORD_DIR/load_commands_installed.json" '
             '"$RECORD_DIR/load_commands.json"']
    positions = [seven_b.index(needle) for needle in steps]
    assert positions == sorted(positions)
    six = text[text.index('say "6.'):text.index('say "7.')]
    assert 'helper load-commands "$RECORD_DIR/load_commands_installed.json"' in six
    assert "MEEP_FFMPEG" in text.split("RETIRED=(", 1)[1].split(")", 1)[0]
    assert "brew" not in text and "/opt/homebrew" not in text


def run_script(platform_tag: str, env: dict):
    return subprocess.run(["bash", str(SCRIPTS[platform_tag])], capture_output=True, text=True,
                          timeout=120, check=False, env=env)


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
@pytest.mark.parametrize("knob", ["MEEP_SIGMA_PATCH", "MEEP_REFERENCE_SOLVE", "MEEP_BUILD_ROLE",
                                  "MEEP_FFMPEG"])
def test_a_retired_knob_is_refused_by_name_before_anything_runs(tmp_path, platform_tag, knob):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), knob: "1",
           "CONDA_ROOT": str(tmp_path / "conda")}
    done = run_script(platform_tag, env)
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stdout.startswith(f"REFUSING: {knob} is retired")


HOST_SCRIPT = {"Darwin/arm64": "osx-arm64", "Linux/x86_64": "linux-64"}.get(
    f"{platform.system()}/{platform.machine()}")


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_build_clears_the_calling_environment_and_plans(tmp_path, platform_tag):
    """The plan in a cleared environment (DRY_RUN): on this host's platform the caller's
    flags are dropped and the identity flags planned; on the other, step 0 refuses."""
    conda = tmp_path / "conda" / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    conda.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "DRY_RUN": "1",
           "CONDA_ROOT": str(tmp_path / "conda"), "CFLAGS": "-O3 -march=native",
           "OMPI_CXX": "clang++", "PYTHONPATH": "/nowhere", "MEEP_ENV_NAME": "meep-gpu-ref-test"}
    done = run_script(platform_tag, env)
    clearing = [line for line in done.stdout.splitlines()
                if line.startswith("reference build: clearing")]
    assert len(clearing) == 1 and " CFLAGS " in clearing[0] + " " and "OMPI_CXX" in clearing[0], \
        done.stdout
    if HOST_SCRIPT != platform_tag:
        assert done.returncode == 1 and "this script builds for" in done.stdout
        return
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PATH         /usr/bin:/bin:/usr/sbin:/sbin" in done.stdout
    assert f"prefix       {tmp_path}/conda/envs/meep-gpu-ref-test" in done.stdout
    assert "role         reference" in done.stdout
    flags = {"osx-arm64": "-O2 -ffp-contract=off -fno-fast-math",
             "linux-64": "-O2 -ffp-contract=off -fno-fast-math -march=x86-64 -mtune=generic"}
    assert f"CFLAGS       {flags[platform_tag]}" in done.stdout
    assert "DRY RUN: nothing is built" in done.stdout


# --------------------------------------------------------------------------
# The lock re-cut
# --------------------------------------------------------------------------

ADDITIONS = json.loads((LOCK_DIR / "additions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_every_addition_is_in_its_lock_and_named_where_its_role_says(platform_tag):
    spec = ADDITIONS["platforms"][platform_tag]
    numerics = NUMERICS["platforms"][platform_tag]
    reference = tool.load_reference(platform_tag, str(LOCK_DIR))
    assert (spec["conda_lock"], spec["pip_lock"]) == (numerics["conda_lock"], numerics["pip_lock"])
    for item in spec["conda"]:
        locked = reference["conda"][item["name"]]
        assert (locked["version"], locked["build"]) == (item["version"], item["build"])
        assert item["role"] in ("meep_links", "corpus_only", "tool"), item
        if item["role"] == "meep_links":
            assert item["name"] in numerics["conda"]["meep_links"]
        if item["role"] == "corpus_only":
            assert item["name"] in CORPUS_ONLY["conda"]
    for item in spec["pypi"]:
        pin = reference["pip"][tool.normalize(item["name"])]
        assert pin["version"] == item["version"] and len(pin["sha256"]) == 1
        assert item["wheel"].startswith(item["name"].replace("-", "_") + "-" + item["version"] + "-")
        assert item["role"] in ("corpus_only", "tests"), item
        if item["role"] == "corpus_only":
            assert item["name"] in CORPUS_ONLY["pypi"]


def solve_of(entries) -> dict:
    """A solve's JSON that links exactly ``entries`` (lock entries)."""
    return {"success": True, "actions": {
        "LINK": [{"name": e["name"], "version": e["version"], "build_string": e["build"]}
                 for e in entries],
        "FETCH": [{"name": e["name"], "version": e["version"], "build": e["build"],
                   "url": e["url"], "md5": e["md5"]} for e in entries]}}


def pypi_answers(directory: pathlib.Path, platform_tag: str) -> pathlib.Path:
    """PyPI's JSON for each addition, as the committed pins give it."""
    directory.mkdir(parents=True, exist_ok=True)
    pins = tool.load_reference(platform_tag, str(LOCK_DIR))["pip"]
    for item in ADDITIONS["platforms"][platform_tag]["pypi"]:
        digest = pins[tool.normalize(item["name"])]["sha256"][0]
        (directory / f"{tool.normalize(item['name'])}-{item['version']}.json").write_text(
            json.dumps({"urls": [{"filename": item["wheel"], "digests": {"sha256": digest},
                                  "yanked": False}]}), encoding="utf-8")
    return directory


@pytest.mark.parametrize("platform_tag", PLATFORMS)
def test_the_committed_locks_are_what_the_recut_makes_of_them(tmp_path, platform_tag):
    spec = ADDITIONS["platforms"][platform_tag]
    conda_text = (LOCK_DIR / spec["conda_lock"]).read_text(encoding="utf-8")
    _prose, entries = recut.read_conda_lock(conda_text)
    solve = tmp_path / "solve.json"
    solve.write_text(json.dumps(solve_of(entries)), encoding="utf-8")
    result = recut.recut(platform_tag, str(LOCK_DIR), solve_json=str(solve),
                         pypi_dir=str(pypi_answers(tmp_path / "pypi", platform_tag)))
    assert result[str(LOCK_DIR / spec["conda_lock"])] == conda_text
    assert result[str(LOCK_DIR / spec["pip_lock"])] == \
        (LOCK_DIR / spec["pip_lock"]).read_text(encoding="utf-8")
    assert [entry["name"] for entry in result["added_conda"]] == sorted(
        item["name"] for item in spec["conda"])
    assert recut.MARKER in conda_text.splitlines()


@pytest.mark.parametrize("change, words", [
    ("moved", "moved: "),
    ("dropped", "is in the lock and not in the solve"),
    ("extra", "is in the solve and is neither in the lock nor named in additions.json"),
    ("addition build", "additions.json pins"),
    ("addition missing", "is named in additions.json and not in the solve"),
    ("failed", "the solve failed"),
])
def test_a_solve_that_is_not_the_lock_plus_its_additions_is_refused_by_name(change, words):
    spec = ADDITIONS["platforms"]["osx-arm64"]
    _prose, entries = recut.read_conda_lock(
        (LOCK_DIR / spec["conda_lock"]).read_text(encoding="utf-8"))
    linked = [dict(entry) for entry in entries]
    if change == "moved":
        linked[0]["build"] = "h0_99"
    elif change == "dropped":
        linked.pop(3)
    elif change == "extra":
        linked.append({"name": "extra", "version": "1", "build": "h0_0", "md5": "f" * 32,
                       "url": recut.CHANNEL_PREFIX + "osx-arm64/extra-1-h0_0.conda"})
    elif change == "addition build":
        index = next(i for i, e in enumerate(linked) if e["name"] == "git")
        linked[index]["build"] = "pl5321h0_9"
    elif change == "addition missing":
        linked = [e for e in linked if e["name"] != "git"]
    solve = {"success": False} if change == "failed" else solve_of(linked)
    with pytest.raises(recut.Refused, match=re.escape(words)):
        recut.judge_solve(entries, spec["conda"], solve)


def test_the_solve_runs_with_an_empty_package_cache_of_its_own(monkeypatch, tmp_path):
    """The dry run lists under FETCH (URL and MD5) only what its package cache lacks: a
    caller's cache that already holds an addition must not reach the solve."""
    seen = {}

    def fake_run(argv, capture_output, text, env, timeout):
        seen["argv"], seen["env"] = argv, env
        seen["scratch"] = os.path.dirname(argv[argv.index("--file") + 1])
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"success": True}),
                                           stderr="")

    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path / "warm"))
    monkeypatch.setattr(recut.subprocess, "run", fake_run)
    assert recut.run_solve("conda", "osx-arm64", ["a==1=h0_0"],
                           {"CONDA_OVERRIDE_OSX": "11.0"}) == {"success": True}
    pkgs = seen["env"]["CONDA_PKGS_DIRS"]
    assert pkgs == os.path.join(seen["scratch"], "pkgs") and str(tmp_path) not in pkgs
    assert seen["env"]["CONDA_OVERRIDE_OSX"] == "11.0"
    assert seen["env"]["CONDA_PLUGINS_USE_SHARDED_REPODATA"] == "true"


def test_the_recut_keeps_the_prose_and_writes_the_rest():
    text = "\n".join(["# prose one", "# prose two", "#", "# platform: osx-arm64", "@EXPLICIT",
                      url("a", "1", "h0_0", "a" * 32), url("b", "2", "h1_0", "b" * 32)])
    prose, entries = recut.read_conda_lock(text)
    assert prose == ["# prose one", "# prose two"]
    added = [{"name": "c", "version": "3", "build": "h2_0", "md5": "c" * 32,
              "url": url("c", "3", "h2_0").split("#")[0]}]
    added[0]["line"] = added[0]["url"] + "#" + "c" * 32
    rendered = recut.render_conda("osx-arm64", prose, entries, added, {"CONDA_OVERRIDE_OSX": "11.0"})
    again_prose, again = recut.read_conda_lock(rendered)
    assert again_prose == prose
    assert [entry["line"] for entry in again] == [e["line"] for e in entries] + [added[0]["line"]]
    assert all(len(line) <= 92 for line in rendered.splitlines() if line.startswith("#"))


# --------------------------------------------------------------------------
# The build record helper
# --------------------------------------------------------------------------

def _source_tree(tmp_path: pathlib.Path, siblings: bool = True) -> pathlib.Path:
    """A stand-in for the extracted tarball: its python/ tree, a file beside the corpus that
    is not one, and (unless told otherwise) the corpus siblings beside python/."""
    top = tmp_path / "meep-1.33.0"
    source = top / "python"
    for relative, data in {"examples/a.py": b"a", "examples/coupler.gds": b"\x00\x06",
                           "tests/utils.py": b"u", "tests/data/x.h5": b"h"}.items():
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_bytes(data)
    (source / "meep.py").write_bytes(b"not corpus")
    (top / "tests").mkdir()
    (top / "tests" / "flux.cpp").write_bytes(b"not a sibling")
    if siblings:
        for relative in record.CORPUS_SIBLINGS:
            (top / relative).parent.mkdir(parents=True, exist_ok=True)
            (top / relative).write_bytes(b"sibling " + relative.encode())
    return source


def _sibling_digests():
    return {relative: hashlib.sha256(b"sibling " + relative.encode()).hexdigest()
            for relative in record.CORPUS_SIBLINGS}


def test_the_corpus_siblings_are_the_file_two_test_simulation_rows_read():
    assert record.CORPUS_SIBLINGS == ("tests/cyl-ellipsoid-eps-ref.h5",)


def test_the_corpus_is_kept_listed_and_verified(tmp_path):
    source = _source_tree(tmp_path)
    dest = tmp_path / "keep" / "meep-1.33.0" / "python"
    listing = tmp_path / "keep" / "corpus_sha256.txt"
    siblings = tmp_path / "keep" / "corpus_siblings_sha256.txt"
    dest.parent.mkdir(parents=True)
    kept = record.keep_corpus(str(source), str(dest), str(listing), str(siblings))
    lines = listing.read_text(encoding="utf-8").splitlines()
    assert kept["files"] == 4 == len(lines)
    assert [line.split("  ", 1)[1] for line in lines] == sorted(
        ["examples/a.py", "examples/coupler.gds", "tests/data/x.h5", "tests/utils.py"])
    assert lines[0] == hashlib.sha256(b"a").hexdigest() + "  examples/a.py"
    assert not (dest / "meep.py").exists()
    # The siblings are kept beside python/, where the rows look for them, and only they.
    assert kept["siblings"] == _sibling_digests()
    assert siblings.read_text(encoding="utf-8") == "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(_sibling_digests().items()))
    for relative in record.CORPUS_SIBLINGS:
        assert (dest.parent / relative).read_bytes() == b"sibling " + relative.encode()
    assert not (dest.parent / "tests" / "flux.cpp").exists()
    assert record.verify_corpus(str(dest), str(listing), str(siblings)) == []
    (dest / "tests" / "utils.py").write_bytes(b"changed")
    (dest / "examples" / "new.py").write_bytes(b"n")
    (sibling,) = record.CORPUS_SIBLINGS
    (dest.parent / sibling).write_bytes(b"changed")
    assert sorted(record.verify_corpus(str(dest), str(listing), str(siblings))) == [
        "examples/new.py is not in the list", f"sibling {sibling} differs",
        "tests/utils.py differs"]
    (dest.parent / sibling).unlink()
    assert f"sibling {sibling} is missing" in record.verify_corpus(
        str(dest), str(listing), str(siblings))
    siblings.write_text("", encoding="utf-8")
    assert any("not the corpus siblings" in problem for problem in record.verify_corpus(
        str(dest), str(listing), str(siblings)))
    with pytest.raises(record.Refused, match="already exists"):
        record.keep_corpus(str(source), str(dest), str(listing), str(siblings))


def test_a_tarball_without_a_corpus_sibling_is_refused_by_name(tmp_path):
    source = _source_tree(tmp_path, siblings=False)
    dest = tmp_path / "keep" / "meep-1.33.0" / "python"
    with pytest.raises(record.Refused, match="has no tests/cyl-ellipsoid-eps-ref.h5 beside"):
        record.keep_corpus(str(source), str(dest), str(tmp_path / "list.txt"),
                           str(tmp_path / "siblings.txt"))


@pytest.mark.parametrize("members, words", [
    ({"meep-1.33.0/../evil.py": b"x"}, "outside the extraction directory"),
    ({"../evil.py": b"x"}, "outside the extraction directory"),
    ({"/abs.py": b"x"}, "outside the extraction directory"),
    ({"meep-1.33.0/link": None}, "not a file or a directory"),
])
def test_the_tarball_is_unpacked_only_inside_its_directory(tmp_path, members, words):
    tarball = _sdist(tmp_path / "meep.tar.gz", members)
    with pytest.raises(record.Refused, match=words):
        record.extract(str(tarball), str(tmp_path / "out"))


def test_a_fetched_file_of_another_digest_is_refused_and_removed(tmp_path):
    source = tmp_path / "meep.tar.gz"
    source.write_bytes(b"tarball")
    out = tmp_path / "got.tar.gz"
    record.fetch(source.as_uri(), hashlib.sha256(b"tarball").hexdigest(), str(out))
    assert out.read_bytes() == b"tarball"
    with pytest.raises(record.Refused, match="not " + "0" * 64):
        record.fetch(source.as_uri(), "0" * 64, str(out))
    assert not out.exists()


def test_a_local_tarball_is_copied_in_place_of_the_download_and_judged_the_same(tmp_path):
    local = tmp_path / "local.tar.gz"
    local.write_bytes(b"tarball")
    out = tmp_path / "got.tar.gz"
    unreachable = "https://example.invalid/meep.tar.gz"
    record.fetch(unreachable, hashlib.sha256(b"tarball").hexdigest(), str(out), str(local))
    assert out.read_bytes() == b"tarball"
    with pytest.raises(record.Refused, match="not " + "0" * 64):
        record.fetch(unreachable, "0" * 64, str(out), str(local))
    assert not out.exists()


def test_load_commands_read_this_interpreter_and_refuse_what_is_not_a_binary(tmp_path):
    found = record.load_commands([os.path.realpath(sys.executable)])
    assert list(found) == [os.path.basename(os.path.realpath(sys.executable))]
    assert isinstance(found[os.path.basename(os.path.realpath(sys.executable))], list)
    text = tmp_path / "x.txt"
    text.write_text("not a binary", encoding="utf-8")
    with pytest.raises(record.Refused, match="neither a 64-bit Mach-O nor a 64-bit ELF"):
        record.load_commands([str(text)])
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps({"libmeep.so": ["libGDSII.so.0"]}), encoding="utf-8")
    after.write_text(json.dumps({"libmeep.so": ["libGDSII.so.0"]}), encoding="utf-8")
    record.same_load_commands(str(before), str(after))
    after.write_text(json.dumps({"libmeep.so": ["libGDSII.so.0", "libavcodec.so.61"]}),
                     encoding="utf-8")
    with pytest.raises(record.Refused, match="load commands of libmeep.so changed"):
        record.same_load_commands(str(before), str(after))


def test_ffmpeg_is_linked_to_the_one_executable_of_the_wheel(tmp_path):
    prefix = tmp_path / "env"
    with pytest.raises(record.Refused, match="holds 0 imageio-ffmpeg executables"):
        (prefix / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        record.link_ffmpeg(str(prefix))
    binary = link_ffmpeg_into(prefix)
    (prefix / "bin" / "ffmpeg").unlink()
    facts = record.link_ffmpeg(str(prefix))
    assert os.path.realpath(prefix / "bin" / "ffmpeg") == os.path.realpath(binary)
    assert facts["version"] == "ffmpeg version test"
    assert facts["target"] == "lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-test"
    assert facts["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    with pytest.raises(record.Refused, match="already exists"):
        record.link_ffmpeg(str(prefix))


def test_a_linked_python_directory_is_read_once(tmp_path):
    """The environment's Python ships lib/python3.1 -> python3.12: one executable, not two."""
    prefix = tmp_path / "env"
    binary = link_ffmpeg_into(prefix)
    (prefix / "bin" / "ffmpeg").unlink()
    (prefix / "lib" / "python3.1").symlink_to("python3.12")
    assert record.site_packages(str(prefix)) == [
        str(prefix / "lib" / "python3.12" / "site-packages")]
    facts = record.link_ffmpeg(str(prefix))
    assert facts["target"] == "lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-test"
    assert os.path.realpath(prefix / "bin" / "ffmpeg") == os.path.realpath(binary)


def test_the_mac_script_finds_libmeep_and_meep_once_beside_a_linked_python_directory(tmp_path):
    """Step 6's image list, run against an installed layout with lib/python3.1 -> python3.12."""
    text = SCRIPTS["osx-arm64"].read_text(encoding="utf-8")
    function = text[text.index("meep_images() {"):text.index('if [ "$REFRESH" = 1 ]; then')]
    prefix = tmp_path / "env"
    lib = prefix / "lib"
    meep = lib / "python3.12" / "site-packages" / "meep"
    meep.mkdir(parents=True)
    for name in ("libmeep.37.dylib", "libmeep.a", "libmeep.la"):
        (lib / name).write_bytes(b"x")
    (lib / "libmeep.dylib").symlink_to("libmeep.37.dylib")
    for name in ("_meep.37.so", "_meep.a", "_meep.la"):
        (meep / name).write_bytes(b"x")
    (meep / "_meep.so").symlink_to("_meep.37.so")
    (lib / "python3.1").symlink_to("python3.12")
    program = "\n".join(["set -euo pipefail", f'ENV_PREFIX="{prefix}"', function, "meep_images"])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [str(lib / "libmeep.37.dylib"), str(meep / "_meep.37.so")]


def test_the_mac_script_reads_the_gdsii_layers_as_plain_integers(tmp_path):
    """Step 8's GDSII check against a stand-in MEEP returning what MEEP 1.33.0 returns: an
    int32 array, whose elements NumPy 2 prints as np.int32(0) inside a list."""
    numpy = pytest.importorskip("numpy")
    assert int(numpy.__version__.split(".")[0]) >= 2
    text = SCRIPTS["osx-arm64"].read_text(encoding="utf-8")
    eight = text[text.index('say "8.'):text.index('say "9.')].splitlines()
    read = next(line for line in eight if line.startswith('GDSII="$('))
    judge = next(line for line in eight if line.startswith('[ "$GDSII" = '))
    fake = tmp_path / "site" / "meep"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "import numpy\n"
        "def verbosity(level):\n    pass\n"
        "def get_GDSII_layers(name):\n"
        "    assert name == 'coupler.gds'\n"
        "    return numpy.array([0, 1, 2, 3, 4, 5, 31, 32], dtype=numpy.int32)\n",
        encoding="utf-8")
    (tmp_path / "corpus" / "examples").mkdir(parents=True)
    program = "\n".join([
        "set -euo pipefail", f'CORPUS_ROOT="{tmp_path / "corpus"}"',
        f'py() {{ PYTHONPATH="{tmp_path / "site"}" "{sys.executable}" -s "$@"; }}',
        read, judge, 'echo "judged: $GDSII"'])
    done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "judged: LAYERS [0, 1, 2, 3, 4, 5, 31, 32]" in done.stdout


def test_an_ffmpeg_that_loads_a_library_outside_the_system_is_refused(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    link_ffmpeg_into(prefix)
    (prefix / "bin" / "ffmpeg").unlink()
    monkeypatch.setattr(record, "macho_dylibs", lambda path: [
        "/usr/lib/libSystem.B.dylib", "/opt/elsewhere/lib/libavcodec.61.dylib"])
    with pytest.raises(record.Refused, match="outside the operating system: "
                                             r"\['/opt/elsewhere/lib/libavcodec.61.dylib'\]"):
        record.link_ffmpeg(str(prefix))
    assert not os.path.lexists(prefix / "bin" / "ffmpeg")     # refused before it links
    monkeypatch.setattr(record, "macho_dylibs", lambda path: [
        "/usr/lib/libSystem.B.dylib", "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"])
    assert record.link_ffmpeg(str(prefix))["links"][0] == "/usr/lib/libSystem.B.dylib"


@pytest.mark.parametrize("platform_tag", sorted(record.TOOLS))
def test_the_environments_programs_must_come_from_its_bin(tmp_path, platform_tag):
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    for name in record.TOOLS[platform_tag]["prefix"]:
        (bin_dir / name).write_text("#!/bin/sh\n", encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    found = record.find_tools(str(tmp_path / "env"), platform_tag)
    assert set(found["prefix"].values()) == {
        str(bin_dir / name) for name in record.TOOLS[platform_tag]["prefix"]}
    assert set(found["host"]) == set(record.TOOLS[platform_tag]["host"])
    (bin_dir / "git").unlink()
    with pytest.raises(record.Refused, match=r"git \("):
        record.find_tools(str(tmp_path / "env"), platform_tag)


def test_the_manifest_names_what_the_record_directory_holds(tmp_path):
    record_dir = tmp_path / "share" / "meep-gpu-build"
    record_dir.mkdir(parents=True)
    listing = record_dir / "corpus_sha256.txt"
    listing.write_text("0" * 64 + "  examples/a.py\n", encoding="utf-8")
    siblings = record_dir / "corpus_siblings_sha256.txt"
    siblings.write_text("1" * 64 + "  tests/cyl-ellipsoid-eps-ref.h5\n", encoding="utf-8")
    (record_dir / "identity_keep.txt").write_text("...\nMEEP identity reference OK\n",
                                                   encoding="utf-8")
    (record_dir / "identity_keep.status").write_text("0\n", encoding="utf-8")
    (record_dir / "identity_keep.outcome").write_text("pass\n", encoding="utf-8")
    (record_dir / "build_record.json").write_text(json.dumps({"sha256": {"libmeep": "1" * 64}}),
                                                  encoding="utf-8")
    (record_dir / "reference_environment.json").write_text(
        json.dumps({"kind": "matches", "verdict": "matches the certification reference"}),
        encoding="utf-8")
    (record_dir / "meep-gpu-ref-osx-arm64.conda.txt").write_text("x", encoding="utf-8")
    arguments = types.SimpleNamespace(
        platform="osx-arm64", role="reference", tarball_url="https://example.invalid/m.tgz",
        tarball_sha256="2" * 64, cflags="-O2", cxxflags="-O2", configure="./configure",
        corpus_root="share/meep-gpu-build/meep-1.33.0/python", corpus_list=str(listing),
        corpus_siblings=str(siblings))
    document = record.manifest(str(record_dir), arguments)
    assert document["schema"] == "meep-gpu/reference-build/1"
    assert document["corpus"]["files"] == 1
    assert document["corpus"]["list"] == "corpus_sha256.txt"
    assert document["corpus"]["siblings"] == [{"path": "tests/cyl-ellipsoid-eps-ref.h5",
                                               "sha256": "1" * 64}]
    assert document["identity"]["keep"] == {"exit_status": 0,
                                            "verdict": ["MEEP identity reference OK"],
                                            "outcome": "pass"}
    assert document["identity"]["flush"]["outcome"] == "not run"
    assert document["images"] == {"libmeep": "1" * 64}
    assert document["reference_environment"]["kind"] == "matches"
    assert list(document["locks_sha256"]) == ["meep-gpu-ref-osx-arm64.conda.txt"]


# --------------------------------------------------------------------------
# The corpus refresh of an existing prefix
# --------------------------------------------------------------------------

TARBALL = "2" * 64
CORPUS_ROOT_RELATIVE = "share/meep-gpu-build/meep-1.33.0/python"


def _old_prefix(tmp_path: pathlib.Path):
    """A prefix as a build kept it before the siblings: python/ and its list, and a record
    written by the build's own writer, whose corpus entry names no siblings."""
    source = _source_tree(tmp_path / "tarball")
    record_dir = tmp_path / "env" / "share" / "meep-gpu-build"
    root = record_dir / "meep-1.33.0" / "python"
    kept = {}
    for relative in record.corpus_files(str(source)):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((source / relative).read_bytes())
        kept[relative] = record.sha256_of(str(root / relative))
    listing = record_dir / "corpus_sha256.txt"
    digest = record.write_listing(str(listing), kept)
    record.write_json(str(record_dir / "reference_build.json"), {
        "schema": "meep-gpu/reference-build/1", "role": "reference",
        "meep": {"version": "1.33.0", "tarball_sha256": TARBALL, "cflags": "-O2"},
        "images": {"libmeep": "1" * 64}, "identity": {"keep": {"outcome": "pass"}},
        "corpus": {"root": CORPUS_ROOT_RELATIVE, "list": "corpus_sha256.txt",
                   "list_sha256": digest, "files": len(kept)}})
    return source, record_dir, root


def _refresh(source, record_dir, staging, tarball=TARBALL):
    return record.refresh_corpus(str(source), str(record_dir), str(staging), tarball,
                                 CORPUS_ROOT_RELATIVE, str(record_dir / "corpus_sha256.txt"),
                                 str(record_dir / "corpus_siblings_sha256.txt"))


def test_a_refresh_adds_the_siblings_and_rewrites_only_the_records_corpus_entry(tmp_path):
    source, record_dir, root = _old_prefix(tmp_path)
    path = record_dir / "reference_build.json"
    before_text = path.read_text(encoding="utf-8")
    before = json.loads(before_text)
    kept_before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    summary = _refresh(source, record_dir, tmp_path / "staging1")
    (sibling,) = record.CORPUS_SIBLINGS
    assert (root.parent / sibling).read_bytes() == b"sibling " + sibling.encode()
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == kept_before
    after_text = path.read_text(encoding="utf-8")
    after = json.loads(after_text)
    assert {k: v for k, v in after.items() if k != "corpus"} == \
        {k: v for k, v in before.items() if k != "corpus"}
    assert after_text == record.render_json(after)
    assert after["corpus"] == {**before["corpus"], "siblings": [
        {"path": sibling, "sha256": _sibling_digests()[sibling]}]}
    assert summary["added"] == [f"../{sibling}"]
    assert summary["record_sha256_before"] == hashlib.sha256(before_text.encode()).hexdigest()
    assert summary["record_sha256_after"] == hashlib.sha256(after_text.encode()).hexdigest()
    assert json.loads((record_dir / record.REFRESH_LOG).read_text()) == summary
    assert record.verify_corpus(str(root), str(record_dir / "corpus_sha256.txt"),
                                str(record_dir / "corpus_siblings_sha256.txt")) == []
    # A second refresh finds nothing to add and leaves the record's bytes as they are; the
    # command line is the same function.
    assert record.main(["refresh-corpus", str(source), str(record_dir),
                        str(tmp_path / "staging2"), "--tarball-sha256", TARBALL,
                        "--corpus-root", CORPUS_ROOT_RELATIVE,
                        "--corpus-list", str(record_dir / "corpus_sha256.txt"),
                        "--corpus-siblings", str(record_dir / "corpus_siblings_sha256.txt")]) == 0
    again = json.loads((record_dir / record.REFRESH_LOG).read_text())
    assert again["added"] == [] and again["record_sha256_before"] == again["record_sha256_after"]
    assert path.read_text(encoding="utf-8") == after_text


@pytest.mark.parametrize("change, words", [
    ("tarball", "built from the tarball"),
    ("format", "not as the build wrote it"),
    ("root", "names the corpus root"),
    ("changed", r"1 file\(s\) with other bytes \(examples/a.py\)"),
    ("extra", r"1 file\(s\) the tarball does not hold \(tests/stray.py\)"),
    ("sibling", r"with other bytes \(\.\./tests/cyl-ellipsoid-eps-ref.h5\)"),
    ("record", "no reference build record"),
])
def test_a_refresh_refuses_what_it_cannot_add_to_and_writes_nothing(tmp_path, change, words):
    source, record_dir, root = _old_prefix(tmp_path)
    path = record_dir / "reference_build.json"
    tarball = TARBALL
    if change == "tarball":
        tarball = "3" * 64
    elif change == "format":
        path.write_text(json.dumps(json.loads(path.read_text())), encoding="utf-8")
    elif change == "root":
        document = json.loads(path.read_text())
        document["corpus"]["root"] = "share/elsewhere/python"
        record.write_json(str(path), document)
    elif change == "changed":
        (root / "examples" / "a.py").write_bytes(b"edited")
    elif change == "extra":
        (root / "tests" / "stray.py").write_bytes(b"")
    elif change == "sibling":
        (root.parent / record.CORPUS_SIBLINGS[0]).parent.mkdir(parents=True)
        (root.parent / record.CORPUS_SIBLINGS[0]).write_bytes(b"other")
    else:
        path.unlink()
    record_text = path.read_text(encoding="utf-8") if path.exists() else None
    listing_text = (record_dir / "corpus_sha256.txt").read_text(encoding="utf-8")
    with pytest.raises(record.Refused, match=words):
        _refresh(source, record_dir, tmp_path / "staging", tarball)
    assert (path.read_text(encoding="utf-8") if path.exists() else None) == record_text
    assert (record_dir / "corpus_sha256.txt").read_text(encoding="utf-8") == listing_text
    assert not (record_dir / "corpus_siblings_sha256.txt").exists()
    assert not (record_dir / record.REFRESH_LOG).exists()
    if change != "sibling":
        assert not (root.parent / record.CORPUS_SIBLINGS[0]).exists()


def refresh_block(platform_tag: str) -> str:
    """The build script's corpus refresh, from its ``if [ "$REFRESH" = 1 ]`` to step 0."""
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    return text[text.index('if [ "$REFRESH" = 1 ]; then'):text.index('say "0.')]


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_refresh_runs_before_the_build_and_touches_no_meep_lock_or_package(platform_tag):
    text = SCRIPTS[platform_tag].read_text(encoding="utf-8")
    block = refresh_block(platform_tag)
    assert "--refresh-corpus) REFRESH=1; shift ;;" in text
    assert text.index('if [ "$REFRESH" = 1 ]; then') < text.index('say "0.')
    order = ["fetch_tarball ", "helper extract ", "helper refresh-corpus ",
             "helper verify-corpus "]
    assert [block.index(step) for step in order] == sorted(block.index(step) for step in order)
    assert block.rstrip().endswith("exit 0\nfi")
    assert block.index("DRY RUN: nothing is refreshed") < block.index("fetch_tarball ")
    assert 'helper verify-corpus "$CORPUS_ROOT" "$CORPUS_LIST" "$CORPUS_SIBLINGS"' in block
    assert '--tarball-sha256 "$MEEP_SHA256"' in block
    for word in ("conda", "make", "pip", "configure", "LOCK_", "autoreconf", "$ENV_PREFIX/lib"):
        assert word not in block, word
    assert "MEEP_TARBALL" in text[text.index("PASS_THROUGH=("):text.index("RETIRED=(")]
    fetch = text[text.index("fetch_tarball() {"):text.index("\n}\n", text.index("fetch_tarball() {"))]
    assert '--from "$MEEP_TARBALL"' in fetch and '"$MEEP_SHA256"' in fetch


@pytest.mark.parametrize("platform_tag", sorted(SCRIPTS))
def test_the_refresh_plans_on_a_built_prefix_and_refuses_another(tmp_path, platform_tag):
    prefix = tmp_path / "env"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "DRY_RUN": "1",
           "MEEP_ENV_PREFIX": str(prefix), "CONDA_ROOT": str(tmp_path / "conda")}

    def run():
        return subprocess.run(["bash", str(SCRIPTS[platform_tag]), "--refresh-corpus"],
                              capture_output=True, text=True, timeout=120, check=False, env=env)

    done = run()
    if HOST_SCRIPT != platform_tag:
        assert done.returncode == 1 and "this script builds for" in done.stdout
        return
    assert done.returncode == 1 and "is not a reference environment" in done.stdout
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (prefix / "bin" / "python").chmod(0o755)
    done = run()   # an interpreter, but no record of a build
    assert done.returncode == 1 and "is not a reference environment" in done.stdout
    (prefix / "share" / "meep-gpu-build").mkdir(parents=True)
    (prefix / "share" / "meep-gpu-build" / "reference_build.json").write_text("{}\n")
    done = run()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "R. refresh the identity corpus" in done.stdout
    assert "DRY RUN: nothing is refreshed" in done.stdout and "=== 0." not in done.stdout

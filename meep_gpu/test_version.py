"""``meep_gpu.__version__``: the version the installer recorded, read on first use.

The number is read from the installed package metadata of the distribution that
installed this copy of the package. A copy that was not installed that way (a
checkout on ``PYTHONPATH``, or an editable install, whose metadata lists no
package files) reads the ``version`` of the ``pyproject.toml`` beside the
package. These tests hold in both situations and state which one they saw.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import meep_gpu

PACKAGE = Path(meep_gpu.__file__).resolve()
#: PEP 440 release segment, with an optional pre-, post- or local part.
VERSION = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[A-Za-z0-9.]+)?")


def _installing_distribution():
    """The distribution whose files include this ``meep_gpu/__init__.py``, or None."""
    for name in metadata.packages_distributions().get("meep_gpu", []):
        distribution = metadata.distribution(name)
        for file in distribution.files or ():
            if file.name == "__init__.py" and file.parent.name == "meep_gpu" \
                    and Path(distribution.locate_file(file)).resolve() == PACKAGE:
                return distribution
    return None


def _pyproject_version():
    project = PACKAGE.parent.parent / "pyproject.toml"
    if not project.is_file():
        return None
    match = re.search(r'(?ms)^\[project\].*?^version\s*=\s*"([^"]+)"',
                      project.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def test_the_version_is_a_release_number():
    assert isinstance(meep_gpu.__version__, str)
    assert VERSION.fullmatch(meep_gpu.__version__), meep_gpu.__version__


def test_the_version_is_the_one_the_installer_recorded_or_the_checkouts():
    distribution = _installing_distribution()
    if distribution is not None:
        assert meep_gpu.__version__ == distribution.version, (
            f"installed by {distribution.metadata['Name']} {distribution.version}, "
            f"but __version__ reads {meep_gpu.__version__}")
    else:
        expected = _pyproject_version()
        assert expected is not None, (
            f"{PACKAGE} was not installed by a distribution that lists it, and no "
            f"pyproject.toml with a [project] version is beside the package")
        assert meep_gpu.__version__ == expected


def test_the_version_is_not_read_at_import():
    """``import meep_gpu`` does not walk the installed metadata; the first read does."""
    probe = ("import sys, importlib.metadata as m\n"
             "calls = []\n"
             "original = m.packages_distributions\n"
             "m.packages_distributions = lambda: (calls.append(1), original())[1]\n"
             "import meep_gpu\n"
             "before = len(calls)\n"
             "meep_gpu.__version__\n"
             "print(before, len(calls))\n")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          cwd=str(PACKAGE.parent.parent), timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    before, after = (int(word) for word in done.stdout.split()[-2:])
    assert (before, after) == (0, 1)


def test_an_unknown_attribute_still_raises_attribute_error():
    try:
        meep_gpu.no_such_attribute  # noqa: B018
    except AttributeError as error:
        assert "no_such_attribute" in str(error)
    else:
        raise AssertionError("meep_gpu.no_such_attribute did not raise AttributeError")

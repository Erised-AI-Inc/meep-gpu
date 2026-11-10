"""The neutral labels a certification record carries for the machine a gate ran on.

A gate stamps its machine as the machine's anonymous id (``neutral_ids.machine_id``),
and its paths under their labels (``neutral_ids.py``): a run's artifact leaves the
machine with its results, so it names no machine, account or local path. A ledger or a
certification record is shipped, and dispatch reads it at run time; every tool that
copies a gate's machine stamp or environment into a ledger writes it through THIS module
(the stamp becomes the one label below, whatever spelling a run wrote, an older run's
raw host name included), and ``tools/ci/check_hygiene.py`` fails a ledger that still
holds a machine stamp.

THE SPELLINGS are the ones 0.9.2 shipped, so a record cut by a tool and a record the
0.9.2 export neutralised read alike:

* :data:`GPU_HOST`, ``the GPU host``: every NVIDIA host field, alone
  (``certification.json`` blocks; ``test_certification_metadata`` reads it as "the one
  machine the artifact stamped"), before a device line (:func:`host_line`, the Triton
  welds), or before where the stamp was read (:func:`machine_stamped`, the CUDA welds).
* :data:`STAGING_ROOT`, ``<staging-root>``: the prefix of a recorded path under the
  directory a certification round was staged in (:func:`neutral_environment`), and of a run directory
  a record wrote home-relative under the staging root (:func:`neutral_home_relative`).

What is still checked. A tool still reads the stamped machine ids and compares them: two
legs that stamp different machines, or a device stamp that names another machine than
its artifact, are refused as before (one machine gives one id). Only what is WRITTEN is the label, and a run that
stamps no machine at all still writes none: the label stands for one measured machine.

Standard library only: the bind tools import it before anything else.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

GPU_HOST = "the GPU host"
STAGING_ROOT = "<staging-root>"
#: The staging root the drivers default to under a home directory
#: (``MGPU_SITE_STAGING_ROOT``, ``parity/meep_gpu/README.md``).
DEFAULT_HOME_STAGING_ROOT = "meep_gpu_validation"

#: The roots under which a path names an account or a machine's temporary space, from
#: neutral pieces (the guard's own rule, ``tools/ci/check_hygiene.py``).
_HOME_ROOTS = ("ho" + "me", "Us" + "ers")
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9._~-])/(?:" + "|".join(_HOME_ROOTS)
                         + r")/(?!user(?:/|$))[A-Za-z0-9._-]+"
                         + r"|(?<![A-Za-z0-9._~-])/priv" + r"ate/(?:var|tmp)/[A-Za-z0-9._-]")


#: A home-relative path, ``~/<directory>...``: where a run lay on the machine that ran it.
_HOME_RELATIVE = re.compile(r"(?<![A-Za-z0-9._$/~-])~/(?P<root>[A-Za-z0-9._-]+)(?=/|$|[\s)\],;:])")


class LabelRefusal(ValueError):
    """A value cannot be written neutrally; the record is not written."""


def host_label(hostname: Optional[str]) -> Optional[str]:
    """:data:`GPU_HOST` for a machine the run stamped, ``None`` when it stamped none."""
    if hostname is None or not str(hostname).strip():
        return None
    return GPU_HOST


def host_line(details: str) -> str:
    """``the GPU host: <details>``, the Triton weld's host line."""
    return f"{GPU_HOST}: {details}"


def machine_stamped(source: str) -> str:
    """``the GPU host (machine-stamped: <source>)``: the CUDA weld's host, read from a leg."""
    return f"{GPU_HOST} (machine-stamped: {source})"


def declared_not_stamped(source: str) -> str:
    """The CUDA weld's host when no leg stamps one and a record declares it."""
    return f"{GPU_HOST} (NOT machine-stamped by this gate; from {source})"


def one_host(hostnames: Iterable[str]) -> str:
    """The label for a set of stamped host names that must name exactly one machine."""
    named = sorted({str(name) for name in hostnames if name and str(name).strip()})
    if len(named) != 1:
        raise LabelRefusal(f"the legs stamp {len(named)} machines; a record names one GPU "
                           f"host, so a campaign that ran on several is not describable")
    return GPU_HOST


def local_paths(text: str) -> List[str]:
    """The local-path prefixes ``text`` still holds (home roots, temporary roots)."""
    return [match.group() for match in _LOCAL_PATH.finditer(text)]


def common_root(paths: Sequence[str]) -> Optional[str]:
    """The deepest directory every one of ``paths`` lies under, when that directory is
    itself under a home root: the staging root of a certification round whose caches, stub and clone a
    driver laid out side by side. ``None`` for fewer than two paths or no shared root."""
    absolute = [os.path.normpath(p) for p in paths if p.startswith("/")]
    if len(absolute) < 2:
        return None
    shared = os.path.commonpath(absolute)
    if shared in absolute:
        # A root that is itself one of the values would write that value as the bare
        # placeholder; the root is the directory it lies in.
        shared = os.path.dirname(shared)
    return shared if local_paths(shared) else None


def neutral_path(value: str, staging_root: Optional[str]) -> str:
    """``value`` with ``staging_root`` (when it is the value's prefix) as :data:`STAGING_ROOT`."""
    if not staging_root:
        return value
    root = staging_root.rstrip("/")
    if value == root:
        return STAGING_ROOT
    if value.startswith(root + "/"):
        return STAGING_ROOT + value[len(root):]
    return value


def home_relative_paths(text: str) -> List[str]:
    """The home-relative paths ``text`` still holds, by their first directory."""
    return [match.group() for match in _HOME_RELATIVE.finditer(text)]


def neutral_home_relative(text: str, root: str = DEFAULT_HOME_STAGING_ROOT) -> str:
    """``text`` with every home-relative path under ``~/<root>`` written under
    :data:`STAGING_ROOT`: ``~/<root>/<campaign>/x`` becomes ``<staging-root>/<campaign>/x``.
    A home-relative path under any other directory is left for the caller to refuse."""
    return _HOME_RELATIVE.sub(
        lambda match: STAGING_ROOT if match.group("root") == root else match.group(), text)


def neutral_environment(environment: Mapping[str, object], staging_root: Optional[str] = None
                        ) -> Tuple[Dict[str, object], Optional[str], List[str]]:
    """``(environment, root, residue)``: every string value with the staging root as
    :data:`STAGING_ROOT`, the root used, and the variables that still name a local path.

    The root is ``staging_root`` when given, else :func:`common_root` of the values that
    name a local path. A non-empty residue is a refusal for the caller to report: it
    means a recorded path lies outside the root, and writing it would ship it.
    """
    strings = [str(value) for value in environment.values() if isinstance(value, str)]
    root = staging_root or common_root([value for value in strings if local_paths(value)])
    out: Dict[str, object] = {}
    residue: List[str] = []
    for name, value in environment.items():
        if isinstance(value, str):
            value = neutral_path(value, root)
            if local_paths(value):
                residue.append(name)
        out[name] = value
    return out, root, sorted(residue)


def neutral_text(text: str, hostnames: Iterable[str]) -> str:
    """``text`` with every whole-word occurrence of a stamped host name (and of its first
    label, the name before its first dot) as :data:`GPU_HOST`."""
    names = set()
    for name in hostnames:
        name = str(name or "").strip()
        if name:
            names.add(name)
            names.add(name.split(".")[0])
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9_-])" + re.escape(name) + r"(?![A-Za-z0-9_-])",
                      GPU_HOST, text)
    return text

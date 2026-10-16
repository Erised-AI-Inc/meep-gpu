#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tier A's cases as one ``--cases-file``: the four probe cases and the family rows.

    MEEP_IDENTITY_CORPUS_MANIFEST=FILE MGPU_SITE_MEEP_SOURCE=DIR \\
        python parity/meep_gpu/record_meep_identity.py --build-record PATH --work DIR \\
            --cases-file parity/meep_gpu/meep_identity_tier_a.py [--mode keep|flush]

WHY ONE MODULE. ``record_meep_identity.py`` mints a mode's whole set in one run and
replaces that mode's fixtures directory with it, and it reads its cases from one
``--cases-file``. Tier A's set is the four probe cases of ``meep_identity_smoke`` (a 2-D
vacuum and an epsilon-block PML case, a 3-D case with two mirrors at two source widths)
together with one small row of Table 2's corpus per family the array path transcribes
(``FAMILY_ROWS``), so the four and the rows are minted together through this module.

HOW A LABEL IS BUILT. A probe label is built by ``meep_identity_smoke.build_case`` and
stepped to the runner's own step rule (``sweep_corpus_lift_parity._timing`` at 200 steps
and a 12,000-step ceiling, the rule its stock cases use), so its N is the N the probe
cases have always been minted at (200, and 375 for ``c``). Every other label is a corpus
row: ``meep_identity_corpus.build``, its Table 2 step count and its grid check, which need
``MEEP_IDENTITY_CORPUS_MANIFEST`` and ``MGPU_SITE_MEEP_SOURCE`` at mint time only (the
replay reads the capsule and needs neither). Through a cases file the runner does not
check the digest smoke's record in process; the minting tool runs
``meep_identity_smoke.py`` itself before a keep mint from a cases file, so the reference
MEEP is still judged by it.

THE FAMILY ROWS were chosen from the corpus Tier C run of the 0.10 freeze, among the 91
rows that hold and whose capsule the schema carries (95 more hold but carry DFT monitors,
which capsule schema 1 refuses): per family, the row with the fewest fixture bytes whose
capsule, loaded on the NumPy reference with its recorded currents, carries the family
(complex storage, a Bloch wavevector, an off-diagonal epsilon, a cylindrical grid, a
conductivity, a susceptibility) and whose replay takes a few seconds (the conductive
family's smallest row, a 3-D ``test_dump_load`` case, replays in 35 s; ``ex.absorber-1d``
in 2). The keep set (the four probe cases and these rows) is held near half the 5 MB
budget (``meep_identity.FIXTURE_BUDGET_BYTES``) so that an equal flush set fits beside it,
which leaves out the one integrated-source row (329 KB). None of the 91 carries a
nonlinearity or a special-kz beta; those families stay with their kernel gates and Tier C.
A flush set is not minted: the only flush reference is x86 MEEP at one thread, and the
minting tool's MEEP-blocked replay cannot drive an x86 FPU to flush
(``docs/development/testing.md``, "Tier A's cases").
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parity.meep_gpu import meep_identity_corpus as corpus  # noqa: E402
from parity.meep_gpu import meep_identity_smoke as smoke  # noqa: E402

#: One corpus row per family, by label (Table 2's labels; see the module docstring), and
#: the families its capsule carries, read off the loaded driver.
FAMILY_ROWS: Dict[str, str] = {
    "ex.material-dispersion": "dispersive: one Lorentzian susceptibility (0-D)",
    "t.test_holey_wvg_bands.TestHoleyWvgBands.test_fields_at_kx":
        "complex storage under a Bloch wavevector, an off-diagonal epsilon, PML and a mirror (2-D)",
    "t.test_ring_cyl.TestRingCyl.test_ring_cyl": "cylindrical at m = 3: complex storage and PML",
    "ex.absorber-1d": "conductive: an absorber's conductivity beside a dispersive medium (1-D)",
}

CASES = tuple(smoke.CASES) + tuple(FAMILY_ROWS)

_CURRENT: Dict[str, Any] = {}


def build(mp: Any, label: str) -> Any:
    """The case's ``mp.Simulation``: a probe case's own builder, or the corpus row's."""
    _CURRENT.clear()
    _CURRENT["label"] = label
    _CURRENT["probe"] = label in smoke.CASES
    if _CURRENT["probe"]:
        return smoke.build_case(mp, label)
    return corpus.build(mp, label)


def steps_planned(mp: Any, sim: Any) -> int:
    """N: the runner's stock rule for a probe case, Table 2's step count for a row."""
    if _CURRENT.get("probe"):
        from parity.meep_gpu.sweep_corpus_lift_parity import _timing  # noqa: PLC0415

        return int(_timing(mp, sim, float(sim.resolution), 200, 12000)["steps_planned"])
    return corpus.steps_planned(mp, sim)


def after_lift(mp: Any, sim: Any, driver: Any) -> Dict[str, Any]:
    """A row's grid check (the corpus module's); a probe case records its label and cells."""
    if _CURRENT.get("probe"):
        return {"label": _CURRENT["label"],
                "grid_cells": int(math.prod(int(value) for value in driver.shape))}
    return corpus.after_lift(mp, sim, driver)

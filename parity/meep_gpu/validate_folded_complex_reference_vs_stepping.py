"""Pin the FOLDED COMPLEX gate's REFERENCE against ``stepping.py``, on NumPy.

The twin of ``validate_fold_reference_vs_stepping.py`` (which belongs to the
symmetry tranche and is not edited here), for the fold under complex64 storage
and under ``grid.beta``.

The device gate (``gate_triton_folded_complex.py``) compares the KERNEL against a
reference transcribed by hand from ``stepping.py``; this compares that REFERENCE
against the array path itself, which is what stops "bit-identical" from meaning
"the kernel reproduces whatever I wrote twice". It must pass on the laptop BEFORE
anything is staged for a device run.

It runs anywhere: the gate module imports ``cupy`` defensively and never needs a
device for its reference half, so nothing here stands in for CuPy. It must NOT —
a module that puts NumPy in ``sys.modules["cupy"]`` poisons every module imported
after it in the same process (``cp is not None`` then selects device paths, and
``cp.asnumpy`` does not exist), which is a coupling this file is imported into a
test process by ``test_triton_folded_complex.py`` and would spread there.

    python -u validate_folded_complex_reference_vs_stepping.py

TWO THINGS ARE PINNED, and the second is the one the whole tranche turns on:

1. the folded complex CURL sub-steps (``step_B`` / ``step_D``), with and without
   beta, in complex64 and — for the beta family — in real float32;
2. the folded complex GHOST FILL, against ``stepping.fill_symmetry_bc_*`` plus
   ``fill_folded_far_ghosts_*``. The array path spells the parity as
   ``phase * plane`` with a PYTHON INT, which on complex64 is a FULL complex
   product; the reference must reproduce that byte for byte, including on the
   engineered signed-zero rows a plane-wise sign flip would get wrong.

The configuration list is chosen for what each row is the ONLY witness to: both
terminations (a folded PERIODIC axis, a folded METALLIC one), both full-count
parities (the reflect row is ``stored - 2`` at an even count and ``stored - 3``
at an odd one), both plane phases, a fold on each of X, Y and Z, two planes at
once with MIXED count parity, an in-plane Bloch phase on an UNFOLDED axis beside
its k = 0 control, and the beta family in both storages. Exit status is the
verdict.

WHAT "BIT-IDENTICAL" MEANS HERE — see :func:`compare_words`. Every finite word,
every zero and every subnormal is compared RAW, by its uint32 bits, unchanged and
still the whole claim. NaN words are the one exemption, and only NaN-against-NaN:
IEEE-754 does not specify a NaN's sign bit or payload, so demanding they agree
asserts something no implementation owes us. A NaN where the other side has a
finite word, an infinity, or a zero is still a FAILURE, and the NaN census rides
in the record so a run that starts producing NaNs cannot hide inside the
exemption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

API = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".."))
if API not in sys.path:
    sys.path.insert(0, API)
_HERE = os.path.join(API, "parity", "meep_gpu")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gate_triton_folded_complex as gate  # noqa: E402

from meep_gpu import stepping                       # noqa: E402
from meep_gpu.fields import Fields                  # noqa: E402
from meep_gpu.grid import Grid, Mirror              # noqa: E402
from meep_gpu.pml import PML                        # noqa: E402
from meep_gpu.triton_kernels import folded_complex  # noqa: E402

NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "By", "Bz", "Dx", "Dy", "Dz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")

BETA_GRATING_13_2 = -0.6850526103319672
BETA_EIGSRC = 0.2

#: (label, folded axes, outer declaration, cell, phase, complex, beta, kx).
#: Every row is here because it is the only witness to something.
CASES: Tuple[Tuple[Any, ...], ...] = (
    ("Y/periodic/+1/even", "Y", "periodic", (1.6, 2.0, 0.0), 1, True, 0.0, 0.0),
    ("Y/periodic/+1/ODD", "Y", "periodic", (1.6, 2.1, 0.0), 1, True, 0.0, 0.0),
    ("Y/periodic/-1/even", "Y", "periodic", (1.6, 2.0, 0.0), -1, True, 0.0, 0.0),
    ("Y/periodic/-1/ODD", "Y", "periodic", (1.6, 2.1, 0.0), -1, True, 0.0, 0.0),
    ("Y/metallic/+1/even", "Y", "metallic", (1.6, 2.0, 0.0), 1, True, 0.0, 0.0),
    ("Y/metallic/+1/ODD", "Y", "metallic", (1.6, 2.1, 0.0), 1, True, 0.0, 0.0),
    ("Y/metallic/-1/ODD", "Y", "metallic", (1.6, 2.1, 0.0), -1, True, 0.0, 0.0),
    ("X/periodic/+1/ODD", "X", "periodic", (2.1, 1.6, 0.0), 1, True, 0.0, 0.0),
    ("Z/periodic/-1/even", "Z", "periodic", (1.2, 1.2, 2.0), -1, True, 0.0, 0.0),
    ("XY/periodic/+1/MIXED", "XY", "periodic", (2.0, 2.1, 1.2), 1, True, 0.0, 0.0),
    # MIXED PHASE, not merely mixed count parity: the two planes carry DIFFERENT
    # coefficients, which is the only configuration in which the fill's
    # composition order is byte-visible at all (measured 8/128 diverging words at
    # (+1,-1) against 0/128 at (+1,+1) and (-1,-1)). The device gate's
    # `reverse_axis_order` leg runs on this grid, so the reference must be pinned
    # on it here first.
    ("XY/periodic/+1,-1/MIXED-PHASE", "XY", "periodic", (2.0, 2.1, 1.2), (1, -1),
     True, 0.0, 0.0),
    ("Y/periodic/+1/even+kx", "Y", "periodic", (1.6, 2.0, 0.0), 1, True, 0.0, 0.23),
    ("Y/periodic/+1/even+beta", "Y", "periodic", (1.6, 2.0, 0.0), 1, True,
     BETA_GRATING_13_2, 0.0),
    ("Y/periodic/+1/ODD+beta", "Y", "periodic", (1.6, 2.1, 0.0), 1, True,
     BETA_EIGSRC, 0.0),
    ("Y/periodic/+1/even+beta+kx", "Y", "periodic", (1.6, 2.0, 0.0), 1, True,
     BETA_GRATING_13_2, 0.23),
    ("Y/periodic/+1/even+beta/REAL", "Y", "periodic", (1.6, 2.0, 0.0), 1, False,
     BETA_GRATING_13_2, 0.0),
    ("Y/metallic/-1/ODD+beta/REAL", "Y", "metallic", (1.6, 2.1, 0.0), -1, False,
     BETA_EIGSRC, 0.0),
)

STEPS = 3


def _build(axes: str, declaration: str, cell, phase, complex_storage: bool,
           beta: float, kx: float, xp):
    phases = (tuple(phase) if isinstance(phase, (tuple, list))
              else tuple(phase for _ in axes))
    planes = tuple(Mirror(name, int(value)) for name, value in zip(axes, phases))
    dimensions = 2 if float(cell[2]) == 0.0 else 3
    declared: Any = declaration
    if dimensions == 2 and declaration != "periodic":
        declared = {name.lower(): declaration for name in axes}
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                courant=0.35, boundaries=declared, symmetry=planes, beta=beta,
                k_point=(kx, 0.0, 0.0), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    thickness = []
    for index in range(3):
        if int(grid.shape[index]) < 6:
            thickness.append((0, 0))
        elif "XYZ"[index] in axes:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    return grid, fields, PML(grid=grid, thickness=tuple(thickness))


def _boundaries_of(grid, pml) -> Tuple[str, ...]:
    kinds = stepping._boundary_kinds(grid, pml)
    out: List[str] = []
    for axis, kind in enumerate(kinds):
        if kind != "mirror":
            out.append(kind)
        elif stepping._stored_past_owned(grid, axis):
            out.append(gate.MIRROR_PERIODIC)
        else:
            out.append(gate.MIRROR_METALLIC)
    return tuple(out)


def _coefficients_of(pml, half_integer: bool) -> Dict[str, Any]:
    """The PML tables the reference steps with, on the HOST.

    The reference half is stepped with NumPy unconditionally, so a table left on
    the device would be mixed into NumPy expressions against host operands. Under
    CuPy that either raises or quietly hands the whole expression to the device —
    and the second outcome is the dangerous one, because it would move the
    reference off the backend this module claims to compute it on. :func:`_host`
    is a no-op when ``run`` was handed NumPy.
    """
    suffix = "_h" if half_integer else ""
    return {f"{stem}_{axis}": _host(getattr(pml, f"{stem}_{axis}{suffix}"))
            for axis in "xyz" for stem in ("kms", "sinv")}


def _seed(fields, grid, xp, offset: int) -> None:
    """Random plus the engineered signed-zero / subnormal rows, BOTH storages.

    Random data alone is PROVABLY blind to the class the fill's needle lives in,
    so every volume gets the targeted planes on top of it. The REAL rows get them
    too: the real arm's own transcription claim is a signed-zero one
    (``curl - (c*g)`` stays a SINGLE subtract, folded_complex.py:918-920), and a
    plain ``rng.uniform`` state carries no signed zero and no subnormal at all,
    so it could not falsify that claim here any more than on the device.
    """
    rng = np.random.default_rng(gate.SEED + 31 + offset)
    rows = stepping._far_reflect_rows(grid)
    folded = [axis for axis in range(3) if grid.is_mirrored(axis)]
    for name in NAMES:
        target = getattr(fields, name)
        if str(target.dtype) == "complex64":
            host = gate.gate._seed_host_complex(tuple(grid.shape), rng)
            for axis in folded:
                host = gate.plant_fill_needles(host, axis, rows[axis])
        else:
            host = gate._seed_host_real(tuple(grid.shape), rng)
        target[...] = xp.asarray(host)


def _host(array) -> np.ndarray:
    """The array's bytes on the HOST, whichever backend ``run`` was handed.

    ``run(xp)`` puts the ``Fields`` on ``xp`` while the reference is always
    stepped with NumPy, so every read of a field crosses a backend boundary that
    is a no-op under NumPy and a device copy under CuPy. CuPy REFUSES the
    implicit crossing — ``np.asarray`` on a ``cupy.ndarray`` raises TypeError
    rather than silently transferring — so the copy is spelled explicitly here.

    ``.get()`` and not ``cupy.asnumpy``: this module never imports cupy, so the
    conversion is asked of the ARRAY, and NumPy arrays — which have no ``.get`` —
    take the other branch. Asking the object needs no module identity, which is
    also what keeps it correct when ``cp`` is not the module it claims to be.
    The transfer is bit-preserving in both directions; nothing about the uint32
    word this module compares depends on which side of the bus it was read from.
    """
    getter = getattr(array, "get", None)
    return array if getter is None else getter()


def _words(array) -> np.ndarray:
    host = np.asarray(_host(array))
    return np.ascontiguousarray(host).view(np.uint32).ravel()


# ---------------------------------------------------------------------------
# THE COMPARATOR POLICY
# ---------------------------------------------------------------------------

_EXPONENT = np.uint32(0xFF)
_SIGNIFICAND = np.uint32(0x7FFFFF)


def _is_nan(words: np.ndarray) -> np.ndarray:
    """IEEE 754-2019 §3.4: exponent all ones AND a non-zero significand.

    Read off the BITS rather than via ``np.isnan`` on a float view, because the
    whole point of this module is that the uint32 word is the unit of comparison.
    """
    return (((words >> np.uint32(23)) & _EXPONENT) == _EXPONENT) & \
           ((words & _SIGNIFICAND) != np.uint32(0))


def compare_words(mine: np.ndarray, theirs: np.ndarray) -> Tuple[int, int, int]:
    """``(mismatched, nan_words, nan_exempt)`` under this module's policy.

    RAW uint32 equality for every finite word, every zero (INCLUDING the sign of
    zero) and every subnormal — unchanged, and still the entire claim. Infinities
    too: ``+inf`` and ``-inf`` are fully specified encodings, and ``+inf`` against
    ``-inf`` is a mismatch.

    THE ONE EXEMPTION is NaN against NaN, and it is not a weakening of the
    arbiter — it is the removal of an assertion the standard never licensed.
    IEEE 754-2019:

    * §6.2.1 — a NaN's SIGN BIT is not interpreted; the standard attaches no
      meaning to it, and nothing constrains what an operation writes there;
    * §6.2.3 — an operation that propagates a NaN operand SHOULD carry the
      payload through. A *should*, not a *shall*;
    * §7.2 — an invalid operation (here ``inf - inf`` inside the curl) delivers a
      quiet NaN whose PAYLOAD the standard does not fix;
    * §5.11 — NaN compares unordered with everything, itself included, so there is
      no equality relation between two NaNs to appeal to in the first place.

    So a raw-bit comparison across a NaN word does not measure the transcription;
    it measures whichever quiet NaN the platform happened to synthesise. MEASURED,
    not argued: with the overflow fill needle in place BOTH platforms produced the
    same NaN population — 709387 word positions over these 17 cases, from
    hash-identical sources — but 1414 of them disagreed in sign or payload under
    numpy 2.2.6 and 0 disagreed under numpy 2.4.3, so a raw-bit verdict read 14/17
    on one and 17/17 on the other. First divergence: case 13, step_D, ``Dx`` word
    268, ``0x7fc00000`` (+qNaN) against ``0xffc00000`` (-qNaN).

    THIS EXEMPTION IS NOT WHAT MAKES THE RUN PASS, and that distinction is the
    point. Four arms on numpy 2.2.6, the platform that failed::

        finite needle + this comparator   17/17   nan_words 0       exempt 0
        3.4e38 needle + this comparator   17/17   nan_words 709387  exempt 1414
        3.4e38 needle + raw-bit compare   14/17   (the original failure)
        finite needle + raw-bit compare   17/17

    The needle fix is what restores the claim: with it, this comparison and a
    raw-bit one are the SAME measurement, and ``nan_exempt`` reads 0. The
    exemption is defence in depth for a NaN the harness did not plant.

    A NaN opposite a FINITE word is a mismatch and must fail: that is the array
    path and the reference disagreeing about whether the result is a number at
    all, which is a real transcription defect and the failure mode this exemption
    must not swallow. And a genuine FDTD run that reaches NaN has already
    diverged, so the exemption covers a state the engine never usefully occupies.

    ``nan_words`` counts the WORD POSITIONS at which either side is NaN — the
    census that stops a silent NaN bloom from hiding inside the exemption.
    ``nan_exempt`` counts only the positions that DIFFER and were forgiven, which
    is the number that has to be zero for this comparison and a raw-bit one to be
    the same measurement.
    """
    differ = mine != theirs
    nan_mine = _is_nan(mine)
    nan_theirs = _is_nan(theirs)
    both = nan_mine & nan_theirs
    return (int(np.count_nonzero(differ & ~both)),
            int(np.count_nonzero(nan_mine | nan_theirs)),
            int(np.count_nonzero(differ & both)))


def run(xp=np, out_path: Optional[str] = None) -> Dict[str, Any]:
    """Every configuration, curl AND fill, against ``stepping.py`` itself."""
    cases: List[Dict[str, Any]] = []
    started_all = time.time()
    for index, (label, axes, declaration, cell, phase, complex_storage, beta,
                kx) in enumerate(CASES, start=1):
        started = time.time()
        grid, fields, pml = _build(axes, declaration, cell, phase,
                                   complex_storage, beta, kx, xp)
        _seed(fields, grid, xp, index)

        mine = {name: np.array(_host(getattr(fields, name)), copy=True)
                for name in NAMES}
        boundaries = _boundaries_of(grid, pml)
        mirror_phases = tuple(grid.mirror_phase(a) or 1 for a in range(3))
        rows = stepping._far_reflect_rows(grid)
        kinds = stepping._boundary_kinds(grid, pml)
        phases = tuple(grid.bloch_phase(a) if kinds[a] == "periodic" else None
                       for a in range(3)) if grid.has_bloch else (None, None, None)

        curl_failures: List[Any] = []
        fill_failures: List[Any] = []
        nan_words = 0
        nan_exempt = 0
        nan_checks = 0

        def check(failures: List[Any], tag: str) -> None:
            """One byte-comparison per component, under :func:`compare_words`."""
            nonlocal nan_words, nan_exempt, nan_checks
            for name in NAMES:
                a = _words(getattr(fields, name))
                b = _words(mine[name])
                mismatched, nans, exempt = compare_words(a, b)
                nan_words += nans
                nan_exempt += exempt
                nan_checks += int(nans > 0)
                if mismatched:
                    failures.append((tag, name, mismatched))

        for _ in range(STEPS):
            for sub_step in ("step_B", "step_D"):
                getattr(stepping, sub_step)(fields, pml)
                gate.folded_reference_step(
                    np, mine, _coefficients_of(pml, sub_step == "step_B"),
                    np.float32(grid.dt / grid.dx), sub_step, boundaries, phases,
                    mirror_phases, rows, beta=grid.beta, dt=grid.dt,
                    complex_storage=complex_storage)
                check(curl_failures, sub_step)

                family = "B" if sub_step == "step_B" else "D"
                getattr(stepping, f"fill_symmetry_bc_{family}")(fields)
                getattr(stepping, f"fill_folded_far_ghosts_{family}")(fields)
                entries = folded_complex.ghost_fill_axis_entries(grid, family)
                gate.reference_ghost_fill(np, mine, family, entries, grid.shape)
                check(fill_failures, family)

                update = (stepping.update_H if sub_step == "step_B"
                          else stepping.update_E)
                update(fields, pml)
                for name in NAMES:
                    mine[name] = np.array(_host(getattr(fields, name)),
                                          copy=True)

        ok = not curl_failures and not fill_failures
        case = {
            "label": label, "shape": [int(n) for n in grid.shape],
            "stored": [int(grid.stored_cells(a)) for a in range(3)],
            "owned": [int(grid.owned_cells(a)) for a in range(3)],
            "codes": list(folded_complex.folded_axis_kinds(grid, pml)[0] or ()),
            "reflect_rows": [None if r is None else int(r) for r in rows],
            "boundaries": list(boundaries), "beta": beta, "kx": kx,
            "storage": "complex" if complex_storage else "real",
            "curl_failures": curl_failures[:6],
            "fill_failures": fill_failures[:6],
            # The NaN census. `nan_words` counts WORD POSITIONS at which either
            # side was NaN, summed over every byte-comparison this case performed
            # — a bloom detector, not a per-array count. `nan_exempt` is the
            # subset the policy actually forgave, and is the number that has to
            # be zero for a raw-bit comparison and this one to be the SAME
            # measurement.
            "nan_words": nan_words,
            "nan_exempt": nan_exempt,
            "nan_checks": nan_checks,
            "identical": ok,
            "seconds": round(time.time() - started, 3),
        }
        cases.append(case)
        print(f"case {index}/{len(CASES)} {label} shape={case['shape']} "
              f"stored={case['stored']} owned={case['owned']} "
              f"codes={case['codes']} rows={case['reflect_rows']}: "
              f"reference==stepping -> {ok}"
              + f"  nan_words={nan_words} nan_exempt={nan_exempt}"
              + ("" if ok else
                 f"  CURL {curl_failures[:3]} FILL {fill_failures[:3]}")
              + f"  elapsed={round(time.time() - started_all, 1)} s",
              flush=True)
        if out_path:                       # partial results, per case
            _save(out_path, cases)
    identical = sum(int(c["identical"]) for c in cases)
    record = {"ran": len(cases), "identical": identical, "steps": STEPS,
              "nan_words": sum(int(c["nan_words"]) for c in cases),
              "nan_exempt": sum(int(c["nan_exempt"]) for c in cases),
              "nan_checks": sum(int(c["nan_checks"]) for c in cases),
              "needle_magnitude": float(gate.FILL_NEEDLE_MAGNITUDE),
              "policy": ("raw uint32 for every finite, zero, subnormal and "
                         "infinite word; NaN-vs-NaN compares by IS-NaN "
                         "(IEEE 754-2019 §6.2.1/§6.2.3/§7.2 leave the sign and "
                         "payload unspecified); NaN-vs-non-NaN FAILS"),
              "cases": cases}
    if out_path:
        _save(out_path, cases, record)
    return record


def _save(path: str, cases: List[Dict[str, Any]],
          record: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(record) if record else {"ran": len(cases), "cases": cases,
                                           "partial": True}
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="write the record (NaN census included) as JSON")
    arguments = parser.parse_args(argv)
    record = run(np, out_path=arguments.out)
    print(f"\n{record['identical']}/{record['ran']} configurations bit-identical "
          f"to stepping.py over {record['steps']} full steps each", flush=True)
    print(f"NaN census: {record['nan_words']} NaN words seen across "
          f"{record['nan_checks']} array-checks, {record['nan_exempt']} of them "
          f"forgiven by the NaN-vs-NaN exemption "
          f"(fill needle {record['needle_magnitude']:g})", flush=True)
    return 0 if record["identical"] == record["ran"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

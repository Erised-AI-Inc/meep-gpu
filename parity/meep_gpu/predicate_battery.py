"""Call the SHIPPED Metal coverage predicates on one lifted (fields, pml, grid) triple.

THE METAL COUNTERPART of ``results/predicate_coverage_2026-08-14_allnine/
predicate_battery.py``, dropped into a BYTE-IDENTICAL copy of that round's
``measure_predicate_coverage.py`` and ``match_param_rows.py`` (sha256 verified at
copy time) — same lift records, same denominators, same three legs, same corpus.
Only the SUBJECT differs, which is what makes this round's number commensurable
with Triton's 702/759 and the hand-CUDA ladder's 157/759 rather than merely
similar-looking.

TRANCHE 3 ADDS THE THREE FOLDED FAMILIES, and adds them to the ENTRY LIST rather
than relying on ``plan_step`` to find them. The composer reaches every registered
arm by itself, so the COMPOSED column would have counted the fold whatever this
file said; the ADMITTED column is a hand-written list of predicates and would
silently have stayed at eight products while the composer selected eleven. An
admitted number BELOW its own composed number is the shape that mistake takes, so
the two lists are kept in step deliberately and the analyzer asserts the direction.

TRANCHE 4 ADDS FOUR MORE — ``cylindrical_complex``, ``cylindrical_real``,
``folded_beta`` (two arms) and ``nonlinear_constitutive`` — for the same reason and
with the same hazard. THE ROUTING VERDICTS ARE THE ONES ENTERED, not the wide ones,
exactly as tranche 3 entered ``folded_composition_curl_coverage`` rather than the
fold-optional predicate the gate certifies against: ``folded_beta`` exposes a wide
curl verdict that admits an UNFOLDED beta run (the shape its gate certifies) and a
composition verdict with the fold MANDATORY, and counting the wide one here would
credit this family with every unfolded beta row the composer routes to
``special_kz``.

TRANCHE 6 ADDS THE SIXTEEN FAMILIES THE 08-17/08-18 ROUNDS LANDED, and adds
NOTHING ELSE. The tranche-5 census ran this file unchanged against a tree that had
grown from sixteen registered families to thirty-two, and its analyzer printed the
exact shape tranche 3's docstring predicted: ``COMPOSED EXCEEDS ADMITTED — the
battery's predicate list is missing a family the arm table carries``. Measured on
that round's own record (162 rows survived its legs; see the corpus note below):
composed 663/663, admitted 593/663, and the 70-slot difference resolves EXACTLY into
the arms this entry list never asked about — 15 ``ADE update_P``, 10 ``no-PML
curl``, 8 ``complex conductive no-PML curl``, 6 ``conductive no-PML curl``,
4 each of ``folded dispersive PML E`` / ``complex no-PML stored E`` /
``nonlinear PML curl`` / ``complex no-PML curl``, 3 each of ``no-PML stored E`` /
``complex folded off-diagonal PML E`` / ``dispersive PML E``, 2 each of
``nonlinear PML magnetic`` / ``complex no-PML off-diagonal``, and 1 each of
``folded off-diagonal dispersive PML E`` and ``conductive PML curl``. The entries
below are those arms and only those arms.

THE ROUTING VERDICT IS AGAIN THE ONE ENTERED, and for these sixteen that is a
weaker claim than it was for tranches 3 and 4, which is worth stating rather than
implying: NONE of them exposes a wide/routing SPLIT. Each registered arm's
``_arm_coverage`` closure delegates to exactly ONE public predicate with the arm's
own slot passed straight through — measured by reading every closure, not by
matching names — so entering that predicate IS entering the routing verdict, and
there is no wider sibling here to enter by mistake. The two families that DO carry a
split (``symmetry``/``folded_complex``'s fold-optional curls and ``folded_beta``'s
two) were already entered at their composition verdicts and are untouched.

ONE FAMILY IS ENTERED AT THREE SLOTS IT ALREADY HELD. ``nonlinear_constitutive``
was entered at ``update_E`` alone, on the argument that the shared clause 10 keeps a
nonlinear run's curls and ``update_H`` on the array path. That is no longer what the
tree does: the family registers THREE further arms — ``nonlinear PML curl`` on
``step_B``/``step_D`` and ``nonlinear PML magnetic`` on ``update_H`` — which reuse
the individually certified ordinary bodies through nonlinear-only spine predicates.
The composer selected all three on this corpus. They are entered here at their own
predicates, and the family name is unchanged, so a removal check that subtracts
``nonlinear_constitutive`` subtracts all four slots together.

``update_P`` IS NO LONGER A NAMED ABSENCE. ``ade_update_p`` registers a WIRED arm on
that slot and ``launch.NO_ARM_REASONS`` is now the EMPTY dict — so the block below,
which read a reason out of that table, was recording ``reason: None`` beside
``metal_arm: None`` on all fifteen polarization slots while ``plan_step`` selected
``ADE update_P`` on every one of them. The absence has become a product; it is asked
for like any other. NOTE THE ANALYZER COUPLING: ``analyze_metal_coverage.roll_up``
hard-codes ``admitted["update_P"] = []`` after the predicate loop, so this entry is
DISCARDED unless that function is patched in step. Both changes are in this
directory and neither is sufficient alone.

A FIFTH PROBE ARTIFACT IS SET, for the cylindrical complex family, on the same
argument the other four rest on: that family multiplies by a complex ROW on the
LEFT and by a complex SCALAR on the left whose real word is a SIGNED ZERO, and
neither pattern was ever classified by the unfolded complex, beta or folded-complex
artifacts. Reading one of theirs would licence an arm from a record that never
measured this call. With it absent the family refuses by name on all four of its
slots and the number would silently be a number about fifteen products.

A THIRD PROBE ARTIFACT IS SET HERE for the same reason the first two are: with
``MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE`` unset the folded complex family
refuses BY NAME on every slot, and its absence would be recorded as a coverage
verdict rather than as a missing measurement. It is a SEPARATE candidate list, not
an alias of the complex one, because the folded fill multiplies by a complex
coefficient on the LEFT — a pattern the unfolded artifact never classified.

THE TWO SEAM SLOTS ARE RECORDED AND ARE NOT IN THE DENOMINATOR. ``fill_B`` and
``fill_D`` are real slots the composer fills, but the 759-slot denominator is
``186 x 4 + 15`` and was fixed by the Triton rounds; adding them would make this
round's number incommensurable with 702/759 and 157/759, which is the whole reason
the denominator is quoted rather than recomputed. They are reported as their own
addendum count instead.

THE ONE STRUCTURAL DIFFERENCE FROM BOTH EARLIER ROUNDS, and it is in this track's
favour rather than a caveat: **nothing is factored out**.

* The Triton round could not run its predicates as written — every one refuses
  ``array module is 'numpy', not cupy`` on this host — so its number is "modulo
  that one clause", computed by stripping the string from a returned tuple.
* The hand-CUDA round could not even do that (its predicate short-circuits), so it
  wrapped the grid in a ``_CupyNamed`` proxy and measured through a shim whose
  soundness it then had to probe per row.
* The Metal predicates ask for THIS host: NumPy on the host side, torch with MPS
  for the device side. ``coverage._metal_backend_reasons`` is satisfied natively,
  so ``covered`` here is the shipped predicate's own verdict on the real target
  and there is no shim, no filter and no soundness question about either.

WHAT IS STILL A CONDITION, stated because it is one. ``MEEP_GPU_SUBNORMAL_POLICY``
must be ``flush``: on MPS the float32 subnormal flush is native and has no lever, so
the resolved default ``keep`` is NOT OFFERABLE and every Metal predicate refuses by
name. This module sets it at import — before any ``meep_gpu`` module is reached —
and RECORDS what it set, because a coverage number measured under a policy the
artifact does not name is a number about an unknown machine.

THE TWO EXPANSION PROBES ARE LOADED HERE, not passed in. The harness's ``probe``
argument carries the TRITON round's artifact (a CuPy measurement); it is recorded
for provenance and deliberately not used, because which complex-multiply arm this
host's reference takes is a different platform's fact. The Metal artifacts are
this host's own, measured by ``gate_metal_complex`` and ``gate_metal_special_kz``.
With either missing, four of the eight products refuse by name and the coverage
number would silently be a number about four families; :func:`evaluate` records
``probe_available`` per row so that can never be read off as coverage.

NO SIXTH ARTIFACT IS ADDED, and that is a measurement rather than an economy. Every
one of the sixteen new complex families takes its expansion from a record one of the
existing four already carries: ``complex_folded_offdiag_update_e`` routes through
``folded_complex._folded_complex_grid_reasons`` and therefore needs the FOLDED
complex artifact; the other seven import ``load_expansion_probe`` from
``complex_fields`` and need the unfolded one. None introduces an operand orientation
the four artifacts have not classified, which is the whole test a new artifact has to
fail before it is added.

A KEY MISMATCH IS RECORDED HERE BECAUSE IT IS LOAD-BEARING FOR THE ARGUMENT ABOVE,
not because this file can fix it. ``launch.plan_step`` builds its context with
``extra`` holding exactly four keys — ``complex_probe``, ``beta_probe``,
``folded_complex_probe``, ``cylindrical_complex_probe`` (launch.py:669, the only
assignment in the module). FOUR of the new arms read ``context.extra.get("probe")``,
a key ``plan_step`` never sets: ``complex_no_pml_curl``,
``complex_no_pml_conductive``, ``complex_no_pml_offdiag_update_e`` and
``complex_folded_offdiag_update_e``. They therefore receive ``None`` from the
composer and fall back to their module's own environment variable inside
``_expansion_reasons`` (complex_fields.py:710-717), which is why they compose at all
on this corpus. The verdict is IDENTICAL either way ONLY while the environment
variable names the same artifact the caller passed — which this module guarantees,
because it sets those variables itself. A caller that passes ``complex_probe=`` to
``plan_step`` and sets no environment variable gets four families refusing by name
for a missing measurement they were handed. Entered below at the record the module
would load, so ADMITTED and COMPOSED stay commensurable; the mismatch belongs in the
package, not in a coverage harness.

WHAT IS COUNTED. Per row and per sub-step:

* ``predicates`` — every shipped Metal predicate's own verdict, by family and
  kernel, exactly as the Triton battery records them.
* ``plan_step`` — what the SHIPPED composer selects per slot, with the arms that
  admitted. This is the number that matters for the composition claim: a predicate
  admitting a slot means a kernel could step it; ``plan_step`` selecting it means
  the composer WOULD. Unlike the Triton round, the builders run here too, so a slot
  can be recorded as admitted-but-unbuildable, which is a distinction that round
  could not make (every Triton builder raises ``ImportError`` on this host).
* ``update_P`` — ASKED, not asserted. ``ade_update_p`` carries a wired arm on that
  slot, so the fifteen polarization slots are a predicate verdict like every other.
  The per-state list is kept and now records the ARM and the per-component verdict
  instead of a reason read out of an empty table.

THE CORPUS CAVEAT THIS FILE CANNOT FIX, recorded so a number produced with it is not
misread. The tranche-5 run measured 162 of 186 rows: its directory was copied
without ``shim/parameterized.py``, so the ``tests_param`` leg's children died at
``import parameterized`` on all six modules and recovered NOTHING (tranche 4:
24 rows in 26.7 s; tranche 5: 0 rows in 3.0 s, each module in 0.1 s). Twenty-four
rows are therefore recorded ``measured: false, note: "child died"``, the denominator
computes 663 rather than 759, and the round's headline is not commensurable with
687/759. Copy the shim directory beside this file before running the legs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
def _find_api_root(start):
    """Locate the repository root BY NAME, never by depth.

    These harness files were promoted out of a results/ tranche into the parity
    dir; a hard-coded ``parents[N]`` silently followed them to the wrong root and
    every child died on ModuleNotFoundError while the parent still wrote a
    full-length census of ``measured: false`` rows.
    """
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}; refusing to guess")


_API = _find_api_root(_HERE)

#: The kernels package these predicates measure. ``measure_predicate_coverage`` digests
#: it (plus the shared engine files) into every census row's ``subject_manifest_sha256``.
#: Declared here rather than inferred from imports: a battery may import a sibling
#: package to compare against it without measuring it.
SUBJECT_PACKAGE = "metal_kernels"


class _HostGrid:
    """The grid ``runtime_reasons`` evaluates the backend clause over: a host array module."""

    xp = __import__("numpy")


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate the battery at all, or ``[]``.

    Read by ``measure_predicate_coverage`` once per interpreter before any row runs
    there. Where the Metal kernels cannot launch, every predicate refuses for the same
    process-level reason, and a census that records such a row as measured hands the
    board a coverage gap that is not one (2026-09-03: five gdsii/sigma rows under
    interpreters without torch, eleven missing halves). The driver refuses the row BY
    NAME instead.

    The list is ``coverage._metal_backend_reasons`` itself, evaluated over a grid whose
    only clause-relevant property is a host (numpy) array module, under the policy this
    battery declares above: the torch import, the MPS build and device, and
    ``compile_shader`` are the process clauses, and a policy the executor cannot honour
    is one too (under ``keep`` every Metal predicate refuses, which is exactly the
    0/759 this module's header warns of). One definition, no second list of strings.
    Batteries whose predicates evaluate in pure Python (Triton, CUDA) declare no such
    function.
    """
    from meep_gpu.metal_kernels import coverage  # noqa: PLC0415
    return list(coverage._metal_backend_reasons(_HostGrid()))

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED. `flush` is the only value
# the MPS executor can honour; the resolved default `keep` makes every predicate
# refuse, which would report 0/759 and read as "the port covers nothing".
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

#: This host's own measured expansion artifacts, newest first.
COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_complex_audit_2026-08-16/complex_expansion_probe.json",
    "parity/meep_gpu/results/metal_complex_2026-08-15/complex_expansion_probe.json",
)
BETA_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_special_kz_2026-08-16_audit/expansion.json",
    "parity/meep_gpu/results/metal_special_kz_2026-08-15_certify/expansion.json",
    "parity/meep_gpu/results/metal_special_kz_2026-08-15/expansion.json",
)
#: The folded complex family's own artifact. SEPARATE, not an alias — see the module
#: docstring: the folded fill multiplies by a complex coefficient on the LEFT, a
#: pattern neither of the two above ever classified, so reading one of theirs would
#: license an arm from a record that never measured this call.
FOLDED_COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_folded_complex_2026-08-16/expansion.json",
)
#: The cylindrical complex family's own artifact. SEPARATE for the reason the
#: folded-complex one is: it carries two operand orientations none of the others
#: classified — the i*m/r coefficient as a complex ROW on the left, and the
#: |m| = 1 axis-increment scalar as a complex SCALAR on the left with a signed-zero
#: real word.
CYLINDRICAL_COMPLEX_PROBE_CANDIDATES: Tuple[str, ...] = (
    "parity/meep_gpu/results/metal_cylindrical_complex_2026-08-16/expansion.json",
)

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D", "update_H", "update_E")
#: The two seam slots the folded families fill. OUTSIDE the 759 denominator by
#: construction — see the module docstring — and counted separately.
FILL_SLOTS: Tuple[str, ...] = ("fill_B", "fill_D")


def _first_existing(candidates: Tuple[str, ...]) -> Optional[str]:
    for relative in candidates:
        path = _API / relative
        if path.exists():
            return str(path)
    return None


COMPLEX_PROBE_PATH = _first_existing(COMPLEX_PROBE_CANDIDATES)
BETA_PROBE_PATH = _first_existing(BETA_PROBE_CANDIDATES)
FOLDED_COMPLEX_PROBE_PATH = _first_existing(FOLDED_COMPLEX_PROBE_CANDIDATES)
CYLINDRICAL_COMPLEX_PROBE_PATH = _first_existing(
    CYLINDRICAL_COMPLEX_PROBE_CANDIDATES)
if COMPLEX_PROBE_PATH:
    os.environ.setdefault("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
                          COMPLEX_PROBE_PATH)
if BETA_PROBE_PATH:
    os.environ.setdefault("MEEP_GPU_METAL_EXPANSION_PROBE", BETA_PROBE_PATH)
if FOLDED_COMPLEX_PROBE_PATH:
    os.environ.setdefault("MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
                          FOLDED_COMPLEX_PROBE_PATH)
if CYLINDRICAL_COMPLEX_PROBE_PATH:
    os.environ.setdefault("MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
                          CYLINDRICAL_COMPLEX_PROBE_PATH)


def _verdict(call: Callable[[], Any]) -> Dict[str, Any]:
    """One predicate's answer, JSON-safe. A RAISE is a refusal, recorded as one."""
    try:
        verdict = call()
    except BaseException as exc:  # noqa: BLE001 - a raising predicate is a refusal
        return {"covered": False, "raised": f"{type(exc).__name__}: {exc}"[:400],
                "reasons": []}
    return {"covered": bool(getattr(verdict, "covered", False)),
            "raised": None,
            "reasons": [str(reason) for reason in
                        (getattr(verdict, "reasons", ()) or ())][:6]}


def _probes() -> Dict[str, Any]:
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields,
        cylindrical_complex,
        folded_complex,
        special_kz,
    )

    return {"complex": complex_fields.load_expansion_probe(),
            "beta": special_kz.load_expansion_probe(),
            "folded_complex": folded_complex.load_expansion_probe(),
            "cylindrical_complex": cylindrical_complex.load_expansion_probe()}


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:
    """Every shipped Metal predicate, on the real lifted objects.

    ``probe`` is the harness's (Triton, CuPy-measured) artifact. It is recorded and
    NOT used — see the module docstring.
    """
    from meep_gpu.metal_kernels import arms, device, launch, subnormal  # noqa: PLC0415
    from meep_gpu.metal_kernels.ade_update_p import (  # noqa: PLC0415
        metal_ade_component_coverage,
        metal_ade_update_p_coverage,
    )
    from meep_gpu.metal_kernels.bfast_curl import (  # noqa: PLC0415
        bfast_pml_curl_coverage,
        bfast_run_constitutive_coverage,
    )
    from meep_gpu.metal_kernels.complex_conductive_pml import (  # noqa: PLC0415
        metal_complex_conductive_pml_constitutive_coverage,
        metal_complex_conductive_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.complex_dispersive_spine import (  # noqa: PLC0415
        metal_complex_dispersive_pml_curl_coverage,
        metal_complex_dispersive_pml_magnetic_coverage,
    )
    from meep_gpu.metal_kernels.complex_dispersive_update_e import (  # noqa: PLC0415
        metal_complex_dispersive_e_coverage,
    )
    from meep_gpu.metal_kernels.complex_folded_offdiag_update_e import (  # noqa: PLC0415
        metal_complex_folded_offdiag_coverage,
    )
    from meep_gpu.metal_kernels.complex_no_pml_conductive import (  # noqa: PLC0415
        metal_complex_conductive_no_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.complex_no_pml_curl import (  # noqa: PLC0415
        metal_complex_no_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.complex_no_pml_offdiag_update_e import (  # noqa: PLC0415
        metal_complex_no_pml_offdiag_coverage,
    )
    from meep_gpu.metal_kernels.complex_no_pml_stored_e import (  # noqa: PLC0415
        metal_complex_stored_e_coverage,
    )
    from meep_gpu.metal_kernels.conductive_pml import (  # noqa: PLC0415
        metal_conductive_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.dispersive_update_e import (  # noqa: PLC0415
        metal_dispersive_e_coverage,
    )
    from meep_gpu.metal_kernels.folded_dispersive_update_e import (  # noqa: PLC0415
        folded_dispersive_e_coverage,
    )
    from meep_gpu.metal_kernels.folded_offdiag_dispersive_update_e import (  # noqa: PLC0415
        folded_offdiag_dispersive_coverage,
    )
    from meep_gpu.metal_kernels.no_pml_conductive import (  # noqa: PLC0415
        metal_conductive_plain_curl_coverage,
    )
    from meep_gpu.metal_kernels.no_pml_curl import (  # noqa: PLC0415
        metal_plain_curl_coverage,
    )
    from meep_gpu.metal_kernels.no_pml_stored_e import (  # noqa: PLC0415
        metal_stored_e_coverage,
    )
    from meep_gpu.metal_kernels.complex_fields import (  # noqa: PLC0415
        complex_constitutive_coverage,
        complex_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.coverage import (  # noqa: PLC0415
        constitutive_coverage,
        pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.cylindrical_complex import (  # noqa: PLC0415
        cylindrical_complex_constitutive_coverage,
        cylindrical_complex_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.cylindrical_real import (  # noqa: PLC0415
        cylindrical_real_constitutive_coverage,
        cylindrical_real_curl_coverage,
    )
    from meep_gpu.metal_kernels.folded_beta import (  # noqa: PLC0415
        folded_beta_complex_constitutive_coverage,
        folded_beta_composition_bloch_curl_coverage,
        folded_beta_composition_curl_coverage,
        folded_beta_constitutive_coverage,
    )
    from meep_gpu.metal_kernels.folded_complex import (  # noqa: PLC0415
        folded_complex_composition_curl_coverage,
        folded_complex_constitutive_coverage,
        folded_complex_fill_coverage,
    )
    from meep_gpu.metal_kernels.folded_offdiag_update_e import (  # noqa: PLC0415
        folded_offdiag_composition_coverage,
    )
    from meep_gpu.metal_kernels.no_pml_constitutive import (  # noqa: PLC0415
        metal_null_constitutive_coverage,
    )
    from meep_gpu.metal_kernels.nonlinear_update_e import (  # noqa: PLC0415
        nonlinear_constitutive_coverage,
        nonlinear_run_constitutive_coverage,
        nonlinear_run_pml_curl_coverage,
    )
    from meep_gpu.metal_kernels.offdiag_update_e import (  # noqa: PLC0415
        offdiag_constitutive_coverage,
    )
    from meep_gpu.metal_kernels.special_kz import (  # noqa: PLC0415
        beta_bloch_pml_curl_coverage,
        beta_pml_curl_coverage,
        beta_run_complex_constitutive_coverage,
        beta_run_constitutive_coverage,
    )
    from meep_gpu.metal_kernels.symmetry import (  # noqa: PLC0415
        folded_composition_curl_coverage,
        folded_constitutive_coverage,
        mirror_ghost_fill_coverage,
    )

    fields = driver.fields
    pml = driver.pml
    grid = driver.grid
    sources = getattr(driver, "_sources", None)
    states = tuple(getattr(fields, "polarizations", ()) or ())

    records = _probes()
    complex_probe, beta_probe = records["complex"], records["beta"]
    folded_complex_probe = records["folded_complex"]
    cylindrical_complex_probe = records["cylindrical_complex"]

    # THE RESIDENCY DECLARATION IS PART OF EVERY METAL PREDICATE, and passing None
    # would make every one of them refuse for a reason that has nothing to do with
    # this corpus. A fresh, empty Residency is exactly what `plan_step` builds for a
    # real composition, so the predicates are asked the question the composer asks.
    residency = device.Residency()

    entries: List[Tuple[str, str, str, Callable[[], Any]]] = []

    def add(family: str, label: str, sub_step: str,
            call: Callable[[], Any]) -> None:
        entries.append((family, label, sub_step, call))

    for name in ("step_B", "step_D"):
        add("pml_curl", "pml_curl", name,
            lambda n=name: pml_curl_coverage(fields, pml, n, residency))
        add("complex_fields", "complex_pml_curl", name,
            lambda n=name: complex_pml_curl_coverage(fields, pml, n, residency,
                                                     complex_probe))
        add("special_kz_real", "beta_pml_curl", name,
            lambda n=name: beta_pml_curl_coverage(fields, pml, n, residency))
        add("special_kz_complex", "beta_bloch_pml_curl", name,
            lambda n=name: beta_bloch_pml_curl_coverage(fields, pml, n, residency,
                                                        beta_probe))
        add("bfast_curl", "bfast_pml_curl", name,
            lambda n=name: bfast_pml_curl_coverage(fields, pml, n, residency))
        # THE ROUTING VERDICTS, not the wide ones. Both folded curl families expose
        # a wide predicate that admits zero folded axes (the gate's subject, where
        # the emitted source reduces to the certified kernel's character for
        # character) and a composition verdict with the fold MANDATORY. Counting the
        # wide one here would credit this family with every unfolded row in the
        # corpus — rows the composer routes to `pml_curl` — so the number would
        # exceed the composed number by exactly the families' overlap.
        add("folded", "folded_pml_curl", name,
            lambda n=name: folded_composition_curl_coverage(fields, pml, n,
                                                            residency))
        add("folded_complex", "folded_complex_pml_curl", name,
            lambda n=name: folded_complex_composition_curl_coverage(
                fields, pml, n, residency, folded_complex_probe))
        # TRANCHE 4. Both folded-beta arms take the ROUTING verdict, with the fold
        # MANDATORY, for the reason the comment above gives: the wide verdict
        # admits an unfolded beta run, which the composer routes to `special_kz`.
        add("folded_beta_real", "folded_beta_pml_curl", name,
            lambda n=name: folded_beta_composition_curl_coverage(
                fields, pml, n, residency))
        add("folded_beta_complex", "folded_beta_bloch_pml_curl", name,
            lambda n=name: folded_beta_composition_bloch_curl_coverage(
                fields, pml, n, residency, beta_probe))
        add("cylindrical_complex", "cylindrical_complex_pml_curl", name,
            lambda n=name: cylindrical_complex_pml_curl_coverage(
                fields, pml, n, residency, cylindrical_complex_probe))
        add("cylindrical_real", "cylindrical_real_curl", name,
            lambda n=name: cylindrical_real_curl_coverage(fields, pml, n,
                                                          residency))
        # TRANCHE 6, curl slots. Eight arms the composer already selects on this
        # corpus and this list never asked about. Each is the SINGLE public
        # predicate its registered arm's `_arm_coverage` delegates to, with the
        # slot passed straight through — none of these families exposes a wide
        # sibling, so there is no routing choice to get wrong here (unlike the two
        # fold families above, whose composition verdicts are the ones entered).
        add("conductive_pml", "conductive_pml_curl", name,
            lambda n=name: metal_conductive_pml_curl_coverage(fields, pml, n,
                                                              residency))
        add("no_pml_curl", "plain_curl", name,
            lambda n=name: metal_plain_curl_coverage(fields, pml, n, residency))
        add("no_pml_conductive", "conductive_plain_curl", name,
            lambda n=name: metal_conductive_plain_curl_coverage(fields, pml, n,
                                                                residency))
        # The four complex no-PML/conductive/dispersive curls take the UNFOLDED
        # complex artifact: each imports `load_expansion_probe` from
        # `complex_fields`, so that record is the one their own fallback reads.
        add("complex_no_pml_curl", "complex_plain_curl", name,
            lambda n=name: metal_complex_no_pml_curl_coverage(
                fields, pml, n, residency, complex_probe))
        add("complex_no_pml_conductive", "complex_conductive_plain_curl", name,
            lambda n=name: metal_complex_conductive_no_pml_curl_coverage(
                fields, pml, n, residency, complex_probe))
        add("complex_conductive_pml", "complex_conductive_pml_curl", name,
            lambda n=name: metal_complex_conductive_pml_curl_coverage(
                fields, pml, n, residency, complex_probe))
        add("complex_dispersive_spine", "complex_dispersive_pml_curl", name,
            lambda n=name: metal_complex_dispersive_pml_curl_coverage(
                fields, pml, n, residency, complex_probe))
        # THE NONLINEAR SPINE, and the reason the family now spans four slots
        # rather than one: `step_B`/`step_D` reuse the certified ordinary PML curl
        # body through a nonlinear-only predicate, because the engine does not read
        # chi2/chi3 in those sub-steps. Same family name as the Pade `update_E`
        # arm, so a removal check subtracts all four together.
        add("nonlinear_constitutive", "nonlinear_run_pml_curl", name,
            lambda n=name: nonlinear_run_pml_curl_coverage(fields, pml, n,
                                                           residency))

    for side, sub_step in (("H", "update_H"), ("E", "update_E")):
        add("constitutive", "constitutive", sub_step,
            lambda s=side: constitutive_coverage(fields, pml, s, residency))
        add("no_pml_constitutive", "null_constitutive", sub_step,
            lambda s=side: metal_null_constitutive_coverage(fields, pml, s,
                                                            residency))
        add("complex_fields", "complex_constitutive", sub_step,
            lambda s=side: complex_constitutive_coverage(fields, pml, s, residency,
                                                         complex_probe))
        add("special_kz_real", "beta_run_constitutive", sub_step,
            lambda s=side: beta_run_constitutive_coverage(fields, pml, s, residency))
        add("special_kz_complex", "beta_run_complex_constitutive", sub_step,
            lambda s=side: beta_run_complex_constitutive_coverage(fields, pml, s,
                                                                  residency))
        add("bfast_curl", "bfast_run_constitutive", sub_step,
            lambda s=side: bfast_run_constitutive_coverage(fields, pml, s,
                                                           residency))
        add("folded", "folded_constitutive", sub_step,
            lambda s=side: folded_constitutive_coverage(fields, pml, s, residency))
        add("folded_complex", "folded_complex_constitutive", sub_step,
            lambda s=side: folded_complex_constitutive_coverage(
                fields, pml, s, residency, folded_complex_probe))
        add("folded_beta_real", "folded_beta_constitutive", sub_step,
            lambda s=side: folded_beta_constitutive_coverage(fields, pml, s,
                                                              residency))
        add("folded_beta_complex", "folded_beta_complex_constitutive", sub_step,
            lambda s=side: folded_beta_complex_constitutive_coverage(
                fields, pml, s, residency, beta_probe))
        add("cylindrical_complex", "cylindrical_complex_constitutive", sub_step,
            lambda s=side: cylindrical_complex_constitutive_coverage(
                fields, pml, s, residency, cylindrical_complex_probe))
        add("cylindrical_real", "cylindrical_real_constitutive", sub_step,
            lambda s=side: cylindrical_real_constitutive_coverage(fields, pml, s,
                                                                   residency))
        # TRANCHE 6, both constitutive slots. The complex conductive PML family is
        # the only new one registered on BOTH sides, and its predicate takes the
        # side the same way the certified pair does.
        add("complex_conductive_pml", "complex_conductive_pml_constitutive",
            sub_step,
            lambda s=side: metal_complex_conductive_pml_constitutive_coverage(
                fields, pml, s, residency, complex_probe))

    # TRANCHE 6, update_H ONLY, and the asymmetry is the arm table's rather than a
    # choice made here: both families register a magnetic arm on `update_H` and
    # leave `update_E` to a separate module (the dispersive E products below).
    # Their predicates take NO side argument, so entering them in the loop above
    # would have asked the same question twice and credited `update_E` with a
    # verdict about `update_H`.
    add("complex_dispersive_spine", "complex_dispersive_pml_magnetic", "update_H",
        lambda: metal_complex_dispersive_pml_magnetic_coverage(
            fields, pml, residency, complex_probe))
    add("nonlinear_constitutive", "nonlinear_run_constitutive", "update_H",
        lambda: nonlinear_run_constitutive_coverage(fields, pml, "H", residency))

    add("offdiag_constitutive", "offdiag_constitutive", "update_E",
        lambda: offdiag_constitutive_coverage(fields, pml, residency))
    add("folded_offdiag_constitutive", "folded_offdiag_constitutive", "update_E",
        lambda: folded_offdiag_composition_coverage(fields, pml, residency))
    # THE PADE BODY IS THIS FAMILY'S update_E ARM. The claim that once stood here —
    # that the shared clause 10 keeps a nonlinear run's curls and update_H on the
    # array path, so update_E is its ONE slot — is no longer what the tree does:
    # three nonlinear-only spine arms were added above, and the composer selects
    # them. The clause-10 argument still holds for the ORDINARY products; it never
    # constrained this family's own spine.
    add("nonlinear_constitutive", "nonlinear_constitutive", "update_E",
        lambda: nonlinear_constitutive_coverage(fields, pml, residency))
    # TRANCHE 6, update_E. Seven single-slot families, each entered at the one
    # public predicate its arm consults. The three dispersive E products and the
    # two stored-E products are disjoint from `constitutive`/`null_constitutive`
    # through the susceptibility and `stores_E` clauses those two already name in
    # their refusals, which is why they are additions rather than replacements.
    add("dispersive_update_e", "dispersive_e", "update_E",
        lambda: metal_dispersive_e_coverage(fields, pml, residency))
    add("complex_dispersive_update_e", "complex_dispersive_e", "update_E",
        lambda: metal_complex_dispersive_e_coverage(fields, pml, residency,
                                                    complex_probe))
    add("folded_dispersive_update_e", "folded_dispersive_e", "update_E",
        lambda: folded_dispersive_e_coverage(fields, pml, residency))
    add("folded_offdiag_dispersive_update_e", "folded_offdiag_dispersive_e",
        "update_E",
        lambda: folded_offdiag_dispersive_coverage(fields, pml, residency))
    add("no_pml_stored_e", "stored_e", "update_E",
        lambda: metal_stored_e_coverage(fields, pml, residency))
    add("complex_no_pml_stored_e", "complex_stored_e", "update_E",
        lambda: metal_complex_stored_e_coverage(fields, pml, residency,
                                                complex_probe))
    add("complex_no_pml_offdiag_update_e", "complex_no_pml_offdiag", "update_E",
        lambda: metal_complex_no_pml_offdiag_coverage(fields, pml, residency,
                                                      complex_probe))
    # THE ONE NEW FAMILY THAT TAKES THE FOLDED COMPLEX ARTIFACT, not the unfolded
    # one: its predicate routes through `folded_complex._folded_complex_grid_
    # reasons`, so the record that licences its multiply is the folded fill's. The
    # fold is MANDATORY inside this single predicate (`_requires_a_fold`), so there
    # is no wide sibling to enter here by mistake.
    add("complex_folded_offdiag_update_e", "complex_folded_offdiag", "update_E",
        lambda: metal_complex_folded_offdiag_coverage(fields, pml, residency,
                                                      folded_complex_probe))
    # TRANCHE 6, update_P. `ade_update_p` registers a WIRED arm on this slot, so it
    # is asked like any other. See the docstring: the analyzer's `roll_up` discards
    # this entry unless it is patched in step.
    add("ade_update_p", "ade_update_p", "update_P",
        lambda: metal_ade_update_p_coverage(fields, pml, residency))

    # THE TWO SEAM SLOTS, in their OWN record. Not appended to `entries`, because
    # `entries` is what the analyzer rolls into the 759 denominator and a fill slot
    # is not one of the four sub-step names that denominator is built from. Kept as
    # an addendum so the fold's seam coverage is measured rather than either
    # silently dropped or silently inflating a cross-track number.
    fill_entries: List[Tuple[str, str, str, Callable[[], Any]]] = []
    for slot, side in (("fill_B", "B"), ("fill_D", "D")):
        fill_entries.append((
            "folded", "mirror_ghost_fill", slot,
            lambda s=side: mirror_ghost_fill_coverage(fields, s, residency)))
        fill_entries.append((
            "folded_complex", "folded_complex_fill", slot,
            lambda s=side: folded_complex_fill_coverage(fields, s, residency,
                                                        folded_complex_probe)))

    results: Dict[str, Any] = {}
    for family, label, sub_step, call in entries:
        results[f"{label}@{sub_step}"] = dict(
            _verdict(call), family=family, kernel=label, sub_step=sub_step)
    fill_results: Dict[str, Any] = {}
    for family, label, slot, call in fill_entries:
        fill_results[f"{label}@{slot}"] = dict(
            _verdict(call), family=family, kernel=label, sub_step=slot)

    # --- update_P: A PRODUCT, ASKED PER STATE AND PER COMPONENT ----------------
    #
    # WHAT CHANGED AND WHY THE OLD BLOCK WAS WRONG RATHER THAN MERELY STALE. It
    # wrote `metal_arm: None` and read the slot's reason out of
    # `launch.NO_ARM_REASONS` — a table that is now the EMPTY dict, because every
    # STEP_ORDER slot including `update_P` is armed. The `.get` therefore returned
    # None and the record said "no arm, and no reason either", which is the exact
    # shape the original comment was written to prevent ("nobody asked" read as "no
    # product carries this"). `ade_update_p` registers a wired arm; the composer
    # selected it on all fifteen polarization slots of the tranche-5 corpus.
    #
    # THE PER-COMPONENT VERDICT IS RECORDED BESIDE THE PER-STATE ONE because the
    # family exposes both and they answer different questions: the slot predicate
    # asks whether the recurrence may run at all, the component predicate asks it
    # for one driven field. A state admitted at the slot with a refusing component
    # is a real configuration and would otherwise be invisible.
    polarization: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        driven: Tuple[str, ...] = ()
        reader = getattr(state, "driven", None)
        if callable(reader):
            try:
                driven = tuple(reader())
            except BaseException:  # noqa: BLE001
                driven = ()
        components = {
            component: _verdict(
                lambda st=state, c=component: metal_ade_component_coverage(
                    fields, pml, st, c))
            for component in driven}
        polarization.append({
            "index": index,
            "kind": str(getattr(getattr(state, "susceptibility", None), "kind",
                                None)),
            "driven": list(driven),
            "metal_arm": "ADE update_P",
            "state": _verdict(
                lambda: metal_ade_update_p_coverage(fields, pml, residency)),
            "components": components,
            # KEPT, and now genuinely empty rather than accidentally so: the table
            # holds no entry because no STEP_ORDER slot is unarmed.
            "reason": launch.NO_ARM_REASONS.get("update_P"),
        })

    def ask(name: str, *args: Any) -> Any:
        attribute = getattr(grid, name, None)
        if attribute is None:
            return None
        if callable(attribute):
            try:
                return attribute(*args)
            except BaseException:  # noqa: BLE001
                return None
        return attribute

    configuration = {
        "shape": [int(v) for v in getattr(grid, "shape", ()) or ()],
        "xp": getattr(getattr(grid, "xp", None), "__name__", None),
        "force_complex_fields": bool(getattr(fields, "force_complex_fields", False)),
        "pml_active": bool(pml is not None and getattr(pml, "is_active", False)),
        "has_bloch": bool(ask("has_bloch") or False),
        "k_point": [float(v) for v in
                    (getattr(grid, "k_point", None) or (0.0, 0.0, 0.0))],
        "beta": float(getattr(grid, "beta", 0.0) or 0.0),
        "bfast_active": bool(getattr(grid, "bfast_active", False)),
        "cylindrical": bool(getattr(grid, "cylindrical", False)),
        "has_symmetry": bool(ask("has_symmetry") or False),
        "mirrored": [bool(ask("is_mirrored", axis)) for axis in range(3)],
        "is_axis": [bool(ask("is_axis", axis)) for axis in range(3)],
        "has_metallic": bool(ask("has_metallic") or False),
        "metallic": [bool(ask("is_metallic", axis)) for axis in range(3)],
        "has_nonlinearity": bool(getattr(fields, "has_nonlinearity", False)),
        "has_offdiagonal_epsilon": bool(
            getattr(fields, "has_offdiagonal_epsilon", False)),
        "has_conductivity": bool(getattr(fields, "has_conductivity", False)),
        "has_magnetic_conductivity": bool(
            getattr(fields, "has_magnetic_conductivity", False)),
        "stores_E": bool(getattr(fields, "stores_E", False)),
        "n_polarizations": len(states),
        "n_sources": (len(tuple(sources)) if sources is not None else None),
        "source_field_types": (
            [str(getattr(s, "field_type", "")) for s in tuple(sources)]
            if sources is not None else None),
    }
    try:
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415

        configuration["boundary_kinds"] = list(
            resolve(grid, pml if configuration["pml_active"] else None) or ())
    except BaseException as exc:  # noqa: BLE001
        configuration["boundary_kinds"] = (
            f"UNRESOLVED {type(exc).__name__}: {exc}"[:200])

    return {
        "predicates": results,
        "fill_predicates": fill_results,
        "polarization": polarization,
        "configuration": configuration,
        "plan_step": plan_step_admission(fields, pml, sources, complex_probe,
                                         beta_probe, folded_complex_probe,
                                         cylindrical_complex_probe),
        "probe_available": {"complex": complex_probe is not None,
                            "beta": beta_probe is not None,
                            "folded_complex": folded_complex_probe is not None,
                            "cylindrical_complex": (
                                cylindrical_complex_probe is not None),
                            "complex_path": COMPLEX_PROBE_PATH,
                            "beta_path": BETA_PROBE_PATH,
                            "folded_complex_path": FOLDED_COMPLEX_PROBE_PATH,
                            "cylindrical_complex_path":
                                CYLINDRICAL_COMPLEX_PROBE_PATH,
                            "harness_probe_backend": (
                                probe.get("backend") if isinstance(probe, dict)
                                else None),
                            "harness_probe_used": False},
        "subnormal_policy": subnormal.mps_policy_report(),
        "arm_table": [{"slot": slot, "family": spec.family, "label": spec.label,
                       "wired": bool(spec.wired)}
                      for slot in arms.registered_slots()
                      for spec in arms.registered(slot)],
    }


def plan_step_admission(fields: Any, pml: Any, sources: Any,
                        complex_probe: Any, beta_probe: Any,
                        folded_complex_probe: Any = None,
                        cylindrical_complex_probe: Any = None
                        ) -> Dict[str, Any]:
    """Run the SHIPPED ``launch.plan_step`` and record what it did, per slot.

    NOTHING IS PATCHED — no gate, no builder, no attribute of the lifted objects,
    and no clause is filtered out of a returned verdict. Both earlier rounds had to
    do one of those to get a number at all; this one does not, because the shipped
    predicates ask for the host they are running on.

    ``selected`` is the composer's answer. ``replaces`` is what the plan would take
    off the array path. ``residency`` is the whole-step verdict, which is a property
    of the composition rather than of any single plan, and it is recorded here
    rather than asserted: on a corpus row the driver's source list is real, so
    ``live_sub_steps`` sees seam passes this backend carries no product for, and the
    verdict refusing is the CORRECT answer for such a row rather than a defect.
    """
    from meep_gpu.metal_kernels import device, launch  # noqa: PLC0415

    record: Dict[str, Any] = {"selected": {}, "replaces": [], "reasons": {},
                              "residency_covered": None, "error": None}
    try:
        plan = launch.plan_step(
            fields, pml, residency=device.Residency(), sources=sources,
            complex_probe=complex_probe, beta_probe=beta_probe,
            folded_complex_probe=folded_complex_probe,
            cylindrical_complex_probe=cylindrical_complex_probe)
    except BaseException as exc:  # noqa: BLE001 - plan_step must never raise
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return record
    record["selected"] = dict(plan.selected)
    record["replaces"] = list(plan.replaces)
    record["reasons"] = {slot: list(reasons)[:3]
                         for slot, reasons in plan.reasons.items()}
    record["residency_covered"] = bool(plan.residency
                                       and plan.residency.covered)
    record["residency_reasons"] = list(
        getattr(plan.residency, "reasons", ()) or ())[:4]
    record["live"] = list(plan.live)
    return record

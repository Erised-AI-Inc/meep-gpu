#!/usr/bin/env python
"""DEVICE byte gate: the Triton H->D weld driven on a NONLINEAR run.

ONE BOARD CELL, TWO SEAM-INSTANCES, **AND NO NEW KERNEL**::

    H_to_D  (nonlinear run -> nonlinear run PML)  2
        examples:3rd-harm-1d.py
        tests:Test3rdHarm1d.test_3rd_harm_1d

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597).

**THIS GATE CERTIFIES NO NEW KERNEL, AND THAT IS THE HEADLINE RATHER THAN A FOOTNOTE.**
``meep_gpu/triton_kernels/nonlinear_fused_hd_pair.py`` adds one predicate, one plan
builder and one checkable identity. The kernel a covered configuration launches is
``fused_hd_pair.fused_constitutive_curl_H_to_D``, character for character, through
``fused_hd_pair.FusedHdPairPlan``. What is new is the ADMISSION -- and what this gate
measures is that the shipped weld, driven on a chi2/chi3 configuration, is bit-identical
to the array path over complete driver steps. The released ``fused_hd_pair`` artifact
does not contain that measurement, because its own predicate refuses every nonlinear
run through ``coverage._grid_reasons`` clause 10.

THE COVERAGE WAS NOT ALREADY OPEN ON THIS BACKEND, and this gate records the check
rather than asserting it: ``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` carries two rows
and neither is ``fused_hd_pair``'s, and ``launch.CERTIFIED_FUSED_PAIR_ARMS`` carries no
``fused_hd_pair`` row at all. (On the Metal board the analogous row has existed since
2026-09-05, so the 2026-09-07 Metal lane supplied an OWED MEASUREMENT there rather than
opening the cell. Here the cell is opened.)

THE MUTATIONS ARE CUT FROM ``fused_hd_pair.py``, in memory, because that is the file
whose bytes this admission launches. Nothing on disk is edited.

THE EXTRA LEG THIS GATE CARRIES is ``equivalence``: the family's own
``nonlinear_fused_hd_pair_equivalence`` states, as data, that this predicate is never
WIDER than the two shipped spine arms it claims to inherit -- the one failure this
construction can have.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_hd_tail_gate_kit as kit  # noqa: E402

from meep_gpu.triton_kernels import fused_hd_pair as _plain  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels import nonlinear_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import nonlinear_update_e as _nonlinear  # noqa: E402

GATE = "triton_nonlinear_fused_hd_pair"

#: The chi3 the fixtures install, per E component. The corpus rows' own form is a
#: SCALAR chi3 with chi2 zero (``3rd-harm-1d.py``'s ``k``); MEEP allocates the pair
#: together, so a zero chi2 rides with it.
CHI3 = 0.045
CHI2 = 0.0


class NonlinearProduct(kit.Product):
    """The kit's descriptor for this admission."""

    gate = GATE
    family = family
    complex_storage = False
    #: THE MUTATIONS ARE CUT FROM THE WELDED FAMILY'S FILE, because that is the file
    #: whose bytes a covered configuration launches. The kit reads it and execs an
    #: in-memory copy; nothing on disk is edited.
    family_path = "meep_gpu/triton_kernels/fused_hd_pair.py"
    cell_arms = (family.ARMS,)
    sources = (
        "meep_gpu/triton_kernels/nonlinear_fused_hd_pair.py",
        "meep_gpu/triton_kernels/fused_hd_pair.py",
        "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
        "meep_gpu/triton_kernels/nonlinear_update_e.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/withdraw_hoist.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/fastpath.py",
        "parity/meep_gpu/triton_hd_tail_gate_kit.py",
        "parity/meep_gpu/gate_triton_fused_hd_pair.py",
        "parity/meep_gpu/gate_triton_nonlinear_fused_hd_pair.py",
    )

    #: The corpus rows are 1-D with a METALLIC z termination and periodic x/y, so the
    #: fixtures carry that shape as well as the all-walled and all-periodic extremes.
    fixtures = (
        ("chi3_ppp", {"boundaries": {}}),
        ("chi3_mmm", {"boundaries": {axis: "metallic" for axis in "xyz"}}),
        ("chi3_ppm", {"boundaries": {"z": "metallic"}}),
        ("chi2_and_chi3_mmm",
         {"boundaries": {axis: "metallic" for axis in "xyz"}, "chi2": 0.02}),
    )
    mutation_fixture = "chi3_mmm"

    #: ONE kernel in the welded family, so every needle matches once.
    mutations: Dict[str, Dict[str, Any]] = {
        "m_own_cell_reads_the_pre_launch_H": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": ("        a = own0\n"
                    "        b = own1\n"
                    "        c = own2\n"),
            "new": ("        a = tl.load(hi0 + idx, mask=live, other=0.0)\n"
                    "        b = tl.load(hi1 + idx, mask=live, other=0.0)\n"
                    "        c = tl.load(hi2 + idx, mask=live, other=0.0)\n"),
            "why": "the curl's OWN cell reads the pre-launch magnetic field instead "
                   "of the register the constitutive half just computed -- the weld "
                   "not welding at all",
        },
        "m_foreign_tap_reads_the_stale_value": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": ("        value = a0\n"
                    "        if COMP == 1:\n"
                    "            value = a1\n"
                    "        if COMP == 2:\n"
                    "            value = a2\n"),
            "new": ("        _sidx = i * (ny * nz) + j * nz + k\n"
                    "        value = tl.load(hi0 + _sidx, mask=valid, other=0.0)\n"
                    "        if COMP == 1:\n"
                    "            value = tl.load(hi1 + _sidx, mask=valid, other=0.0)\n"
                    "        if COMP == 2:\n"
                    "            value = tl.load(hi2 + _sidx, mask=valid, "
                    "other=0.0)\n"),
            "why": "every foreign tap reads the PRE-LAUNCH H instead of recomputing "
                   "it -- exactly the value an in-place weld would have read at a "
                   "neighbour the launch had not yet written. This is the defect the "
                   "whole design removes",
        },
        "m_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": "        tl.store(ho0 + idx, own0, mask=live)\n",
            "new": "        tl.store(hi0 + idx, own0, mask=live)\n",
            "why": "the stepped magnetic field is written into the PRE-LAUNCH buffer "
                   "-- the race the scratch output exists to remove",
        },
        "m_f_w_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": "        tl.store(wo0 + idx, src0, mask=live)\n",
            "new": "        tl.store(wi0 + idx, src0, mask=live)\n",
            "why": "on this side the newly written f_w_H IS B exactly, so a foreign "
                   "recompute reads B where it needs B_prev",
        },
        "m_rotation_skipped": {
            "target": "host", "expected": "CAUGHT",
            "why": "the launch happens and the engine's H / f_w_H references are NOT "
                   "moved onto the freshly written scratch",
        },
        "m_half_integer_rebind": {
            "target": "host", "expected": "CAUGHT",
            "why": "the coefficient group is rebound to the HALF-INTEGER sub-lattice: "
                   "a half-cell error in the absorber profile, smooth and wrong",
        },
    }

    byte_neutral: Dict[str, Any] = {
        "hits": 1,
        "old": ("        a = own0\n"
                "        b = own1\n"
                "        c = own2\n"),
        "new": ("        a = tl.load(ho0 + idx, mask=live, other=0.0)\n"
                "        b = tl.load(ho1 + idx, mask=live, other=0.0)\n"
                "        c = tl.load(ho2 + idx, mask=live, other=0.0)\n"),
        "why": "the curl half re-loads its own cell's magnetic field from the scratch "
               "this program just wrote. PREDICTED NULL: one program's store and load "
               "of one address are ordered, so this is inert",
    }

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        return {"force_complex_fields": False, "dimensions": 3}

    def pml_spec(self, spec: Mapping[str, Any]) -> Any:
        return 2

    def configure(self, driver: Any, spec: Mapping[str, Any]) -> None:
        """Install MEEP's instantaneous chi2/chi3 per E component.

        THE PAIR IS INSTALLED TOGETHER, as MEEP does: ``set_chi3`` allocates a zero
        ``chi2`` and vice versa, and this engine's setter takes both maps.
        """
        shape = tuple(int(n) for n in driver.grid.shape)
        xp = driver.xp
        chi2 = float(spec.get("chi2", CHI2))
        chi3 = float(spec.get("chi3", CHI3))
        driver.fields.set_nonlinear_volumes(
            {name: xp.asarray(np.full(shape, np.float32(chi2), np.float32))
             for name in ("Ex", "Ey", "Ez")},
            {name: xp.asarray(np.full(shape, np.float32(chi3), np.float32))
             for name in ("Ex", "Ey", "Ez")})
        if not getattr(driver.fields, "has_nonlinearity", False):
            raise RuntimeError(
                "the chi2/chi3 volumes did not register: this fixture is a LINEAR "
                "run and every leg on it would be measuring the wrong admission")

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        return "nonlinear"

    def build_weld(self, driver: Any, kernel: Any = None, sources: Any = (),
                   module: Any = None) -> Any:
        """The plan is the WELDED family's, built through this admission.

        A mutated ``module`` is a copy of ``fused_hd_pair``; its KERNEL is handed to
        the shipped builder through the ``kernel=`` door, which is exactly how the
        admission forwards it. Reaching into the mutated module's own plan builder
        instead would bypass the admission this gate exists to certify.
        """
        if module is not None:
            kernel = module.fused_constitutive_curl_H_to_D_kernel()
        return family.plan_nonlinear_fused_hd_pair(
            driver.fields, driver.pml, sources=sources, kernel=kernel)

    def singles(self, driver: Any) -> Dict[str, Any]:
        fields, pml = driver.fields, driver.pml
        return {
            "update_H": _nonlinear.plan_nonlinear_run_constitutive(fields, pml, "H"),
            "step_D": _nonlinear.plan_nonlinear_run_pml_curl(fields, pml, "step_D"),
        }

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.nonlinear_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        from meep_gpu.triton_kernels import kernels as _kernels  # noqa: PLC0415

        name = type(owner).__name__
        if name == "FusedHdPairPlan":
            return _plain.fused_constitutive_curl_H_to_D_kernel()
        if name == "ConstitutivePlan":
            return _kernels.constitutive_step
        if name == "PmlCurlPlan":
            return _kernels.pml_curl_step
        if name == "NonlinearConstitutivePlan":
            return None
        return None

    def host_launcher(self, tag: str,
                      driver: Any) -> Optional[Callable[[Any], Any]]:
        if tag == "m_rotation_skipped":
            return kit.RotationSkipped
        if tag == "m_half_integer_rebind":
            return kit.half_integer_rebind(driver.pml)
        return None

    # -- host legs -----------------------------------------------------------
    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        """The welded family's lift, re-checked, PLUS the no-new-kernel declaration."""
        findings: List[str] = []
        constitutive_equal = (_plain.certified_constitutive_tail()
                              == _plain.lifted_constitutive_tail())
        curl_equal = _plain.certified_curl_tail() == _plain.lifted_curl_tail()
        if not constitutive_equal:
            findings.append(
                "the welded family's constitutive lift no longer equals the "
                "certified body")
        if not curl_equal:
            findings.append(
                "the welded family's curl lift no longer equals the certified body")
        if not family.CERTIFIES_NO_NEW_KERNEL:
            findings.append("this admission claims a kernel of its own")
        if family.INSTALLABLE != _plain.INSTALLABLE:
            findings.append(
                "the admission's INSTALLABLE differs from the welded family's, which "
                "is the flag launch._declared_uninstallable actually reads")
        # THE COVERAGE-WAS-NOT-ALREADY-OPEN CHECK, recorded as data.
        extra = getattr(triton_launch, "CERTIFIED_FUSED_PAIR_EXTRA_ARMS", {})
        primary = getattr(triton_launch, "CERTIFIED_FUSED_PAIR_ARMS", {})
        already_open = (_plain.FAMILY in extra) or (_plain.FAMILY in primary)
        measures = {
            "certifies_no_new_kernel": bool(family.CERTIFIES_NO_NEW_KERNEL),
            "welded_family": family.WELDED_FAMILY,
            "welded_kernel": family.WELDED_KERNEL,
            "welded_constitutive_lift_equal": constitutive_equal,
            "welded_curl_lift_equal": curl_equal,
            "extra_arms_rows_today": sorted(extra),
            "primary_arms_rows_with_hd": sorted(
                name for name in primary if "hd" in name),
            "the_cell_was_already_predicate_admitted": bool(already_open),
            "what_that_means": (
                "FALSE here: neither table carries a fused_hd_pair row on this "
                "backend, so no product admitted this cell before this admission "
                "existed. On the Metal board the analogous EXTRA_ARMS row has "
                "existed since 2026-09-05, and a lane there supplies an OWED "
                "measurement rather than opening the cell"),
            "arms": list(family.ARMS),
        }
        return findings, measures

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        fields, pml = driver.fields, driver.pml

        class _View:
            def __init__(self, inner: Any, **overrides: Any) -> None:
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_overrides", overrides)

            def __getattr__(self, name: str) -> Any:
                overrides = object.__getattribute__(self, "_overrides")
                if name in overrides:
                    return overrides[name]
                return getattr(object.__getattribute__(self, "_inner"), name)

        class _IntegratedElectric:
            """A source whose withdraw DOES WORK, by ``withdraw_hoist``'s own three
            conditions (a callable ``withdraw``, ``is_integrated``, and at least one
            source point). The third is not decoration: a stand-in without it is
            ADMITTED, and the refusal leg would then be measuring nothing -- caught on
            this lane's first device smoke, 2026-09-08."""

            component = "Ez"
            is_integrated = True
            field_type = "D"
            _n_source_points = 4

            def withdraw(self, _fields: Any) -> None:
                return None

        linear = _View(fields, has_nonlinearity=False)
        folded = _View(fields, grid=_View(fields.grid,
                                          is_mirrored=lambda axis: axis == 1,
                                          has_symmetry=lambda: True))
        return [
            {"name": "admits_the_fixture", "fields": fields, "sources": (),
             "must_refuse": False},
            {"name": "undeclared_source_list", "fields": fields, "sources": None,
             "must_refuse": True, "needle": "the source set was not declared"},
            {"name": "standing_integrated_electric_withdraw", "fields": fields,
             "sources": (_IntegratedElectric(),), "must_refuse": True,
             "needle": "standing integrated"},
            # THE INVERTED CLAUSE, IN BOTH DIRECTIONS: a LINEAR run belongs to the
            # ordinary weld and this admission must not overlap it.
            {"name": "linear_run_belongs_to_the_ordinary_weld", "fields": linear,
             "sources": (), "must_refuse": True,
             "needle": "no chi2/chi3 is installed"},
            {"name": "folded_grid", "fields": folded, "sources": (),
             "must_refuse": True, "needle": "folded"},
            {"name": "no_absorber", "fields": fields, "pml": None, "sources": (),
             "must_refuse": True},
        ]

    def arbitration_incumbents(self) -> Dict[str, str]:
        return {"note": "recorded per fixture in composer_selected. This admission "
                        "installs nothing of its own: it builds the welded family's "
                        "plan, and that family holds no CERTIFIED_FUSED_PRODUCTS row"}

    def extra_legs(self) -> Dict[str, Callable[[], Any]]:
        return {"equivalence": leg_equivalence}


PRODUCT = NonlinearProduct()


def leg_equivalence() -> List[Dict[str, Any]]:
    """The identity this admission rests on, DRIVEN rather than read.

    ``nonlinear_fused_hd_pair_equivalence`` evaluates the two shipped spine arms and
    the scoped weld predicate directly, and returns
    ``weld_never_wider_than_the_halves`` -- the one failure this construction can
    have. This leg runs it on every fixture and on the deliberately LINEAR view.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in PRODUCT.fixtures:
        driver = kit.build_driver(PRODUCT, spec, seed=20260907)
        try:
            identity = family.nonlinear_fused_hd_pair_equivalence(
                driver.fields, driver.pml, ())
        finally:
            driver.close()
        findings = ([] if identity["weld_never_wider_than_the_halves"]
                    else [f"{name}: the admission is WIDER than a half it inherits"])
        rows.append({"leg": "equivalence", "case": name, "passed": not findings,
                     "findings": findings, **identity})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    return kit.run(PRODUCT, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main())

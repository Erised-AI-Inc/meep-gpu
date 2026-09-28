"""A THIRD transcription of the no-absorber curl sub-step, plus the byte comparator.

Imported by both legs so the kernel is measured against the same reference on the
laptop (NumPy, against ``stepping.py``) and on the device (CuPy, against the
kernel). It deliberately does NOT import ``meep_gpu.stepping``: two independent
readings of one contract is the point, and an import would compare the kernel
against whatever the array path happens to do rather than against what was read
out of it. ``validate_no_pml_ref.py`` is what welds the two together.

Transcribed from, with line numbers checked against the tree on 2026-08-10:

* term tables        ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``      :213-223
* ghost rule         ``stepping._shift_up`` / ``_shift_down``          :1723 / :1787
* curl grouping      ``stepping._curl_from_operands``                  :1601
* ownership mask     ``stepping._mask_non_owned_cells``                :1865
* the update         ``stepping._apply_curl``, the plain branch        :508
* the E derivation   ``stepping._read_component``                      :2383
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

SEED = 20260810

# target, first (g1), first_axis, second (g2), second_axis
B_TERMS: Tuple[Tuple[str, str, int, str, int], ...] = (
    ("Bx", "Ez", 1, "Ey", 2),
    ("By", "Ex", 2, "Ez", 0),
    ("Bz", "Ey", 0, "Ex", 1),
)
D_TERMS: Tuple[Tuple[str, str, int, str, int], ...] = (
    ("Dx", "Hz", 1, "Hy", 2),
    ("Dy", "Hx", 2, "Hz", 0),
    ("Dz", "Hy", 0, "Hx", 1),
)

# vec.hpp iyee_shift, transcribed from ``fields.IYEE_SHIFTS`` (fields.py:214-219).
# The D family is shift-1 on its OWN axis; the B family is shift-1 on the two
# TRANSVERSE axes. So a metallic axis masks cell 0 of Bx on x alone, and of Dx on
# y and z — which is the asymmetry the kernel's BACKWARD branch encodes.
IYEE: Dict[str, Tuple[int, int, int]] = {
    "Bx": (0, 1, 1), "By": (1, 0, 1), "Bz": (1, 1, 0),
    "Dx": (1, 0, 0), "Dy": (0, 1, 0), "Dz": (0, 0, 1),
}


def _face(axis: int, index: int):
    return tuple(index if a == axis else slice(None) for a in range(3))


def shift_up(xp, field, axis: int, boundary: str):
    """field[i+1] with the far-face ghost rule (``stepping._shift_up``)."""
    shifted = xp.roll(field, -1, axis=axis)
    if boundary == "periodic":
        return shifted
    if boundary == "metallic":
        shifted[_face(axis, -1)] = 0
        return shifted
    raise ValueError(f"unknown boundary {boundary!r}")


def shift_down(xp, field, axis: int, boundary: str):
    """field[i-1] with the near-face ghost rule (``stepping._shift_down``)."""
    shifted = xp.roll(field, 1, axis=axis)
    if boundary == "periodic":
        return shifted
    if boundary == "metallic":
        shifted[_face(axis, 0)] = 0
        return shifted
    raise ValueError(f"unknown boundary {boundary!r}")


def derive_electric(xp, displacement: Dict[str, Any], inverse: Dict[str, Any]
                    ) -> Dict[str, Any]:
    """``stepping._read_component``'s derived branch: E = D * inv_eps, D on the LEFT."""
    return {"E" + name[1]: xp.multiply(displacement[name], inverse["E" + name[1]])
            for name in ("Dx", "Dy", "Dz")}


def reference_plain_step(xp, sources: Dict[str, Any], targets: Dict[str, Any],
                         dtdx, sub_step: str, boundaries: Tuple[str, str, str],
                         grouping: str = "array_order") -> None:
    """Advance the three targets one no-absorber curl sub-step, IN PLACE.

    ``grouping`` selects the parenthesisation:

    * ``array_order``  — ``dtdx * ((sf - f) + (s - ss))``, ``stepping.py``'s tree;
    * ``kernel_order`` — ``dtdx * (((sf - f) + s) - ss)``, C's left-to-right, the
      NEGATIVE CONTROL: it must NOT match at a non-power-of-two Courant.
    """
    backward = sub_step == "step_D"
    terms = D_TERMS if backward else B_TERMS
    shift = shift_down if backward else shift_up
    for target, first, first_axis, second, second_axis in terms:
        f = sources[first]
        s = sources[second]
        sf = shift(xp, f, first_axis, boundaries[first_axis])
        ss = shift(xp, s, second_axis, boundaries[second_axis])
        if grouping == "array_order":
            curl = dtdx * ((sf - f) + (s - ss))
        elif grouping == "kernel_order":
            curl = dtdx * (((sf - f) + s) - ss)
        else:
            raise ValueError(f"unknown grouping {grouping!r}")
        # stepping._mask_non_owned_cells: cell 0 of every METALLIC axis on which
        # this target's Yee shift is 0.
        for axis in range(3):
            if boundaries[axis] == "metallic" and IYEE[target][axis] == 0:
                curl[_face(axis, 0)] = 0
        targets[target] -= curl          # stepping._apply_curl, the plain branch


# ---------------------------------------------------------------------------
# The byte comparator (identical definitions to probe_triton_bit_identity)
# ---------------------------------------------------------------------------

def _ordered_key(raw: np.ndarray) -> np.ndarray:
    signed = raw.view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(-2147483648) - signed, signed)


def to_host(array: Any) -> np.ndarray:
    get = getattr(array, "get", None)
    return np.ascontiguousarray(get() if callable(get) else array)


def bit_compare(a_dev: Any, b_dev: Any) -> Dict[str, Any]:
    a, b = to_host(a_dev), to_host(b_dev)
    af, bf = a.ravel(), b.ravel()
    ua, ub = af.view(np.uint32), bf.view(np.uint32)
    identical = bool(np.array_equal(ua, ub))
    out: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": int(np.count_nonzero(ua != ub)),
        "total_floats": int(ua.size),
        # Recorded beside the byte verdict so the two can never be confused: a
        # magnitude comparison calls a signed-zero difference identical.
        "allclose_would_say": bool(np.allclose(af, bf, rtol=0, atol=0)),
    }
    if not identical:
        ulp = np.abs(_ordered_key(af) - _ordered_key(bf))
        out["max_ulp"] = int(ulp.max())
        out["max_abs_diff"] = float(
            np.max(np.abs(af.astype(np.float64) - bf.astype(np.float64))))
        bad = int(np.flatnonzero(ua != ub)[0])
        out["first_diff_index"] = bad
        out["first_diff_bits"] = [f"0x{int(ua[bad]):08x}", f"0x{int(ub[bad]):08x}"]
    return out


def combine(parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    identical = all(p["bit_identical"] for p in parts.values())
    out: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": sum(p["differing_floats"] for p in parts.values()),
        "total_floats": sum(p["total_floats"] for p in parts.values()),
        "per_array": parts,
    }
    if not identical:
        out["max_ulp"] = max(p.get("max_ulp", 0) for p in parts.values())
        out["max_abs_diff"] = max(p.get("max_abs_diff", 0.0) for p in parts.values())
    return out

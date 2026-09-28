"""Deferred raw-CUDA kernels for the MEEP-compatible FDTD core.

The active implementation remains the shared NumPy/CuPy array path. Modules in
this package are retained for the separately validated fused-kernel fast path
and are not imported by :mod:`meep_gpu`.
"""

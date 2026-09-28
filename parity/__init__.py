"""Certification and benchmark harness. Not part of the installed distribution.

``pyproject.toml`` packages ``meep_gpu`` only; nothing under this directory is
installed, and using the package needs nothing from it. ``parity/meep_gpu``
holds the gates and probes that certify each kernel family against the
package's own array path, the comparisons against stock MEEP, and the benchmark
drivers. Its artifact tree, ``parity/meep_gpu/results/``, is written by those
runs and is not tracked.
"""

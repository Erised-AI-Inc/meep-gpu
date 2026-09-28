# License and attribution

meep_gpu is derived from MEEP's C++ FDTD implementation. It is distributed under
MEEP's terms, the GNU General Public License, version 2 or (at your option) any
later version (SPDX identifier <code>GPL-2.0-or-later</code>).

| File at the repository root | Content |
|---|---|
| <code>LICENSE</code> | The full text of the GNU General Public License, version 2 |
| <code>NOTICE</code> | The attribution and copyright notices, including MEEP's |

Every file of the package is covered by that licence, whether or not the file
carries its own notice. Source citations of the MEEP files a module follows are
carried in the module.

## Relationship to MEEP

meep_gpu is an independent project. It is not affiliated with or endorsed by the
MEEP developers. It installs beside a MEEP installation and imports the
<code>meep</code> module of that installation when a simulation is lifted; it
does not contain, replace or modify MEEP.

MEEP is documented at <https://meep.readthedocs.io/> and developed at
<https://github.com/NanoComp/meep>.

## Other software

The package uses other software at run time when it is installed: NumPy, and,
depending on the host, CuPy, Triton or PyTorch. Each is distributed under its own
licence by its own project, and none is redistributed with this package.

This page is not legal advice and does not replace the notices in the released
files.

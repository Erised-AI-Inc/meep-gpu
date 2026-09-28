# How to cite

Cite this software, and cite MEEP, which builds every simulation this package
steps. The citation metadata is `CITATION.cff` at the root of the repository;
code hosts and reference managers read it directly.

> Ivan Biggs, Alicia Zeng and Yanlin Dou, *meep-gpu: single-GPU time stepping for
> MEEP simulations on NVIDIA and Apple hardware*, version 0.9.0 (2026),
> <https://github.com/Erised-AI-Inc/meep-gpu>.

> A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and
> S. G. Johnson, "MEEP: A flexible free-software package for electromagnetic
> simulations by the FDTD method", *Computer Physics Communications* 181,
> 687–702 (2010), doi:10.1016/j.cpc.2009.11.008.

In BibTeX:

```bibtex
@software{meep_gpu_2026,
  author  = {Biggs, Ivan and Zeng, Alicia and Dou, Yanlin},
  title   = {{meep-gpu: single-GPU time stepping for MEEP simulations on NVIDIA and Apple hardware}},
  version = {0.9.0},
  year    = {2026},
  url     = {https://github.com/Erised-AI-Inc/meep-gpu},
  license = {GPL-2.0-or-later}
}

@article{oskooi_2010,
  author  = {Oskooi, A. F. and Roundy, D. and Ibanescu, M. and Bermel, P. and
             Joannopoulos, J. D. and Johnson, S. G.},
  title   = {{MEEP}: A flexible free-software package for electromagnetic
             simulations by the {FDTD} method},
  journal = {Computer Physics Communications},
  volume  = {181},
  number  = {3},
  pages   = {687--702},
  year    = {2010},
  doi     = {10.1016/j.cpc.2009.11.008}
}
```

When you report a result computed with the package, state the version, the MEEP
version and precision, the kernel table that served (`table` in
`driver.fast_path_report()`), and the hardware.

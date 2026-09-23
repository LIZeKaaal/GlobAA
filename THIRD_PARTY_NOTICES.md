# Third-party notices

This file records the sources and licenses of code and data included in GlobAA.
The root [LICENSE](LICENSE) contains Apache-2.0. Third-party material retains
its original attribution and applicable license terms.

Copyright (c) 2026 lizekai (original contributions and modifications only).
Date: 2026-09-23.

## AndersonAcceleration

- Project: [yangliu-op/AndersonAcceleration](https://github.com/yangliu-op/AndersonAcceleration)
- Upstream author attribution: Yang Liu (also written as Liu Yang in the source)
- Upstream revision: [`2eb2947a172d5fd28e1452447cfac30c10acf30b`](https://github.com/yangliu-op/AndersonAcceleration/tree/2eb2947a172d5fd28e1452447cfac30c10acf30b)
- License: Apache License, Version 2.0; the complete text is in [LICENSE](LICENSE)
- GlobAA modifications and integration: lizekai, 2026

| Local file | Relationship to the upstream revision |
| --- | --- |
| `lsqr.py` | Numerical implementation reused unchanged; GlobAA attribution comments added |
| `Anderson.py` | Adapted and extended with FAA coefficient filtering, retained-history management, effective-memory diagnostics, and related checks |
| `optim_algo.py` | Adapted to fixed-point maps and relative residuals; changed restart and residual-safeguard behavior; extended controls and recording; removed unused upstream routines/branches; added `GlobalAndersonSolver` |

Original source attributions are retained. The upstream revision has no
`NOTICE` file; this document provides GlobAA's own provenance summary.

The upstream repository accompanies:

Wenqing Ouyang, Yang Liu, and Andre Milzarek. *Descent Properties of an Anderson
Accelerated Gradient Method With Restarting*. arXiv:2206.01372.
[Paper](https://arxiv.org/abs/2206.01372).

## SciPy-derived LSQR material

The header of the upstream `lsqr.py` identifies it as a PyTorch adaptation and
modification of `scipy.sparse.linalg.lsqr`. Its SciPy-derived material is subject
to the SciPy BSD-3-Clause terms, in addition to the attribution to the direct
upstream project above.

The complete SciPy copyright notice, conditions, and disclaimer are included in
[licenses/SCIPY-BSD-3-Clause.txt](licenses/SCIPY-BSD-3-Clause.txt). This license
text is copied from SciPy 1.11.4, the version in the reference environment, and
matches [that release's license](https://github.com/scipy/scipy/blob/v1.11.4/LICENSE.txt).
The exact historical SciPy revision adapted by the upstream author is not
specified in `lsqr.py`.

The LSQR research references already present in the source are retained.

## Madelon dataset

The bundled Madelon training files are attributed to Isabelle Guyon and the
UCI Machine Learning Repository, under CC BY 4.0. See
[datasets/ISTA/README.md](datasets/ISTA/README.md) for the source, citation,
license link, and the way the experiment uses these files.

## Installed dependencies

NumPy, SciPy, PyTorch, Matplotlib, and scikit-fem are installed separately via
`requirements.txt`. They retain their respective licenses. Their package source
trees are not bundled here; the copied SciPy-derived LSQR material is addressed
separately above.

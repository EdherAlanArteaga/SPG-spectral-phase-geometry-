## SPG — Spectral Phase-space Geometry

Python implementation of the SPG framework for brain state analysis.

**Author:** Edher Alan Arteaga Marroquin  
**Preprint:** https://doi.org/10.5281/zenodo.20712285

---

## Observables

| Observable | Module | Paper |
|---|---|---|
| `lambda2` | `spg_core.py` | Mathematical foundation |
| `C_dyn` | `spg_core.py` | Mathematical foundation (Theorem 1) |
| `d_eff` | `spg_manifold.py` | Paper 2 — EEG manifold |
| `CI` | `spg_manifold.py` | Paper 2 — EEG manifold |
| `lambda_star` | `spg_crossover.py` | Mathematical foundation (Theorem 2) |
| `D_cross` | `spg_crossover.py` | Mathematical foundation (Theorem 2) |

---

## Installation

```bash
pip install numpy scipy
```

## Quick start

```python
import numpy as np
from spg_core import spg
from spg_manifold import eeg_to_manifold

# Single window
signal = np.random.randn(1024)
result = spg(signal)
print(result['lambda2'], result['C_dyn'])

# Full EEG recording (single channel)
fs = 256  # Hz
recording = np.random.randn(fs * 120)  # 2 minutes
manifold = eeg_to_manifold(recording, fs=fs)
print(f"d_eff = {manifold['d_eff']:.3f}  CI = {manifold['CI']:.3f}")
```

## Datasets

The EEG datasets used in Paper 2 are publicly available:

- **Sleep-EDF:** https://physionet.org/content/sleep-edfx/
- **DoC (disorders of consciousness):** https://figshare.com/...

Download and place EDF files in `data/` before running reproduce scripts.

## Files

```
spg_core.py       Core pipeline: signal -> lambda2, C_dyn
spg_manifold.py   Manifold position: (lambda2, C_dyn) -> d_eff, CI
spg_crossover.py  Spectral crossover: D_cross, lambda_star (Theorem 2)
README.md         This file
```


# HistoRange

[![tests](https://github.com/JiayuEllieSu/HistoRange/actions/workflows/tests.yml/badge.svg)](https://github.com/JiayuEllieSu/HistoRange/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

HistoRange is a computational framework for spatial transcriptomics that uses registered H&E histology to define a spatially varying anisotropic transport geometry for secreted ligand signaling. It separates spatial association detection from post-detection effective-range estimation so that a fitted scale is reported quantitatively only when the data support it.

The code implements the core HistoRange workflow, including molecular scoring, histology-derived transport geometry, finite-element transport, contact exposure, covariance-aware spatial association testing, range estimation, and the accompanying specificity, robustness, and histology-evidence calculations.

## How it works

1. **Histology.** The registered RGB image is converted to optical density, separated into haematoxylin and eosin channels with a fixed stain basis, and thresholded into a tissue support that keeps internal holes (lumens, air spaces) as barriers.
2. **Geometry.** A structure tensor on the eosin channel gives local tissue-line orientation θ and coherency. Coherent, uncrossed regions get an anisotropic tensor with det = 1 and at most a 3:1 conductance ratio; weak or crossing regions revert to isotropic transport.
3. **Transport.** For a secreted interaction, the ligand source field is propagated through the screened equation −∇·(A∇u) + ℓ⁻²u = q with no-flux boundaries, solved with P1 finite elements at every candidate range ℓ.
4. **Detection.** A spatial GLS model with Matérn-3/2 residual covariance tests whether the propagated exposure predicts the receptor response. A parametric bootstrap with a fixed wrong-ligand tail calibration gives secreted p-values; contact-annotated interactions use a local-adjacency exposure and the raw bootstrap p. All tests in a specimen share one Benjamini–Hochberg family.
5. **Range.** Only detected secreted interactions enter a separate all-bin negative-binomial estimator (T1). A range is reported only when the optimum is interior to the trusted grid and its support statistic exceeds the specimen's near-null threshold, using simulation-calibrated, boundary-aware bands.
6. **Evidence layers.** Source specificity, range robustness, and the held-out contribution of matched histology are evaluated separately and never change a detection verdict.

## Installation

HistoRange needs Python 3.10 or newer, NumPy and SciPy.

```bash
git clone https://github.com/JiayuEllieSu/HistoRange.git
cd HistoRange
pip install .
```

For development (tests, linting, building):

```bash
pip install -e ".[dev]"
```

The import name is `histrange`.

## Quickstart

You provide a registered H&E image `image` (`rows × cols × 3` RGB array) with its `pixel_size_um`, observation centres `coords` (`n × 2`, in microns, `x` = column and `y` = row of the image), raw `ligand_counts`, `receptor_counts` and `library_size` per observation, the spot or bin size `spot_um`, and the T1 statistics of the specimen's near-null modules `near_null_lambdas`.

```python
import numpy as np

import histrange
from histrange import molecular, projection, spatial, transport
from histrange import range as range_estimation  # aliased so it does not shadow the builtin
from histrange.settings import trusted_range_grid

# Histology -> geometry -> mesh -> transport operator
hist = histrange.prepare_histology(image, pixel_size_um=pixel_size_um)
geo = histrange.build_geometry(hist)
mesh = histrange.build_mesh(hist)  # 4 um mesh; 8 and 16 um fallbacks for large sections
op = transport.assemble(mesh, geo)
op.validate()  # raises OperatorError if symmetry, mass or no-flux checks fail

# Observations off tissue or inside holes are dropped, never projected onto tissue
keep = mesh.contains(coords)
coords, ligand_counts = coords[keep], ligand_counts[keep]
receptor_counts, library_size = receptor_counts[keep], library_size[keep]
src = projection.source_operator(mesh, coords, radius_um=spot_um / 2)
rec = projection.receiver_operator(mesh, coords, radius_um=spot_um / 2)

# Exposure fields over the specimen's trusted range grid
d_nn = spot_um  # median nearest-neighbour distance between observations
diameter = mesh.geodesic_diameter()
ranges = trusted_range_grid(mesh.spacing_um, d_nn, diameter)
load = projection.deposit(src, molecular.bounded_score(ligand_counts, library_size))
exposures = [projection.interpolate(rec, op.solve(ell, load)) for ell in ranges]

# Detection: spatial GLS on receptor-competent receivers
elig = molecular.eligibility(ligand_counts, receptor_counts, library_size)
r = elig["receptor_competent"]
y = molecular.bounded_score(receptor_counts, library_size)[r]
baseline = np.column_stack([np.ones(r.sum()), np.log(library_size[r])])
cov = spatial.fit_covariance(y, baseline, coords[r], spatial.phi_bounds(d_nn, diameter))
fit = spatial.profile(y, baseline, [h[r] for h in exposures], cov)
p_sec = spatial.calibrated_secreted_pvalue(spatial.bootstrap_pvalue(fit))  # 9,999 draws
# Collect p_sec (secreted) and raw p_boot (contact) for every eligible interaction, then
# q = spatial.benjamini_hochberg(all_p); detected = q <= 0.05

# Range: detected secreted interactions only, on all bins including receptor zeros
t1 = range_estimation.fit_range_profile(
    receptor_counts, library_size, np.zeros((len(coords), 0)), exposures, ranges
)
tau = range_estimation.support_threshold(near_null_lambdas)
claim = range_estimation.report_range(
    t1["ell_hat"], t1["best_index"], len(ranges), t1["Lambda"], tau, ranges[0], ranges[-1], level=0.95
)
print(claim)  # {'status': 'interval' | 'upper_bound' | 'uninformative' | 'unresolved', ...}
```

### Full worked example

[`examples/synthetic_walkthrough.py`](examples/synthetic_walkthrough.py) runs the whole secreted workflow on a simulated 400 × 400 µm section with a known fibre angle and a known signaling range, including a Benjamini–Hochberg family, the near-null support threshold, boundary-aware reporting, and the anisotropy and mesh robustness arms:

```bash
python examples/synthetic_walkthrough.py
```

It finishes in about half a minute and prints each stage, ending with the range claims at 90% and 95%.

## Package layout

| Module | Contents | Methods |
|---|---|---|
| `histrange.molecular` | CP10K, complex counts, bounded activation scores, QC, eligibility | §3.2–3.4 |
| `histrange.histology` | Optical density, stain separation, tissue support | §3.5 |
| `histrange.geometry` | Structure tensor, crossing statistic, transport tensor; isotropic, rotated and morphology-mismatch controls | §3.6, §3.18 |
| `histrange.mesh` | P1 lattice mesh with mesh ladder, point location, geodesic diameter | §3.8 |
| `histrange.transport` | Anisotropic stiffness and mass assembly, screened solves, operator validation | §3.7–3.8 |
| `histrange.projection` | Footprint source deposition and receiver averaging | §3.8 |
| `histrange.contact` | Local-adjacency exposure for contact-annotated interactions | §3.10 |
| `histrange.spatial` | Matérn-3/2 REML covariance (dense or Vecchia), GLS range profile, bootstrap, calibration, BH | §3.11–3.12 |
| `histrange.vecchia` | Max–min ordering and sparse Vecchia factors | §3.11 |
| `histrange.range` | T1 negative-binomial range profile, support threshold, boundary-aware reporting | §3.13–3.15 |
| `histrange.evidence` | Checkerboard folds, partial R², geometry gain, robustness criteria | §3.16–3.18 |
| `histrange.settings` | Every fixed analysis setting, and the master and trusted range grids | §3.9 |

Detailed function-level documentation is in [`docs/api.md`](docs/api.md). Every fixed setting, its value, and the Methods section it comes from are listed in [`docs/settings.md`](docs/settings.md). The settings are frozen to the values reported in the manuscript; change them only for sensitivity analyses, and report any change.

## Running the tests

```bash
pip install -e ".[test]"
pytest              # full suite, about 30 s
pytest -m "not slow"  # skip the end-to-end example
```

The suite checks exact discrete identities (mass conservation, constant solutions, P1 interpolation exactness, exact Vecchia factors at full conditioning), recovery of known orientations, anisotropy ratios and ranges on synthetic data, the frozen eligibility and reporting thresholds, and that the README quickstart and the example run end to end. No real data are needed.

## Data availability

All datasets analysed in the manuscript are public:

- **IPMN benchmark** — Visium, GEO [GSE233254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233254) (sections GSM7421780 LG_1, GSM7421787 HG_1, GSM7421792 PDAC_3); cell-type annotations in [GSE233293](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233293).
- **Colorectal cancer** — Visium HD, GEO [GSE280315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315) (GSM8594568); matched Xenium in [GSE280314](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280314).
- **Lung adenocarcinoma** — 10x Genomics Visium HD dataset `Visium_HD_Human_Lung_Cancer_HD_Only_A`.
- **Pancreatic ductal adenocarcinoma** — 10x Genomics Visium HD 3′ dataset `Visium_HD_3prime_Human_Pancreatic_Cancer`.
- **EGFR-mutant NSCLC** — GEO [GSE301973](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE301973).

## Citation

If you use HistoRange, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button). A manuscript reference will be added on publication.

## License

MIT — see [LICENSE](LICENSE).

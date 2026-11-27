# Basin-Preserving Discretizations of Modern Hopfield Retrieval Dynamics

Code that reproduces every numerical result and figure of

**Francisco R. Villatoro**, *Basin-Preserving Discretizations of Modern
Hopfield Retrieval Dynamics: Energy Cells, Dissipation, and the Attention
Limit* (2026).

Repository: https://github.com/FrancisRVillatoro/modern-hopfield-basin-discretizations  
Archive: https://doi.org/10.5281/zenodo.22048178 (all versions; the DOI resolves to the latest one)

The repository contains code only. Running the scripts regenerates every
data file and both figures of the article.

## Contents

| Directory | Script | Results in the article |
|---|---|---|
| `scripts/` | `experiments.py` | Campaigns A, B and D; Table 1; the EE, CS and ETD rows of Table 2 |
| | `expSAVg.py` | SAV row of Table 2; Campaign F |
| | `expE2.py` | Campaign E: basin labels, disagreement fractions, distances to the separatrices, Figure 1 (`fig_basins2.pdf`) |
| | `expI2.py` | Campaign I: convergence orders and energy-identity residual |
| `campaign_E_audit/` | `certify_planar_critical_points.py` | Campaign E: interval enclosure of the seven critical points, Morse indices and escape levels |
| | `audit_planar_horizons.py` | Campaign E: horizon and tolerance checks, boundary nodes, rerun of the discrete maps, interval energy gaps of the disagreeing nodes |
| | `audit_independent_boundary.py` | Campaign E: DOP853 integration of the 894 boundary nodes |
| `campaigns_G_H_audit/` | `audit_gh.py` | Campaign G (Table 3) and Campaign H (Table 4, dissipation ratios, separation and spectral covariates) |
| `campaign_I_audit/` | `audit_campaign_i.py` | Campaign I: nonlinear residuals, solver status, energy-identity defects and work counts |
| `campaign_J/` | `campaign_j.py` | Campaign J in dimension 32: recall and disagreement rates, Figure 2 (`fig_recall_corruption.pdf`) |
| | `audit_terminal_states.py` | Campaign J: Hessians at the endpoints, endpoint distances and paired bootstrap interval |
| `campaign_J_robustness/` | `campaign_j_robustness.py` | Dimension and temperature controls of Campaign J (Table 5) |
| `low_temperature_bounds/` | `low_temperature_bounds.py` | Bounds of Section 4.4 quoted in Remark 4.13 and in Campaign E |

Each directory other than `scripts/` has its own README with details.

## Installation

The scripts need Python 3 with NumPy, SciPy, Matplotlib and mpmath.
`requirements.txt` and `environment.yml` pin the tested versions: NumPy 2.3.5,
SciPy 1.17.0, Matplotlib 3.10.8 and mpmath 1.3.0. The scripts in `scripts/`
were tested with CPython 3.13 and the other scripts with CPython 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

With conda, `conda env create -f environment.yml` creates the same environment.

## Running the scripts

Run from the repository root. The scripts are independent and can be run in
any order.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/experiments.py
python scripts/expSAVg.py
python scripts/expE2.py
python scripts/expI2.py
python campaign_E_audit/certify_planar_critical_points.py
python campaign_E_audit/audit_planar_horizons.py
python campaign_E_audit/audit_independent_boundary.py
python campaigns_G_H_audit/audit_gh.py campaigns_G_H_audit/results_gh.json
python campaign_I_audit/audit_campaign_i.py campaign_I_audit/output
python campaign_J/campaign_j.py
python campaign_J/audit_terminal_states.py campaign_J/audit
python campaign_J_robustness/campaign_j_robustness.py
python low_temperature_bounds/low_temperature_bounds.py
```

The scripts in `scripts/` write JSON files to `results/` and the Campaign E
figure to `figures/`. The Campaign E audits, `campaign_j.py` and
`campaign_j_robustness.py` write their JSON, CSV and PDF files to their own
directories. `audit_gh.py`, `audit_campaign_i.py` and
`audit_terminal_states.py` write to the path given on the command line, and
`low_temperature_bounds.py` prints its values. Git ignores all generated files.
The longest computations are the two-dimensional basin calculations of
Campaign E and the flow integrations of Campaign J and its controls.

## Random seeds

Unless a script states otherwise, random draws use the seed `20260809`. The ten
pattern instances of Campaign H use the seeds 1 to 10. Campaign J uses the
seed root `20261122`, and its dimension and temperature controls use
`2026112207`.

## Changes from version 1.0.0

Version 1.0.0 (https://doi.org/10.5281/zenodo.22048179) archived the
first-generation scripts together with their outputs. This version adds the
code of the revised analyses: the audits of Campaigns E and I, the thirty-start
Campaign G and ten-seed Campaign H runs, Campaign J with its controls, and the
evaluation of the low-temperature bounds. It removes the generated JSON files
and figures, which the scripts regenerate, the superseded scripts `expG.py` and
`expH2.py`, the legacy and pilot scripts, the earlier revision scripts in
`siads-revision/`, which the audits supersede, and the files `MANIFEST.md`,
`SHA256SUMS.txt` and `verify_repository.py`, which described version 1.0.0.

## Funding

This work was supported by the Universidad de Málaga, Spain, through project
PRO-B4-2026-011 of its Plan Propio de Investigación, Transferencia y
Divulgación Científica.

## License

The code is released under the MIT License.

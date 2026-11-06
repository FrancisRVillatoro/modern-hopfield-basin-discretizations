# Basin-Preserving Discretizations of Modern Hopfield Retrieval Dynamics

Reproducibility code and archived numerical outputs for

**Francisco R. Villatoro**, *Basin-Preserving Discretizations of Modern
Hopfield Retrieval Dynamics: Energy Cells, Dissipation, and the Attention
Limit* (2026).

Repository: https://github.com/FrancisRVillatoro/modern-hopfield-basin-discretizations

## What is in this repository?

All eight Python files and all seven JSON result files present in the archived
working bundle have been retained.  The scripts have only been made portable:
hard-coded `/home/claude/...` output paths were replaced by repository-relative
`results/` and `figures/` directories.  The numerical algorithms and parameters
were not changed.

The current-paper reproducibility scripts are:

- `scripts/experiments.py`: baseline Campaigns A--F, first-generation driver.
- `scripts/expSAVg.py`: authoritative gamma-resplit SAV results used for
  Campaign C (SAV column) and Campaign F.
- `scripts/expE2.py`: authoritative Campaign E; generates both
  `results/resultsE2.json` and `figures/fig_basins2.pdf`.
- `scripts/expG.py`: Campaign G.
- `scripts/expH2.py`: Campaign H.
- `scripts/expI2.py`: authoritative Campaign I discrete-gradient benchmark.

Historical material is retained for provenance:

- `scripts/legacy/expI.py` and `results/resultsI.json`: first-generation
  Campaign I, superseded by `expI2.py` / `resultsI2.json`.
- `scripts/pilot/expJ_pilot.py`: an exploratory Campaign J pilot, not used in
  the manuscript.
- `figures/fig_basins.pdf`: first-generation Campaign E figure, no longer used
  in the current manuscript.
- `scripts/legacy/expE1_legacy.py`: reconstruction of the missing standalone
  script for that legacy figure from the Campaign E code in `experiments.py`.
  The original ZIP contained the PDF but not its standalone plotting script.

## Authoritative result mapping

| Manuscript campaign | Authoritative script | Archived result/output |
|---|---|---|
| A | `scripts/experiments.py` | `results/results.json` |
| B | `scripts/experiments.py` | `results/results.json` |
| C: EE/CS/ETD | `scripts/experiments.py` | `results/results.json` |
| C: SAV | `scripts/expSAVg.py` | `results/resultsSAVg.json` |
| D | `scripts/experiments.py` | `results/results.json` |
| E | `scripts/expE2.py` | `results/resultsE2.json`, `figures/fig_basins2.pdf` |
| F | `scripts/expSAVg.py` | `results/resultsSAVg.json` |
| G | `scripts/expG.py` | `results/resultsG.json` |
| H | `scripts/expH2.py` | `results/resultsH.json` |
| I | `scripts/expI2.py` | `results/resultsI2.json` |

`results/results.json` is intentionally retained unchanged.  Its first-generation
Campaign E/F and SAV values are superseded by `resultsE2.json` and
`resultsSAVg.json`, respectively.

## Figures

The current manuscript uses

```text
figures/fig_basins2.pdf
```

and `scripts/expE2.py` explicitly generates this file.

The previous manuscript figure

```text
figures/fig_basins.pdf
```

is retained as a legacy output.  Its standalone generator was absent from the
working ZIP, so `scripts/legacy/expE1_legacy.py` was reconstructed from the
first-generation Campaign E code in `experiments.py` and the archived figure.

## Installation

A tested environment is recorded in `requirements.txt` and `environment.yml`.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the current manuscript results

Run from the repository root:

```bash
python scripts/experiments.py
python scripts/expSAVg.py
python scripts/expE2.py
python scripts/expG.py
python scripts/expH2.py
python scripts/expI2.py
```

The expensive two-dimensional basin calculation is `expE2.py`.

To regenerate the historical first-generation basin figure:

```bash
python scripts/legacy/expE1_legacy.py
```

## Validation

Run

```bash
python verify_repository.py
```

to check required files, JSON syntax, Python syntax, and the archived figure
hashes listed in `SHA256SUMS.txt`.

## Random seeds

The principal fixed seed in the manuscript is `20260809`.  Campaign H uses
the explicit seed set `(1, 2, 3)` as recorded in `expH2.py`.

## Funding

This work was supported by research project **PPRO-B4-2026-004**.

## Citation and preservation

The repository is prepared for a GitHub release `v1.0.0` and archival through
Zenodo.  `.zenodo.json` is included for Zenodo metadata and `CITATION.cff` for
GitHub citation metadata.  After Zenodo mints the DOI, add the version-specific
DOI to the manuscript and, optionally, to `CITATION.cff` in a post-release
commit.

## License

The software in this repository is released under the MIT License.

## siads-revision/

Validation and replication scripts added for the revised version (horizon and tolerance audit of Campaign E; Campaign H over ten seeds; Campaign G over thirty initializations), with their raw outputs. The Zenodo record 10.5281/zenodo.22048179 archives the original package and is unchanged.

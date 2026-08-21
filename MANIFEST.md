# Repository audit and manifest

## Audit of the supplied working ZIP

Original Python files found: **8**
- expJ_pilot.py
expI.py
expSAVg.py
expI2.py
expG.py
expH2.py
expE2.py
experiments.py

Original JSON files found: **7**
- results.json
resultsE2.json
resultsH.json
resultsI2.json
resultsSAVg.json
resultsG.json
resultsI.json

Both archived basin figures were present:
- `fig_basins2.pdf`
- `fig_basins.pdf`

### Figure-generation audit

- `expE2.py` explicitly states `Outputs: resultsE2.json and fig_basins2.pdf`
  and contains the `savefig` call for `fig_basins2.pdf`.
- No original Python file in the supplied ZIP contains `fig_basins.pdf` or a
  `savefig` call for that filename.
- Therefore the standalone generator of the old figure was genuinely absent.
  `scripts/legacy/expE1_legacy.py` has been reconstructed from the first-
  generation Campaign E implementation in `experiments.py`; it is labelled
  as reconstructed and is not presented as the lost original file.

## Files deliberately excluded from the GitHub reproducibility package

The supplied working ZIP also contained obsolete manuscript assembly files
(`main.tex`, `body.tex`, section fragments, `main.pdf`) and an internal audit
note.  They are not required to reproduce the numerical claims and were
excluded to avoid mixing outdated manuscript drafts with the code/data release.

The current manuscript itself should be deposited through arXiv/journal
submission, while this repository is the persistent code-and-results object.

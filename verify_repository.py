from pathlib import Path
import json, py_compile, hashlib, sys

ROOT = Path(__file__).resolve().parent

required = [
    "README.md", "LICENSE", "CITATION.cff", ".zenodo.json",
    "requirements.txt", "environment.yml", "MANIFEST.md",
    "scripts/experiments.py", "scripts/expSAVg.py", "scripts/expE2.py",
    "scripts/expG.py", "scripts/expH2.py", "scripts/expI2.py",
    "scripts/legacy/expI.py", "scripts/legacy/expE1_legacy.py",
    "scripts/pilot/expJ_pilot.py",
    "results/results.json", "results/resultsE2.json", "results/resultsG.json",
    "results/resultsH.json", "results/resultsI.json", "results/resultsI2.json",
    "results/resultsSAVg.json",
    "figures/fig_basins.pdf", "figures/fig_basins2.pdf",
]
missing = [p for p in required if not (ROOT/p).is_file()]
if missing:
    print("MISSING:")
    print("\n".join(missing))
    raise SystemExit(1)

for p in (ROOT/"results").glob("*.json"):
    with p.open(encoding="utf-8") as fh:
        json.load(fh)

for p in (ROOT/"scripts").rglob("*.py"):
    py_compile.compile(str(p), doraise=True)

for p in [ROOT/"figures"/"fig_basins.pdf", ROOT/"figures"/"fig_basins2.pdf"]:
    if not p.read_bytes().startswith(b"%PDF"):
        raise RuntimeError(f"{p} is not a PDF")

print("Repository structure: OK")
print("JSON files: OK")
print("Python syntax: OK")
print("Archived PDFs: OK")

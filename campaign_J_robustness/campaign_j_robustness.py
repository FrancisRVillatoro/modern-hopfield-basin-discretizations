#!/usr/bin/env python3
"""Run the paired dimension and temperature controls for Campaign J."""
import argparse
import csv
import runpy
from pathlib import Path

DIMENSION = 64
LOADS = (32, 128)
RHO = 0.6
NOISE_LEVELS = (0.5, 2.0)
TEMPERATURES = (12, 18, 24)
BANKS = 16
SEED = 2026112207
BOOTSTRAPS = 10_000


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, np):
    rng = np.random.default_rng(SEED + 1)
    draws = {size: rng.integers(BANKS, size=(BOOTSTRAPS, BANKS)) for size in LOADS}
    rates = {
        "recall_raw_cue": "cues",
        "recall_one_step": "cues",
        "recall_iterated_attention": "cues",
        "recall_flow": "cues",
        "recall_overrelaxed": "cues",
        "disagree_attention_flow": "resolved_attention_flow",
        "disagree_overrelaxed_flow": "resolved_overrelaxed_flow",
    }
    counts = (
        "cues",
        "flow_unresolved_T40",
        "flow_unresolved_T80",
        "flow_unresolved_T320",
        "flow_audit_endpoint_changes",
        "flow_audit_label_changes",
        "iterated_attention_unresolved",
        "overrelaxed_unresolved",
        "resolved_attention_flow",
        "resolved_overrelaxed_flow",
    )
    summary = []
    for size in LOADS:
        for beta_d in TEMPERATURES:
            for sigma in NOISE_LEVELS:
                subset = [
                    row for row in rows
                    if (row["patterns"], row["beta_d"], row["sigma"]) == (size, beta_d, sigma)
                ]
                result = {
                    "dimension": DIMENSION, "patterns": size, "load": size / DIMENSION,
                    "rho": RHO, "beta_d": beta_d, "sigma": sigma, "banks": BANKS,
                }
                result.update({field: sum(row[field] for row in subset) for field in counts})
                for field, denominator in rates.items():
                    hits = np.array([row[field] for row in subset])
                    totals = np.array([row[denominator] for row in subset])
                    indices = draws[size]
                    sampled_hits = hits[indices].sum(axis=1)
                    sampled_totals = totals[indices].sum(axis=1)
                    valid = sampled_totals > 0
                    lo, hi = np.quantile(sampled_hits[valid] / sampled_totals[valid], [0.025, 0.975])
                    result[field] = hits.sum() / totals.sum()
                    result[field + "_lo"] = lo
                    result[field + "_hi"] = hi
                summary.append(result)
    return summary


def main():
    this_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-script", type=Path,
        default=this_path.parent.parent / "campaign_J" / "campaign_j.py",
        help="Path to the original Campaign J implementation.",
    )
    parser.add_argument("--output", type=Path, default=this_path.parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    replicate = runpy.run_path(str(args.base_script))["campaign_replicate"]
    parameters = replicate.__globals__
    np = parameters["np"]
    parameters.update(
        DIMENSION=DIMENSION,
        NOISE_LEVELS=NOISE_LEVELS,
        RESIDUAL_LIMIT=1e-9 * np.sqrt(DIMENSION),
        ENDPOINT_LIMIT=1e-6 * np.sqrt(DIMENSION),
        BASE_SEED=SEED,
    )
    rows = []
    load_seeds = np.random.SeedSequence(SEED).spawn(len(LOADS))
    for size, load_seed in zip(LOADS, load_seeds):
        bank_seeds = load_seed.spawn(BANKS)
        for beta_d in TEMPERATURES:
            parameters["BETA"] = beta_d / DIMENSION
            for bank, seed in enumerate(bank_seeds):
                # Reinstantiating this generator pairs the bank and queries across beta.
                rng = np.random.default_rng(seed)
                for row in replicate(size, RHO, bank, rng):
                    rows.append({"beta_d": beta_d, **row})
            print(f"N={size}, beta*d={beta_d}: {BANKS} independent banks complete", flush=True)
    write_csv(args.output / "campaign_j_robustness_seed_counts.csv", rows)
    write_csv(args.output / "campaign_j_robustness_summary.csv", summarize(rows, np))


if __name__ == "__main__":
    main()

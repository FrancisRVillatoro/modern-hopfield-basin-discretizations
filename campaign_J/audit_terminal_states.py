"""Replay Campaign J and audit its terminal-state classification."""

import argparse
import json
from pathlib import Path

import campaign_j as campaign
import numpy as np
from scipy.special import softmax


def terminal_metrics(states, patterns):
    weights = softmax(campaign.BETA * (states @ patterns.T), axis=1)
    mean = weights @ patterns
    centered = patterns[np.newaxis, :, :] - mean[:, np.newaxis, :]
    covariances = np.einsum(
        "bni,bnj,bn->bij", centered, centered, weights, optimize=True,
    )
    hessians = np.eye(campaign.DIMENSION) - campaign.BETA * covariances
    eigenvalues = np.linalg.eigvalsh(hessians)[:, 0]
    residuals = np.linalg.norm(states - mean, axis=1)
    return eigenvalues, residuals


def replicate(size, rho, bank, rng):
    patterns, initial, targets = campaign.memory_and_cues(size, rho, rng)
    one_step = initial + campaign.retrieval_rhs(initial, patterns)
    attention, attention_ok = campaign.iterate(initial, patterns, theta=1.0)
    overrelaxed, overrelaxed_ok = campaign.iterate(initial, patterns, theta=1.9)
    early = campaign.integrate(initial, patterns, 40.0, 5e-9, 5e-11)
    flow = campaign.integrate(initial, patterns, 80.0, 5e-11, 5e-13)
    resolved_at_80 = campaign.residual(flow, patterns) <= campaign.RESIDUAL_LIMIT
    early_mismatch = (
        (campaign.residual(early, patterns) <= campaign.RESIDUAL_LIMIT)
        & resolved_at_80
        & ~campaign.close_endpoints(early, flow)
    )
    if np.any(~resolved_at_80):
        flow[~resolved_at_80] = campaign.integrate(
            flow[~resolved_at_80], patterns, 240.0, 5e-11, 5e-13,
        )
    flow_ok = (campaign.residual(flow, patterns) <= campaign.RESIDUAL_LIMIT) & ~early_mismatch

    endpoints = {"attention": attention, "overrelaxed": overrelaxed, "flow": flow}
    resolved = {"attention": attention_ok, "overrelaxed": overrelaxed_ok, "flow": flow_ok}
    metrics = {name: terminal_metrics(state, patterns) for name, state in endpoints.items()}
    distances = {
        name: np.linalg.norm(endpoints[name] - flow, axis=1)
        for name in ("attention", "overrelaxed")
    }
    first_correct = campaign.decode(one_step, patterns) == targets
    flow_correct = flow_ok & (campaign.decode(flow, patterns) == targets)

    for index, sigma in enumerate(campaign.NOISE_LEVELS):
        selection = slice(campaign.CUES * index, campaign.CUES * (index + 1))
        conditions = {
            "patterns": size, "rho": rho, "bank": bank, "sigma": sigma, "cues": campaign.CUES,
        }
        for name, (eigenvalues, residuals) in metrics.items():
            accepted = resolved[name][selection]
            minima = eigenvalues[selection][accepted]
            conditions[f"{name}_unresolved"] = int(accepted.size - np.count_nonzero(accepted))
            conditions[f"{name}_nonpositive_hessians"] = int(np.count_nonzero(minima <= 0))
            conditions[f"{name}_min_eigenvalue"] = float(minima.min()) if minima.size else None
            values = residuals[selection]
            conditions[f"{name}_max_residual"] = float(values.max())
        for name, distance in distances.items():
            values = distance[selection][(resolved[name] & flow_ok)[selection]]
            same = values[values <= campaign.ENDPOINT_LIMIT]
            distinct = values[values > campaign.ENDPOINT_LIMIT]
            conditions[f"{name}_same_max_distance"] = float(same.max()) if same.size else None
            conditions[f"{name}_distinct_min_distance"] = float(distinct.min()) if distinct.size else None
        first = first_correct[selection]
        final = flow_correct[selection]
        conditions["one_step_correct"] = int(np.count_nonzero(first))
        conditions["flow_correct"] = int(np.count_nonzero(final))
        conditions["one_correct_flow_incorrect"] = int(np.count_nonzero(first & ~final))
        conditions["one_incorrect_flow_correct"] = int(np.count_nonzero(~first & final))
        yield conditions


def summary(rows):
    result = {"queries": sum(row["cues"] for row in rows), "methods": {}}
    for name in ("attention", "overrelaxed", "flow"):
        result["methods"][name] = {
            "unresolved": sum(row[f"{name}_unresolved"] for row in rows),
            "nonpositive_hessians": sum(row[f"{name}_nonpositive_hessians"] for row in rows),
            "min_eigenvalue": min(
                row[f"{name}_min_eigenvalue"] for row in rows if row[f"{name}_min_eigenvalue"] is not None
            ),
            "max_residual": max(row[f"{name}_max_residual"] for row in rows),
        }
    for name in ("attention", "overrelaxed"):
        result["methods"][name].update({
            "same_max_distance": max(
                row[f"{name}_same_max_distance"] for row in rows if row[f"{name}_same_max_distance"] is not None
            ),
            "distinct_min_distance": min(
                row[f"{name}_distinct_min_distance"] for row in rows if row[f"{name}_distinct_min_distance"] is not None
            ),
        })

    target = [row for row in rows if (row["patterns"], row["rho"], row["sigma"]) == (64, 0.6, 0.5)]
    counts = np.array([row["one_step_correct"] - row["flow_correct"] for row in target])
    totals = np.array([row["cues"] for row in target])
    indices = np.random.default_rng(campaign.BASE_SEED + 1).integers(
        len(target), size=(campaign.BOOTSTRAP_SAMPLES, len(target)),
    )
    bootstrap = counts[indices].sum(axis=1) / totals[indices].sum(axis=1)
    result["paired_example"] = {
        "patterns": 64, "rho": 0.6, "sigma": 0.5, "queries": int(totals.sum()),
        "one_correct_flow_incorrect": sum(row["one_correct_flow_incorrect"] for row in target),
        "one_incorrect_flow_correct": sum(row["one_incorrect_flow_correct"] for row in target),
        "recall_difference": float(counts.sum() / totals.sum()),
        "bootstrap_percentile_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=Path(__file__).parent, type=Path)
    output = parser.parse_args().output
    output.mkdir(exist_ok=True, parents=True)
    seeds = iter(np.random.SeedSequence(campaign.BASE_SEED).spawn(
        len(campaign.LOADS) * len(campaign.CORRELATIONS) * campaign.REPLICATES,
    ))
    rows = []
    for size in campaign.LOADS:
        for rho in campaign.CORRELATIONS:
            for bank in range(campaign.REPLICATES):
                rows.extend(replicate(size, rho, bank, np.random.default_rng(next(seeds))))
            print(f"Audited {campaign.REPLICATES} banks at N={size}, rho={rho}", flush=True)
    campaign.write_csv(output / "terminal_audit_banks.csv", rows)
    report = json.dumps(summary(rows), indent=2) + "\n"
    (output / "terminal_audit_summary.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled query corruption for tied modern Hopfield retrieval.

Usage: python campaign_j.py [output-directory]
The CSV files contain the replicate counts and seed-bootstrap summaries.
"""
import argparse
import csv
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax

DIMENSION = 32
LOADS = (16, 64)
CORRELATIONS = (0.0, 0.6)
NOISE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
BETA = 18.0 / DIMENSION
REPLICATES = 16
CUES = 128
BASE_SEED = 20261122
BOOTSTRAP_SAMPLES = 10_000
RESIDUAL_LIMIT = 1e-9 * np.sqrt(DIMENSION)
ENDPOINT_LIMIT = 1e-6 * np.sqrt(DIMENSION)
MAP_STEP_LIMIT = 2000

METHODS = (
    ("raw_cue", "Raw cue", "#6c7075", "x"),
    ("one_step", "One update", "#16161d", "s"),
    ("iterated_attention", r"Iterated attention", "#087f8c", "o"),
    ("flow", "Continuous flow", "#d5532c", "^"),
    ("overrelaxed", r"Iterated $\theta=1.9$", "#7451a6", "d"),
)
AGREEMENTS = ("attention", "overrelaxed")


def retrieval_rhs(states, patterns):
    return softmax(BETA * (states @ patterns.T), axis=-1) @ patterns - states


def residual(states, patterns):
    return np.linalg.norm(retrieval_rhs(states, patterns), axis=1)


def decode(states, patterns):
    # All patterns have the same Euclidean norm.
    return np.argmax(states @ patterns.T, axis=1)


def close_endpoints(left, right):
    return np.linalg.norm(left - right, axis=1) <= ENDPOINT_LIMIT


def integrate(initial, patterns, time_limit, rtol, atol):
    def rhs(_, flat_states):
        states = flat_states.reshape(initial.shape)
        return retrieval_rhs(states, patterns).ravel()

    solution = solve_ivp(
        rhs, (0.0, time_limit), initial.ravel(), method="DOP853", t_eval=[time_limit],
        rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(initial.shape)


def iterate(initial, patterns, theta):
    states = initial.copy()
    active = np.arange(len(initial))
    for _ in range(MAP_STEP_LIMIT):
        vector_field = retrieval_rhs(states[active], patterns)
        pending = np.linalg.norm(vector_field, axis=1) > RESIDUAL_LIMIT / 10
        active = active[pending]
        if len(active) == 0:
            break
        states[active] += theta * vector_field[pending]
    return states, residual(states, patterns) <= RESIDUAL_LIMIT


def memory_and_cues(size, rho, rng):
    shared = rng.normal(size=DIMENSION)
    independent = rng.normal(size=(size, DIMENSION))
    patterns = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * independent
    patterns *= np.sqrt(DIMENSION) / np.linalg.norm(patterns, axis=1, keepdims=True)
    targets = np.repeat(np.arange(size), CUES // size)
    noise = rng.normal(size=(CUES, DIMENSION))
    initial = np.concatenate([patterns[targets] + sigma * noise for sigma in NOISE_LEVELS])
    return patterns, initial, np.tile(targets, len(NOISE_LEVELS))


def campaign_replicate(size, rho, replicate, rng):
    patterns, initial, targets = memory_and_cues(size, rho, rng)
    one_step = initial + retrieval_rhs(initial, patterns)
    attention, attention_ok = iterate(initial, patterns, theta=1.0)
    overrelaxed, overrelaxed_ok = iterate(initial, patterns, theta=1.9)
    early = integrate(initial, patterns, 40.0, 5e-9, 5e-11)
    flow = integrate(initial, patterns, 80.0, 5e-11, 5e-13)

    early_ok = residual(early, patterns) <= RESIDUAL_LIMIT
    resolved_T80 = residual(flow, patterns) <= RESIDUAL_LIMIT
    resolved_both = early_ok & resolved_T80
    early_disagreement = resolved_both & ~close_endpoints(early, flow)
    early_label_change = decode(early, patterns) != decode(flow, patterns)
    slow = ~resolved_T80
    if np.any(slow):
        flow[slow] = integrate(flow[slow], patterns, 240.0, 5e-11, 5e-13)
    flow_ok = (residual(flow, patterns) <= RESIDUAL_LIMIT) & ~early_disagreement
    recovery = {
        "raw_cue": decode(initial, patterns) == targets,
        "one_step": decode(one_step, patterns) == targets,
        "iterated_attention": attention_ok & (decode(attention, patterns) == targets),
        "flow": flow_ok & (decode(flow, patterns) == targets),
        "overrelaxed": overrelaxed_ok & (decode(overrelaxed, patterns) == targets),
    }
    resolved = {
        "attention": attention_ok & flow_ok,
        "overrelaxed": overrelaxed_ok & flow_ok,
    }
    disagreement = {
        "attention": resolved["attention"] & ~close_endpoints(attention, flow),
        "overrelaxed": resolved["overrelaxed"] & ~close_endpoints(overrelaxed, flow),
    }
    for index, sigma in enumerate(NOISE_LEVELS):
        indices = slice(index * CUES, (index + 1) * CUES)
        yield {
            "dimension": DIMENSION, "patterns": size, "load": size / DIMENSION, "rho": rho,
            "beta": BETA, "sigma": sigma, "seed_root": BASE_SEED, "replicate": replicate,
            "cues": CUES, "flow_unresolved_T40": CUES - np.count_nonzero(early_ok[indices]),
            "flow_unresolved_T80": CUES - np.count_nonzero(resolved_T80[indices]),
            "flow_unresolved_T320": CUES - np.count_nonzero(flow_ok[indices]),
            "flow_audit_endpoint_changes": np.count_nonzero(early_disagreement[indices]),
            "flow_audit_label_changes": np.count_nonzero(early_label_change[indices]),
            "iterated_attention_unresolved": CUES - np.count_nonzero(attention_ok[indices]),
            "overrelaxed_unresolved": CUES - np.count_nonzero(overrelaxed_ok[indices]),
            **{f"recall_{name}": np.count_nonzero(hits[indices]) for name, hits in recovery.items()},
            **{f"resolved_{name}_flow": np.count_nonzero(hits[indices]) for name, hits in resolved.items()},
            **{f"disagree_{name}_flow": np.count_nonzero(hits[indices]) for name, hits in disagreement.items()},
        }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(seed_rows):
    samples = np.random.default_rng(BASE_SEED + 1).integers(
        REPLICATES, size=(BOOTSTRAP_SAMPLES, REPLICATES)
    )
    summary = []
    for n in LOADS:
        for rho in CORRELATIONS:
            for sigma in NOISE_LEVELS:
                selected = [row for row in seed_rows if (row["patterns"], row["rho"], row["sigma"]) == (n, rho, sigma)]
                record = {"patterns": n, "load": n / DIMENSION, "rho": rho, "sigma": sigma}
                statistics = [(f"recall_{name}", "cues") for name, *_ in METHODS]
                statistics += [(f"disagree_{name}_flow", f"resolved_{name}_flow") for name in AGREEMENTS]
                for numerator, denominator in statistics:
                    counts = np.array([row[numerator] for row in selected])
                    totals = np.array([row[denominator] for row in selected])
                    draws = counts[samples].sum(axis=1) / totals[samples].sum(axis=1)
                    low, high = np.quantile(draws, [0.025, 0.975])
                    record[numerator] = counts.sum() / totals.sum()
                    record[f"{numerator}_lo"] = low
                    record[f"{numerator}_hi"] = high
                summary.append(record)
    return summary


def draw(output, summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.0), sharex=True, sharey=True)
    for row_index, n in enumerate(LOADS):
        for column_index, rho in enumerate(CORRELATIONS):
            ax = axes[row_index, column_index]
            subset = [row for row in summary if (row["patterns"], row["rho"]) == (n, rho)]
            for name, label, color, marker in METHODS:
                field = f"recall_{name}"
                values = [row[field] for row in subset]
                low = [row[field + "_lo"] for row in subset]
                high = [row[field + "_hi"] for row in subset]
                ax.fill_between(NOISE_LEVELS, low, high, color=color, alpha=0.105, linewidth=0)
                linestyle = {"raw_cue": ":", "one_step": "--"}.get(name, "-")
                ax.plot(NOISE_LEVELS, values, label=label, color=color, marker=marker, markersize=3.2,
                        linewidth=1.2, linestyle=linestyle)
            ax.set_title(rf"$N/d={n / DIMENSION:g},\ \rho={rho:g}$")
            ax.set_xlim(-0.03, 2.04)
            ax.set_ylim(0.0, 1.045)
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5,
               frameon=False, handlelength=2.0, columnspacing=1.5)
    fig.supxlabel(r"Gaussian corruption amplitude $\sigma$", y=0.022)
    fig.supylabel("Decoded recall accuracy", x=0.015)
    fig.tight_layout(rect=(0.035, 0.05, 1, 0.91))
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, default=Path(__file__).resolve().parent, nargs="?")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    seeds = iter(np.random.SeedSequence(BASE_SEED).spawn(len(LOADS) * len(CORRELATIONS) * REPLICATES))
    for n in LOADS:
        for rho in CORRELATIONS:
            for replicate in range(REPLICATES):
                rng = np.random.default_rng(next(seeds))
                seed_rows.extend(campaign_replicate(n, rho, replicate, rng))
            print(f"N={n}, rho={rho}: completed {REPLICATES} independent memories", flush=True)
    write_csv(args.output / "campaign_j_seed_counts.csv", seed_rows)
    summary = summarize(seed_rows)
    write_csv(args.output / "campaign_j_summary.csv", summary)
    draw(args.output / "fig_recall_corruption.pdf", summary)


if __name__ == "__main__":
    main()

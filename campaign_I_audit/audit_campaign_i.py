"""Replay the archived large-step discrete-gradient run and log solver residuals."""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import root


SEED = 20260809
DIMENSION = 64
PATTERN_COUNT = 4
BETA = 0.15
STEP_SIZES = (0.5, 1.0, 2.0, 4.0)
RUNS = 10
STEPS = 200
RESIDUAL_DIAGNOSTIC = 1e-10


def energy(x, patterns):
    logits = BETA * (patterns.T @ x)
    maximum = logits.max(axis=0)
    return (
        0.5 * (x * x).sum(axis=0)
        - (maximum + np.log(np.exp(logits - maximum).sum(axis=0))) / BETA
    )


def gradient(x, patterns):
    logits = BETA * (patterns.T @ x)
    weights = np.exp(logits - logits.max(axis=0))
    return x - patterns @ (weights / weights.sum(axis=0))


def discrete_gradient(x, y, energy_x, patterns):
    midpoint_gradient = gradient(0.5 * (x + y), patterns)
    difference = y - x
    norm_squared = difference @ difference
    if norm_squared < 1e-30:
        return midpoint_gradient
    correction = (energy(y, patterns) - energy_x - midpoint_gradient @ difference) / norm_squared
    return midpoint_gradient + correction * difference


def solve_step(x, step_size, patterns):
    energy_x = energy(x, patterns)
    calls = 0

    def residual(y):
        nonlocal calls
        calls += 1
        return y - x + step_size * discrete_gradient(x, y, energy_x, patterns)

    initial = x - step_size * gradient(x, patterns)
    result = root(residual, initial, method="hybr", tol=1e-12)
    return result, calls


def replay():
    rng = np.random.default_rng(SEED)
    q, _ = np.linalg.qr(rng.standard_normal((DIMENSION, PATTERN_COUNT)))
    patterns = np.sqrt(DIMENSION) * q
    pattern_norm = np.linalg.norm(patterns, axis=0).max()
    rows = []
    for step_size in STEP_SIZES:
        for replicate in range(RUNS):
            state = rng.standard_normal(DIMENSION)
            state *= 1.5 * pattern_norm / np.linalg.norm(state)
            for step in range(STEPS):
                result, calls = solve_step(state, step_size, patterns)
                next_state = result.x
                displacement = next_state - state
                energy_increment = energy(next_state, patterns) - energy(state, patterns)
                defect = energy_increment + displacement @ displacement / step_size
                rows.append({
                    "step_size": step_size,
                    "replicate": replicate,
                    "step": step,
                    "solver_success": bool(result.success),
                    "solver_status": int(result.status),
                    "residual_l2": float(np.linalg.norm(result.fun)),
                    "energy_increment": float(energy_increment),
                    "energy_identity_defect": float(defect),
                    "absolute_defect_minus_residual_projection": float(
                        abs(defect - displacement @ result.fun / step_size)
                    ),
                    "archive_evaluation_units": 1.5 * calls,
                })
                state = next_state
    return rows


def summarize(rows):
    report = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "random_seed": SEED,
        "residual_diagnostic": RESIDUAL_DIAGNOSTIC,
        "by_step_size": {},
    }
    for step_size in STEP_SIZES:
        selected = [row for row in rows if row["step_size"] == step_size]
        failures = [row for row in selected if not row["solver_success"]]
        report["by_step_size"][str(step_size)] = {
            "solver_calls": len(selected),
            "solver_non_successes": len(failures),
            "non_successes_above_residual_diagnostic": sum(
                row["residual_l2"] > RESIDUAL_DIAGNOSTIC for row in failures
            ),
            "maximum_residual_l2": max(row["residual_l2"] for row in selected),
            "maximum_energy_increment": max(row["energy_increment"] for row in selected),
            "maximum_absolute_energy_identity_defect": max(
                abs(row["energy_identity_defect"]) for row in selected
            ),
            "maximum_absolute_defect_minus_residual_projection": max(
                row["absolute_defect_minus_residual_projection"] for row in selected
            ),
            "mean_archive_evaluation_units": float(np.mean([
                row["archive_evaluation_units"] for row in selected
            ])),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for the JSON and CSV results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = replay()
    summary = summarize(rows)
    with (args.output / "campaign_i_solver_trace.csv").open("w", newline="") as stream:
        columns = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "campaign_i_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

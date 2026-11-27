"""Reproduce and extend the endpoint-label audit of Hopfield Campaign E.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_planar_horizons.py
Labels come from numerical integration and iteration, without error enclosure.
The energy gap of each observed disagreement is enclosed by interval arithmetic.
"""
from functools import cache
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from certify_planar_critical_points import energy as interval_energy, iv


PATTERNS = np.array([[2.0, -1.0, -0.5], [0.0, 1.5, -1.8]])
BETA = 1.5
GRID_SIZE = 301
# These upper bounds are from certify_planar_critical_points.py.
ESCAPE_UPPER_BY_PATTERN = ("-1.145436380054", "-0.869210114866", "-1.145436380054")


def attention(points):
    logits = BETA * (PATTERNS.T @ points)
    weights = np.exp(logits - logits.max(axis=0))
    weights /= weights.sum(axis=0)
    return PATTERNS @ weights


def field(_time, state):
    points = state.reshape(2, -1)
    return (attention(points) - points).ravel()


def integrate(points, horizon, tolerance):
    result = solve_ivp(
        field, (0.0, horizon), points.ravel(), method="RK45",
        rtol=tolerance, atol=tolerance, t_eval=[horizon],
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1].reshape(2, -1), result.nfev


def classify(points, minima):
    distances = np.linalg.norm(points.T[:, None, :] - minima[None, :, :], axis=2)
    return distances.argmin(axis=1), distances.min(axis=1)


def boundary_nodes(labels):
    image = labels.reshape(GRID_SIZE, GRID_SIZE)
    boundary = np.zeros_like(image, dtype=bool)
    for rows, columns in ((slice(1, None), slice(None)), (slice(None), slice(1, None))):
        if rows.start == 1:
            previous = (slice(None, -1), columns)
        else:
            previous = (rows, slice(None, -1))
        following = (rows, columns)
        change = image[following] != image[previous]
        boundary[following] |= change
        boundary[previous] |= change
    return np.flatnonzero(boundary)


def compare(points, selected, minima, reference, horizon, tolerance):
    endpoints, calls = integrate(points[:, selected], horizon, tolerance)
    labels, distances = classify(endpoints, minima)
    return {
        "nodes": int(selected.size),
        "horizon": horizon,
        "rtol_atol": tolerance,
        "label_changes": int(np.count_nonzero(labels != reference[selected])),
        "max_distance_to_assigned_minimum": float(distances.max()),
        "gradient_evaluations": calls,
    }


def compare_discrete(points, minima, reference):
    """Compare the reported numerical labels with the enclosed escape energies."""
    upper_bounds = tuple(map(iv.mpf, ESCAPE_UPPER_BY_PATTERN))
    coordinates = [iv.mpf(-3) + iv.mpf(i) / 50 for i in range(GRID_SIZE)]

    @cache
    def gap_lower_bound(i):
        row, column = divmod(i, GRID_SIZE)
        point_energy = interval_energy((coordinates[column], coordinates[row]))
        return (point_energy - upper_bounds[reference[i]]).a

    records = []
    for theta, iterations in ((0.5, 1500), (1.0, 800), (1.5, 800), (1.9, 4000)):
        endpoint = points.copy()
        for _ in range(iterations):
            endpoint = (1 - theta) * endpoint + theta * attention(endpoint)
        labels, distances = classify(endpoint, minima)
        disagreement = labels != reference
        gap = min(gap_lower_bound(int(i)) for i in np.flatnonzero(disagreement))
        if gap <= iv.mpf("0.0265"):
            raise RuntimeError("A disagreement falls below the expected energy margin")
        records.append({
            "theta": theta,
            "iterations": iterations,
            "unresolved_nodes_at_1e_4": int(np.count_nonzero(distances >= 1e-4)),
            "label_changes_from_reference": int(np.count_nonzero(disagreement)),
            "lower_bound_on_minimum_energy_minus_escape_upper": str(gap),
        })
    return records


def main():
    minima = []
    for pattern in PATTERNS.T:
        result = root(lambda state: field(0.0, state), pattern, tol=1e-12)
        if np.linalg.norm(result.fun) > 1e-12:
            raise RuntimeError("The minimum solver returned a large residual")
        minima.append(result.x)
    minima = np.array(minima)
    np.testing.assert_array_less(np.linalg.norm(minima - PATTERNS.T, axis=1), 0.005)

    grid = np.linspace(-3.0, 3.0, GRID_SIZE)
    xx, yy = np.meshgrid(grid, grid)
    points = np.vstack((xx.ravel(), yy.ravel()))
    ref_points, calls = integrate(points, 60.0, 1e-11)
    reference, distances = classify(ref_points, minima)
    batch = np.arange(reference.size)
    boundary = boundary_nodes(reference)
    random_sample = np.random.default_rng(1).choice(reference.size, 400, replace=False)

    output = {
        "method": "RK45 with a joint adaptive controller within each batch",
        "patterns": PATTERNS.T.tolist(),
        "beta": BETA,
        "minima_in_pattern_order": minima.tolist(),
        "grid": {"size_per_axis": GRID_SIZE, "interval": [-3.0, 3.0]},
        "reference": {
            "nodes": int(reference.size),
            "horizon": 60.0,
            "rtol_atol": 1e-11,
            "label_counts": np.bincount(reference).tolist(),
            "max_distance_to_assigned_minimum": float(distances.max()),
            "gradient_evaluations": calls,
        },
        "whole_grid_double_horizon": compare(points, batch, minima, reference, 120.0, 1e-11),
        "four_neighbour_boundary_double_horizon": compare(
            points, boundary, minima, reference, 120.0, 1e-9,
        ),
        "random_sample_same_horizon": compare(
            points, random_sample, minima, reference, 60.0, 1e-9,
        ),
        "discrete_comparison": compare_discrete(points, minima, reference),
    }
    path = Path(__file__).with_name("planar_horizon_audit.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()

"""Check every sampled boundary node with a scalar DOP853 integration."""

import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

import audit_planar_horizons as planar


def main():
    solves = [root(lambda x: planar.field(0.0, x), p, tol=1e-12) for p in planar.PATTERNS.T]
    minima = np.array([s.x for s in solves])
    if np.max(np.linalg.norm([planar.field(0.0, x) for x in minima], axis=1)) > 1e-12:
        raise RuntimeError("Could not numerically refine the three reference minima.")

    grid = np.linspace(-3.0, 3.0, planar.GRID_SIZE)
    xx, yy = np.meshgrid(grid, grid)
    points = np.vstack((xx.ravel(), yy.ravel()))
    reference, _ = planar.integrate(points, 60.0, 1e-11)
    reference_labels, _ = planar.classify(reference, minima)
    boundary = planar.boundary_nodes(reference_labels)

    labels = []
    minimum_distances = []
    for index in boundary:
        solution = solve_ivp(
            planar.field, (0.0, 120.0), points[:, index], t_eval=[120.0],
            method="DOP853", rtol=1e-12, atol=1e-13,
        )
        if not solution.success:
            raise RuntimeError(f"Integration failed at grid node {index}: {solution.message}")
        distances = np.linalg.norm(solution.y[:, -1] - minima, axis=1)
        labels.append(int(distances.argmin()))
        minimum_distances.append(float(distances.min()))

    result = {
        "boundary_nodes": int(len(boundary)),
        "changed_labels": int(np.count_nonzero(np.array(labels) != reference_labels[boundary])),
        "largest_distance_to_assigned_minimum": max(minimum_distances),
        "method": "DOP853",
        "horizon": 120.0,
        "rtol": 1e-12,
        "atol": 1e-13,
    }
    text = json.dumps(result, indent=2) + "\n"
    Path(__file__).with_name("independent_boundary_results.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

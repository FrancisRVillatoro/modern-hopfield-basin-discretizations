import argparse
import json
from pathlib import Path

import numpy as np


def model(patterns, beta):
    def probability(states):
        logits = beta * (patterns.T @ states)
        weights = np.exp(logits - np.max(logits, axis=0))
        return weights / np.sum(weights, axis=0)

    def spectrum(state):
        p = probability(state)
        covariance = np.diag(p) - np.outer(p, p)
        return np.linalg.eigvalsh(beta * (patterns @ covariance @ patterns.T))

    def energy(states):
        logits = beta * (patterns.T @ states)
        return 0.5 * np.sum(states**2, axis=0) - np.logaddexp.reduce(logits, axis=0) / beta

    return probability, spectrum, energy


def directions(rng, dimension, count):
    samples = rng.standard_normal((dimension, count))
    return samples / np.linalg.norm(samples, axis=0)


def overrelaxation():
    rng = np.random.default_rng(20260809)
    d, n = 64, 4
    q, _ = np.linalg.qr(rng.standard_normal((d, n)))
    patterns = np.sqrt(d) * q
    rows = []
    tol = 1e-10
    for beta in [0.09, 0.10, 0.12, 0.15, 0.20, 0.25]:
        p, spectrum, _ = model(patterns, beta)
        eps = (n - 1) * np.exp(-beta * (d - 2 * np.sqrt(d)))
        ell = 2 * beta * d * eps
        radius = 1 - 2 * np.sqrt(d) * eps
        equilibrium = patterns[:, 0].copy()
        for _ in range(600):
            equilibrium = patterns @ p(equilibrium)
        lam = float(spectrum(equilibrium)[-1])
        sphere = equilibrium[:, None] + radius * directions(rng, d, 200)
        sampled = max(lam, *(float(spectrum(x)[-1]) for x in sphere.T))
        initial = equilibrium[:, None] + 0.8 * radius * directions(rng, d, 30)
        metrics = []
        for theta in [1, 2 / (2 - ell), 2 / (2 - lam)]:
            state = initial.copy()
            counts = np.zeros(initial.shape[1], dtype=int)
            observed = 0.0
            for _ in range(100):
                errors = np.linalg.norm(state - equilibrium[:, None], axis=0)
                if np.all(errors <= tol):
                    break
                updated = (1 - theta) * state + theta * (patterns @ p(state))
                updated_errors = np.linalg.norm(updated - equilibrium[:, None], axis=0)
                valid_ratio = errors > 10 * tol
                if np.any(valid_ratio):
                    observed = max(observed, float(np.max(updated_errors[valid_ratio] / errors[valid_ratio])))
                counts += errors > tol
                state = updated
            else:
                raise RuntimeError(f"Iteration cap reached at beta={beta}, theta={theta}")
            metrics.append(dict(theta=theta, maximum_observed_factor=observed,
                                mean_iterations=float(np.mean(counts)),
                                sample_sd=float(np.std(counts, ddof=1)), iterations=counts.tolist()))
        rows.append(dict(beta=beta, ell=ell, lambda_max=lam, sampled_spectral_maximum=sampled,
                         q_certified=ell / (2 - ell), metrics=metrics))
    return rows


def correlation_and_load():
    rows = []
    for d, n in [(64, 4), (64, 32), (16, 32)]:
        for rho in [0, 0.3, 0.6]:
            for beta in ([0.15, 0.3, 0.6] if n <= d else [0.5, 1, 2]):
                seeds = []
                for seed in range(1, 11):
                    rng = np.random.default_rng(seed)
                    common = rng.standard_normal((d, 1))
                    independent = rng.standard_normal((d, n))
                    samples = np.sqrt(1 - rho) * independent + np.sqrt(rho) * common
                    patterns = np.sqrt(d) * samples / np.linalg.norm(samples, axis=0)
                    p, spectrum, energy = model(patterns, beta)
                    sig2 = np.linalg.svd(patterns, compute_uv=False)[0]**2
                    gram = patterns.T @ patterns
                    margins = gram.diagonal() - gram
                    np.fill_diagonal(margins, np.inf)
                    separation = np.min(margins, axis=0)
                    radius = 0.05 * np.sqrt(d)
                    eps = (n - 1) * np.exp(-beta * (separation - 2 * np.sqrt(d) * radius))
                    ell = 2 * beta * sig2 * eps
                    certified = (2 * np.sqrt(d) * eps < radius) & (ell < 1)
                    state = patterns.copy()
                    for _ in range(500):
                        state = patterns @ p(state)
                    residual = np.linalg.norm(state - patterns @ p(state), axis=0)
                    lam = np.array([spectrum(x)[-1] for x in state.T])
                    empirical = ((np.linalg.norm(state - patterns, axis=0) < radius)
                                 & (lam < 0.5) & (residual < 1e-9))

                    queries = (2 * np.sqrt(d) * directions(rng, d, 500)
                               * rng.random(500)**(1 / d))
                    gradient = queries - patterns @ p(queries)
                    ratios = []
                    for theta in [0.5, 1, 1.5, 1.9]:
                        decrease = energy(queries) - energy(queries - theta * gradient)
                        bound = 0.5 * theta * (2 - theta) * np.sum(gradient**2, axis=0)
                        ratios.append(float(np.min(decrease / bound)))
                    seeds.append(dict(seed=seed, certified_count=int(np.sum(certified)),
                                      empirical_count=int(np.sum(empirical)),
                                      certified_without_proxy=int(np.sum(certified & ~empirical)),
                                      minimum_normalized_separation=float(np.min(separation) / d),
                                      normalized_sigma_squared=float(sig2 / d),
                                      minimum_slack=float(np.min(ell[certified] / lam[certified]))
                                      if np.any(certified) else None,
                                      maximum_fixed_point_residual=float(np.max(residual)),
                                      minimum_dissipation_ratio=min(ratios)))
                rows.append(dict(d=d, n=n, rho=rho, beta=beta, seeds=seeds))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {"G": overrelaxation(), "H": correlation_and_load()}
    args.output.write_text(json.dumps(results, indent=2) + "\n")

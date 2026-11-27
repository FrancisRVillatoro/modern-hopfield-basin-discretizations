#!/usr/bin/env python3
"""Bounds on energy cells quoted in Remark 4.13 and in Campaign E.

Run: python low_temperature_bounds.py
The script writes no files and runs in a few seconds. It evaluates, for the
orthogonal instance of Campaigns A-C and G and for the planar landscape of
Campaign E, the escape-gap bounds of Proposition 4.11 and Corollary 4.6 and
the first-order expression of Remark 4.13.
"""
import numpy as np

GRID = 400_001        # radii sampled when maximizing a certified gap over r
SEGMENT = 200_001     # points sampled on a segment between two patterns


def lse(z, beta):
    m = z.max(axis=0)
    return m + np.log(np.exp(beta * (z - m)).sum(axis=0)) / beta


def softmax(z, beta):
    w = np.exp(beta * (z - z.max(axis=0)))
    return w / w.sum(axis=0)


def energy(x, patterns, beta):
    """Energy at one point of shape (d,) or at the columns of an array (d, n)."""
    return 0.5 * (x * x).sum(axis=0) - lse(patterns.T @ x, beta)


def attractor(mu, patterns, beta):
    """Equilibrium near pattern mu: attention iteration followed by Newton steps."""
    x = patterns[:, mu].copy()
    for _ in range(5000):
        x = patterns @ softmax(patterns.T @ x, beta)
    for _ in range(50):
        p = softmax(patterns.T @ x, beta)
        gradient = x - patterns @ p
        if np.linalg.norm(gradient) < 1e-15:
            break
        hessian = np.eye(x.size) - beta * patterns @ (np.diag(p) - np.outer(p, p)) @ patterns.T
        x = x - np.linalg.solve(hessian, gradient)
    return x


def margins(patterns, mu):
    """Delta_{mu j} and D_{mu j} for j != mu."""
    xi = patterns[:, mu]
    others = [j for j in range(patterns.shape[1]) if j != mu]
    delta = np.array([(xi - patterns[:, j]) @ xi for j in others])
    dist = np.array([np.linalg.norm(xi - patterns[:, j]) for j in others])
    return others, delta, dist


def inscribed_radius(patterns, mu):
    """rbar_mu and the label j attaining it."""
    others, delta, dist = margins(patterns, mu)
    k = int(np.argmin(delta / dist))
    return float(delta[k] / dist[k]), others[k], float(dist[k])


def eps_tilde(patterns, mu, r, beta):
    _, delta, dist = margins(patterns, mu)
    r = np.atleast_1d(r)
    return np.exp(-beta * (delta[:, None] - r[None, :] * dist[:, None])).sum(axis=0)


def proposition_bound(patterns, mu, beta):
    """Largest 1/2 r^2 - eps~/beta over radii satisfying the assumptions of Proposition 4.11."""
    rbar = inscribed_radius(patterns, mu)[0]
    r = np.linspace(rbar / GRID, rbar * (1 - 1 / GRID), GRID)
    eps = eps_tilde(patterns, mu, r, beta)
    ell = 2 * beta * np.linalg.norm(patterns, 2) ** 2 * eps
    admissible = (ell < 1) & (eps / beta < 0.5 * r ** 2)
    if not admissible.any():
        return None
    gap = np.where(admissible, 0.5 * r ** 2 - eps / beta, -np.inf)
    k = int(np.argmax(gap))
    return float(r[k]), float(gap[k])


def corollary_bound(patterns, mu, beta):
    """Largest certified gap 1/2 (1 - ell)(r - 2 M eps)^2 of Corollary 4.6 over admissible radii."""
    _, delta, _ = margins(patterns, mu)
    n = patterns.shape[1]
    big_m = np.linalg.norm(patterns, axis=0).max()
    sigma2 = np.linalg.norm(patterns, 2) ** 2
    separation = delta.min()
    r = np.linspace(0, separation / (2 * big_m), GRID)[1:-1]
    eps = (n - 1) * np.exp(-beta * (separation - 2 * big_m * r))
    ell = 2 * beta * sigma2 * eps
    admissible = (2 * big_m * eps < r) & (ell < 1)
    gap = np.where(admissible, 0.5 * (1 - ell) * (r - 2 * big_m * eps) ** 2, -np.inf)
    k = int(np.argmax(gap))
    return float(r[k]), float(gap[k])


def segment_maximum(patterns, beta, a, b):
    t = np.linspace(0.0, 1.0, SEGMENT)
    return float(energy(a[:, None] + t[None, :] * (b - a)[:, None], patterns, beta).max())


def point(v):
    return "(" + ", ".join(f"{c:g}" for c in v) + ")"


def binary_entropy(lam):
    return -lam * np.log(lam) - (1 - lam) * np.log(1 - lam)


def first_order_level(patterns, mu, beta):
    """-|xi_mu|^2/2 + rbar_mu^2/2 - h(lambda_mu)/beta of Remark 4.13."""
    rbar, _, dist = inscribed_radius(patterns, mu)
    xi = patterns[:, mu]
    return -0.5 * xi @ xi + 0.5 * rbar ** 2 - binary_entropy(rbar / dist) / beta


def orthogonal_instance():
    print("Orthogonal instance of Campaigns A-C and G: d = 64, N = 4, beta = 0.15")
    rng = np.random.default_rng(20260809)
    d, n, beta = 64, 4, 0.15
    q, _ = np.linalg.qr(rng.standard_normal((d, n)))
    patterns = np.sqrt(d) * q
    rbar, _, dist = inscribed_radius(patterns, 0)
    e_star = float(energy(attractor(0, patterns, beta), patterns, beta))
    r_prop, gap_prop = proposition_bound(patterns, 0, beta)
    r_cor, gap_cor = corollary_bound(patterns, 0, beta)
    print(f"  rbar = {rbar:.6f} (4 sqrt 2 = {4 * np.sqrt(2):.6f}); limit rbar^2/2 = {0.5 * rbar ** 2:.6f}")
    print(f"  E(x*) = {e_star:.10f}")
    print(f"  Proposition 4.11(c): escape gap >= {gap_prop:.6f} (best radius r = {r_prop:.4f})")
    print(f"  Corollary 4.6: largest certified escape gap = {gap_cor:.6f} (at r = {r_cor:.4f})")
    condition = beta * dist ** 2
    print(f"  equal norms: beta D^2 = {condition:.3f} > 8 log 2 = {8 * np.log(2):.3f}: {condition > 8 * np.log(2)}")
    upper = -0.5 * patterns[:, 0] @ patterns[:, 0] + 0.5 * rbar ** 2 - np.log(2) / beta - e_star
    print(f"  pair bound at the midpoint (Remark 4.13): escape gap <= {upper:.6f}")


def planar_instance():
    print("Planar landscape of Campaign E: beta = 1.5")
    patterns = np.array([[2.0, -1.0, -0.5], [0.0, 1.5, -1.8]])
    beta = 1.5
    # Certified escape levels from campaign_E_audit/certify_planar_critical_points.py.
    certified = {0: (-1.145436380056, -1.145436380054),
                 1: (-0.869210114869, -0.869210114866),
                 2: (-1.145436380056, -1.145436380054)}
    for mu in range(3):
        xi = patterns[:, mu]
        rbar, j, _ = inscribed_radius(patterns, mu)
        e_star = float(energy(attractor(mu, patterns, beta), patterns, beta))
        lower, upper = certified[mu]
        own = proposition_bound(patterns, mu, beta)
        other = proposition_bound(patterns, j, beta)
        segment = segment_maximum(patterns, beta, xi, patterns[:, j])
        print(f"  minimum near {point(xi)}: E(x*) = {e_star:.12f}; neighbor attaining rbar = {point(patterns[:, j])}")
        print(f"    escape gap from the certified level: [{lower - e_star:.6f}, {upper - e_star:.6f}]")
        print(f"    Proposition 4.11(c): escape gap >= {own[1]:.6f} (r = {own[0]:.4f}); "
              f"assumptions hold at the neighbor: {other is not None}")
        print(f"    Proposition 4.11(d): max of E on the segment = {segment:.9f}, "
              f"above the certified level by {segment - upper:.3e}")
        print(f"    Remark 4.13: first-order level = {first_order_level(patterns, mu, beta):.6f}")


if __name__ == "__main__":
    orthogonal_instance()
    planar_instance()

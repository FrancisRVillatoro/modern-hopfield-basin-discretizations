"""Campaign G: certified vs oracle overrelaxation for the relaxed attention family.
Same patterns as Campaigns A-C (seed 20260809), sweep in beta at fixed r=1.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json

rng = np.random.default_rng(20260809)
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N)))
X = np.sqrt(d) * Q
M = np.linalg.norm(X, axis=0).max()
sig2 = np.linalg.svd(X, compute_uv=False)[0] ** 2
Delta = min((X[:, 0] - X[:, nu]) @ X[:, 0] for nu in range(1, N))
r = 1.0
out = {}

def model(beta):
    def p(x):
        z = beta * (X.T @ x); z -= z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
    def A(x):
        pp = p(x); H = np.diag(pp) - np.outer(pp, pp)
        return beta * (X @ H @ X.T)
    return p, A

def run(beta):
    p, A = model(beta)
    eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r))
    ell = 2 * beta * sig2 * eps
    if 2 * M * eps > r or ell >= 1:
        return None
    rho = r - 2 * M * eps
    # attractor
    x = X[:, 0].copy()
    for _ in range(600):
        x = X @ p(x)
    xs = x
    Ax = A(xs)
    lam = np.linalg.eigvalsh(Ax)
    lmax, lmin = float(lam[-1]), float(lam[0])
    # sup of ||A|| over sampled sphere of radius rho around x*
    lbar = lmax
    for _ in range(200):
        u = rng.standard_normal(d); u *= rho / np.linalg.norm(u)
        lbar = max(lbar, float(np.linalg.norm(A(xs + u), 2)))
    th_c = 2 / (2 - ell)                    # certified minimax
    th_o = 2 / (2 - lmax - max(lmin, 0.0))  # oracle from attractor spectrum
    def obs(theta, tol=1e-10, iters_max=4000, nrep=10):
        fac, its = 0.0, []
        for _ in range(nrep):
            u = rng.standard_normal(d); u *= 0.8 * rho / np.linalg.norm(u)
            y = xs + u; k = 0
            while np.linalg.norm(y - xs) > tol and k < iters_max:
                yn = (1 - theta) * y + theta * (X @ p(y))
                e0, e1 = np.linalg.norm(y - xs), np.linalg.norm(yn - xs)
                if e0 > 10 * tol:
                    fac = max(fac, e1 / e0)
                y = yn; k += 1
            its.append(k)
        return fac, float(np.mean(its))
    q1o, it1 = obs(1.0)
    qco, itc = obs(th_c)
    qoo, ito = obs(th_o)
    return dict(beta=beta, eps=float(eps), ell=float(ell), rho=float(rho),
                lmax_star=lmax, lmin_star=lmin, lbar_ball=lbar,
                q1_cert=float(ell), q1_obs=q1o, it_theta1=it1,
                theta_cert=float(th_c), qstar_cert=float(ell / (2 - ell)),
                q_obs_at_theta_cert=qco, it_theta_cert=itc,
                theta_oracle=float(th_o), q_obs_at_oracle=qoo, it_oracle=ito)

for beta in [0.09, 0.10, 0.12, 0.15, 0.20, 0.25]:
    res = run(beta)
    out[f"beta={beta}"] = res

print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS_DIR / 'resultsG.json', 'w'), indent=1)

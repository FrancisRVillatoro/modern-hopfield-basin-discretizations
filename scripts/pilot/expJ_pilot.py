"""Pilot for Campaign J: real (non-synthetic) pattern matrices and masked retrieval.
Patterns are actual images from the bundled sklearn digits set (8x8, d=64), stored as
columns of X.  Queries are half-masked images.  Two questions:
  (Q1) how do the certified quantities of Section 3.3 behave on natural, correlated,
       possibly overcomplete pattern matrices?
  (Q2) does the energy-cell criterion PREDICT which queries retrieve their target?
Cost: seconds on one CPU core, no downloads, no training.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(20260809)
imgs = load_digits().images.reshape(-1, 64)
imgs = imgs / np.linalg.norm(imgs, axis=1, keepdims=True) * np.sqrt(64)  # |xi| = sqrt(d)

def make(N, beta, seed=0):
    r = np.random.default_rng(seed)
    idx = r.choice(len(imgs), N, replace=False)
    X = imgs[idx].T                                   # (64, N)
    def p(x):
        z = beta * (X.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
    def E(x):
        z = beta * (X.T @ x); m = z.max(axis=0)
        return 0.5 * (x * x).sum(axis=0) - (m + np.log(np.exp(z - m).sum(axis=0))) / beta
    def A(x):
        pp = p(x); H = np.diag(pp) - np.outer(pp, pp); return beta * (X @ H @ X.T)
    return X, p, E, A, idx

print(f"{'N':>4} {'d/N':>6} {'beta':>6} {'mu_coh':>8} {'Delta/M2':>9} "
      f"{'sig2/d':>7} {'cert%':>6} {'retr%':>6} {'cellOK':>7} {'falsePos':>9}")
for N in [16, 64, 128]:
    for beta in [0.5, 2.0, 8.0]:
        X, p, E, A, idx = make(N, beta, seed=N)
        d, M = 64, np.sqrt(64)
        sig2 = np.linalg.svd(X, compute_uv=False)[0] ** 2
        G = X.T @ X / 64.0
        coh = np.abs(G - np.diag(np.diag(G))).max()          # mutual coherence
        r_ball = 0.05 * M
        cert = 0; Deltas = []
        atts = np.zeros_like(X)
        for mu in range(N):
            Delta = min((X[:, mu] - X[:, n]) @ X[:, mu] for n in range(N) if n != mu)
            Deltas.append(Delta / 64.0)
            eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r_ball))
            ell = 2 * beta * sig2 * eps
            cert += int(2 * M * eps <= r_ball and ell < 1)
            x = X[:, mu].copy()
            for _ in range(400): x = X @ p(x)
            atts[:, mu] = x
        # masked-image retrieval: bottom half of the image zeroed
        mask = np.ones((8, 8)); mask[4:, :] = 0.0; mask = mask.ravel()
        succ = 0; cell_pred = 0; false_pos = 0
        # numerically inferred escape energy per attractor: lowest energy on the
        # boundary between its basin and another, estimated along attractor-to-attractor
        # segments (cheap 1-D scan) -- a conservative proxy \hat c*
        chat = np.full(N, np.inf)
        for mu in range(N):
            for nu in range(N):
                if nu == mu: continue
                ts = np.linspace(0, 1, 101)
                seg = atts[:, mu][:, None] * (1 - ts) + atts[:, nu][:, None] * ts
                chat[mu] = min(chat[mu], float(E(seg).max()))
        for mu in range(N):
            q = X[:, mu] * mask
            x = q.copy()
            for _ in range(200): x = X @ p(x)          # theta = 1 (attention)
            ok = np.linalg.norm(x - atts[:, mu]) < 0.05 * M
            succ += ok
            inside = E(q) < chat[mu]                    # energy-cell criterion
            cell_pred += inside
            if inside and not ok: false_pos += 1
        print(f"{N:>4} {64/N:>6.2f} {beta:>6.1f} {coh:>8.3f} {np.min(Deltas):>9.3f} "
              f"{sig2/64:>7.2f} {100*cert/N:>5.0f}% {100*succ/N:>5.0f}% "
              f"{100*cell_pred/N:>6.0f}% {false_pos:>9d}")

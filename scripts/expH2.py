from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json
def run(d, N, rho, beta, seed, rfrac):
    rng = np.random.default_rng(seed)
    g0 = rng.standard_normal(d)
    G = np.sqrt(1 - rho) * rng.standard_normal((d, N)) + np.sqrt(rho) * g0[:, None]
    X = np.sqrt(d) * G / np.linalg.norm(G, axis=0); M = np.sqrt(d)
    sig2 = np.linalg.svd(X, compute_uv=False)[0] ** 2
    def p(x):
        z = beta * (X.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
    def A(x):
        pp = p(x); H = np.diag(pp) - np.outer(pp, pp); return beta * (X @ H @ X.T)
    r = rfrac * M
    cert = 0; emp = 0; slacks = []; lams = []; errs = []
    Deltas = []
    for mu in range(N):
        Delta = min((X[:, mu] - X[:, n]) @ X[:, mu] for n in range(N) if n != mu); Deltas.append(Delta)
        eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r)); ell = 2 * beta * sig2 * eps
        c_ok = (2 * M * eps <= r) and (ell < 1)
        x = X[:, mu].copy()
        for _ in range(500): x = X @ p(x)
        err = np.linalg.norm(x - X[:, mu]) / M
        lam = float(np.linalg.eigvalsh(A(x))[-1])
        e_ok = (err < 0.05) and (lam < 0.5)
        cert += c_ok; emp += e_ok; errs.append(err); lams.append(lam)
        if c_ok: slacks.append(ell / max(lam, 1e-300))
    return dict(cert=cert / N, emp=emp / N, med_err=float(np.median(errs)), med_lam=float(np.median(lams)),
                minDelta_over_M2=float(min(Deltas) / M**2), slack_min=float(min(slacks)) if slacks else None,
                sig2_over_d=float(sig2 / d))
out = {}
for d, N in [(64, 4), (64, 32), (16, 32)]:
    for rho in [0.0, 0.3, 0.6]:
        for beta in ([0.15, 0.3, 0.6] if N <= d else [0.5, 1.0, 2.0]):
            rows = [run(d, N, rho, beta, s, 0.05) for s in (1, 2, 3)]
            agg = {k: (min(r[k] for r in rows) if k in ('cert', 'emp', 'minDelta_over_M2') else
                       (max(r[k] for r in rows) if k in ('med_err', 'med_lam', 'sig2_over_d') else
                        min((r[k] for r in rows if r[k] is not None), default=None))) for k in rows[0]}
            out[f"d={d},N={N},rho={rho},beta={beta}"] = agg
            print(f"d={d:3d} N={N:3d} rho={rho:.1f} beta={beta:.2f} | cert={agg['cert']:.2f} emp={agg['emp']:.2f} med_err={agg['med_err']:.3f} med_lam={agg['med_lam']:.3f} minDelta/M2={agg['minDelta_over_M2']:.3f} sig2/d={agg['sig2_over_d']:.2f} slack={agg['slack_min']}")
json.dump(out, open(RESULTS_DIR / 'resultsH.json', 'w'), indent=1)

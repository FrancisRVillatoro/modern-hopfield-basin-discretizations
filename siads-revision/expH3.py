"""Campaign H, revision: certified vs empirical retrieval over TEN independent
seeds per configuration (was three), reporting worst case and median, with the
same geometry, radii and criteria as the published campaign (seed conventions
unchanged; seeds 1..10)."""
import numpy as np, json

def run(d, N, rho, beta, seed, rfrac=0.05):
    rng = np.random.default_rng(seed)
    g0 = rng.standard_normal(d)
    G = np.sqrt(1 - rho) * rng.standard_normal((d, N)) + np.sqrt(rho) * g0[:, None]
    Xp = np.sqrt(d) * G / np.linalg.norm(G, axis=0); M = np.sqrt(d)
    sig2 = np.linalg.svd(Xp, compute_uv=False)[0] ** 2
    def p(x):
        z = beta * (Xp.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
    def A(x):
        pp = p(x); H = np.diag(pp) - np.outer(pp, pp); return beta * (Xp @ H @ Xp.T)
    r = rfrac * M
    cert = 0; emp = 0; slacks = []
    for mu in range(N):
        Delta = min((Xp[:, mu] - Xp[:, n]) @ Xp[:, mu] for n in range(N) if n != mu)
        eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r)); ell = 2 * beta * sig2 * eps
        c_ok = (2 * M * eps < r) and (ell < 1)
        x = Xp[:, mu].copy()
        for _ in range(500): x = Xp @ p(x)
        err = np.linalg.norm(x - Xp[:, mu]) / M
        lam = float(np.linalg.eigvalsh(A(x))[-1])
        e_ok = (err < 0.05) and (lam < 0.5)
        cert += c_ok; emp += e_ok
        if c_ok: slacks.append(ell / max(lam, 1e-300))
    return cert / N, emp / N, (min(slacks) if slacks else None)

out = {}
SEEDS = range(1, 11)
for d, N in [(64, 4), (64, 32), (16, 32)]:
    for rho in [0.0, 0.3, 0.6]:
        for beta in ([0.15, 0.3, 0.6] if N <= d else [0.5, 1.0, 2.0]):
            rows = [run(d, N, rho, beta, s) for s in SEEDS]
            certs = [r[0] for r in rows]; emps = [r[1] for r in rows]
            sl = [r[2] for r in rows if r[2] is not None]
            agg = dict(cert_min=min(certs), cert_med=float(np.median(certs)),
                       emp_min=min(emps), emp_med=float(np.median(emps)),
                       slack_min=(float(min(sl)) if sl else None))
            out[f"d={d},N={N},rho={rho},beta={beta}"] = agg
            print(f"d={d:3d} N={N:3d} rho={rho:.1f} beta={beta:.2f} | "
                  f"cert_min={agg['cert_min']:.2f} cert_med={agg['cert_med']:.2f} "
                  f"emp_min={agg['emp_min']:.2f} emp_med={agg['emp_med']:.2f} "
                  f"slack_min={agg['slack_min']}")
json.dump(out, open('/home/claude/siads/resultsH3.json', 'w'), indent=1)

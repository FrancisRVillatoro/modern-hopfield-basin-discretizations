"""Campaign G, revision: certified vs oracle overrelaxation with THIRTY
initializations per theta (was ten), reporting mean and standard deviation of
the iteration counts.  Same patterns and conventions (seed 20260809)."""
import numpy as np, json

rng = np.random.default_rng(20260809)
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N)))
X = np.sqrt(d) * Q
sig2 = np.linalg.svd(X, compute_uv=False)[0] ** 2
Delta = min((X[:, 0] - X[:, nu]) @ X[:, 0] for nu in range(1, N))
M = np.linalg.norm(X, axis=0).max(); r = 1.0
out = {}

def model(beta):
    def p(x):
        z = beta * (X.T @ x); z -= z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
    def A(x):
        pp = p(x); H = np.diag(pp) - np.outer(pp, pp); return beta * (X @ H @ X.T)
    return p, A

def run(beta, nrep=30):
    p, A = model(beta)
    eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r))
    ell = 2 * beta * sig2 * eps
    if 2 * M * eps > r or ell >= 1: return None
    rho = r - 2 * M * eps
    x = X[:, 0].copy()
    for _ in range(600): x = X @ p(x)
    xs = x
    lam = np.linalg.eigvalsh(A(xs)); lmax = float(lam[-1])
    lbar = lmax
    for _ in range(200):
        u = rng.standard_normal(d); u *= rho / np.linalg.norm(u)
        lbar = max(lbar, float(np.linalg.norm(A(xs + u), 2)))
    th_c = 2 / (2 - ell); th_o = 2 / (2 - lmax)
    def obs(theta, tol=1e-10, iters_max=4000):
        fac, its = 0.0, []
        for _ in range(nrep):
            u = rng.standard_normal(d); u *= 0.8 * rho / np.linalg.norm(u)
            y = xs + u; k = 0
            while np.linalg.norm(y - xs) > tol and k < iters_max:
                yn = (1 - theta) * y + theta * (X @ p(y))
                e0, e1 = np.linalg.norm(y - xs), np.linalg.norm(yn - xs)
                if e0 > 10 * tol: fac = max(fac, e1 / e0)
                y = yn; k += 1
            its.append(k)
        return fac, float(np.mean(its)), float(np.std(its))
    q1o, it1, s1 = obs(1.0)
    qco, itc, sc = obs(th_c)
    qoo, ito, so = obs(th_o)
    return dict(beta=beta, ell=float(ell), lmax=lmax, lbar=float(lbar),
                q1_obs=q1o, it1=it1, sd1=s1,
                theta_cert=float(th_c), qstar_cert=float(ell/(2-ell)), q_obs_cert=qco,
                itc=itc, sdc=sc, theta_oracle=float(th_o), q_obs_or=qoo, ito=ito, sdo=so)

for beta in [0.09, 0.10, 0.12, 0.15, 0.20, 0.25]:
    res = run(beta); out[f"beta={beta}"] = res
    if res:
        print(f"b={beta:.2f} ell={res['ell']:.3g} lmax={res['lmax']:.3g} lbar={res['lbar']:.3g} "
              f"q*={res['qstar_cert']:.4g} qobs*={res['q_obs_cert']:.4g} th*={res['theta_cert']:.4g} "
              f"it1={res['it1']:.1f}+-{res['sd1']:.1f} itc={res['itc']:.1f}+-{res['sdc']:.1f} "
              f"ito={res['ito']:.1f}+-{res['sdo']:.1f}")
json.dump(out, open('/home/claude/siads/resultsG2.json', 'w'), indent=1)

"""Campaign I: Gonzalez midpoint discrete gradient (DG) as exact-true-energy second-order benchmark.
Same configuration as Campaigns C and F (d=64, N=4, beta=0.15, seed 20260809).
Cost unit: one softmax-gradient evaluation Xp(.) = 1 unit; one energy E(.) = 0.5 unit (single X^T matvec + lse).
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json
rng = np.random.default_rng(20260809)
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N))); X = np.sqrt(d) * Q; beta = 0.15
M = np.linalg.norm(X, axis=0).max()
def p(x):
    z = beta * (X.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
def E(x):
    z = beta * (X.T @ x); m = z.max(axis=0); return 0.5 * (x * x).sum(axis=0) - (m + np.log(np.exp(z - m).sum(axis=0))) / beta
def grad(x): return x - X @ p(x)

def dg_step(x, dt, tol=1e-13, itmax=200):
    """Solve y = x - dt * DGgrad(x,y) by fixed point; count evaluations."""
    Ex = E(x); y = x - dt * grad(x); nev = 1.5  # one grad + one energy E(x)
    for _ in range(itmax):
        mid = 0.5 * (x + y); g = grad(mid); Ey = E(y); nev += 1.5
        dxy = y - x; n2 = dxy @ dxy
        corr = (Ey - Ex - g @ dxy) / n2 * dxy if n2 > 1e-30 else 0.0
        ynew = x - dt * (g + corr)
        if np.linalg.norm(ynew - y) < tol: y = ynew; break
        y = ynew
    return y, nev

def sav_run(x0, dt, n, R=None):
    R = R or max(np.linalg.norm(x0), M) + 1; C0 = M * R + np.log(N) / beta + 1.0
    def E1v(y):
        z = beta * (X.T @ y); m = z.max(); return -((m + np.log(np.exp(z - m).sum())) / beta)
    def bvec(y): return (-(X @ p(y))) / np.sqrt(E1v(y) + C0)
    x_prev = x0.copy(); th = 1 - np.exp(-dt); x = (1 - th) * x0 + th * (X @ p(x0)); r = np.sqrt(E1v(x) + C0)
    Es = [float(E(x0)), float(E(x))]
    for k in range(1, n):
        xb = 1.5 * x - 0.5 * x_prev; b = bvec(xb); Ad = 1 / dt + 0.5
        g = (1 / dt - 0.5) * x - b * r + 0.25 * b * (b @ x)
        Ainvg = g / Ad; corr = (0.25 * (b @ Ainvg) / (1 + 0.25 * (b @ b) / Ad)) * (b / Ad)
        xn = Ainvg - corr; r = r + 0.5 * (b @ (xn - x)); x_prev, x = x, xn; Es.append(float(E(x)))
    return x, np.array(Es)

# --- reference & trajectory error (Campaign C setup) ---
x_star = X[:, 0].copy()
for _ in range(200): x_star = X @ p(x_star)
u = rng.standard_normal(d); x0 = x_star + 0.9 * u / np.linalg.norm(u); T = 4.0
def rk4(x, h, nst):
    f = lambda y: -grad(y)
    for _ in range(nst):
        k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3); x = x + h/6*(k1+2*k2+2*k3+k4)
    return x
x_ref = rk4(x0.copy(), 1e-3, int(T / 1e-3))
res = {}
for dt in [0.4, 0.2, 0.1, 0.05, 0.025]:
    n = int(round(T / dt)); x = x0.copy(); nev = 0.0; Es = [float(E(x))]
    for _ in range(n):
        x, ne = dg_step(x, dt); nev += ne; Es.append(float(E(x)))
    Es = np.array(Es); dE = np.diff(Es)
    # exact-dissipation check: E(x^{k+1}) - E(x^k) = -(1/dt) ||x^{k+1}-x^k||^2 to roundoff
    res[f"dt={dt}"] = dict(err=float(np.linalg.norm(x - x_ref)), evals_per_step=nev / n,
                           max_dE=float(dE.max()))
eoc = [round(np.log2(res[f"dt={a}"]["err"] / res[f"dt={b}"]["err"]), 2)
       for a, b in [(0.4, 0.2), (0.2, 0.1), (0.1, 0.05), (0.05, 0.025)]]
out = dict(trajectory=res, eoc=eoc)

# --- exact identity check per step at dt=0.4 ---
x = x0.copy(); ident = 0.0
for _ in range(10):
    y, _ = dg_step(x, 0.4); ident = max(ident, abs(E(y) - E(x) + (1/0.4) * (y - x) @ (y - x))); x = y
out['identity_residual_dt0.4'] = float(ident)

# --- Campaign F setup: true energy monotonicity, DG vs SAV, evaluations to tolerance ---
viol = {}
for dt in [0.5, 1.0, 2.0]:
    worst_dg = 0.0; worst_sav = 0.0; ev_dg = []; ev_sav = []
    for rep in range(10):
        x0v = rng.standard_normal(d); x0v *= (1.5 * M) / np.linalg.norm(x0v)
        # DG
        x = x0v.copy(); nev = 0.0; Es = [float(E(x))]
        for k in range(200):
            x, ne = dg_step(x, dt); nev += ne; Es.append(float(E(x)))
        worst_dg = max(worst_dg, float(np.diff(Es).max())); ev_dg.append(nev / 200)
        _, Es2 = sav_run(x0v, dt, 200); worst_sav = max(worst_sav, float(np.diff(Es2).max())); ev_sav.append(1.0)
    viol[f"dt={dt}"] = dict(DG_worst_increase=f"{worst_dg:.2e}", DG_evals_per_step=round(float(np.mean(ev_dg)), 2),
                            SAV_worst_increase=f"{worst_sav:.2e}", SAV_evals_per_step=1.0)
out['true_energy'] = viol
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS_DIR / 'resultsI.json', 'w'), indent=1)

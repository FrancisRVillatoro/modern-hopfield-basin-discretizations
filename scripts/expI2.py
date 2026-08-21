from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json
from scipy.optimize import root
rng = np.random.default_rng(20260809)
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N))); X = np.sqrt(d) * Q; beta = 0.15
M = np.linalg.norm(X, axis=0).max()
def p(x):
    z = beta * (X.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
def E(x):
    z = beta * (X.T @ x); m = z.max(axis=0); return 0.5 * (x * x).sum(axis=0) - (m + np.log(np.exp(z - m).sum(axis=0))) / beta
def grad(x): return x - X @ p(x)
def dg_grad(x, y, Ex):
    mid = 0.5 * (x + y); g = grad(mid); dxy = y - x; n2 = dxy @ dxy
    if n2 < 1e-30: return g
    return g + (E(y) - Ex - g @ dxy) / n2 * dxy
def dg_step_newton(x, dt):
    Ex = E(x); cnt = [0]
    def F(y): cnt[0] += 1; return y - x + dt * dg_grad(x, y, Ex)
    y0 = x - dt * grad(x)
    sol = root(F, y0, method='hybr', tol=1e-12)
    y = sol.x
    # polish with a few Picard iterations for the exact identity
    pass
    return y, 1.5 * cnt[0], sol.success
viol = {}
for dt in [0.5, 1.0, 2.0, 4.0]:
    worst = 0.0; evs = []; fails = 0; ident = 0.0
    for rep in range(10):
        x0v = rng.standard_normal(d); x0v *= (1.5 * M) / np.linalg.norm(x0v)
        x = x0v.copy(); Es = [float(E(x))]; nev = 0.0
        for k in range(200):
            y, ne, ok = dg_step_newton(x, dt); nev += ne; fails += (not ok)
            ident = max(ident, abs(E(y) - E(x) + (y - x) @ (y - x) / dt)); x = y; Es.append(float(E(x)))
        worst = max(worst, float(np.diff(Es).max())); evs.append(nev / 200)
    viol[f"dt={dt}"] = dict(worst_increase=f"{worst:.2e}", evals_per_step=round(float(np.mean(evs)), 1),
                            solver_failures=fails, identity_residual=f"{ident:.1e}")
    print(dt, viol[f"dt={dt}"])
json.dump(viol, open(RESULTS_DIR / 'resultsI2.json', 'w'), indent=1)

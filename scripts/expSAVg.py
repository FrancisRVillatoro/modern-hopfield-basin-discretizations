"""gamma-resplit SAV (gamma=1/2): trajectory order (Campaign C setup) and true-energy test (Campaign F setup)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json
rng = np.random.default_rng(20260809)
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N))); X = np.sqrt(d) * Q; beta = 0.15
M = np.linalg.norm(X, axis=0).max(); gam = 0.5
C0 = M**2 / (2*gam) + np.log(N)/beta + 1.0
def p(x):
    z = beta*(X.T@x); z = z - z.max(axis=0); e = np.exp(z); return e/e.sum(axis=0)
def lse(x):
    z = beta*(X.T@x); m = z.max(axis=0); return (m + np.log(np.exp(z-m).sum(axis=0)))/beta
def E(x): return 0.5*(x*x).sum(axis=0) - lse(x)
def grad(x): return x - X@p(x)
def E1g(x): return gam/2*(x@x) - lse(x)
def bvec(x): return (gam*x - X@p(x))/np.sqrt(E1g(x)+C0)

def sav_run(x0, dt, n):
    th = 1 - np.exp(-dt)
    x_prev = x0.copy(); x = (1-th)*x0 + th*(X@p(x0))
    r = np.sqrt(E1g(x)+C0)
    Es = [float(E(x0)), float(E(x))]; law = 0.0
    for k in range(1, n):
        xb = 1.5*x - 0.5*x_prev; b = bvec(xb)
        Ad = 1/dt + (1-gam)/2
        g = (1/dt - (1-gam)/2)*x - b*r + 0.25*b*(b@x)
        Ai = g/Ad; corr = (0.25*(b@Ai)/(1 + 0.25*(b@b)/Ad))*(b/Ad)
        xn = Ai - corr
        rn = r + 0.5*(b@(xn-x))
        Emod0 = (1-gam)/2*(x@x) + r*r - C0
        Emod1 = (1-gam)/2*(xn@xn) + rn*rn - C0
        law = max(law, abs(Emod1 - Emod0 + (1/dt)*((xn-x)@(xn-x))))
        x_prev, x, r = x, xn, rn
        Es.append(float(E(x)))
    return x, np.array(Es), law

# Campaign C: trajectory order
x_star = X[:, 0].copy()
for _ in range(200): x_star = X@p(x_star)
u = rng.standard_normal(d); x0 = x_star + 0.9*u/np.linalg.norm(u); T = 4.0
def rk4(x, h, nst):
    f = lambda y: -grad(y)
    for _ in range(nst):
        k1=f(x); k2=f(x+h/2*k1); k3=f(x+h/2*k2); k4=f(x+h*k3); x = x + h/6*(k1+2*k2+2*k3+k4)
    return x
x_ref = rk4(x0.copy(), 1e-3, int(T/1e-3))
errs = []
for dt in [0.4, 0.2, 0.1, 0.05, 0.025]:
    xe, _, _ = sav_run(x0, dt, int(round(T/dt)))
    errs.append(float(np.linalg.norm(xe - x_ref)))
eoc = [round(np.log2(errs[i]/errs[i+1]), 2) for i in range(4)]
print("errors:", [f"{e:.3e}" for e in errs]); print("EOC:", eoc)

# Campaign F: true energy
out = {}
for dt in [0.5, 1.0, 2.0]:
    worst = 0.0; worst_law = 0.0
    for rep in range(10):
        x0v = rng.standard_normal(d); x0v *= (1.5*M)/np.linalg.norm(x0v)
        _, Es, law = sav_run(x0v, dt, 200)
        worst = max(worst, float(np.diff(Es).max())); worst_law = max(worst_law, law)
    out[f"dt={dt}"] = dict(worst_true_increase=f"{worst:.2e}", worst_law_residual=f"{worst_law:.2e}")
    print(dt, out[f"dt={dt}"])
json.dump(dict(errors=errs, eoc=eoc, F=out), open(RESULTS_DIR / 'resultsSAVg.json', 'w'), indent=1)

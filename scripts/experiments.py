"""Numerical experiments for the manuscript on time integration of modern
Hopfield retrieval dynamics.  Sections A-F as described in sec:experiments.
All computations in float64, fixed seeds, single file, numpy only.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import json

rng = np.random.default_rng(20260809)
results = {}

# ---------- model ----------------------------------------------------------
def make_model(X, beta):
    d, N = X.shape
    def lse(z):  # (N,) or (N,m) -> scalar or (m,)
        zb = beta * z
        m = zb.max(axis=0)
        return (m + np.log(np.exp(zb - m).sum(axis=0))) / beta
    def p(x):    # x: (d,) or (d,m)
        z = beta * (X.T @ x)
        z -= z.max(axis=0)
        e = np.exp(z)
        return e / e.sum(axis=0)
    def E(x):
        return 0.5 * (x * x).sum(axis=0) - lse(X.T @ x)
    def gradE(x):
        return x - X @ p(x)
    def hessE(x):  # single x only
        pp = p(x)
        H = np.diag(pp) - np.outer(pp, pp)
        return np.eye(d) - beta * (X @ H @ X.T)
    return p, E, gradE, hessE

def psi(x, theta, p, X):
    return (1 - theta) * x + theta * (X @ p(x))

# ---------- configuration for quantitative experiments (B, C, F) ------------
d, N = 64, 4
Q, _ = np.linalg.qr(rng.standard_normal((d, N)))
X = np.sqrt(d) * Q                      # orthogonal patterns, |xi| = 8
beta = 0.15
M = np.linalg.norm(X, axis=0).max()
sig2 = np.linalg.svd(X, compute_uv=False)[0] ** 2
p, E, gradE, hessE = make_model(X, beta)
Delta = min((X[:, 0] - X[:, nu]) @ X[:, 0] for nu in range(1, N))
r_ball = 1.0
eps = (N - 1) * np.exp(-beta * (Delta - 2 * M * r_ball))
ell = 2 * beta * sig2 * eps
nu_const = beta * sig2 / 2 - 1
results['config'] = dict(d=d, N=N, beta=beta, M=float(M), sigma2=float(sig2),
                         Delta=float(Delta), r=r_ball, eps=float(eps),
                         ell=float(ell), nu=float(nu_const))

# ---------- A: descent inequality verification ------------------------------
thetas = np.arange(0.1, 2.0, 0.1)
nsamp = 2000
Xs = rng.standard_normal((d, nsamp))
Xs *= (2 * M * rng.random(nsamp) ** (1 / d)) / np.linalg.norm(Xs, axis=0)
ratios = []
for th in thetas:
    g = gradE(Xs); g2 = (g * g).sum(axis=0)
    dec = E(Xs) - E(psi(Xs, th, p, X))
    bound = 0.5 * th * (2 - th) * g2
    mask = bound > 1e-12
    ratios.append(float((dec[mask] / bound[mask]).min()))
results['A_min_ratio_over_thetas'] = dict(zip([f"{t:.1f}" for t in thetas],
                                              [round(v, 4) for v in ratios]))
results['A_global_min_ratio'] = round(min(ratios), 4)

# ---------- B: local contraction rates --------------------------------------
# attractor near xi_1
xs = X[:, 0].copy()
for _ in range(200):
    xs = X @ p(xs)
x_star = xs
Hs = hessE(x_star)
ell_emp = float(np.linalg.norm(np.eye(d) - Hs, 2))   # spectral norm of beta X H X^T
results['B_retrieval_error'] = float(np.linalg.norm(x_star - X[:, 0]))
results['B_retrieval_bound_2Meps'] = float(2 * M * eps)
results['B_ell_bound'] = float(ell)
results['B_ell_empirical'] = ell_emp

def emp_factor_family(theta, nrep=20, r0=0.8, iters=25):
    fac = []
    for _ in range(nrep):
        u = rng.standard_normal(d); u *= r0 / np.linalg.norm(u)
        x = x_star + u
        for _ in range(iters):
            xn = psi(x, theta, p, X)
            e0, e1 = np.linalg.norm(x - x_star), np.linalg.norm(xn - x_star)
            if e1 < 1e-13:
                break
            fac.append(e1 / e0)
            x = xn
    return float(np.max(fac))

def ie_step_local(x, dt, tol=1e-14, itmax=500):
    y = x.copy()
    for _ in range(itmax):
        yn = (x + dt * (X @ p(y))) / (1 + dt)
        if np.linalg.norm(yn - y) < tol:
            return yn
        y = yn
    return y

def emp_factor_ie(dt, nrep=20, r0=0.8, iters=25):
    fac = []
    for _ in range(nrep):
        u = rng.standard_normal(d); u *= r0 / np.linalg.norm(u)
        x = x_star + u
        for _ in range(iters):
            xn = ie_step_local(x, dt)
            e0, e1 = np.linalg.norm(x - x_star), np.linalg.norm(xn - x_star)
            if e1 < 1e-13:
                break
            fac.append(e1 / e0)
            x = xn
    return float(np.max(fac))

tab = {}
for th in [0.25, 0.5, 1.0]:
    tab[f"family_theta={th}"] = dict(pred=float(1 - th * (1 - ell)),
                                     emp=round(emp_factor_family(th), 6))
for dt in [0.25, 1.0, 4.0, 16.0]:
    tab[f"IE_dt={dt}"] = dict(pred=float(1 / (1 + dt * (1 - ell))),
                              emp=round(emp_factor_ie(dt), 6))
results['B_contraction'] = tab

# ---------- C: order of accuracy --------------------------------------------
x0 = x_star + 0.9 * (lambda u: u / np.linalg.norm(u))(rng.standard_normal(d))
T = 4.0
def rk4(x, h, nst):
    f = lambda y: -gradE(y)
    for _ in range(nst):
        k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x
x_ref = rk4(x0.copy(), 1e-3, int(T/1e-3))

def run_scheme(name, dt):
    n = int(round(T/dt)); x = x0.copy()
    if name == 'SAV':
        R = max(np.linalg.norm(x0), M) + 1
        C0 = M*R + np.log(N)/beta + 1.0
        E1 = lambda y: -(E(y)*0 + ( (X.T@y)*0 ).sum() )  # placeholder, defined below
        def E1v(y): return -( (1/beta)*np.log(np.exp(beta*(X.T@y) - (beta*(X.T@y)).max()).sum()) + (beta*(X.T@y)).max()/beta )
        def bvec(y): return (-(X @ p(y))) / np.sqrt(E1v(y) + C0)
        r = np.sqrt(E1v(x) + C0)
        th = 1 - np.exp(-dt)
        x_prev = x.copy()
        x = psi(x, th, p, X)                       # ETD startup
        r = np.sqrt(E1v(x) + C0)                   # re-init r at x^1 (consistent)
        for k in range(1, n):
            xb = 1.5*x - 0.5*x_prev
            b = bvec(xb)
            A_diag = 1/dt + 0.5
            g = (1/dt - 0.5)*x - b*r + 0.25*b*(b @ x)
            # Sherman-Morrison for (A_diag I + 0.25 b b^T) xnew = g
            Ainvg = g / A_diag
            corr = (0.25 * (b @ Ainvg) / (1 + 0.25 * (b @ b) / A_diag)) * (b / A_diag)
            xnew = Ainvg - corr
            r = r + 0.5 * (b @ (xnew - x))
            x_prev, x = x, xnew
        return x
    th_map = {'EE': dt, 'CS': dt/(1+dt), 'ETD': 1-np.exp(-dt)}
    th = th_map[name]
    for _ in range(n):
        x = psi(x, th, p, X)
    return x

dts = [0.4, 0.2, 0.1, 0.05, 0.025]
err = {s: [float(np.linalg.norm(run_scheme(s, dt) - x_ref)) for dt in dts]
       for s in ['EE', 'CS', 'ETD', 'SAV']}
eoc = {s: [round(np.log2(err[s][i]/err[s][i+1]), 2) for i in range(len(dts)-1)]
       for s in err}
results['C_errors'] = {s: [f"{e:.3e}" for e in err[s]] for s in err}
results['C_eoc'] = eoc

# ---------- D: tunneling of the global prox in 1D ---------------------------
Xb1 = np.array([[1.0, -1.3]]); beta1 = 3.0
p1, E1d, g1d, _ = make_model(Xb1, beta1)
grid = np.linspace(-4, 4, 400001)
Eg = E1d(grid[None, :])
# attractors and their energies
def polish(x0):
    x = x0
    for _ in range(400):
        x = float((Xb1 @ p1(np.array([x]))).item())
    return x
a1 = polish(1.0); a2 = polish(-1.3)
E_a1, E_a2 = float(E1d(np.array([a1]))), float(E1d(np.array([a2])))
x_g = a2 if E_a2 < E_a1 else a1
delta = abs(E_a1 - E_a2)
x0t = 0.8
dtstar = (x_g - x0t)**2 / (2*delta)
def global_prox(x0v, dt):
    obj = Eg + (grid - x0v)**2 / (2*dt)
    return grid[int(np.argmin(obj))]
dts_scan = np.geomspace(0.05, 50, 400)
onset = None
for dt in dts_scan:
    y = global_prox(x0t, dt)
    if E1d(np.array([y])) < E_a1 - 1e-12 or abs(y - a2) < abs(y - a1):
        onset = float(dt); break
results['D_tunneling'] = dict(a_shallow=round(a1, 6), a_deep=round(a2, 6),
                              E_shallow=round(E_a1, 6), E_deep=round(E_a2, 6),
                              delta=round(delta, 6), x0=x0t,
                              dt_star_bound=round(dtstar, 4),
                              dt_onset_empirical=round(onset, 4),
                              nu_1d=float(beta1*np.linalg.svd(Xb1,compute_uv=False)[0]**2/2 - 1))

# ---------- E: 2D basins vs theta -------------------------------------------
X2 = np.array([[2.0, -1.0, -0.5], [0.0, 1.5, -1.8]])
beta2 = 1.5
p2, E2, g2, _ = make_model(X2, beta2)
# attractors
atts = []
for j in range(3):
    x = X2[:, j].copy()
    for _ in range(600):
        x = X2 @ p2(x)
    atts.append(x)
atts = np.array(atts).T          # (2,3)
ng = 301
gx = np.linspace(-3, 3, ng)
GX, GY = np.meshgrid(gx, gx)
P0 = np.vstack([GX.ravel(), GY.ravel()])     # (2, ng^2)
E0 = E2(P0)

def basins(theta, iters):
    Z = P0.copy()
    for _ in range(iters):
        Z = (1 - theta) * Z + theta * (X2 @ p2(Z))
    Dst = ((Z[:, :, None] - atts[:, None, :])**2).sum(axis=0)  # (npts,3)
    lab = Dst.argmin(axis=1)
    conv = Dst.min(axis=1) < 1e-6
    return lab, conv

lab_ref, conv_ref = basins(0.05, 4000)       # proxy for the flow
res_E = {}
for th, it in [(0.5, 1200), (1.0, 600), (1.5, 600), (1.9, 3000)]:
    lab, conv = basins(th, it)
    both = conv_ref & conv
    dis = both & (lab != lab_ref)
    frac = float(dis.sum()) / float(both.sum())
    Emin_dis = float(E0[dis].min()) if dis.any() else None
    res_E[f"theta={th}"] = dict(disagree_frac=round(frac, 5),
                                nonconv_frac=round(1 - float(conv.sum())/conv.size, 5),
                                min_E_on_disagreement=None if Emin_dis is None else round(Emin_dis, 4))
# estimate lowest saddle value: min E on boundary pixels of reference partition
labimg = lab_ref.reshape(ng, ng)
bnd = np.zeros_like(labimg, bool)
bnd[:-1, :] |= labimg[:-1, :] != labimg[1:, :]
bnd[:, :-1] |= labimg[:, :-1] != labimg[:, 1:]
c_saddle_est = float(E0.reshape(ng, ng)[bnd].min())
res_E['saddle_estimate_cstar_proxy'] = round(c_saddle_est, 4)
res_E['attractor_energies'] = [round(float(E2(atts[:, j])), 4) for j in range(3)]
results['E_basins'] = res_E

# ---------- F: SAV true-energy monotonicity ---------------------------------
def sav_run(x0v, dt, nsteps):
    R = max(np.linalg.norm(x0v), M) + 1
    C0 = M*R + np.log(N)/beta + 1.0
    def E1v(y):
        z = beta*(X.T@y); m = z.max()
        return -((m + np.log(np.exp(z-m).sum()))/beta)
    def bvec(y): return (-(X @ p(y))) / np.sqrt(E1v(y) + C0)
    x_prev = x0v.copy()
    th = 1 - np.exp(-dt)
    x = psi(x0v, th, p, X)
    r = np.sqrt(E1v(x) + C0)
    Es = [float(E(x0v)), float(E(x))]
    Emod = [None, float(0.5*(x@x) + r*r - C0)]
    for k in range(1, nsteps):
        xb = 1.5*x - 0.5*x_prev
        b = bvec(xb)
        A_diag = 1/dt + 0.5
        g = (1/dt - 0.5)*x - b*r + 0.25*b*(b @ x)
        Ainvg = g / A_diag
        corr = (0.25*(b @ Ainvg)/(1 + 0.25*(b @ b)/A_diag)) * (b / A_diag)
        xnew = Ainvg - corr
        r = r + 0.5*(b @ (xnew - x))
        x_prev, x = x, xnew
        Es.append(float(E(x)))
        Emod.append(float(0.5*(x@x) + r*r - C0))
    return np.array(Es), np.array([e for e in Emod if e is not None])

viol = {}
for dt in [0.5, 1.0, 2.0]:
    worst = 0.0; nv = 0; worst_mod = 0.0
    for rep in range(10):
        x0v = rng.standard_normal(d); x0v *= (1.5*M) / np.linalg.norm(x0v)
        Es, Emod = sav_run(x0v, dt, 200)
        dE = np.diff(Es)
        nv += int((dE > 1e-10).sum())
        if dE.max() > worst: worst = float(dE.max())
        dEm = np.diff(Emod)
        if dEm.max() > worst_mod: worst_mod = float(dEm.max())
    viol[f"dt={dt}"] = dict(true_E_violations=nv, worst_true_increase=f"{worst:.2e}",
                            worst_modified_increase=f"{worst_mod:.2e}")
results['F_sav'] = viol

print(json.dumps(results, indent=1))
with open(RESULTS_DIR / 'results.json', 'w') as fh:
    json.dump(results, fh, indent=1)

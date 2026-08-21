"""Campaign E, second generation.  d=2, N=3 modern Hopfield energy.
- reference flow basins by adaptive RK45 (rtol=atol=1e-12)
- all critical points by damped Newton from a dense multistart, classified by Hessian
- escape energy c* from the actual saddle values (component-connectivity check)
- stable manifolds of index-1 saddles integrated backward as continuous basin boundaries
- discrete basins of Psi_theta on the same grid; disagreement located in energy and in
  distance to the continuous separatrix
Outputs: resultsE2.json and fig_basins2.pdf
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np, json
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[2.0, -1.0, -0.5], [0.0, 1.5, -1.8]]); beta = 1.5
d, N = X.shape

def p(x):
    z = beta * (X.T @ x); z = z - z.max(axis=0); e = np.exp(z); return e / e.sum(axis=0)
def E(x):
    z = beta * (X.T @ x); m = z.max(axis=0)
    return 0.5 * (x * x).sum(axis=0) - (m + np.log(np.exp(z - m).sum(axis=0))) / beta
def grad(x): return x - X @ p(x)
def hess(x):
    pp = p(x); H = np.diag(pp) - np.outer(pp, pp); return np.eye(d) - beta * (X @ H @ X.T)
def f(t, x): return -grad(x)

# ---------- 1. critical points ----------
def newton(x0, it=100):
    x = x0.copy()
    for _ in range(it):
        g = grad(x)
        if np.linalg.norm(g) < 1e-14: break
        try: dx = np.linalg.solve(hess(x), g)
        except np.linalg.LinAlgError: return None
        lam = 1.0
        while lam > 1e-6 and np.linalg.norm(grad(x - lam*dx)) > np.linalg.norm(g): lam *= 0.5
        x = x - lam * dx
    return x if np.linalg.norm(grad(x)) < 1e-12 else None

crit = []
for a in np.linspace(-3, 3, 41):
    for b in np.linspace(-3, 3, 41):
        c = newton(np.array([a, b]))
        if c is not None and all(np.linalg.norm(c - q) > 1e-7 for q in crit): crit.append(c)
crit = np.array(crit)
idx = [int((np.linalg.eigvalsh(hess(c)) < 0).sum()) for c in crit]
minima = crit[[i for i, k in enumerate(idx) if k == 0]]
saddles = crit[[i for i, k in enumerate(idx) if k == 1]]
maxima = crit[[i for i, k in enumerate(idx) if k == 2]]
Emin = np.array([E(m) for m in minima]); Esad = np.array([E(s) for s in saddles])

# ---------- 2. reference flow basins ----------
ng = 301; gx = np.linspace(-3, 3, ng); GX, GY = np.meshgrid(gx, gx)
P0 = np.vstack([GX.ravel(), GY.ravel()])
def flow_endpoint(x0, T=60.0):
    sol = solve_ivp(f, (0, T), x0, method='RK45', rtol=1e-12, atol=1e-12,
                    events=None, dense_output=False)
    return sol.y[:, -1]
# vectorised: integrate the whole grid as one big system (independent ODEs)
def F_all(t, Z):
    Zm = Z.reshape(2, -1); return (-(Zm - X @ p(Zm))).ravel()
sol = solve_ivp(F_all, (0, 60.0), P0.ravel(), method='RK45', rtol=1e-11, atol=1e-11)
Zend = sol.y[:, -1].reshape(2, -1)
Dm = ((Zend[:, :, None] - minima.T[:, None, :])**2).sum(axis=0)
lab_flow = Dm.argmin(axis=1); conv_flow = Dm.min(axis=1) < 1e-8
# tolerance-independence check on a subsample
sub = rng_idx = np.random.default_rng(1).choice(P0.shape[1], 400, replace=False)
sol2 = solve_ivp(F_all, (0, 60.0), P0[:, sub].ravel(), method='RK45', rtol=1e-9, atol=1e-9)
Z2 = sol2.y[:, -1].reshape(2, -1)
lab2 = ((Z2[:, :, None] - minima.T[:, None, :])**2).sum(axis=0).argmin(axis=1)
tol_agree = float((lab2 == lab_flow[sub]).mean())

# ---------- 3. escape energies c* per minimum ----------
# for minimum j: c*_j = smallest saddle value s such that the sublevel component of x*_j at level s
# also contains another critical point; compute via connectivity on a fine grid.
def cell_contains(level, center, target, ngc=601):
    gxc = np.linspace(-3.5, 3.5, ngc); GXc, GYc = np.meshgrid(gxc, gxc)
    Eg = E(np.vstack([GXc.ravel(), GYc.ravel()])).reshape(ngc, ngc)
    mask = Eg < level
    from scipy.ndimage import label
    lab, _ = label(mask)
    def cell_of(pt):
        i = int(round((pt[1] + 3.5) / 7 * (ngc - 1))); j = int(round((pt[0] + 3.5) / 7 * (ngc - 1)))
        return lab[i, j]
    return cell_of(center) != 0 and cell_of(center) == cell_of(target)
cstar = []
for j, m in enumerate(minima):
    cands = []
    for s, es in zip(saddles, Esad):
        # just above the saddle value the saddle joins the component; c* = es if the joined
        # component then contains another critical point (always true for an index-1 saddle
        # separating two minima); check that at level es+1e-6 the cell of m contains s
        if cell_contains(es + 1e-6, m, s): cands.append(es)
    cstar.append(min(cands) if cands else np.inf)
cstar = np.array(cstar)

# ---------- 4. stable manifolds of index-1 saddles ----------
manifolds = []
for s in saddles:
    w, V = np.linalg.eigh(hess(s))
    vs = V[:, w > 0][:, 0]        # stable direction of the flow -grad: eigen of Hess >0
    for sgn in (+1, -1):
        x0 = s + sgn * 1e-6 * vs
        solb = solve_ivp(f, (0, -40.0), x0, method='RK45', rtol=1e-11, atol=1e-11,
                         dense_output=True, max_step=0.05)
        traj = solb.y
        keep = (np.abs(traj) < 3.2).all(axis=0)
        manifolds.append(traj[:, keep])

# ---------- 5. discrete basins and disagreement analysis ----------
def basins_disc(theta, iters):
    Z = P0.copy()
    for _ in range(iters): Z = (1 - theta) * Z + theta * (X @ p(Z))
    Dd = ((Z[:, :, None] - minima.T[:, None, :])**2).sum(axis=0)
    return Dd.argmin(axis=1), Dd.min(axis=1) < 1e-8
E0 = E(P0)
# distance of grid points to nearest manifold point
allman = np.hstack(manifolds) if manifolds else np.zeros((2, 0))
def dist_to_sep(pts):
    if allman.shape[1] == 0: return np.full(pts.shape[1], np.nan)
    out = np.empty(pts.shape[1])
    for k in range(0, pts.shape[1], 2000):
        blk = pts[:, k:k+2000]
        D2 = ((blk[:, :, None] - allman[:, None, :])**2).sum(axis=0)
        out[k:k+2000] = np.sqrt(D2.min(axis=1))
    return out
res = {}
for th, it in [(0.5, 1500), (1.0, 800), (1.5, 800), (1.9, 4000)]:
    lab, conv = basins_disc(th, it)
    both = conv & conv_flow
    dis = both & (lab != lab_flow)
    n_dis = int(dis.sum())
    frac = n_dis / int(both.sum())
    e_dis = E0[dis]
    # per-flow-basin check: any disagreeing point with E < c*_{lab_flow}?
    viol = int(((e_dis < cstar[lab_flow[dis]] - 1e-9)).sum()) if n_dis else 0
    dsep = dist_to_sep(P0[:, dis]) if n_dis else np.array([])
    res[f"theta={th}"] = dict(disagree_frac=float(frac), n_disagree=n_dis,
                              min_E_disagree=float(e_dis.min()) if n_dis else None,
                              violations_below_cstar=viol,
                              median_dist_to_separatrix=float(np.median(dsep)) if n_dis else None,
                              max_dist_to_separatrix=float(dsep.max()) if n_dis else None,
                              nonconv_frac=float(1 - conv.mean()))
grid_h = gx[1] - gx[0]
out = dict(minima=minima.tolist(), E_minima=Emin.tolist(), saddles=saddles.tolist(),
           E_saddles=Esad.tolist(), maxima=maxima.tolist(), n_critical=int(len(crit)),
           cstar=cstar.tolist(), grid_h=float(grid_h), flow_tol_agreement_subsample=tol_agree,
           flow_nonconv_frac=float(1 - conv_flow.mean()), discrete=res)
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS_DIR / 'resultsE2.json', 'w'), indent=1)

# ---------- 6. figure ----------
lab19, _ = basins_disc(1.9, 4000)
fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.1), sharey=True)
for ax, lab, ttl in [(axs[0], lab_flow, 'flow (RK45, tol $10^{-11}$)'), (axs[1], lab19, r'$\Psi_{1.9}$')]:
    ax.imshow(lab.reshape(ng, ng), origin='lower', extent=[-3, 3, -3, 3], cmap='Pastel1', interpolation='nearest')
    for man in manifolds: ax.plot(man[0], man[1], 'k-', lw=1.0)
    for j, m in enumerate(minima):
        cs = ax.contour(gx, gx, E(P0).reshape(ng, ng), levels=[cstar[j]], colors=['0.35'], linewidths=0.8, linestyles='--')
    ax.plot(X[0], X[1], 'k^', ms=6); ax.plot(minima[:, 0], minima[:, 1], 'ko', mfc='w', ms=5)
    ax.plot(saddles[:, 0], saddles[:, 1], 'kx', ms=7, mew=1.5)
    if len(maxima): ax.plot(maxima[:, 0], maxima[:, 1], 'k+', ms=8, mew=1.5)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel('$x_1$')
axs[0].set_ylabel('$x_2$')
fig.suptitle('Solid: stable manifolds of index-1 saddles (continuous basin boundaries). Dashed: $E=c^*$ per attractor.', fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES_DIR / 'fig_basins2.pdf', bbox_inches='tight')
print('figure saved')

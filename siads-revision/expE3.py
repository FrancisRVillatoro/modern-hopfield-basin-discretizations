"""Campaign E, third generation (SIADS revision): reproduces the published census,
basins, escape energies, separatrices and disagreement analysis, and ADDS the
validation requested in review: (i) a final-time/convergence audit of the flow
labels on the full grid, and (ii) a tolerance and horizon re-integration of ALL
basin-boundary nodes (where misclassification is most plausible), not a random
subsample.  Also regenerates the color figure fig_basins2.pdf.
Deterministic except where seeded (20260809 conventions).
"""
import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.ndimage import label as cclabel
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

# ---------- 1. critical points (damped Newton multistart) ----------
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

# ---------- 2. reference flow basins on the grid ----------
ng = 301; gx = np.linspace(-3, 3, ng); GX, GY = np.meshgrid(gx, gx)
P0 = np.vstack([GX.ravel(), GY.ravel()])
def F_all(t, Z):
    Zm = Z.reshape(2, -1); return (-(Zm - X @ p(Zm))).ravel()
sol = solve_ivp(F_all, (0, 60.0), P0.ravel(), method='RK45', rtol=1e-11, atol=1e-11)
Zend = sol.y[:, -1].reshape(2, -1)
Dm = ((Zend[:, :, None] - minima.T[:, None, :])**2).sum(axis=0)
lab_flow = Dm.argmin(axis=1)
dist_attr = np.sqrt(Dm.min(axis=1))
# --- NEW validation (i): final-time audit on the full grid ---
horizon = dict(max_dist_to_attractor_T60=float(dist_attr.max()),
               frac_dist_gt_1e6=float((dist_attr > 1e-6).mean()),
               frac_dist_gt_1e8=float((dist_attr > 1e-8).mean()))

# --- NEW validation (ii): re-integrate ALL basin-boundary nodes ---
L = lab_flow.reshape(ng, ng)
bmask = np.zeros_like(L, dtype=bool)
bmask[:-1, :] |= L[:-1, :] != L[1:, :]
bmask[1:, :]  |= L[1:, :] != L[:-1, :]
bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
bmask[:, 1:]  |= L[:, 1:] != L[:, :-1]
bidx = np.where(bmask.ravel())[0]
solB = solve_ivp(F_all, (0, 120.0), P0[:, bidx].ravel(), method='RK45', rtol=1e-9, atol=1e-9)
ZB = solB.y[:, -1].reshape(2, -1)
labB = ((ZB[:, :, None] - minima.T[:, None, :])**2).sum(axis=0).argmin(axis=1)
boundary = dict(n_boundary_nodes=int(bidx.size),
                n_label_changes_tol1e9_T120=int((labB != lab_flow[bidx]).sum()))

# ---------- 3. escape energies by sublevel-component connectivity ----------
def cell_contains(level, center, target, ngc=601):
    gxc = np.linspace(-3.5, 3.5, ngc); GXc, GYc = np.meshgrid(gxc, gxc)
    Eg = E(np.vstack([GXc.ravel(), GYc.ravel()])).reshape(ngc, ngc)
    lab2, _ = cclabel(Eg < level)
    def cell_of(pt):
        i = int(round((pt[1] + 3.5) / 7 * (ngc - 1))); j = int(round((pt[0] + 3.5) / 7 * (ngc - 1)))
        return lab2[i, j]
    return cell_of(center) != 0 and cell_of(center) == cell_of(target)
cstar = []
for j, m in enumerate(minima):
    cands = [es for s, es in zip(saddles, Esad) if cell_contains(es + 1e-6, m, s)]
    cstar.append(min(cands) if cands else np.inf)
cstar = np.array(cstar)

# ---------- 4. stable manifolds of index-1 saddles ----------
manifolds = []
for s in saddles:
    w, V = np.linalg.eigh(hess(s)); vs = V[:, w > 0][:, 0]
    for sgn in (+1, -1):
        solb = solve_ivp(f, (0, -40.0), s + sgn * 1e-6 * vs, method='RK45',
                         rtol=1e-11, atol=1e-11, max_step=0.05)
        tr = solb.y; keep = (np.abs(tr) < 3.2).all(axis=0); manifolds.append(tr[:, keep])
allman = np.hstack(manifolds)

# ---------- 5. discrete basins, violations, distances ----------
def basins_disc(theta, iters):
    Z = P0.copy()
    for _ in range(iters): Z = (1 - theta) * Z + theta * (X @ p(Z))
    Dd = ((Z[:, :, None] - minima.T[:, None, :])**2).sum(axis=0)
    return Dd.argmin(axis=1), Dd.min(axis=1) < 1e-8
E0 = E(P0)
def dist_to_sep(pts):
    out = np.empty(pts.shape[1])
    for k in range(0, pts.shape[1], 2000):
        blk = pts[:, k:k+2000]
        D2 = ((blk[:, :, None] - allman[:, None, :])**2).sum(axis=0)
        out[k:k+2000] = np.sqrt(D2.min(axis=1))
    return out
res = {}; conv_flow = dist_attr < 1e-8
for th, it in [(0.5, 1500), (1.0, 800), (1.5, 800), (1.9, 4000)]:
    lab, conv = basins_disc(th, it)
    both = conv & conv_flow
    dis = both & (lab != lab_flow)
    n_dis = int(dis.sum())
    e_dis = E0[dis]
    viol = int((e_dis < cstar[lab_flow[dis]] - 1e-9).sum()) if n_dis else 0
    dsep = dist_to_sep(P0[:, dis]) if n_dis else np.array([])
    res[f"theta={th}"] = dict(n_disagree=n_dis, violations=viol,
                              median_dist_sep=float(np.median(dsep)) if n_dis else None,
                              max_dist_sep=float(dsep.max()) if n_dis else None)
out = dict(E_minima=Emin.tolist(), E_saddles=Esad.tolist(), n_critical=int(len(crit)),
           cstar=cstar.tolist(), horizon=horizon, boundary=boundary, discrete=res,
           total_disagree=int(sum(v['n_disagree'] for v in res.values())))
json.dump(out, open('/home/claude/siads/resultsE3.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# ---------- 6. color figure (same filename as in the manuscript) ----------
lab19, _ = basins_disc(1.9, 4000)
fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True)
for ax, lab, ttl in [(axs[0], lab_flow, 'flow (RK45, tol $10^{-11}$)'), (axs[1], lab19, r'$\Psi_{1.9}$')]:
    ax.imshow(lab.reshape(ng, ng), origin='lower', extent=[-3, 3, -3, 3],
              cmap='Pastel1', vmin=0, vmax=8, interpolation='nearest')
    for man in manifolds: ax.plot(man[0], man[1], 'k-', lw=1.0)
    for c in np.unique(cstar):
        ax.contour(gx, gx, E0.reshape(ng, ng), levels=[c], colors=['0.35'],
                   linewidths=0.8, linestyles='--')
    ax.plot(X[0], X[1], 'k^', ms=6); ax.plot(minima[:, 0], minima[:, 1], 'ko', mfc='w', ms=5)
    ax.plot(saddles[:, 0], saddles[:, 1], 'kx', ms=7, mew=1.5)
    if len(maxima): ax.plot(maxima[:, 0], maxima[:, 1], 'k+', ms=8, mew=1.5)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel('$x_1$')
axs[0].set_ylabel('$x_2$')
fig.tight_layout(); fig.savefig('/home/claude/siads/fig_basins2.pdf', bbox_inches='tight')
print('figure saved')

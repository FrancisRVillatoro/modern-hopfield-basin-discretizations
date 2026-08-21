"""Reconstructed legacy Campaign E figure generator.

This script reconstructs the first-generation figure ``fig_basins.pdf`` from
the Campaign E code and parameters in ``experiments.py``.  The original
standalone plotting script was not present in the archived working bundle.

Legacy construction:
  * reference = Psi_0.05, used as a near-flow proxy;
  * comparison = Psi_1.9;
  * 301 x 301 grid on [-3,3]^2;
  * one escape-level proxy obtained as the minimum energy on boundary pixels
    of the reference basin partition.

The current manuscript uses ``expE2.py`` / ``fig_basins2.pdf`` instead.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

X = np.array([[2.0, -1.0, -0.5], [0.0, 1.5, -1.8]])
beta = 1.5

def p(x):
    z = beta * (X.T @ x)
    z -= z.max(axis=0)
    e = np.exp(z)
    return e / e.sum(axis=0)

def E(x):
    z = beta * (X.T @ x)
    m = z.max(axis=0)
    return 0.5 * (x*x).sum(axis=0) - (
        m + np.log(np.exp(z-m).sum(axis=0))
    ) / beta

atts = []
for j in range(3):
    x = X[:, j].copy()
    for _ in range(600):
        x = X @ p(x)
    atts.append(x)
atts = np.array(atts).T

ng = 301
gx = np.linspace(-3, 3, ng)
GX, GY = np.meshgrid(gx, gx)
P0 = np.vstack([GX.ravel(), GY.ravel()])
E0 = E(P0)

def basins(theta, iters):
    Z = P0.copy()
    for _ in range(iters):
        Z = (1-theta)*Z + theta*(X @ p(Z))
    dst = ((Z[:, :, None] - atts[:, None, :])**2).sum(axis=0)
    return dst.argmin(axis=1), dst.min(axis=1) < 1e-6

lab_ref, _ = basins(0.05, 4000)
lab19, _ = basins(1.9, 3000)

labimg = lab_ref.reshape(ng, ng)
boundary = np.zeros_like(labimg, dtype=bool)
boundary[:-1, :] |= labimg[:-1, :] != labimg[1:, :]
boundary[:, :-1] |= labimg[:, :-1] != labimg[:, 1:]
cstar_proxy = float(E0.reshape(ng, ng)[boundary].min())

fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.6), sharey=True)
for ax, lab, title in [
    (axs[0], lab_ref, r"reference ($\Psi_{0.05}$, proxy for the flow)"),
    (axs[1], lab19, r"$\Psi_{1.9}$"),
]:
    ax.imshow(
        lab.reshape(ng, ng), origin="lower", extent=[-3,3,-3,3],
        cmap="Pastel1", interpolation="nearest"
    )
    ax.contour(
        gx, gx, E0.reshape(ng, ng), levels=[cstar_proxy],
        colors="k", linewidths=1.2, linestyles="--"
    )
    ax.plot(X[0], X[1], "k^", ms=6, label="patterns")
    ax.plot(atts[0], atts[1], "ko", mfc="w", ms=5, label="attractors")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

axs[0].set_ylabel(r"$x_2$")
axs[0].legend(loc="upper left", fontsize=8)
fig.suptitle(
    r"Basins of attraction; black curve: level set $E=c^\ast$ "
    r"(grid saddle estimate)",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_basins.pdf", bbox_inches="tight")
print(f"saved {FIGURES_DIR / 'fig_basins.pdf'}")
print(f"grid saddle proxy c* = {cstar_proxy:.8f}")

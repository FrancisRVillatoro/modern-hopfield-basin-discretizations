# Campaign E reproducibility checks

Run the first two scripts in this directory with Python 3, NumPy, SciPy, and mpmath 1.3 or later. The third script is described at the end.

```bash
python3 certify_planar_critical_points.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_planar_horizons.py
```

## Critical points and escape levels

A critical point is a convex combination of the three patterns, so the first
script starts from the rectangle $[-1,2]\times[-1.8,1.5]$. It uses outward-rounded
intervals to enclose the gradient and Hessian. The Hessian uses the softmax
covariance in the pairwise form
$$
\nabla^{2} E(x) = I-\beta \sum_{i<j} p_i(x)p_j(x)
   (\xi_i-\xi_j)(\xi_i-\xi_j)^\top.
$$

For a box $B$ with midpoint $b$, gradient $g$, interval Hessian $H(B)$,
and an invertible point preconditioner $C$, any root in $B$ belongs to

$$
K(B) = b-Cg(b) +(I-CH(B))(B-b).
$$

An image strictly inside the box and an infinity norm bound
$\lVert I-CH(B)\rVert_{\infty}<1$ give existence and uniqueness by
the contraction theorem. In the full subdivision, boxes are discarded only
if the gradient excludes zero, if their Krawczyk image is disjoint from the
box, or if their possible critical points lie inside a previously verified
root box. The Hessian determinant and first diagonal entry determine the
Morse index in dimension two.

The script also verifies four chords from the lowest two saddle points to
minimum points. For each saddle $s$ and minimum $m$, it establishes
$E(s+t(m-s))<E(s)$ for $0<t\le1$. It uses negative directional curvature
near the stationary endpoint and direct interval evaluation by subdivision
on the rest of the chord. The lowest saddle attaches to the minima near
$(-0.5,-1.8)$ and $(2,0)$. The second saddle attaches to the minima near
$(-1,1.5)$ and $(2,0)$. The first three critical levels are those of the
minima. Since the energy is coercive and the critical points are
nondegenerate, the sublevel topology is constant between successive critical
values and a single index-one saddle joins at most two components. Together
with the chords, this fixes the escape levels. They lie in
$[-1.145436380056,-1.145436380054]$ for the first pair and in
$[-0.869210114869,-0.869210114866]$ for the third minimum.

## Numerical flow and discrete labels

The second script checks the finite-horizon flow labels under two
integration horizons and two tolerances. A boundary grid node has at least
one horizontal or vertical neighbor with another reference label. All
90,601 grid points retain their reference labels after doubling the horizon,
as do all 894 boundary nodes at reduced accuracy and the previously sampled
400 grid points. The discrete maps are run with the original iteration caps.
Each RK45 run uses one adaptive controller for the full batch advanced in
that run. Flow and map labels come from numerical endpoint comparisons and
the plotted separatrices are numerical traces. The script evaluates the
energy at exact grid coordinates by interval arithmetic on all observed
disagreement nodes to verify that each lies at least 0.0265 above the
upper endpoint of its reference label's escape energy.

## Independent boundary integration

The following program integrates each of the 894 detected boundary initial
conditions separately with DOP853 to T=120 at relative tolerance 1e-12 and
absolute tolerance 1e-13. It writes the label comparison and the maximum
distance to the assigned minimum in independent_boundary_results.json.

~~~bash
python3 audit_independent_boundary.py
~~~

# Campaigns G and H

`audit_gh.py` runs the overrelaxation sweep of Campaign G with thirty
initializations per inverse temperature and the study of correlated patterns
of Campaign H with ten pattern seeds per configuration. It writes one JSON file
with the entries `G` and `H` to the path given on the command line.

```bash
OPENBLAS_NUM_THREADS=1 python audit_gh.py results_gh.json
```

Each Campaign G row gives the inverse temperature, the bound `ell` for `r = 1`,
the largest eigenvalue of the softmax curvature at the equilibrium and the
maximum of that eigenvalue over the equilibrium and 200 points of a sphere
around it, the certified factor `ell/(2-ell)`, and,
for the relaxation parameters 1, `theta_star` and `theta_o`, the largest
observed contraction factor and the mean and sample standard deviation of the
iteration counts. Each Campaign H configuration lists, for every pattern seed,
the numbers of certified and empirically retrieved patterns, the normalized
minimum separation, the normalized squared largest singular value, the minimum
slack over certified patterns, the largest fixed-point residual and the minimum
ratio of observed to certified energy decrease.

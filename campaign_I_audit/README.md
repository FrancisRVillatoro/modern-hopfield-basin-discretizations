# Campaign I nonlinear solver audit

This replay uses the same seed, pattern bank, initial radius, step sizes,
hybrid Newton solver and fixed run length as the archived
`scripts/expI2.py`. Every step advances with the returned state,
including when the solver reports non-success. The audit logs the returned
status and both residual diagnostics for every attempt.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python audit_campaign_i.py reproduced
```

Tested with Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. The JSON gives
the full-precision summary, and the CSV contains all 8,000 attempts.
The nonlinear residual is the Euclidean norm of
`y - x + h * discrete_gradient(x, y)`. The identity defect is
`E(y) - E(x) + ||y-x||² / h`. In exact arithmetic, the latter is the
inner product of the nonlinear residual vector with the step, divided
by `h`. The reported difference between these scalars checks that relation
in the floating-point implementation.

The 1.5 evaluation units per nonlinear function call reproduce the archive's
accounting. This count is specific to the implementation and runs of fixed
length. The residual threshold `1e-10` is used only in a diagnostic column.

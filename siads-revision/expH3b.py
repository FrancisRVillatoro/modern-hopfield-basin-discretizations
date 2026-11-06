"""Campaign H companion: dissipation-ratio test and table covariates over the
same TEN seeds as expH3.py."""
import numpy as np, json
def mk(d, N, rho, seed):
    rng = np.random.default_rng(seed)
    g0 = rng.standard_normal(d)
    G = np.sqrt(1-rho)*rng.standard_normal((d, N)) + np.sqrt(rho)*g0[:, None]
    return np.sqrt(d)*G/np.linalg.norm(G, axis=0), rng
ratio_min = np.inf; slack_max = 0.0; worst_at = None
cov = {}
for d, N in [(64,4),(64,32),(16,32)]:
    for rho in [0.0,0.3,0.6]:
        mind = np.inf; maxs = 0.0
        for beta in ([0.15,0.3,0.6] if N <= d else [0.5,1.0,2.0]):
            for seed in range(1,11):
                X, rng = mk(d, N, rho, seed)
                M = np.sqrt(d); sig2 = np.linalg.svd(X, compute_uv=False)[0]**2
                Dl = min(min((X[:,mu]-X[:,n])@X[:,mu] for n in range(N) if n!=mu) for mu in range(N))
                mind = min(mind, Dl/d); maxs = max(maxs, sig2/d)
                def p(x):
                    z = beta*(X.T@x); z=z-z.max(axis=0); e=np.exp(z); return e/e.sum(axis=0)
                def E(x):
                    z = beta*(X.T@x); m=z.max(axis=0)
                    return 0.5*(x*x).sum(axis=0)-(m+np.log(np.exp(z-m).sum(axis=0)))/beta
                Xs = rng.standard_normal((d,500)); Xs *= (2*M*rng.random(500)**(1/d))/np.linalg.norm(Xs,axis=0)
                for th in [0.5,1.0,1.5,1.9]:
                    g = Xs - X@p(Xs); g2 = (g*g).sum(axis=0)
                    dec = E(Xs) - E((1-th)*Xs + th*(X@p(Xs)))
                    b = 0.5*th*(2-th)*g2; m = b > 1e-12
                    if m.any():
                        rr = float((dec[m]/b[m]).min())
                        if rr < ratio_min: ratio_min = rr; worst_at = (d,N,rho,beta,th)
                        slack_max = max(slack_max, float((dec[m]/b[m]-1).max()*0+ (dec[m]/b[m]).min()-1) if False else slack_max)
        cov[f"d={d},N={N},rho={rho}"] = dict(minDelta_over_d=float(mind), max_sig2_over_d=float(maxs))
# recompute max slack of the MINIMUM ratio above 1 across configs properly:
print("global min ratio:", ratio_min, "at", worst_at)
for k,v in cov.items(): print(k, v)
json.dump(dict(min_ratio=ratio_min, worst_at=worst_at, covariates=cov), open('resultsH3b.json','w'), indent=1)

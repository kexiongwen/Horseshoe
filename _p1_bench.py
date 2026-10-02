"""P1 optimization benchmark: per-step cost after optimization, plus the
paper-scale approximate chain behavior with the corrected direct-route
construction (sqrt-weighted dsyrk product)."""
import time

import numpy as np

import horseshoe as H

n, p = 2000, 20000
rng0 = np.random.default_rng(0)
beta_true = np.zeros(p)
beta_true[:23] = 2.0 ** (-np.arange(-2.0, 3.5 + 1e-9, 0.25))
X = rng0.standard_normal((n, p))
y = X @ beta_true + np.sqrt(2.0) * rng0.standard_normal(n)
print(f"data ready: n={n}, p={p}", flush=True)

t0 = time.perf_counter()
H.horseshoe(y, X, burnin=0, mcmc=5, thin=1, n_keep=5, beta_true=beta_true,
            is_sim=True, verbose=0, approx_xlx=False,
            rng=np.random.default_rng(1))
t_exact = (time.perf_counter() - t0) / 5
print(f"EXACT  (optimized): {t_exact:.2f} s/iter   "
      "(pre-optimization measured 1.88-3.00 s/iter)", flush=True)

r = H.horseshoe(y, X, burnin=300, mcmc=700, thin=1, n_keep=100,
                beta_true=beta_true, is_sim=True, verbose=100,
                approx_xlx=True, delta=1e-4, a0=1, b0=1,
                rng=np.random.default_rng(5171))
t_ap = r.elapsed / 1000
s2 = r.sigma_sq_samples
print(f"\nAPPROX : {t_ap:.3f} s/iter avg over 1000 iters; "
      f"sigma2 tail {s2[-200:].mean():.3g}; mean|S| {r.mean_active_set:.0f} "
      f"(n/2={n/2:.0f})", flush=True)
print(f"per-step speedup vs optimized exact: {t_exact/t_ap:.1f}x;  "
      f"vs pre-optimization exact (1.88 s): {1.88/t_ap:.1f}x", flush=True)
print(f"beta corr vs truth: "
      f"{np.corrcoef(r.beta_mean, beta_true[:100])[0,1]:.3f}, "
      f"coverage {100*r.coverage:.1f}%, mse {r.mse:.4f}", flush=True)

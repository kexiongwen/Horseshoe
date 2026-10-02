# Horseshoe MCMC — Python Implementation

Implementation of the horseshoe samplers described in

> Johndrow, Orenstein & Bhattacharya (2020).
> *Scalable Approximate MCMC Algorithms for the Horseshoe Prior.* JMLR.

It contains both the **exact** and the **approximate** (thresholding +
Woodbury) horseshoe samplers, modular samplers for the local scales
(`eta_sampler.py`, Appendix S1), for the regression coefficients
(`beta_sampler.py`, the surrogate-variable draw of eq. (5)/§2.2), for
the global-precision Metropolis step (`xi_sampler.py`, eqs. (3)-(4) with
all three evaluation routes) and for the noise variance
(`sigma2_sampler.py`, step 3: the exact conjugate draw), plus a
simulation driver and a correctness verification suite.

## Dependencies

```
python >= 3.10, numpy, scipy
(optional, for plots: matplotlib)
```

## Project layout

| File | Contents |
|---|---|
| `horseshoe.py` | `horseshoe()` main sampler and `HorseshoeResult` dataclass; `plot_diagnostics()` |
| `xi_sampler.py` | `update_xi()`: ξ RW-Metropolis step (eqs. (3)–(4)); three evaluation routes (`exact` / thresholded direct / Woodbury); truncated-target rejection; returns an `XiUpdate` |
| `sigma2_sampler.py` | `update_sigma2()`: exact conjugate draw, σ² ~ InvGamma((n+a0)/2, (ssr+b0)/2) |
| `beta_sampler.py` | `sample_beta()`: β surrogate-variable draw (eq. (5)/§2.2); pure deterministic math, three routes (`exact` / `approx_direct` / `woodbury`) |
| `eta_sampler.py` | `sample_eta()` and friends: exact rejection sampler for the local scales (Appendix S1), with lower-truncation support |
| `sims.py` | simulation driver; paper's default configuration n=2000, p=20000, δ=1e-4 |
| `verify_port.py` | correctness verification suite (9 checks, see below) |
| `_p1_bench.py` | performance benchmark at the paper scale |

## Quick start

```bash
python verify_port.py          # correctness checks (see below)
python sims.py --test          # small smoke test (n=200, p=500)
python sims.py                 # paper's default configuration (slow)
python sims.py --algorithm exact --n 1000 --p 2000 --mcmc 5000
```

or as a library:

```python
import numpy as np
from horseshoe import horseshoe

res = horseshoe(y, X, burnin=1000, mcmc=5000, approx_xlx=True,
                delta=1e-4, beta_true=beta_true, rng=np.random.default_rng(1))
print(res.beta_mean, res.sigma_sq_mean, res.acc_rate_xi, res.mean_active_set)
```

## Prior truncation variants (optional)

`horseshoe()` takes two truncation arguments (both off by default = the
main-text model of the paper):

- **`eta_lb = b > 0`**: lower-truncates each η_j to (b, ∞) (equivalently an
  upper cap λ_j = η_j^{-1/2} ≤ b^{-1/2} on the local scales) — the modified
  prior of Appendix B eq. (26) / Appendix S2's convergence analysis.
  Implementation: coordinates with M > 1 use a truncated-exponential
  proposal with acceptance probability (1+b)/(1+t) (naive rejection of
  t ≤ b needs ~e^{γb} retries for strongly identified coordinates and
  hangs); M ≤ 1 uses support rejection under the same envelope. KS ≤ 0.0023
  in all tested regimes.
- **`xi_bounds = (lo, hi)`**: restricts ξ to [lo, hi] (equivalently
  τ ∈ [hi^{-1/2}, lo^{-1/2}]); proposals outside the interval are rejected
  outright (= MH on the truncated target).

These implement the modified prior that the paper's Appendix B/S2
convergence analysis works on.  Note `eta_lb` caps the local scales and
biases E[σ²] downward on the test design (0.48 at `eta_lb=0.5` vs 1.02
untruncated, true σ² = 2).

## Verification (`python verify_port.py`)

1. **Envelope validity**: f_L ≤ f on dense grids for M ∈ {1e-8 … 1}
   (holds by construction).
2. **η sampler exactness**: KS distance of N=200,000 samples against the
   numerically integrated CDF, M ∈ {1e-8 … 50}, all < 0.004; a legacy
   reuse-uniform variant is reported alongside for comparison (max KS
   0.12). Average draws per accepted sample ≈ 1.0–1.4 (high acceptance,
   as the paper claims).
3. **Woodbury identity and SVD determinant**: machine-precision agreement
   (~1e-13) with a direct solve / `slogdet`; the `diag(chol(Gram))` vs
   singular-value discrepancy is demonstrated; the direct route's
   `cho_factor`-diagonal logdet and the woodbury route's Sylvester
   `|Γ_S|·|A|` logdet both have `slogdet` regression tests (guarding
   against "extra sqrt"-type bugs).
4. **End-to-end**: exact vs approximate on a small sparse problem; both
   recover the signal (corr > 0.8 asserted for each), their posterior
   means agree (mutual corr > 0.95 asserted) and exact E[σ²] stays in a
   sane range — route-level construction regressions break these.
5. **Approximate β conditional**: Monte Carlo comparison against the
   analytic transition density of the approximate β update (the symmetric
   Appendix B eq. (30) form; max relative deviation 3.9e-3).
6. **Truncation variants**: the `eta_lb` sampler over 6 (m, lb) pairs
   (including m·lb ≫ 1 cases that hang naive rejection) with KS ≤ 0.0024
   (truth CDF computed in log space); `xi_bounds` rejects out-of-bounds
   proposals with probability 1; an end-to-end `eta_lb=0.5` exact run
   stays healthy (corr > 0.9, coverage > 0.8 asserted; E[σ²] biased low
   relative to untruncated, as capping λ must); config recording asserted.
7. **Real-data mode (`is_sim=False`)**: a synthetic train/test split
   exercises keep_id selection (all true signals land in the first half),
   the streaming beta_hat, test-set losses, the exact route's
   `id1=None` branch, and the config field; both routes run untruncated
   (the default).
8. **Pipeline conditionals**: the direct route's M construction and the β
   draw of all three routes through the actual code — deterministic mean
   (zero randomness; machine precision), MH decisions bracketed around
   hand-computed log ratios on **both the direct and woodbury routes**,
   and Monte-Carlo covariance vs the analytic eq. (28) / eq. (30)
   targets.  Regression tests for the dsyrk weight-squaring bug
   (`dsyrk(X·w)` is `X diag(w²) X'`) and the Sylvester logdet.
9. **Edge cases**: `mcmc = 0`, `thin > mcmc`, `burnin < 0`, `thin = 0`
   are rejected with `ValueError` (configurations that store no draws);
   the relative-loss diagnostics stay finite under an all-null truth
   (zero-norm denominators fall back to 1, so the losses read
   1 − |error|).

`HorseshoeResult.config` records all run parameters (n/p/burnin/mcmc/
a0/b0/δ/eta_lb/xi_bounds/...) for reproducibility.

## Main API

`horseshoe(y, X, *, burnin, mcmc, thin, mh_scale_ub, mh_scale_lb, phasein,
a0, b0, approx_xlx, delta, n_keep, beta_true,
is_sim, y_test, X_test, rng, a_eta, b_eta, legacy_eta_sampler,
lam_init, tau_init, sigma_sq_init, eta_lb, xi_bounds,
save_samples, save_dir, simtype, verbose) -> HorseshoeResult`

- `delta` may be a scalar or an array of length `burnin+mcmc` (a decreasing
  schedule gives the asymptotically exact algorithm of the paper's §3.3).
- The credible intervals, `se` and `coverage` use saved draws from
  post-burnin iteration 5001 on (the cutoff counts post-burnin
  iterations, so it does not drift with `thin`); with fewer qualifying
  draws the second half of the chain is used.  `mcmc = 0`, `thin > mcmc`,
  `burnin < 0` and `thin < 1` are rejected with `ValueError`.
- `eta_lb` / `xi_bounds` implement the truncated prior of Appendix B
  (optional variant; see "Prior truncation variants" above).
- `legacy_eta_sampler=True` switches the local-scale sampler back to its
  legacy (inexact) behavior: the component-selection uniform is reused for
  the acceptance test.
- `HorseshoeResult` fields: `beta_mean / beta_median / lambda_mean /
  sigma_sq_mean / beta_samples / xi_samples / sigma_sq_samples / ci_lo /
  ci_hi / coverage / mse / keep_id / acc_rate_xi / mean_active_set /
  elapsed / config`, among others.

"""Correctness verification suite.

Run:  python verify_port.py

Checks
------
1. Envelope validity: f_L <= f on dense grids for a range of M.
2. Exactness of the eta rejection samplers: KS distance between large
   samples and the numerically-integrated CDF of h(t) ~ exp(-M t)/(1+t),
   for both the sampler (independent acceptance uniform, per Appendix S1)
   and a legacy variant that reuses the component-selection uniform
   (shown to be inexact for M near 1).
3. Woodbury identity and SVD/Cholesky log determinants used by the
   sampler, plus a demonstration that the legacy diag(chol(X_S' X_S))
   choice is not the vector of singular values.
4. End-to-end: exact vs approximate samplers on a small sparse simulation;
   posterior means should agree closely.
5. Approximate beta conditional: Monte Carlo covariance of the beta draw
   vs the analytic transition density (Appendix B eq. (30)).
6. Truncation variants: eta_lb sampler exactness, xi_bounds semantics,
   and an end-to-end eta_lb run.
7. Real-data mode (is_sim=False): keep_id selection, streaming beta_hat,
   test-set losses, and config recording.
8. Pipeline conditionals: the direct route's M construction and the beta
   draw of all three routes through the actual update_xi / sample_beta
   code -- deterministic mean (zero randomness), MH decisions bracketed
   around hand-computed log ratios on BOTH the direct and the woodbury
   routes, and Monte-Carlo covariance vs the analytic eq. (28) / eq. (30)
   targets.  Regression test for the dsyrk weight-squaring bug
   (dsyrk(X*w) is X diag(w^2) X', not X diag(w) X').
9. Edge cases: input validation (mcmc/thin/burnin combinations that store
   no draws are rejected), and finite relative-loss diagnostics under an
   all-null truth (zero-norm denominators fall back to 1).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import exp1

from beta_sampler import sample_beta
from eta_sampler import approx_constants, hseta_rs_small_m, neg_log_post, sample_eta
from horseshoe import horseshoe
from xi_sampler import log_marg, update_xi

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def header(msg):
    print(f"\n=== {msg} ===")


# ---------------------------------------------------------------------------
# 1. envelope validity: f_L <= f
# ---------------------------------------------------------------------------
def piecewise_fL(x, M, a=0.2, b=10.0):
    A, I, B = (neglog(M, k / M) for k in (a, 1.0, b))
    lam2, lam3 = M * (I - A) / (1 - a), M * (B - I) / (b - 1.0)
    y = np.where(x < a / M, np.log1p(x), np.nan)
    y = np.where((x >= a / M) & (x < 1 / M), A + lam2 * (x - a / M), y)
    y = np.where((x >= 1 / M) & (x < b / M), I + lam3 * (x - 1 / M), y)
    y = np.where(x >= b / M, B + M * (x - b / M), y)
    return y


def neglog(M, x):
    return M * x + np.log1p(x)


def check_envelope():
    header("1. envelope validity f_L <= f")
    worst = 0.0
    for M in (1e-8, 1e-4, 1e-2, 0.1, 0.5, 1.0):
        grid = np.unique(
            np.concatenate([np.linspace(0, 12 / M, 200000),
                            np.logspace(-12, np.log10(12 / M), 200000)])
        )
        viol = np.max(piecewise_fL(grid, M) - neglog(M, grid))
        worst = max(worst, viol)
        assert viol < 1e-9, (M, viol)
    print(f"  {PASS} max(f_L - f) over all M and grids = {worst:.2e}")


# ---------------------------------------------------------------------------
# 2. eta sampler exactness (KS test against numerically integrated CDF)
# ---------------------------------------------------------------------------
def true_cdf_fn(M):
    Z = np.exp(M) * exp1(M)  # integral of e^{-Mt}/(1+t), t in [0, inf)

    def cdf(pts):
        out = np.empty_like(pts)
        interior = pts[pts <= 30.0 / M]
        if interior.size:
            grid = np.unique(
                np.concatenate([
                    [0.0],
                    np.logspace(-14, np.log10(max(interior.max(), 1e-13)), 300000),
                ])
            )
            c = np.concatenate(
                [[0.0], cumulative_trapezoid(np.exp(-M * grid) / (1 + grid), grid)]
            ) / Z
            out[pts <= 30.0 / M] = np.interp(interior, grid, c)
        out[pts > 30.0 / M] = 1.0  # tail mass beyond 30/M is negligible
        return out

    return cdf


def ks_statistic(samples, cdf):
    s = np.sort(samples)
    F = cdf(s)
    n = s.size
    return np.max(np.abs(np.arange(1, n + 1) / n - F))


def check_eta_sampler():
    header("2. eta rejection sampler vs true CDF (KS, N=200000)")
    rng = np.random.default_rng(11)
    Ms = [1e-8, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 50.0]
    crit = 1.95 / np.sqrt(200_000)  # ~alpha=0.001 critical value
    print(f"  KS critical value (alpha~0.001): {crit:.4f}")
    print(f"  {'M':>8} {'KS (sampler)':>13} {'KS (legacy)':>12} {'avg draws':>9}")
    worst_port, worst_legacy = 0.0, 0.0
    for M in Ms:
        cdf = true_cdf_fn(M)
        s_port, nd = hseta_rs_small_m(np.full(200_000, M), rng=rng)
        s_leg, _ = hseta_rs_small_m(
            np.full(200_000, M), rng=rng, independent_accept_uniform=False
        )
        ks_p, ks_m = ks_statistic(s_port, cdf), ks_statistic(s_leg, cdf)
        worst_port, worst_legacy = max(worst_port, ks_p), max(worst_legacy, ks_m)
        print(f"  {M:>8.0e} {ks_p:>10.4f} {ks_m:>12.4f} {nd.mean():>9.2f}")
    assert worst_port < 2 * crit, worst_port
    print(f"  {PASS} port: max KS {worst_port:.4f} < {2 * crit:.4f}")
    print(f"  NOTE  legacy reuse-uniform variant: max KS {worst_legacy:.4f} "
          f"({'inexact' if worst_legacy > 2 * crit else 'close'} -- reuse of the "
          "component-selection uniform biases acceptance; see README)")


# ---------------------------------------------------------------------------
# 3. Woodbury identity, SVD logdet, and the diag(chol(Gram)) issue
# ---------------------------------------------------------------------------
def check_woodbury():
    header("3. Woodbury identity and log determinant (s_delta < n)")
    rng = np.random.default_rng(3)
    n, s = 80, 20
    X1 = rng.standard_normal((n, s))
    lam1 = rng.uniform(0.1, 3.0, s)
    xi = 10.0**rng.uniform(-1, 2)
    y = rng.standard_normal(n)

    eta1 = 1.0 / lam1**2
    A = np.diag(xi * eta1) + X1.T @ X1
    x_wb = y - X1 @ np.linalg.solve(A, X1.T @ y)
    M = np.eye(n) + (X1 * (lam1**2)[None, :]) @ X1.T / xi
    x_direct = np.linalg.solve(M, y)
    err_x = np.max(np.abs(x_wb - x_direct)) / np.max(np.abs(x_direct))
    assert err_x < 1e-10, err_x
    print(f"  {PASS} M^(-1)y via Woodbury vs direct solve: rel err {err_x:.2e}")

    svals = np.linalg.svd(X1 * lam1[None, :], compute_uv=False)
    ld_svd = np.sum(np.log1p(svals**2 / xi))
    ld_true = np.linalg.slogdet(M)[1]
    assert abs(ld_svd - ld_true) < 1e-8 * abs(ld_true)
    print(f"  {PASS} logdet via SVD vs slogdet: {ld_svd:.6f} vs {ld_true:.6f}")

    # Sylvester identity now used by the sampler's woodbury route:
    # logdet M = log|D_S| - s log xi + 2 sum log diag chol(A), with A the
    # same matrix the solves factor -- no eigendecomposition needed
    A_syl = X1.T @ X1
    A_syl[np.diag_indices_from(A_syl)] += xi / lam1**2
    ld_syl = (2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(A_syl))))
              + np.sum(np.log(lam1**2)) - s * np.log(xi))
    assert abs(ld_syl - ld_true) < 1e-8 * abs(ld_true), (ld_syl, ld_true)
    print(f"  {PASS} logdet via Sylvester |D_S||A| == slogdet: "
          f"{ld_syl:.6f} (err {abs(ld_syl - ld_true):.1e})")

    # direct-route logdet through cho_factor, as used in the sampler's MH
    # step (regression-guards the extra-sqrt bug that once halved this
    # determinant)
    from scipy.linalg import cho_factor

    f = cho_factor(M, lower=True, check_finite=False)
    ld_chol = 2.0 * np.sum(np.log(np.diag(f[0])))
    assert abs(ld_chol - ld_true) < 1e-8 * abs(ld_true), (ld_chol, ld_true)
    print(f"  {PASS} logdet via cho_factor diag == slogdet: {ld_chol:.6f}")

    dchol = np.diag(np.linalg.cholesky((X1 * lam1[None, :]).T @ (X1 * lam1[None, :])))
    ld_chol = np.sum(np.log1p(dchol**2 / xi))
    print(f"  NOTE  sum log(1 + diag(chol(Gram))^2/xi) = {ld_chol:.6f} "
          f"(the legacy Woodbury-route approximation); true logdet = "
          f"{ld_true:.6f}. diag(chol) is not the vector of singular values; "
          "the gap grows with column correlation, so the sampler follows the "
          "paper (SVD).")
    assert abs(ld_chol - ld_true) > 1e-3  # demonstrate the two differ here


# ---------------------------------------------------------------------------
# 4. end-to-end comparison on a small sparse problem
# ---------------------------------------------------------------------------
def ess(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    acf = np.empty(51)
    denom = x @ x
    for k in range(51):
        acf[k] = x[: x.size - k] @ x[k:] / denom
    # initial positive sequence estimator (Geyer)
    tau, k = 1.0, 1
    while k + 1 < 51 and acf[k] + acf[k + 1] > 0:
        tau += 2 * (acf[k] + acf[k + 1])
        k += 2
    return x.size / tau


def check_end_to_end():
    header("4. end-to-end: exact vs approximate (n=150, p=400)")
    rng_data = np.random.default_rng(7)
    n, p = 150, 400
    beta_true = np.zeros(p)
    beta_true[:23] = 2.0 ** (-np.arange(-2.0, 3.5 + 1e-9, 0.25))
    X = rng_data.standard_normal((n, p))
    y = X @ beta_true + np.sqrt(2.0) * rng_data.standard_normal(n)

    common = dict(
        burnin=500, mcmc=4000, thin=1, n_keep=100,
        beta_true=beta_true, is_sim=True, verbose=0,
    )
    r_exact = horseshoe(y, X, approx_xlx=False,
                        rng=np.random.default_rng(5171), **common)
    r_approx = horseshoe(y, X, approx_xlx=True, delta=1e-4,
                         rng=np.random.default_rng(5171), **common)

    bt = beta_true[:100]
    print(f"  {'':16}{'exact':>12}{'approx':>12}")
    print(f"  {'E[beta]~truth r':16}"
          f"{np.corrcoef(r_exact.beta_mean, bt)[0,1]:>12.3f}"
          f"{np.corrcoef(r_approx.beta_mean, bt)[0,1]:>12.3f}")
    print(f"  {'E[sigma^2]':16}{r_exact.sigma_sq_mean:>12.3f}"
          f"{r_approx.sigma_sq_mean:>12.3f}")
    print(f"  {'coverage %':16}{100*r_exact.coverage:>12.1f}"
          f"{100*r_approx.coverage:>12.1f}")
    print(f"  {'mse':16}{r_exact.mse:>12.4f}{r_approx.mse:>12.4f}")
    print(f"  {'ESS log(xi)':16}{ess(r_exact.xi_samples):>12.0f}"
          f"{ess(r_approx.xi_samples):>12.0f}")
    print(f"  {'seconds':16}{r_exact.elapsed:>12.1f}{r_approx.elapsed:>12.1f}")
    print(f"  {'mean |S|':16}{'-':>12}{r_approx.mean_active_set:>12.1f}")

    c = np.corrcoef(r_exact.beta_mean, r_approx.beta_mean)[0, 1]
    rel = (np.abs(r_exact.beta_mean - r_approx.beta_mean).max()
           / np.abs(r_exact.beta_mean).max())
    assert np.isfinite(r_approx.beta_mean).all()
    assert np.corrcoef(r_approx.beta_mean, bt)[0, 1] > 0.8
    assert np.isfinite(r_approx.sigma_sq_mean) and r_approx.sigma_sq_mean > 0
    assert 0.5 < r_approx.mean_active_set < p  # thresholding active but sane
    assert np.corrcoef(r_exact.beta_mean, bt)[0, 1] > 0.8  # untruncated exact healthy
    assert 0.2 < r_exact.sigma_sq_mean < 6.0  # guards the runaway regime
    c_ea = np.corrcoef(r_exact.beta_mean, r_approx.beta_mean)[0, 1]
    assert c_ea > 0.95  # the two routes must agree on the same posterior
    print(f"  {PASS} both chains recover the signal and agree: "
          f"exact corr {np.corrcoef(r_exact.beta_mean, bt)[0,1]:.3f}, "
          f"approx corr {np.corrcoef(r_approx.beta_mean, bt)[0,1]:.3f}, "
          f"mutual corr {c_ea:.3f}")
    print(f"  {PASS} untruncated exact chain healthy "
          f"(E[sigma^2] = {r_exact.sigma_sq_mean:.2f})")


# ---------------------------------------------------------------------------
# 5. approximate beta draw vs the paper's transition density (Appendix B,
#    eq. (30)), incl. the errata: eq. (9)'s dropped PSD term and eq. (30)'s
#    2*Gamma W'Md^-1 W Gamma_d in place of the (asymmetric) sum
# ---------------------------------------------------------------------------
def check_beta_conditional():
    header("5. approximate beta | eta, xi draw vs Appendix B eq. (30)")
    rng = np.random.default_rng(4)
    n, p = 40, 100
    X = rng.standard_normal((n, p))
    z = rng.standard_normal(n)
    xi = 0.2
    gd = 10.0 ** rng.uniform(-3, 3, p) / xi  # diag(Gamma) = xi^-1 lambda^2
    delta = 0.5
    S = np.flatnonzero(gd > delta)
    gd_d = np.where(np.isin(np.arange(p), S), gd, 0.0)
    Gam, Gam_d = np.diag(gd), np.diag(gd_d)
    M_d = np.eye(n) + X @ Gam_d @ X.T
    M = np.eye(n) + X @ Gam @ X.T
    Md_inv = np.linalg.inv(M_d)

    A1 = Gam @ X.T @ Md_inv @ X @ Gam_d          # Gamma W' Md^-1 W Gamma_d
    A2 = Gam_d @ X.T @ Md_inv @ X @ Gam          # its transpose
    mid = Gam_d @ X.T @ Md_inv @ M @ Md_inv @ X @ Gam_d
    sig30_correct = Gam - (A1 + A2) + mid        # eq. (30) with A + A'
    sig30_printed = Gam - (2 * A1 - mid)         # eq. (30) as printed (2A)

    # Monte Carlo of the Section 2.2 beta draw (sigma^2 = 1); the mean term
    # mu_d = Gamma_d W' Md^-1 z is included automatically through (z - v)
    N = 400_000
    u = rng.normal(size=(N, p)) * np.sqrt(gd)    # u ~ N(0, Gamma), full D
    f = rng.normal(size=(N, n))
    v = u @ X.T + f                              # v = Wu + f, full W
    vs = np.linalg.solve(M_d, (z - v).T).T       # Md^-1 (z - v)
    beta = u + (vs @ X) * gd_d                   # u + Gamma_d W' Md^-1 (z - v)

    rel = lambda A_, B_: np.abs(A_ - B_).max() / np.abs(B_).max()
    sig_mc = np.cov(beta.T)
    e30 = rel(sig_mc, sig30_correct)
    print(f"  |S| = {S.size}/{p} (delta = {delta})")
    print(f"  {'MC vs corrected eq(30):':36s}{e30:.2e}")
    assert e30 < 0.05, e30
    print(f"  {PASS} the port's beta update (u with full D, v with full W,\n"
          "        correction on S only) matches the corrected eq. (30)")

    lab_sym = "symmetry of A = Gamma W\u2032Md\u207b\u00b9W Gamma_d"
    print(f"  {lab_sym:36s}max|A-A'| = {np.abs(A1 - A1.T).max():.2e} "
          f"(max|A| = {np.abs(A1).max():.2e})")
    lab_asy = "printed eq(30) asymmetry (2A vs A+A\u2032)"
    print(f"  {lab_asy:36s}max|Sig-Sig'| = "
          f"{np.abs(sig30_printed - sig30_printed.T).max():.2e}"
          "  -> erratum, S x S^c blocks only")

    Gs = Gam[np.ix_(S, S)]
    sig9SS = np.linalg.inv(X[:, S].T @ X[:, S] + np.linalg.inv(Gs))
    gap = rel(sig30_correct[np.ix_(S, S)], sig9SS)
    print(f"  {'eq(30)(S,S) vs eq(9)(S,S)':36s}{gap:.2e} "
          "(eq. (9) drops an O(delta) PSD term)")
    sig_exact = Gam - Gam @ X.T @ np.linalg.solve(M, X) @ Gam
    print(f"  {'MC vs exact Sigma':36s}{rel(sig_mc, sig_exact):.2e} "
          "(approximation gap, shrinks with delta)")


# ---------------------------------------------------------------------------
# 6. prior-truncation variant (eta_lb / xi_bounds): sampler exactness
#    and an end-to-end truncated run
# ---------------------------------------------------------------------------
def make_test_data(n, p, seed=7):
    rng = np.random.default_rng(seed)
    beta_true = np.zeros(p)
    beta_true[:23] = 2.0 ** (-np.arange(-2.0, 3.5 + 1e-9, 0.25))
    X = rng.standard_normal((n, p))
    y = X @ beta_true + np.sqrt(2.0) * rng.standard_normal(n)
    return y, X, beta_true


def truncated_ks(m, lb, draws):
    # truth CDF on a grid tailored to (m, lb): fine near lb where the mass
    # sits when m*lb is large, logarithmic out to the tail; computed in log
    # space so that e^{-m t} does not underflow for large m*lb
    width = min(20.0, 50.0 / m)
    grid = np.unique(np.concatenate([
        np.linspace(lb, lb + width, 100000),
        np.logspace(np.log10(lb + width),
                    np.log10(max(2e6, lb + 60.0 / m)), 100000),
    ]))
    logw = -m * grid - np.log1p(grid)
    w = np.exp(logw - logw[0])
    cdf = np.concatenate([[0.0], np.cumsum(w[:-1] * np.diff(grid))])
    cdf /= cdf[-1]
    s = np.sort(draws)
    F = np.interp(s, grid, cdf)
    return float(np.max(np.abs(np.arange(1, s.size + 1) / s.size - F)))


def check_truncation():
    header("6. truncation: eta_lb sampler exactness + end-to-end run")
    rng = np.random.default_rng(3)
    print("  truncated eta sampler, target ~ e^{-m t}/(1+t) on (lb, inf):")
    for m, lb in ((0.7, 0.5), (0.1, 2.0), (3.0, 0.3),
                  (1e-4, 5.0), (50.0, 0.5), (1e4, 2.0)):
        s = sample_eta(np.full(200_000, m), rng=np.random.default_rng(0), lb=lb)
        assert s.min() > lb, (m, lb, s.min())
        ks = truncated_ks(m, lb, s)
        print(f"    m={m:>8g} lb={lb:>4.1f}: KS = {ks:.4f}")
        assert ks < 0.006, (m, lb, ks)
    print(f"  {PASS} truncated sampler exact for all (m, lb), incl. the "
          "m*lb >> 1 cases that break naive rejection")

    # xi_bounds: proposals outside [lo, hi] must never be accepted
    y, X, beta_true = make_test_data(150, 400)
    r = horseshoe(y, X, burnin=200, mcmc=800, thin=1, n_keep=10,
                  beta_true=beta_true, is_sim=True, verbose=0,
                  approx_xlx=False,
                  xi_bounds=(1.0, 100.0),
                  rng=np.random.default_rng(2))
    assert r.xi_samples.min() >= 1.0 and r.xi_samples.max() <= 100.0
    print(f"  {PASS} xi_bounds respected: xi in "
          f"[{r.xi_samples.min():.2f}, {r.xi_samples.max():.2f}]")

    # eta_lb end-to-end: the truncated exact chain stays healthy; capping
    # the local scales biases E[sigma^2] downward relative to untruncated
    r = horseshoe(y, X, burnin=500, mcmc=4000, thin=1, n_keep=100,
                  beta_true=beta_true, is_sim=True, verbose=0,
                  approx_xlx=False, eta_lb=0.5,
                  rng=np.random.default_rng(5171))
    corr = np.corrcoef(r.beta_mean, beta_true[:100])[0, 1]
    print(f"  exact chain with eta_lb=0.5: E[sigma^2] = {r.sigma_sq_mean:.3f}, "
          f"corr(beta, truth) = {corr:.3f}, coverage = {100 * r.coverage:.0f}%")
    assert corr > 0.9 and r.coverage > 0.8
    assert 0.2 < r.sigma_sq_mean < 6.0
    assert r.config["eta_lb"] == 0.5 and r.config["approx_xlx"] is False
    print(f"  {PASS} eta_lb=0.5 end-to-end healthy "
          f"(corr {corr:.3f}; E[sigma^2] biased low vs untruncated, as "
          "capping lambda_j must); config recorded")


# ---------------------------------------------------------------------------
# 7. real-data mode (is_sim=False): keep_id selection, streaming beta_hat,
#    test-set losses, and the exact-mode id1 path
# ---------------------------------------------------------------------------
def check_real_data_path():
    header("7. real-data mode: keep_id, streaming beta_hat, test losses")
    rng = np.random.default_rng(21)
    n, p, n_test = 120, 300, 60
    beta_true = np.zeros(p)
    beta_true[:10] = 2.0 * 2.0 ** (-0.25 * np.arange(10))
    X = rng.standard_normal((n + n_test, p))
    y = X @ beta_true + 0.5 * rng.standard_normal(n + n_test)
    Xtr, Xte = X[:n], X[n:]
    ytr, yte = y[:n], y[n:]

    # both routes run untruncated (the default; both are healthy at this
    # size since the direct-route fix); the eta_lb variant's end-to-end
    # behavior is covered by check 6
    for approx, extra in ((True, {}), (False, {})):
        tag = "approx" if approx else "exact"
        r = horseshoe(ytr, Xtr, burnin=600, mcmc=800, thin=2, n_keep=40,
                      is_sim=False,
                      y_test=yte, X_test=Xte,
                      approx_xlx=approx, delta=1e-3,
                      rng=np.random.default_rng(1), verbose=0, **extra)
        assert r.beta_samples.shape == (40, 400)
        assert r.coverage is None and r.mse is None and r.pred_loss is None
        assert np.isfinite(r.l1_loss).all() and np.isfinite(r.per_expl).all()
        assert np.isfinite(r.beta_hat).all()
        kid = r.keep_id
        assert kid.size == 40 and np.unique(kid).size == 40
        # keep_id is chosen at i == burnin from |beta_hat| *at that time*;
        # the final streaming mean re-ranks the null tail, so require the
        # unambiguous part: all true signals in the first half, plus a loose
        # overlap with the final ranking
        top20 = set(np.argsort(np.abs(r.beta_hat))[::-1][:20].tolist())
        overlap = len(set(kid[:20].tolist()) & top20)
        corr = np.corrcoef(r.beta_hat, beta_true)[0, 1]
        print(f"  [{tag}] corr(beta_hat, truth) = {corr:.3f}, "
              f"signals in first half = "
              f"{len(set(range(10)) & set(kid[:20].tolist()))}/10, "
              f"final-ranking overlap = {overlap}/20, "
              f"per_expl range [{r.per_expl.min():.2f}, {r.per_expl.max():.2f}]")
        assert set(range(10)) <= set(kid[:20].tolist())
        assert overlap >= 10
        assert corr > 0.4
        assert r.config["is_sim"] is False and r.config["n"] == n
    print(f"  {PASS} real-data mode works in both approx and exact routes "
          "(keep_id selection, streaming mean, test-set losses, config)")


# ---------------------------------------------------------------------------
# 8. pipeline conditionals: the direct route's M construction and the beta
#    draw through the actual code (regression test for the dsyrk
#    weight-squaring bug: dsyrk(X*w) is X diag(w^2) X', not X diag(w) X')
# ---------------------------------------------------------------------------
def check_pipeline_conditionals():
    header("8. pipeline conditionals (direct-route M, beta draw, MH decision)")
    rng = np.random.default_rng(42)
    n, p = 40, 90
    X = rng.standard_normal((n, p))
    lam = rng.uniform(0.1, 1.5, p)
    y = rng.standard_normal(n)
    xi, sigma_sq = 0.5, 1.3
    a0 = b0 = 1.0
    gd = lam**2 / xi
    Gam = np.diag(gd)
    M = np.eye(n) + X @ Gam @ X.T
    Minv = np.linalg.inv(M)
    mu_exact = Gam @ X.T @ Minv @ y                          # eq. (28)
    Sig_exact = sigma_sq * (Gam - Gam @ X.T @ Minv @ X @ Gam)
    rel = lambda A_, B_: np.abs(A_ - B_).max() / np.abs(B_).max()

    def run_update(approx, delta, prop=None, u=1e-300):
        # prop = current xi (default) accepts for any uniform and keeps the
        # active set identical on both sides of the MH ratio
        return update_xi(y, X, lam, xi, xi if prop is None else prop, u,
                         a0=a0, b0=b0, approx_xlx=approx, delta=delta)

    def draw(upd, mode, u_, resid):
        return sample_beta(u_, resid, X=X, lam=lam, xi=xi,
                           sigma_sq=sigma_sq, mode=mode, factor=upd.factor,
                           matrix=upd.matrix, X1=upd.X1, lam1=upd.lam1,
                           id1=upd.id1, A_factor=upd.factor,
                           active_mask=upd.active_mask)

    # (a) deterministic mean (u = 0, v = 0): the draw must equal the
    #     analytic mean exactly, in all three routes
    delta_wb = np.sort(gd)[::-1][10]             # |S| = 10 < n/2 -> Woodbury
    upd = run_update(False, 1e-4)
    routes = [("exact", upd, "exact", mu_exact, Sig_exact)]
    upd = run_update(True, 0.0)                  # all active -> approx_direct
    assert not upd.woodbury
    routes.append(("approx_direct", upd, "approx_direct", mu_exact, Sig_exact))
    upd = run_update(True, delta_wb)
    assert upd.woodbury and upd.active_size == 10
    S = gd > delta_wb
    gd_d = np.where(S, gd, 0.0)
    Gam_d = np.diag(gd_d)
    M_d = np.eye(n) + (X * gd_d) @ X.T
    Md_inv = np.linalg.inv(M_d)
    A1 = Gam @ X.T @ Md_inv @ X @ Gam_d           # eq. (30), corrected
    mid = Gam_d @ X.T @ Md_inv @ M @ Md_inv @ X @ Gam_d
    mu_d = Gam_d @ X.T @ Md_inv @ y
    Sig_d = sigma_sq * (Gam - (A1 + A1.T) + mid)
    routes.append(("woodbury", upd, "woodbury", mu_d, Sig_d))
    for tag, upd, mode, mu_t, _ in routes:
        b_mean = draw(upd, mode, np.zeros(p), y / np.sqrt(sigma_sq))
        err = (np.linalg.norm(b_mean - mu_t) / np.linalg.norm(mu_t))
        assert err < 1e-10, (tag, err)
        print(f"  {PASS} {tag:13s} deterministic mean rel err {err:.1e}")
    err = rel(routes[2][1].x_curr, np.linalg.solve(M_d, y))
    assert err < 1e-10, err
    print(f"  {PASS} woodbury x_curr vs direct solve of M_delta: {err:.1e}")

    # (b) MH decision bracketing on the direct route: a uniform with
    #     log(u) just below / above the hand-computed log ratio must
    #     accept / reject (catches logdet and ssr bugs inside update_xi;
    #     when log ratio >= 0 acceptance is certain and only the accept
    #     probe applies)
    lr_true = log_marg(y, np.linalg.solve(M, y), xi, np.linalg.slogdet(M)[1],
                       n, a0, b0)
    for prop in (0.31, 1.7, 2.3):
        Mp = np.eye(n) + (X * lam**2) @ X.T / prop
        log_acc = (log_marg(y, np.linalg.solve(Mp, y), prop,
                            np.linalg.slogdet(Mp)[1], n, a0, b0)
                   - lr_true + np.log(prop) - np.log(xi))
        probes = []
        for off, expect in ((-1e-6, True), (1e-6, False)):
            log_u = log_acc + off
            if log_u >= 0.0:
                if not expect:
                    continue
                log_u = -1e-9
            probes.append((np.exp(log_u), expect))
        for u, expect in probes:
            got = run_update(False, 1e-4, prop=prop, u=u).moved
            assert got == expect, (prop, log_acc, expect, got)
        print(f"  {PASS} MH decision at prop xi = {prop:.2f}: hand log "
              f"ratio {log_acc:+.3f} bracketed ({len(probes)}/2 probes apply)")

    # (b2) MH decision bracketing on the WOODBURY route.  The active set
    #      depends on the proposal through the E6 union rule, so the hand
    #      side must rebuild both thresholded matrices per proposal; the
    #      hand log ratio uses slogdet + a direct solve
    rngb = np.random.default_rng(42)
    n2, p2 = 200, 400
    X2 = rngb.standard_normal((n2, p2))
    lam2 = rngb.uniform(0.05, 2.0, p2)
    y2 = rngb.standard_normal(n2)
    xi2 = 0.5
    gd2 = lam2**2 / xi2
    delta2 = float(np.sort(gd2)[::-1][50])  # |S| = 50 at prop == xi
    for prop in (0.21, 0.5, 1.4, 3.2):
        S2 = (lam2**2) * max(1.0 / xi2, 1.0 / prop) > delta2
        XS = np.where(S2, lam2**2, 0.0)
        M_c = np.eye(n2) + (X2 * XS) @ X2.T / xi2
        M_p = np.eye(n2) + (X2 * XS) @ X2.T / prop
        log_acc = (log_marg(y2, np.linalg.solve(M_p, y2), prop,
                            np.linalg.slogdet(M_p)[1], n2, 1.0, 1.0)
                   - log_marg(y2, np.linalg.solve(M_c, y2), xi2,
                              np.linalg.slogdet(M_c)[1], n2, 1.0, 1.0)
                   + np.log(prop) - np.log(xi2))
        probes = []
        for off, expect in ((-1e-6, True), (1e-6, False)):
            log_u = log_acc + off
            if log_u >= 0.0:
                if not expect:
                    continue
                log_u = -1e-9
            probes.append((np.exp(log_u), expect))
        for u, expect in probes:
            upd = update_xi(y2, X2, lam2, xi2, prop, u, a0=1.0, b0=1.0,
                            approx_xlx=True, delta=delta2)
            assert upd.woodbury, (prop, upd.active_size)
            assert upd.moved == expect, (prop, log_acc, expect, upd.moved)
        print(f"  {PASS} woodbury MH decision at prop xi = {prop:.2f} "
              f"(|S| = {int(S2.sum())}): hand log ratio {log_acc:+.2f} "
              f"bracketed ({len(probes)}/2 probes apply)")

    # (c) Monte-Carlo covariance through the pipeline (fresh u, v each
    #     draw, exactly as the main loop calls them) vs the analytic
    #     eq. (28) (exact route) and eq. (30) (woodbury route)
    N = 40_000
    tau = xi**-0.5
    for tag, upd, mode, mu_t, Sig_t in (routes[0], routes[2]):
        rng2 = np.random.default_rng(7)
        betas = np.empty((N, p))
        for k in range(N):
            u_ = rng2.standard_normal(p) * (tau * lam)
            v_ = X @ u_ + rng2.standard_normal(n)
            betas[k] = draw(upd, mode, u_, y / np.sqrt(sigma_sq) - v_)
        err = rel(np.cov(betas.T), Sig_t)
        assert err < 0.05, (tag, err)
        print(f"  {PASS} {tag:13s} MC covariance vs analytic: rel err "
              f"{err:.1e} (N={N}, MC noise ~{1 / np.sqrt(N):.1e})")


# ---------------------------------------------------------------------------
# 9. edge cases: input validation and the all-null-truth relative losses
# ---------------------------------------------------------------------------
def check_edges():
    header("9. edge cases: validation, all-null-truth losses")
    X = np.random.default_rng(1).standard_normal((40, 60))
    y = np.random.default_rng(0).standard_normal(40)
    beta0 = np.zeros(60)
    base = dict(burnin=10, mcmc=100, beta_true=beta0, is_sim=True, verbose=0)
    for extra, msg in (
        (dict(mcmc=0), "mcmc = 0"),
        (dict(mcmc=3, thin=5), "thin > mcmc"),
        (dict(burnin=-1), "burnin < 0"),
        (dict(thin=0), "thin = 0"),
    ):
        try:
            horseshoe(y, X, **{**base, **extra})
        except ValueError:
            print(f"  {PASS} ValueError on {msg}")
        else:
            raise AssertionError(f"{msg} was not rejected")

    # all-null truth: the relative losses must stay finite (zero-norm
    # denominators fall back to 1, so the losses read 1 - |error|)
    r = horseshoe(y, X, burnin=50, mcmc=100, beta_true=beta0,
                  is_sim=True, verbose=0)
    assert np.isfinite(r.per_expl).all() and np.isfinite(r.l1_loss).all()
    assert np.isfinite(r.beta_mean).all() and np.isfinite(r.mse)
    print(f"  {PASS} all-null truth: losses finite "
          f"(per_expl in [{r.per_expl.min():.2f}, {r.per_expl.max():.2f}], "
          f"l1 in [{r.l1_loss.min():.2f}, {r.l1_loss.max():.2f}])")


if __name__ == "__main__":
    np.seterr(over="ignore", under="ignore")
    check_envelope()
    check_eta_sampler()
    check_woodbury()
    check_end_to_end()
    check_beta_conditional()
    check_truncation()
    check_real_data_path()
    check_pipeline_conditionals()
    check_edges()
    print("\nall checks passed.")

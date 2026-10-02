"""Scalable (approximate) MCMC for the horseshoe linear model.

Implementation of the blocked Metropolis-within-Gibbs sampler of

    Johndrow, Orenstein & Bhattacharya (2020),
    "Scalable Approximate MCMC Algorithms for the Horseshoe Prior",
    Journal of Machine Learning Research.

Model (the paper's equation (2); xi = tau^-2 and eta_j = lambda_j^-2 are the
global and local *precisions*):

    y = X beta + eps,                     eps ~ N(0, sigma^2 I_n)
    beta_j | sigma^2, xi, eta_j ~ N(0, sigma^2 xi^-1 eta_j^-1)
    eta_j^-1/2 ~ half-Cauchy(0, 1),       xi^-1/2 ~ half-Cauchy(0, 1)
    sigma^2 ~ InvGamma(a0/2, b0/2)        (a0 = b0 = 1 in the paper's sims)

One Gibbs sweep of the exact algorithm (the paper's equation (4)):

    1. eta_j independently from p(eta_j | beta, xi, sigma^2) with the exact
       rejection sampler of Appendix S1 (eta_sampler.py);
    2. random-walk Metropolis on log(xi) targeting p(xi | eta), which blocks
       (beta, sigma^2, xi) together -- the key difference from Polson et al.
       (2014) and Bhattacharya et al. (2016), and the source of the improved
       mixing of the global parameter;
    3. sigma^2 ~ InvGamma((n+a0)/2, (ssr+b0)/2) with ssr = y' M_xi^-1 y;
    4. beta | ... by the surrogate variable trick of Bhattacharya et al.
       (2016): u ~ N(0, xi^-1 D), v = Xu + f, v* = M_xi^-1 (y/sigma - v),
       beta = sigma (u + xi^-1 D X' v*),

where D = diag(eta^-1) = diag(lambda^2) and M_xi = I_n + xi^-1 X D X'.

Approximate algorithm (approx_xlx=True, Section 2.2 of the paper): D is
hard-thresholded to D_delta with active set

    S = { j : max(xi^-1, xi*^-1) eta_j^-1 > delta }   (i.e. xi_min^-1 eta_j^-1 > delta)

recomputed every iteration (xi* is the proposal; the set must cover the
coordinates active under xi OR xi*, since the same D_delta serves both
terms of the MH ratio).  When |S| is below a flop-count crossover (|S|
<~ 3500 at the paper scale), no n x n matrix is ever formed: the log
determinant and the solves both ride on the cho_factor of
A = xi D_S^-1 + X_S'X_S -- logdet M = log|D_S| - |S| log(xi) + log|A| by
the Sylvester identity, so the eigendecomposition the paper prescribes
is never needed -- and solves use M_xi^-1 = I - X_S A^-1 X_S'.
Thresholded coordinates are still drawn from a (near-exact) Gaussian, so
the active set can change from iteration to iteration.

Deviation from the original implementation (documented in README.md):
the eta acceptance uniform is drawn independently of the envelope-component
uniform, exactly as Appendix S1 specifies (the legacy code reused one
uniform, which is inexact); the Woodbury-branch log determinant rides on
the cho_factor of A = xi D_S^-1 + X_S'X_S through the Sylvester identity
(see xi_sampler.py), so the eigendecomposition the paper prescribes is
never computed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from beta_sampler import sample_beta
from sigma2_sampler import update_sigma2
from xi_sampler import update_xi
from eta_sampler import (
    DEFAULT_A,
    DEFAULT_B,
    hseta_rs_small_m,
    sample_eta,
)

__all__ = ["HorseshoeResult", "horseshoe", "plot_diagnostics"]


@dataclass
class HorseshoeResult:
    """Posterior summaries and stored draws (kept coordinates only)."""

    beta_mean: np.ndarray          # posterior mean of beta[keep_id]
    beta_median: np.ndarray
    lambda_mean: np.ndarray        # posterior mean of lambda[keep_id]
    sigma_sq_mean: float
    beta_hat: np.ndarray           # p-vector: mean of beta over saved draws
    beta_samples: np.ndarray       # (n_keep, n_saved)
    lambda_samples: np.ndarray
    eta_samples: np.ndarray
    tau_samples: np.ndarray        # (n_saved,)
    xi_samples: np.ndarray
    sigma_sq_samples: np.ndarray
    ci_lo: np.ndarray | None       # 2.5% pointwise credible limits
    ci_hi: np.ndarray | None
    coverage: float | None         # simulation only
    mse: float | None              # simulation only
    se: np.ndarray | None          # posterior std of beta[keep_id]
    keep_id: np.ndarray            # indices of stored coordinates (0-based)
    l1_loss: np.ndarray            # per-draw L1 loss diagnostic
    per_expl: np.ndarray           # per-draw "percent explained" diagnostic
    pred_loss: np.ndarray | None   # simulation only
    acc_rate_xi: float
    mean_active_set: float         # average |S| (approx_xlx only; nan otherwise)
    elapsed: float
    config: dict = field(default_factory=dict)   # run parameters (reproducibility)


def horseshoe(
    y,
    X,
    *,
    burnin=1000,
    mcmc=20000,
    thin=1,
    mh_scale_ub=0.8,
    mh_scale_lb=0.8,
    phasein=1,
    a0=1.0,
    b0=1.0,
    approx_xlx=True,
    delta=1e-4,
    n_keep=100,
    beta_true=None,
    is_sim=True,
    y_test=None,
    X_test=None,
    rng=None,
    a_eta=DEFAULT_A,
    b_eta=DEFAULT_B,
    legacy_eta_sampler=False,
    lam_init=None,
    tau_init=None,
    sigma_sq_init=None,
    eta_lb=0.0,
    xi_bounds=None,
    save_samples=False,
    save_dir="Outputs",
    simtype="",
    verbose=1000,
):
    """Run the (approximate) horseshoe Gibbs sampler.  See module docstring.

    Returns a HorseshoeResult.  `legacy_eta_sampler=True` switches the
    local scale sampler back to the legacy (inexact) behavior (the
    component-selection uniform is reused for the acceptance test);
    `delta` may be an
    array of length burnin+mcmc giving a per-iteration threshold schedule.

    Prior-truncation variant (paper Appendix B / S2 convergence analysis;
    optional): `eta_lb = b > 0` lower-truncates eta_j to (b, inf) (upper-caps
    the local scales lambda_j = eta_j^{-1/2} at b^{-1/2}), sampled exactly
    by rejecting draws <= b with the same envelope; `xi_bounds = (lo, hi)`
    restricts xi to [lo, hi] (equivalently tau = xi^{-1/2} in
    [hi^{-1/2}, lo^{-1/2}]; corresponds to [b_xi^{-2}, a_xi^{-2}]), by
    rejecting xi proposals outside the bounds, which is MH on the
    correspondingly truncated p(xi | eta).  Defaults reproduce the
    untruncated main-text model (b = 0, bounds = (0, inf)).
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if burnin < 0 or mcmc < 1 or thin < 1 or thin > mcmc:
        raise ValueError(
            "need burnin >= 0, mcmc >= 1 and 1 <= thin <= mcmc "
            "(otherwise no draws are stored and every summary is nan)"
        )
    N = burnin + mcmc
    n_saved = mcmc // thin
    half_burn = burnin // 2

    if is_sim and beta_true is None:
        raise ValueError("beta_true is required when is_sim=True")
    beta_true = np.asarray(beta_true, dtype=float).reshape(-1) if is_sim else None
    if not is_sim and (y_test is None or X_test is None):
        raise ValueError("y_test and X_test are required when is_sim=False")
    # denominators of the relative-loss diagnostics; an all-null truth
    # (zero norm) falls back to 1.0, so the losses read "1 - |error|"
    # instead of nan
    if is_sim:
        bt_norm = float(np.sqrt(beta_true @ beta_true))
        bt_l1 = float(np.abs(beta_true).sum())
        bt_norm = bt_norm if bt_norm > 0.0 else 1.0
        bt_l1 = bt_l1 if bt_l1 > 0.0 else 1.0

    delta_arr = np.asarray(delta, dtype=float)
    if delta_arr.ndim == 0:
        delta_arr = np.full(N, float(delta_arr))
    elif delta_arr.size != N:
        raise ValueError(f"delta must be scalar or length {N}")

    n_keep = min(n_keep, p)
    Xw = np.empty_like(X)  # reusable n x p buffer for X * colweights

    if xi_bounds is not None:
        xi_lo, xi_hi = float(xi_bounds[0]), float(xi_bounds[1])
        if not (0.0 <= xi_lo < xi_hi):
            raise ValueError("xi_bounds must be (lo, hi) with 0 <= lo < hi")
    else:
        xi_lo, xi_hi = 0.0, np.inf

    # state (lam_init/tau_init/sigma_sq_init allow starting in a chosen
    # regime, e.g. tau small and nulls' lambda small -- the sparse basin)
    beta = np.ones(p)
    lam = np.ones(p) if lam_init is None else np.asarray(lam_init, float).copy()
    xi = 1.0 if tau_init is None else float(tau_init) ** -2
    xi = min(max(xi, xi_lo), xi_hi)
    tau = 1.0 / np.sqrt(xi)
    sigma_sq = 1.0 if sigma_sq_init is None else float(sigma_sq_init)
    beta_hat = np.zeros(p)         # streaming mean (real-data mode)

    # stored draws
    beta_out = np.zeros((n_keep, n_saved))
    lam_out = np.zeros((n_keep, n_saved))
    eta_out = np.zeros((n_keep, n_saved))
    tau_out = np.zeros(n_saved)
    xi_out = np.zeros(n_saved)
    sig_out = np.zeros(n_saved)
    l1_out = np.zeros(n_saved)
    pexp_out = np.zeros(n_saved)
    ploss_out = np.zeros(n_saved) if is_sim else None

    n_prop = n_acc = 0
    active_sizes = [] if approx_xlx else None
    keep_id = np.arange(n_keep)
    beta_sum = None

    t0 = time.perf_counter()
    t_last = t0

    for i in range(1, N + 1):
        # ---------------- update xi: RW Metropolis on log(xi) ------------
        # proposal + acceptance uniform are drawn here (nothing else
        # consumes the stream in between, so drawing both up front
        # preserves the stream order exactly); the linear algebra and the
        # MH decision live in xi_sampler
        std_mh = (
            (mh_scale_ub * (phasein - i) + mh_scale_lb * i) / phasein
            if i < phasein
            else mh_scale_lb
        )
        prop_xi = np.exp(rng.normal(np.log(xi), std_mh))
        accept_u = rng.random()

        upd = update_xi(
            y, X, lam, xi, prop_xi, accept_u,
            a0=a0, b0=b0, approx_xlx=approx_xlx,
            delta=delta_arr[i - 1],
            xi_lo=xi_lo, xi_hi=xi_hi, work=Xw,
        )
        n_prop += 1
        if upd.moved:
            n_acc += 1
        xi = upd.xi
        x_curr = upd.x_curr
        f_curr = upd.factor
        M_curr = upd.matrix
        id1 = upd.id1
        which_in = upd.active_mask
        woodbury = upd.woodbury
        rxlx = upd.active_size
        if approx_xlx:
            active_sizes.append(rxlx)
        X1 = upd.X1
        lam1 = upd.lam1
        tau = 1.0 / np.sqrt(xi)

        # ---------------- update sigma^2 ---------------------------------
        # exact conjugate draw; the Gamma variate is drawn in the loop so
        # the random stream stays with the main loop (sigma2_sampler holds
        # the deterministic math)
        ssr = y @ x_curr
        sigma_sq = update_sigma2(rng.gamma(0.5 * (n + a0), 2.0 / (ssr + b0)))

        # ---------------- update beta (surrogate-variable trick) ---------
        # u/v draws live here (random stream owned by the loop); the
        # deterministic assembly of beta from them lives in beta_sampler
        u = rng.standard_normal(p) * (tau * lam)
        v = X @ u + rng.standard_normal(n)
        resid = y / np.sqrt(sigma_sq) - v
        beta = sample_beta(
            u, resid,
            X=X, lam=lam, xi=xi, sigma_sq=sigma_sq,
            mode="woodbury" if woodbury else
            ("approx_direct" if approx_xlx else "exact"),
            factor=f_curr, matrix=M_curr,
            X1=X1, lam1=lam1, id1=id1, A_factor=f_curr,
            active_mask=which_in if approx_xlx else None,
        )

        # ---------------- streaming diagnostics (real data) --------------
        if not is_sim:
            if i < half_burn:
                beta_hat = (i - 1.0) / i * beta_hat + beta / i
            elif i == half_burn:
                beta_hat = beta.copy()
            else:
                w = 1.0 / (i - half_burn)
                beta_hat = (1.0 - w) * beta_hat + w * beta

        # ---------------- update eta (exact rejection sampler) -----------
        gamma_rate = (beta**2) * xi / (2.0 * sigma_sq)
        if legacy_eta_sampler:
            eta, _ = hseta_rs_small_m(
                gamma_rate, a=a_eta, b=b_eta, rng=rng,
                independent_accept_uniform=False, lb=eta_lb,
            )
        else:
            eta = sample_eta(gamma_rate, rng=rng, a=a_eta, b=b_eta, lb=eta_lb)
        bad = eta <= 0
        if bad.any():
            eta[bad] = np.finfo(float).eps
        lam = 1.0 / np.sqrt(eta)

        if verbose and i % verbose == 0:
            dt = time.perf_counter() - t_last
            t_last = time.perf_counter()
            msg = (
                f"iter {i:6d}/{N}  {dt:.1f}s  acc(xi)={n_acc / max(n_prop, 1):.3f}"
            )
            if approx_xlx:
                msg += f"  |S|={rxlx}"
            print(msg)

        # ---------------- choose stored coordinates ----------------------
        if i == burnin or (i == 1 and burnin == 0):
            if is_sim:
                keep_id = np.arange(n_keep)
                beta_sum = np.zeros(p)
            else:
                half = n_keep // 2
                order = np.argsort(np.abs(beta_hat))[::-1]
                big_id = order[:half]
                other_id = np.setdiff1d(np.arange(p), big_id)[: n_keep - half]
                keep_id = np.concatenate([big_id, other_id])

        # ---------------- store draws (and losses) -----------------------
        # losses are only needed for stored iterations; computing them here
        # skips an O(np) matvec per non-stored iteration
        if i > burnin and (i - burnin) % thin == 0:
            j = (i - burnin) // thin - 1
            if is_sim:
                err = beta - beta_true
                pexp_out[j] = 1.0 - np.sqrt(err @ err) / bt_norm
                l1_out[j] = 1.0 - np.abs(err).sum() / bt_l1
                e = X @ err
                ploss_out[j] = (e @ e) / p
                beta_sum += beta
            else:
                # predictions from the full streaming beta_hat: the
                # per-iteration active set is not a selection statement
                # (thresholded coordinates are still sampled)
                res1 = y_test - X_test @ beta_hat
                pexp_out[j] = (
                    1.0 - np.sqrt(res1 @ res1) / np.sqrt(y_test @ y_test)
                )
                l1_out[j] = 1.0 - np.abs(res1).sum() / np.abs(y_test).sum()
            beta_out[:, j] = beta[keep_id]
            lam_out[:, j] = lam[keep_id]
            eta_out[:, j] = eta[keep_id]
            tau_out[j] = tau
            xi_out[j] = xi
            sig_out[j] = sigma_sq

    elapsed = time.perf_counter() - t0

    # -------- posterior summaries -----------------------------------------
    beta_mean = beta_out.mean(axis=1)
    beta_median = np.median(beta_out, axis=1)
    # Summaries conventionally start at post-burnin iteration 5001
    # (comparability with the published results).  The cutoff counts
    # post-burnin iterations, not saved draws -- saved draw j sits at
    # post-burnin iteration (j + 1) * thin -- so it does not drift with
    # thin (thin = 1 reproduces saved index 5000 exactly); fall back to
    # the second half of the chain when fewer draws qualify.
    j0 = -(-5001 // thin) - 1  # first saved index with iteration >= 5001
    tail = beta_out[:, j0:] if n_saved > j0 else beta_out[:, n_saved // 2 :]
    ci_lo = np.quantile(tail, 0.025, axis=1)
    ci_hi = np.quantile(tail, 0.975, axis=1)
    se = tail.std(axis=1, ddof=1)

    if is_sim:
        beta_hat_tail = tail.mean(axis=1)
        bt = beta_true[keep_id]
        coverage = float(np.mean((bt > ci_lo) & (bt < ci_hi)))
        mse = float(np.mean((bt - beta_hat_tail) ** 2))
        beta_hat_full = beta_sum / n_saved
        pred_loss_hat = float(
            ((X @ (beta_hat_full - beta_true)) @ (X @ (beta_hat_full - beta_true))) / n
        )
    else:
        coverage = mse = None
        pred_loss_hat = None
        beta_hat_full = beta_hat

    run_config = {
        "n": n, "p": p, "burnin": burnin, "mcmc": mcmc, "thin": thin,
        "a0": a0, "b0": b0, "approx_xlx": approx_xlx,
        "delta": float(delta) if np.ndim(delta) == 0 else "schedule",
        "eta_lb": float(eta_lb),
        "xi_bounds": (None if xi_bounds is None
                      else (float(xi_bounds[0]), float(xi_bounds[1]))),
        "legacy_eta_sampler": legacy_eta_sampler, "n_keep": n_keep,
        "is_sim": is_sim,
    }
    res = HorseshoeResult(
        beta_mean=beta_mean,
        beta_median=beta_median,
        lambda_mean=lam_out.mean(axis=1),
        sigma_sq_mean=float(sig_out.mean()),
        beta_hat=beta_hat_full,
        beta_samples=beta_out,
        lambda_samples=lam_out,
        eta_samples=eta_out,
        tau_samples=tau_out,
        xi_samples=xi_out,
        sigma_sq_samples=sig_out,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        coverage=coverage,
        mse=mse,
        se=se,
        keep_id=keep_id,
        l1_loss=l1_out,
        per_expl=pexp_out,
        pred_loss=ploss_out,
        acc_rate_xi=n_acc / max(n_prop, 1),
        mean_active_set=float(np.mean(active_sizes)) if active_sizes else float("nan"),
        elapsed=elapsed,
        config=run_config,
    )

    if verbose:
        if is_sim:
            print(f"coverage {100 * coverage:.1f}%")
            print(f"mse {mse:.4f}")
        print(f"{elapsed:.1f} seconds elapsed")

    if save_samples:
        import os

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"post_reg_horse_conc_{simtype}_{n}_{p}.npz",
        )
        np.savez(
            path,
            betaout=beta_out,
            lambdaout=lam_out,
            etaout=eta_out,
            tauout=tau_out,
            xiout=xi_out,
            sigmaSqout=sig_out,
            l1out=l1_out,
            pexpout=pexp_out,
            ci_hi=ci_hi,
            ci_lo=ci_lo,
            coverage=coverage if coverage is not None else np.nan,
            BetaHat=res.beta_hat,
            mse=mse if mse is not None else np.nan,
            se=se,
            BetaTrue=beta_true if beta_true is not None else np.zeros(0),
            keep_id=keep_id,
            BURNIN=burnin,
            MCMC=mcmc,
            t=elapsed,
        )
        print(f"saved samples to {path}")

    return res


def plot_diagnostics(res: HorseshoeResult, n_trace=25):
    """Rough equivalents of the paper's diagnostic figures (after the run,
    rather than live during sampling)."""
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(np.log(res.xi_samples), ".")
    ax[0, 0].set_title(r"$\log\xi$")
    ax[0, 1].plot(np.log(1.0 / res.sigma_sq_samples), ".")
    ax[0, 1].set_title(r"$\log\sigma^{-2}$")
    ax[1, 0].plot(res.beta_samples[0], ".")
    ax[1, 0].set_title(r"$\beta_1$")
    ax[1, 1].plot(res.eta_samples[0], ".")
    ax[1, 1].set_title(r"$\eta_1$")
    fig1.tight_layout()

    fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(res.l1_loss, ".")
    ax[0].set_title("L1 loss")
    ax[1].plot(res.per_expl, ".")
    ax[1].set_title("proportion explained")
    fig2.tight_layout()

    k = min(n_trace, res.beta_samples.shape[0])
    fig3, ax = plt.subplots(5, 5, figsize=(14, 12))
    for j in range(min(k, 25)):
        ax.flat[j].plot(res.beta_samples[j], ".")
        ax.flat[j].set_title(rf"$\beta_{{{j + 1}}}$")
    fig3.tight_layout()
    plt.show()
    return fig1, fig2, fig3

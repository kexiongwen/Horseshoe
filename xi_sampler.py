"""Random-walk Metropolis update of the global precision xi.

Implements step 2 of the paper's algorithm (4): a log-normal random-walk
proposal on xi, accepted with probability

    min{1, p(xi* | eta) xi* / (p(xi | eta) xi)},

where p(xi | eta) is the marginal in eq. (3) (up to constants; the
omega/2 term cancels in the ratio):

    log p(xi | eta) = -.5 logdet(M_xi) - .5 (n + a0) log(y' M_xi^-1 y + b0)
                      -.5 log(xi) - log(1 + xi),
    M_xi = I_n + xi^-1 X D X',   D = diag(lambda_j^2).

Three evaluation routes (Section 2.2), selected per iteration:

* exact route        (approx_xlx=False): the n x n matrices
  M(xi) and M(xi*) are formed via a dsyrk symmetric product and factored;
* approx direct route: same, with D masked to the active set
  S = { j : xi_min^-1 eta_j^-1 > delta } (E6-corrected direction: the set
  must cover the coordinates active under xi OR the proposal xi*, since
  the same D_delta serves both terms of the MH ratio);
* Woodbury route: no n x n matrix is formed -- the log determinant and
  the solves both ride on the cho_factor of A = xi D_S^-1 + X_S'X_S
  (Sylvester: |I + X_S Gamma_S X_S'| = |Gamma_S| |A|, so no
  eigendecomposition is needed), via M^-1 = I - X_S A^-1 X_S'.

Truncated target (Appendix B): with xi_bounds, proposals outside the
interval are declared dead before any proposal-side linear algebra (= MH
on p(xi | eta) restricted to the interval).

Like ``beta_sampler``, the module owns no RNG: the caller draws the
proposal and the acceptance uniform (nothing else consumes the stream in
between, so drawing both up front preserves the stream order exactly).
Arithmetic order matches the original inline implementation bit for bit,
except the direct route's sqrt-weight correction (dsyrk(A) returns A A',
so the product needs X * sqrt(w), not X * w) and the Woodbury route's
Sylvester logdet -- both are mathematically exact corrections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk

__all__ = ["XiUpdate", "update_xi", "log_marg"]


@dataclass
class XiUpdate:
    """Everything the main loop needs from one xi update."""

    xi: float                      # global precision after the update
    moved: bool                    # proposal accepted AND usable
    x_curr: np.ndarray             # M_xi^-1 y for the (possibly new) state
    factor: object | None          # cho_factor of the state's matrix
                                   # (M for direct/exact, A for Woodbury)
    matrix: np.ndarray | None      # dense M retained for the LU fallback
    id1: np.ndarray | None         # active-set indices (None for exact)
    active_mask: np.ndarray | None  # boolean mask of S (None for exact)
    X1: np.ndarray | None          # X[:, S] (Woodbury route only)
    lam1: np.ndarray | None        # lambda[S] (Woodbury route only)
    woodbury: bool                 # whether the Woodbury route was taken
    active_size: int               # |S| (no active set on the exact route; unused, set to n)


def log_marg(y, x_solve, xi_val, ldet, n, a0, b0):
    """log p(xi | eta) up to an additive constant (paper eq. (3)); the
    omega/2 term is dropped because it cancels in the MH ratio."""
    ssr = y @ x_solve + b0
    return (
        -0.5 * ldet
        - 0.5 * (n + a0) * np.log(ssr)
        - 0.5 * np.log(xi_val)
        - np.log1p(xi_val)
    )


def update_xi(
    y,
    X,
    lam,
    xi,
    prop_xi,
    accept_u,
    *,
    a0,
    b0,
    approx_xlx,
    delta,
    xi_lo=0.0,
    xi_hi=np.inf,
    work=None,
):
    """One RW-Metropolis sweep of the global precision.

    Parameters
    ----------
    y : (n,) response; X : (n, p) design; lam : (p,) local scales.
    xi, prop_xi : current and proposed global precision.
    accept_u : uniform in [0, 1) for the MH decision (drawn by the caller).
    a0, b0 : InvGamma shape and rate of the sigma^2 prior (b0 shifts the
        ssr in eq. (3): ssr = y' M^-1 y + b0).
    approx_xlx : whether the thresholded (approximate) algorithm is on.
    delta : the active-set threshold for this iteration (scalar).
    xi_lo, xi_hi : truncated-target bounds (Appendix B); defaults reproduce
        the untruncated main-text model.
    work : optional (n, p) buffer reused across iterations for X * sqrt(w).

    Returns an :class:`XiUpdate`.
    """
    n, p = X.shape
    dg_n = np.diag_indices(n)
    if work is None:
        work = np.empty_like(X)

    # active set S = { j : max(xi^-1, xi*^-1) eta_j^-1 > delta }
    # (i.e. xi_min^-1 eta_j^-1 > delta; erratum E6: the union direction --
    # the same D_delta serves both MH-ratio terms, so the set must cover
    # the coordinates active under xi OR the proposal xi*)
    X1 = lam1 = id1 = active_mask = None
    woodbury = False
    if approx_xlx:
        active_mask = (lam**2) * max(1.0 / xi, 1.0 / prop_xi) > delta
        id1 = np.flatnonzero(active_mask)
        active_size = int(active_mask.sum())
        # route by flop count, chol flops weighted x3 (dpotrf runs ~3x
        # slower per flop than dsyrk on this stack; empirical crossover
        # s* ~ 3500 at n=2000, p=20000): woodbury ~ 2 n s^2 + 4 s^3 (Gram
        # dsyrk + two s x s Choleskys), direct ~ 2 n^2 p + 2 n^3 (full
        # dsyrk + two n x n Choleskys, nearly independent of s)
        woodbury = (2.0 * n * active_size**2 + 4.0 * active_size**3
                    < 2.0 * n * n * p + 2.0 * n**3)
        if woodbury:
            lam1 = lam[id1]
            X1 = X[:, id1]
    else:
        active_size = n

    # a proposal outside xi_bounds is dead on arrival: skip all
    # proposal-side linear algebra up front (MH on the truncated
    # p(xi | eta) of Appendix B)
    prop_ok = xi_lo <= prop_xi <= xi_hi
    prop_valid = False  # set per branch: proposal usable if accepted

    if woodbury:
        # ---- small active set: Woodbury identities on s x s systems ----
        # solves against M = I + X_S D_S X_S'/xi use
        # A = xi diag(eta_S) + X_S'X_S (cho_factor'd once, reused);
        # the log determinant rides on the same factors by Sylvester,
        # |I + X_S Gamma_S X_S'| = |Gamma_S| |Gamma_S^-1 + X_S'X_S|, so no
        # eigendecomposition is needed (dsyrk touches the lower triangle
        # only, matching the lower=True factors).
        if id1.size:
            XX1 = dsyrk(1.0, X1, trans=1, lower=1)
            X1y = X1.T @ y
            log_det_gam = float(np.sum(np.log(lam1**2)))
            lr_prop = -np.inf
            if prop_ok:
                A_prop = XX1.copy()
                A_prop[np.diag_indices_from(XX1)] += prop_xi / lam1**2
                f_prop = cho_factor(
                    A_prop, lower=True, overwrite_a=True, check_finite=False
                )
                x_prop = y - X1 @ cho_solve(f_prop, X1y, check_finite=False)
                ldet_prop = float(
                    2.0 * np.sum(np.log(np.diag(f_prop[0])))
                    + log_det_gam - id1.size * np.log(prop_xi)
                )
                lr_prop = log_marg(y, x_prop, prop_xi, ldet_prop, n, a0, b0)
            XX1[np.diag_indices_from(XX1)] += xi / lam1**2
            f_curr = cho_factor(
                XX1, lower=True, overwrite_a=True, check_finite=False
            )
            x_curr = y - X1 @ cho_solve(f_curr, X1y, check_finite=False)
            ldet_curr = float(
                2.0 * np.sum(np.log(np.diag(f_curr[0])))
                + log_det_gam - id1.size * np.log(xi)
            )
            lr_curr = log_marg(y, x_curr, xi, ldet_curr, n, a0, b0)
            prop_valid = prop_ok
        else:  # empty active set: M = I, no factor needed
            x_prop = x_curr = y
            f_curr = f_prop = None
            lr_prop = log_marg(y, y, prop_xi, 0.0, n, a0, b0) if prop_ok else -np.inf
            lr_curr = log_marg(y, y, xi, 0.0, n, a0, b0)
            prop_valid = prop_ok
        matrix = None
    else:
        # ---- direct route: M = I + X diag(w) X' / xi, w = lambda^2 ----
        # masked weights avoid copying X[:, active]; dsyrk computes the
        # symmetric product in half the flops of a general matmul; the
        # Cholesky factor of each M is formed once and reused for
        # M^-1 y here and for the beta draw below.
        w = np.where(active_mask, lam**2, 0.0) if approx_xlx else lam**2
        # dsyrk(A) returns A A', so the columns carry sqrt(w):
        # dsyrk(X * sqrt(w)) = X diag(w) X'.  Scaling by w itself would
        # give X diag(w^2) X' -- the weight-squaring bug fixed 2026-10-02
        # (verify_port.py check 8 pins this).
        np.multiply(X, np.sqrt(w), out=work)
        C = dsyrk(1.0, work, trans=0, lower=1)
        f_curr = f_prop = None
        lr_prop = lr_curr = -np.inf
        if prop_ok:
            M_prop = C * (1.0 / prop_xi)
            M_prop[dg_n] += 1.0
            try:
                f_prop = cho_factor(
                    M_prop, lower=True, overwrite_a=True, check_finite=False
                )
                x_prop = cho_solve(f_prop, y, check_finite=False)
                lr_prop = log_marg(
                    y, x_prop, prop_xi, 2.0 * np.sum(np.log(np.diag(f_prop[0]))),
                    n, a0, b0,
                )
            except np.linalg.LinAlgError:
                f_prop = None
        M_curr = C * (1.0 / xi)
        M_curr[dg_n] += 1.0
        try:
            # keep M_curr intact (no overwrite): if the Cholesky fails
            # on an ill-conditioned collapsed-regime state, x_curr stays
            # zero (legacy behavior) and the beta draw below falls back to
            # an LU solve with M_curr
            f_curr = cho_factor(
                M_curr, lower=True, overwrite_a=False, check_finite=False
            )
            x_curr = cho_solve(f_curr, y, check_finite=False)
            lr_curr = log_marg(
                y, x_curr, xi, 2.0 * np.sum(np.log(np.diag(f_curr[0]))),
                n, a0, b0,
            )
        except np.linalg.LinAlgError:
            x_curr = np.zeros(n)
            f_curr = None
        if f_curr is None:
            # current state not positive definite cannot happen for
            # I + PSD; refuse to move so the chain stays defined
            lr_prop = -np.inf
        prop_valid = f_prop is not None

    log_acc = (lr_prop - lr_curr) + (np.log(prop_xi) - np.log(xi))
    accept = np.log(accept_u) < log_acc
    if accept and prop_valid:
        moved = True
        xi = prop_xi
        x_curr = x_prop
        f_curr = f_prop
        matrix = None  # the factor supersedes the matrix
    else:
        moved = False
        matrix = M_curr if not woodbury else None

    return XiUpdate(
        xi=xi,
        moved=moved,
        x_curr=x_curr,
        factor=f_curr,
        matrix=matrix,
        id1=id1,
        active_mask=active_mask,
        X1=X1,
        lam1=lam1,
        woodbury=woodbury,
        active_size=active_size,
    )

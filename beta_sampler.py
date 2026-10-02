"""Surrogate-variable sampling of the regression coefficients (beta).

Implements step 4 of the paper's algorithm (4) and its approximate
variant of Section 2.2.  Given the current precisions, the draw is

    u  ~ N(0, tau^2 Lambda)            (Lambda = diag(lambda_j^2))
    v  = X u + f,   f ~ N(0, I_n)
    v* = M_delta^-1 (y/sigma - v)
    beta = sigma (u + xi^-1 D_delta X' v*)

with M_delta = I_n + xi^-1 X D_delta X'.  Three routes, selected by
``mode``:

* ``"exact"``         : full D; v* via the n x n Cholesky factor of M
  (LU fallback on the retained dense matrix when the factorization fails
  on numerically degenerate states);
* ``"approx_direct"`` : same solve, but the correction
  xi^-1 D X' v* is masked to the active set S;
* ``"woodbury"``      : no n x n matrix is ever formed:
  v* = (I - X_S A^-1 X_S') resid with A = xi D_S^-1 + X_S'X_S (the
  s x s factor computed in the xi update), and the correction is added
  on S only.

``approx_direct`` and ``woodbury`` draw from the same thresholded
conditional with different linear algebra; which one runs is chosen by
the flop-count crossover in ``xi_sampler.update_xi`` (carried through
``XiUpdate.woodbury``), not by the caller.

The module is purely deterministic -- u and v are drawn by the caller so
that the random stream is owned by the main loop.  Arithmetic order
matches the original inline implementation bit for bit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve

__all__ = ["compute_v_star", "sample_beta"]


def compute_v_star(resid, factor=None, matrix=None):
    """v* = M^-1 resid for the exact / approx_direct routes.

    ``factor`` is a scipy ``cho_factor`` result for M; ``matrix`` is the
    dense M kept for the LU fallback used when the Cholesky failed.  It is
    stored as the dsyrk lower triangle (zero upper triangle), so the full
    symmetric matrix is rebuilt from the lower one before the general LU
    solve -- solving the half-zero array directly would be garbage.
    """
    if factor is not None:
        return cho_solve(factor, resid, check_finite=False)
    if matrix is not None:
        low = np.tril(matrix)
        return np.linalg.solve(low + low.T - np.diag(np.diag(low)), resid)
    raise ValueError("need a Cholesky factor or the dense matrix M")


def sample_beta(
    u,
    resid,
    *,
    X,
    lam,
    xi,
    sigma_sq,
    mode,
    factor=None,
    matrix=None,
    X1=None,
    lam1=None,
    id1=None,
    A_factor=None,
    active_mask=None,
):
    """Draw beta = sigma (u + xi^-1 D_delta X' v*).

    Parameters
    ----------
    u, resid : (p,) and (n,) arrays -- the surrogate draw and
        resid = y/sigma - v.
    X : (n, p) design matrix.
    lam : (p,) local scales (the full vector; masked/selected as needed).
    xi, sigma_sq : global precision and noise variance.
    mode : "exact" | "approx_direct" | "woodbury".
    factor, matrix : Cholesky factor of M (or of A for ``woodbury``) and
        the retained dense M for the LU fallback (direct/exact routes).
    X1, lam1, id1 : active-set columns, local scales and indices
        (``woodbury`` only; ``id1`` may be empty, in which case the draw
        is beta = sigma * u).
    A_factor : cho_factor of A = xi D_S^-1 + X_S'X_S (``woodbury`` only).
    active_mask : (p,) boolean mask of the active set
        (``approx_direct`` only).
    """
    if mode == "woodbury":
        beta = np.sqrt(sigma_sq) * u
        if id1 is not None and id1.size:
            # U v* with U = xi^-1 D_delta X' = (lambda_S^2 / xi) .* X_S'
            v_star = resid - X1 @ cho_solve(
                A_factor, X1.T @ resid, check_finite=False
            )
            beta[id1] += np.sqrt(sigma_sq) * ((lam1**2 / xi) * (X1.T @ v_star))
        return beta

    v_star = compute_v_star(resid, factor=factor, matrix=matrix)
    # (lambda^2 / xi) .* X' v*; in the approximate direct route the
    # correction applies only on the active set
    corr = (lam**2 / xi) * (X.T @ v_star)
    if mode == "approx_direct":
        corr = corr * active_mask
    elif mode != "exact":
        raise ValueError(f"unknown mode: {mode!r}")
    return np.sqrt(sigma_sq) * (u + corr)

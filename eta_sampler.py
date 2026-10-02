"""Exact rejection samplers for the horseshoe local scales (precisions eta_j).

Implements the sampler of Appendix S1 of Johndrow, Orenstein & Bhattacharya,
"Scalable Approximate MCMC Algorithms for the Horseshoe Prior".

The full conditional of the local precision is

    p(eta_j | beta, xi, sigma^2) proportional to exp(-M * t) / (1 + t),
    t > 0,   with M = beta_j^2 * xi / (2 * sigma^2),

whose negative log-density (up to constants) is

    f(x) = M * x + log(1 + x),  x > 0,

an increasing concave function.  Because f is concave, chords lie below it,
so a piecewise-linear lower bound f_L <= f with knots at a/M, 1/M, b/M
(defaults a = 1/5, b = 10) yields an envelope density
h_L(x) proportional to exp(-f_L(x)) that is a mixture of

    1. the exact 1/(1+x) piece on [0, a/M),
    2/3. truncated exponentials on [a/M, 1/M) and [1/M, b/M),
    4. an untruncated exponential on [b/M, inf),

all samplable by inverse CDF.  For M > 1 the plain Exp(M) rejection sampler
(hseta1) is already efficient and is used instead.

Numerically careful log(1+x) / exp(x)-1 are taken from numpy's log1p /
expm1, which are accurate for small arguments.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "neg_log_post",
    "approx_constants",
    "hseta1",
    "hseta_rs_small_m",
    "sample_eta",
]

# Default knot multipliers for the envelope (Figure S1 of the paper).
DEFAULT_A = 0.2  # 'a' in Appendix S1, 0 < a < 1
DEFAULT_B = 10.0  # 'b' in Appendix S1, b > 1


def neg_log_post(M, x):
    """f(x) = M*x + log(1+x): negative log-density of h(t) ~ exp(-M t)/(1+t).

    Vectorized; M and x broadcast against each other.
    """
    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float)
    return M * x + np.log1p(x)


def approx_constants(M, a=DEFAULT_A, b=DEFAULT_B):
    """Masses of the four envelope segments of h_L (Appendix S1).

    Returns a dict with

        nu     : total mass of h_L (its normalizing constant),
        nuvec  : (n, 4) per-segment masses (mixture weights are nuvec / nu),
        A, I, B: f at the knots a/M, 1/M, b/M,
        lam2, lam3 : slopes of the two chords,
        H2, H3  : truncated-exponential normalizers 1 - exp(-(I-A)), 1 - exp(-(B-I)).

    M, a, b are 1-D arrays of equal length (a < 1 < b elementwise).
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    a = np.broadcast_to(np.asarray(a, dtype=float), M.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), M.shape)

    A = neg_log_post(M, a / M)
    I = neg_log_post(M, 1.0 / M)
    B = neg_log_post(M, b / M)
    lam2 = M * (I - A) / (1.0 - a)
    lam3 = M * (B - I) / (b - 1.0)

    nu1 = np.log1p(a / M)  # integral of 1/(1+x) on [0, a/M)
    nu2 = (np.exp(-A) - np.exp(-I)) / lam2
    nu3 = (np.exp(-I) - np.exp(-B)) / lam3
    nu4 = np.exp(-B) / M
    nuvec = np.column_stack([nu1, nu2, nu3, nu4])

    return {
        "nu": nu1 + nu2 + nu3 + nu4,
        "nuvec": nuvec,
        "A": A,
        "I": I,
        "B": B,
        "lam2": lam2,
        "lam3": lam3,
        "H2": -np.expm1(-(I - A)),
        "H3": -np.expm1(-(B - I)),
    }


def hseta1(M, rng, lb=0.0):
    """Plain rejection sampler for eta ~ h(t) proportional to exp(-M t)/(1+t).

    Proposes from Exp(M) and accepts with probability
    1/(1+t); efficient when M > 1.  Vectorized over M with a retry loop on
    the rejected coordinates.  ``lb`` lower-truncates the target to
    (lb, inf) exactly (Appendix B eq. (26) with b > 0): for lb > 0 the
    proposal is the exponential restricted to (lb, inf), accepted with
    probability (1+lb)/(1+t).

    Returns (samples, numdraws).
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    samp = np.zeros(M.size)
    numdraw = np.ones(M.size, dtype=int)

    idx = np.arange(M.size)
    m = M
    if lb <= 0.0:
        while idx.size:
            t = -np.log1p(-rng.random(idx.size)) / m  # Exp(m) draws
            u = rng.random(idx.size)
            done = u < 1.0 / (1.0 + t)
            samp[idx[done]] = t[done]
            numdraw[idx[~done]] += 1
            keep = ~done
            idx = idx[keep]
            m = m[keep]
    else:
        # target restricted to (lb, inf).  Naively rejecting t <= lb from
        # the untruncated proposal needs ~e^{M*lb} retries when M*lb is
        # large (strongly identified coordinates), so instead propose from
        # the exponential restricted to (lb, inf) and accept with
        # probability (1+lb)/(1+t): g(t) = M e^{-M(t-lb)}, and
        # g(t) (1+lb)/(1+t) proportional to e^{-Mt}/(1+t) on (lb, inf).
        while idx.size:
            t = lb - np.log1p(-rng.random(idx.size)) / m  # Exp(m) on (lb,inf)
            u = rng.random(idx.size)
            # support guard: in float64 t == lb has probability ~1e-12 (the
            # guard is inert), but in float32 the subtraction rounds onto or
            # below lb ~0.1% of the time when m*lb is large (2026-10-02
            # fp32 study) and the atom leaks below the truncation point
            done = (u * (1.0 + t) < (1.0 + lb)) & (t > lb)
            samp[idx[done]] = t[done]
            numdraw[idx[~done]] += 1
            keep = ~done
            idx = idx[keep]
            m = m[keep]
    return samp, numdraw


def hseta_rs_small_m(
    M,
    a=DEFAULT_A,
    b=DEFAULT_B,
    rng=None,
    independent_accept_uniform=True,
    lb=0.0,
):
    """Appendix S1 rejection sampler for the case M <= 1.

    Draws from the 4-piece envelope h_L of ``approx_constants`` and accepts
    with probability exp(-(f - f_L)(t)).  Entries with M > 1 are routed to
    ``hseta1``.  ``lb`` lower-truncates the target to (lb, inf) by rejecting
    draws t <= lb with the same envelope (exact).  Returns (samples,
    numdraws) over the full input vector.

    Correctness note
    ----------------
    Appendix S1 of the paper specifies: "(i) draw z ~ h_L and u ~ U(0,1)
    *independently* ... (ii) accept z if u < exp(-(f - f_L)(z))".  With
    ``independent_accept_uniform=True`` (default) a fresh uniform is drawn
    for the acceptance test, as specified.  The legacy implementation
    reuses the uniform that selected the mixture component as the
    acceptance uniform; given the drawn t, that uniform is only uniform on
    the component's CDF interval, so the acceptance probability is not
    exp(-(f - f_L)(t)) and the sampler is inexact (the effect is quantified
    in verify_port.py).  Set ``independent_accept_uniform=False`` to
    reproduce the legacy behavior exactly.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = np.atleast_1d(np.asarray(M, dtype=float))
    # M = 0 would make the density improper in the tail and break the knot
    # arithmetic below; clip to the smallest positive double.
    M = np.clip(M, np.finfo(float).tiny, None)

    out = np.zeros(M.size)
    full_numdraw = np.ones(M.size, dtype=int)

    big = M > 1.0
    if big.any():
        out[big], full_numdraw[big] = hseta1(M[big], rng, lb=lb)

    small = np.flatnonzero(~big)
    if small.size == 0:
        return out, full_numdraw

    m = M[small]
    a = np.broadcast_to(np.asarray(a, dtype=float), m.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), m.shape)

    c = approx_constants(m, a, b)
    A, If, B = c["A"], c["I"], c["B"]
    lam2, lam3, H2, H3 = c["lam2"], c["lam3"], c["H2"], c["H3"]
    cum = np.cumsum(c["nuvec"] / c["nu"][:, None], axis=1)
    cum = np.hstack([np.zeros((m.size, 1)), cum])  # (n, 5) mixture CDF

    # pack the per-entry constants row-wise into a (14, k) contiguous block
    # (rows: m, a, b, A, I, B, lam2, lam3, H2, H3, c1..c4): reads stay
    # contiguous and each retry pass compacts ONE array instead of
    # gathering through `pos` a dozen times
    st = np.stack([
        m, a, b, A, If, B, lam2, lam3, H2, H3,
        cum[:, 1], cum[:, 2], cum[:, 3], cum[:, 4],
    ])

    pos = np.arange(m.size)  # positions still pending, into the small set
    xout = np.zeros(m.size)
    numdraw = np.ones(m.size, dtype=int)

    # the four envelope pieces are evaluated on the full pending set and
    # selected by segment; unselected lanes may overflow to inf/nan
    # harmlessly (np.where discards them).  The selected lane uses the same
    # arithmetic as the segment-wise form, so sampled values are
    # bit-identical to the previous implementation for a given seed.
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        while pos.size:
            k = pos.size
            u_sel = rng.random(k)  # picks the mixture component
            # cum row is [0, c1, c2, c3, 1]; the 1-based segment index is
            # 1 + #{j : c_j < u}
            ind = 1 + ((u_sel > st[10]).astype(np.int64)
                       + (u_sel > st[11])
                       + (u_sel > st[12])
                       + (u_sel > st[13]))

            ua = rng.random(k)  # position within the component

            m_, a_, b_ = st[0], st[1], st[2]
            A_, I_, B_ = st[3], st[4], st[5]
            l2, l3, H2_, H3_ = st[6], st[7], st[8], st[9]

            am = a_ / m_
            t1 = (1.0 + am) ** ua - 1.0
            t2 = am - np.log1p(-ua * H2_) / l2
            t3 = 1.0 / m_ - np.log1p(-ua * H3_) / l3
            t4 = b_ / m_ - np.log1p(-ua) / m_
            t = np.where(ind == 1, t1,
                         np.where(ind == 2, t2,
                                  np.where(ind == 3, t3, t4)))

            # acceptance ratio exp(f_L(t) - f(t)), f(t) = m t + log(1+t)
            r1 = np.exp(-m_ * t1)
            r2 = np.exp(A_ + l2 * (t2 - am) - (m_ * t2 + np.log1p(t2)))
            r3 = np.exp(I_ + l3 * (t3 - 1.0 / m_) - (m_ * t3 + np.log1p(t3)))
            r4 = np.exp(B_ + m_ * (t4 - b_ / m_) - (m_ * t4 + np.log1p(t4)))
            rat = np.where(ind == 1, r1,
                           np.where(ind == 2, r2,
                                    np.where(ind == 3, r3, r4)))

            if independent_accept_uniform:
                u_acc = rng.random(k)
            else:  # legacy behavior: reuse the component-selection uniform
                u_acc = u_sel
            # support rejection is safe here: this branch handles M <= 1,
            # for which a non-negligible fraction of the target mass lies
            # above any O(1) lb (retries stay O(1)); M > 1 goes to hseta1,
            # which uses a truncated-exponential proposal instead.
            done = (u_acc < rat) & (t > lb)

            xout[pos[done]] = t[done]
            numdraw[pos[~done]] += 1
            keep = ~done
            pos = pos[keep]
            st = st[:, keep]

    out[small] = xout
    full_numdraw[small] = numdraw
    return out, full_numdraw


def sample_eta(gamma_rate, rng=None, a=DEFAULT_A, b=DEFAULT_B, lb=0.0):
    """Sample eta_j ~ p(eta_j | beta, xi, sigma^2) for every j.

    gamma_rate[j] = beta_j^2 * xi / (2 sigma^2) is the 'M' of the target
    density exp(-M t)/(1+t).  Routes M > 1 to plain rejection (hseta1) and
    M <= 1 to the Appendix S1 envelope sampler.  ``lb`` lower-truncates the
    target to (lb, inf) (the b > 0 prior-truncation variant of Appendix B).
    """
    if rng is None:
        rng = np.random.default_rng()
    eta, _ = hseta_rs_small_m(gamma_rate, a=a, b=b, rng=rng, lb=lb)
    return eta

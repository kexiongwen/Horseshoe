"""Update of the noise variance sigma^2 (step 3 of the paper's eq. (4)).

Integrating beta out of the likelihood, the conditional posterior of the
noise precision kappa = 1/sigma^2 given (eta, xi) under the
InvGamma(a0/2, b0/2) prior is

    kappa | ...  ~  Gamma( (n + a0)/2 ,  rate = (ssr + b0)/2 ),
    ssr = y' M_xi^-1 y,

i.e. sigma^2 | ... ~ InvGamma((n + a0)/2, (ssr + b0)/2).  The update is
the exact conjugate draw.  

Like ``beta_sampler`` and ``xi_sampler``, the module owns no RNG: the
caller draws the Gamma variate, so the random stream stays with the main
loop and the stream order is unchanged relative to the original inline
code.
"""
from __future__ import annotations

__all__ = ["update_sigma2"]


def update_sigma2(gamma_draw):
    """Exact conjugate update: sigma^2 = 1/g with g ~ Gamma((n+a0)/2,
    scale = 2/(ssr+b0)) drawn by the caller.  Returns the new sigma^2."""
    return 1.0 / gamma_draw

"""Simulation driver reproducing the paper's experimental setup.

Default configuration replicates the paper's simulation: n = 2000,
p = 20000, BetaTrue(1:23) = 2.^(-(-2:.25:3.5)) (23 decaying signals from
4 down to 2^-3.5), noise variance 2, approximate algorithm with
delta = 1e-4, conjugate update for sigma^2, burnin 1000 + 20000 draws.

Examples
--------
    python sims.py --test              # small fast check (n=200, p=500)
    python sims.py                     # paper configuration (slow)
    python sims.py --algorithm exact   # exact (non-thresholded) algorithm
    python sims.py --n 1000 --p 5000 --mcmc 5000
"""
from __future__ import annotations

import argparse

import numpy as np

from horseshoe import horseshoe, plot_diagnostics


def make_sim(n, p, sigma_true=2.0, seed=0):
    """Generate one replicate of the paper's simulation design."""
    beta_true = np.zeros(p)
    beta_true[:23] = 2.0 ** (-np.arange(-2.0, 3.5 + 1e-9, 0.25))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X @ beta_true + np.sqrt(sigma_true) * rng.standard_normal(n)
    return y, X, beta_true


def run(
    n=2000,
    p=20000,
    nmc=20000,
    burnin=1000,
    algorithm="approx",  # 'approx' | 'exact'
    delta=1e-4,
    mh_scale=0.8,
    n_keep=100,
    seed=5171,
    data_seed=0,
    plot=False,
    save=True,
    save_sim_vars=False,
    out_dir="Outputs",
    verbose=1000,
    simtype="F",
):
    y, X, beta_true = make_sim(n, p, sigma_true=2.0, seed=data_seed)

    if save_sim_vars:
        import os

        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f"sim_vars_{n}_{p}.npz"), X=X, y=y)
        print(f"saved simulation variables to {out_dir}/sim_vars_{n}_{p}.npz")

    rng = np.random.default_rng(seed)
    res = horseshoe(
        y, X,
        burnin=burnin, mcmc=nmc, thin=1,
        mh_scale_ub=mh_scale, mh_scale_lb=mh_scale,
        a0=1.0, b0=1.0,
        approx_xlx=(algorithm == "approx"),
        delta=delta,
        n_keep=n_keep, beta_true=beta_true, is_sim=True,
        rng=rng, save_samples=save, save_dir=out_dir,
        simtype=simtype, verbose=verbose,
    )
    if plot:
        plot_diagnostics(res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=20000)
    ap.add_argument("--burnin", type=int, default=1000)
    ap.add_argument("--mcmc", type=int, default=20000, help="post-burnin draws")
    ap.add_argument("--algorithm", choices=["approx", "exact"], default="approx")
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--mh-scale", type=float, default=0.8)
    ap.add_argument("--n-keep", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5171)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--test", action="store_true",
                    help="small fast configuration (n=200, p=500, 2000 draws)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--save-sim-vars", action="store_true",
                    help="also save X and y to an .npz (large: n*p*8 bytes)")
    ap.add_argument("--out-dir", default="Outputs")
    ap.add_argument("--quiet", type=int, default=1000, help="progress interval")
    args = ap.parse_args()

    kw = dict(
        n=args.n, p=args.p, nmc=args.mcmc, burnin=args.burnin,
        algorithm=args.algorithm, delta=args.delta,
        mh_scale=args.mh_scale, n_keep=args.n_keep, seed=args.seed,
        data_seed=args.data_seed, plot=args.plot, save=not args.no_save,
        save_sim_vars=args.save_sim_vars, out_dir=args.out_dir,
        verbose=args.quiet,
    )
    if args.test:
        kw.update(n=200, p=500, nmc=2000, burnin=500, n_keep=50)

    print(f"running '{kw['algorithm']}' with n={kw['n']}, p={kw['p']}, "
          f"burnin={kw['burnin']}, mcmc={kw['nmc']}")
    run(**kw)


if __name__ == "__main__":
    main()

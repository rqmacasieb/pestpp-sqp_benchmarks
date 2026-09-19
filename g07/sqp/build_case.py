"""
Build a multi-seed run of G07 (the classic 10-decision-variable,
8-inequality-constraint nonlinear test problem) for PESTPP-SQP.

G07 has no varying dimensionality to sweep, so this builds exactly one case
("g07"), fanned out into N_SEEDS independent runs, one per Latin-Hypercube
sampled starting point within the template's [-10, 10] decision-variable
bounds:

    g07/sqp/g07/lhs_starting_values.csv
    g07/sqp/g07/seed1/template/ ...
    g07/sqp/g07/seed2/template/ ...
    ...

Reuses ../template/g07.pst verbatim as the starting configuration -- it's
already tuned for PESTPP-SQP (sqp_num_reals=20, sqp_subset_size=5,
sqp_alpha_mults, sqp_update_hessian, etc.). This script only overrides
parval1 (per-seed LHS starting point), random_seed, and (optionally) noptmax;
every other pestpp_option is left exactly as the template has it.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-sqp binary (on your PATH, next to run_case.py, or passed to it with --exe).

Usage:
    python build_case.py                   # build all 50 seeds
    python build_case.py --noptmax 50

Then run the seeds in parallel on this computer with run_case.py (same
directory):
    python run_case.py --seeds 1 2 3
    python run_case.py --concurrent 4

Note: the LHS design draws N_SEEDS points at once, so a seed's starting point
depends on how many seeds were built. To use the canonical design, build all
50 (the default) and select which seeds to run with run_case.py --seeds,
rather than building fewer seeds with --n_seeds.
"""
import os
import argparse
import shutil
import numpy as np
import pandas as pd
import pyemu
from scipy import stats
from scipy.spatial.distance import pdist

SQP_DIR = os.path.dirname(os.path.abspath(__file__))
G07_DIR = os.path.dirname(SQP_DIR)
TEMPLATE_DIR = os.path.join(G07_DIR, "template")
TEMPLATE_PST = os.path.join(TEMPLATE_DIR, "g07.pst")
CASE_NAME = "g07"
N_DECVARS = 10
PARNAMES = ["x{0}".format(i + 1) for i in range(N_DECVARS)]

PARLBND = -10.0
PARUBND = 10.0

N_SEEDS = 50
BASE_SEED = 1  # seeds run BASE_SEED .. BASE_SEED + N_SEEDS - 1
LHS_SEED = 42  # seeds the LHS design (one draw of N_SEEDS starting points)


def generate_starting_values(n_samples, n_dimensions, bounds, seed, iterations=1000):
    """Maximin Latin-Hypercube design: n_samples points in n_dimensions,
    scaled to per-dimension bounds. Reproducible from seed alone."""
    np.random.seed(seed)

    # basic LHS: one random point per equal-probability interval, shuffled per dimension
    cut = np.linspace(0, 1, n_samples + 1)
    samples = np.zeros((n_samples, n_dimensions))
    for i in range(n_dimensions):
        samples[:, i] = stats.uniform(cut[:-1], np.diff(cut)).rvs(n_samples)
        np.random.shuffle(samples[:, i])

    # maximin: randomly swap coordinates, keeping swaps that increase the minimum pairwise distance
    best_samples = samples.copy()
    best_min_dist = np.min(pdist(samples))
    for _ in range(iterations):
        dim = np.random.randint(0, n_dimensions)
        i, j = np.random.choice(n_samples, 2, replace=False)
        samples[i, dim], samples[j, dim] = samples[j, dim], samples[i, dim]
        min_dist = np.min(pdist(samples))
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_samples = samples.copy()
        else:
            samples[i, dim], samples[j, dim] = samples[j, dim], samples[i, dim]
    samples = best_samples

    bounds = np.asarray(bounds)
    return bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])


def write_lhs_starting_values(case_dir, seeds, lhs_seed=LHS_SEED):
    """Draw one Latin-Hypercube design of len(seeds) starting points across
    all 10 decision variables at once and save it as
    g07/sqp/g07/lhs_starting_values.csv, indexed by seed number. Every seed's
    build then reads its starting point from this same design, so the
    N_SEEDS starting points properly space-fill the decision space instead of
    each seed drawing an independent uniform sample."""
    os.makedirs(case_dir, exist_ok=True)
    bounds = np.tile([PARLBND, PARUBND], (N_DECVARS, 1))
    samples = generate_starting_values(len(seeds), N_DECVARS, bounds, seed=lhs_seed)
    df = pd.DataFrame(samples, index=list(seeds), columns=PARNAMES)
    df.index.name = "seed"
    path = os.path.join(case_dir, "lhs_starting_values.csv")
    df.to_csv(path)
    return df, path


def build_seed_case(seed, init_vals, case_dir, noptmax):
    seed_dir = os.path.join(case_dir, "seed{0}".format(seed))
    seed_template_dir = os.path.join(seed_dir, "template")
    if os.path.exists(seed_template_dir):
        shutil.rmtree(seed_template_dir)
    os.makedirs(seed_template_dir)

    for fname in ("par.tpl", "obs.ins", "forward_run.py"):
        shutil.copy(os.path.join(TEMPLATE_DIR, fname), os.path.join(seed_template_dir, fname))

    cwd = os.getcwd()
    os.chdir(seed_template_dir)
    try:
        # start from the already-tuned template pst verbatim -- only the
        # starting point, random_seed, and noptmax change per seed
        pst = pyemu.Pst(TEMPLATE_PST)

        par = pst.parameter_data
        for name, val in zip(PARNAMES, init_vals):
            par.loc[name, "parval1"] = val

        pst.pestpp_options["random_seed"] = seed
        pst.control_data.noptmax = noptmax

        pst_name = "{0}.pst".format(CASE_NAME)
        pst.write(pst_name)
        num_reals = int(pst.pestpp_options.get("sqp_num_reals", 20))
    finally:
        os.chdir(cwd)

    return seed_template_dir, pst_name, num_reals


def build_case(n_seeds=N_SEEDS, base_seed=BASE_SEED, noptmax=None):
    case_dir = os.path.join(SQP_DIR, CASE_NAME)

    if noptmax is None:
        noptmax = pyemu.Pst(TEMPLATE_PST).control_data.noptmax  # template's own value (100)

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = write_lhs_starting_values(case_dir, seeds)

    num_reals = None
    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        _, _, num_reals = build_seed_case(seed, init_vals, case_dir, noptmax)

    print("built case: {0} ({1} dec vars, {2} seeds, noptmax={3})".format(
        case_dir, N_DECVARS, n_seeds, noptmax))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  each seed: template dir {0}/seed<N>/template/, g07.pst; "
          "up to {1} parallel workers (sqp_num_reals)".format(case_dir, num_reals))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build a multi-seed G07 PESTPP-SQP run")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                         help="number of seeds to build (default {0})".format(N_SEEDS))
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--noptmax", type=int, default=None,
                         help="defaults to the template's own noptmax (see ../template/g07.pst)")
    args = parser.parse_args()

    build_case(n_seeds=args.n_seeds, base_seed=args.base_seed, noptmax=args.noptmax)

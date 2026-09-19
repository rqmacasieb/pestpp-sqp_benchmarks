"""
Build N-dimensional Rosenbrock test cases: the chained Rosenbrock sum objective with 3
pairwise-block-replicated linear constraints. n = 2, 20, 50 and 100 decision
variables (n must be even; n=2 is the original 2D three-linear-constraint
case, 1 block).

Each case fans out into N_SEEDS independent runs, one per Latin-Hypercube
sampled starting point within [-2.2, 2.2], laid out as

    rosenbrock_2d/lhs_starting_values.csv
    rosenbrock_2d/seed1/template/ ...
    rosenbrock_2d/seed2/template/ ...
    ...

for PESTPP-SQP. Seed s draws its starting point from the case's LHS design
and also sets that same integer as pestpp_options["random_seed"].

Ensemble size heuristics (starting values; adjust if convergence is poor or
too slow): sqp_num_reals = max(20, 2n) (20, 40, 100, 200 for the four cases),
sqp_subset_size = max(10, n). run_case.py caps the parallel workers at your
core count.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-sqp binary (on your PATH, next to run_case.py, or passed to it with --exe).

Usage:
    python build_case.py                   # build every case, 50 seeds each
    python build_case.py 20                # build one case
    python build_case.py 20 --noptmax 30

Then run the seeds in parallel on this computer with run_case.py (same
directory):
    python run_case.py 20 --seeds 1 2 3
    python run_case.py 20 --concurrent 4

Note: the LHS design draws N_SEEDS points at once, so a seed's starting point
depends on how many seeds were built. To use the canonical design, build all
50 (the default) and select which seeds to run with run_case.py --seeds,
rather than building fewer seeds.
"""
import os
import re
import argparse
import shutil
import numpy as np
import pandas as pd
import pyemu
from scipy import stats
from scipy.spatial.distance import pdist

METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
ROSENBROCK_DIR = os.path.dirname(METHOD_DIR)
TEMPLATE_DIR = os.path.join(ROSENBROCK_DIR, "template")
SQP_DIR = os.path.join(ROSENBROCK_DIR, "sqp")
FORWARD_MODEL = "rosenbrock_nd_three_linear_constrained.py"
CASE_FMT = "rosenbrock_{0}d"

FIELD_INNER = 13
PARLBND = -2.2
PARUBND = 2.2

N_SEEDS = 50
BASE_SEED = 1  # seeds run BASE_SEED .. BASE_SEED + N_SEEDS - 1
LHS_SEED = 42  # seeds the LHS design (one draw of N_SEEDS starting points) per case
# n=2 reproduces the original 2D three-linear-constraint case exactly (1 block,
# same constraint coefficients and RHS), just with x1/x2 names
NOPTMAX_BY_CASE = {2: 20, 20: 50, 50: 100, 100: 150}
DEFAULT_NOPTMAX = 20
DEFAULT_CASES = (2, 20, 50, 100)


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


def _tpl_field(name):
    return "~{0:^{1}}~".format(name, FIELD_INNER)


def _write_par_tpl_and_dat(case_dir, n, init_vals):
    parnames = ["x{0}".format(i + 1) for i in range(n)]

    with open(os.path.join(case_dir, "par.dat.tpl"), "w") as f:
        f.write("ptf ~\n")
        f.write(" ".join(_tpl_field(p) for p in parnames) + "\n")

    with open(os.path.join(case_dir, "par.dat"), "w") as f:
        f.write(" ".join("{0:.8E}".format(v) for v in init_vals) + "\n")

    return parnames


def _stage_model_files(seed_template_dir):
    """Copy the forward model and write constraints.dat.ins -- always
    exactly 3 constraints, regardless of n_decvars."""
    for fname in ("obs.dat.ins", FORWARD_MODEL):
        shutil.copy(os.path.join(TEMPLATE_DIR, fname), os.path.join(seed_template_dir, fname))

    with open(os.path.join(seed_template_dir, "constraints.dat.ins"), "w") as f:
        f.write("pif ~\nl1  w  !constraint1!\nl1  w  !constraint2!\nl1  w  !constraint3!\n")


def _set_constraint_data(obs, n_decvars):
    """Constraint senses and right-hand sides: the 2D three-linear-constraint
    case replicated over n_decvars // 2 consecutive (x[2k], x[2k+1]) blocks
    and summed, so the RHS scale with the number of blocks."""
    n_blocks = n_decvars // 2
    obs.loc["constraint1", "obgnme"] = "l_constraint"
    obs.loc["constraint1", "obsval"] = -0.5 * n_blocks
    obs.loc["constraint2", "obgnme"] = "l_constraint"
    obs.loc["constraint2", "obsval"] = 0.0 * n_blocks
    obs.loc["constraint3", "obgnme"] = "g_constraint"
    obs.loc["constraint3", "obsval"] = -1.0 * n_blocks


def write_lhs_starting_values(case_dir, n_decvars, seeds, lhs_seed=LHS_SEED):
    """Draw one Latin-Hypercube design of len(seeds) starting points across
    all n_decvars dimensions at once and save it as a CSV in the case
    directory (e.g. rosenbrock_2d/lhs_starting_values.csv), indexed by seed number,
    columns x1..xn. Every seed's build then reads its starting point from
    this same design, so the N_SEEDS starting points properly space-fill the
    decision space instead of each seed drawing an independent uniform sample."""
    os.makedirs(case_dir, exist_ok=True)
    bounds = np.tile([PARLBND, PARUBND], (n_decvars, 1))
    samples = generate_starting_values(len(seeds), n_decvars, bounds, seed=lhs_seed)
    parnames = ["x{0}".format(i + 1) for i in range(n_decvars)]
    df = pd.DataFrame(samples, index=list(seeds), columns=parnames)
    df.index.name = "seed"
    path = os.path.join(case_dir, "lhs_starting_values.csv")
    df.to_csv(path)
    return df, path


def build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, noptmax, num_reals, subset_size):
    seed_dir = os.path.join(case_dir, "seed{0}".format(seed))
    seed_template_dir = os.path.join(seed_dir, "template")
    if os.path.exists(seed_template_dir):
        shutil.rmtree(seed_template_dir)
    os.makedirs(seed_template_dir)

    _stage_model_files(seed_template_dir)
    parnames = _write_par_tpl_and_dat(seed_template_dir, n_decvars, init_vals)

    cwd = os.getcwd()
    os.chdir(seed_template_dir)
    try:
        pst = pyemu.helpers.pst_from_io_files(
            "par.dat.tpl", "par.dat",
            ["obs.dat.ins", "constraints.dat.ins"],
            ["obs.dat", "constraints.dat"])

        par = pst.parameter_data
        par.loc[:, "partrans"] = "none"
        par.loc[:, "parlbnd"] = PARLBND
        par.loc[:, "parubnd"] = PARUBND
        par.loc[:, "parchglim"] = "relative"
        for name, val in zip(parnames, init_vals):
            par.loc[name, "parval1"] = val

        obs = pst.observation_data
        obs.loc["obs", "obgnme"] = "obj_fn"
        obs.loc["obs", "obsval"] = 0.0
        _set_constraint_data(obs, n_decvars)
        obs.loc[:, "weight"] = 1.0

        pst.pestpp_options["opt_obj_func"] = "obs"
        pst.pestpp_options["sqp_num_reals"] = num_reals
        pst.pestpp_options["sqp_update_hessian"] = "true"
        pst.pestpp_options["sqp_subset_size"] = subset_size
        pst.pestpp_options["par_sigma_range"] = 10
        pst.pestpp_options["sqp_alpha_mults"] = "-0.05, 0.01, 0.1, 0.5, 1.0"
        pst.pestpp_options["random_seed"] = seed

        pst.model_command = ["python {0}".format(FORWARD_MODEL)]
        pst.control_data.noptmax = noptmax

        # pyemu sorts parameter_data alphabetically (x1, x10, x11, ..., x2, ...);
        # reorder numerically so the pst file lists x1, x2, x3, ... in sequence
        pst.parameter_data = pst.parameter_data.loc[parnames]

        pst_name = "{0}.pst".format(case_name)
        pst.write(pst_name)
    finally:
        os.chdir(cwd)

    return seed_template_dir, pst_name


def build_case(n_decvars, n_seeds=N_SEEDS, base_seed=BASE_SEED, noptmax=None, num_reals=None,
                subset_size=None):
    if n_decvars % 2 != 0:
        raise ValueError("n_decvars must be even (pairwise-block constraints): got {0}".format(n_decvars))

    case_name = CASE_FMT.format(n_decvars)
    case_dir = os.path.join(METHOD_DIR, case_name)

    if noptmax is None:
        noptmax = NOPTMAX_BY_CASE.get(n_decvars, DEFAULT_NOPTMAX)
    # ensemble-based SQP needs more realizations to estimate gradient/Hessian
    # info reliably as the decision space grows
    if num_reals is None:
        num_reals = max(20, 2 * n_decvars)
    if subset_size is None:
        subset_size = max(10, n_decvars)

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = write_lhs_starting_values(case_dir, n_decvars, seeds)

    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, noptmax, num_reals, subset_size)

    print("built case: {0} ({1} dec vars, {2} blocks, {3} seeds, noptmax={4}, num_reals={5}, "
          "subset_size={6})".format(case_dir, n_decvars, n_decvars // 2, n_seeds, noptmax,
                                    num_reals, subset_size))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  run it:  python run_case.py {0} --seeds {1}".format(n_decvars, base_seed))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build multi-seed Rosenbrock N-D PESTPP-SQP cases")
    parser.add_argument("n_decvars", nargs="*", type=int, default=None,
                         help="which case(s) to (re)build, e.g. 50 -- "
                              "defaults to all ({0})".format(", ".join(map(str, DEFAULT_CASES))))
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                         help="number of seeds to build per case (default {0})".format(N_SEEDS))
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--noptmax", type=int, default=None,
                         help="noptmax applied uniformly to every case built this call; defaults to "
                              "a per-case value ({0})".format(NOPTMAX_BY_CASE))
    args = parser.parse_args()

    for n in (args.n_decvars if args.n_decvars else list(DEFAULT_CASES)):
        build_case(n, n_seeds=args.n_seeds, base_seed=args.base_seed, noptmax=args.noptmax)

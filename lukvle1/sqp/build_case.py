"""
Build N-dimensional Luksan-Vlcek Problem 1 (LUKVLE1) test cases -- chained
Rosenbrock objective with n-2 nonlinear equality constraints -- for
PESTPP-SQP.

Each case (n = 20 and 200 decision variables) fans out into N_SEEDS
independent runs, one per Latin-Hypercube sampled starting point within
[-2.2, 2.2], laid out as

    lukvle1_20d/lhs_starting_values.csv
    lukvle1_20d/seed1/template/ ...
    lukvle1_20d/seed2/template/ ...
    ...

Seed s draws its decision-variable starting values from the case's LHS design
and also sets that same integer as pestpp_options["random_seed"] -- both
reproducible from the seed number alone.

Ensemble size heuristics (starting values; adjust if convergence is poor or
too slow): sqp_num_reals = max(20, 3n), sqp_subset_size = max(10, 2n). Note
this means 60 parallel model runs per iteration for n=20 and 600 for n=200 --
run_case.py caps the workers at your core count. sqp_viol_pad is a flat 2.5:
it is a PER-CONSTRAINT violation tolerance, not a total-sum tolerance, so it
does not scale with n or the number of constraints.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-sqp binary (on your PATH, next to run_case.py, or passed to it with --exe).

Usage:
    python build_case.py                   # build every case, 50 seeds each
    python build_case.py 20               # build one case
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
LUKVLE1_DIR = os.path.dirname(METHOD_DIR)
TEMPLATE_DIR = os.path.join(LUKVLE1_DIR, "template")
SQP_DIR = os.path.join(LUKVLE1_DIR, "sqp")
FORWARD_MODEL = "lukvle1_nd_equality_constrained.py"
CASE_FMT = "lukvle1_{0}d"

FIELD_INNER = 13
PARLBND = -2.2
PARUBND = 2.2

N_SEEDS = 50
BASE_SEED = 1  # seeds run BASE_SEED .. BASE_SEED + N_SEEDS - 1
LHS_SEED = 42  # seeds the LHS design (one draw of N_SEEDS starting points) per case
NOPTMAX_BY_CASE = {20: 50, 200: 150}
DEFAULT_NOPTMAX = 20
DEFAULT_CASES = (20, 200)


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


def _write_constraints_ins(case_dir, n_decvars):
    """n_decvars - 2 nonlinear equality constraints, regardless of case size."""
    n_constraints = n_decvars - 2
    with open(os.path.join(case_dir, "constraints.dat.ins"), "w") as f:
        f.write("pif ~\n")
        for i in range(1, n_constraints + 1):
            f.write("l1  w  !constraint{0}!\n".format(i))
    return n_constraints


def write_lhs_starting_values(case_dir, n_decvars, seeds, lhs_seed=LHS_SEED):
    """Draw one Latin-Hypercube design of len(seeds) starting points across
    all n_decvars dimensions at once and save it as a CSV in the case
    directory (e.g. lukvle1_20d/lhs_starting_values.csv), indexed by seed number,
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


def build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, noptmax, num_reals, subset_size, viol_pad):
    n_constraints = n_decvars - 2
    seed_dir = os.path.join(case_dir, "seed{0}".format(seed))
    seed_template_dir = os.path.join(seed_dir, "template")
    if os.path.exists(seed_template_dir):
        shutil.rmtree(seed_template_dir)
    os.makedirs(seed_template_dir)

    for fname in ("obs.dat.ins", FORWARD_MODEL):
        shutil.copy(os.path.join(TEMPLATE_DIR, fname), os.path.join(seed_template_dir, fname))

    _write_constraints_ins(seed_template_dir, n_decvars)
    parnames = _write_par_tpl_and_dat(seed_template_dir, n_decvars, init_vals)

    constraint_names = ["constraint{0}".format(i + 1) for i in range(n_constraints)]

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
        # all n_decvars - 2 constraints are equality (c_i(x) = 0)
        obs.loc[constraint_names, "obgnme"] = "e_constraint"
        obs.loc[constraint_names, "obsval"] = 0.0
        obs.loc[:, "weight"] = 1.0

        pst.pestpp_options["opt_obj_func"] = "obs"
        pst.pestpp_options["sqp_num_reals"] = num_reals
        pst.pestpp_options["sqp_update_hessian"] = "true"
        pst.pestpp_options["sqp_subset_size"] = subset_size
        pst.pestpp_options["par_sigma_range"] = 20
        pst.pestpp_options["sqp_alpha_mults"] = "-0.05, 0.01, 0.1, 0.5, 1.0"
        pst.pestpp_options["sqp_viol_pad"] = viol_pad
        pst.pestpp_options["sqp_num_refined_search_pts"] = 2
        pst.pestpp_options["sqp_scale_down_factor"] = 0.9
        pst.pestpp_options["sqp_rescale_search_dir"] = "true"
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
                subset_size=None, viol_pad=None):
    if n_decvars < 3:
        raise ValueError("n_decvars must be >= 3 (need at least one equality constraint): got {0}".format(n_decvars))

    case_name = CASE_FMT.format(n_decvars)
    case_dir = os.path.join(METHOD_DIR, case_name)
    n_constraints = n_decvars - 2

    if noptmax is None:
        noptmax = NOPTMAX_BY_CASE.get(n_decvars, DEFAULT_NOPTMAX)
    # ensemble-based SQP needs more realizations to estimate gradient/Hessian
    # info reliably as the decision space grows
    if num_reals is None:
        num_reals = max(20, 3 * n_decvars)
    if subset_size is None:
        subset_size = max(10, 2 * n_decvars)
    if viol_pad is None:
        viol_pad = 2.5

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = write_lhs_starting_values(case_dir, n_decvars, seeds)

    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, noptmax, num_reals, subset_size, viol_pad)

    print("built case: {0} ({1} dec vars, {2} equality constraints, {3} seeds, noptmax={4}, num_reals={5}, "
          "subset_size={6}, viol_pad={7})".format(
              case_dir, n_decvars, n_constraints, n_seeds, noptmax, num_reals, subset_size, viol_pad))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  run it:  python run_case.py {0} --seeds {1}".format(n_decvars, base_seed))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build multi-seed LUKVLE1 N-D PESTPP-SQP cases")
    parser.add_argument("n_decvars", nargs="*", type=int, default=None,
                         help="which case(s) to (re)build, e.g. 20 -- "
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

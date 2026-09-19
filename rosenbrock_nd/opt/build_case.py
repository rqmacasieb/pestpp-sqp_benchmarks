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

for PESTPP-OPT (sequential linear programming with linearized chance
constraints). Each case is the *same problem* as the ../sqp build.

Reuses ../sqp's per-case, per-seed LHS starting-value design verbatim (loaded
from ../sqp/rosenbrock_{n}d/lhs_starting_values.csv if that case has already
been built there, else regenerated with the same LHS_SEED, which is
reproducibly identical) -- so seed s's PESTPP-OPT parval1 values are exactly
seed s's PESTPP-SQP starting point.

The PST is set up with opt_dec_var_groups / opt_obj_func(obs) /
opt_direction(min) / opt_risk(0.5): risk-neutral, so FOSM/chance-constraint
calculations are skipped and each SLP iteration costs n + 1 model runs (one
base run plus one finite-difference run per decision variable). The "obs"
objective is zero-weighted (a non-zero weight marks an observation as part of
PESTPP-OPT's notional calibration dataset used to build the FOSM J matrix,
and the objective can't be both that and the opt_obj_func-referenced
objective) and stays in its own "obj_fn" group ("l_"/"g_" group prefixes are
reserved for constraints).

noptmax: by default it matches that *same seed's* completed PESTPP-SQP run.
If ../sqp/rosenbrock_{n}d/seed{s}/{master,template}/rosenbrock_{n}d.rec exists,
its last "number of model runs:" line sets noptmax = round(sqp_total_runs /
(n_decvars + 1)). If that record isn't found for a seed, the case's default noptmax
(NOPTMAX_BY_CASE) is used instead, with a warning. Pass --noptmax to set one
value for every seed and skip the matching entirely.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-opt binary (on your PATH, next to run_case.py, or passed to it with --exe).

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

_MODEL_RUNS_RE = re.compile(r"number of model runs:\s*(\d+)")


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


def _sqp_total_model_runs(n_decvars, seed):
    """Total model runs of the matching PESTPP-SQP seed run (last 'number of
    model runs:' line in its .rec), used to size noptmax so this seed's run
    spends roughly the same model-run budget. Returns None if that SQP run
    hasn't been done."""
    case_name = CASE_FMT.format(n_decvars)
    for sub in ("master", "template"):
        rec_file = os.path.join(SQP_DIR, case_name, "seed{0}".format(seed), sub,
                                 "{0}.rec".format(case_name))
        if not os.path.exists(rec_file):
            continue
        last_count = None
        with open(rec_file, "r") as f:
            for line in f:
                m = _MODEL_RUNS_RE.search(line)
                if m:
                    last_count = int(m.group(1))
        if last_count is not None:
            return last_count
    return None


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


def load_or_generate_lhs_starting_values(case_dir, n_decvars, seeds, lhs_seed=LHS_SEED):
    """Reuse ../sqp's per-case LHS starting-value design verbatim if it's
    already been built (guarantees identical starting points across suites
    regardless of RNG/library-version drift); otherwise generate a fresh
    design with the same lhs_seed, which is reproducibly identical to what
    ../sqp/build_case.py would produce anyway."""
    os.makedirs(case_dir, exist_ok=True)
    case_name = os.path.basename(case_dir)
    sqp_csv = os.path.join(SQP_DIR, case_name, "lhs_starting_values.csv")
    dest = os.path.join(case_dir, "lhs_starting_values.csv")

    if os.path.exists(sqp_csv):
        df = pd.read_csv(sqp_csv, index_col="seed")
        missing = [s for s in seeds if s not in df.index]
        if missing:
            raise ValueError(
                "{0} is missing seed(s) {1} -- rebuild the ../sqp case with "
                "matching N_SEEDS/BASE_SEED first".format(sqp_csv, missing))
        df.to_csv(dest)
        return df, dest

    bounds = [[PARLBND, PARUBND]] * n_decvars
    samples = generate_starting_values(len(seeds), n_decvars, bounds, seed=lhs_seed)
    parnames = ["x{0}".format(i + 1) for i in range(n_decvars)]
    df = pd.DataFrame(samples, index=list(seeds), columns=parnames)
    df.index.name = "seed"
    df.to_csv(dest)
    return df, dest


def build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, noptmax):
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
        # the objective observation must NOT carry an "l_"/"g_" (or
        # "less_"/"greater_") group prefix -- that naming is reserved for
        # PESTPP-OPT constraints -- so it stays its own, unconstrained group
        obs.loc["obs", "obgnme"] = "obj_fn"
        obs.loc["obs", "obsval"] = 0.0
        _set_constraint_data(obs, n_decvars)
        obs.loc[:, "weight"] = 1.0
        # a non-zero weight marks an observation as part of PESTPP-OPT's
        # notional calibration dataset (used to build the FOSM J matrix for
        # chance constraints); the objective observation can't be both that
        # and the (opt_obj_func-referenced) objective, so it must be zero-weighted
        obs.loc["obs", "weight"] = 0.0

        pst.pestpp_options["opt_dec_var_groups"] = "pargp"
        pst.pestpp_options["opt_obj_func"] = "obs"
        pst.pestpp_options["opt_direction"] = "min"
        pst.pestpp_options["opt_risk"] = 0.5
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

    # PESTPP-OPT's per-SLP-iteration response matrix needs one base run plus
    # one finite-difference run per decision variable (risk-neutral, no
    # separate FOSM J matrix here)
    return seed_template_dir, pst_name, n_decvars + 1


def build_case(n_decvars, n_seeds=N_SEEDS, base_seed=BASE_SEED, noptmax=None):
    if n_decvars % 2 != 0:
        raise ValueError("n_decvars must be even (pairwise-block constraints): got {0}".format(n_decvars))

    case_name = CASE_FMT.format(n_decvars)
    case_dir = os.path.join(METHOD_DIR, case_name)

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = load_or_generate_lhs_starting_values(case_dir, n_decvars, seeds)

    default_noptmax = NOPTMAX_BY_CASE.get(n_decvars, DEFAULT_NOPTMAX)
    explicit_noptmax = noptmax is not None
    unmatched = []
    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        seed_noptmax = noptmax
        if not explicit_noptmax:
            sqp_total_runs = _sqp_total_model_runs(n_decvars, seed)
            if sqp_total_runs is None:
                unmatched.append(seed)
                seed_noptmax = default_noptmax
            else:
                seed_noptmax = max(1, round(sqp_total_runs / (n_decvars + 1)))
        build_seed_case(n_decvars, seed, init_vals, case_name, case_dir, seed_noptmax)

    print("built case: {0} ({1} dec vars, {2} blocks, {3} seeds)".format(
        case_dir, n_decvars, n_decvars // 2, n_seeds))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  run it:  python run_case.py {0} --seeds {1}".format(n_decvars, base_seed))
    if unmatched:
        print("  WARNING: no completed ../sqp run found for {0} of {1} seeds{2} -- used the default "
              "noptmax={3} for those. Run ../sqp first and rebuild to match the SQP model-run "
              "budget, or pass --noptmax.".format(
                  len(unmatched), n_seeds, " ({0})".format(unmatched) if len(unmatched) <= 10 else "",
                  default_noptmax))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build multi-seed Rosenbrock N-D PESTPP-OPT cases")
    parser.add_argument("n_decvars", nargs="*", type=int, default=None,
                         help="which case(s) to (re)build, e.g. 50 -- "
                              "defaults to all ({0})".format(", ".join(map(str, DEFAULT_CASES))))
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                         help="number of seeds to build per case (default {0})".format(N_SEEDS))
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--noptmax", type=int, default=None,
                         help="noptmax applied uniformly to every seed; defaults to matching each "
                              "seed's completed ../sqp run (falls back to a per-case default)")
    args = parser.parse_args()

    for n in (args.n_decvars if args.n_decvars else list(DEFAULT_CASES)):
        build_case(n, n_seeds=args.n_seeds, base_seed=args.base_seed, noptmax=args.noptmax)

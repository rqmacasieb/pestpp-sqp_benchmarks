"""
Build a multi-seed run of G07 (the classic 10-decision-variable,
8-inequality-constraint nonlinear test problem) for PESTPP-OPT (sequential
linear programming with linearized chance constraints).

G07 has no varying dimensionality to sweep, so this builds exactly one case
("g07"), fanned out into N_SEEDS independent runs, one per starting point:

    g07/opt/g07/lhs_starting_values.csv
    g07/opt/g07/seed1/template/ ...
    g07/opt/g07/seed2/template/ ...
    ...

Reuses ../sqp's per-seed LHS starting-value design verbatim (loaded from
../sqp/g07/lhs_starting_values.csv if that case has already been built
there, else regenerated with the same LHS_SEED, which is reproducibly
identical) -- so seed s's PESTPP-OPT decision-variable parval1 values are
exactly seed s's PESTPP-SQP starting point.

There are no adjustable parameters here other than the decision variables
themselves (no separate calibration dataset), so opt_risk is left at its
default of 0.5 (risk-neutral) -- this skips FOSM/chance-constraint
calculations entirely and keeps the model-run count at n_decvars + 1 = 11
per SLP iteration (one base run plus one finite-difference run per decision
variable), plus one final run.

Starting from ../template/g07.pst (the already-tested PESTPP-SQP template),
this script strips every sqp_* pestpp_option and replaces them with
opt_obj_func/opt_direction/opt_risk, and zero-weights the "obj" observation
-- a non-zero weight marks an observation as part of PESTPP-OPT's notional
calibration dataset (used to build the FOSM J matrix for chance
constraints); the objective observation can't be both that and the
(opt_obj_func-referenced) objective, so it must be zero-weighted. The
constraint (g1..g8) observations keep weight=1.0 and their existing
l_constraint group (already the correct "less than" sense for PESTPP-OPT --
see ../template/forward_run.py's G07 constraint formulas, all <= 0).

noptmax: by default it matches that *same seed's* completed PESTPP-SQP run.
If ../sqp/g07/seed{s}/{master,template}/g07.rec exists, its last "number of
model runs:" line sets noptmax = round(sqp_total_runs / (n_decvars + 1)), so
OPT spends roughly the same model-run budget as SQP. If that record isn't
found for a seed, the template's own noptmax is used instead (with a
warning). Pass --noptmax to set one value for every seed and skip the
matching entirely.

An LHS-drawn starting value could in principle land very close to 0 for some
decision variable -- a "relative" derivative increment (the template's
default) would shrink toward 0 there too, and PESTPP-OPT genuinely depends
on classic finite-difference derinc to build its response matrix; this
script switches parameter_groups to an "absolute" increment so it's
independent of parval1.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-opt binary (on your PATH, next to run_case.py, or passed to it with --exe).

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
import re
import argparse
import shutil
import numpy as np
import pandas as pd
import pyemu
from scipy import stats
from scipy.spatial.distance import pdist

OPT_DIR = os.path.dirname(os.path.abspath(__file__))
G07_DIR = os.path.dirname(OPT_DIR)
TEMPLATE_DIR = os.path.join(G07_DIR, "template")
TEMPLATE_PST = os.path.join(TEMPLATE_DIR, "g07.pst")
SQP_DIR = os.path.join(G07_DIR, "sqp")
CASE_NAME = "g07"
N_DECVARS = 10
PARNAMES = ["x{0}".format(i + 1) for i in range(N_DECVARS)]

PARLBND = -10.0
PARUBND = 10.0

N_SEEDS = 50
BASE_SEED = 1  # seeds run BASE_SEED .. BASE_SEED + N_SEEDS - 1
LHS_SEED = 42  # must match ../sqp/build_case.py's LHS_SEED -- keeps starting points identical

_MODEL_RUNS_RE = re.compile(r"number of model runs:\s*(\d+)")


def generate_starting_values(n_samples, n_dimensions, bounds, seed, iterations=1000):
    """Maximin Latin-Hypercube design: n_samples points in n_dimensions,
    scaled to per-dimension bounds. Reproducible from seed alone. (Same
    design as ../sqp/build_case.py.)"""
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


def _sqp_total_model_runs(seed):
    """Total model runs of the matching PESTPP-SQP seed run (last 'number of
    model runs:' line in its .rec), used to size noptmax so this seed's
    PESTPP-OPT run spends roughly the same model-run budget. Returns None if
    that SQP run hasn't been done."""
    for sub in ("master", "template"):
        rec_file = os.path.join(SQP_DIR, CASE_NAME, "seed{0}".format(seed), sub,
                                 "{0}.rec".format(CASE_NAME))
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


def load_or_generate_lhs_starting_values(case_dir, seeds, lhs_seed=LHS_SEED):
    """Reuse ../sqp's LHS starting-value design verbatim if it's already
    been built (guarantees identical starting points across suites
    regardless of RNG/library-version drift); otherwise generate a fresh
    design with the same lhs_seed, which is reproducibly identical to what
    ../sqp/build_case.py would produce anyway."""
    os.makedirs(case_dir, exist_ok=True)
    sqp_csv = os.path.join(SQP_DIR, CASE_NAME, "lhs_starting_values.csv")
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

    bounds = [[PARLBND, PARUBND]] * N_DECVARS
    samples = generate_starting_values(len(seeds), N_DECVARS, bounds, seed=lhs_seed)
    df = pd.DataFrame(samples, index=list(seeds), columns=PARNAMES)
    df.index.name = "seed"
    df.to_csv(dest)
    return df, dest


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
        pst = pyemu.Pst(TEMPLATE_PST)

        par = pst.parameter_data
        for name, val in zip(PARNAMES, init_vals):
            par.loc[name, "parval1"] = val

        # an LHS-drawn starting value could in principle land very close to
        # 0 for some decision variable -- a "relative" derivative increment
        # would shrink toward 0 there too, and PESTPP-OPT genuinely depends
        # on classic finite-difference derinc to build its response matrix;
        # use an absolute increment so it's independent of parval1
        pst.parameter_groups.loc[:, "inctyp"] = "absolute"

        # drop every sqp_* option from the template and replace with the
        # PESTPP-OPT equivalents
        for k in list(pst.pestpp_options):
            if k.startswith("sqp_"):
                del pst.pestpp_options[k]
        pst.pestpp_options["opt_obj_func"] = "obj"
        pst.pestpp_options["opt_direction"] = "min"
        pst.pestpp_options["opt_risk"] = 0.5
        pst.pestpp_options["random_seed"] = seed

        # a non-zero weight marks an observation as part of PESTPP-OPT's
        # notional calibration dataset (used to build the FOSM J matrix for
        # chance constraints); the objective observation can't be both that
        # and the (opt_obj_func-referenced) objective, so it must be
        # zero-weighted (g1..g8 keep weight=1.0 and their l_constraint group)
        pst.observation_data.loc["obj", "weight"] = 0.0

        pst.control_data.noptmax = noptmax

        pst_name = "{0}.pst".format(CASE_NAME)
        pst.write(pst_name)
    finally:
        os.chdir(cwd)

    # PESTPP-OPT's per-SLP-iteration response matrix needs one base run plus
    # one finite-difference run per decision variable (risk-neutral, no
    # separate FOSM J matrix here)
    return seed_template_dir, pst_name, N_DECVARS + 1


def build_case(n_seeds=N_SEEDS, base_seed=BASE_SEED, noptmax=None):
    case_dir = os.path.join(OPT_DIR, CASE_NAME)

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = load_or_generate_lhs_starting_values(case_dir, seeds)

    template_noptmax = pyemu.Pst(TEMPLATE_PST).control_data.noptmax
    explicit_noptmax = noptmax is not None
    unmatched = []
    num_runs_per_iter = None
    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        seed_noptmax = noptmax
        if not explicit_noptmax:
            sqp_total_runs = _sqp_total_model_runs(seed)
            if sqp_total_runs is None:
                unmatched.append(seed)
                seed_noptmax = template_noptmax
            else:
                seed_noptmax = max(1, round(sqp_total_runs / (N_DECVARS + 1)))
        _, _, num_runs_per_iter = build_seed_case(seed, init_vals, case_dir, seed_noptmax)

    print("built case: {0} ({1} dec vars, {2} seeds)".format(case_dir, N_DECVARS, n_seeds))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  each seed: template dir {0}/seed<N>/template/, g07.pst; "
          "up to {1} parallel workers".format(case_dir, num_runs_per_iter))
    if unmatched:
        print("  WARNING: no completed ../sqp run found for {0} of {1} seeds{2} -- used the "
              "template's noptmax={3} for those. Run ../sqp first and rebuild to match "
              "the SQP model-run budget, or pass --noptmax.".format(
                  len(unmatched), n_seeds, " ({0})".format(unmatched) if len(unmatched) <= 10 else "",
                  template_noptmax))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build a multi-seed G07 PESTPP-OPT run")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                         help="number of seeds to build (default {0})".format(N_SEEDS))
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--noptmax", type=int, default=None,
                         help="noptmax applied uniformly to every seed; defaults to matching each "
                              "seed's completed ../sqp run (falls back to the template's noptmax)")
    args = parser.parse_args()

    build_case(n_seeds=args.n_seeds, base_seed=args.base_seed, noptmax=args.noptmax)

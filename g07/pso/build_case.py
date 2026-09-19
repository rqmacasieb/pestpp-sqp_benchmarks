"""
Build a multi-seed run of G07 (the classic 10-decision-variable,
8-inequality-constraint nonlinear test problem) for PESTPP-MOU
(single-objective, particle swarm optimization generator).

G07 has no varying dimensionality to sweep, so this builds exactly one case
("g07"), fanned out into N_SEEDS independent runs, laid out as

    g07/pso/g07/lhs_starting_values.csv
    g07/pso/g07/seed1/template/ ...
    g07/pso/g07/seed2/template/ ...
    ...

Unlike PESTPP-SQP (which perturbs an ensemble around one starting point),
PESTPP-MOU/PSO needs an actual *population*: it draws mou_population_size
members uniformly across parlbnd/parubnd and completely ignores the control
file's parval1 unless an explicit mou_dv_population_file is supplied. So each
seed here gets its own freshly-drawn (maximin LHS) population of
mou_population_size members across the template's [-10, 10] decision-variable
bounds, written to seed{s}/template/dv_pop.csv and wired in via
pst.pestpp_options["mou_dv_population_file"] -- with member 0 overwritten by
that *same seed's* exact SQP starting point (loaded from
../sqp/g07/lhs_starting_values.csv if present, else regenerated with the
same LHS_SEED). This keeps PESTPP-SQP and PESTPP-MOU/PSO seed-for-seed linked
(one shared point genuinely present in both) while still giving PSO a
properly disperse initial swarm.

Starting from ../template/g07.pst (the already-tested PESTPP-SQP template),
this script strips every sqp_* pestpp_option, renames the "obj" observation's
group from obj_fn to less_than_obj (PESTPP-MOU distinguishes objectives from
constraints via mou_objectives, but every objective/constraint group name
still needs a valid sense prefix -- "less"/"l_" or "greater"/"g_"; g1..g8
already use l_constraint, the correct sense for G07's all-<=0 constraints,
so they're untouched), and sets mou_generator/mou_objectives/
mou_population_size/mou_dv_population_file. Unlike PESTPP-OPT, PESTPP-MOU
has no FOSM/chance-constraint machinery, so "obj" keeps its weight=1.0.

mou_population_size matches ../sqp's sqp_num_reals (20, from the template) --
keeps the two optimizers' per-generation/iteration model-run budgets
comparable.

noptmax: by default it matches that *same seed's* completed PESTPP-SQP run.
If ../sqp/g07/seed{s}/{master,template}/g07.rec exists, its last "number of
model runs:" line sets noptmax = round(sqp_total_runs / mou_population_size).
If that record isn't found for a seed, the template's own noptmax is used
instead (with a warning). Pass --noptmax to set one value for every seed and
skip the matching entirely.

Requirements: python with numpy, scipy, pandas and pyemu, plus a compiled
pestpp-mou binary (on your PATH, next to run_case.py, or passed to it with --exe).

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

PSO_DIR = os.path.dirname(os.path.abspath(__file__))
G07_DIR = os.path.dirname(PSO_DIR)
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
# seed for each seed's own initial-swarm LHS draw -- offset well clear of
# LHS_SEED/BASE_SEED/random_seed ranges so the swarm draw isn't a near-repeat
# of any other draw made with the same seed integer; still fully
# reproducible from the seed alone
POP_LHS_SEED_BASE = 900000

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
    PESTPP-MOU/PSO run spends roughly the same model-run budget. Returns
    None if that SQP run hasn't been done."""
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
    been built; otherwise generate a fresh design with the same lhs_seed,
    reproducibly identical to what ../sqp/build_case.py would produce."""
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


def write_dv_population(seed_template_dir, num_reals, seed, sqp_start_vals,
                         pop_lhs_seed_base=POP_LHS_SEED_BASE):
    """Draw this seed's own maximin-LHS initial swarm (num_reals members
    across the template's decision-variable bounds) and overwrite member 0
    with sqp_start_vals -- that same seed's exact PESTPP-SQP starting
    point."""
    bounds = [[PARLBND, PARUBND]] * N_DECVARS
    pop_seed = pop_lhs_seed_base + N_DECVARS * 1000 + seed
    samples = generate_starting_values(num_reals, N_DECVARS, bounds, seed=pop_seed)
    samples[0] = sqp_start_vals

    member_names = ["gen=0_member={0}".format(i) for i in range(num_reals)]
    df = pd.DataFrame(samples, index=member_names, columns=PARNAMES)
    df.index.name = "real_name"
    path = os.path.join(seed_template_dir, "dv_pop.csv")
    df.to_csv(path)
    return path


def build_seed_case(seed, init_vals, case_dir, noptmax, num_reals):
    seed_dir = os.path.join(case_dir, "seed{0}".format(seed))
    seed_template_dir = os.path.join(seed_dir, "template")
    if os.path.exists(seed_template_dir):
        shutil.rmtree(seed_template_dir)
    os.makedirs(seed_template_dir)

    for fname in ("par.tpl", "obs.ins", "forward_run.py"):
        shutil.copy(os.path.join(TEMPLATE_DIR, fname), os.path.join(seed_template_dir, fname))

    # PESTPP-MOU/PSO ignores parval1 for population init -- it needs a real
    # dv_pop_file to get a controlled, reproducible initial swarm instead of
    # silently falling back to its own internal uniform-random draw
    dv_pop_path = write_dv_population(seed_template_dir, num_reals, seed, init_vals)
    dv_pop_fname = os.path.basename(dv_pop_path)

    cwd = os.getcwd()
    os.chdir(seed_template_dir)
    try:
        pst = pyemu.Pst(TEMPLATE_PST)

        par = pst.parameter_data
        for name, val in zip(PARNAMES, init_vals):
            par.loc[name, "parval1"] = val

        pst.parameter_groups.loc[:, "inctyp"] = "absolute"

        # drop every sqp_* option from the template, plus opt_obj_func
        # (PESTPP-SQP/OPT's single-objective option -- PESTPP-MOU reads
        # mou_objectives instead), and replace with the PESTPP-MOU equivalents
        for k in list(pst.pestpp_options):
            if k.startswith("sqp_") or k == "opt_obj_func":
                del pst.pestpp_options[k]

        # PESTPP-MOU distinguishes objectives from constraints via
        # mou_objectives, but every objective/constraint group name still
        # needs a valid sense prefix ("less"/"l_" or "greater"/"g_") -- g1..g8
        # already use l_constraint (correct for G07's all-<=0 constraints)
        # and need no change; only "obj"'s group needs renaming
        pst.observation_data.loc["obj", "obgnme"] = "less_than_obj"

        pst.pestpp_options["mou_generator"] = "pso"
        pst.pestpp_options["mou_objectives"] = "obj"
        pst.pestpp_options["mou_population_size"] = num_reals
        pst.pestpp_options["mou_dv_population_file"] = dv_pop_fname
        pst.pestpp_options["random_seed"] = seed

        pst.control_data.noptmax = noptmax

        pst_name = "{0}.pst".format(CASE_NAME)
        pst.write(pst_name)
    finally:
        os.chdir(cwd)

    return seed_template_dir, pst_name


def build_case(n_seeds=N_SEEDS, base_seed=BASE_SEED, noptmax=None, num_reals=None):
    case_dir = os.path.join(PSO_DIR, CASE_NAME)

    template_pst = pyemu.Pst(TEMPLATE_PST)
    # match ../sqp's ensemble size for the same case -- keeps the two
    # optimizers' per-generation/iteration model-run budgets comparable
    if num_reals is None:
        num_reals = int(template_pst.pestpp_options.get("sqp_num_reals", 20))
    template_noptmax = template_pst.control_data.noptmax

    seeds = list(range(base_seed, base_seed + n_seeds))
    lhs_df, lhs_path = load_or_generate_lhs_starting_values(case_dir, seeds)

    explicit_noptmax = noptmax is not None
    unmatched = []
    for seed in seeds:
        init_vals = lhs_df.loc[seed].values
        seed_noptmax = noptmax
        if not explicit_noptmax:
            sqp_total_runs = _sqp_total_model_runs(seed)
            if sqp_total_runs is None:
                unmatched.append(seed)
                seed_noptmax = template_noptmax
            else:
                seed_noptmax = max(1, round(sqp_total_runs / num_reals))
        build_seed_case(seed, init_vals, case_dir, seed_noptmax, num_reals)

    print("built case: {0} ({1} dec vars, {2} seeds, mou_population_size={3})".format(
        case_dir, N_DECVARS, n_seeds, num_reals))
    print("  LHS starting values: {0}".format(lhs_path))
    print("  each seed: template dir {0}/seed<N>/template/, g07.pst; "
          "up to {1} parallel workers".format(case_dir, num_reals))
    if unmatched:
        print("  WARNING: no completed ../sqp run found for {0} of {1} seeds{2} -- used the "
              "template's noptmax={3} for those. Run ../sqp first and rebuild to match "
              "the SQP model-run budget, or pass --noptmax.".format(
                  len(unmatched), n_seeds, " ({0})".format(unmatched) if len(unmatched) <= 10 else "",
                  template_noptmax))
    return case_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build a multi-seed G07 PESTPP-MOU/PSO run")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                         help="number of seeds to build (default {0})".format(N_SEEDS))
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--noptmax", type=int, default=None,
                         help="noptmax applied uniformly to every seed; defaults to matching each "
                              "seed's completed ../sqp run (falls back to the template's noptmax)")
    args = parser.parse_args()

    build_case(n_seeds=args.n_seeds, base_seed=args.base_seed, noptmax=args.noptmax)

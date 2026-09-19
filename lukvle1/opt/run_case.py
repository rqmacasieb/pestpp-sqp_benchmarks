"""
Run the LUKVLE1 PESTPP-OPT cases built by build_case.py on this computer.

Each seed of a case is one PESTPP-OPT run: a master plus a group of parallel
workers, all started locally with pyemu.os_utils.start_workers. For seed N of
case lukvle1_20d this

    lukvle1_20d/seedN/template/  ->  copied to lukvle1_20d/seedN/master/   (the master's
                                                              dir, and where the
                                                              results end up)
                            ->  copied to lukvle1_20d/seedN/worker_0/ ... worker_<k-1>/
                                (removed again when the run finishes)

and writes everything the master and workers print to lukvle1_20d/seedN/run.log.

Run build_case.py first. You also need the compiled `pestpp-opt` binary,
either on your PATH, next to this script, or passed with --exe.

Usage:
    python run_case.py 20 --seeds 1 2 3       # these seeds of the 20-variable case
    python run_case.py 20 --concurrent 4      # every built seed, 4 at a time
    python run_case.py                         # every built seed of every built case
    python run_case.py 20 --exe /path/to/pestpp-opt --num_workers 8

Parallelism: each seed uses --num_workers parallel workers (default: as many as
the run can use -- n decision variables + 1 -- but no more than your cores divided by
--concurrent). With --concurrent > 1 the cores are shared between the seeds
running at once. Cases are run one after another.

A seed that already has a master/ directory is skipped, so re-running this
script resumes where it stopped; pass --overwrite to redo (and delete the
results of) those seeds. Note that a seed interrupted part-way also has a
master/ directory, so use --overwrite for it.
"""
import os
import re
import sys
import shutil
import argparse
import multiprocessing as mp
import pyemu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_FMT = "lukvle1_{0}d"
EXE_NAME = "pestpp-opt"
BASE_PORT = 4500  # seed N of an n-variable case listens on BASE_PORT + 100 * n + N (unique per case and seed)


def case_dir_for(n_decvars):
    return os.path.join(SCRIPT_DIR, CASE_FMT.format(n_decvars))


def find_seeds(n_decvars):
    seeds = []
    case_dir = case_dir_for(n_decvars)
    pst_name = CASE_FMT.format(n_decvars) + ".pst"
    if os.path.isdir(case_dir):
        for name in os.listdir(case_dir):
            if name.startswith("seed") and name[4:].isdigit() \
                    and os.path.exists(os.path.join(case_dir, name, "template", pst_name)):
                seeds.append(int(name[4:]))
    return sorted(seeds)


def find_cases():
    pat = re.compile("^" + CASE_FMT.format("(\\d+)") + "$")
    cases = []
    for name in os.listdir(SCRIPT_DIR):
        m = pat.match(name)
        if m and find_seeds(int(m.group(1))):
            cases.append(int(m.group(1)))
    return sorted(cases)


def find_exe(exe):
    """Resolve the pestpp binary to an absolute path: an explicit --exe, else
    one on PATH, else one sitting next to this script."""
    if exe is not None:
        found = shutil.which(exe) or (os.path.abspath(exe) if os.path.exists(exe) else None)
    else:
        found = shutil.which(EXE_NAME)
        local = os.path.join(SCRIPT_DIR, EXE_NAME)
        if found is None and os.path.exists(local):
            found = local
    if found is None:
        sys.exit("could not find {0} -- put it on your PATH, next to this script, or pass "
                 "--exe".format(exe or EXE_NAME))
    return os.path.abspath(found)


def workers_needed(n_decvars, seed):
    """Most parallel model runs the algorithm can use at once for this case."""
    case_name = CASE_FMT.format(n_decvars)
    pst = pyemu.Pst(os.path.join(case_dir_for(n_decvars), "seed{0}".format(seed), "template",
                                  case_name + ".pst"))
    return pst.npar_adj + 1


def run_seed(n_decvars, seed, exe, num_workers, port):
    """Run one seed's master + workers, sending all output to seedN/run.log.
    Returns (n_decvars, seed, error message or None)."""
    seed_dir = os.path.join(case_dir_for(n_decvars), "seed{0}".format(seed))
    pst_name = CASE_FMT.format(n_decvars) + ".pst"
    cwd = os.getcwd()
    saved = [os.dup(1), os.dup(2)]
    err = None
    try:
        with open(os.path.join(seed_dir, "run.log"), "w") as log:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log.fileno(), 1)  # child processes inherit the log too
            os.dup2(log.fileno(), 2)
            try:
                pyemu.os_utils.start_workers(
                    os.path.join(seed_dir, "template"), exe, pst_name,
                    num_workers=num_workers, worker_root=seed_dir, port=port,
                    master_dir=os.path.join(seed_dir, "master"), verbose=True)
            except Exception as e:
                err = str(e)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
    finally:
        for fd in saved:
            os.close(fd)
        os.chdir(cwd)
    return n_decvars, seed, err


def _run_seed_star(args):
    return run_seed(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the LUKVLE1 PESTPP-OPT cases locally")
    parser.add_argument("n_decvars", nargs="*", type=int, default=None,
                         help="which case(s) to run, e.g. 20 (default: every built case)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="seeds to run (default: every seed built for the case)")
    parser.add_argument("--concurrent", type=int, default=1,
                         help="number of seeds to run at the same time (default 1)")
    parser.add_argument("--num_workers", type=int, default=None,
                         help="parallel workers per seed (default: min(what the run can use, "
                              "cores // concurrent))")
    parser.add_argument("--exe", default=None, help="path to the {0} binary".format(EXE_NAME))
    parser.add_argument("--base_port", type=int, default=BASE_PORT,
                         help="seed N of an n-variable case uses port base_port + 100 * n + N "
                              "(default base {0})".format(BASE_PORT))
    parser.add_argument("--overwrite", action="store_true",
                         help="rerun seeds that already have a master/ directory, deleting it")
    args = parser.parse_args()

    cases = args.n_decvars if args.n_decvars else find_cases()
    if not cases:
        sys.exit("no built cases found under {0} -- run build_case.py first".format(SCRIPT_DIR))
    exe = find_exe(args.exe)

    failed = []
    for n in cases:
        built = find_seeds(n)
        if not built:
            sys.exit("case {0} has not been built -- run build_case.py {1} first".format(
                CASE_FMT.format(n), n))
        seeds = args.seeds if args.seeds is not None else built
        missing = [s for s in seeds if s not in built]
        if missing:
            sys.exit("{0}: seed(s) {1} have not been built -- run build_case.py first".format(
                CASE_FMT.format(n), missing))

        todo = []
        for s in seeds:
            if os.path.exists(os.path.join(case_dir_for(n), "seed{0}".format(s), "master")) \
                    and not args.overwrite:
                print("{0} seed {1}: master/ already exists -- skipping (use --overwrite to "
                      "redo)".format(CASE_FMT.format(n), s))
            else:
                todo.append(s)
        if not todo:
            continue

        concurrent = max(1, min(args.concurrent, len(todo)))
        num_workers = args.num_workers
        if num_workers is None:
            num_workers = max(1, min(workers_needed(n, todo[0]), (os.cpu_count() or 1) // concurrent))
        print("{0}: running seeds {1} with {2}: {3} at a time, {4} workers each".format(
            CASE_FMT.format(n), todo, exe, concurrent, num_workers))

        jobs = [(n, s, exe, num_workers, args.base_port + 100 * n + s) for s in todo]
        # separate processes (not threads) because start_workers changes the working directory
        with mp.get_context("spawn").Pool(concurrent) as pool:
            for n_dv, seed, err in pool.imap_unordered(_run_seed_star, jobs):
                tag = "{0} seed {1}".format(CASE_FMT.format(n_dv), seed)
                if err is None:
                    print("{0}: done  ({1})".format(
                        tag, os.path.join(case_dir_for(n_dv), "seed{0}".format(seed), "master")))
                else:
                    print("{0}: FAILED -- {1} (see seed{2}/run.log)".format(tag, err, seed))
                    failed.append((n_dv, seed))

    if failed:
        sys.exit("failed: {0}".format(", ".join("{0} seed {1}".format(CASE_FMT.format(n), s)
                                                for n, s in failed)))
    print("all done")

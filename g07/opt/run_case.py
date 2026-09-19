"""
Run the G07 PESTPP-OPT cases built by build_case.py on this computer.

Each seed is one PESTPP-OPT run: a master plus a group of parallel workers, all
started locally with pyemu.os_utils.start_workers. For seed N this

    g07/seedN/template/   ->  copied to g07/seedN/master/   (the master's dir, and
                                                            where the results end up)
                          ->  copied to g07/seedN/worker_0/ ... worker_<k-1>/
                              (removed again when the run finishes)

and writes everything the master and workers print to g07/seedN/run.log.

Run build_case.py first. You also need the compiled `pestpp-opt` binary, either
on your PATH or passed with --exe.

Usage:
    python run_case.py                       # every built seed, one after another
    python run_case.py --seeds 1 2 3         # only these seeds
    python run_case.py --concurrent 4        # run 4 seeds at the same time
    python run_case.py --exe /path/to/pestpp-opt --num_workers 8

Parallelism: each seed uses --num_workers parallel workers (default: as many as
the run can use -- n decision variables + 1 = 11 -- but no more than your cores divided by
--concurrent). With --concurrent > 1 the cores are shared between the seeds
running at once.

A seed that already has a master/ directory is skipped, so re-running this
script resumes where it stopped; pass --overwrite to redo (and delete the
results of) those seeds. Note that a seed interrupted part-way also has a
master/ directory, so use --overwrite for it.
"""
import os
import sys
import shutil
import argparse
import multiprocessing as mp
import pyemu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_NAME = "g07"
CASE_DIR = os.path.join(SCRIPT_DIR, CASE_NAME)
PST_NAME = "{0}.pst".format(CASE_NAME)
EXE_NAME = "pestpp-opt"
BASE_PORT = 15500  # seed N's master listens on BASE_PORT + N (unique per seed so concurrent seeds don't collide)


def find_seeds(case_dir):
    seeds = []
    if os.path.isdir(case_dir):
        for name in os.listdir(case_dir):
            if name.startswith("seed") and name[4:].isdigit() \
                    and os.path.exists(os.path.join(case_dir, name, "template", PST_NAME)):
                seeds.append(int(name[4:]))
    return sorted(seeds)


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


def workers_needed(seed):
    """Most parallel model runs the algorithm can use at once for this seed."""
    pst = pyemu.Pst(os.path.join(CASE_DIR, "seed{0}".format(seed), "template", PST_NAME))
    return pst.npar_adj + 1


def run_seed(seed, exe, num_workers, port):
    """Run one seed's master + workers, sending all output to seedN/run.log.
    Returns (seed, error message or None)."""
    seed_dir = os.path.join(CASE_DIR, "seed{0}".format(seed))
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
                    os.path.join(seed_dir, "template"), exe, PST_NAME,
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
    return seed, err


def _run_seed_star(args):
    return run_seed(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the G07 PESTPP-OPT cases locally")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="seeds to run (default: every seed found under {0}/)".format(CASE_NAME))
    parser.add_argument("--concurrent", type=int, default=1,
                         help="number of seeds to run at the same time (default 1)")
    parser.add_argument("--num_workers", type=int, default=None,
                         help="parallel workers per seed (default: min(what the run can use, "
                              "cores // concurrent))")
    parser.add_argument("--exe", default=None, help="path to the {0} binary".format(EXE_NAME))
    parser.add_argument("--base_port", type=int, default=BASE_PORT,
                         help="seed N's master uses port base_port + N (default {0})".format(BASE_PORT))
    parser.add_argument("--overwrite", action="store_true",
                         help="rerun seeds that already have a master/ directory, deleting it")
    args = parser.parse_args()

    built = find_seeds(CASE_DIR)
    if not built:
        sys.exit("no built seeds found under {0} -- run build_case.py first".format(CASE_DIR))
    seeds = args.seeds if args.seeds is not None else built
    missing = [s for s in seeds if s not in built]
    if missing:
        sys.exit("seed(s) {0} have not been built -- run build_case.py first".format(missing))

    todo = []
    for s in seeds:
        if os.path.exists(os.path.join(CASE_DIR, "seed{0}".format(s), "master")) and not args.overwrite:
            print("seed {0}: master/ already exists -- skipping (use --overwrite to redo)".format(s))
        else:
            todo.append(s)
    if not todo:
        sys.exit("nothing to run")

    exe = find_exe(args.exe)
    concurrent = max(1, min(args.concurrent, len(todo)))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, min(workers_needed(todo[0]), (os.cpu_count() or 1) // concurrent))
    print("running seeds {0} with {1}: {2} at a time, {3} workers each".format(
        todo, exe, concurrent, num_workers))

    jobs = [(s, exe, num_workers, args.base_port + s) for s in todo]
    failed = []
    # separate processes (not threads) because start_workers changes the working directory
    with mp.get_context("spawn").Pool(concurrent) as pool:
        for seed, err in pool.imap_unordered(_run_seed_star, jobs):
            if err is None:
                print("seed {0}: done  ({1})".format(
                    seed, os.path.join(CASE_DIR, "seed{0}".format(seed), "master")))
            else:
                print("seed {0}: FAILED -- {1} (see seed{0}/run.log)".format(seed, err))
                failed.append(seed)

    if failed:
        sys.exit("failed seeds: {0}".format(failed))
    print("all {0} seed(s) finished".format(len(todo)))

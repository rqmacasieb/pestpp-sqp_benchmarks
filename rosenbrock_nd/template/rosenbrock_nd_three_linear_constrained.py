import os


def evaluate(x):
    """x: sequence of n decision-variable values, ordered x1..xn. Returns
    (objective, constraint1, constraint2, constraint3) -- same Rosenbrock-sum
    + pairwise-block-linear-constraint math the file-based path always used,
    factored out here so ppw_worker can reuse it without a
    par.dat/obs.dat/constraints.dat file round trip."""
    n = len(x)
    # see: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.rosen.html
    result = sum(100.0*(x[i+1] - x[i]**2.0)**2.0 + (1.0 - x[i])**2.0 for i in range(n - 1))

    # 3 constraints regardless of N: pairwise-block replication of the 2D
    # three-linear-constraint case, summed over consecutive (x[2k], x[2k+1]) blocks
    constraint1 = 0.0
    constraint2 = 0.0
    constraint3 = 0.0
    for k in range(0, n, 2):
        a, b = x[k], x[k + 1]
        constraint1 += -2.25*a + b
        constraint2 += a + 1.5*b
        constraint3 += b - 0.5*a

    return result, constraint1, constraint2, constraint3


def helper():
    """File-based forward run: read par.dat (one line of whitespace-
    separated values, in x1..xn order -- see build_case_local.py's
    par.dat.tpl field layout), write obs.dat/constraints.dat in the exact
    format obs.dat.ins/constraints.dat.ins expect. This is exactly what
    running this file as a script always did."""
    with open(os.path.join("par.dat"), 'r') as f:
        x = [float(v) for v in f.readline().strip().split()]

    result, constraint1, constraint2, constraint3 = evaluate(x)

    with open(os.path.join("obs.dat"), 'w') as f:
        f.write("{0:20.8E}\n".format(result))

    with open(os.path.join("constraints.dat"), 'w') as f:
        f.write("{0:20.8E}\n".format(constraint1))
        f.write("{0:20.8E}\n".format(constraint2))
        f.write("{0:20.8E}\n".format(constraint3))


def ppw_worker(pst_name, host, port):
    """In-memory PyPestWorker loop -- mirrors
    ../../g07/template/forward_run.py's ppw_worker exactly, reusing
    evaluate() above instead of the file-based helper() so no
    par.dat/obs.dat/constraints.dat round trip is needed per realization."""
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name, host, port, verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return

    obs = ppw._pst.observation_data.copy()
    obs = obs.loc[ppw.obs_names, "obsval"]

    while True:
        # don't trust pvals.index's natural order -- sort numerically by
        # the xN suffix, same defensive re-sort g07/template/forward_run.py
        # uses for its own x1..x10 parameters
        parnames = sorted(pvals.index, key=lambda name: int(name[1:]))
        x = [pvals[name] for name in parnames]
        result, constraint1, constraint2, constraint3 = evaluate(x)

        sim = {"obs": result, "constraint1": constraint1,
               "constraint2": constraint2, "constraint3": constraint3}
        obs.update(sim)

        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    helper()

import os

with open(os.path.join("par.dat"), 'r') as f:
    x = [float(v) for v in f.readline().strip().split()]

n = len(x)
# Powell's singular function: pairwise-block-of-4 replication -- apply the
# classic 4-variable Powell formula to each consecutive block
# (x[4k], x[4k+1], x[4k+2], x[4k+3]) and sum across blocks. See:
# https://www.sfu.ca/~ssurjano/powell.html
result = 0.0
for k in range(0, n, 4):
    x1, x2, x3, x4 = x[k], x[k + 1], x[k + 2], x[k + 3]
    result += (x1 + 10.0 * x2) ** 2.0 + 5.0 * (x3 - x4) ** 2.0 \
        + (x2 - 2.0 * x3) ** 4.0 + 10.0 * (x1 - x4) ** 4.0

# 3 constraints regardless of N: block-of-4 replication (matching the
# objective's own decomposition), summed over consecutive blocks -- all
# linear (unlike the 4d base case's nonlinear constraint1). Dropping the
# x1**2 term keeps constraint1 a well-behaved linear constraint even when
# summed across many blocks -- same simplification used by
# ../../rosenbrock/template/rosenbrock_nd_three_linear_constrained.py, which
# likewise drops the nonlinear term present in its 2-par base case. All three
# still evaluate to exactly 0 at the true unconstrained optimum x=0; the
# RHS values are chosen (see build_nd_cases*.py) so x=0 stays infeasible,
# forcing a nontrivial constrained optimum, without requiring an
# actively-binding *nonlinear* boundary to be held across every block at
# once -- that combination is what made high-dimensional cases numerically
# fragile (ensemble/CMA feasibility-seeking could stall and eventually
# produce denormal parameter values).
constraint1 = 0.0  # linear, less-than
constraint2 = 0.0  # linear, less-than
constraint3 = 0.0  # linear, greater-than
for k in range(0, n, 4):
    x1, x2, x3, x4 = x[k], x[k + 1], x[k + 2], x[k + 3]
    constraint1 += x2 - x3
    constraint2 += x2 + 2.0 * x3 - x4
    constraint3 += x1 - x4

with open(os.path.join("obs.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(result))

with open(os.path.join("constraints.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(constraint1))
    f.write("{0:20.8E}\n".format(constraint2))
    f.write("{0:20.8E}\n".format(constraint3))

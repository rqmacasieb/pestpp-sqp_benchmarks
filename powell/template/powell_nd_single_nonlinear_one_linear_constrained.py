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

# 2 constraints regardless of N:
#   constraint1 -- the classic textbook Powell nonlinear constraint (a
#     squared term on the first block's x1), taken from block 0 (x1, x2, x3)
#     only -- not replicated/summed across every block. Nonlinear and
#     actively binding: x=0 is infeasible, so the constrained optimum sits
#     on this boundary, and the curvature ensemble-SQP has to estimate for
#     it never grows with N (see
#     powell_nd_single_active_nonlinear_constrained.py's docstring for why
#     summing a squared term over every block was numerically fragile).
#   constraint2 -- linear, block-of-4 summed over ALL blocks (same pattern as
#     powell_nd_three_linear_constrained.py's constraint2), RHS set loose
#     (satisfied at x=0 and at the standard start point) so it stays
#     non-binding -- constraint1 carries the active-constraint story here.
constraint1 = x[0] ** 2.0 + x[1] - x[2]  # nonlinear, block 0 only
constraint2 = 0.0  # linear, less-than, all blocks
for k in range(0, n, 4):
    x1, x2, x3, x4 = x[k], x[k + 1], x[k + 2], x[k + 3]
    constraint2 += x2 + 2.0 * x3 - x4

with open(os.path.join("obs.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(result))

with open(os.path.join("constraints.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(constraint1))
    f.write("{0:20.8E}\n".format(constraint2))

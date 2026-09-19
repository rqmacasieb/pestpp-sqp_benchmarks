import os
import math

with open(os.path.join("par.dat"), 'r') as f:
    x = [float(v) for v in f.readline().strip().split()]

n = len(x)
# Luksan-Vlcek Problem 1 (LUKVLE1): chained Rosenbrock objective
# see: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.rosen.html
result = sum(100.0 * (x[i] ** 2.0 - x[i + 1]) ** 2.0 + (x[i] - 1.0) ** 2.0 for i in range(n - 1))

# n-2 nonlinear equality constraints coupling each variable triplet through
# trig, exponential, and cubic terms
constraints = []
for i in range(n - 2):
    xi, xi1, xi2 = x[i], x[i + 1], x[i + 2]
    c = (3.0 * xi1 ** 3.0 + 2.0 * xi2 - 5.0
         + math.sin(xi1 - xi2) * math.sin(xi1 + xi2)
         + 4.0 * xi1 - xi * math.exp(xi - xi1) - 3.0)
    constraints.append(c)

with open(os.path.join("obs.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(result))

with open(os.path.join("constraints.dat"), 'w') as f:
    for c in constraints:
        f.write("{0:20.8E}\n".format(c))

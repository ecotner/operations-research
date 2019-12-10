# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # LP and MIP

# ## Test out `cvxpy` and linear programming

import cvxpy as cvx
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

cvx.installed_solvers()

# +
n = 10
m = 5
np.random.seed(1)
A = np.random.randn(n, m)
b = np.random.randn(n)

x = cvx.Variable(m)
constraints = [
    0 <= x, x <= 1
]

obj = cvx.Minimize(cvx.sum_squares(A*x - b))

prob = cvx.Problem(obj, constraints)
prob.solve(solver=cvx.CVXOPT);
# -

print("Using CVXOPT:")
print("status:", prob.status)
print("optimal objective:", prob.value)
print(f"optimal vars - x={x.value}")

# ## Try out some MIP

# We'll solve a small traveling salesman problem using integer programming. We are given a set of $N$ locations $L \in \mathbb{R}^{N\times 2}$, with a distance matrix $D \in \mathbb{R}^{N\times N}$. We have a set of boolean decision variables $X \in \mathbb{R}^{N\times N}$ where the $X_{ij}$ entry says to travel from the $i$th location to the $j$th location. We will also introduce $N$ slack variables $1 \le u_i \le N-1$ which describes the visitation order for the $i$th location ($u_0$ is fixed to zero).
#
# We want to find the route that minimizes the total distance traveled, while still visiting all the locations, and returning back to the origin. This means we must solve the following problem:
#
# $$
# \min_X \sum_{i,j=1}^{N} D_{ij} X_{ij} \\
# \text{subject to} \\
# \sum_{i=1,i\neq j}^{N} X_{ij} = 1, \qquad
# \sum_{j=1,j\neq i}^{N} X_{ij} = 1, \qquad
# X_{ii} = 0, \\
# u_0 = 0,\,1 \le u_i \le N-1;\, i \in [1, N], \qquad
# u_i - u_j + NX_{ij} \le N-1;\, \forall i \neq j
# $$

# +
# Simple traveling salesman problem
from scipy.spatial.distance import pdist, squareform
N = 15    # number of locations
np.random.seed(1)
locations = np.random.randn(N, 2)    # (x,y) coordinates of the locations
D = squareform(pdist(locations))    # distance matrix between locations

# Location visitation variables; 
X = cvx.Variable((N, N), boolean=True)
u = cvx.Variable(N-1, integer=True)

# Constraints
constraints = [
    cvx.sum(X, axis=0) == 1,
    cvx.sum(X, axis=1) == 1,
    cvx.trace(X) == 0,
    1 <= u, u <= N-1
]
constraints.extend(
    [u[i-1] - u[j-1] + N*X[i,j] <= N-1 for i in range(1, N) for j in range(1, N) if i != j]
)

# Objective
obj = cvx.Minimize(cvx.sum(cvx.multiply(D, X)))

# Problem
prob = cvx.Problem(obj, constraints)
# %time prob.solve(solver=cvx.GLPK_MI, glpk={"tmlim": 1, "presolve": "GLP_ON"})
# -

print("Problem status:", prob.status)
print("Location index:", np.arange(N))
print("argmax(X):     ", np.argmax(X.value, axis=1))
print("Visitation order:", [0]+u.value.astype(int).tolist()+[0])

order = list(list(zip(*sorted(
    zip(
        list(range(N))+[0], 
        [0]+u.value.astype(int).tolist()+[N]
    ), 
    key=lambda e: e[1]
)))[0])
tour = locations[order]
plt.plot(*tour.T)
for i, t in enumerate(tour[:-1]):
    plt.scatter(*t, marker=f"${i}$", color='black', s=100*len(str(i)))
plt.title(f"Traveling salesman tour\nTotal distance: {prob.value}")
plt.show()

cvx.__version__



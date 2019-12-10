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

# # Testing MIPCL-PY package (and MIPCL solver!)

# Gonna try out some of the example problems/code in the MIPCL-PY documentation

import mipcl_py.mipshell.mipshell as mip


class CVP(mip.Problem):
    def model(self,B,v):
        self.v = v
        m, n = len(v),  len(B[0])
        self.x = x = mip.VarVector((n,),"x",mip.INT,lb=-mip.VAR_INF)
        self.y = y = mip.VarVector((m,),"y",lb=-mip.VAR_INF)
        self.t = t = mip.Var("t")

        mip.minimize(t)
        for i in range(m):
            mip.sum_(B[i][j]*x[j] for j in range(n)) + y[i] == v[i]
        mip.norm(y) <= t

    def printSolution(self):
        t, x, y, v = self.t, self.x, self.y, self.v
        m, n = len(y), len(x)

        print('Minimum distance = {:.4f}\n'.format(t.val))
        str = 'x = ({:d}'.format(int(x[0].val + 0.5))
        for j in range(1,n):
            str+=', {:d}'.format(int(x[j].val + 0.5))
        print(str+')\n')

        print(' _____________________________________')
        print('|         v        |         Bx       |')
        print('|------------------+------------------|')
        for i in range(m):
            print('| {:16.4f} | {:16.4f} |'.format(v[i],v[i]-y[i].val))
        print(' -------------------------------------')


# +
B = (
    (-11, 0, 2, -1, 1, 0, 9, -1),
    (-3, 0, 1, -3, -1, 1, 0, 0),
    (0, -2, 1, 2, 0, 1, 0, 1),
    (-1, -103, 0, 0, 0, 1, 0, 1),
    (0, 1, 0, 1, -3, 0, 1, -3),
    (-1, 0, -5, 3, 0, 0, 77, -8),
    (10, 2, 0, -1, 1, 8, 0, -1),
    (3, 0, -1, 914, 0, 1, 0, 0),
    (0, -2, 1, 2, 0, 1, 0, 1),
    (-97, 1, 0, 0, 0, 142, 0, 1),
    (2, -432, 0, 1, -3, 0, -6, -13),
    (4, 0, 5, 265, 0, 0, 12, -8)
)
v = (12, -1700, 424, -25, 191, 0, 12, 304, -19, 514, 0, 77)

prob = CVP("testCVP")
prob.model(B,v)
prob.optimize(False)
prob.printSolution()
# -



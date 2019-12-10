# Operations research

## Different packages
There are a huge number of different packages you can use to interface with LP (Linear Programming) and MILP (Mixed Integer LP) solvers. This is just an example of one's I've tested.

* `cvxpy`
    * Very simple to install: `pip install cvxpy`
    * Comes with a bunch of [different built-in solvers](https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options) by default that can handle a variety of problems, not just LP/MILP
        * LP, MILP, QP ([Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming)), SOCP ([Second-Order Cone Programming](https://en.wikipedia.org/wiki/Second-order_cone_programming)), SDP ([Semidefinite Programming](https://en.wikipedia.org/wiki/Semidefinite_programming)), and EXP ([EXponential cone Programming](https://yalmip.github.io/tutorial/exponentialcone/))
* `cvxopt`
    * Made by the same people who made `cvxpy`
    * Simple to install with `pip install cvxopt`
    *
* `mipcl-py`
    * Python interface to the MIPCL solver (i.e. it doesn't use any other backends)
    * Supposedly very fast more on that in next section
    * Installation is moderately more difficult, but not too hard; you can download the files [here](http://www.mipcl-cpp.appspot.com/download.html), and there is a short install guide [here](https://github.com/onebitbrain/MIPCL/blob/master/mipcl_py/README.txt) (which I will now paraphrase).
        * Download the `mipcl-py` file appropriate for your OS; this will download an `mipcl-py-X.X.XYYYYY.tar.gz` archive
        * Put it somewhere you want the install directory to be (I made a new directory `~/mipcl-py`)
        * Extract the archive with `tar -xzvf <file>`
        * Enter the mipshell directory `cd <install_dir>/mipcl_py/mipshell`
        * Make a symbolic link to the shared library
            * If you use Python3, type `ln -s mipcl-py3.so mipcl.so`
            * If you use Python2, type `ln -s mipcl-py2.so mipcl.so`
        * Modify the `PYTHONPATH` environment variable to include the `MIPCL-PY` module and examples:
            * On LINUX computers, if you use bash, add the following line to the end of .bashrc: `PYTHONPATH=<install_dir>; export PYTHONPATH`
        * Test that you can run `python -c "import mipcl_py"` without error
        

## Different solvers
There are a huge number of different solvers (CBC, Gurobi, CPLEX, Mosek, GLPK, GLOP, SCIP, etc.)
Here is a [comparison](http://plato.asu.edu/talks/ismp2018.pdf) of the performance of various different solvers. See page 11 for LP and page 20 for MILP solver comparison.
If you can get a free version of Gurobi or CPLEX, do that.
Commercial solvers are so much faster than the free ones it's ridiculous.
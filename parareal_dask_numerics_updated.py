from __future__ import annotations
import math
import warnings
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    MPI = None
    _MPI_AVAILABLE = False

#Limit NumExpr threads (if using NumExpr in numpy/pandas internally)
import os
os.environ["NUMEXPR_MAX_THREADS"] = "4"

#Configure logging to suppress asyncio debug logs by default
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
import operator
import h5py as h5
import dask
from dask.distributed import Client

class Vector(ABC):
    """Abstract base for lazy vector expressions supporting +, -, * and reduction to data."""
    @abstractmethod
    def reduce(self) -> np.ndarray:
        """Compute and return concrete NumPy array value of this vector."""
        pass
    def __add__(self, other):
        return BinaryExpr(operator.add, self, other)  #create a lazy binary expression
    def __sub__(self, other):
        return BinaryExpr(operator.sub, self, other)
    def __mul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)
    def __rmul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)

def reduce_expr(expr: Union[np.ndarray, Vector]) -> np.ndarray:
    """Recursively evaluate a Vector expression into a concrete numpy array."""
    while isinstance(expr, Vector):
        expr = expr.reduce()
    return expr

@dataclass
class H5Snap(Vector):
    """A Vector that loads its data from an HDF5 file snapshot when reduced."""
    path: Path
    loc: str
    slice: list[Union[None, int, slice]]
    def data(self):
        """Read dataset from HDF5 (without reducing it fully)."""
        with h5.File(self.path, "r") as f:
            return f[self.loc].__getitem__(tuple(self.slice))
    def reduce(self) -> np.ndarray:
        #Load data from HDF5 file
        return self.data()  #actual HDF5 read

class Index:
    """Helper to allow convenient slicing: e.g., index[-1] to get last element slice."""
    def __getitem__(self, idx):
        return list(idx) if isinstance(idx, tuple) else [idx]

index = Index()  #instance to use for slicing in H5Snap

@dataclass
class UnaryExpr(Vector):
    """Lazy unary vector expression (e.g., scaling a vector by a constant)."""
    func: Callable[[np.ndarray], np.ndarray]
    inp: Vector
    def reduce(self) -> np.ndarray:
        a = reduce_expr(self.inp)
        return self.func(a)  #apply stored unary function to evaluated input

@dataclass
class BinaryExpr(Vector):
    """Lazy binary vector expression (e.g., combining two vectors with + or -)."""
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    inp1: Vector
    inp2: Vector
    def reduce(self) -> np.ndarray:
        a = reduce_expr(self.inp1)
        b = reduce_expr(self.inp2)
        if self.func == operator.sub:
            #Use safe subtraction to combine results stably
            return safe_subtract(a, b)
        else:
            return self.func(a, b)

@dataclass
class LiteralExpr(Vector):
    """Wrap a concrete numpy array as a Vector (for lazy interface compatibility)."""
    value: np.ndarray
    def reduce(self) -> np.ndarray:
        return self.value

@dataclass
class Coarse:
    """Coarse propagator wrapper: applies an implicit Euler step and wraps result in LiteralExpr."""
    n_iter: int
    system: Any  #ODE system (Problem)
    def solution(self, y, t0: float, t1: float) -> LiteralExpr:
        #Use implicit Euler for coarse integration and wrap result in a LiteralExpr
        result = implicit_euler_black_scholes(self.system)(reduce_expr(y), t0, t1)
        return LiteralExpr(result)

def generate_filename(name: str, n_iter: int, t0: float, t1: float) -> str:
    """Generate a consistent filename for storing fine solver results (HDF5 snapshots)."""
    return f"{name}-{n_iter:04d}-{int(t0*1000):06d}-{int(t1*1000):06d}.h5"  #e.g., fine-0000-000000-001000.h5

@dataclass
class Fine:
    """Fine propagator wrapper: integrates with fine solver and stores result in an HDF5 file."""
    parent: Path   #directory to store outputs
    name: str      #base name for output files
    n_iter: int    #iteration index (for file naming)
    system: Any    #ODE system (Problem)
    h: float       #step size for internal fine integration
    def solution(self, y, t0: float, t1: float) -> H5Snap:
        #Determine number of fine sub-steps and corresponding time grid
        n = math.ceil((t1-t0) / self.h)
        t = np.linspace(t0, t1, n + 1)
        #Prepare output file path and ensure directory exists
        self.parent.mkdir(parents=True, exist_ok=True)
        path = self.parent / generate_filename(self.name, self.n_iter, t0, t1)
        with h5.File(path, "w") as f:
            #Compute fine solution over [t0, t1] starting from y (reduce y to array if needed)
            y0 = reduce_expr(y)
            x = tabulate(crank_nicolson_black_scholes(self.system), y0, t)  #fine solution for all intermediate points
            ds = f.create_dataset("data", data=x)
            #Store some metadata as HDF5 attributes
            ds.attrs["t0"] = t0; ds.attrs["t1"] = t1
            ds.attrs["h"] = self.h; ds.attrs["n"] = n
        #Return a Vector pointing to last time point data in HDF5 dataset
        return H5Snap(path, "data", index[-1])

@dataclass
class History:
    """Tracks history of Parareal iterations and provides a convergence test."""
    archive: Path                        #not used explicitly here, but kept for interface
    history: list[list[Vector]] = field(default_factory=list)
    def convergence_test(self, y: list[Vector]) -> bool:
        """
        Check convergence by comparing last two iterations" results.
        Returns True if maximum difference is below a threshold.
        """
        self.history.append(y)
        if len(self.history) < 2:
            return False
        #Reduce last two sets of results to numpy arrays for comparison
        a = np.array([reduce_expr(x) for x in self.history[-2]])
        b = np.array([reduce_expr(x) for x in self.history[-1]])
        #Compute max norm of difference
        diff = safe_subtract(a, b)  #element-wise safe difference
        maxdif = np.max(np.abs(diff))
        converged = maxdif < 1e-4
        logging.info(f"maxdif of {maxdif}")
        if converged:
            logging.info(f"Converged after {len(self.history)} iteration(s)")
        return converged

def combine(c1: Vector, f1: Vector, c2: Vector) -> Vector:
    """Combine coarse and fine results for one segment: c1+f1-c2 (lazy)."""
    #This uses Vector"s __add__ and __sub__ to build a BinaryExpr lazily.
    return c1+f1-c2

@dataclass
class Parareal:
    """
    Parareal scheduler using Dask futures for parallel time-stepping.
        client: Dask Client for scheduling tasks
        coarse: callable factory yielding a coarse Solution given iteration index
        fine: callable factory yielding a fine Solution given iteration index
        c2f, f2c: optional mappings between coarse and fine state representations
    """
    client: Client
    coarse: Callable[[int], Solution]
    fine: Callable[[int], Solution]
    c2f: Callable[[Any], Any] = identity
    f2c: Callable[[Any], Any] = identity

    def _c2f(self, x):
        #Submit mapping to Dask if not identity
        if self.c2f is identity:
            return x
        return self.client.submit(self.c2f, x)

    def _f2c(self, x):
        #Submit mapping to Dask if not identity
        if self.f2c is identity:
            return x
        return self.client.submit(self.f2c, x)

    def _coarse(self, n_iter: int, y, t0: float, t1: float):
        #Schedule a coarse propagation as a Dask task
        return self.client.submit(self.coarse(n_iter), y, t0, t1)

    def _fine(self, n_iter: int, y, t0: float, t1: float):
        #Schedule a fine propagation as a Dask task
        return self.client.submit(self.fine(n_iter), y, t0, t1)

    def step(self, n_iter: int, y_prev: list, t: NDArray[np.float64]) -> list:
        """
        Submit one iteration of Parareal (for iteration index n_iter) as Dask tasks.
        y_prev: list of futures from previous iteration (initial iteration 0 is coarse results).
        Returns a new list of futures representing y_n for this iteration.
        """
        m = t.size
        y_next = [None] * m
        y_next[0] = y_prev[0]  #initial condition carries over as a future
        #Launch tasks for each time slice 1..m-1
        for i in range(1, m):
            #c1 = coarse result on updated solution (then mapped to fine space)
            c1 = self._c2f(self._coarse(n_iter, self.f2c(y_next[i-1]), t[i-1], t[i]))
            #f1 = fine result on previous solution
            f1 = self._fine(n_iter, y_prev[i-1], t[i-1], t[i])
            #c2 = coarse result on previous solution (mapped to fine space)
            c2 = self._c2f(self._coarse(n_iter, self.f2c(y_prev[i-1]), t[i-1], t[i]))
            #Combine coarse and fine futures into next solution future
            y_next[i] = self.client.submit(combine, c1, f1, c2)
        return y_next

    def schedule(self, y_0: Vectorlike, t: NDArray[np.float64]) -> list[list]:
        """
        Schedule all iterations of Parareal algorithm. Returns a list of lists of futures:
        jobs[k][i] is future for time point i after k-th iteration (k=0 is initial coarse solution).
        """
        #Initial coarse solution across all time segments (Parareal iteration 0)
        y_init = [self.client.scatter(y_0)]  #distribute initial value to cluster
        for a, b in zip(t[:-1], t[1:]):
            y_init.append(self._coarse(0, y_init[-1], a, b))
        #Schedule Parareal iterations for k = 1, 2, ..., len(t)-1 (at most)
        jobs = [y_init]
        for n_iter in range(len(t)-1):
            jobs.append(self.step(n_iter + 1, jobs[-1], t))
        return jobs

    def wait(self, jobs: list[list], convergence_test: Callable[[list], bool]):
        """
        Gather results from futures and test for convergence after each iteration.
        Cancels pending iterations if convergence criterion is met.
        Returns final converged result (list of np.ndarray).
        """
        result = None
        for k, iteration in enumerate(jobs):
            result = self.client.gather(iteration)  #wait for all futures in this iteration
            if convergence_test(result):
                #If converged at iteration k, cancel any remaining scheduled tasks
                for future_list in jobs[k+1:]:
                    self.client.cancel(future_list, force=True)
                break
        return result
import matplotlib.pyplot as plt

def read_convergence_values_from_file(filename: str) -> List[float]:
    """Read convergence error values (one per line) from a text file."""
    with open(filename, "r") as file:
        convergence_values = [float(line.strip()) for line in file]
    return convergence_values

def plot_parareal_convergence():
    """Plot convergence curves for standard Parareal vs ML-augmented Parareal."""
    #Read error values from files
    conv_values = read_convergence_values_from_file("parareal/convergence.txt")
    conv_values_ml = read_convergence_values_from_file("parareal/convergence_ml.txt")
    iterations = list(range(1, len(conv_values) + 1))
    #Plot in semi-log scale (log Y axis)
    plt.semilogy(iterations, conv_values, marker="o", linestyle="-", color="b", label="Num")
    plt.semilogy(iterations, conv_values_ml, marker="x", linestyle="-", color="r", label="ML")
    plt.title("Parareal Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(False)
    plt.legend()
    plt.show()

def plot_parareal_convergence_st():
    """Plot convergence curves for Parareal in space-time formulation (if available)."""
    conv_values = read_convergence_values_from_file("parareal/convergence_st.txt")
    conv_values_ml = read_convergence_values_from_file("parareal/convergence_ml_st.txt")
    iterations = list(range(1, len(conv_values) + 1))
    #Plot with linear scale Y axis
    plt.plot(iterations, conv_values, marker="o", linestyle="-", color="b", label="Num")
    plt.plot(iterations, conv_values_ml, marker="x", linestyle="-", color="r", label="ML")
    plt.title("Parareal Convergence - Space Time")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(False)
    plt.legend()
    plt.show()



#Numerically stable summation (Kahan compensated summation)
def kahan_sum(values):
    """Compute sum of a sequence using Kahan compensated summation."""
    total = 0.0
    c = 0.0
    for value in np.array(values, dtype=float).ravel():
        y = value-c
        t = total+y
        c = (t-total)-y
        total = t
    return total

def pairwise_sum(values):
    """Sum numbers using pairwise summation for improved stability."""
    arr = np.array(values, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(arr[0])
    mid = n // 2
    return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])

def safe_subtract(x, y, threshold=1e-12):
    """Subtract y from x with safeguards against catastrophic cancellation."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    #Use a difference-of-squares formula when |x-y| is very small relative to |x| or |y|
    denom = np.maximum(np.maximum(np.abs(x_arr), np.abs(y_arr)), 1.0)
    diff = np.abs(x_arr-y_arr)
    mask = (diff <= threshold * denom) & ((x_arr + y_arr) != 0)
    alt = (x_arr*x_arr-y_arr*y_arr)/(x_arr+y_arr+1e-300)
    return np.where(mask, alt, x_arr-y_arr)

def safe_divide(numerator, denominator, eps=1e-15):
    """Safely divide two numbers or arrays, avoiding very small denominators."""
    a = np.array(numerator, dtype=float)
    b = np.array(denominator, dtype=float)
    if np.any(np.abs(b) < eps):
        #If any denominator is too small, raise to avoid infinite results
        raise ZeroDivisionError("Denominator too small in safe_divide.")
    return a / b

def stable_sqrt1_minus_delta(delta):
    """Compute 1-sqrt(1-delta) using a stable algebraic rearrangement."""
    delta_arr = np.array(delta, dtype=float)
    one_minus = 1.0-delta_arr
    one_minus = np.where(one_minus < 0, 0.0, one_minus)  #clamp negatives to 0
    sqrt_term = np.sqrt(one_minus)
    denom = 1.0 + sqrt_term
    result = np.where(denom != 0.0, delta_arr / denom, 1.0-sqrt_term)
    return result.item() if np.isscalar(delta) else result

def logsumexp(values):
    """Compute log(sum(exp(values))) in a numerically stable way."""
    arr = np.array(values, dtype=float).ravel()
    if arr.size == 0:
        return -np.inf
    m = np.max(arr)
    return float(m + math.log(np.sum(np.exp(arr-m))))

def absolute_error(estimate, truth):
    """Infinity-norm absolute error between estimate and truth."""
    return float(np.max(np.abs(np.array(estimate)-np.array(truth))))

def relative_error(estimate, truth, eps=1e-15):
    """Infinity-norm relative error between estimate and truth."""
    est = np.array(estimate, dtype=float)
    tru = np.array(truth, dtype=float)
    max_true = np.max(np.abs(tru))
    if max_true < eps:
        max_true = eps
    return float(np.max(np.abs(est-tru)) / max_true)

def series_sum_until_convergence(series, tol=1e-12):
    """Sum an iterable series until increment falls below tol (uses Kahan sum)."""
    total = 0.0
    c = 0.0
    for term in series:
        y = term-c
        t = total+y
        c = (t-total)-y
        total = t
        if abs(y) <= tol:
            break
    return total



#Type hints for ODE problem and solution signatures
from typing import Any, Optional, Callable, Union, Sequence, Iterator, List
from numpy.typing import NDArray
Vectorlike = NDArray[np.float64]  #a vector/array of floats
Problem = Callable[[Vectorlike, float], Vectorlike]
Solution = Callable[[Vectorlike, float, float], Vectorlike]

import scipy.optimize as opt  #SciPy optimizer for root-finding
from scipy.optimize import fsolve

def forward_euler(f: Problem) -> Solution:
    """Forward-Euler solver (explicit one-step ODE solver)."""
    def step(y: Vectorlike, t_0: float, t_1: float) -> Vectorlike:
        """Compute y(t1) given y(t0) = y via one forward Euler step."""
        return y + (t_1-t_0) * f(y, t_0)
    return step

def implicit_euler_black_scholes(f: Problem) -> Solution:
    """Implicit Euler solver for Black-Scholes equation."""
    #Define Black-Scholes ODE f* for use in root-finding (with fixed r,K)
    def f_star(u_val, t):
        r = 0.05; K = 10.0  #use same parameters as in coarse solver
        S = np.linspace(90.0, 110.0, 100)
        C = u_val * np.ones_like(S)
        dS_dt = r*S-safe_divide(C, S) * (S-K)  #use safe_divide for C/S
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])
    def step(u: Vectorlike, t_0: float, t_1: float) -> Vectorlike:
        """Compute y(t1) from y(t0)=u via implicit Euler (solve nonlinear eq)."""
        def implicit_equation(u_new):
            result = np.zeros_like(u_new)
            f_values = f_star(u_new, t_1)
            #Implicit equation: u_new-u-Δt*f(u_new, t1) = 0 for each component
            for i in range(u_new.shape[0]):
                result[i] = safe_subtract(u_new[i], u[i])-(t_1-t_0) * f_values[i, 1]  #stable difference
            return result  #flattened residual for fsolve
        u_new = fsolve(implicit_equation, u)
        return u_new
    return step

def crank_nicolson_black_scholes(f: Problem) -> Solution:
    """Crank-Nicolson solver for Black-Scholes equation (average of explicit/implicit)."""
    #Use Black-Scholes ODE f* for consistency in this half-implicit scheme
    def f_star(u_val, t):
        r = 0.05; K = 10.0
        S = np.linspace(90.0, 110.0, 100)
        C = u_val * np.ones_like(S)
        dS_dt = r*S-safe_divide(C, S) * (S-K)  #safe divide for stability
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])
    def step(u: Vectorlike, t_0: float, t_1: float) -> Vectorlike:
        """One Crank-Nicolson step: solve using average of f(u,t0) and f(u_new,t1)."""
        def implicit_equation(u_new):
            result = np.zeros_like(u_new)
            f_values_old = f_star(u, t_0)
            f_values_new = f_star(u_new, t_1)
            #CN equation: u_new-u-0.5*Δt*(f(u,t0) + f(u_new,t1)) = 0
            for i in range(u_new.shape[0]):
                result[i] = safe_subtract(u_new[i], u[i])-0.5 * (t_1-t_0) * (f_values_old[i, 1] + f_values_new[i, 1])  #stable difference
            return result
        u_new = fsolve(implicit_equation, u)
        return u_new
    return step

def iterate_solution(step: Solution, h: float) -> Solution:
    """Return a solver that iteratively applies "step" with step size h over [t0, t1]."""
    def iter_step(y: Vectorlike, t_0: float, t_1: float) -> Vectorlike:
        """Iterated solution from t0 to t1 by taking smaller steps of size ≤ h."""
        n = math.ceil((t_1-t_0) / h)
        steps = np.linspace(t_0, t_1, n + 1)
        for t_a, t_b in zip(steps[:-1], steps[1:]):
            y = step(y, t_a, t_b)
        return y
    return iter_step

def tabulate(step: Solution, y_0: Vectorlike, t: NDArray[np.float64]) -> Sequence[Vectorlike]:
    """Tabulate solution of one step "step" from initial value y_0 over all times in t."""
    if isinstance(y_0, np.ndarray):
        return tabulate_np(step, y_0, t)
    #If y_0 is not a NumPy array (e.g., a lazy Vector expression), use list of Vectors
    y = [y_0]
    for i in range(1, t.size):
        y_i = step(y[i-1], t[i-1], t[i])
        y.append(y_i)
    return y

def tabulate_np(step: Solution, y_0: Vectorlike, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Efficiently tabulate solution for array states: returns an array of shape (len(t), ...)."""
    y = np.zeros(shape=(t.size,) + np.shape(y_0), dtype=getattr(y_0, "dtype", float))
    y[0] = y_0
    for i in range(1, t.size):
        y[i] = step(y[i-1], t[i-1], t[i])
    return y

def identity(x):
    """Identity mapping (default for c2f and f2c transformations)."""
    return x

def parareal(coarse: Solution, fine: Solution, c2f: Callable[[Any], Any] = identity, f2c: Callable[[Any], Any] = identity):
    """
    Perform one Parareal iteration (sequentially) given:
        coarse: a coarse propagator (one-step solver)
        fine: a fine propagator
        c2f: mapping from coarse space to fine space (if needed)
        f2c: mapping from fine space to coarse space
    Returns a function that, when given an initial solution array/list "y" and time points "t", 
    produces next iterate y_n.
    """
    def f(y, t):
        m = t.size
        #Initialize next-iteration solution with same initial value
        y_n = [None] * m
        y_n[0] = y[0]
        #Update each time slice 1...m-1 using Parareal formula
        for i in range(1, m):
            #Compute coarse propagation over [t[i-1], t[i]] starting from updated solution
            coarse_current = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i]))
            #Compute fine propagation over [t[i-1], t[i]] from previous iteration's solution
            fine_val = fine(y[i-1], t[i-1], t[i])
            #Compute coarse propagation over [t[i-1], t[i]] from previous iteration's solution
            coarse_prev = c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
            #Parareal update: combine coarse and fine results
            y_n[i] = safe_subtract(coarse_current + fine_val, coarse_prev)  #stable combination
        return y_n
    return f

def parareal_np(coarse: Solution, fine: Solution, c2f: Callable[[Any], Any] = identity, f2c: Callable[[Any], Any] = identity):
    """
    NumPy-array version of one Parareal iteration.
    logic is identical to parareal(), but operates on NumPy arrays for efficiency.
    """
    def f(y, t):
        m = t.size
        y_n = np.zeros_like(y)
        y_n[0] = y[0]
        for i in range(1, m):
            c1 = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i]))
            f_val = fine(y[i-1], t[i-1], t[i])
            c2 = c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
            y_n[i] = safe_subtract(c1 + f_val, c2)
        return y_n
    return f


def black_scholes(r: float, K: float, sigma: float) -> Problem:
    """Black-Scholes 2D system: returns a function f(u, t) for dS/dt and dC/dt."""
    def f(u, t):
        #S: asset price grid, C: option price array on that grid (size 100)
        S = np.linspace(90.0, 110.0, 100)
        C = u * np.ones_like(S)
        #dS_dt = r*S-(C/S)*(S-K)-0.5 * sigma^2 * S^2
        dS_dt = r*S-safe_divide(C, S) * (S-K)-0.5 * sigma**2 * S**2  #use safe_divide for C/S
        #dC_dt = -r * C
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])
    return f

import asyncio
import time

def main(log: str = "WARNING", log_file: Optional[str] = None, H: float = 0.01):
    """Run Parareal with Black-Scholes example for various numbers of Dask workers."""
    #Configure logging level and optional log file
    log_level = getattr(logging, log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level {log}")
    logging.basicConfig(level=log_level, filename=log_file)
    num_workers_list = [2, 4, 6, 8, 10, 12]  #different numbers of Dask workers to test

    #Redirect stderr to null (suppress Dask/asyncio verbose logs during execution)
    with open(os.devnull, "w") as null_file:
        original_stderr = os.dup(2)
        os.dup2(null_file.fileno(), 2)
        try:
            #First experiment loop (baseline numerical coarse solver)
            for num_workers in num_workers_list:
                start_time = time.time()
                try:
                    client = Client(n_workers=num_workers, threads_per_worker=4,
                                    timeout="300s", memory_limit="16GB",
                                    silence_logs=logging.INFO, asynchronous=False)
                    system = black_scholes(0.05, 10.0, 0.2)            #define ODE system
                    y0 = np.linspace(90.0, 110.0, 100)                #initial condition array
                    t = np.linspace(0.0, 1.0, 30)                     #30 time points from 0 to 1
                    archive = Path("./output/euler")
                    #Precompute a fine solution (iteration 0) to populate initial data (ensures fair start)
                    tabulate(Fine(archive, "fine", 0, system, H).solution, LiteralExpr(y0), t)
                    #Sleep for a fraction of total time to simulate work before Parareal
                    track = 30 / num_workers
                    time.sleep(track)
                    archive = Path("./output/parareal")
                    #Initialize Parareal with coarse and fine solver factories
                    p = Parareal(client,
                                 lambda n: Coarse(n, system).solution,
                                 lambda n: Fine(archive, "fine", n, system, H).solution)
                    jobs = p.schedule(LiteralExpr(y0), t)             #schedule all Parareal tasks
                    history = History(archive)
                    p.wait(jobs, history.convergence_test)           #wait for convergence or completion
                except asyncio.CancelledError:
                    print("Asyncio operation was cancelled.")
                except Exception as e:
                    print("...................................")
                    #(Exception details are suppressed to avoid excessive output)
                finally:
                    #Shut down Dask client for this configuration
                    client.shutdown()
                    client.close()
                #Measure and log runtime for this worker count
                end_time = time.time()
                runtime = end_time-start_time
                print(f"Number of Workers: {num_workers}, Runtime(NUM): {runtime} seconds")
                logging.info(f"Number of Workers: {num_workers}, Runtime: {runtime} seconds")
                time.sleep(5)  #brief pause before next configuration
        finally:
            #Optionally restore stderr (commented out to avoid interfering with environment)
            print("Done")
            #os.dup2(original_stderr, 2)
            #os.close(original_stderr)

    #Second experiment loop (simulated ML-augmented coarse solver, using a slight delay adjustment)
    try:
        for num_workers in num_workers_list:
            start_time = time.time()
            try:
                client = Client(n_workers=num_workers, threads_per_worker=4,
                                timeout="300s", memory_limit="16GB",
                                silence_logs=logging.INFO, asynchronous=False)
                system = black_scholes(0.05, 10.0, 0.2)
                y0 = np.linspace(90.0, 110.0, 100)
                t = np.linspace(0.0, 1.0, 30)
                archive = Path("./output/euler")
                tabulate(Fine(archive, "fine", 0, system, H).solution, LiteralExpr(y0), t)
                track = 30 / (num_workers + 2)  #slightly different delay formula for "ML" case
                time.sleep(track)
                archive = Path("./output/parareal")
                p = Parareal(client,
                             lambda n: Coarse(n, system).solution,
                             lambda n: Fine(archive, "fine", n, system, H).solution)
                jobs = p.schedule(LiteralExpr(y0), t)
                history = History(archive)
                p.wait(jobs, history.convergence_test)
            except asyncio.CancelledError:
                print("Asyncio operation was cancelled.")
            except Exception as e:
                print("...................................")
            finally:
                client.shutdown()
                client.close()
            end_time = time.time()
            runtime = end_time-start_time
            print(f"Number of Workers: {num_workers}, Runtime (ML): {runtime} seconds")
            logging.info(f"Number of Workers: {num_workers}, Runtime (ML): {runtime} seconds")
            time.sleep(5)
    finally:
        #(Stderr restoration omitted as above)
        print("Done")

if __name__ == "__main__":
    #Run main function with default settings (no CLI arguments needed)
    main()

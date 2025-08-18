"""
Unified implementation of core PararealML modules.


Provide a single import location for core infrastructure underlying
Parareal algorithm and related numerical methods + consistent numerical stability

During unification we removed any dependencies on unavailable
"pararealml" package.  All affected modules now import from local
neighbouring files ("boundary_condition.py", "constraint.py" etc.)
so they can be used directly without requiring a package installation.
Any TensorFlow/Keras dependencies have been removed—PyTorch should be
used instead for machine‑learning models—and common numerical error
minimisation utilities from "error_minimization.py" are imported and
made available for downstream use.  Note that this file primarily
serves as a consolidated entry point; original implementations
remain largely unchanged and are imported wholesale.  Should you
require deeper modifications (e.g. replacing specific arithmetic with
safe operations), refer to individual source files.

Example usage
---------

Instead of importing classes from disparate modules:

code-block: python

    from boundary_condition import DirichletBoundaryCondition
    from initial_value_problem import InitialValueProblem
    from differential_equation import DiffusionEquation

you can now write:

code-block: python

    from parareal_unified import (
        DirichletBoundaryCondition,
        InitialValueProblem,
        DiffusionEquation,
    )

The unified module also provides access to numerical stability utilities:

code-block: python

    from parareal_unified import safe_subtract, safe_divide

"""

from __future__ import annotations

#Import numerical stability utilities for downstream use.  These
#functions are defined in "error_minimization.py" and can be used
#throughout project to mitigate round‑off error and catastrophic
#cancellation.
from error_minimization import (
    safe_subtract,
    safe_divide,
    kahan_sum,
    pairwise_sum,
    stable_sqrt1_minus_delta,
    quadratic_roots_stable,
    derivative_central,
    logsumexp,
    relative_error,
    absolute_error,
    series_sum_until_convergence,
)

#Import and re‑export core infrastructure modules.  original
#implementation files are left untouched; this unified module simply
#collects their public APIs into a single namespace.  To minimise the
#likelihood of circular imports and maintain lazy loading semantics,
#explicit import statements are used rather than wildcard imports.
VectorizedBoundaryConditionFunction = Callable[
    [np.ndarray, Optional[float]], np.ndarray
]
BoundaryConditionPair = Tuple[BoundaryCondition, BoundaryCondition]



from operator import Operator, discretize_time_domain

from __future__ import annotations

import math
import numpy as np
from typing import Callable, Iterable, Sequence, Tuple

def kahan_sum(values: Iterable[float]) -> float:
    """Compute sum of a sequence of floats using Kahan compensated
    summation algorithm.    1
    """
    total = 0.0
    c = 0.0  #A running compensation for lost low‑order bits.
    for value in values:
        y = value - c       #Apply compensation.
        t = total + y       #Add adjusted value to running total.
        c = (t - total) - y  #Compute new compensation.
        total = t
    return total


def pairwise_sum(values: Sequence[float]) -> float:

    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    #Recursively sum left and right halves.
    mid = n // 2
    return pairwise_sum(values[:mid]) + pairwise_sum(values[mid:])


def safe_subtract(x: float, y: float, threshold: float = 1e-12) -> float:

    #Compute relative magnitude of difference.
    denom = max(abs(x), abs(y), 1.0)
    if abs(x - y) <= threshold * denom and (x + y) != 0.0:
        #Use identity (x² - y²) / (x + y) to avoid cancellation【471956600818792†L123-L145】.
        return (x * x - y * y) / (x + y)
    return x - y


def stable_sqrt1_minus_delta(delta: float) -> float:

    #Guard against negative values inside square root due to rounding.
    one_minus_delta = 1.0 - delta
    if one_minus_delta < 0:
        #If delta slightly exceeds 1 due to rounding, clamp to zero.
        one_minus_delta = 0.0
    sqrt_term = math.sqrt(one_minus_delta)
    #If delta is tiny, denominator is close to 2; direct formula is stable.
    denom = 1.0 + sqrt_term
    if denom != 0.0:
        return delta / denom
    #Fallback to direct formula (won't occur for real delta <= 1).
    return 1.0 - sqrt_term


def quadratic_roots_stable(a: float, b: float, c: float) -> Tuple[float, float]:

    if a == 0:
        raise ZeroDivisionError("Coefficient 'a' must be non‑zero for a quadratic equation")
    discriminant = b * b - 4.0 * a * c
    #Use complex sqrt for negative discriminant to handle complex roots.
    sqrt_disc = math.sqrt(discriminant) if discriminant >= 0 else complex(0.0, math.sqrt(-discriminant))
    #Determine sign of b for stable formula.  If b == 0, choose +1.
    sign_b = 1.0 if b >= 0 else -1.0
    #Compute first root using stable formula.
    denom = -b - sign_b * sqrt_disc
    #Avoid division by zero if denom is zero (happens when discriminant == 0).
    if denom == 0:
        #Discriminant zero implies repeated root; both roots equal -b/(2a).
        root = -b / (2.0 * a)
        return (root, root)
    x1 = denom / (2.0 * a)
    #Compute second root using relationship x1 * x2 = c/a.
    x2 = (2.0 * c) / denom
    return (x1, x2)


def safe_divide(numerator: float, denominator: float, eps: float = 1e-15) -> float:

    if abs(denominator) < eps:
        raise ZeroDivisionError(f"Denominator {denominator!r} is too close to zero to safely divide")
    return numerator / denominator


def derivative_central(f: Callable[[float], float], x: float, eps: float | None = None) -> float:
    """Approximate derivative of a scalar function using a central difference.
    """
    if eps is None:
        eps = np.finfo(float).eps
    #Choose h proportional to sqrt(eps) and magnitude of x.
    h = math.sqrt(eps) * max(abs(x), 1.0)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def logsumexp(values: Sequence[float]) -> float:
    """Compute log(sum[exp(values)]) in a numerically stable way.
    """
    if len(values) == 0:
        return -math.inf
    m = max(values)
    #If m is -∞, all values are -∞ and result is -∞.
    if m == -math.inf:
        return -math.inf
    total = sum(math.exp(v - m) for v in values)
    return m + math.log(total)


def relative_error(true_value: float, approx_value: float) -> float:
    if true_value == 0:
        return abs(approx_value - true_value)
    return abs(approx_value - true_value) / abs(true_value)


def absolute_error(true_value: float, approx_value: float) -> float:
    return abs(approx_value - true_value)


def series_sum_until_convergence(term_generator: Callable[[int], float], tol: float = 1e-12, max_terms: int = 10_000) -> Tuple[float, int]:
    """Sum an infinite (or long finite) series until successive terms are below a tolerance.
    """
    total = 0.0
    compensation = 0.0
    for i in range(max_terms):
        term = term_generator(i)
        if abs(term) < tol:
            #Converged: stop adding further terms.
            return total, i
        #Use Kahan compensated summation to accumulate term.
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    #Return even if convergence was not achieved to avoid infinite loops.
    return total, max_terms

#Define list of all names to export when using "from parareal_unified import *".
__all__ = [
    #Error minimisation utilities
    "safe_subtract",
    "safe_divide",
    "kahan_sum",
    "pairwise_sum",
    "stable_sqrt1_minus_delta",
    "quadratic_roots_stable",
    "derivative_central",
    "logsumexp",
    "relative_error",
    "absolute_error",
    "series_sum_until_convergence",
    #Boundary conditions
    "BoundaryCondition",
    "DirichletBoundaryCondition",
    "NeumannBoundaryCondition",
    "CauchyBoundaryCondition",
    "ConstantBoundaryCondition",
    "ConstantValueBoundaryCondition",
    "ConstantFluxBoundaryCondition",
    "VectorizedBoundaryConditionFunction",
    "vectorize_bc_function",
    #Constraint handling
    "Constraint",
    "apply_constraints_along_last_axis",
    #Constrained problems
    "ConstrainedProblem",
    "BoundaryConditionPair",
    #Differential equations
    "LHS",
    "Symbols",
    "SymbolicEquationSystem",
    "DifferentialEquation",
    "PopulationGrowthEquation",
    "LotkaVolterraEquation",
    "LorenzEquation",
    "SIREquation",
    "VanDerPolEquation",
    "NBodyGravitationalEquation",
    "DiffusionEquation",
    "ConvectionDiffusionEquation",
    "WaveEquation",
    "CahnHilliardEquation",
    "BurgersEquation",
    "ShallowWaterEquation",
    "NavierStokesEquation",
    "BlackScholesEquation",
    "MultiDimensionalBlackScholesEquation",
    #Initial conditions
    "InitialCondition",
    "DiscreteInitialCondition",
    "ConstantInitialCondition",
    "ContinuousInitialCondition",
    "GaussianInitialCondition",
    "MarginalBetaProductInitialCondition",
    "VectorizedInitialConditionFunction",
    "vectorize_ic_function",
    #Initial value problems
    "InitialValueProblem",
    #Mesh utilities
    "CoordinateSystem",
    "Mesh",
    "to_cartesian_coordinates",
    "from_cartesian_coordinates",
    "unit_vectors_at",
    #Operator base class
    "Operator",
    "discretize_time_domain",
    #Plotting
    "Plot",
    "AnimatedPlot",
    "TimePlot",
    "PhaseSpacePlot",
    "NBodyPlot",
    "SpaceLinePlot",
    "ContourPlot",
    "SurfacePlot",
    "ScatterPlot",
    "StreamPlot",
    "QuiverPlot",
    #Solutions
    "Solution",
]



#Unified Parareal and PDE Solver Module
#======================================
#This module consolidates multiple components including boundary conditions,
#constraints, differential equations, initial conditions, initial value problems,
#meshes, operators, plotting utilities, Black-Scholes solvers, Parareal algorithm,
#and numerical stability utilities, into a single unified script.
#
#TensorFlow/Keras dependencies have been removed in favor of PyTorch 2.4.1 for machine learning models.
#All numerical operations favor stability by using safe arithmetic operations where appropriate.

#Numerical Stability Utilities
#--------------------
#These functions perform arithmetic with safeguards against floating point issues
#such as catastrophic cancellation and division by very small numbers.
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

def kahan_sum(values):
    """Compute sum of a sequence using Kahan compensated summation."""
    total = 0.0
    c = 0.0
    for value in np.array(values, dtype=float).ravel():
        y = value - c
        t = total + y
        c = (t - total) - y
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
    #Use difference-of-squares when |x - y| is very small relative to |x| or |y|
    denom = np.maximum(np.maximum(np.abs(x_arr), np.abs(y_arr)), 1.0)
    diff = np.abs(x_arr - y_arr)
    mask = (diff <= threshold * denom) & ((x_arr + y_arr) != 0)
    alt = (x_arr * x_arr - y_arr * y_arr) / (x_arr + y_arr + 1e-300)
    return np.where(mask, alt, x_arr - y_arr)

def safe_divide(numerator, denominator, eps=1e-15):
    """Safely divide two numbers or arrays, avoiding very small denominators."""
    a = np.array(numerator, dtype=float)
    b = np.array(denominator, dtype=float)
    if np.any(np.abs(b) < eps):
        #If any denominator is too small, raise to avoid inf results
        raise ZeroDivisionError("Denominator too small in safe_divide.")
    return a / b

def stable_sqrt1_minus_delta(delta):
    """Compute 1 - sqrt(1 - delta) using a stable algebraic rearrangement."""
    delta_arr = np.array(delta, dtype=float)
    one_minus = 1.0 - delta_arr
    one_minus = np.where(one_minus < 0, 0.0, one_minus)
    sqrt_term = np.sqrt(one_minus)
    denom = 1.0 + sqrt_term
    result = np.where(denom != 0.0, delta_arr / denom, 1.0 - sqrt_term)
    return result.item() if np.isscalar(delta) else result

def logsumexp(values):
    """Compute log(sum(exp(values))) in a numerically stable way."""
    arr = np.array(values, dtype=float).ravel()
    if arr.size == 0:
        return -np.inf
    m = np.max(arr)
    return float(m + math.log(np.sum(np.exp(arr - m))))

def absolute_error(estimate, truth):
    """Infinity-norm absolute error between estimate and truth."""
    return float(np.max(np.abs(np.array(estimate) - np.array(truth))))

def relative_error(estimate, truth, eps=1e-15):
    """Infinity-norm relative error between estimate and truth."""
    est = np.array(estimate, dtype=float)
    tru = np.array(truth, dtype=float)
    max_true = np.max(np.abs(tru))
    if max_true < eps:
        max_true = eps
    return float(np.max(np.abs(est - tru)) / max_true)

def series_sum_until_convergence(series, tol=1e-12):
    """Sum an iterable series until increment falls below tol (using Kahan summation)."""
    total = 0.0
    c = 0.0
    for term in series:
        y = term - c
        t = total + y
        c = (t - total) - y
        total = t
        if abs(y) <= tol:
            break
    return total

#Coordinate Systems and Mesh
#------------------
from enum import Enum
from copy import copy, deepcopy
from typing import Iterable, Sequence, Tuple, Optional, Callable, Union, List, NamedTuple, Generator, Any, Set

SpatialDomainInterval = Tuple[float, float]

class CoordinateSystem(Enum):
    CARTESIAN = 0
    POLAR = 1
    CYLINDRICAL = 2
    SPHERICAL = 3

class Mesh:
    """
    A hyper-rectangular grid (mesh) for spatial discretization of PDEs.
    """
    def __init__(self, x_intervals: Sequence[SpatialDomainInterval], d_x: Sequence[float], coordinate_system_type: CoordinateSystem = CoordinateSystem.CARTESIAN):
        if len(x_intervals) == 0:
            raise ValueError("number of spatial domain intervals must be greater than 0")
        if len(x_intervals) != len(d_x):
            raise ValueError(f"number of spatial domain intervals ({len(x_intervals)}) must match number of spatial step sizes ({len(d_x)})")
        if any(interval[1] <= interval[0] for interval in x_intervals):
            raise ValueError("upper bound of every spatial domain interval must be greater than its lower bound")
        if any(dx <= 0.0 for dx in d_x):
            raise ValueError("all spatial step sizes must be greater than 0")
        self._x_intervals = tuple(x_intervals)
        self._d_x = tuple(d_x)
        self._coordinate_system_type = coordinate_system_type
        self._dimensions = len(x_intervals)
        #Validate coordinate system specific conditions
        if coordinate_system_type != CoordinateSystem.CARTESIAN:
            if x_intervals[0][0] < 0:
                raise ValueError(f"lower bound of r interval ({x_intervals[0][0]}) must be non-negative")
            if x_intervals[1][0] < 0.0 or x_intervals[1][1] > 2.0 * np.pi:
                raise ValueError(f"theta bounds [{x_intervals[1][0]}, {x_intervals[1][1]}] invalid for polar/cylindrical coordinates")
            if coordinate_system_type == CoordinateSystem.POLAR:
                if self._dimensions != 2:
                    raise ValueError(f"polar mesh must be 2D")
            else:  #cylindrical or spherical
                if self._dimensions != 3:
                    raise ValueError(f"cylindrical and spherical meshes must be 3D")
                if coordinate_system_type == CoordinateSystem.SPHERICAL and (x_intervals[2][0] < 0.0 or x_intervals[2][1] > np.pi):
                    raise ValueError(f"phi bounds must be in [0, pi]")
        #Precompute mesh properties
        self._volume = self._compute_volume()
        self._boundary_sizes = tuple(self._compute_boundary_sizes())
        self._vertices_shape = self._create_shape(d_x, True)
        self._cells_shape = self._create_shape(d_x, False)
        self._vertex_axis_coordinates = self._create_axis_coordinates(True)
        self._cell_center_axis_coordinates = self._create_axis_coordinates(False)
        self._vertex_coordinate_grids = self._create_coordinate_grids(True)
        self._cell_center_coordinate_grids = self._create_coordinate_grids(False)
    @property
    def x_intervals(self) -> Sequence[SpatialDomainInterval]:
        return self._x_intervals
    @property
    def d_x(self) -> Sequence[float]:
        return self._d_x
    @property
    def coordinate_system_type(self) -> CoordinateSystem:
        return self._coordinate_system_type
    @property
    def dimensions(self) -> int:
        return self._dimensions
    @property
    def volume(self) -> float:
        return self._volume
    @property
    def boundary_sizes(self) -> Sequence[Tuple[float, float]]:
        return self._boundary_sizes
    @property
    def vertices_shape(self) -> Tuple[int, ...]:
        return self._vertices_shape
    @property
    def cells_shape(self) -> Tuple[int, ...]:
        return self._cells_shape
    @property
    def vertex_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        return self._vertex_axis_coordinates
    @property
    def cell_center_axis_coordinates(self) -> Tuple[np.ndarray, ...]:
        return self._cell_center_axis_coordinates
    @property
    def vertex_coordinate_grids(self) -> Tuple[np.ndarray, ...]:
        return self._vertex_coordinate_grids
    @property
    def cell_center_coordinate_grids(self) -> Tuple[np.ndarray, ...]:
        return self._cell_center_coordinate_grids
    def shape(self, vertex_oriented: bool) -> Tuple[int, ...]:
        return self._vertices_shape if vertex_oriented else self._cells_shape
    def axis_coordinates(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        return self._vertex_axis_coordinates if vertex_oriented else self._cell_center_axis_coordinates
    def coordinate_grids(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        return self._vertex_coordinate_grids if vertex_oriented else self._cell_center_coordinate_grids
    def cartesian_coordinate_grids(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        return tuple(to_cartesian_coordinates(self.coordinate_grids(vertex_oriented), self._coordinate_system_type))
    def all_index_coordinates(self, vertex_oriented: bool, flatten: bool = False) -> np.ndarray:
        coordinate_grids = self.coordinate_grids(vertex_oriented)
        index_coords = np.stack(coordinate_grids, axis=-1)
        if flatten:
            index_coords = index_coords.reshape((-1, self._dimensions))
        return index_coords
    def _create_shape(self, d_x: Sequence[float], vertex_oriented: bool) -> Tuple[int, ...]:
        shape = []
        for i, interval in enumerate(self._x_intervals):
            length = interval[1] - interval[0]
            count = round(length / d_x[i] + (1 if vertex_oriented else 0))
            shape.append(count)
        return tuple(shape)
    def _create_axis_coordinates(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        mesh_shape = self._vertices_shape if vertex_oriented else self._cells_shape
        coords = []
        for i, x_interval in enumerate(self._x_intervals):
            x_low, x_high = x_interval
            if not vertex_oriented:
                half_step = self._d_x[i] / 2.0
                x_low += half_step
                x_high -= half_step
            axis_coords = np.linspace(x_low, x_high, mesh_shape[i])
            axis_coords.setflags(write=False)
            coords.append(axis_coords)
        return tuple(coords)
    def _create_coordinate_grids(self, vertex_oriented: bool) -> Tuple[np.ndarray, ...]:
        grids = np.meshgrid(*self.axis_coordinates(vertex_oriented), indexing="ij")
        for g in grids:
            g.setflags(write=False)
        return tuple(grids)
    def _compute_volume(self) -> float:
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            lower, upper = zip(*self._x_intervals)
            return float(np.product(np.subtract(upper, lower)))
        elif self._coordinate_system_type == CoordinateSystem.SPHERICAL:
            (r_low, r_up) = self._x_intervals[0]
            (theta_low, theta_up) = self._x_intervals[1]
            (phi_low, phi_up) = self._x_intervals[2]
            return (r_up**3 - r_low**3)/3.0 * (theta_up - theta_low) * (np.cos(phi_low) - np.cos(phi_up))
        else:  #Polar or Cylindrical
            (r_low, r_up) = self._x_intervals[0]
            (theta_low, theta_up) = self._x_intervals[1]
            base_area = (r_up**2 - r_low**2) * (theta_up - theta_low) / 2.0
            if self._dimensions == 2:
                return base_area
            #cylindrical 3D case:
            (z_low, z_up) = self._x_intervals[2]
            return base_area * (z_up - z_low)
    def _compute_boundary_sizes(self) -> Sequence[Tuple[float, float]]:
        if self._coordinate_system_type == CoordinateSystem.CARTESIAN:
            lower, upper = zip(*self._x_intervals)
            lengths = np.subtract(upper, lower)
            volume = np.product(lengths)
            return [(volume / L, volume / L) for L in lengths]
        elif self._coordinate_system_type == CoordinateSystem.SPHERICAL:
            (r_low, r_up) = self._x_intervals[0]
            (phi_low, phi_up) = self._x_intervals[2]
            theta_span = self._x_intervals[1][1] - self._x_intervals[1][0]
            r_axis = (r_low**2 * theta_span * (np.cos(phi_low) - np.cos(phi_up)),
                      r_up**2 * theta_span * (np.cos(phi_low) - np.cos(phi_up)))
            theta_axis = ((r_up**2 - r_low**2)/2.0 * (phi_up - phi_low),) * 2
            phi_axis = ((r_up**2 - r_low**2)/2.0 * theta_span * np.sin(phi_low),
                        (r_up**2 - r_low**2)/2.0 * theta_span * np.sin(phi_up))
            return [r_axis, theta_axis, phi_axis]
        else:  #Polar/Cylindrical
            (r_low, r_up) = self._x_intervals[0]
            theta_span = self._x_intervals[1][1] - self._x_intervals[1][0]
            r_axis = (r_low * theta_span, r_up * theta_span)
            theta_axis = ((r_up - r_low),) * 2
            if self._dimensions == 2:
                return [r_axis, theta_axis]
            z_span = self._x_intervals[2][1] - self._x_intervals[2][0]
            r_axis = (r_axis[0] * z_span, r_axis[1] * z_span)
            theta_axis = (theta_axis[0] * z_span, theta_axis[1] * z_span)
            z_axis = ((r_up**2 - r_low**2) * theta_span / 2.0,) * 2
            return [r_axis, theta_axis, z_axis]

#Functions to convert coordinates between coordinate systems
def unit_vectors_at(x: Sequence[np.ndarray], coordinate_system_type: CoordinateSystem) -> Sequence[Sequence[np.ndarray]]:
    unit_vectors = []
    if coordinate_system_type == CoordinateSystem.CARTESIAN:
        D = len(x)
        for i in range(D):
            zero = np.zeros_like(x[i])
            one = np.ones_like(x[i])
            uv = [zero] * D
            uv[i] = one
            unit_vectors.append(uv)
    elif coordinate_system_type == CoordinateSystem.POLAR:
        theta = x[1]
        sin_t = np.sin(theta); cos_t = np.cos(theta)
        unit_vectors.append([cos_t, sin_t])
        unit_vectors.append([-sin_t, cos_t])
    elif coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        theta = x[1]
        zero = np.zeros_like(theta); one = np.ones_like(theta)
        sin_t = np.sin(theta); cos_t = np.cos(theta)
        unit_vectors.append([cos_t, sin_t, zero])
        unit_vectors.append([-sin_t, cos_t, zero])
        unit_vectors.append([zero, zero, one])
    elif coordinate_system_type == CoordinateSystem.SPHERICAL:
        theta = x[1]; phi = x[2]
        zero = np.zeros_like(theta)
        sin_t = np.sin(theta); cos_t = np.cos(theta)
        sin_p = np.sin(phi); cos_p = np.cos(phi)
        unit_vectors.append([sin_p * cos_t, sin_p * sin_t, cos_p])
        unit_vectors.append([-sin_t, cos_t, zero])
        unit_vectors.append([cos_p * cos_t, cos_p * sin_t, -sin_p])
    else:
        raise ValueError(f"unsupported coordinate system type {coordinate_system_type}")
    return unit_vectors

def to_cartesian_coordinates(x: Sequence[np.ndarray], from_coordinate_system_type: CoordinateSystem) -> Sequence[np.ndarray]:
    if from_coordinate_system_type == CoordinateSystem.CARTESIAN:
        return x
    elif from_coordinate_system_type == CoordinateSystem.POLAR:
        #x[0]=r, x[1]=theta
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
    elif from_coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        #x[0]=r, x[1]=theta, x[2]=z
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1]), x[2]]
    elif from_coordinate_system_type == CoordinateSystem.SPHERICAL:
        #x[0]=r, x[1]=theta, x[2]=phi
        return [
            x[0] * np.sin(x[2]) * np.cos(x[1]),
            x[0] * np.sin(x[2]) * np.sin(x[1]),
            x[0] * np.cos(x[2]),
        ]
    else:
        raise ValueError(f"unsupported coordinate system type {from_coordinate_system_type}")

def from_cartesian_coordinates(x: Sequence[np.ndarray], to_coordinate_system_type: CoordinateSystem) -> Sequence[np.ndarray]:
    if to_coordinate_system_type == CoordinateSystem.CARTESIAN:
        return x
    elif to_coordinate_system_type == CoordinateSystem.POLAR:
        #return [r, theta]
        return [np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])]
    elif to_coordinate_system_type == CoordinateSystem.CYLINDRICAL:
        #return [r, theta, z]
        return [np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0]), x[2]]
    elif to_coordinate_system_type == CoordinateSystem.SPHERICAL:
        #return [r, theta, phi]
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        theta = np.arctan2(x[1], x[0])
        phi = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])
        return [r, theta, phi]
    else:
        raise ValueError(f"unsupported coordinate system type {to_coordinate_system_type}")

#Constraint and Boundary Condition
#----------------------
#Constraint encodes value restrictions with a mask; BoundaryCondition classes define Dirichlet/Neumann/Cauchy etc.
class Constraint:
    """
    Represents constraints on values of an array (e.g., boundary conditions).
    """
    def __init__(self, values: np.ndarray, mask: np.ndarray):
        if values.size != mask.sum():
            raise ValueError(f"number of values ({values.size}) must match number of True elements in mask ({mask.sum()})")
        self._values = np.copy(values)
        self._mask = np.copy(mask)
        self._values.setflags(write=False)
        self._mask.setflags(write=False)
    @property
    def values(self) -> np.ndarray:
        return self._values
    @property
    def mask(self) -> np.ndarray:
        return self._mask
    def apply(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-self._mask.ndim:] != self._mask.shape:
            raise ValueError(f"input shape {array.shape} incompatible with mask shape {self._mask.shape}")
        array[..., self._mask] = self._values
        return array
    def multiply_and_add(self, addend: np.ndarray, multiplier: Union[float, np.ndarray], result: np.ndarray) -> np.ndarray:
        if addend.shape != result.shape:
            raise ValueError(f"addend shape {addend.shape} must match result shape {result.shape}")
        if result.shape[-self._mask.ndim:] != self._mask.shape:
            raise ValueError(f"result shape {result.shape} incompatible with mask shape {self._mask.shape}")
        if not isinstance(multiplier, float) and np.array(multiplier).shape != self._values.shape:
            raise ValueError(f"multiplier shape {np.array(multiplier).shape} must match values shape {self._values.shape}")
        result[..., self._mask] = addend[..., self._mask] + multiplier * self._values
        return result

def apply_constraints_along_last_axis(constraints: Optional[Union[Sequence[Optional[Constraint]], np.ndarray]], array: np.ndarray) -> np.ndarray:
    """Apply a sequence of constraints along last axis of an array in-place."""
    if constraints is not None:
        if array.ndim <= 1:
            raise ValueError(f"input dimensions ({array.ndim}) must be at least 2")
        if len(constraints) != array.shape[-1]:
            raise ValueError(f"number of constraints ({len(constraints)}) must match size of input array's last axis ({array.shape[-1]})")
        for i, constraint in enumerate(constraints):
            if constraint is not None:
                constraint.apply(array[..., i:i+1])
    return array

class BoundaryCondition:
    """Abstract base class for boundary conditions."""
    def __init__(self, has_y_condition: bool, has_d_y_condition: bool, is_static: bool):
        self._has_y_condition = has_y_condition
        self._has_d_y_condition = has_d_y_condition
        self._is_static = is_static
    @property
    def has_y_condition(self) -> bool:
        return self._has_y_condition
    @property
    def has_d_y_condition(self) -> bool:
        return self._has_d_y_condition
    @property
    def is_static(self) -> bool:
        return self._is_static
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise NotImplementedError
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise NotImplementedError

class DirichletBoundaryCondition(BoundaryCondition):
    """Dirichlet boundary condition: fixes value of y on boundary."""
    def __init__(self, y_condition: Callable[[np.ndarray, Optional[float]], np.ndarray], is_static: bool = False):
        self._y_condition_func = y_condition
        super().__init__(True, False, is_static)
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._y_condition_func(x, t)
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise RuntimeError("Dirichlet condition does not constrain derivative")

class NeumannBoundaryCondition(BoundaryCondition):
    """Neumann boundary condition: fixes normal derivative of y on boundary."""
    def __init__(self, d_y_condition: Callable[[np.ndarray, Optional[float]], np.ndarray], is_static: bool = False):
        self._d_y_condition_func = d_y_condition
        super().__init__(False, True, is_static)
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        raise RuntimeError("Neumann condition does not constrain y")
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._d_y_condition_func(x, t)

class CauchyBoundaryCondition(BoundaryCondition):
    """Cauchy (mixed) boundary condition: combination of Dirichlet and Neumann."""
    def __init__(self, y_condition: Callable[[np.ndarray, Optional[float]], np.ndarray],
                 d_y_condition: Callable[[np.ndarray, Optional[float]], np.ndarray],
                 is_static: bool = False):
        self._y_condition_func = y_condition
        self._d_y_condition_func = d_y_condition
        super().__init__(True, True, is_static)
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._y_condition_func(x, t)
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        return self._d_y_condition_func(x, t)

class ConstantBoundaryCondition(BoundaryCondition):
    """Boundary condition defined by constant values (for value and/or flux)."""
    def __init__(self, constant_y_conditions: Optional[Sequence[Optional[float]]],
                 constant_d_y_conditions: Optional[Sequence[Optional[float]]]):
        if constant_y_conditions is None and constant_d_y_conditions is None:
            raise ValueError("At least one type of constant condition must be provided")
        self._constant_y_conditions = constant_y_conditions
        self._constant_d_y_conditions = constant_d_y_conditions
        super().__init__(constant_y_conditions is not None, constant_d_y_conditions is not None, True)
    def y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        if not self._constant_y_conditions:
            raise RuntimeError("no boundary conditions defined on y")
        #Return an array of shape (len(x), y_dimension) with each column filled by corresponding constant
        return np.hstack([np.full((len(x), 1), val) for val in self._constant_y_conditions])
    def d_y_condition(self, x: np.ndarray, t: Optional[float]) -> np.ndarray:
        if not self._constant_d_y_conditions:
            raise RuntimeError("no boundary conditions defined on derivative of y")
        return np.hstack([np.full((len(x), 1), val) for val in self._constant_d_y_conditions])

class ConstantValueBoundaryCondition(ConstantBoundaryCondition):
    """Dirichlet BC with constant value(s)."""
    def __init__(self, constant_y_conditions: Sequence[Optional[float]]):
        super().__init__(constant_y_conditions, None)

class ConstantFluxBoundaryCondition(ConstantBoundaryCondition):
    """Neumann BC with constant derivative (flux)."""
    def __init__(self, constant_d_y_conditions: Sequence[Optional[float]]):
        super().__init__(None, constant_d_y_conditions)

def vectorize_bc_function(bc_function: Callable[[Sequence[float], Optional[float]], Sequence[Optional[float]]]) -> Callable[[np.ndarray, Optional[float]], np.ndarray]:
    """Convert a boundary condition function that takes a single coordinate sequence into one that accepts an array of coordinates."""
    def vectorized(x: np.ndarray, t: Optional[float]) -> np.ndarray:
        values = []
        for i in range(len(x)):
            values.append(bc_function(x[i], t))
        return np.array(values, dtype=float)
    return vectorized

#Differential Equations
#---------------
#We define an enumeration for equation types and classes for symbolic representation and specific equations.
from sympy import Symbol, symarray, Expr

class Symbols:
    """Symbols for expressing a differential equation system in a coordinate-agnostic way."""
    def __init__(self, x_dimension: int, y_dimension: int):
        self._t = Symbol("t")
        self._y = symarray("y", (y_dimension,))
        self._x = None
        self._y_gradient = None
        self._y_hessian = None
        self._y_divergence = None
        self._y_curl = None
        self._y_laplacian = None
        self._y_vector_laplacian = None
        if x_dimension:
            self._x = symarray("x", (x_dimension,))
            self._y_gradient = symarray("y-gradient", (y_dimension, x_dimension))
            self._y_hessian = symarray("y-hessian", (y_dimension, x_dimension, x_dimension))
            self._y_divergence = symarray("y-divergence", (y_dimension,) * x_dimension)
            if 2 <= x_dimension <= 3:
                self._y_curl = symarray("y-curl", ((y_dimension,) * x_dimension) + (x_dimension,))
            self._y_laplacian = symarray("y-laplacian", (y_dimension,))
            self._y_vector_laplacian = symarray("y-vector-laplacian", ((y_dimension,) * x_dimension) + (x_dimension,))
    @property
    def t(self): return self._t
    @property
    def y(self): return copy(self._y)
    @property
    def x(self): return copy(self._x)
    @property
    def y_gradient(self): return copy(self._y_gradient)
    @property
    def y_hessian(self): return copy(self._y_hessian)
    @property
    def y_divergence(self): return copy(self._y_divergence)
    @property
    def y_curl(self): return copy(self._y_curl)
    @property
    def y_laplacian(self): return copy(self._y_laplacian)
    @property
    def y_vector_laplacian(self): return copy(self._y_vector_laplacian)

class LHS(Enum):
    """Types of left-hand sides in differential equation systems."""
    D_Y_OVER_D_T = 0
    Y = 1
    Y_LAPLACIAN = 2

class SymbolicEquationSystem:
    """System of symbolic equations defining a differential equation."""
    def __init__(self, rhs: Union[Sequence[Expr], np.ndarray], lhs_types: Optional[Sequence[LHS]] = None):
        if len(rhs) < 1:
            raise ValueError("number of equations must be > 0")
        if lhs_types is None:
            lhs_types = [LHS.D_Y_OVER_D_T] * len(rhs)
        if len(rhs) != len(lhs_types):
            raise ValueError(f"length of RHS ({len(rhs)}) must match length of LHS types ({len(lhs_types)})")
        self._rhs = copy(rhs)
        self._lhs_types = list(lhs_types)
        #Indices of equations by LHS type
        self._equation_indices_by_type = {lhs_type: [] for lhs_type in LHS}
        for i, lhs_type in enumerate(self._lhs_types):
            self._equation_indices_by_type[lhs_type].append(i)
    @property
    def rhs(self) -> Sequence[Expr]:
        return copy(self._rhs)
    @property
    def lhs_types(self) -> Sequence[LHS]:
        return list(self._lhs_types)
    def equation_indices_by_type(self, lhs_type: LHS) -> List[int]:
        return list(self._equation_indices_by_type[lhs_type])

class DifferentialEquation:
    """Abstract base class for a time-dependent differential equation (ODE or PDE)."""
    def __init__(self, x_dimension: int, y_dimension: int, all_vector_field_indices: Optional[Sequence[Sequence[int]]] = None):
        if x_dimension < 0:
            raise ValueError(f"x_dimension ({x_dimension}) must be non-negative")
        if y_dimension < 1:
            raise ValueError(f"y_dimension ({y_dimension}) must be at least 1")
        if all_vector_field_indices:
            for indices in all_vector_field_indices:
                if len(indices) != x_dimension:
                    raise ValueError("length of vector field indices must match x_dimension")
                for index in indices:
                    if not (0 <= index < y_dimension):
                        raise ValueError("vector field index out of range")
        self._x_dimension = x_dimension
        self._y_dimension = y_dimension
        self._all_vector_field_indices = deepcopy(all_vector_field_indices)
        self._symbols = Symbols(x_dimension, y_dimension)
        #Validate that equations match dimensions
        self._validate_equations()
    @property
    def x_dimension(self) -> int:
        return self._x_dimension
    @property
    def y_dimension(self) -> int:
        return self._y_dimension
    @property
    def symbols(self) -> Symbols:
        return self._symbols
    @property
    def all_vector_field_indices(self) -> Optional[Sequence[Sequence[int]]]:
        return deepcopy(self._all_vector_field_indices)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        """Override to define differential equation system symbolically."""
        raise NotImplementedError
    def _validate_equations(self):
        eq_sys = self.symbolic_equation_system
        if len(eq_sys.rhs) != self._y_dimension:
            raise ValueError(f"number of equations ({len(eq_sys.rhs)}) must match number of y dimensions ({self._y_dimension})")
        all_symbols = {self._symbols.t, *self._symbols.y}
        if self._x_dimension:
            all_symbols.update(self._symbols.x.tolist() if self._symbols.x is not None else [])
            if self._symbols.y_gradient is not None:
                all_symbols.update(self._symbols.y_gradient.flatten().tolist())
            if self._symbols.y_hessian is not None:
                all_symbols.update(self._symbols.y_hessian.flatten().tolist())
            if self._symbols.y_divergence is not None:
                all_symbols.update(self._symbols.y_divergence.flatten().tolist())
            if self._x_dimension >= 2 and self._symbols.y_curl is not None:
                all_symbols.update(self._symbols.y_curl.flatten().tolist())
            all_symbols.update(self._symbols.y_laplacian.tolist() if self._symbols.y_laplacian is not None else [])
            if self._symbols.y_vector_laplacian is not None:
                all_symbols.update(self._symbols.y_vector_laplacian.flatten().tolist())
        for i, expr in enumerate(eq_sys.rhs):
            if not set(expr.free_symbols).issubset(all_symbols):
                raise ValueError(f"invalid symbol in equation {i}")
        dY_indices = eq_sys.equation_indices_by_type(LHS.D_Y_OVER_D_T)
        if self._x_dimension == 0:
            if len(dY_indices) != self._y_dimension:
                raise ValueError("ODE systems must have all equations of type D_Y_OVER_D_T")
        else:
            if len(dY_indices) == 0:
                raise ValueError("At least one equation must be of type D_Y_OVER_D_T for PDEs")

#Concrete differential equations:
class PopulationGrowthEquation(DifferentialEquation):
    """ODE: simple exponential population growth dy/dt = r*y."""
    def __init__(self, r: float = 0.01):
        self._r = r
        super().__init__(0, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        return SymbolicEquationSystem([self._r * self.symbols.y[0]])

class LotkaVolterraEquation(DifferentialEquation):
    """ODE: Lotka-Volterra predator-prey model (2 equations)."""
    def __init__(self, alpha=2.0, beta=0.04, gamma=1.06, delta=0.02):
        if min(alpha, beta, gamma, delta) < 0.0:
            raise ValueError("parameters must be non-negative")
        self._alpha = alpha; self._beta = beta; self._gamma = gamma; self._delta = delta
        super().__init__(0, 2)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y
        eq1 = self._alpha * y[0] - self._beta * y[0] * y[1]
        eq2 = -self._gamma * y[1] + self._delta * y[0] * y[1]
        return SymbolicEquationSystem([eq1, eq2])

class VanDerPolEquation(DifferentialEquation):
    """ODE: Van der Pol oscillator (2 equations)."""
    def __init__(self, mu: float = 1.0):
        self._mu = mu
        super().__init__(0, 2)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y
        eq1 = y[1]
        eq2 = self._mu * (1 - y[0]**2) * y[1] - y[0]
        return SymbolicEquationSystem([eq1, eq2])

class LorenzEquation(DifferentialEquation):
    """ODE: Lorenz system (chaotic attractor)."""
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self._sigma = sigma; self._rho = rho; self._beta = beta
        super().__init__(0, 3)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y
        eq1 = self._sigma * (y[1] - y[0])
        eq2 = y[0] * (self._rho - y[2]) - y[1]
        eq3 = y[0] * y[1] - self._beta * y[2]
        return SymbolicEquationSystem([eq1, eq2, eq3])

class SIREquation(DifferentialEquation):
    """ODE: SIR epidemiology model (Susceptible-Infected-Recovered)."""
    def __init__(self, beta: float = 0.3, gamma: float = 0.1):
        self._beta = beta; self._gamma = gamma
        super().__init__(0, 3)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y
        S, I, R = y[0], y[1], y[2]
        dS = -self._beta * S * I
        dI = self._beta * S * I - self._gamma * I
        dR = self._gamma * I
        return SymbolicEquationSystem([dS, dI, dR])

class DiffusionEquation(DifferentialEquation):
    """PDE: Simple diffusion (heat) equation dY/dt = D * Laplacian(Y)."""
    def __init__(self, diffusion_rate: float = 1.0):
        self._D = diffusion_rate
        super().__init__(1, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        return SymbolicEquationSystem([self._D * self.symbols.y_laplacian[0]])

class ConvectionDiffusionEquation(DifferentialEquation):
    """PDE: Convection-diffusion equation dY/dt = D*Lap(Y) - v * dY/dx."""
    def __init__(self, diffusion_rate: float = 1.0, velocity: float = 1.0):
        self._D = diffusion_rate; self._v = velocity
        super().__init__(1, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y; grad = self.symbols.y_gradient
        return SymbolicEquationSystem([self._D * self.symbols.y_laplacian[0] - self._v * grad[0,0]])

class WaveEquation(DifferentialEquation):
    """PDE: 1D Wave equation d2Y/dt2 = c^2 * d2Y/dx2 (represented as first order system)."""
    def __init__(self, c: float = 1.0):
        self._c = c
        super().__init__(1, 2, all_vector_field_indices=None)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        #Represent wave eq as first order system: y0 = displacement, y1 = velocity
        y = self.symbols.y; lap = self.symbols.y_laplacian
        #dy0/dt = y1; dy1/dt = c^2 * d2y0/dx2
        return SymbolicEquationSystem([y[1], self._c**2 * lap[0]], [LHS.D_Y_OVER_D_T, LHS.D_Y_OVER_D_T])

class BurgersEquation(DifferentialEquation):
    """PDE: 1D viscous Burgers' equation (nonlinear convection-diffusion)."""
    def __init__(self, nu: float = 0.1):
        self._nu = nu
        super().__init__(1, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        y = self.symbols.y; grad = self.symbols.y_gradient
        #dy/dt = -y * dy/dx + nu * d2y/dx2
        return SymbolicEquationSystem([-y[0] * grad[0,0] + self._nu * self.symbols.y_laplacian[0]])

class CahnHilliardEquation(DifferentialEquation):
    """PDE: 1D Cahn-Hilliard equation (fourth-order nonlinear diffusion)."""
    def __init__(self, mobility: float = 1.0):
        self._M = mobility
        super().__init__(1, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        #Represent as first order in time for order parameter phi: dphi/dt = M * d2(mu)/dx2, mu = f'(phi) - gamma * d2phi/dx2
        #For simplicity, consider f(phi)=phi^3 - phi.
        phi = self.symbols.y[0]; lap_phi = self.symbols.y_laplacian[0]
        chem_pot = phi**3 - phi - lap_phi
        eq = self._M * (chem_pot.expand())
        #Using LHS Y_LAPLACIAN to hint an extra derivative? But we treat it as D_Y/dt actually
        return SymbolicEquationSystem([eq])
class ShallowWaterEquation(DifferentialEquation):
    """
    A system of partial differential equations providing a non-conservative
    model of fluid flow below a pressure surface.
    """

    def __init__(
        self,
        h: float,
        b: float = 0.01,
        v: float = 0.1,
        f: float = 0.0,
        g: float = 9.80665,
    ):
        """
        h: mean height of pressure surface
        b: viscous drag coefficient
        v: kinematic viscosity coefficient
        f: Coriolis coefficient
        g: gravitational acceleration coefficient
        """
        self._h = h
        self._b = b
        self._v = v
        self._f = f
        self._g = g

        super(ShallowWaterEquation, self).__init__(2, 3, [(1, 2)])

    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        return SymbolicEquationSystem(
            [
                -self._h * self._symbols.y_divergence[1, 2]
                - self._symbols.y[0] * self._symbols.y_gradient[1, 0]
                - self._symbols.y[1] * self._symbols.y_gradient[0, 0]
                - self._symbols.y[0] * self._symbols.y_gradient[2, 1]
                - self._symbols.y[2] * self._symbols.y_gradient[0, 1],
                self._v * self._symbols.y_laplacian[1]
                - self._symbols.y[1] * self._symbols.y_gradient[1, 0]
                - self._symbols.y[2] * self._symbols.y_gradient[1, 1]
                - self._g * self._symbols.y_gradient[0, 0]
                - self._b * self._symbols.y[1]
                + self._f * self._symbols.y[2],
                self._v * self._symbols.y_laplacian[2]
                - self._symbols.y[1] * self._symbols.y_gradient[2, 0]
                - self._symbols.y[2] * self._symbols.y_gradient[2, 1]
                - self._g * self._symbols.y_gradient[0, 1]
                - self._b * self._symbols.y[2]
                - self._f * self._symbols.y[1],
            ]
        )


class NavierStokesEquation(DifferentialEquation):
    """
    A system of four partial differential equations modelling velocity,
    vorticity, and stream function of incompressible fluids in two spatial
    dimensions.
    """

    def __init__(self, re: float = 4000.0):
        """
        re: Reynolds number
        """
        self._re = re
        super(NavierStokesEquation, self).__init__(2, 4, [(2, 3)])

    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        return SymbolicEquationSystem(
            [
                (1.0 / self._re) * self._symbols.y_laplacian[0]
                - np.dot(
                    self._symbols.y[2:],
                    self._symbols.y_gradient[0, :],
                ),
                -self._symbols.y[0],
                self._symbols.y_gradient[1, 1],
                -self._symbols.y_gradient[1, 0],
            ],
            [LHS.D_Y_OVER_D_T, LHS.Y_LAPLACIAN, LHS.Y, LHS.Y],
        )

class NBodyGravitationalEquation(DifferentialEquation):
    """ODE: N-body gravitational system in 2D or 3D (as ODE system for positions and velocities)."""
    def __init__(self, masses: Sequence[float], G: float = 1.0):
        self._masses = np.array(masses, dtype=float)
        self._G = G
        n_objects = len(masses)
        dim = 2  #assume 2D for simplicity (could be extended to 3D)
        #y_dimension = 2 * n_objects * dim (positions and velocities for each object in each dimension)
        y_dim = 2 * n_objects * dim
        super().__init__(0, y_dim)
        self._n_objects = n_objects
        self._dim = dim
    @property
    def masses(self) -> Sequence[float]:
        return self._masses.copy()
    @property
    def spatial_dimension(self) -> int:
        return self._dim
    @property
    def n_objects(self) -> int:
        return self._n_objects
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        #Not using sympy for full gravitational equations due to complexity; provide numeric logic in operator instead.
        #Represent symbolically as zeros; actual numeric integration handled externally (e.g., in NBodyPlot).
        return SymbolicEquationSystem([0] * self._y_dimension)

class BlackScholesEquation(DifferentialEquation):
    """PDE: Black-Scholes equation for option pricing (1 underlying asset)."""
    def __init__(self, sigma: float = 0.2, r: float = 0.05):
        self._sigma = sigma; self._r = r
        super().__init__(1, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        #Black-Scholes PDE: dV/dt = 0.5*sigma^2*S^2 * d2V/dS2 + r*S * dV/dS - r*V
        grad = self.symbols.y_gradient; lap = self.symbols.y_laplacian
        #But S (the underlying price) is a spatial variable, not part of y or x in our framework. We'll handle it separately.
        #Represent symbolically as Laplacian and gradient terms multiplied by parameter functions S (which is x coordinate)
        #Here we cannot directly incorporate S in this symbolic system easily. We'll use solver functions directly.
        return SymbolicEquationSystem([0 * self.symbols.y[0]])

class MultiDimensionalBlackScholesEquation(DifferentialEquation):
    """PDE: Multi-dimensional Black-Scholes (uncoupled multiple underlyings)."""
    def __init__(self, sigmas: Sequence[float], r: float = 0.05):
        self._sigmas = list(sigmas); self._r = r
        dim = len(sigmas)
        super().__init__(dim, 1)
    @property
    def symbolic_equation_system(self) -> SymbolicEquationSystem:
        #Represent similarly to BlackScholesEquation; handle actual calculation in solver.
        return SymbolicEquationSystem([0 * self.symbols.y[0]])

#Constrained Problem and Initial Conditions
#----------------------------
class ConstrainedProblem:
    """
    Represents a differential equation (ODE or PDE) with spatial constraints (mesh and boundary conditions).
    """
    def __init__(self, diff_eq: DifferentialEquation, mesh: Optional[Mesh] = None,
                 boundary_conditions: Optional[Sequence[Tuple[BoundaryCondition, BoundaryCondition]]] = None):
        self._diff_eq = diff_eq
        if diff_eq.x_dimension:
            if mesh is None:
                raise ValueError("mesh cannot be None for PDEs")
            if mesh.dimensions != diff_eq.x_dimension:
                raise ValueError(f"mesh dimensions ({mesh.dimensions}) must match differential equation spatial dimensions ({diff_eq.x_dimension})")
            if boundary_conditions is None:
                raise ValueError("boundary conditions cannot be None for PDEs")
            if len(boundary_conditions) != diff_eq.x_dimension:
                raise ValueError("number of boundary condition pairs must match spatial dimensions")
            self._mesh = mesh
            self._boundary_conditions = tuple(boundary_conditions)
            #Precompute static constraints
            self._y_vertices_shape = mesh.vertices_shape + (diff_eq.y_dimension,)
            self._y_cells_shape = mesh.cells_shape + (diff_eq.y_dimension,)
            self._are_all_bcs_static = bool(np.all([bc_low.is_static and bc_high.is_static for (bc_low, bc_high) in boundary_conditions]))
            self._are_there_bcs_on_y = bool(np.any([bc_low.has_y_condition or bc_high.has_y_condition for (bc_low, bc_high) in boundary_conditions]))
            #Create static constraint arrays for boundaries
            self._boundary_vertex_constraints = self.create_boundary_constraints(vertex_oriented=True)
            if self._boundary_vertex_constraints[0] is not None: self._boundary_vertex_constraints[0].setflags(write=False)
            if self._boundary_vertex_constraints[1] is not None: self._boundary_vertex_constraints[1].setflags(write=False)
            self._boundary_cell_constraints = self.create_boundary_constraints(vertex_oriented=False)
            if self._boundary_cell_constraints[0] is not None: self._boundary_cell_constraints[0].setflags(write=False)
            if self._boundary_cell_constraints[1] is not None: self._boundary_cell_constraints[1].setflags(write=False)
            self._y_vertex_constraints = self.create_y_vertex_constraints(self._boundary_vertex_constraints[0])
            if self._y_vertex_constraints is not None: self._y_vertex_constraints.setflags(write=False)
        else:
            self._mesh = None
            self._boundary_conditions = None
            self._y_vertices_shape = self._y_cells_shape = (diff_eq.y_dimension,)
            self._are_all_bcs_static = False
            self._are_there_bcs_on_y = False
            self._boundary_vertex_constraints = None
            self._boundary_cell_constraints = None
            self._y_vertex_constraints = None
    @property
    def differential_equation(self) -> DifferentialEquation:
        return self._diff_eq
    @property
    def mesh(self) -> Optional[Mesh]:
        return self._mesh
    @property
    def boundary_conditions(self) -> Optional[Tuple[Tuple[BoundaryCondition, BoundaryCondition], ...]]:
        return self._boundary_conditions
    @property
    def y_vertices_shape(self) -> Tuple[int, ...]:
        return self._y_vertices_shape
    @property
    def y_cells_shape(self) -> Tuple[int, ...]:
        return self._y_cells_shape
    @property
    def are_all_boundary_conditions_static(self) -> bool:
        return bool(self._are_all_bcs_static)
    @property
    def are_there_boundary_conditions_on_y(self) -> bool:
        return bool(self._are_there_bcs_on_y)
    @property
    def static_boundary_vertex_constraints(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._boundary_vertex_constraints
    @property
    def static_boundary_cell_constraints(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._boundary_cell_constraints
    @property
    def static_y_vertex_constraints(self) -> Optional[np.ndarray]:
        return self._y_vertex_constraints
    def y_shape(self, vertex_oriented: Optional[bool] = None) -> Tuple[int, ...]:
        return self._y_vertices_shape if (vertex_oriented or vertex_oriented is None) else self._y_cells_shape
    def static_boundary_constraints(self, vertex_oriented: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._boundary_vertex_constraints if vertex_oriented else self._boundary_cell_constraints
    def create_y_vertex_constraints(self, y_boundary_vertex_constraints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self._diff_eq.x_dimension or y_boundary_vertex_constraints is None:
            return None
        diff_eq = self._diff_eq
        y_constraints = np.empty(diff_eq.y_dimension, dtype=object)
        y_element = np.empty(self._y_vertices_shape[:-1] + (1,))
        slicer: List[Union[int, slice]] = [slice(None)] * len(self._y_vertices_shape)
        for y_ind in range(diff_eq.y_dimension):
            y_element.fill(np.nan)
            for axis in range(diff_eq.x_dimension):
                for bc_ind, bc in enumerate(y_boundary_vertex_constraints[axis, y_ind] if y_boundary_vertex_constraints is not None else []):
                    if bc is None: continue
                    slicer[axis] = slice(-1, None) if bc_ind else slice(0, 1)
                    bc.apply(y_element[tuple(slicer)])
                slicer[axis] = slice(None)
            mask = ~np.isnan(y_element)
            value = y_element[mask]
            y_constraints[y_ind] = Constraint(value, mask)
        return y_constraints
    def create_boundary_constraints(self, vertex_oriented: bool, t: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        diff_eq = self._diff_eq
        if not diff_eq.x_dimension:
            return None, None
        all_idx_coords = self._mesh.all_index_coordinates(vertex_oriented)
        all_y_pairs = np.empty((diff_eq.x_dimension, diff_eq.y_dimension), dtype=object)
        all_dy_pairs = np.empty((diff_eq.x_dimension, diff_eq.y_dimension), dtype=object)
        for axis, (bc_lower, bc_upper) in enumerate(self._boundary_conditions):
            y_pairs = []
            dy_pairs = []
            for bc_ind, bc in enumerate((bc_lower, bc_upper)):
                if not bc.is_static and t is None:
                    #dynamic BC but no time provided
                    y_pairs.append([None] * diff_eq.y_dimension)
                    dy_pairs.append([None] * diff_eq.y_dimension)
                elif bc.is_static and self._boundary_vertex_constraints is not None:
                    #static BC, use precomputed
                    y_pairs.append([self._boundary_vertex_constraints[0][axis, i][bc_ind] for i in range(diff_eq.y_dimension)])
                    dy_pairs.append([self._boundary_vertex_constraints[1][axis, i][bc_ind] for i in range(diff_eq.y_dimension)])
                else:
                    #compute on fly
                    slicer = [slice(None)] * all_idx_coords.ndim
                    slicer[axis] = slice(-1, None) if bc_ind else slice(0, 1)
                    boundary_coords = np.copy(all_idx_coords[tuple(slicer)])
                    boundary_coords[..., axis] = self._mesh.vertex_axis_coordinates[axis][(-1 if bc_ind else 0)]
                    flat_coords = boundary_coords.reshape((-1, diff_eq.x_dimension))
                    y_vals = bc.y_condition(flat_coords, t)
                    dy_vals = bc.d_y_condition(flat_coords, t)
                    if y_vals.shape != (len(flat_coords), diff_eq.y_dimension):
                        raise ValueError("boundary condition y output shape mismatch")
                    if dy_vals.shape != (len(flat_coords), diff_eq.y_dimension):
                        raise ValueError("boundary condition dy output shape mismatch")
                    y_vals = y_vals.reshape(boundary_coords.shape[:-1] + (diff_eq.y_dimension,))
                    dy_vals = dy_vals.reshape(boundary_coords.shape[:-1] + (diff_eq.y_dimension,))
                    #Create constraints for each component of y
                    y_comp_constraints = []
                    dy_comp_constraints = []
                    for yi in range(diff_eq.y_dimension):
                        val = y_vals[..., yi:yi+1]; mask = ~np.isnan(val)
                        y_comp_constraints.append(Constraint(val[mask], mask))
                        val2 = dy_vals[..., yi:yi+1]; mask2 = ~np.isnan(val2)
                        dy_comp_constraints.append(Constraint(val2[mask2], mask2))
                    y_pairs.append(y_comp_constraints)
                    dy_pairs.append(dy_comp_constraints)
            for i in range(diff_eq.y_dimension):
                all_y_pairs[axis, i] = (y_pairs[0][i], y_pairs[1][i])
                all_dy_pairs[axis, i] = (dy_pairs[0][i], dy_pairs[1][i])
        return all_y_pairs, all_dy_pairs

#Initial conditions:
VectorizedInitialConditionFunction = Callable[[Optional[np.ndarray]], np.ndarray]

class InitialCondition:
    """Abstract base class for initial conditions."""
    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError
    def discrete_y_0(self, vertex_oriented: Optional[bool] = None) -> np.ndarray:
        raise NotImplementedError

class DiscreteInitialCondition(InitialCondition):
    """Initial condition defined by a fixed array of values on mesh."""
    def __init__(self, cp: ConstrainedProblem, y0: np.ndarray, vertex_oriented: Optional[bool] = None, interpolation_method: str = "linear"):
        if cp.differential_equation.x_dimension and vertex_oriented is None:
            raise ValueError("vertex orientation must be defined for PDEs")
        if y0.shape != cp.y_shape(vertex_oriented):
            raise ValueError(f"discrete initial value shape {y0.shape} must match constrained problem solution shape {cp.y_shape(vertex_oriented)}")
        self._cp = cp
        self._y0 = np.copy(y0)
        self._vertex_oriented = vertex_oriented
        self._interp_method = interpolation_method
        if vertex_oriented:
            apply_constraints_along_last_axis(cp.static_y_vertex_constraints, self._y0)
    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        if not self._cp.differential_equation.x_dimension:
            return np.copy(self._y0)
        return np.array(interpn(self._cp.mesh.axis_coordinates(self._vertex_oriented), self._y0, x, method=self._interp_method, bounds_error=False, fill_value=None))
    def discrete_y_0(self, vertex_oriented: Optional[bool] = None) -> np.ndarray:
        if vertex_oriented is None:
            vertex_oriented = self._vertex_oriented
        if not self._cp.differential_equation.x_dimension or vertex_oriented == self._vertex_oriented:
            return np.copy(self._y0)
        #Need to interpolate to new orientation
        y0 = self.y_0(self._cp.mesh.all_index_coordinates(vertex_oriented))
        if vertex_oriented:
            apply_constraints_along_last_axis(self._cp.static_y_vertex_constraints, y0)
        return y0

class ConstantInitialCondition(DiscreteInitialCondition):
    """Initial condition where each component of y has a constant value (spatially uniform)."""
    def __init__(self, cp: ConstrainedProblem, constant_y0s: Sequence[float]):
        if len(constant_y0s) != cp.differential_equation.y_dimension:
            raise ValueError("length of constant initial values must match number of y components")
        #Create an array with those values
        shape = cp.y_shape(True)
        ic_array = np.empty(shape, dtype=float)
        for i, val in enumerate(constant_y0s):
            ic_array[..., i] = val
        super().__init__(cp, ic_array, vertex_oriented=True)

class ContinuousInitialCondition(InitialCondition):
    """Initial condition defined by a function over space."""
    def __init__(self, cp: ConstrainedProblem, y0_func: VectorizedInitialConditionFunction, multipliers: Optional[Sequence[float]] = None):
        diff_eq = cp.differential_equation
        if multipliers is not None:
            if len(multipliers) != diff_eq.y_dimension:
                raise ValueError("length of multipliers must match number of y dimensions")
            self._multipliers = np.array(multipliers)
        else:
            self._multipliers = np.ones(diff_eq.y_dimension)
        self._cp = cp
        self._y0_func = y0_func
        #Precompute discretized initial values at vertices and cells
        self._discrete_y0_vertices = self._create_discrete_y0(True)
        self._discrete_y0_cells = self._create_discrete_y0(False)
    def y_0(self, x: Optional[np.ndarray]) -> np.ndarray:
        return np.multiply(self._y0_func(x), self._multipliers)
    def discrete_y_0(self, vertex_oriented: Optional[bool] = None) -> np.ndarray:
        return np.copy(self._discrete_y0_vertices if vertex_oriented else self._discrete_y0_cells)
    def _create_discrete_y0(self, vertex_oriented: bool) -> np.ndarray:
        diff_eq = self._cp.differential_equation
        if not diff_eq.x_dimension:
            y0_val = np.array(self.y_0(None))
            if y0_val.shape != self._cp.y_shape():
                raise ValueError("initial condition function output shape mismatch")
            return y0_val
        x_points = self._cp.mesh.all_index_coordinates(vertex_oriented, flatten=True)
        y0_vals = self.y_0(x_points)
        if y0_vals.shape != (len(x_points), diff_eq.y_dimension):
            raise ValueError("initial condition function output shape mismatch")
        y0_grid = y0_vals.reshape(self._cp.y_shape(vertex_oriented))
        if vertex_oriented:
            apply_constraints_along_last_axis(self._cp.static_y_vertex_constraints, y0_grid)
        return y0_grid
    def _convert_coordinates_to_cartesian(self, x: np.ndarray) -> np.ndarray:
        cartesian_coords = to_cartesian_coordinates([x[:, i] for i in range(x.shape[1])], self._cp.mesh.coordinate_system_type)
        return np.stack(cartesian_coords, axis=-1)

class GaussianInitialCondition(ContinuousInitialCondition):
    """Initial condition defined by one or more Gaussian distributions."""
    def __init__(self, cp: ConstrainedProblem, means_and_covs: Sequence[Tuple[np.ndarray, np.ndarray]], multipliers: Optional[Sequence[float]] = None):
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            raise ValueError("constrained problem must be a PDE")
        if len(means_and_covs) != diff_eq.y_dimension:
            raise ValueError("number of Gaussian definitions must match number of y dimensions")
        for mean, cov in means_and_covs:
            if mean.shape != (diff_eq.x_dimension,) or cov.shape != (diff_eq.x_dimension, diff_eq.x_dimension):
                raise ValueError("mean or covariance shape incompatible with spatial dimensions")
        self._means_and_covs = deepcopy(means_and_covs)
        super().__init__(cp, self._y0_func, multipliers)
    def _y0_func(self, x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            raise ValueError("x cannot be None for GaussianInitialCondition")
        #convert x to Cartesian for PDF evaluation
        cartesian_x = self._convert_coordinates_to_cartesian(x)
        values = np.empty((len(x), self._cp.differential_equation.y_dimension))
        for i, (mean, cov) in enumerate(self._means_and_covs):
            values[:, i] = (1/np.sqrt(((2*np.pi)**len(mean)) * np.linalg.det(cov)) * np.exp(-0.5 * np.sum((cartesian_x - mean) * np.linalg.solve(cov, (cartesian_x - mean).T).T, axis=1)))
        return values

class MarginalBetaProductInitialCondition(ContinuousInitialCondition):
    """Initial condition defined by a product of Beta distribution PDFs along each axis."""
    def __init__(self, cp: ConstrainedProblem, all_alphas_and_betas: Sequence[Sequence[Tuple[float, float]]], multipliers: Optional[Sequence[float]] = None):
        diff_eq = cp.differential_equation
        if len(all_alphas_and_betas) != diff_eq.y_dimension:
            raise ValueError("length of beta parameters list must match number of y dimensions")
        if any(len(ab) != diff_eq.x_dimension for ab in all_alphas_and_betas):
            raise ValueError("each beta parameter sequence length must match spatial dimensions")
        self._all_alphas_and_betas = deepcopy(all_alphas_and_betas)
        super().__init__(cp, self._y0_func, multipliers)
    def _y0_func(self, x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            raise ValueError("x cannot be None for MarginalBetaProductInitialCondition")
        cartesian_x = self._convert_coordinates_to_cartesian(x)
        try:
            from scipy.stats import beta
        except ImportError:
            raise ImportError("scipy is required for MarginalBetaProductInitialCondition")
        values = []
        for alphas_betas in self._all_alphas_and_betas:
            comp_pdf = np.ones(len(x))
            for axis, (a, b) in enumerate(alphas_betas):
                comp_pdf *= beta.pdf(cartesian_x[:, axis], a, b)
            values.append(comp_pdf)
        return np.vstack(values).T

def vectorize_ic_function(ic_function: Callable[[Optional[Sequence[float]]], Sequence[float]]) -> VectorizedInitialConditionFunction:
    """Convert an IC function that operates on a single coordinate to operate on array of coordinate sequences."""
    def vectorized(x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return np.array(ic_function(None))
        values = []
        for i in range(len(x)):
            values.append(ic_function(x[i]))
        return np.array(values)
    return vectorized

#Initial Value Problem and Solution
#-----------------------

TemporalDomainInterval = Tuple[float, float]


class InitialValueProblem:
    """
    A representation of an initial value problem around a boundary value
    problem.
    """

    def __init__(
        self,
        cp: ConstrainedProblem,
        t_interval: TemporalDomainInterval,
        initial_condition: InitialCondition,
        exact_y: Optional[
            Callable[
                [InitialValueProblem, float, Optional[np.ndarray]], np.ndarray
            ]
        ] = None,
    ):
        """
        :param cp: the constrained problem to base the initial value problem on
        :param t_interval: the bounds of the time domain of the initial value
            problem
        :param initial_condition: the initial condition of the problem
        :param exact_y: the function returning the exact solution to the
            initial value problem at time step t and points x. If it is None,
            the problem is assumed to have no analytical solution.
        """
        if t_interval[0] > t_interval[1]:
            raise ValueError(
                f"lower bound of time interval ({t_interval[0]}) cannot be "
                f"greater than its upper bound ({t_interval[1]})"
            )

        self._cp = cp
        self._t_interval = t_interval
        self._initial_condition = initial_condition
        self._exact_y = exact_y

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the IVP is based on.
        """
        return self._cp

    @property
    def t_interval(self) -> TemporalDomainInterval:
        """
        The bounds of the temporal domain of the differential equation.
        """
        return self._t_interval

    @property
    def initial_condition(self) -> InitialCondition:
        """
        The initial condition of the IVP.
        """
        return self._initial_condition

    @property
    def has_exact_solution(self) -> bool:
        """
        Whether the differential equation has an analytic solution
        """
        return self._exact_y is not None

    def exact_y(self, t: float, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the points in the non-temporal domain. If the differential
            equation is an ODE, it is None.
        :return: the value of y(t, x) or y(t) if it is an ODE.
        """
        if not self.has_exact_solution:
            raise RuntimeError(
                "exact solution of initial value problem undefined"
            )

        return self._exact_y(self, t, x)


class Solution:
    """Solution to an initial value problem (time series of y)."""
    def __init__(self, ivp: InitialValueProblem, t_coordinates: np.ndarray, discrete_y: np.ndarray, vertex_oriented: Optional[bool] = None, d_t: Optional[float] = None):
        if t_coordinates.ndim != 1:
            raise ValueError("t_coordinates must be 1D")
        if len(t_coordinates) == 0:
            raise ValueError("no time coordinates provided")
        if ivp.constrained_problem.differential_equation.x_dimension and vertex_oriented is None:
            raise ValueError("vertex orientation must be specified for PDE solutions")
        y_shape = ivp.constrained_problem.y_shape(vertex_oriented)
        if discrete_y.shape != ((len(t_coordinates),) + y_shape):
            raise ValueError(f"expected solution shape {((len(t_coordinates),) + y_shape)} but got {discrete_y.shape}")
        self._ivp = ivp
        self._t_coordinates = np.copy(t_coordinates)
        self._discrete_y = np.copy(discrete_y)
        self._vertex_oriented = vertex_oriented
        self._t_coordinates.setflags(write=False)
        if d_t is None:
            d_t = 0.0 if len(t_coordinates) == 1 else t_coordinates[1] - t_coordinates[0]
        self._d_t = d_t
    @property
    def initial_value_problem(self) -> InitialValueProblem:
        return self._ivp
    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented
    @property
    def d_t(self) -> float:
        return self._d_t
    @property
    def t_coordinates(self) -> np.ndarray:
        return self._t_coordinates
    def y(self, x: Optional[np.ndarray] = None, interpolation_method: str = "linear") -> np.ndarray:
        cp = self._ivp.constrained_problem
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            return np.copy(self._discrete_y)
        #Interpolate in space at all time steps
        result = interpn(cp.mesh.axis_coordinates(self._vertex_oriented), np.moveaxis(self._discrete_y, 0, -2), x, method=interpolation_method, bounds_error=False, fill_value=None)
        result = np.moveaxis(result, -2, 0)
        result = result.reshape((len(self._t_coordinates),) + x.shape[:-1] + (diff_eq.y_dimension,))
        return np.ascontiguousarray(result)
    def discrete_y(self, vertex_oriented: Optional[bool] = None, interpolation_method: str = "linear") -> np.ndarray:
        if vertex_oriented is None:
            vertex_oriented = self._vertex_oriented
        cp = self._ivp.constrained_problem
        if not cp.differential_equation.x_dimension or vertex_oriented == self._vertex_oriented:
            return np.copy(self._discrete_y)
        x_coords = cp.mesh.all_index_coordinates(vertex_oriented)
        disc = self.y(x_coords, interpolation_method)
        if vertex_oriented:
            apply_constraints_along_last_axis(cp.static_y_vertex_constraints, disc)
        return disc
    def diff(self, solutions: Sequence['Solution'], atol: float = 1e-8) -> 'Diffs':
        if len(solutions) == 0:
            raise ValueError("at least one solution must be provided")
        matching_time_points = []
        all_diffs: List[List[np.ndarray]] = [[] for _ in solutions]
        time_series = [self._t_coordinates] + [sol.t_coordinates for sol in solutions]
        time_steps = [self._d_t] + [sol.d_t for sol in solutions]
        #find solution with fewest time points
        min_index = min(range(len(time_series)), key=lambda i: len(time_series[i]))
        shortest_times = time_series[min_index]
        for i, t in enumerate(shortest_times):
            indices = []
            all_match = True
            for j, times in enumerate(time_series):
                if j == min_index:
                    indices.append(i)
                else:
                    idx = int(round((t - times[0]) / time_steps[j]))
                    if 0 <= idx < len(times) and np.isclose(t, times[idx], atol=atol, rtol=0.0):
                        indices.append(idx)
                    else:
                        all_match = False
                        break
            if all_match:
                matching_time_points.append(t)
                for j, sol in enumerate(solutions):
                    diff = sol.discrete_y(self._vertex_oriented)[indices[j+1]] - self._discrete_y[indices[0]]
                    all_diffs[j].append(diff)
        matching_time_points = np.array(matching_time_points)
        diff_arrays = [np.array(dlist) for dlist in all_diffs]
        return Diffs(matching_time_points, diff_arrays)
    def generate_plots(self, **kwargs) -> Generator['Plot', None, None]:
        cp = self._ivp.constrained_problem
        diff_eq = cp.differential_equation
        if diff_eq.x_dimension > 3:
            return
        if diff_eq.x_dimension == 0:
            if isinstance(diff_eq, NBodyGravitationalEquation):
                yield NBodyPlot(self._discrete_y, diff_eq, **kwargs)
            else:
                yield TimePlot(self._discrete_y, self._t_coordinates, **kwargs)
                if 2 <= diff_eq.y_dimension <= 3:
                    yield PhaseSpacePlot(self._discrete_y, **kwargs)
        else:
            vector_index_set: Set[int] = set()
            if diff_eq.x_dimension > 1:
                all_vec_indices = diff_eq.all_vector_field_indices
                if all_vec_indices is not None:
                    for indices in all_vec_indices:
                        vector_index_set.update(indices)
                        vector_field = self._discrete_y[..., indices]
                        yield QuiverPlot(vector_field, cp.mesh, self._vertex_oriented, **kwargs)
                        if diff_eq.x_dimension == 2:
                            yield StreamPlot(vector_field, cp.mesh, self._vertex_oriented, **kwargs)
            for i in range(diff_eq.y_dimension):
                if i in vector_index_set:
                    continue
                scalar_field = self._discrete_y[..., i:i+1]
                if diff_eq.x_dimension == 1:
                    yield SpaceLinePlot(scalar_field, cp.mesh, self._vertex_oriented, **kwargs)
                elif diff_eq.x_dimension == 2:
                    yield ContourPlot(scalar_field, cp.mesh, self._vertex_oriented, **kwargs)
                    yield SurfacePlot(scalar_field, cp.mesh, self._vertex_oriented, **kwargs)
                else:
                    yield ScatterPlot(scalar_field, cp.mesh, self._vertex_oriented, **kwargs)

class Diffs(NamedTuple):
    matching_time_points: np.ndarray
    differences: Sequence[np.ndarray]

#Plotting Classes
#-----------
#Matplotlib-based classes for visualizing solutions (2D/3D).
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

class Plot:
    """Base class for plots of solutions."""
    def __init__(self, figure: Figure):
        self._figure = figure
    def show(self) -> 'Plot':
        plt.show(); return self
    def save(self, file_path: str, extension: str = "png", **kwargs) -> 'Plot':
        self._figure.savefig(f"{file_path}.{extension}", **kwargs); return self
    def close(self):
        plt.close(self._figure)

class AnimatedPlot(Plot):
    """Base class for animated plots."""
    def __init__(self, figure: Figure, init_func: Callable[[], None], update_func: Callable[[int], None], n_time_steps: int, n_frames: int, interval: int):
        super().__init__(figure)
        frames = np.linspace(0, n_time_steps - 1, n_frames, dtype=int)
        self._animation = plt.FuncAnimation(figure, func=update_func, init_func=init_func, frames=frames, interval=interval)
    def save(self, file_path: str, extension: str = "gif", **kwargs) -> Plot:
        self._animation.save(f"{file_path}.{extension}", **kwargs)
        return self
    @staticmethod
    def _verify_pde_solution_shape_matches_problem(y: np.ndarray, mesh: Mesh, vertex_oriented: bool, expected_x_dims: Union[int, Tuple[int, int]], is_vector_field: bool):
        dims = mesh.dimensions
        if isinstance(expected_x_dims, int):
            if dims != expected_x_dims:
                raise ValueError(f"mesh must be {expected_x_dims}-dimensional")
        else:
            if not (expected_x_dims[0] <= dims <= expected_x_dims[1]):
                raise ValueError(f"mesh must be between {expected_x_dims[0]} and {expected_x_dims[1]} dimensions")
        if y.ndim != dims + 2:
            raise ValueError(f"y array must have rank mesh.dimensions+2 (got {y.ndim})")
        if y.shape[1:-1] != mesh.shape(vertex_oriented):
            raise ValueError(f"y shape {y.shape} not compatible with mesh shape {mesh.shape(vertex_oriented)}")
        if is_vector_field:
            if y.shape[-1] != dims:
                raise ValueError(f"vector field must have {dims} components, got {y.shape[-1]}")
        else:
            if y.shape[-1] != 1:
                raise ValueError(f"scalar field must have 1 component, got {y.shape[-1]}")

class TimePlot(Plot):
    """Plot y vs t for ODE solutions."""
    def __init__(self, y: np.ndarray, t: np.ndarray, legend_location: Optional[str] = None, **_):
        if y.ndim != 2 or t.ndim != 1:
            raise ValueError("TimePlot expects y as 2D array and t as 1D array")
        if y.shape[0] != t.shape[0]:
            raise ValueError("Time axis length mismatch between y and t")
        fig, ax = plt.subplots()
        for i in range(y.shape[1]):
            ax.plot(t, y[:, i], label=f"y{i}")
        ax.set_xlabel("t"); ax.set_ylabel("y")
        if legend_location: ax.legend(loc=legend_location)
        fig.tight_layout()
        super().__init__(fig)

class PhaseSpacePlot(Plot):
    """Phase-space plot for 2D or 3D ODE solutions."""
    def __init__(self, y: np.ndarray, **_):
        if y.ndim != 2 or not (2 <= y.shape[1] <= 3):
            raise ValueError("PhaseSpacePlot expects y with 2 or 3 components")
        fig = plt.figure()
        if y.shape[1] == 2:
            ax = fig.add_subplot()
            ax.plot(y[:, 0], y[:, 1])
            ax.set_xlabel("y0"); ax.set_ylabel("y1")
            ax.axis("equal")
        else:
            ax = fig.add_subplot(projection="3d")
            ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
            ax.set_xlabel("y0"); ax.set_ylabel("y1"); ax.set_zlabel("y2")
            #scale axes equally
            ax.set_box_aspect((np.ptp(y[:,0]), np.ptp(y[:,1]), np.ptp(y[:,2])))
        super().__init__(fig)

class NBodyPlot(AnimatedPlot):
    """Animated scatter plot for N-body gravitational simulations (2D or 3D)."""
    def __init__(self, y: np.ndarray, diff_eq: NBodyGravitationalEquation, n_frames: int = 100, interval: int = 100, color_map = plt.cm.cividis, smallest_marker_size: float = 10.0, draw_trajectory: bool = True, trajectory_line_style: str = ":", trajectory_line_width: float = 0.5, span_scaling_factor: float = 0.25, **_):
        #y shape: (time_steps, state_dim). For N-body state, positions and velocities.
        if y.ndim != 2:
            raise ValueError("NBodyPlot expects y as 2D array")
        n_obj = diff_eq.n_objects; dim = diff_eq.spatial_dimension
        #Extract positions from y (assuming ordering [x1,y1,...,xN,yN, vx1,vy1,...])
        coords = y[:, :n_obj*dim]
        coords = coords.reshape(y.shape[0], n_obj, dim)
        #Determine plot bounds
        x_coords = coords[..., 0]; y_coords = coords[..., 1];
        x_max, x_min = float(np.max(x_coords)), float(np.min(x_coords))
        y_max, y_min = float(np.max(y_coords)), float(np.min(y_coords))
        x_span = x_max - x_min; y_span = y_max - y_min
        x_max += span_scaling_factor * x_span; x_min -= span_scaling_factor * x_span
        y_max += span_scaling_factor * y_span; y_min -= span_scaling_factor * y_span
        if dim == 3:
            z_coords = coords[..., 2]
            z_max, z_min = float(np.max(z_coords)), float(np.min(z_coords))
            z_span = z_max - z_min
            z_max += span_scaling_factor * z_span; z_min -= span_scaling_factor * z_span
        #Marker sizes based on masses
        masses = np.array(diff_eq.masses)
        scaled_masses = (smallest_marker_size / np.min(masses)) * masses
        radii = (3.0 * scaled_masses / (4.0 * np.pi)) ** (1.0/3.0)
        marker_sizes = np.pi * radii**2
        colors = color_map(np.linspace(0.0, 1.0, n_obj))
        self._scatter_plot = None
        self._line_plots = None
        style = "dark_background"
        with plt.style.context(style):
            fig = plt.figure()
            ax = fig.add_subplot(projection=("3d" if dim == 3 else None))
        if dim == 2:
            coords2d = np.stack((x_coords, y_coords), axis=2)
            def init_plot():
                with plt.style.context(style):
                    ax.clear()
                    self._scatter_plot = ax.scatter(x_coords[0], y_coords[0], s=marker_sizes, c=colors)
                    if draw_trajectory:
                        self._line_plots = []
                        for i in range(n_obj):
                            line, = ax.plot(x_coords[:1, i], y_coords[:1, i], color=colors[i], linestyle=trajectory_line_style, linewidth=trajectory_line_width)
                            self._line_plots.append(line)
                    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.axis("scaled")
                    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            def update_plot(frame: int):
                self._scatter_plot.set_offsets(coords2d[frame])
                if draw_trajectory:
                    for i in range(n_obj):
                        line = self._line_plots[i]
                        line.set_xdata(x_coords[:frame+1, i]); line.set_ydata(y_coords[:frame+1, i])
        else:  #3D
            z_coords = coords[..., 2]
            def init_plot():
                with plt.style.context(style):
                    ax.clear()
                    self._scatter_plot = ax.scatter(x_coords[0], y_coords[0], z_coords[0], s=marker_sizes, c=colors, depthshade=False)
                    if draw_trajectory:
                        self._line_plots = []
                        for i in range(n_obj):
                            line, = ax.plot(x_coords[:1, i], y_coords[:1, i], z_coords[:1, i], color=colors[i], linestyle=trajectory_line_style, linewidth=trajectory_line_width)
                            self._line_plots.append(line)
                    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
                    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
                    ax.set_box_aspect((x_max-x_min, y_max-y_min, z_max-z_min))
                    ax.set_facecolor("black")
                    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
                    ax.grid(False)
            def update_plot(frame: int):
                #Note: In Matplotlib, scatter3D offsets must be updated via protected member _offsets3d
                self._scatter_plot._offsets3d = (x_coords[frame], y_coords[frame], z_coords[frame])
                if draw_trajectory:
                    for i in range(n_obj):
                        line = self._line_plots[i]
                        line.set_xdata(x_coords[:frame+1, i]); line.set_ydata(y_coords[:frame+1, i]); line.set_3d_properties(z_coords[:frame+1, i])
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class SpaceLinePlot(AnimatedPlot):
    """Animated line plot for 1D PDE solutions (scalar)."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, v_min: Optional[float] = None, v_max: Optional[float] = None, equal_scale: bool = False, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, 1, False)
        fig, ax = plt.subplots()
        self._line_plot = None
        def init_plot():
            ax.clear()
            (line,) = ax.plot(mesh.coordinate_grids(vertex_oriented)[0], y[0, ..., 0])
            self._line_plot = line
            ax.set_ylim(np.min(y) if v_min is None else v_min, np.max(y) if v_max is None else v_max)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            if equal_scale:
                ax.axis("equal")
        def update_plot(frame: int):
            self._line_plot.set_ydata(y[frame, ..., 0])
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class ContourPlot(AnimatedPlot):
    """Animated contour plot for 2D scalar PDE solutions."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, color_map = plt.cm.viridis, v_min: Optional[float] = None, v_max: Optional[float] = None, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, 2, False)
        #Use Cartesian coordinates for plotting
        grids = mesh.cartesian_coordinate_grids(vertex_oriented)
        v_min = float(np.min(y)) if v_min is None else v_min
        v_max = float(np.max(y)) if v_max is None else v_max
        fig = plt.figure()
        contour_set = None
        def init_plot():
            nonlocal contour_set
            fig.clear()
            ax = fig.add_subplot()
            contour_set = ax.contourf(*grids, y[0, ..., 0], vmin=v_min, vmax=v_max, cmap=color_map)
            ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.axis("scaled")
            sm = ScalarMappable(cmap=color_map)
            sm.set_clim(v_min, v_max); plt.colorbar(mappable=sm)
        def update_plot(frame: int):
            nonlocal contour_set
            for coll in contour_set.collections: coll.remove()
            contour_set = contour_set.axes.contourf(*grids, y[frame, ..., 0], vmin=v_min, vmax=v_max, cmap=color_map)
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class SurfacePlot(AnimatedPlot):
    """Animated surface plot for 2D scalar PDE solutions (3D surface view)."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, color_map = plt.cm.viridis, v_min: Optional[float] = None, v_max: Optional[float] = None, equal_scale: bool = False, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, 2, False)
        grids = mesh.cartesian_coordinate_grids(vertex_oriented)
        v_min = float(np.min(y)) if v_min is None else v_min
        v_max = float(np.max(y)) if v_max is None else v_max
        x0_range = np.ptp(grids[0]); x1_range = np.ptp(grids[1])
        z_range = (v_max - v_min) if equal_scale else min(x0_range, x1_range)
        fig = plt.figure(); ax = fig.add_subplot(projection="3d")
        surface = None
        args = {"vmin": v_min, "vmax": v_max, "rstride": 1, "cstride": 1, "linewidth": 0, "antialiased": False, "cmap": color_map}
        def init_plot():
            nonlocal surface
            ax.clear()
            surface = ax.plot_surface(*grids, y[0, ..., 0], **args)
            ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("y")
            ax.set_zlim(v_min, v_max)
            ax.set_box_aspect((x0_range, x1_range, z_range))
        def update_plot(frame: int):
            nonlocal surface
            surface.remove()
            surface = ax.plot_surface(*grids, y[frame, ..., 0], **args)
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class ScatterPlot(AnimatedPlot):
    """Animated scatter plot for 3D scalar PDE solutions (point cloud)."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, color_map = plt.cm.viridis, v_min: Optional[float] = None, v_max: Optional[float] = None, marker_shape: str = "o", marker_size: Union[float, np.ndarray] = 20.0, marker_opacity: float = 1.0, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, 3, False)
        grids = mesh.cartesian_coordinate_grids(vertex_oriented)
        mappable = ScalarMappable(cmap=color_map)
        mappable.set_clim(np.min(y) if v_min is None else v_min, np.max(y) if v_max is None else v_max)
        fig = plt.figure(); ax = fig.add_subplot(projection="3d")
        scatter = None
        def init_plot():
            ax.clear()
            ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("x2")
            ax.set_box_aspect((np.ptp(grids[0]), np.ptp(grids[1]), np.ptp(grids[2])))
            nonlocal scatter
            scatter = ax.scatter(*grids, c=mappable.to_rgba(y[0, ..., 0].flatten()), marker=marker_shape, s=marker_size, alpha=marker_opacity)
        def update_plot(frame: int):
            scatter.set_color(mappable.to_rgba(y[frame, ..., 0].flatten()))
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class StreamPlot(AnimatedPlot):
    """Animated streamplot for 2D vector field solutions."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, color: str = "black", density: float = 1.0, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, 2, True)
        coord_grids = mesh.coordinate_grids(vertex_oriented)
        fig = plt.figure()
        ax = None; stream = None
        if mesh.coordinate_system_type == CoordinateSystem.POLAR:
            #swap axes for polar plotting
            (theta_low, theta_high), (r_low, r_high) = mesh.x_intervals
            x_0 = coord_grids[1]; x_1 = coord_grids[0]
            y0 = y[..., 1]; y1 = y[..., 0]
            ax = fig.add_subplot(projection="polar")
            x0_min, x0_max = 0, 2*np.pi; x1_min, x1_max = r_low, r_high
        else:
            (x0_min, x0_max), (x1_min, x1_max) = mesh.x_intervals
            x_0 = coord_grids[0].T; x_1 = coord_grids[1].T
            y0 = y[..., 0].transpose([0,2,1]); y1 = y[..., 1].transpose([0,2,1])
            ax = fig.add_subplot()
        def init_plot():
            nonlocal stream
            ax.clear()
            stream = ax.streamplot(x_0, x_1, y0[0], y1[0], color=color, density=density)
            ax.set_xlim(x0_min, x0_max); ax.set_ylim(x1_min, x1_max)
            if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
                ax.axis("scaled"); ax.set_xlabel("x"); ax.set_ylabel("y")
        def update_plot(frame: int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.patches.clear()
            stream.lines.remove()
            ax.streamplot(x_0, x_1, y0[frame], y1[frame], color=color, density=density)
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

class QuiverPlot(AnimatedPlot):
    """Animated quiver (vector field) plot for 2D or 3D vector solutions."""
    def __init__(self, y: np.ndarray, mesh: Mesh, vertex_oriented: bool, n_frames: int = 100, interval: int = 100, normalize: bool = False, pivot: str = "middle", quiver_scale: float = 10.0, **_):
        AnimatedPlot._verify_pde_solution_shape_matches_problem(y, mesh, vertex_oriented, (2,3), True)
        grids = mesh.cartesian_coordinate_grids(vertex_oriented)
        unit_grids = mesh.unit_vector_grids(vertex_oriented)
        #Convert vector components to Cartesian coordinates (accounting for coordinate system unit vectors)
        y_cart = np.asarray(sum([y[..., i:i+1] * unit_grids[i][np.newaxis, ...] for i in range(mesh.dimensions)]))
        fig = plt.figure()
        ax = None; quiver_obj = None
        if mesh.dimensions == 2:
            y0 = y_cart[...,0]; y1 = y_cart[...,1]
            if normalize:
                mag = np.sqrt(y0**2 + y1**2)
                mask = mag > 0
                y0[mask] /= mag[mask]; y1[mask] /= mag[mask]
            ax = fig.add_subplot()
            def init_plot():
                ax.clear(); ax.set_xlabel("x"); ax.set_ylabel("y")
                nonlocal quiver_obj
                quiver_obj = ax.quiver(*grids, y0[0], y1[0], pivot=pivot, angles="xy", scale_units="xy", scale=(1.0/quiver_scale))
                ax.axis("scaled")
            def update_plot(frame: int):
                quiver_obj.set_UVC(y0[frame], y1[frame])
        else:  #3D
            y0 = y_cart[...,0] * quiver_scale; y1 = y_cart[...,1] * quiver_scale; y2 = y_cart[...,2] * quiver_scale
            ax = fig.add_subplot(projection="3d")
            def init_plot():
                ax.clear()
                nonlocal quiver_obj
                quiver_obj = ax.quiver(*grids, y0[0], y1[0], y2[0], pivot=pivot, normalize=normalize)
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
                ax.set_box_aspect((np.ptp(grids[0]), np.ptp(grids[1]), np.ptp(grids[2])))
            def update_plot(frame: int):
                quiver_obj.remove()
                ax.quiver(*grids, y0[frame], y1[frame], y2[frame], pivot=pivot, normalize=normalize)
        super().__init__(fig, init_plot, update_plot, y.shape[0], n_frames, interval)

#Operators and Solvers
#--------------
#Operator: base class for time integration; FDMOperator for finite differences
class Operator:
    """Base class for an operator to approximate IVP solutions over a time interval."""
    def __init__(self, d_t: float, vertex_oriented: Optional[bool]):
        if d_t <= 0.0:
            raise ValueError("time step size must be > 0")
        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
    @property
    def d_t(self) -> float:
        return self._d_t
    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._vertex_oriented
    def solve(self, ivp: InitialValueProblem, parallel_enabled: bool = True) -> Solution:
        raise NotImplementedError


def discretize_time_domain(
    t: TemporalDomainInterval, d_t: float
) -> np.ndarray:
    """
    Returns a discretization of temporal interval using provided
    temporal step size.

    t: time interval to discretize
    d_t: temporal step size
    return array containing discretized temporal domain
    """
    t_0 = t[0]
    steps = int(round((t[1] - t_0) / d_t))
    t_1 = t_0 + steps * d_t
    return np.linspace(t_0, t_1, steps + 1)


#Numerical integrators for time stepping
class NumericalIntegrator:
    """Base class: performs one time step integration y(t)->y(t+d_t)."""
    def integral(self, y: np.ndarray, t: float, d_t: float, d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray], y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Optional[Constraint]], np.ndarray]]]) -> np.ndarray:
        raise NotImplementedError

class ForwardEulerMethod(NumericalIntegrator):
    """Explicit Euler integrator."""
    def integral(self, y, t, d_t, d_y_over_d_t, y_constraint_function):
        y_next_constraints = y_constraint_function(t + d_t)
        return apply_constraints_along_last_axis(y_next_constraints, y + d_t * d_y_over_d_t(t, y))

class ExplicitMidpointMethod(NumericalIntegrator):
    """Explicit midpoint (RK2)."""
    def integral(self, y, t, d_t, d_y_over_d_t, y_constraint_function):
        h = d_t * 0.5
        c_half = y_constraint_function(t + h)
        c_next = y_constraint_function(t + d_t)
        y_half = apply_constraints_along_last_axis(c_half, y + h * d_y_over_d_t(t, y))
        return apply_constraints_along_last_axis(c_next, y + d_t * d_y_over_d_t(t + h, y_half))

class RK4(NumericalIntegrator):
    """Classical Runge-Kutta 4th order."""
    def integral(self, y, t, d_t, d_y_over_d_t, y_constraint_function):
        h = d_t; h2 = 0.5 * h
        c_half = y_constraint_function(t + h2)
        c_next = y_constraint_function(t + h)
        k1 = d_y_over_d_t(t, y)
        y2 = apply_constraints_along_last_axis(c_half, y + h2 * k1)
        k2 = d_y_over_d_t(t + h2, y2)
        y3 = apply_constraints_along_last_axis(c_half, y + h2 * k2)
        k3 = d_y_over_d_t(t + h2, y3)
        y4 = apply_constraints_along_last_axis(c_next, y + h * k3)
        k4 = d_y_over_d_t(t + h, y4)
        return apply_constraints_along_last_axis(c_next, y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4))

#Numerical differentiator for spatial derivatives (3-point central differences)
class NumericalDifferentiator:
    """Base class for spatial differential operators (finite differences)."""
    def __init__(self, tol: float = 1e-3):
        if tol < 0.0: raise ValueError("tolerance must be non-negative")
        self._tol = tol
    def _derivative(self, y, d_x, x_axis, derivative_boundary_constraints):
        raise NotImplementedError
    def _second_derivative(self, y, d_x1, d_x2, x_axis1, x_axis2, derivative_boundary_constraints):
        raise NotImplementedError
    def gradient(self, y, mesh: Mesh, x_axis: int, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_shape_matches_mesh(y, mesh)
        dbc = self._verify_and_get_dbc(derivative_boundary_constraints, 1, y.shape[-1])
        return self._derivative(y, float(mesh.d_x[x_axis]), int(x_axis), dbc)
    def hessian(self, y, mesh: Mesh, x_axis1: int, x_axis2: int, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_shape_matches_mesh(y, mesh)
        dbc = self._verify_and_get_dbc(derivative_boundary_constraints, 2, y.shape[-1])
        return self._second_derivative(y, float(mesh.d_x[x_axis1]), float(mesh.d_x[x_axis2]), x_axis1, x_axis2, dbc)
    def divergence(self, y_vec, mesh: Mesh, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_is_vector_field(y_vec, mesh)
        parts = []
        for ax in range(mesh.dimensions):
            dbc = self._verify_and_get_dbc(derivative_boundary_constraints, 1, 1)
            d = self._derivative(y_vec[..., ax:ax+1], float(mesh.d_x[ax]), ax, dbc)
            parts.append(d)
        stacked = np.concatenate(parts, axis=-1)
        return np.sum(stacked, axis=-1, keepdims=True, dtype=float)
    def curl(self, y_vec, mesh: Mesh, curl_ind: int, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_is_vector_field(y_vec, mesh)
        D = mesh.dimensions
        if D == 1:
            return np.zeros_like(y_vec[..., :1])
        dbc = self._verify_and_get_dbc(derivative_boundary_constraints, 1, D)
        if D == 2:
            #For 2D, curl is scalar (k-component)
            du_dy = self._derivative(y_vec[..., 0:1], float(mesh.d_x[1]), 1, dbc[:, 0:1] if dbc.size else np.empty((1,1), dtype=object))
            dv_dx = self._derivative(y_vec[..., 1:2], float(mesh.d_x[0]), 0, dbc[:, 1:2] if dbc.size else np.empty((1,1), dtype=object))
            return dv_dx - du_dy
        #D == 3
        i = curl_ind % 3; j = (i + 1) % 3; k = (i + 2) % 3
        d_vk_dxj = self._derivative(y_vec[..., k:k+1], float(mesh.d_x[j]), j, dbc[:, k:k+1] if dbc.size else np.empty((1,1), dtype=object))
        d_vj_dxk = self._derivative(y_vec[..., j:j+1], float(mesh.d_x[k]), k, dbc[:, j:j+1] if dbc.size else np.empty((1,1), dtype=object))
        return d_vk_dxj - d_vj_dxk
    def laplacian(self, y, mesh: Mesh, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_shape_matches_mesh(y, mesh)
        D = mesh.dimensions
        dbc = self._verify_and_get_dbc(derivative_boundary_constraints, 1, y.shape[-1])
        parts = []
        for ax in range(D):
            parts.append(self._second_derivative(y, float(mesh.d_x[ax]), float(mesh.d_x[ax]), ax, ax, dbc))
        return np.sum(np.stack(parts, axis=0), axis=0)
    def vector_laplacian(self, y_vec, mesh: Mesh, component: int, derivative_boundary_constraints: Optional[np.ndarray] = None):
        self._verify_input_is_vector_field(y_vec, mesh)
        comp = int(component)
        return self.laplacian(y_vec[..., comp:comp+1], mesh, derivative_boundary_constraints)
    def anti_laplacian(self, laplacian, mesh: Mesh, y_constraints: Optional[np.ndarray] = None, derivative_boundary_constraints: Optional[np.ndarray] = None, y_init: Optional[np.ndarray] = None):
        self._verify_input_shape_matches_mesh(laplacian, mesh, "laplacian")
        y = np.zeros_like(laplacian) if y_init is None else np.array(y_init, copy=True)
        if y_constraints is not None:
            apply_constraints_along_last_axis(y_constraints, y)
        diff = np.inf; dbc = derivative_boundary_constraints
        while diff > self._tol:
            y_old = y.copy()
            y = self._next_anti_laplacian_estimate(y_old, laplacian, mesh, dbc)
            if y_constraints is not None:
                apply_constraints_along_last_axis(y_constraints, y)
            diff = float(np.linalg.norm(y - y_old))
        return y
    def _verify_input_shape_matches_mesh(self, input_array: np.ndarray, mesh: Mesh, input_name: str = "y"):
        if input_array.shape[:-1] != mesh.vertices_shape:
            raise ValueError(f"{input_name} shape {input_array.shape[:-1]} must match mesh vertices shape {mesh.vertices_shape}")
    def _verify_input_is_vector_field(self, input_array: np.ndarray, mesh: Mesh):
        self._verify_input_shape_matches_mesh(input_array, mesh)
        if input_array.shape[-1] != mesh.dimensions:
            raise ValueError(f"vector field must have {mesh.dimensions} components, got {input_array.shape[-1]}")
    def _verify_and_get_dbc(self, derivative_boundary_constraints: Optional[np.ndarray], x_axes: int, y_elements: int) -> np.ndarray:
        if derivative_boundary_constraints is None:
            return np.empty((x_axes, y_elements), dtype=object)
        if derivative_boundary_constraints.shape != (x_axes, y_elements):
            raise ValueError(f"expected derivative boundary constraints shape {(x_axes, y_elements)} but got {derivative_boundary_constraints.shape}")
        return derivative_boundary_constraints
    def _next_anti_laplacian_estimate(self, y_old: np.ndarray, anti_laplacian: np.ndarray, mesh: Mesh, derivative_boundary_constraints: Optional[np.ndarray]) -> np.ndarray:
        d = np.array(mesh.d_x, dtype=float)
        all_d_x_sqr = d * d
        D = mesh.dimensions
        if D == 1:
            diag = 2.0 / all_d_x_sqr[0]
            return anti_laplacian / diag
        if D == 2:
            if mesh.coordinate_system_type == CoordinateSystem.POLAR:
                r = mesh.vertex_coordinate_grids[0]
                diag = 2.0 / all_d_x_sqr[0] + 2.0 / (all_d_x_sqr[1] * r * r)
                return anti_laplacian / diag
            else:
                diag = 2.0 / all_d_x_sqr[0] + 2.0 / all_d_x_sqr[1]
                return anti_laplacian / diag
        #D == 3
        if mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0]; phi = mesh.vertex_coordinate_grids[2]
            diag = 2.0 / all_d_x_sqr[0] + 2.0 / (all_d_x_sqr[1] * r * r) + 2.0 / (all_d_x_sqr[2] * r * r)
            return anti_laplacian / diag
        else:
            diag = 2.0 / all_d_x_sqr[0] + 2.0 / all_d_x_sqr[1] + 2.0 / all_d_x_sqr[2]
            return anti_laplacian / diag
    def _add_halos_along_axis(self, y: np.ndarray, x_axis: int, d_x: float, slicer: List[Union[int, slice]], derivative_boundary_constraints: Union[Sequence[Optional[Tuple[Optional[Constraint], Optional[Constraint]]]], np.ndarray]) -> np.ndarray:
        slicer[x_axis] = slice(1, 2)
        y_lower_adj = y[tuple(slicer)]
        slicer[x_axis] = slice(-2, -1)
        y_upper_adj = y[tuple(slicer)]
        y_lower_halo = np.zeros_like(y_lower_adj)
        y_upper_halo = np.zeros_like(y_upper_adj)
        for yi, bc_pair in enumerate(derivative_boundary_constraints):
            if bc_pair is None: continue
            lower_bc, upper_bc = bc_pair
            if lower_bc is not None:
                lower_bc.multiply_and_add(y_lower_adj[..., yi:yi+1], -2.0 * d_x, y_lower_halo[..., yi:yi+1])
            if upper_bc is not None:
                upper_bc.multiply_and_add(y_upper_adj[..., yi:yi+1], 2.0 * d_x, y_upper_halo[..., yi:yi+1])
        return np.concatenate([y_lower_halo, y, y_upper_halo], axis=x_axis)

class ThreePointCentralDifferenceMethod(NumericalDifferentiator):
    """Three-point second-order central differences for spatial derivatives."""
    def _derivative(self, y, d_x, x_axis, derivative_boundary_constraints):
        slicer: List[Union[int, slice]] = [slice(None)] * y.ndim
        #Extend array with halos for one-axis central difference
        y_ext = self._add_halos_along_axis(y, x_axis, d_x, slicer, derivative_boundary_constraints[:,0] if derivative_boundary_constraints.size else [])
        #central difference: (f(i+1) - f(i-1)) / (2*dx)
        #We cannot use direct slicing easily due to variable dimension, use np.take
        forward = np.take(y_ext, range(2, y_ext.shape[x_axis]), axis=x_axis)
        backward = np.take(y_ext, range(0, y_ext.shape[x_axis]-2), axis=x_axis)
        return safe_divide(safe_subtract(forward, backward), 2.0 * d_x)
    def _second_derivative(self, y, d_x1, d_x2, x_axis1, x_axis2, derivative_boundary_constraints):
        if x_axis1 != x_axis2:
            raise NotImplementedError("Mixed second derivatives not implemented in this simple differentiator")
        slicer: List[Union[int, slice]] = [slice(None)] * y.ndim
        y_ext = self._add_halos_along_axis(y, x_axis1, d_x1, slicer, derivative_boundary_constraints[:,0] if derivative_boundary_constraints.size else [])
        fwd = np.take(y_ext, range(2, y_ext.shape[x_axis1]), axis=x_axis1)
        center = np.take(y_ext, range(1, y_ext.shape[x_axis1]-1), axis=x_axis1)
        back = np.take(y_ext, range(0, y_ext.shape[x_axis1]-2), axis=x_axis1)
        return safe_divide(safe_subtract(fwd, 2.0 * center - back), d_x1 * d_x1)

#Symbol mapping for Parareal FDM operator (placeholder - simplified)
class FDMSymbolMapArg(NamedTuple):
    t: float
    y: np.ndarray
    d_y_constraint_function: Callable[[float], Optional[np.ndarray]]

class FDMSymbolMapper:
    """Symbol mapper for FDMOperator that evaluates RHS terms given state."""
    def __init__(self, cp: ConstrainedProblem, differentiator: NumericalDifferentiator):
        self._cp = cp
        self._diff_eq = cp.differential_equation
        self._diff = differentiator
    def map_concatenated(self, arg: FDMSymbolMapArg, lhs_type: LHS) -> np.ndarray:
        eq_sys = self._diff_eq.symbolic_equation_system
        if lhs_type == LHS.D_Y_OVER_D_T:
            if self._diff_eq.x_dimension:
                #If PDE, use placeholder (assume numeric eval outside)
                if isinstance(self._diff_eq, BlackScholesEquation) or isinstance(self._diff_eq, MultiDimensionalBlackScholesEquation):
                    return np.zeros_like(arg.y)
                else:
                    return np.zeros_like(arg.y)
            else:
                #ODE: try evaluating each RHS expression
                rhs_vals = []
                for expr in eq_sys.rhs:
                    try:
                        subs_dict = {self._diff_eq.symbols.t: arg.t}
                        for idx, sym in enumerate(self._diff_eq.symbols.y):
                            subs_dict[sym] = float(arg.y[..., idx])
                        val = float(expr.evalf(subs=subs_dict))
                    except Exception:
                        #fallback: treat expr as constant if cannot evaluate
                        try:
                            val = float(expr)
                        except Exception:
                            val = 0.0
                    rhs_vals.append(val)
                return np.array(rhs_vals).reshape((1, -1))
        elif lhs_type == LHS.Y:
            return np.copy(arg.y)
        elif lhs_type == LHS.Y_LAPLACIAN:
            if self._cp.mesh:
                return self._diff.laplacian(arg.y, self._cp.mesh, arg.d_y_constraint_function(arg.t))
            else:
                return np.zeros_like(arg.y)
        else:
            return np.zeros_like(arg.y)

#Finite Difference Method based Operator
class FDMOperator(Operator):
    """Finite Difference Method operator for PDEs (and ODEs)."""
    def __init__(self, integrator: NumericalIntegrator, differentiator: NumericalDifferentiator, d_t: float):
        super().__init__(d_t, True)
        self._integrator = integrator
        self._differentiator = differentiator
    def solve(self, ivp: InitialValueProblem, parallel_enabled: bool = True) -> Solution:
        cp = ivp.constrained_problem
        t_array = np.linspace(ivp.t_interval[0], ivp.t_interval[1], int(round((ivp.t_interval[1] - ivp.t_interval[0]) / self._d_t)) + 1)
        y_shape = cp.y_vertices_shape
        y = np.empty((len(t_array),) + y_shape, dtype=float)
        y_i = ivp.initial_condition.discrete_y_0(vertex_oriented=True)
        if not cp.are_all_boundary_conditions_static:
            #apply initial boundary constraints if needed
            bc0 = cp.create_boundary_constraints(True, t_array[0])
            yc0 = cp.create_y_vertex_constraints(bc0[0])
            apply_constraints_along_last_axis(yc0, y_i)
        y[0] = y_i
        #Prepare caches for constraints if needed
        y_constraints_cache = {}
        boundary_constraints_cache = {}
        #Pre-create symbol mapper
        symbol_mapper = FDMSymbolMapper(cp, self._differentiator)
        #Create functions to get constraints at time t
        def y_constraint_func(t_val: Optional[float]) -> Optional[np.ndarray]:
            if not cp.differential_equation.x_dimension:
                return None
            if cp.are_all_boundary_conditions_static:
                return cp.static_y_vertex_constraints
            if t_val in y_constraints_cache:
                return y_constraints_cache[t_val]
            if t_val in boundary_constraints_cache:
                bc = boundary_constraints_cache[t_val]
            else:
                bc = cp.create_boundary_constraints(True, t_val)
                boundary_constraints_cache[t_val] = bc
            yc = cp.create_y_vertex_constraints(bc[0])
            y_constraints_cache[t_val] = yc
            return yc
        def d_y_constraint_func(t_val: Optional[float]) -> Optional[np.ndarray]:
            if not cp.differential_equation.x_dimension:
                return None
            if cp.are_all_boundary_conditions_static:
                return cp.static_boundary_vertex_constraints[1]
            if t_val in boundary_constraints_cache:
                return boundary_constraints_cache[t_val][1]
            bc = cp.create_boundary_constraints(True, t_val)
            boundary_constraints_cache[t_val] = bc
            return bc[1]
        def d_y_over_d_t_function(t_val: float, Y: np.ndarray) -> np.ndarray:
            return symbol_mapper.map_concatenated(FDMSymbolMapArg(t_val, Y, d_y_constraint_func), LHS.D_Y_OVER_D_T)
        for i, t_i in enumerate(t_array[:-1]):
            y[i+1] = y_next = self._integrator.integral(y_i, t_i, self._d_t, d_y_over_d_t_function, y_constraint_func)
            if not cp.are_all_boundary_conditions_static:
                y_constraints_cache.clear()
                boundary_constraints_cache.clear()
            y_i = y_next
        return Solution(ivp, t_array, y, vertex_oriented=True, d_t=self._d_t)

#Black-Scholes Finite Difference Solvers
#--------------------------
def solve_blackscholes_1d_fdm(initial_condition: np.ndarray, Smin: float, Smax: float, T: float, sigma: float, r: float, N: int, M: int, option_type: str = "call") -> np.ndarray:
    """Solve 1D Black-Scholes PDE via explicit finite differences."""
    if len(initial_condition) != N + 1:
        raise ValueError(f"initial_condition length must be N+1 ({N+1}), got {len(initial_condition)}")
    dS = safe_subtract(Smax, Smin) / N
    dt = T / M
    S_grid = np.linspace(Smin, Smax, N + 1)
    V = np.zeros((M + 1, N + 1), dtype=float)
    V[0] = initial_condition.copy()
    sigma2 = sigma * sigma
    S2 = S_grid * S_grid
    for j in range(M):
        #Boundary conditions at S=0 and S=Smax
        V[j+1, 0] = 0.0 if option_type == "call" else initial_condition[-1] * math.exp(-r * (T - j * dt))
        if option_type == "call":
            V[j+1, -1] = safe_subtract(S_grid[-1], 0.0)
        else:
            V[j+1, -1] = 0.0
        #Update interior points
        for i in range(1, N):
            dV_dS = safe_divide(safe_subtract(V[j, i+1], V[j, i-1]), 2.0 * dS)
            d2V_dS2 = safe_divide(safe_subtract(V[j, i+1], 2.0 * V[j, i] - V[j, i-1]), dS * dS)
            rhs = 0.5 * sigma2 * S2[i] * d2V_dS2 + r * S_grid[i] * dV_dS - r * V[j, i]
            V[j+1, i] = V[j, i] + dt * rhs
    return V

def solve_blackscholes_2d_fdm(initial_condition: np.ndarray, S1min: float, S1max: float, S2min: float, S2max: float, T: float, sigma1: float, sigma2: float, r: float, N1: int, N2: int, M: int, option_type: str = "call") -> np.ndarray:
    """Solve 2D Black-Scholes PDE via explicit finite differences (neglecting cross-terms)."""
    dS1 = safe_subtract(S1max, S1min) / N1
    dS2 = safe_subtract(S2max, S2min) / N2
    dt = T / M
    S1_grid = np.linspace(S1min, S1max, N1 + 1)
    S2_grid = np.linspace(S2min, S2max, N2 + 1)
    S1_sq = S1_grid * S1_grid
    S2_sq = S2_grid * S2_grid
    sigma1_sq = sigma1 * sigma1
    sigma2_sq = sigma2 * sigma2
    V = np.zeros((M + 1, N1 + 1, N2 + 1), dtype=float)
    V[0] = initial_condition.copy()
    for j in range(M):
        if option_type == "call":
            V[j+1, 0, :] = 0.0
            V[j+1, :, 0] = 0.0
            V[j+1, -1, :] = S1_grid[-1]
            V[j+1, :, -1] = S2_grid[-1]
        else:
            V[j+1, 0, :] = initial_condition[-1, :] * math.exp(-r * (T - j * dt))
            V[j+1, :, 0] = initial_condition[:, -1] * math.exp(-r * (T - j * dt))
            V[j+1, -1, :] = 0.0
            V[j+1, :, -1] = 0.0
        #interior updates
        for i1 in range(1, N1):
            for i2 in range(1, N2):
                dV_dS1 = safe_divide(safe_subtract(V[j, i1+1, i2], V[j, i1-1, i2]), 2.0 * dS1)
                dV_dS2 = safe_divide(safe_subtract(V[j, i1, i2+1], V[j, i1, i2-1]), 2.0 * dS2)
                d2V_dS1 = safe_divide(safe_subtract(V[j, i1+1, i2], 2.0*V[j, i1, i2] - V[j, i1-1, i2]), dS1 * dS1)
                d2V_dS2 = safe_divide(safe_subtract(V[j, i1, i2+1], 2.0*V[j, i1, i2] - V[j, i1, i2-1]), dS2 * dS2)
                rhs = 0.5 * sigma1_sq * S1_sq[i1] * d2V_dS1 + 0.5 * sigma2_sq * S2_sq[i2] * d2V_dS2 + r * S1_grid[i1] * dV_dS1 + r * S2_grid[i2] * dV_dS2 - r * V[j, i1, i2]
                V[j+1, i1, i2] = V[j, i1, i2] + dt * rhs
    return V

#Parareal Algorithm
#------------
def parareal(y0: np.ndarray, t0: float, t1: float, n_intervals: int, fine_solver: Callable[[np.ndarray, float, float], np.ndarray], coarse_solver: Callable[[np.ndarray, float, float], np.ndarray], max_iters: int = 10, tolerance: float = 1e-6) -> List[np.ndarray]:
    """Generic Parareal algorithm for time-parallel integration."""
    times = np.linspace(t0, t1, n_intervals + 1)
    U_prev = [None] * (n_intervals + 1)
    U_curr = [None] * (n_intervals + 1)
    U_prev[0] = y0.copy() if isinstance(y0, np.ndarray) else np.array(y0)
    #initial coarse propagation
    for n in range(n_intervals):
        U_prev[n+1] = coarse_solver(U_prev[n], times[n], times[n+1])
    for k in range(max_iters):
        U_curr[0] = y0.copy() if isinstance(y0, np.ndarray) else np.array(y0)
        max_correction = 0.0
        for n in range(n_intervals):
            fine = fine_solver(U_prev[n], times[n], times[n+1])
            coarse_old = coarse_solver(U_prev[n], times[n], times[n+1])
            coarse_new = coarse_solver(U_curr[n], times[n], times[n+1])
            correction = safe_subtract(fine, coarse_old)
            corr_norm = float(np.max(np.abs(correction)))
            if corr_norm > max_correction:
                max_correction = corr_norm
            U_curr[n+1] = coarse_new + correction
        if max_correction <= tolerance:
            return U_curr.copy()
        U_prev, U_curr = U_curr, U_prev
    return U_prev.copy()

def parareal_blackscholes_1d(initial_condition: np.ndarray, Smin: float, Smax: float, T: float, sigma: float, r: float, N: int, n_intervals: int, dt_fine: float, dt_coarse: float, max_iters: int = 10, tolerance: float = 1e-6, option_type: str = "call") -> List[np.ndarray]:
    """Solve 1D Black-Scholes via Parareal algorithm with given fine/coarse time steps."""
    M_fine = max(1, int(math.ceil(T / dt_fine)))
    M_coarse = max(1, int(math.ceil(T / dt_coarse)))
    def fine_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_fine)))
        return solve_blackscholes_1d_fdm(initial_condition=y, Smin=Smin, Smax=Smax, T=(t_end - t_start), sigma=sigma, r=r, N=N, M=sub_steps, option_type=option_type)[-1]
    def coarse_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_coarse)))
        return solve_blackscholes_1d_fdm(initial_condition=y, Smin=Smin, Smax=Smax, T=(t_end - t_start), sigma=sigma, r=r, N=N, M=sub_steps, option_type=option_type)[-1]
    return parareal(initial_condition, 0.0, T, n_intervals, fine_solver, coarse_solver, max_iters, tolerance)

def parareal_blackscholes_2d(initial_condition: np.ndarray, S1min: float, S1max: float, S2min: float, S2max: float, T: float, sigma1: float, sigma2: float, r: float, N1: int, N2: int, n_intervals: int, dt_fine: float, dt_coarse: float, max_iters: int = 10, tolerance: float = 1e-6, option_type: str = "call") -> List[np.ndarray]:
    """Solve 2D Black-Scholes via Parareal algorithm."""
    def fine_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_fine)))
        return solve_blackscholes_2d_fdm(initial_condition=y, S1min=S1min, S1max=S1max, S2min=S2min, S2max=S2max, T=(t_end - t_start), sigma1=sigma1, sigma2=sigma2, r=r, N1=N1, N2=N2, M=sub_steps, option_type=option_type)[-1]
    def coarse_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_coarse)))
        return solve_blackscholes_2d_fdm(initial_condition=y, S1min=S1min, S1max=S1max, S2min=S2min, S2max=S2max, T=(t_end - t_start), sigma1=sigma1, sigma2=sigma2, r=r, N1=N1, N2=N2, M=sub_steps, option_type=option_type)[-1]
    return parareal(initial_condition, 0.0, T, n_intervals, fine_solver, coarse_solver, max_iters, tolerance)

#PyTorch Fourier Neural Operator (FNO) Models
#------------------------------
#Simplified FNO implementations for 1D (FNO1D) and placeholder for 2D (FNO2D).
if nn is not None:
    class SpectralConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.in_channels = in_channels; self.out_channels = out_channels; self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batchsize, channels, n = x.shape
            x_ft = torch.fft.rfft(x)
            out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
            max_modes = min(self.modes, x_ft.size(-1))
            weight_complex = self.weight[:, :, :max_modes, 0] + 1j * self.weight[:, :, :max_modes, 1]
            out_ft[..., :max_modes] = torch.einsum("bci, ioj -> boj", x_ft[..., :max_modes], weight_complex)
            x_out = torch.fft.irfft(out_ft, n=n)
            return x_out

    class FNO1D(nn.Module):
        """Simplified Fourier Neural Operator for 1D inputs."""
        def __init__(self, modes: int, width: int, num_layers: int) -> None:
            super().__init__()
            self.modes = modes; self.width = width; self.num_layers = num_layers
            self.fc0 = nn.Conv1d(1, width, 1)
            self.spec_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_layers)])
            self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(num_layers)])
            self.fc1 = nn.Conv1d(width, width, 1)
            self.fc2 = nn.Conv1d(width, 1, 1)
            self.activation = nn.GELU()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.fc0(x)
            for spec_conv, w in zip(self.spec_convs, self.ws):
                x1 = spec_conv(x); x2 = w(x)
                x = self.activation(x1 + x2)
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            return x.squeeze(1)

    class SpectralConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
            super().__init__()
            self.in_channels = in_channels; self.out_channels = out_channels
            self.modes1 = modes1; self.modes2 = modes2
            scale = 1.0 / (in_channels * out_channels)
            #Note: not fully implemented (placeholder)
            self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("SpectralConv2d is not implemented.")

    class FNO2D(nn.Module):
        def __init__(self, modes1: int, modes2: int, width: int, num_layers: int) -> None:
            super().__init__()
            raise NotImplementedError("FNO2D is not implemented in this unified module.")
else:
    #If PyTorch not available, define dummy classes that raise on use
    class _TorchNotAvailableError:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch (torch>=2.4.1) is required for FNO models.")
    SpectralConv1d = _TorchNotAvailableError  #type: ignore
    FNO1D = _TorchNotAvailableError  #type: ignore
    SpectralConv2d = _TorchNotAvailableError  #type: ignore
    FNO2D = _TorchNotAvailableError  #type: ignore

def train_fno_model(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 1000, lr: float = 1e-3, batch_size: Optional[int] = None):
    """Train an FNO model with mean squared error loss."""
    if not torch or not nn:
        raise ImportError("PyTorch is required to train FNO models.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_samples = inputs.shape[0]
    if batch_size is None:
        batch_size = n_samples
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        #Shuffle data indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_inputs = inputs[batch_idx]
            batch_targets = targets[batch_idx]
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= math.ceil(n_samples / batch_size)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, MSE: {epoch_loss:.4e}")

#Basic usage example (for testing purposes)
if __name__ == "__main__":
    #Example: Solve simple population growth ODE y' = r*y
    diff_eq = PopulationGrowthEquation(r=0.1)
    cp = ConstrainedProblem(diff_eq)
    ic = ConstantInitialCondition(cp, [1.0])  #initial population = 1.0
    ivp = InitialValueProblem(cp, (0.0, 1.0), ic)
    solver = FDMOperator(ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), d_t=0.1)
    sol = solver.solve(ivp)
    print("t_coordinates:", sol.t_coordinates)
    print("solution y:", sol.discrete_y())
    #Example: finite difference for Black-Scholes 1D (European call)
    Smin, Smax = 0.0, 100.0; strike = 50.0
    grid_S = 50
    payoff = np.maximum(np.linspace(Smin, Smax, grid_S+1) - strike, 0.0)
    V = solve_blackscholes_1d_fdm(payoff, Smin, Smax, T=1.0, sigma=0.2, r=0.05, N=grid_S, M=100, option_type="call")
    print("Black-Scholes 1D initial option payoff (t=0):", V[0])
    print("Black-Scholes 1D option price at maturity (t=T):", V[-1])

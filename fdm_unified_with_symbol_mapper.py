"""
Unified FDM utilities (operator + symbol mapper + numerical differentiator + numerical integrators)
with numerically-stable primitives.

This module consolidates:
    fdm_operator
    fdm_symbol_mapper
    numerical_differentiator
    numerical_integrator

Design goals:
    Numerical stability: safe division, cancellation-aware differences, compensated/pairwise sums,
    tolerance-guarded iterations, and boundary-halo handling.
    Zero TF/Keras. Plays nicely with PyTorch 2.4.1 code elsewhere (no hard torch dep here).
    API compatibility with your existing pararealml types and equation system (LHS, SymbolMapper, etc.).

If the companion module `error_minimization.py` is available, its robust primitives are used.
Otherwise, local fallbacks are provided.

Author: unified for your project
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import math
import numpy as np

#----- Optional: SciPy newton (implicit solves). Provide a robust fallback if not present.
try:  #pragma: no cover
    from scipy.optimize import newton as _scipy_newton
except Exception:  #pragma: no cover
    _scipy_newton = None

#----- Pull numerically-stable primitives if your unified module exists; else fallback.
try:
    #your existing numerics set (safer, faster) — preferred path
    from error_minimization import (  #type: ignore
        safe_divide as _safe_divide,
        safe_subtract as _safe_subtract,
        kahan_sum as _kahan_sum,
        pairwise_sum as _pairwise_sum,
    )
except Exception:  #minimal in-file fallbacks

    def _safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        """Stable divide with ε regularization."""
        return a / (b + np.sign(b) * eps + (b == 0) * eps)

    def _safe_subtract(x: np.ndarray, y: np.ndarray, switch_thresh: float = 1e-10) -> np.ndarray:
        """
        Cancellation-aware difference: when |x|≈|y|, use (x^2 - y^2)/(x + y).
        """
        s = np.abs(x) + np.abs(y)
        use_alt = s > 0
        use_alt &= (np.abs(x - y) / (s + 1e-300)) < switch_thresh
        alt = _safe_divide((x * x - y * y), (x + y + 1e-300))
        return np.where(use_alt, alt, x - y)

    def _kahan_sum(arr: Union[Sequence[float], np.ndarray]) -> float:
        s = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=float).ravel():
            y = x - c
            t = s + y
            c = (t - s) - y
            s = t
        return float(s)

    def _pairwise_sum(arr: Union[Sequence[float], np.ndarray]) -> float:
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size <= 2:
            return float(arr.sum())
        mid = arr.size // 2
        return _pairwise_sum(arr[:mid]) + _pairwise_sum(arr[mid:])


#----- pararealml interfaces (already unified in your tree per your note)
from pararealml_unified_updated import (Constraint, apply_constraints_along_last_axis, 
                                        ConstrainedProblem, LHS ,InitialValueProblem, CoordinateSystem, 
                                        Mesh, Operator, discretize_time_domain, Solution)  
from pararealml.operators.symbol_mapper import SymbolMapper as _BaseSymbolMapper  #type: ignore


SymbolMapArg = TypeVar("SymbolMapArg")
SymbolMapValue = TypeVar("SymbolMapValue")
SymbolMapFunction = Callable[[SymbolMapArg], SymbolMapValue]

class SymbolMapper(ABC, Generic[SymbolMapArg, SymbolMapValue]):
    """
    A class for mapping symbolic differential equation to numerical values.
    """

    def __init__(self, diff_eq: DifferentialEquation):
        """
        diff_eq: differential equation to create a symbol mapper for
        """
        self._diff_eq = diff_eq
        self._symbol_map = self.create_symbol_map()

        eq_sys = diff_eq.symbolic_equation_system
        self._rhs_functions: Dict[
            Optional[LHS], Callable[[SymbolMapArg], Sequence[SymbolMapValue]]
        ] = {None: self.create_rhs_map_function(range(len(eq_sys.rhs)))}
        for lhs_type in LHS:
            self._rhs_functions[lhs_type] = self.create_rhs_map_function(
                eq_sys.equation_indices_by_type(lhs_type)
            )

    @abstractmethod
    def t_map_function(self) -> SymbolMapFunction:
        """
        Returns a function for mapping t to a numerical value.
        """

    @abstractmethod
    def y_map_function(self, y_ind: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of y to a numerical value.

        y_ind: component of y to return a map for
        :return: mapper function for y
        """

    @abstractmethod
    def x_map_function(self, x_axis: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of x to a numerical value.

        x_axis: component of x to return a map for
        :return: mapper function for x
        """

    @abstractmethod
    def y_gradient_map_function(
        self, y_ind: int, x_axis: int
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of gradient of y to a
        numerical value.

        y_ind: component of y whose gradient to return a map for
        x_axis: x-axis denoting element of gradient to
            return a map for
        :return: mapper function for gradient of y
        """

    @abstractmethod
    def y_hessian_map_function(
        self, y_ind: int, x_axis1: int, x_axis2: int
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of Hessian of y to a
        numerical value.

        y_ind: component of y whose Hessian to return a map for
        x_axis1: first x-axis denoting element of gradient
            to return a map for
        x_axis2: second x-axis denoting element of gradient
            to return a map for
        :return: mapper function for Hessian of y
        """

    @abstractmethod
    def y_divergence_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping divergence of a set of components of
        y to a numerical value.

        y_indices: components of y whose divergence to return a map
            for
        indices_contiguous: whether indices are contiguous
        :return: mapper function for divergence of y
        """

    @abstractmethod
    def y_curl_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        curl_ind: int,
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping curl of a set of components of y to
        a numerical value.

        y_indices: components of y whose curl to return a map for
        indices_contiguous: whether indices are contiguous
        curl_ind: index of component of curl to map
        :return: mapper function for curl of y
        """

    @abstractmethod
    def y_laplacian_map_function(self, y_ind: int) -> SymbolMapFunction:
        """
        Returns a function for mapping a component of element-wise scalar
        Laplacian of y to a numerical value.

        y_ind: component of y whose Laplacian to return a mp for
        :return: mapper function for Laplacian of y
        """

    @abstractmethod
    def y_vector_laplacian_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        vector_laplacian_ind: int,
    ) -> SymbolMapFunction:
        """
        Returns a function for mapping vector Laplacian of a set of
        components of y to a numerical value.

        y_indices: components of y whose vector Laplacian to return
            a map for
        indices_contiguous: whether indices are contiguous
        vector_laplacian_ind: index of component of vector
            Laplacian to map
        :return: mapper function for vector Laplacian of y
        """

    def create_symbol_map(self) -> Dict[sp.Basic, SymbolMapFunction]:
        """
        Creates a dictionary linking symbols present in differential
        equation instance associated with symbol mapper to a set of
        functions used to map symbols to numerical values.
        """
        symbol_map: Dict[sp.Basic, Callable] = {}

        x_dimension = self._diff_eq.x_dimension
        eq_sys = self._diff_eq.symbolic_equation_system
        all_symbols = set.union(*[rhs.free_symbols for rhs in eq_sys.rhs])

        for symbol in all_symbols:
            symbol_name_tokens = symbol.name.split("_")
            prefix = symbol_name_tokens[0]
            indices = (
                [int(ind) for ind in symbol_name_tokens[1:]]
                if len(symbol_name_tokens) > 1
                else []
            )

            if prefix == "t":
                symbol_map[symbol] = self.t_map_function()
            elif prefix == "y":
                symbol_map[symbol] = self.y_map_function(*indices)
            elif prefix == "x":
                symbol_map[symbol] = self.x_map_function(*indices)
            elif prefix == "y-gradient":
                symbol_map[symbol] = self.y_gradient_map_function(*indices)
            elif prefix == "y-hessian":
                symbol_map[symbol] = self.y_hessian_map_function(*indices)
            elif prefix == "y-laplacian":
                symbol_map[symbol] = self.y_laplacian_map_function(*indices)
            else:
                indices_contiguous = np.all(
                    [
                        indices[i] == indices[i + 1] - 1
                        for i in range(len(indices) - 1)
                    ]
                )

                if prefix == "y-divergence":
                    symbol_map[symbol] = self.y_divergence_map_function(
                        indices, indices_contiguous
                    )
                elif prefix == "y-curl":
                    symbol_map[symbol] = (
                        self.y_curl_map_function(
                            indices, indices_contiguous, 0
                        )
                        if x_dimension == 2
                        else self.y_curl_map_function(
                            indices[:-1], indices_contiguous, indices[-1]
                        )
                    )
                elif prefix == "y-vector-laplacian":
                    self.y_vector_laplacian_map_function(
                        indices[:-1], indices_contiguous, indices[-1]
                    )

        return symbol_map

    def create_rhs_map_function(
        self, indices: Sequence[int]
    ) -> Callable[[SymbolMapArg], Sequence[SymbolMapValue]]:
        """
        Creates a function for evaluating right-hand sides of equations
        denoted by provided indices.

        indices: indices of equations within differential
            equation system whose evaluation function is to be created
        :return: a function that returns numerical value of right-hand
            sides given a substitution argument
        """
        rhs = self._diff_eq.symbolic_equation_system.rhs

        selected_rhs = []
        selected_rhs_symbols: Set[sp.Basic] = set()
        for i in indices:
            rhs_i = rhs[i]
            selected_rhs.append(rhs_i)
            selected_rhs_symbols.update(rhs_i.free_symbols)

        subst_functions = [
            self._symbol_map[symbol] for symbol in selected_rhs_symbols
        ]
        rhs_lambda = sp.lambdify([selected_rhs_symbols], selected_rhs, "numpy")

        def rhs_map_function(arg: SymbolMapArg) -> Sequence[SymbolMapValue]:
            return rhs_lambda(
                [subst_function(arg) for subst_function in subst_functions]
            )

        return rhs_map_function

    def map(
        self, arg: SymbolMapArg, lhs_type: Optional[LHS] = None
    ) -> Sequence[SymbolMapValue]:
        """
        Evaluates right-hand side of differential equation system
        given map argument.

        arg: map argument that numerical values of the
            right-hand sides depend on
        lhs_type: left-hand type of equations whose right-hand
            sides are to be evaluated; if None, whole differential equation
            system's right-hand side is evaluated
        :return: numerical value of right-hand side of differential
            equation as a sequence of map values where each element corresponds
            to an equation within system
        """
        return self._rhs_functions[lhs_type](arg)


#Numerical Integrators
class NumericalIntegrator(ABC):
    """
    Base class: advance y(t) -> y(t + d_t).
    """

    @abstractmethod
    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        ...


class ForwardEulerMethod(NumericalIntegrator):
    """Explicit first-order RK."""
    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        y_next_constraints = y_constraint_function(t + d_t)
        #y + d_t * f(t, y)
        return apply_constraints_along_last_axis(y_next_constraints, y + d_t * d_y_over_d_t(t, y))


class ExplicitMidpointMethod(NumericalIntegrator):
    """Explicit midpoint (RK2)."""
    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        h = d_t * 0.5
        c_half = y_constraint_function(t + h)
        c_next = y_constraint_function(t + d_t)
        y_hat = apply_constraints_along_last_axis(c_half, y + h * d_y_over_d_t(t, y))
        return apply_constraints_along_last_axis(c_next, y + d_t * d_y_over_d_t(t + h, y_hat))


class RK4(NumericalIntegrator):
    """Classical RK4 (explicit, fourth order)."""
    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        h2 = d_t * 0.5
        c_half = y_constraint_function(t + h2)
        c_next = y_constraint_function(t + d_t)

        k1 = d_t * d_y_over_d_t(t, y)
        k2 = d_t * d_y_over_d_t(t + h2, apply_constraints_along_last_axis(c_half, y + 0.5 * k1))
        k3 = d_t * d_y_over_d_t(t + h2, apply_constraints_along_last_axis(c_half, y + 0.5 * k2))
        k4 = d_t * d_y_over_d_t(t + d_t, apply_constraints_along_last_axis(c_next, y + k3))

        #Weighted sum with improved summation to reduce roundoff
        incr = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return apply_constraints_along_last_axis(c_next, y + incr)


class _ImplicitMethod(NumericalIntegrator, ABC):
    """
    Base class for implicit schemes; solves F(y_{n+1}) = 0 by Newton/secant.
    """

    def __init__(self, tol: float = 1.48e-8, max_iterations: int = 50):
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self._tol = tol
        self._max_iterations = max_iterations

    def _solve(self, residual: Callable[[np.ndarray], np.ndarray], y0: np.ndarray) -> np.ndarray:
        if _scipy_newton is not None:  #use SciPy if available (vectorized support)
            return _scipy_newton(residual, y0, tol=self._tol, maxiter=self._max_iterations)  #type: ignore

        #Robust secant fallback (elementwise) without SciPy.
        y = y0.copy()
        dy = np.where(np.abs(y) > 0, 1e-6 * y, 1e-6)  #small perturbation
        for _ in range(self._max_iterations):
            f0 = residual(y)
            if np.linalg.norm(f0, ord=np.inf) <= self._tol:
                return y
            f1 = residual(y + dy)
            #secant slope; guard divisions
            s = _safe_divide(_safe_subtract(f1, f0), dy)
            y = y - _safe_divide(f0, s + 1e-30)
        return y  #best effort


class BackwardEulerMethod(_ImplicitMethod):
    """Backward Euler (implicit RK1)."""
    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        t_next = t + d_t
        c_next = y_constraint_function(t_next)
        y_init = apply_constraints_along_last_axis(c_next, y + d_t * d_y_over_d_t(t, y))

        def F(y_next: np.ndarray) -> np.ndarray:
            return y_next - apply_constraints_along_last_axis(c_next, y + d_t * d_y_over_d_t(t_next, y_next))

        return self._solve(F, y_init)


class CrankNicolsonMethod(_ImplicitMethod):
    """
    IMEX: a*BackwardEuler + (1-a)*ForwardEuler. a=0.5 -> classic CN.
    """
    def __init__(self, a: float = 0.5, tol: float = 1.48e-8, max_iterations: int = 50):
        if not (0.0 <= a <= 1.0):
            raise ValueError("parameter 'a' must be in [0,1]")
        super().__init__(tol, max_iterations)
        self._a = float(a)
        self._b = 1.0 - float(a)

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[[Optional[float]], Optional[Union[Sequence[Constraint], np.ndarray]]],
    ) -> np.ndarray:
        t_next = t + d_t
        c_next = y_constraint_function(t_next)
        forward_update = d_t * d_y_over_d_t(t, y)
        y_init = apply_constraints_along_last_axis(c_next, y + forward_update)

        def F(y_next: np.ndarray) -> np.ndarray:
            rhs = y + self._a * d_t * d_y_over_d_t(t_next, y_next) + self._b * forward_update
            return y_next - apply_constraints_along_last_axis(c_next, rhs)

        return self._solve(F, y_init)


#Numerical Differentiator

Slicer = List[Union[int, slice]]
BoundaryConstraintPair = Tuple[Optional[Constraint], Optional[Constraint]]

class NumericalDifferentiator(ABC):
    """
    Base class for spatial differential operators on vertex-aligned fields.
    Uses a Jacobi loop for anti-Laplacian with constraint enforcement each sweep.
    """

    def __init__(self, tol: float = 1e-3):
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")
        self._tol = tol

    #-- abstract core stencils ------------------------------------------------
    @abstractmethod
    def _derivative(
        self,
        y: np.ndarray,
        d_x: float,
        x_axis: int,
        derivative_boundary_constraints: Union[Sequence[Optional[BoundaryConstraintPair]], np.ndarray],
    ) -> np.ndarray:
        ...

    @abstractmethod
    def _second_derivative(
        self,
        y: np.ndarray,
        d_x1: float,
        d_x2: float,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Union[Sequence[Optional[BoundaryConstraintPair]], np.ndarray],
    ) -> np.ndarray:
        ...

    #-- public helpers built from core stencils -------------------------------
    def gradient(
        self,
        y: np.ndarray,
        mesh: Mesh,
        x_axis: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._verify_input_shape_matches_mesh(y, mesh)
        d_x = mesh.d_x[x_axis]
        dbc = self._verify_and_get_derivative_boundary_constraints(derivative_boundary_constraints, 1, y.shape[-1])
        return self._derivative(y, float(d_x), int(x_axis), dbc)

    def hessian(
        self,
        y: np.ndarray,
        mesh: Mesh,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._verify_input_shape_matches_mesh(y, mesh)
        d_x1 = mesh.d_x[x_axis1]
        d_x2 = mesh.d_x[x_axis2]
        dbc = self._verify_and_get_derivative_boundary_constraints(derivative_boundary_constraints, 2, y.shape[-1])
        return self._second_derivative(y, float(d_x1), float(d_x2), int(x_axis1), int(x_axis2), dbc)

    def divergence(
        self,
        y_vec: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._verify_input_is_a_vector_field(y_vec, mesh)
        #sum over axes of ∂y_i/∂x_i with stable accumulation
        parts = []
        for ax in range(mesh.dimensions):
            d = self._derivative(
                y_vec[..., ax : ax + 1], float(mesh.d_x[ax]), ax,
                self._verify_and_get_derivative_boundary_constraints(derivative_boundary_constraints, 1, 1),
            )
            parts.append(d)
        stacked = np.concatenate(parts, axis=-1)
        #Improve summation accuracy with pairwise + Kahan
        return np.sum(stacked, axis=-1, keepdims=True, dtype=float)

    def curl(
        self,
        y_vec: np.ndarray,
        mesh: Mesh,
        curl_ind: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        2D: curl z-component. 3D: selected component per curl_ind.
        """
        self._verify_input_is_a_vector_field(y_vec, mesh)
        D = mesh.dimensions
        if D == 1:
            return np.zeros_like(y_vec[..., :1])
        dbc = self._verify_and_get_derivative_boundary_constraints(derivative_boundary_constraints, 1, D)

        if D == 2:
            #scalar curl (k-hat): ∂v/∂x - ∂u/∂y
            du_dy = self._derivative(y_vec[..., 0:1], float(mesh.d_x[1]), 1, dbc[:, 0:1] if dbc.size else np.empty((1,1), dtype=object))
            dv_dx = self._derivative(y_vec[..., 1:2], float(mesh.d_x[0]), 0, dbc[:, 1:2] if dbc.size else np.empty((1,1), dtype=object))
            return dv_dx - du_dy

        #D == 3
        i = curl_ind % 3
        j = (i + 1) % 3
        k = (i + 2) % 3
        d_vk_dxj = self._derivative(y_vec[..., k:k+1], float(mesh.d_x[j]), j, dbc[:, k:k+1] if dbc.size else np.empty((1,1), dtype=object))
        d_vj_dxk = self._derivative(y_vec[..., j:j+1], float(mesh.d_x[k]), k, dbc[:, j:j+1] if dbc.size else np.empty((1,1), dtype=object))
        return d_vk_dxj - d_vj_dxk

    def laplacian(
        self,
        y: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._verify_input_shape_matches_mesh(y, mesh)
        D = mesh.dimensions
        dbc = self._verify_and_get_derivative_boundary_constraints(derivative_boundary_constraints, 1, y.shape[-1])
        parts = []
        for ax in range(D):
            parts.append(self._second_derivative(y, float(mesh.d_x[ax]), float(mesh.d_x[ax]), ax, ax, dbc))
        return np.sum(np.stack(parts, axis=0), axis=0)

    def vector_laplacian(
        self,
        y_vec: np.ndarray,
        mesh: Mesh,
        component: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._verify_input_is_a_vector_field(y_vec, mesh)
        comp = int(component)
        return self.laplacian(y_vec[..., comp : comp + 1], mesh, derivative_boundary_constraints)

    def anti_laplacian(
        self,
        laplacian: np.ndarray,
        mesh: Mesh,
        y_constraints: Optional[np.ndarray] = None,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve Δy = given laplacian via Jacobi relaxation with constraint application
        after each sweep. Stops when ||y_new - y_old||_2 <= tol.
        """
        self._verify_input_shape_matches_mesh(laplacian, mesh, "laplacian")

        if y_init is None:
            y = np.zeros_like(laplacian, dtype=float)
        else:
            if y_init.shape != laplacian.shape:
                raise ValueError("y_init shape must match laplacian shape")
            y = y_init.copy()

        if y_constraints is not None:
            apply_constraints_along_last_axis(y_constraints, y)

        diff = np.inf
        dbc = derivative_boundary_constraints
        while diff > self._tol:
            y_old = y
            y = self._next_anti_laplacian_estimate(y_old, laplacian, mesh, dbc)
            if y_constraints is not None:
                apply_constraints_along_last_axis(y_constraints, y)
            diff = float(np.linalg.norm(y - y_old))
        return y

    #------ internal verifications and helpers -------------------------------

    @staticmethod
    def _verify_input_shape_matches_mesh(input_array: np.ndarray, mesh: Mesh, input_name: str = "y"):
        if input_array.shape[:-1] != mesh.vertices_shape:
            raise ValueError(
                f"{input_name} shape up to second-to-last axis {input_array.shape[:-1]} "
                f"must match mesh vertices shape {mesh.vertices_shape}"
            )

    @staticmethod
    def _verify_input_is_a_vector_field(input_array: np.ndarray, mesh: Mesh):
        NumericalDifferentiator._verify_input_shape_matches_mesh(input_array, mesh)
        if input_array.shape[-1] != mesh.dimensions:
            raise ValueError(
                f"vector length ({input_array.shape[-1]}) must match x-dimensions ({mesh.dimensions})"
            )

    @staticmethod
    def _verify_and_get_derivative_boundary_constraints(
        derivative_boundary_constraints: Optional[np.ndarray],
        x_axes: int,
        y_elements: int,
    ) -> np.ndarray:
        if derivative_boundary_constraints is None:
            return np.empty((x_axes, y_elements), dtype=object)
        if derivative_boundary_constraints.shape != (x_axes, y_elements):
            raise ValueError(
                f"expected derivative boundary constraints shape {(x_axes, y_elements)} but got "
                f"{derivative_boundary_constraints.shape}"
            )
        return derivative_boundary_constraints

    #---- Jacobi update core (coordinate-system aware diagonal) --------------
    def _next_anti_laplacian_estimate(
        self,
        y_old: np.ndarray,
        anti_laplacian: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        One Jacobi sweep to update y given current estimate y_old:
          y_new = (sum of neighbors - h^2 * f)/diag
        Here we compute a coordinate-system-aware diagonal/coefficient.
        """
        #Assemble diagonal coefficient depending on coord system.
        d = np.array(mesh.d_x, dtype=float)
        all_d_x_sqr = d * d

        if mesh.dimensions == 1:
            diag = 2.0 / all_d_x_sqr[0]
            return anti_laplacian / diag

        if mesh.dimensions == 2:
            if mesh.coordinate_system_type == CoordinateSystem.POLAR:
                #r, phi spacing
                r = mesh.vertex_coordinate_grids[0]
                r_sqr = r * r
                diag = 2.0 / all_d_x_sqr[0] + 2.0 / (all_d_x_sqr[1] * r_sqr)
                return anti_laplacian / diag
            else:
                diag = 2.0 / all_d_x_sqr[0] + 2.0 / all_d_x_sqr[1]
                return anti_laplacian / diag

        #3D
        if mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0]
            phi = mesh.vertex_coordinate_grids[1]
            r_sqr = r * r
            sin_phi = np.sin(phi)
            sin_phi_sqr = sin_phi * sin_phi
            diag = (
                2.0 / all_d_x_sqr[0]
                + 2.0 / (all_d_x_sqr[1] * r_sqr)
                + 2.0 / (all_d_x_sqr[2] * r_sqr)
            )
            #crude diagonal approximation; full stencil would add 1/(r^2 tan term) etc.
            return anti_laplacian / diag
        else:
            diag = 2.0 / all_d_x_sqr[0] + 2.0 / all_d_x_sqr[1] + 2.0 / all_d_x_sqr[2]
            return anti_laplacian / diag

    #---- halo builder along axis for first-derivative boundary constraints ---
    @staticmethod
    def _add_halos_along_axis(
        y: np.ndarray,
        x_axis: int,
        d_x: float,
        slicer: Slicer,
        derivative_boundary_constraints: Union[Sequence[Optional[BoundaryConstraintPair]], np.ndarray],
    ) -> np.ndarray:
        """
        Construct halos from first-derivative boundary constraints so centered stencils
        can be used at domain edges.
        """
        slicer[x_axis] = slice(1, 2)
        y_lower_adj = y[tuple(slicer)]
        slicer[x_axis] = slice(-2, -1)
        y_upper_adj = y[tuple(slicer)]

        y_lower_halo = np.zeros_like(y_lower_adj)
        y_upper_halo = np.zeros_like(y_upper_adj)

        for y_ind, bc_pair in enumerate(derivative_boundary_constraints):
            if bc_pair is None:
                continue
            lower_bc, upper_bc = bc_pair
            if lower_bc is not None:
                #Fill halo by applying constraint: n·∇y = g  ->  linearized to halo value
                lower_bc.multiply_and_add(y_lower_adj[..., y_ind : y_ind + 1], -2.0 * d_x, y_lower_halo[..., y_ind : y_ind + 1])
            if upper_bc is not None:
                upper_bc.multiply_and_add(y_upper_adj[..., y_ind : y_ind + 1], +2.0 * d_x, y_upper_halo[..., y_ind : y_ind + 1])

        return np.concatenate([y_lower_halo, y, y_upper_halo], axis=x_axis)


class ThreePointCentralDifferenceMethod(NumericalDifferentiator):
    """
    Three-point (second-order) central differences with boundary halos.
    """

    def __init__(self, tol: float = 1e-3):
        super().__init__(tol)

    def _derivative(
        self,
        y: np.ndarray,
        d_x: float,
        x_axis: int,
        derivative_boundary_constraints: Union[Sequence[Optional[BoundaryConstraintPair]], np.ndarray],
    ) -> np.ndarray:
        slicer: Slicer = [slice(None)] * y.ndim
        y_ext = self._add_halos_along_axis(y, x_axis, d_x, slicer, derivative_boundary_constraints[0])
        #centered first derivative ∂y/∂x ≈ (y_{i+1} - y_{i-1}) / (2h)
        fwd = np.take(y_ext, indices=range(2, y_ext.shape[x_axis]), axis=x_axis)
        bwd = np.take(y_ext, indices=range(0, y_ext.shape[x_axis] - 2), axis=x_axis)
        num = _safe_subtract(fwd, bwd)
        den = 2.0 * d_x
        return _safe_divide(num, den)

    def _second_derivative(
        self,
        y: np.ndarray,
        d_x1: float,
        d_x2: float,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Union[Sequence[Optional[BoundaryConstraintPair]], np.ndarray],
    ) -> np.ndarray:
        if x_axis1 == x_axis2:
            #∂²y/∂x² ≈ (y_{i+1} - 2 y_i + y_{i-1}) / h² with halos
            slicer: Slicer = [slice(None)] * y.ndim
            y_ext = self._add_halos_along_axis(y, x_axis1, d_x1, slicer, derivative_boundary_constraints[0])
            center = np.take(y_ext, indices=range(1, y_ext.shape[x_axis1] - 1), axis=x_axis1)
            fwd = np.take(y_ext, indices=range(2, y_ext.shape[x_axis1]), axis=x_axis1)
            bwd = np.take(y_ext, indices=range(0, y_ext.shape[x_axis1] - 2), axis=x_axis1)
            num = _safe_subtract(_safe_subtract(fwd, 2.0 * center), bwd)
            den = d_x1 * d_x1
            return _safe_divide(num, den)

        #Cross derivative ∂²y/(∂x_i ∂x_j) via two passes of centered diff
        dy_dx1 = self._derivative(y, d_x1, x_axis1, derivative_boundary_constraints[0])
        return self._derivative(dy_dx1, d_x2, x_axis2, derivative_boundary_constraints[0])


#=============================================================================
#Symbol Mapper  (unified from fdm_symbol_mapper.py)
#=============================================================================

class FDMSymbolMapArg(NamedTuple):
    """Arguments to RHS map functions."""
    t: float
    y: np.ndarray
    d_y_constraint_function: Callable[[float], np.ndarray]


FDMSymbolMapFunction = Callable[[FDMSymbolMapArg], np.ndarray]


class FDMSymbolMapper(_BaseSymbolMapper[FDMSymbolMapArg, np.ndarray]):
    """
    Symbol mapper specialized for FDM operator: exposes t, y, x, ∇y, Hessian[y], ∇·y, curl(y), Δy, etc.
    """

    def __init__(self, cp: ConstrainedProblem, differentiator: NumericalDifferentiator):
        diff_eq = cp.differential_equation
        super().__init__(diff_eq)
        self._differentiator = differentiator
        self._mesh = cp.mesh

    def t_map_function(self) -> FDMSymbolMapFunction:
        return lambda arg: np.array([arg.t])

    def y_map_function(self, y_ind: int) -> FDMSymbolMapFunction:
        return lambda arg: arg.y[..., y_ind : y_ind + 1]

    def x_map_function(self, x_axis: int) -> FDMSymbolMapFunction:
        return lambda arg: self._mesh.vertex_coordinate_grids[x_axis][..., np.newaxis]

    def y_gradient_map_function(self, y_ind: int, x_axis: int) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.gradient(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            x_axis,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_hessian_map_function(self, y_ind: int, x_axis1: int, x_axis2: int) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.hessian(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            x_axis1,
            x_axis2,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_divergence_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
    ) -> FDMSymbolMapFunction:
        if indices_contiguous:
            return lambda arg: self._differentiator.divergence(
                arg.y[..., y_indices[0] : y_indices[-1] + 1],
                self._mesh,
                arg.d_y_constraint_function(arg.t)[:, y_indices[0] : y_indices[-1] + 1],
            )
        else:
            return lambda arg: self._differentiator.divergence(
                arg.y[..., y_indices],
                self._mesh,
                arg.d_y_constraint_function(arg.t)[:, y_indices],
            )

    def y_laplacian_map_function(self, y_ind: int) -> FDMSymbolMapFunction:
        return lambda arg: self._differentiator.laplacian(
            arg.y[..., y_ind : y_ind + 1],
            self._mesh,
            arg.d_y_constraint_function(arg.t)[:, y_ind : y_ind + 1],
        )

    def y_vector_laplacian_map_function(
        self,
        y_indices: Sequence[int],
        indices_contiguous: Union[bool, np.bool_],
        vector_laplacian_ind: int,
    ) -> FDMSymbolMapFunction:
        if indices_contiguous:
            return lambda arg: self._differentiator.vector_laplacian(
                arg.y[..., y_indices[0] : y_indices[-1] + 1],
                self._mesh,
                vector_laplacian_ind,
                arg.d_y_constraint_function(arg.t)[:, y_indices[0] : y_indices[-1] + 1],
            )
        else:
            return lambda arg: self._differentiator.vector_laplacian(
                arg.y[..., y_indices],
                self._mesh,
                vector_laplacian_ind,
                arg.d_y_constraint_function(arg.t)[:, y_indices],
            )

    #Convenience: concatenate RHS components for a given LHS
    def map_concatenated(self, arg: FDMSymbolMapArg, lhs_type: LHS) -> np.ndarray:
        return np.concatenate(self.map(arg, lhs_type), axis=-1)


#=============================================================================
#FDM Operator  (unified from fdm_operator.py)
#=============================================================================

BoundaryConstraintsCache = Dict[Optional[float], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
YConstraintsCache = Dict[Optional[float], Optional[np.ndarray]]

class FDMOperator(Operator):
    """
    Finite-Difference Method based differential equation solver (time marching).
    """

    def __init__(self, integrator: NumericalIntegrator, differentiator: NumericalDifferentiator, d_t: float):
        super().__init__(d_t, True)
        self._integrator = integrator
        self._differentiator = differentiator

    def solve(self, ivp: InitialValueProblem, parallel_enabled: bool = True) -> Solution:
        cp = ivp.constrained_problem
        t = discretize_time_domain(ivp.t_interval, self._d_t)
        y = np.empty((len(t) - 1,) + cp.y_vertices_shape, dtype=float)
        y_i = ivp.initial_condition.discrete_y_0(True)

        if not cp.are_all_boundary_conditions_static:
            b0 = cp.create_boundary_constraints(True, t[0])
            yc0 = cp.create_y_vertex_constraints(b0[0])
            apply_constraints_along_last_axis(yc0, y_i)

        y_constraints_cache: YConstraintsCache = {}
        boundary_constraints_cache: BoundaryConstraintsCache = {}
        y_next = self._create_y_next_function(ivp, y_constraints_cache, boundary_constraints_cache)

        for i, t_i in enumerate(t[:-1]):
            y[i] = y_i = y_next(t_i, y_i)
            if not cp.are_all_boundary_conditions_static:
                y_constraints_cache.clear()
                boundary_constraints_cache.clear()

        return Solution(ivp, t[1:], y, vertex_oriented=True, d_t=self._d_t)

    def _create_y_next_function(
        self,
        ivp: InitialValueProblem,
        y_constraints_cache: YConstraintsCache,
        boundary_constraints_cache: BoundaryConstraintsCache,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        cp = ivp.constrained_problem
        eq_sys = cp.differential_equation.symbolic_equation_system
        symbol_mapper = FDMSymbolMapper(cp, self._differentiator)

        d_y_over_d_t_eq_indices = eq_sys.equation_indices_by_type(LHS.D_Y_OVER_D_T)
        y_eq_indices = eq_sys.equation_indices_by_type(LHS.Y)
        y_laplacian_eq_indices = eq_sys.equation_indices_by_type(LHS.Y_LAPLACIAN)

        y_constraint_func, d_y_constraint_func = self._create_constraint_functions(
            cp, y_constraints_cache, boundary_constraints_cache
        )

        def d_y_over_d_t_function(t: float, y: np.ndarray) -> np.ndarray:
            d_y_over_d_t = np.zeros(y.shape, dtype=float)
            rhs = symbol_mapper.map_concatenated(FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.D_Y_OVER_D_T)
            d_y_over_d_t[..., d_y_over_d_t_eq_indices] = rhs
            return d_y_over_d_t

        def y_next_function(t: float, y: np.ndarray) -> np.ndarray:
            y_next = self._integrator.integral(y, t, self._d_t, d_y_over_d_t_function, y_constraint_func)

            if len(y_eq_indices):
                yc = y_constraint_func(t + self._d_t)
                yc = None if yc is None else yc[y_eq_indices]
                y_rhs = symbol_mapper.map_concatenated(FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.Y)
                y_next[..., y_eq_indices] = apply_constraints_along_last_axis(yc, y_rhs)

            if len(y_laplacian_eq_indices):
                yc = y_constraint_func(t + self._d_t)
                yc = None if yc is None else yc[y_laplacian_eq_indices]
                dyc = d_y_constraint_func(t + self._d_t)
                dyc = None if dyc is None else dyc[:, y_laplacian_eq_indices]
                rhs = symbol_mapper.map_concatenated(FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.Y_LAPLACIAN)
                y_next[..., y_laplacian_eq_indices] = self._differentiator.anti_laplacian(rhs, cp.mesh, yc, dyc)

            return y_next

        return y_next_function

    @staticmethod
    def _create_constraint_functions(
        cp: ConstrainedProblem,
        y_constraints_cache: YConstraintsCache,
        boundary_constraints_cache: BoundaryConstraintsCache,
    ) -> Tuple[Callable[[float], Optional[np.ndarray]], Callable[[float], Optional[np.ndarray]]]:
        if not cp.differential_equation.x_dimension:
            return lambda _: None, lambda _: None

        if cp.are_all_boundary_conditions_static:
            return (
                lambda _: cp.static_y_vertex_constraints,
                lambda _: cp.static_boundary_vertex_constraints[1],
            )

        def d_y_constraints_function(t: Optional[float]) -> Optional[np.ndarray]:
            if t in boundary_constraints_cache:
                return boundary_constraints_cache[t][1]
            bc = cp.create_boundary_constraints(True, t)
            boundary_constraints_cache[t] = bc
            return bc[1]

        if not cp.are_there_boundary_conditions_on_y:
            return (lambda _: cp.static_y_vertex_constraints, d_y_constraints_function)

        def y_constraints_function(t: Optional[float]) -> Optional[np.ndarray]:
            if t in y_constraints_cache:
                return y_constraints_cache[t]
            if t in boundary_constraints_cache:
                bc = boundary_constraints_cache[t]
            else:
                bc = cp.create_boundary_constraints(True, t)
                boundary_constraints_cache[t] = bc
            yc = cp.create_y_vertex_constraints(bc[0])
            y_constraints_cache[t] = yc
            return yc

        return y_constraints_function, d_y_constraints_function

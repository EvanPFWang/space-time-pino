

from __future__ import annotations

import math
import numpy as np
from typing import Callable, Iterable, Sequence, Tuple

def kahan_sum(values: Iterable[float]) -> float:
    """Computesum of a sequence of floats usingKahan compensated
    summation algorithm.
    """
    total = 0.0
    c = 0.0  #running compensation for lost low‑order bits.
    for value in values:
        y = value - c       #apply compensation.
        t = total + y       #sum adjusted value to currtotal.
        c = (t - total) - y  #compute new compensation.
        total = t
    return total


def pairwise_sum(values: Sequence[float]) -> float:

    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    #recursively sumleft and right halves.
    mid = n // 2
    return pairwise_sum(values[:mid]) + pairwise_sum(values[mid:])


def safe_subtract(x: float, y: float, threshold: float = 1e-12) -> float:

    #Compute relative magnitude ofdifference.
    denom = max(abs(x), abs(y), 1.0)
    if abs(x - y) <= threshold * denom and (x + y) != 0.0:
        #identity (x^2 - y^2) / (x + y) to avoid cancellation
        return (x * x - y * y) / (x + y)
    return x - y


def stable_sqrt1_minus_delta(delta: float) -> float:

    #guard for negative values in sqrt due to rounding.
    one_minus_delta = 1.0 - delta
    if one_minus_delta < 0:
        #if delta slightly exceeds 1 due to rounding, clamp to zero.
        one_minus_delta = 0.0
    sqrt_term = math.sqrt(one_minus_delta)
    #if delta tiny,denom close to 2 + direct formula stable
    denom = 1.0 + sqrt_term
    if denom != 0.0:
        return delta / denom
    #Fallback todirect formula (won't occur for real delta <= 1).
    return 1.0 - sqrt_term


def quadratic_roots_stable(a: float, b: float, c: float) -> Tuple[float, float]:

    if a == 0:
        raise ZeroDivisionError("Coefficient 'a' must be non‑zero for a quadratic equation")
    discriminant = b * b - 4.0 * a * c
    #use complex sqrt for neg discriminant to handle complex roots.
    sqrt_disc = math.sqrt(discriminant) if discriminant >= 0 else complex(0.0, math.sqrt(-discriminant))
    #determine sign of b for stable formula.  If b == 0, choose +1.
    sign_b = 1.0 if b >= 0 else -1.0
    #computefirst root usingstable formula.
    denom = -b - sign_b * sqrt_disc
    #avoid division by zero if denom is zero (happens when discriminant == 0).
    if denom == 0:
        #discriminant zero implies repeated root; both roots equal -b/(2a).
        root = -b / (2.0 * a)
        return (root, root)
    x1 = denom / (2.0 * a)
    #computesecond root usingrelationship x1 * x2 = c/a.
    x2 = (2.0 * c) / denom
    return (x1, x2)


def safe_divide(numerator: float, denominator: float, eps: float = 1e-15) -> float:

    if abs(denominator) < eps:
        raise ZeroDivisionError(f"Denominator {denominator!r} is too close to zero to safely divide")
    return numerator / denominator


def derivative_central(f: Callable[[float], float], x: float, eps: float | None = None) -> float:
    """Approximatederivative of a scalar function using a central difference.
    """
    if eps is None:
        eps = np.finfo(float).eps
    #choose h proportional to sqrt(eps) and magnitude of x
    h = math.sqrt(eps) * max(abs(x), 1.0)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def logsumexp(values: Sequence[float]) -> float:
    """Compute log(sum[exp(values)]) in a numerically stable way.
    """
    if len(values) == 0:
        return -math.inf
    m = max(values)
    #If m is -∞, all values are -∞ and result is -∞
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
        #Use Kahan compensated summation to accumulateterm.
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    #return even if convergence was not achieved to avoid infinite loops.
    return total, max_terms


__all__ = [
    "kahan_sum",
    "pairwise_sum",
    "safe_subtract",
    "stable_sqrt1_minus_delta",
    "quadratic_roots_stable",
    "safe_divide",
    "derivative_central",
    "logsumexp",
    "relative_error",
    "absolute_error",
    "series_sum_until_convergence",
]
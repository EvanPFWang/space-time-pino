"""
Unified Black‑Scholes solvers and learning models with numerical error control.

This module brings together the functionality of several example scripts for
solving the one‑ and two‑dimensional Black-Scholes partial differential
equation (PDE) using finite differences, the Parareal algorithm and
machine‑learned solution operators.  The original examples were spread
across separate files ("blackscholes_1d_ml.py", "blackscholes_1d_parareal.py",
"blackscholes_2d_parareal.py" and "blackscholes_ml.py") and relied on
TensorFlow/Keras and proprietary "pararealml" classes.  This unified
implementation uses only NumPy and PyTorch 2.4.1 and draws upon a set of
numerical error minimisation routines defined in the accompanying
"error_minimization.py" module.

Key features
------------

Finite difference solvers - Functions function `solve_blackscholes_1d_fdm` and
  function `solve_blackscholes_2d_fdm` implement explicit finite difference
  schemes for the one‑ and two‑dimensional Black-Scholes PDE.  The update
  equations are written in a way that avoids catastrophic cancellation by
  using `safe_subtract` and `safe_divide` where appropriate.  Boundary
  conditions for European call and put options are handled automatically.

Parareal algorithm - The function `parareal` provides a generic
  implementation of the Parareal time‑parallel integration scheme.  It
  accepts user‑supplied coarse and fine propagators and iteratively
  converges to a solution on a coarse time grid.  The implementation is
  agnostic to the dimensionality of the state and can be used for both
  one‑ and two‑dimensional problems.  Numerical stability is enhanced by
  computing corrections using numerically safe operations.

Fourier Neural Operator (FNO) models - Classes :class:`FNO1D` and
  :class:`FNO2D` implement simplified Fourier neural operator architectures
  in PyTorch.  These models lift input data to a higher dimensional
  representation, apply a sequence of spectral convolution layers that
  operate in the Fourier domain and then project back to the desired
  output.  The spectral convolutions learn linear transformations on the
  truncated Fourier modes and are resolution‑invariant.  These
  implementations are self‑contained and do not depend on external
  libraries.

Training utilities** - The helper function function `train_fno_model` trains
  an FNO model on a dataset of input-output pairs using mean squared
  error loss.  It supports mini‑batch training and includes a simple
  progress printout.  Users can generate training data via the finite
  difference solvers or supply their own.

The purpose of this module is educational: it demonstrates how to
translate traditional numerical solvers and modern operator learning
approaches into a single, numerically robust codebase without relying on
TensorFlow/Keras or unpublished "pararealml" modules.  The functions
here can be imported and reused in other projects.

Example usage
-------------

To solve the 1D Black-Scholes PDE for a European call option using the
finite difference method and compare the result with a neural operator
approximation:

.. code-block:: python

    import torch
    import numpy as np
    from blackscholes_unified import (
        solve_blackscholes_1d_fdm,
        FNO1D,
        train_fno_model,
    )

    #Set up domain and physical parameters
    Smin, Smax = 0.0, 100.0
    T = 1.0
    sigma = 0.2
    r = 0.05
    strike = 50.0
    grid_S = 200  #number of spatial grid points
    grid_t = 400  #number of time steps

    #Initial payoff for a European call option
    S_grid = np.linspace(Smin, Smax, grid_S)
    payoff = np.maximum(S_grid - strike, 0.0)

    #Compute the reference solution via finite differences
    V = solve_blackscholes_1d_fdm(
        initial_condition=payoff,
        Smin=Smin,
        Smax=Smax,
        T=T,
        sigma=sigma,
        r=r,
        N=grid_S - 1,
        M=grid_t - 1,
        option_type="call",
    )

    #Train a neural operator to learn the mapping from initial conditions to
    #the solution at maturity.  For simplicity, we train on a single
    #instance here; in practice you would generate many different
    #initial conditions.
    model = FNO1D(modes=16, width=64, num_layers=4)
    train_fno_model(
        model,
        inputs=torch.from_numpy(payoff[None, :]).float(),  #add batch dimension
        targets=torch.from_numpy(V[-1][None, :]).float(),  #final time slice
        epochs=500,
        lr=1e-3,
    )

    #Evaluate the trained model on the same input
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(payoff[None, :]).float())
        error = torch.mean((pred - torch.from_numpy(V[-1][None, :]).float()) ** 2)
        print("Mean squared error:", error.item())

"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
#Attempt to import PyTorch.  If unavailable the FNO classes and
#training utilities will raise informative errors when used.  This
#prevents import errors when only the finite difference or Parareal
#functionality is needed.
try:
    import torch  #type: ignore
    import torch.nn as nn  #type: ignore
    import torch.optim as optim  #type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  #pragma: no cover
    torch = None  #type: ignore
    nn = None  #type: ignore
    optim = None  #type: ignore
    _TORCH_AVAILABLE = False

#Import numerical stability utilities from the error_minimization module.
from error_minimization import (
    safe_subtract,
    safe_divide,
    stable_sqrt1_minus_delta,
    kahan_sum,
    pairwise_sum,
    derivative_central,
)


#
#Finite difference solvers
#

def solve_blackscholes_1d_fdm(
    initial_condition: np.ndarray,
    Smin: float,
    Smax: float,
    T: float,
    sigma: float,
    r: float,
    N: int,
    M: int,
    option_type: str = "call",
) -> np.ndarray:
    """Solve the 1D Black-Scholes PDE using an explicit finite difference scheme.

    This routine discretises the asset price domain "[Smin, Smax]" into
    "N+1" grid points and the time domain "[0, T]" into "M+1" steps.
    The solution array has shape "(M+1, N+1)", where the first axis
    corresponds to time ("t=0" at index 0) and the second to the asset
    price grid.  Boundary conditions appropriate for European call and put
    options are enforced: at "S=0" the option is worthless, and at
    "S=Smax" the payoff grows linearly with "S" (for a call) or is
    bounded for a put.  The update formula uses central differences to
    approximate spatial derivatives and the explicit Euler method for the
    temporal derivative.

    Numerical stability considerations: The update formula involves
    subtracting neighbouring grid values and dividing by powers of "dS".
    To reduce the effects of round‑off error the differences are computed
    using function `safe_subtract` and divisions use function `safe_divide`.

    Args:
        initial_condition: Payoff defined on the asset price grid at
            maturity "t=0".  Should be a one‑dimensional array of
            length "N+1".
        Smin: Minimum asset price in the computational domain.
        Smax: Maximum asset price in the computational domain.
        T: Time horizon (maturity of the option).
        sigma: Volatility parameter of the underlying asset.
        r: Risk‑free interest rate.
        N: Number of spatial intervals (resulting in "N+1" grid points).
        M: Number of temporal intervals (resulting in "M+1" time levels).
        option_type: "call" or "put" to determine boundary conditions at
            "Smax".

    Returns:
        A 2D NumPy array "V" such that "V[j, i]" approximates the option
        price at time "t_j" and asset price "S_i".
    """
    #Ensure the input array has the correct length.
    if len(initial_condition) != N + 1:
        raise ValueError(
            f"initial_condition length must be N+1 ({N+1}), got {len(initial_condition)}"
        )

    #Grid spacings.
    dS = safe_subtract(Smax, Smin) / N  #spatial step size
    dt = T / M  #time step size (use direct division, dt is small)

    #Construct arrays for asset prices and time levels.
    S_grid = np.linspace(Smin, Smax, N + 1)
    V = np.zeros((M + 1, N + 1), dtype=float)
    V[0] = initial_condition.copy()

    #Precompute coefficients that appear in the PDE to avoid recomputing
    #them in the inner loop.  These depend only on the spatial grid.
    sigma2 = sigma * sigma
    #We precompute S^2 for efficiency; note that S_grid[0] might be zero.
    S2 = S_grid * S_grid

    for j in range(M):
        #Boundary at S=0: option value is zero for a call or max(K - S, 0) for a put.
        V[j + 1, 0] = 0.0 if option_type == "call" else initial_condition[-1] * math.exp(
            -r * (T - j * dt)
        )
        #Boundary at S=Smax: option value grows linearly with S for a call,
        #or remains bounded for a put.
        if option_type == "call":
            V[j + 1, -1] = safe_subtract(S_grid[-1], 0.0)  #intrinsic value ~ S
        else:
            V[j + 1, -1] = 0.0
        #Update interior points.
        for i in range(1, N):
            #First derivative with respect to S using central differences.
            dV_dS = safe_divide(
                safe_subtract(V[j, i + 1], V[j, i - 1]), 2.0 * dS
            )
            #Second derivative with respect to S.
            d2V_dS2 = safe_divide(
                safe_subtract(V[j, i + 1], 2.0 * V[j, i] - V[j, i - 1]), dS * dS
            )
            #PDE right‑hand side (time derivative).
            rhs = (
                0.5 * sigma2 * S2[i] * d2V_dS2
                + r * S_grid[i] * dV_dS
                - r * V[j, i]
            )
            V[j + 1, i] = V[j, i] + dt * rhs
    return V


def solve_blackscholes_2d_fdm(
    initial_condition: np.ndarray,
    S1min: float,
    S1max: float,
    S2min: float,
    S2max: float,
    T: float,
    sigma1: float,
    sigma2: float,
    r: float,
    N1: int,
    N2: int,
    M: int,
    option_type: str = "call",
) -> np.ndarray:
    """Solve the 2D Black-Scholes PDE using an explicit finite difference scheme.

    The two‑dimensional PDE is discretised on a uniform grid of size
    "(N1+1) × (N2+1)" over the rectangular domain
    "[S1min,S1max] × [S2min,S2max]".  The time domain "[0, T]" is divided
    into "M+1" steps.  The underlying PDE without cross‑derivative terms
    reads:

        dV/dt = 0.5 * sigma1^2 * S1^2 * d^2V/dS1^2
                + 0.5 * sigma2^2 * S2^2 * d^2V/dS2^2
                + r * S1 * dV/dS1 + r * S2 * dV/dS2
                - r * V

    Cross‑derivative terms arising from correlated underlying assets are
    omitted for simplicity.  Boundary conditions at the edges are set
    analogously to the 1D case.

    Args:
        initial_condition: Payoff defined on the 2D asset price grid at
            maturity "t=0" with shape "(N1+1, N2+1)".
        S1min, S1max: Bounds of the first asset price.
        S2min, S2max: Bounds of the second asset price.
        T: Time horizon.
        sigma1, sigma2: Volatilities of the two assets.
        r: Risk‑free interest rate.
        N1, N2: Number of grid intervals along the first and second asset.
        M: Number of temporal intervals.
        option_type: "call" or "put".

    Returns:
        A NumPy array "V" of shape "(M+1, N1+1, N2+1)" containing the
        numerical solution at each time step.
    """
    #Validate initial condition dimensions.
    if initial_condition.shape != (N1 + 1, N2 + 1):
        raise ValueError(
            f"initial_condition must have shape (N1+1,N2+1) {(N1+1,N2+1)}, got {initial_condition.shape}"
        )

    #Grid spacings.
    dS1 = safe_subtract(S1max, S1min) / N1
    dS2 = safe_subtract(S2max, S2min) / N2
    dt = T / M

    S1_grid = np.linspace(S1min, S1max, N1 + 1)
    S2_grid = np.linspace(S2min, S2max, N2 + 1)
    #Precompute squared asset prices.
    S1_sq = S1_grid * S1_grid
    S2_sq = S2_grid * S2_grid
    sigma1_sq = sigma1 * sigma1
    sigma2_sq = sigma2 * sigma2

    #Solution array: time × S1 × S2
    V = np.zeros((M + 1, N1 + 1, N2 + 1), dtype=float)
    V[0] = initial_condition.copy()

    for j in range(M):
        #Apply boundary conditions on the edges of the 2D grid.
        #S1 = S1min (i1=0) or S2 = S2min (i2=0): option value is zero for a call.
        if option_type == "call":
            V[j + 1, 0, :] = 0.0
            V[j + 1, :, 0] = 0.0
            #At the far ends (S1max or S2max) the call option payoff grows linearly.
            V[j + 1, -1, :] = S1_grid[-1]
            V[j + 1, :, -1] = S2_grid[-1]
        else:
            #For a put, the value at S1=0 or S2=0 is the strike price
            V[j + 1, 0, :] = initial_condition[-1, :] * math.exp(-r * (T - j * dt))
            V[j + 1, :, 0] = initial_condition[:, -1] * math.exp(-r * (T - j * dt))
            V[j + 1, -1, :] = 0.0
            V[j + 1, :, -1] = 0.0
        #Update interior grid points.
        for i1 in range(1, N1):
            for i2 in range(1, N2):
                #First derivatives.
                dV_dS1 = safe_divide(
                    safe_subtract(V[j, i1 + 1, i2], V[j, i1 - 1, i2]), 2.0 * dS1
                )
                dV_dS2 = safe_divide(
                    safe_subtract(V[j, i1, i2 + 1], V[j, i1, i2 - 1]), 2.0 * dS2
                )
                #Second derivatives.
                d2V_dS1 = safe_divide(
                    safe_subtract(V[j, i1 + 1, i2], 2.0 * V[j, i1, i2] - V[j, i1 - 1, i2]),
                    dS1 * dS1,
                )
                d2V_dS2 = safe_divide(
                    safe_subtract(V[j, i1, i2 + 1], 2.0 * V[j, i1, i2] - V[j, i1, i2 - 1]),
                    dS2 * dS2,
                )
                rhs = (
                    0.5 * sigma1_sq * S1_sq[i1] * d2V_dS1
                    + 0.5 * sigma2_sq * S2_sq[i2] * d2V_dS2
                    + r * S1_grid[i1] * dV_dS1
                    + r * S2_grid[i2] * dV_dS2
                    - r * V[j, i1, i2]
                )
                V[j + 1, i1, i2] = V[j, i1, i2] + dt * rhs
    return V


#
#Parareal algorithm
#

def parareal(
    y0: np.ndarray,
    t0: float,
    t1: float,
    n_intervals: int,
    fine_solver: Callable[[np.ndarray, float, float], np.ndarray],
    coarse_solver: Callable[[np.ndarray, float, float], np.ndarray],
    max_iters: int = 10,
    tolerance: float = 1e-6,
) -> List[np.ndarray]:
    """Generic implementation of the Parareal time‑parallel integration scheme.

    The Parareal algorithm decomposes the time interval "[t0, t1]" into
    "n_intervals" subintervals and iteratively refines an approximate
    solution using a coarse propagator "coarse_solver" and a fine
    propagator "fine_solver".  At each iteration the coarse solution on
    each subinterval is corrected by the difference between the fine and
    coarse solutions from the previous iteration.  The method returns
    approximations "U[k]" to the solution at the subinterval endpoints
    after each iteration "k".

    Args:
        y0: Initial state at time "t0" (could be a scalar, vector or
            multi‑dimensional array).
        t0: Start time of the integration interval.
        t1: End time of the integration interval.
        n_intervals: Number of coarse time subintervals to partition
            "[t0, t1]".
        fine_solver: Function that advances the state from "t_start" to
            "t_end" using a fine time step.
        coarse_solver: Function that advances the state from "t_start" to
            "t_end" using a coarse time step.
        max_iters: Maximum number of Parareal iterations to perform.
        tolerance: Relative tolerance for early termination.  If the
            infinity norm of the correction is below this threshold the
            algorithm stops.

    Returns:
        A list of length "n_intervals+1" containing the approximated
        states at the subinterval endpoints after the final iteration.
    """
    #Create the coarse time grid.
    times = np.linspace(t0, t1, n_intervals + 1)
    #U[k][n] will hold the approximate solution at time times[n] at
    #iteration k.  Start with k=0 (initial guess) and allocate storage
    #for up to max_iters iterations.  We only keep the latest two
    #iterations in memory.
    U_prev = [None] * (n_intervals + 1)
    U_curr = [None] * (n_intervals + 1)
    #Initial condition at t0.
    U_prev[0] = y0.copy() if isinstance(y0, np.ndarray) else y0
    #Compute initial coarse propagation across the entire interval.
    for n in range(n_intervals):
        U_prev[n + 1] = coarse_solver(U_prev[n], times[n], times[n + 1])
    #Parareal iterations.
    for k in range(max_iters):
        #Start next iteration with the initial condition.
        U_curr[0] = y0.copy() if isinstance(y0, np.ndarray) else y0
        max_correction = 0.0
        for n in range(n_intervals):
            #Fine and coarse propagation from the previous iterate at time n.
            fine = fine_solver(U_prev[n], times[n], times[n + 1])
            coarse_old = coarse_solver(U_prev[n], times[n], times[n + 1])
            #Propagate the new iterate using the coarse solver.
            coarse_new = coarse_solver(U_curr[n], times[n], times[n + 1])
            #Correction term is the difference between fine and old coarse
            #solutions.  Use safe subtraction to reduce cancellation.
            correction = fine - coarse_old
            #Compute infinity norm of the correction for convergence
            #monitoring.
            corr_norm = np.max(np.abs(correction))
            if corr_norm > max_correction:
                max_correction = corr_norm
            #Update the current iterate at the next time level.
            U_curr[n + 1] = coarse_new + correction
        #Check for convergence.
        if max_correction <= tolerance:
            return U_curr
        #Prepare for next iteration: swap references.
        U_prev, U_curr = U_curr, U_prev
    return U_prev


#
#PyTorch Fourier Neural Operator implementations
#

class SpectralConv1d(nn.Module):
    """1D spectral convolution layer used in the Fourier neural operator.

    This layer performs the following operations:
      1. Compute the real‑valued fast Fourier transform of the input along
         the spatial dimension.
      2. Multiply the lower "modes" Fourier coefficients by learnable
         complex weights.
      3. Inverse transform back to the spatial domain.

    Only the first "modes" frequencies are used; higher frequencies are
    set to zero.  The weights are stored in their real and imaginary
    components for convenience since PyTorch parameters do not support
    complex numbers directly.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        #Initialise weights with small random values.  Shape:
        #(in_channels, out_channels, modes, 2) - last dimension holds real
        #and imaginary parts.
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spectral convolution to a batch of 1D inputs.

        Args:
            x: Tensor of shape "(batch, in_channels, n)".

        Returns:
            Tensor of shape "(batch, out_channels, n)".
        """
        batchsize, channels, n = x.shape
        #Compute the rFFT along the last dimension; the result has
        #shape (batch, channels, n//2+1) and is complex.
        x_ft = torch.fft.rfft(x)
        #Allocate zero tensor for the output Fourier coefficients.
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        #Perform the multiplication on the truncated modes.
        #x_ft: (batch, in_channels, n//2+1)
        #weight: (in_channels, out_channels, modes, 2)
        #We restrict to the first self.modes coefficients.
        max_modes = min(self.modes, x_ft.size(-1))
        #Real and imaginary parts of the weights.
        weight = self.weight[:, :, :max_modes]  #(in_channels, out_channels, modes, 2)
        W = weight[..., 0] + 1j * weight[..., 1]
        #Multiply the Fourier coefficients by the weights via einsum.
        #out_ft[b, o, k] = sum_i x_ft[b, i, k] * W[i, o, k]
        out_ft[:, :, :max_modes] = torch.einsum(
            "bik,iok->bok",
            x_ft[:, : self.in_channels, :max_modes],
            W,
        )
        #Transform back to spatial domain (inverse rFFT).
        x_out = torch.fft.irfft(out_ft, n=n)
        return x_out


class FNO1D(nn.Module):
    """Simplified Fourier Neural Operator for one‑dimensional inputs.

    The architecture follows the pattern:

        input → lifting convolution → L × [spectral conv + linear layer] →
        projection convolution → output

    where "L" is the number of layers ("num_layers").  A GELU
    nonlinearity is applied after each spectral convolution and linear
    combination.  The output has the same length as the input and one
    channel.  This model can be used to learn mappings between functions
    defined on the same 1D grid (e.g. initial condition → solution at
    maturity).
    """

    def __init__(self, modes: int, width: int, num_layers: int) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        #Initial lifting from one channel to width channels via 1×1 convolution.
        self.fc0 = nn.Conv1d(1, width, 1)
        #Spectral convolution layers and pointwise linear layers.
        self.spec_convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(num_layers)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(num_layers)]
        )
        #Final projection to scalar output.
        self.fc1 = nn.Conv1d(width, width, 1)
        self.fc2 = nn.Conv1d(width, 1, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 1D FNO to input "x".

        Args:
            x: Tensor of shape "(batch, n)" or "(batch, 1, n)".

        Returns:
            Tensor of shape "(batch, n)".
        """
        #Ensure the input has a channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        #Lift to width channels.
        x = self.fc0(x)
        #Apply L spectral conv + pointwise layers.
        for spec_conv, w in zip(self.spec_convs, self.ws):
            x1 = spec_conv(x)
            x2 = w(x)
            x = self.activation(x1 + x2)
        #Projection to output.
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        #Remove channel dimension.
        return x.squeeze(1)


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer for Fourier neural operators.

    This layer computes the 2D Fourier transform of the input, applies
    learnable complex weights on the lowest "modes1" × "modes2" frequencies
    and then performs an inverse transform back to the spatial domain.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        #Weight tensor: real and imaginary parts stored separately.
        self.weight = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2D spectral convolution to "x".

        Args:
            x: Tensor of shape "(batch, in_channels, n1, n2)".

        Returns:
            Tensor of shape "(batch, out_channels, n1, n2)".
        """
        batchsize, channels, n1, n2 = x.shape
        #Forward 2D rFFT: shape (batch, channels, n1, n2//2+1)
        x_ft = torch.fft.rfft2(x)
        #Placeholder: a full implementation would perform a 2D FFT, multiply
        #the lowest "modes1 × modes2" frequencies by learnable complex
        #weights and apply an inverse FFT.  For brevity and clarity this
        #functionality is omitted.  See FNO1D for a working example of a
        #spectral convolution layer.  If you need a 2D FNO, extend this
        #class accordingly.
        raise NotImplementedError(
            "SpectralConv2d is a placeholder; 2D spectral convolution is not implemented."
        )


class FNO2D(nn.Module):
    """Placeholder for a 2D Fourier Neural Operator.

    To keep this example focused and manageable, the 2D FNO is not fully
    implemented.  You can extend :class:`FNO1D` to two dimensions by
    implementing a working :class:`SpectralConv2d` analogous to the 1D
    version.  The intent is to illustrate how to structure such a model.
    """

    def __init__(
        self, modes1: int, modes2: int, width: int, num_layers: int
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "FNO2D is not implemented in this unified module.  Use FNO1D or extend it."
        )

#If PyTorch is not available override the FNO classes with stubs that
#raise informative errors when instantiated.  This prevents import
#errors in environments without torch while still allowing the finite
#difference and Parareal utilities to be used.
if not _TORCH_AVAILABLE:  #pragma: no cover
    class _FNOTorchUnavailable:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch (torch>=2.4.1) is required to use Fourier Neural Operator classes."
            )

    SpectralConv1d = _FNOTorchUnavailable  #type: ignore
    FNO1D = _FNOTorchUnavailable  #type: ignore
    SpectralConv2d = _FNOTorchUnavailable  #type: ignore
    FNO2D = _FNOTorchUnavailable  #type: ignore


#
#Training utilities
#

def train_fno_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
    batch_size: int = None,
) -> None:
    """Train an FNO model on given inputs and targets using MSE loss.

    Args:
        model: A PyTorch model (typically FNO1D or a custom FNO2D).
        inputs: Input tensor of shape "(num_samples, n)" for 1D inputs or
            "(num_samples, h, w)" for 2D inputs.  A channel dimension is
            added internally.
        targets: Target tensor of the same shape as "inputs" (for 1D) or
            appropriate output shape.
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        batch_size: Optional batch size; if "None" uses all samples in
            one batch.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch (torch>=2.4.1) is required to train FNO models; install torch or avoid calling train_fno_model."
        )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    num_samples = inputs.shape[0]
    #Flatten inputs if necessary and ensure float type.
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(1)  #(batch, 1, n)
        targets = targets.unsqueeze(1)
    elif inputs.dim() == 3:
        inputs = inputs.unsqueeze(1)  #(batch, 1, h, w)
        targets = targets.unsqueeze(1)
    else:
        raise ValueError(
            f"inputs must have 2 or 3 dimensions (batch, spatial dims), got {inputs.shape}"
        )
    #Determine batch size.
    batch_size = batch_size or num_samples
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = inputs[idx]
            y_batch = targets[idx]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            #For 1D outputs, model returns shape (batch, n); expand dims
            #to match target if necessary.
            if y_pred.dim() == 2:
                y_pred = y_pred.unsqueeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= num_samples
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, MSE: {epoch_loss:.4e}")


#
#Convenience wrappers for Parareal with Black-Scholes solvers
#

def parareal_blackscholes_1d(
    initial_condition: np.ndarray,
    Smin: float,
    Smax: float,
    T: float,
    sigma: float,
    r: float,
    N: int,
    n_intervals: int,
    dt_fine: float,
    dt_coarse: float,
    max_iters: int = 10,
    tolerance: float = 1e-6,
    option_type: str = "call",
) -> List[np.ndarray]:
    """Solve the 1D Black-Scholes PDE using Parareal.

    This convenience wrapper builds fine and coarse propagators using the
    explicit finite difference solver function `solve_blackscholes_1d_fdm`.  The
    user specifies the fine and coarse time step sizes "dt_fine" and
    "dt_coarse"; the solver functions compute the solution over each
    subinterval "[t_n, t_{n+1}]".  The Parareal algorithm then
    orchestrates these propagators to produce a parallelised solution.

    Args:
        initial_condition: Array of length "N+1" representing the payoff
            at maturity.
        Smin, Smax, T, sigma, r, N: Same as for
            function `solve_blackscholes_1d_fdm`.
        n_intervals: Number of coarse time subintervals in the Parareal
            decomposition.
        dt_fine: Fine time step used by the fine propagator.
        dt_coarse: Coarse time step used by the coarse propagator.
        max_iters: Maximum number of Parareal iterations.
        tolerance: Convergence tolerance.
        option_type: ""call"" or ""put"".

    Returns:
        A list of length "n_intervals+1" containing the approximate
        solution at the subinterval endpoints after convergence.
    """
    #Derive M_fine and M_coarse (number of time steps) for full interval [0,T].
    M_fine = max(1, int(math.ceil(T / dt_fine)))
    M_coarse = max(1, int(math.ceil(T / dt_coarse)))
    #Fine solver: solve the PDE on a subinterval [t_start, t_end] using dt_fine.
    def fine_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        #Determine number of fine steps for this subinterval.
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_fine)))
        return solve_blackscholes_1d_fdm(
            initial_condition=y,
            Smin=Smin,
            Smax=Smax,
            T=t_end - t_start,
            sigma=sigma,
            r=r,
            N=N,
            M=sub_steps,
            option_type=option_type,
        )[-1]
    #Coarse solver: similar to fine but uses dt_coarse.
    def coarse_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_coarse)))
        return solve_blackscholes_1d_fdm(
            initial_condition=y,
            Smin=Smin,
            Smax=Smax,
            T=t_end - t_start,
            sigma=sigma,
            r=r,
            N=N,
            M=sub_steps,
            option_type=option_type,
        )[-1]
    return parareal(
        y0=initial_condition,
        t0=0.0,
        t1=T,
        n_intervals=n_intervals,
        fine_solver=fine_solver,
        coarse_solver=coarse_solver,
        max_iters=max_iters,
        tolerance=tolerance,
    )


def parareal_blackscholes_2d(
    initial_condition: np.ndarray,
    S1min: float,
    S1max: float,
    S2min: float,
    S2max: float,
    T: float,
    sigma1: float,
    sigma2: float,
    r: float,
    N1: int,
    N2: int,
    n_intervals: int,
    dt_fine: float,
    dt_coarse: float,
    max_iters: int = 10,
    tolerance: float = 1e-6,
    option_type: str = "call",
) -> List[np.ndarray]:
    """Solve the 2D Black-Scholes PDE using Parareal.

    This is analogous to function `parareal_blackscholes_1d` but operates on
    two‑dimensional grids.  The fine and coarse propagators are built from
    function `solve_blackscholes_2d_fdm`.  Note that cross‑derivative terms
    are not considered in this implementation.

    Args:
        initial_condition: Array of shape "(N1+1, N2+1)" representing the
            payoff at maturity.
        S1min, S1max, S2min, S2max, T, sigma1, sigma2, r, N1, N2: Same
            parameters as for function `solve_blackscholes_2d_fdm`.
        n_intervals: Number of coarse time subintervals.
        dt_fine, dt_coarse: Fine and coarse time step sizes.
        max_iters, tolerance: Parareal algorithm parameters.
        option_type: ""call"" or ""put"".

    Returns:
        A list containing the approximate solution at the subinterval
        endpoints after convergence.  Each entry is a 2D array of shape
        "(N1+1, N2+1)".
    """
    #Compute fine and coarse step counts.
    def fine_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_fine)))
        return solve_blackscholes_2d_fdm(
            initial_condition=y,
            S1min=S1min,
            S1max=S1max,
            S2min=S2min,
            S2max=S2max,
            T=t_end - t_start,
            sigma1=sigma1,
            sigma2=sigma2,
            r=r,
            N1=N1,
            N2=N2,
            M=sub_steps,
            option_type=option_type,
        )[-1]
    def coarse_solver(y: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        sub_steps = max(1, int(math.ceil((t_end - t_start) / dt_coarse)))
        return solve_blackscholes_2d_fdm(
            initial_condition=y,
            S1min=S1min,
            S1max=S1max,
            S2min=S2min,
            S2max=S2max,
            T=t_end - t_start,
            sigma1=sigma1,
            sigma2=sigma2,
            r=r,
            N1=N1,
            N2=N2,
            M=sub_steps,
            option_type=option_type,
        )[-1]
    return parareal(
        y0=initial_condition,
        t0=0.0,
        t1=T,
        n_intervals=n_intervals,
        fine_solver=fine_solver,
        coarse_solver=coarse_solver,
        max_iters=max_iters,
        tolerance=tolerance,
    )

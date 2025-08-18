import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split
CPU_DEVICE_TYPE = "CPU"


from enum import Enum
from typing import Callable, Dict, Generic, Optional, Sequence, Set, TypeVar, Union
from    parareal_unified_updated import Operator,discretize_time_domain,TemporalDomainInterval, VectorizedInitialConditionFunction
#Define LHS types for differential equations (time derivative, state, Laplacian, etc.)
class LHS(Enum):
    """Types of left-hand sides in differential equation systems."""
    D_Y_OVER_D_T = 0
    Y = 1
    Y_LAPLACIAN = 2

from abc import ABC, abstractmethod

from collections.abc import Iterator

class Dataset:
    """
    Generates and holds all data needed to train a physics-informed model (domain, initial, boundary collocation points).
    """
    def __init__(self, cp: Any, t_interval: Tuple[float, float],
                 y_0_functions: Iterable[Any],
                 point_sampler: CollocationPointSampler,
                 n_domain_points: int, n_boundary_points: int = 0, vertex_oriented: bool = False):
        x_dim = cp.differential_equation.x_dimension
        if n_domain_points <= 0:
            raise ValueError(f"number of domain points ({n_domain_points}) must be > 0")
        if n_boundary_points < 0:
            raise ValueError(f"number of boundary points ({n_boundary_points}) must be >= 0")
        if x_dim == 0 and n_boundary_points:
            raise ValueError("number of boundary points must be 0 for ODEs")

        self._cp = cp
        self._t_interval = t_interval
        self._y_0_functions = list(y_0_functions)
        self._point_sampler = point_sampler
        self._n_domain_points = n_domain_points
        self._n_boundary_points = n_boundary_points
        self._vertex_oriented = vertex_oriented

        #Evaluate initial conditions on mesh
        if x_dim:
            x_coords = cp.mesh.all_index_coordinates(vertex_oriented, flatten=True)
            self._initial_value_data = np.vstack([func(x_coords).flatten() for func in self._y_0_functions])
        else:
            self._initial_value_data = np.array([func(None) for func in self._y_0_functions])
        self._initial_value_data.setflags(write=False)

        #Sample collocation points in domain
        domain_points = point_sampler.sample_domain_points(n_domain_points, t_interval, cp.mesh if x_dim else None)
        self._domain_collocation_data = (np.concatenate((domain_points.t, domain_points.x), axis=1) if x_dim else domain_points.t)
        self._domain_collocation_data.setflags(write=False)

        #Initial collocation points (t=0 and spatial coordinates)
        if x_dim:
            x0 = cp.mesh.all_index_coordinates(vertex_oriented, flatten=True)
            t0 = np.zeros((x0.shape[0], 1))
            self._initial_collocation_data = np.concatenate((t0, x0), axis=1)
        else:
            self._initial_collocation_data = np.zeros((1, 1))
        self._initial_collocation_data.setflags(write=False)

        #Boundary collocation points
        if x_dim and n_boundary_points > 0:
            boundary_sets = point_sampler.sample_boundary_points(n_boundary_points, t_interval, cp.mesh)
            #Flatten boundary data: each entry in boundary_sets is AxialBoundaryPoints for an axis
            data_list = []
            for axis_idx, axial_points in enumerate(boundary_sets):
                for end, colloc in enumerate([axial_points.lower_boundary_points, axial_points.upper_boundary_points]):
                    if colloc is None:
                        continue
                    t_arr = colloc.t
                    x_arr = colloc.x
                    #Determine normal direction index (axis index for this boundary)
                    axis_col = np.full((t_arr.shape[0], 1), axis_idx)
                    #Evaluate true solution at these boundary points if needed (for training, assume cp has boundary values?)
                    #Here we just store placeholders for y and normal derivative values:
                    y_arr = np.empty((t_arr.shape[0], cp.differential_equation.y_dimension))
                    d_y_n_arr = np.empty_like(y_arr)
                    #Mark Dirichlet vs Neumann: assume if cp provides a boundary condition value, fill y; if Neumann, fill d_y_n.
                    #For simplicity, treat all as Dirichlet (y provided) and Neumann (d_y/dn) as NaN:
                    y_vals = cp.boundary_conditions[axis_idx][end].value_at(x_arr) if hasattr(cp, "boundary_conditions") else None
                    if y_vals is not None:
                        y_arr[:] = y_vals
                        d_y_n_arr[:] = np.nan
                    else:
                        y_arr[:] = np.nan
                        d_y_n_arr[:] = 0.0  #or np.nan if unknown
                    data_list.append(np.concatenate((t_arr, x_arr, y_arr, d_y_n_arr, axis_col), axis=1))
            if data_list:
                self._boundary_collocation_data = np.vstack(data_list)
            else:
                self._boundary_collocation_data = None
        else:
            self._boundary_collocation_data = None

    @property
    def constrained_problem(self) -> Any:
        return self._cp

    @property
    def initial_value_data(self) -> np.ndarray:
        return self._initial_value_data

    @property
    def domain_collocation_data(self) -> np.ndarray:
        return self._domain_collocation_data

    @property
    def initial_collocation_data(self) -> np.ndarray:
        return self._initial_collocation_data

    @property
    def boundary_collocation_data(self) -> Optional[np.ndarray]:
        return self._boundary_collocation_data

    def get_iterator(self, n_batches: int, n_ic_repeats: int = 1, shuffle: bool = True) -> "DatasetIterator":
        return DatasetIterator(self, n_batches, n_ic_repeats, shuffle)

class DatasetIterator(Iterator):
    """
    Iterator to iterate over dataset batches, computing Cartesian product of initial value data and collocation data.
    """
    def __init__(self, dataset: Dataset, n_batches: int, n_ic_repeats: int = 1, shuffle: bool = True):
        self._dataset = dataset
        self._n_batches = n_batches
        self._n_ic_repeats = n_ic_repeats
        self._shuffle = shuffle

        iv_count = dataset.initial_value_data.shape[0]
        domain_count = dataset.domain_collocation_data.shape[0]
        initial_count = dataset.initial_collocation_data.shape[0]
        boundary_count = 0 if dataset.boundary_collocation_data is None else dataset.boundary_collocation_data.shape[0]

        self._total_domain_size = iv_count * domain_count
        self._total_initial_size = n_ic_repeats * iv_count * initial_count
        self._total_boundary_size = iv_count * boundary_count

        if (self._total_domain_size % n_batches != 0 or
            self._total_initial_size % n_batches != 0 or
            self._total_boundary_size % n_batches != 0):
            raise ValueError("number of batches must divide total data sizes")

        self._domain_batch_size = self._total_domain_size // n_batches
        self._initial_batch_size = self._total_initial_size // n_batches
        self._boundary_batch_size = self._total_boundary_size // n_batches if boundary_count > 0 else 0

        #Precompute indices for Cartesian product selection
        self._domain_indices = self._create_cartesian_indices(iv_count, domain_count)
        base_initial_indices = self._create_cartesian_indices(iv_count, initial_count)
        #Repeat initial collocation data for n_ic_repeats times
        self._initial_indices = np.tile(base_initial_indices, (n_ic_repeats, 1))
        self._boundary_indices = (self._create_cartesian_indices(iv_count, boundary_count) if boundary_count > 0 else None)

        self._batch_index = 0

    def _create_cartesian_indices(self, n1: int, n2: int) -> np.ndarray:
        #Create all pairs (i, j) for i in range(n1), j in range(n2)
        i_indices = np.repeat(np.arange(n1), n2)
        j_indices = np.tile(np.arange(n2), n1)
        return np.vstack((i_indices, j_indices)).T

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> "DatasetIterator":
        self._batch_index = 0
        if self._shuffle:
            np.random.shuffle(self._domain_indices)
            np.random.shuffle(self._initial_indices)
            if self._boundary_indices is not None:
                np.random.shuffle(self._boundary_indices)
        return self

    def __next__(self):
        if self._batch_index >= self._n_batches:
            raise StopIteration
        batch = self[self._batch_index]
        self._batch_index += 1
        return batch

    def __getitem__(self, index: int):
        #Domain batch
        d_start = index * self._domain_batch_size
        d_idx = self._domain_indices[d_start: d_start + self._domain_batch_size]
        iv_idx_dom = d_idx[:, 0]
        coll_idx_dom = d_idx[:, 1]
        domain_iv = self._dataset.initial_value_data[iv_idx_dom]
        domain_coll = self._dataset.domain_collocation_data[coll_idx_dom]
        #Split collocation data into t and x parts
        t_dom = torch.tensor(domain_coll[:, :1], dtype=torch.float32)
        x_dom = torch.tensor(domain_coll[:, 1:], dtype=torch.float32) if self._dataset.constrained_problem.differential_equation.x_dimension else None

        #Initial batch
        i_start = index * self._initial_batch_size
        i_idx = self._initial_indices[i_start: i_start + self._initial_batch_size]
        iv_idx_init = i_idx[:, 0]
        coll_idx_init = i_idx[:, 1]
        init_iv = self._dataset.initial_value_data[iv_idx_init]
        init_coll = self._dataset.initial_collocation_data[coll_idx_init]
        t_init = torch.tensor(init_coll[:, :1], dtype=torch.float32)
        x_init = torch.tensor(init_coll[:, 1:], dtype=torch.float32) if self._dataset.constrained_problem.differential_equation.x_dimension else None
        #Compute "true" solution at initial collocation points (should equal initial condition values)
        y_init = torch.tensor(init_iv, dtype=torch.float32)

        #Boundary batch (if any)
        if self._boundary_indices is not None and self._boundary_batch_size > 0:
            b_start = index * self._boundary_batch_size
            b_idx = self._boundary_indices[b_start: b_start + self._boundary_batch_size]
            iv_idx_b = b_idx[:, 0]
            coll_idx_b = b_idx[:, 1]
            boundary_data = self._dataset.boundary_collocation_data
            subset = boundary_data[coll_idx_b]
            #Columns: t, x..., y, d_y/d_n, axis
            t_b = torch.tensor(subset[:, 0:1], dtype=torch.float32)
            #spatial coordinates (assuming x_dim columns)
            x_cols = 1 + self._dataset.constrained_problem.differential_equation.x_dimension
            x_b = torch.tensor(subset[:, 1:x_cols], dtype=torch.float32)
            y_b = torch.tensor(subset[:, x_cols:x_cols + self._dataset.constrained_problem.differential_equation.y_dimension], dtype=torch.float32)
            d_y_n_b = torch.tensor(subset[:, x_cols + self._dataset.constrained_problem.differential_equation.y_dimension:
                                            x_cols + 2*self._dataset.constrained_problem.differential_equation.y_dimension], dtype=torch.float32)
            axis_b = torch.tensor(subset[:, -1], dtype=torch.float32)
        else:
            t_b = None; x_b = None; y_b = None; d_y_n_b = None; axis_b = None

        #Convert initial value data and domain initial value data to torch as well
        u_domain = torch.tensor(domain_iv, dtype=torch.float32)
        u_initial = torch.tensor(init_iv, dtype=torch.float32)

        domain_batch = (u_domain, t_dom, x_dom)
        initial_batch = (u_initial, t_init, x_init, y_init)
        boundary_batch = None if t_b is None else (u_domain, t_b, x_b, y_b, d_y_n_b, axis_b)
        return domain_batch, initial_batch, boundary_batch
#Spectral convolution layer in 2D (complex weights on Fourier modes)
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        #weight1 and weight2 store real and imaginary parts of complex weights for Fourier modes
        self.weight1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weight2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x shape: (batch, in_channels, n1, n2)
        batch_size, in_ch, n1, n2 = x.shape
        #Compute full 2D FFT of input
        x_ft = torch.fft.fft2(x, dim=(-2, -1))
        #Prepare output spectrum as zeros
        out_ft = torch.zeros(batch_size, self.out_channels, n1, n2, dtype=torch.cfloat, device=x.device)
        #Define complex weights from real and imag parts
        W1 = self.weight1[..., 0] + 1j * self.weight1[..., 1]  #shape (in_ch, out_ch, modes1, modes2)
        W2 = self.weight2[..., 0] + 1j * self.weight2[..., 1]
        #Truncate mode indices to not exceed actual Fourier shape
        max_m1 = min(self.modes1, n1)
        max_m2 = min(self.modes2, n2)
        #Compute contributions for low-frequency (top) modes and high-frequency (bottom) modes
        out_ft[:, :, :max_m1, :max_m2] = torch.einsum('bixy, ioxy->boxy', x_ft[:, :, :max_m1, :max_m2], W1[:, :, :max_m1, :max_m2])
        out_ft[:, :, -max_m1:, :max_m2] = torch.einsum('bixy, ioxy->boxy', x_ft[:, :, -max_m1:, :max_m2], W2[:, :, :max_m1, :max_m2])
        #Inverse FFT to get spatial output, take real part
        x_out = torch.fft.ifft2(out_ft, s=(n1, n2)).real
        return x_out
class FNO2d(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        #Fully-connected layers (1x1 conv in effect) for input lifting and output projection
        self.fc0 = nn.Linear(width, width)
        self.conv_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(4)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)  #intermediate output dense
        self.fc2 = nn.Linear(128, 1)      #final output to 1 value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x expected shape (batch, N) or (batch, width) depending on input preparation
        #Lift input to width channels
        x0 = self.fc0(x)  #shape (batch, width)
        #Reshape to (batch, width, 1, 1) to treat as a 1x1 image with width channels
        x_tensor = x0.view(-1, self.width, 1, 1)
        #Apply Fourier layers
        for conv, lin in zip(self.conv_layers, self.w_layers):
            x1 = conv(x_tensor)
            x2 = lin(x_tensor)
            x_tensor = torch.relu(x1 + x2)
        #Project back to output
        x_tensor = x_tensor.view(-1, self.width)
        x_tensor = torch.relu(self.fc1(x_tensor))
        out = self.fc2(x_tensor)
        return out  #shape (batch, 1)
class FNO(nn.Module):
    """
    Wrapper for Fourier Neural Operator (2D) that matches expected interface.
    """
    def __init__(self, modes1: int = 4, modes2: int = 4, width: int = 64, branch_net_input_size: int = 13):
        super().__init__()
        self._fno = FNO2d(modes1, modes2, width)
        self._branch_net_input_size = branch_net_input_size

    @property
    def branch_net_input_size(self) -> int:
        return self._branch_net_input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Forward through internal FNO2d model
        return self._fno(x)


class CollocationPoints:
    """Collocation points sampled from a spatio-temporal domain."""
    def __init__(self, t: np.ndarray, x: Optional[np.ndarray]):
        self.t = t
        self.x = x

class AxialBoundaryPoints:
    """Boundary collocation points for lower and upper boundaries of a spatial axis."""
    def __init__(self, lower_boundary_points: Optional[CollocationPoints], upper_boundary_points: Optional[CollocationPoints]):
        self.lower_boundary_points = lower_boundary_points
        self.upper_boundary_points = upper_boundary_points

class CollocationPointSampler(ABC):
    """Base class for collocation point samplers."""
    @abstractmethod
    def sample_domain_points(self, n_points: int, t_interval: Tuple[float, float], mesh: Optional[Any]) -> CollocationPoints:
        pass

    @abstractmethod
    def sample_boundary_points(self, n_points: int, t_interval: Tuple[float, float], mesh: Any) -> Sequence[AxialBoundaryPoints]:
        pass

class UniformRandomCollocationPointSampler(CollocationPointSampler):
    """
    A uniform random collocation point sampler.
    """

    def sample_domain_points(
        self,
        n_points: int,
        t_interval: TemporalDomainInterval,
        mesh: Optional[Mesh],
    ) -> CollocationPoints:
        if n_points <= 0:
            raise ValueError(
                f"number of domain points ({n_points}) must be greater than 0"
            )

        t = np.random.uniform(*t_interval, (n_points, 1))
        if mesh is not None:
            x_lower_bounds, x_upper_bounds = zip(*mesh.x_intervals)
            x = np.random.uniform(
                x_lower_bounds, x_upper_bounds, (n_points, mesh.dimensions)
            )
        else:
            x = None
        return CollocationPoints(t, x)

    def sample_boundary_points(
        self, n_points: int, t_interval: TemporalDomainInterval, mesh: Mesh
    ) -> Sequence[AxialBoundaryPoints]:
        if n_points <= 0:
            raise ValueError(
                f"number of boundary points ({n_points}) must be greater "
                f"than 0"
            )

        (lower_t_bound, upper_t_bound) = t_interval
        (lower_x_bounds, upper_x_bounds) = zip(*mesh.x_intervals)

        all_n_boundary_points = np.random.multinomial(
            n_points, np.full(2 * mesh.dimensions, 0.5 / mesh.dimensions)
        )

        boundary_points = []
        for axis in range(mesh.dimensions):
            axial_boundary_points: List[Optional[CollocationPoints]] = []
            for axis_end in range(2):
                n_samples = all_n_boundary_points[2 * axis + axis_end]
                if n_samples == 0:
                    axial_boundary_points.append(None)
                    continue

                t = np.random.uniform(
                    lower_t_bound, upper_t_bound, (n_samples, 1)
                )
                x = np.random.uniform(
                    lower_x_bounds,
                    upper_x_bounds,
                    (n_samples, mesh.dimensions),
                )
                x[:, axis] = mesh.x_intervals[axis][axis_end]
                axial_boundary_points.append(CollocationPoints(t, x))

            boundary_points.append(AxialBoundaryPoints(*axial_boundary_points))

        return boundary_points

SymbolMapArg = TypeVar("SymbolMapArg")
SymbolMapValue = TypeVar("SymbolMapValue")
SymbolMapFunction = Callable[[SymbolMapArg], SymbolMapValue]



class SymbolMapper(Generic[SymbolMapArg, SymbolMapValue]):
    """
    Maps symbolic differential equation expressions to numerical functions.
    """
    def __init__(self, diff_eq):
        """
        diff_eq: differential equation (with symbolic_equation_system and x_dimension)
        """
        self._diff_eq = diff_eq
        #Initialize symbol map dictionary
        self._symbol_map: Dict = self.create_symbol_map()
        #Pre-create RHS evaluation functions for each LHS type
        eq_sys = diff_eq.symbolic_equation_system
        self._rhs_functions: Dict[Optional[LHS], Callable[[SymbolMapArg], Sequence[SymbolMapValue]]] = {
            None: self.create_rhs_map_function(range(len(eq_sys.rhs)))
        }
        for lhs_type in LHS:
            indices = eq_sys.equation_indices_by_type(lhs_type) if hasattr(eq_sys, "equation_indices_by_type") else []
            self._rhs_functions[lhs_type] = self.create_rhs_map_function(indices)

    #Abstract mapping functions for time, state, spatial coordinates, and derivatives:
    def t_map_function(self) -> SymbolMapFunction:
        raise NotImplementedError
    def y_map_function(self, y_ind: int) -> SymbolMapFunction:
        raise NotImplementedError
    def x_map_function(self, x_axis: int) -> SymbolMapFunction:
        raise NotImplementedError
    def y_gradient_map_function(self, y_ind: int, x_axis: int) -> SymbolMapFunction:
        raise NotImplementedError
    def y_hessian_map_function(self, y_ind: int, x_axis1: int, x_axis2: int) -> SymbolMapFunction:
        raise NotImplementedError
    def y_divergence_map_function(self, y_indices: Sequence[int], indices_contiguous: Union[bool, np.bool_]) -> SymbolMapFunction:
        raise NotImplementedError
    def y_curl_map_function(self, y_indices: Sequence[int], indices_contiguous: Union[bool, np.bool_], curl_ind: int) -> SymbolMapFunction:
        raise NotImplementedError
    def y_laplacian_map_function(self, y_ind: int) -> SymbolMapFunction:
        raise NotImplementedError
    def y_vector_laplacian_map_function(self, y_indices: Sequence[int], indices_contiguous: Union[bool, np.bool_], vector_laplacian_ind: int) -> SymbolMapFunction:
        raise NotImplementedError

    def create_symbol_map(self) -> Dict:
        """
        Construct mapping from each sympy symbol in diff eq to a function.
        """
        symbol_map: Dict = {}
        eq_sys = self._diff_eq.symbolic_equation_system
        all_symbols: Set = set().union(*[rhs.free_symbols for rhs in eq_sys.rhs])
        x_dim = self._diff_eq.x_dimension if hasattr(self._diff_eq, "x_dimension") else 0
        for symbol in all_symbols:
            name_tokens = symbol.name.split("_")
            prefix = name_tokens[0]
            indices = [int(ind) for ind in name_tokens[1:]] if len(name_tokens) > 1 else []
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
                #Divergence, curl, vector-laplacian use contiguous index logic
                indices_contiguous = np.all([indices[i] == indices[i+1] - 1 for i in range(len(indices)-1)])
                if prefix == "y-divergence":
                    symbol_map[symbol] = self.y_divergence_map_function(indices, indices_contiguous)
                elif prefix == "y-curl":
                    symbol_map[symbol] = (self.y_curl_map_function(indices, indices_contiguous, 0) if x_dim == 2
                                          else self.y_curl_map_function(indices[:-1], indices_contiguous, indices[-1]))
                elif prefix == "y-vector-laplacian":
                    symbol_map[symbol] = self.y_vector_laplacian_map_function(indices[:-1], indices_contiguous, indices[-1])
        return symbol_map

    def create_rhs_map_function(self, indices: Sequence[int]) -> Callable[[SymbolMapArg], Sequence[SymbolMapValue]]:
        """
        Create a function that evaluates right-hand sides of equations with given indices.
        """
        rhs_exprs = [self._diff_eq.symbolic_equation_system.rhs[i] for i in indices]
        rhs_symbols = set().union(*[expr.free_symbols for expr in rhs_exprs])
        #Prepare substitution functions for each symbol
        subst_functions = [self._symbol_map[sym] for sym in rhs_symbols]
        #Lambdify selected RHS expressions to a numpy-callable function
        #Note: sympy lambdify might not support all, but assume basic support.
        import sympy as sp
        rhs_lambda = sp.lambdify([rhs_symbols], rhs_exprs, "numpy")
        def rhs_map_function(arg: SymbolMapArg) -> Sequence[SymbolMapValue]:
            #Substitute each symbol in rhs with its numeric value
            subs_vals = [f(arg) for f in subst_functions]
            return rhs_lambda([subs_vals])
        return rhs_map_function

    def map(self, arg: SymbolMapArg, lhs_type: Optional[LHS] = None) -> Sequence[SymbolMapValue]:
        """
        Evaluate differential equation's RHS (or subset for given lhs_type) at provided argument.
        """
        return self._rhs_functions[lhs_type](arg)



class PhysicsInformedRegressor(nn.Module):
    """
    Wraps a base regression model with physics-informed loss computation.
    """
    def __init__(self, model: nn.Module, cp: Any,
                 diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0,
                 ic_loss_weight: Union[float, Sequence[float]] = 1.0,
                 bc_loss_weight: Union[float, Sequence[float]] = 1.0,
                 vertex_oriented: bool = False):
        super().__init__()
        diff_eq = cp.differential_equation
        y_dim = diff_eq.y_dimension
        #Ensure loss weights are sequences of length y_dim
        if isinstance(diff_eq_loss_weight, float):
            diff_eq_loss_weight = (diff_eq_loss_weight,) * y_dim
        if isinstance(ic_loss_weight, float):
            ic_loss_weight = (ic_loss_weight,) * y_dim
        if isinstance(bc_loss_weight, float):
            bc_loss_weight = (bc_loss_weight,) * y_dim
        if not (len(diff_eq_loss_weight) == len(ic_loss_weight) == len(bc_loss_weight) == y_dim):
            raise ValueError("length of all loss weight sequences must equal y_dim")

        self.base_model = model  #underlying PyTorch model (e.g., FNO)
        self._cp = cp
        self._diff_eq_loss_weights = tuple(diff_eq_loss_weight)
        self._ic_loss_weights = tuple(ic_loss_weight)
        self._bc_loss_weights = tuple(bc_loss_weight)
        #Initialize symbol mapper for PDE and precompute LHS functions for each equation
        self._symbol_mapper = PhysicsInformedMLSymbolMapper(cp)
        self._diff_eq_lhs_functions = self._create_diff_eq_lhs_functions()

    def _create_diff_eq_lhs_functions(self) -> List[PhysicsInformedMLSymbolMapFunction]:
        diff_eq = self._cp.differential_equation
        lhs_funcs: List[PhysicsInformedMLSymbolMapFunction] = []
        for i, lhs_type in enumerate(diff_eq.symbolic_equation_system.lhs_types):
            if lhs_type == LHS.D_Y_OVER_D_T:
                lhs_funcs.append(lambda arg, _i=i: arg.auto_diff.batch_gradient(arg.t, arg.y_hat[:, _i:_i+1], 0))
            elif lhs_type == LHS.Y:
                lhs_funcs.append(lambda arg, _i=i: arg.y_hat[:, _i:_i+1])
            elif lhs_type == LHS.Y_LAPLACIAN:
                lhs_funcs.append(lambda arg, _i=i: arg.auto_diff.batch_laplacian(arg.x, arg.y_hat[:, _i:_i+1], self._cp.mesh.coordinate_system_type))
            else:
                raise ValueError(f"unsupported LHS type ({lhs_type})")
        return lhs_funcs

    def forward(self, inputs):
        #Combine inputs if provided as tuple (u, t, x), else assume already concatenated tensor
        if isinstance(inputs, tuple):
            u, t, x = inputs
            if x is None:
                #If ODE, input is just initial condition and time
                input_tensor = torch.cat((u, t), dim=1)
            else:
                input_tensor = torch.cat((u, t, x), dim=1)
        else:
            input_tensor = inputs
        return self.base_model(input_tensor)

    def compute_differential_equation_loss(self, domain_batch, training: bool = False) -> torch.Tensor:
        u, t, x = domain_batch
        #Ensure gradients can be computed w.rt t and x
        with AutoDifferentiator(persistent=True) as auto_diff:
            auto_diff.watch(t);
            if x is not None: auto_diff.watch(x)
            #Forward pass
            y_hat = self((u, t, x))
            #Build symbol map argument
            sym_arg = PhysicsInformedMLSymbolMapArg(auto_diff, t, x, y_hat)
            rhs_vals = self._symbol_mapper.map(sym_arg)
            #Compute residual = LHS - RHS for each equation
            residuals = []
            for i, rhs_i in enumerate(rhs_vals):
                lhs_val = self._diff_eq_lhs_functions[i](sym_arg)
                residuals.append(lhs_val - torch.tensor(rhs_i, dtype=torch.float32, device=y_hat.device))
            diff_eq_residual = torch.cat(residuals, dim=1)
        #Mean squared residual for each equation component
        mse = (diff_eq_residual ** 2).mean(dim=0)
        return mse  #shape (y_dim,)

    def compute_initial_condition_loss(self, initial_batch, training: bool = False) -> torch.Tensor:
        u, t, x, y_true = initial_batch
        y_pred = self((u, t, x))
        #Compute difference between predicted and true initial condition
        diff = torch.from_numpy(safe_subtract(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()))
        diff = diff.to(dtype=torch.float32, device=y_pred.device)
        mse = (diff ** 2).mean(dim=0)
        return mse  #shape (y_dim,)

    def compute_boundary_condition_loss(self, boundary_batch, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if boundary_batch is None:
            return (torch.zeros(len(self._diff_eq_loss_weights)), torch.zeros(len(self._diff_eq_loss_weights)))
        u, t, x, y_true, d_y_dn_true, axis = boundary_batch
        #Compute predictions at boundary
        with AutoDifferentiator() as auto_diff:
            auto_diff.watch(x)
            y_pred = self((u, t, x))
        #Compute normal derivative via auto_diff
        d_y_dn_pred = auto_diff.batch_gradient(x, y_pred, axis)
        #Compute Dirichlet BC error (y_pred - y_true), treat NaN in y_true as no Dirichlet condition
        mask_dir = torch.isnan(y_true)
        dir_error = torch.where(mask_dir, torch.zeros_like(y_true),
                                torch.from_numpy(safe_subtract(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy())).to(y_pred.device))
        #Compute Neumann BC error (d_y_dn_pred - d_y_dn_true), treat NaN in true as no Neumann condition
        mask_neu = torch.isnan(d_y_dn_true)
        neu_error = torch.where(mask_neu, torch.zeros_like(d_y_dn_true),
                                torch.from_numpy(safe_subtract(d_y_dn_pred.detach().cpu().numpy(), d_y_dn_true.detach().cpu().numpy())).to(y_pred.device))
        #Mean squared errors
        mse_dir = (dir_error ** 2).mean(dim=0)
        mse_neu = (neu_error ** 2).mean(dim=0)
        return mse_dir, mse_neu

class SKLearnKerasRegressor:
    """
    A wrapper for Keras regression models to implement the implicit
    Scikit-learn model interface.
    """

    def __init__(
        self,
        build_fn: Callable[..., tf.keras.Model],
        batch_size: int = 256,
        epochs: int = 1000,
        verbose: Union[int, str] = "auto",
        callbacks: Sequence[tf.keras.callbacks.Callback] = (),
        validation_split: float = 0.0,
        validation_frequency: int = 1,
        lazy_load_to_gpu: bool = False,
        prefetch_buffer_size: int = 1,
        max_predict_batch_size: Optional[int] = None,
        **build_args: Any,
    ):
        """
        build_fn: a function that compiles and returns the Keras model
            to wrap
        batch_size: the training batch size
        epochs: the number of training epochs
        verbose: controls the level of training and evaluation
            information printed to the stdout stream
        callbacks: any callbacks for the training of the model
        validation_split: the proportion of the training data to use for
            validation
        validation_frequency: the number of training epochs between each
            validation
        lazy_load_to_gpu: whether to avoid loading the entire training
            data set onto the GPU all at once by using lazy loading instead
        prefetch_buffer_size: the number of batches to prefetch if using
            lazy loading to the GPU
        max_predict_batch_size: the maximum batch size to use for
            predictions
        build_args: all the parameters to pass to `build_fn`
        """
        self.build_fn = build_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_frequency = validation_frequency
        self.lazy_load_to_gpu = lazy_load_to_gpu
        self.prefetch_buffer_size = prefetch_buffer_size
        self.max_predict_batch_size = max_predict_batch_size
        self.build_args = build_args

        self._model: Optional[tf.keras.Model] = None

    @property
    def model(self) -> tf.keras.Model:
        """
        The underlying Tensorflow model.
        """
        return self._model

    @model.setter
    def model(self, model: tf.keras.Model):
        self._model = model

    def get_params(self, **_: Any) -> Dict[str, Any]:
        params = {
            "build_fn": self.build_fn,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
            "validation_split": self.validation_split,
            "lazy_load_to_gpu": self.lazy_load_to_gpu,
            "prefetch_buffer_size": self.prefetch_buffer_size,
            "max_predict_batch_size": self.max_predict_batch_size,
        }
        params.update(self.build_args)
        return params

    def set_params(self, **parameters: Any) -> SKLearnKerasRegressor:
        build_fn_arg_names = list(
            inspect.signature(self.build_fn).parameters.keys()
        )
        build_args = {}
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in build_fn_arg_names:
                build_args[key] = value
            else:
                raise ValueError(f"invalid parameter '{key}'")

        self.build_args.update(build_args)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> SKLearnKerasRegressor:
        self._model = self.build_fn(**self.build_args)

        if self.lazy_load_to_gpu:
            with tf.device(CPU_DEVICE_TYPE):
                if self.validation_split:
                    (
                        x_train,
                        x_validate,
                        y_train,
                        y_validate,
                    ) = train_test_split(
                        x,
                        y,
                        test_size=self.validation_split,
                    )
                    training_dataset = (
                        tf.data.Dataset.from_tensor_slices((x_train, y_train))
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )
                    validation_dataset = (
                        tf.data.Dataset.from_tensor_slices(
                            (x_validate, y_validate)
                        )
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )

                else:
                    training_dataset = (
                        tf.data.Dataset.from_tensor_slices((x, y))
                        .batch(self.batch_size)
                        .prefetch(self.prefetch_buffer_size)
                    )
                    validation_dataset = None

            self._model.fit(
                training_dataset,
                epochs=self.epochs,
                validation_data=validation_dataset,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )

        else:
            self._model.fit(
                x,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                validation_freq=self.validation_frequency,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if (
            self.max_predict_batch_size is None
            or len(x) <= self.max_predict_batch_size
        ):
            return self._infer(tf.convert_to_tensor(x, tf.float32)).numpy()

        batch_start_ind = 0
        outputs = []
        while batch_start_ind < len(x):
            batch_end_ind = min(
                batch_start_ind + self.max_predict_batch_size, len(x)
            )
            batch = x[batch_start_ind:batch_end_ind]
            outputs.append(
                self._infer(tf.convert_to_tensor(batch, tf.float32)).numpy()
            )
            batch_start_ind += len(batch)

        return np.concatenate(outputs, axis=0)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.lazy_load_to_gpu:
            with tf.device(CPU_DEVICE_TYPE):
                dataset = (
                    tf.data.Dataset.from_tensor_slices((x, y))
                    .batch(self.batch_size)
                    .prefetch(self.prefetch_buffer_size)
                )

            loss = self._model.evaluate(dataset, verbose=self.verbose)
        else:
            loss = self._model.evaluate(x, y, verbose=self.verbose)

        if isinstance(loss, Sequence):
            return -loss[0]
        return -loss

    @tf.function
    def _infer(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Propagates the inputs through the underlying model.

        inputs: the model inputs
        :return: the model outputs
        """
        return self._model(inputs)


class PhysicsInformedMLOperator(Operator):

    def __init__(
            self,
            sampler: CollocationPointSampler,
            d_t: float,
            vertex_oriented: bool,
            auto_regressive: bool = False,
    ):
        super(PhysicsInformedMLOperator, self).__init__(d_t, vertex_oriented)
        self._sampler = sampler
        self._auto_regressive = auto_regressive
        self._model: Optional[PhysicsInformedRegressor] = None

    @property
    def auto_regressive(self) -> bool:
        return self._auto_regressive

    @property
    def model(self) -> Optional[PhysicsInformedRegressor]:
        return self._model

    @model.setter
    def model(self, model: Optional[PhysicsInformedRegressor]):
        self._model = model

    def solve(
            self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            x_tensor = tf.convert_to_tensor(x, tf.float32)
            u = ivp.initial_condition.y_0(x).reshape((1, -1))
            u_tensor = tf.tile(
                tf.convert_to_tensor(u, tf.float32), (x.shape[0], 1)
            )
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, tf.float32)

        t_tensor = tf.constant(
            self._d_t if self._auto_regressive else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1),
        )

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_i_tensor = self._infer((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_i_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regressive:
                    u_tensor = (
                        tf.tile(
                            tf.reshape(y_i_tensor, (1, -1)),
                            (x_tensor.shape[0], 1),
                        )
                        if diff_eq.x_dimension
                        else tf.reshape(y_i_tensor, u_tensor.shape)
                    )
                else:
                    t_tensor = tf.constant(
                        t[i + 1],
                        dtype=tf.float32,
                        shape=(u_tensor.shape[0], 1),
                    )

        return Solution(
            ivp, t, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t
        )

    def train(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            training_data_args: DataArgs,
            optimization_args: OptimizationArgs,
            model_args: Optional[ModelArgs] = None,
            validation_data_args: Optional[DataArgs] = None,
            test_data_args: Optional[DataArgs] = None,
    ) -> Tuple[tf.keras.callbacks.History, Optional[Sequence[float]]]:

        if model_args is None and self._model is None:
            raise ValueError(
                "the model arguments cannot be None if operator's model "
                "is None"
            )

        # print("training_data_args: ", training_data_args)

        if self._auto_regressive:
            if t_interval != (0.0, self._d_t):
                raise ValueError(
                    "in auto-regressive mode, training time interval "
                    f"{t_interval} must range from 0 to time step size of "
                    f"the operator ({self._d_t})"
                )

            diff_eq = cp.differential_equation
            t_symbol = diff_eq.symbols.t
            eq_sys = diff_eq.symbolic_equation_system
            if any([t_symbol in rhs.free_symbols for rhs in eq_sys.rhs]):
                raise ValueError(
                    "auto-regressive mode is not compatible with differential "
                    "equations whose right-hand sides contain any t terms"
                )

            if (
                    diff_eq.x_dimension
                    and not cp.are_all_boundary_conditions_static
            ):
                raise ValueError(
                    "auto-regressive mode is not compatible with dynamic "
                    "boundary conditions"
                )

        training_dataset = self._create_dataset(
            cp, t_interval, training_data_args
        )
        validation_dataset = self._create_dataset(
            cp, t_interval, validation_data_args
        )
        test_dataset = self._create_dataset(cp, t_interval, test_data_args)
        # print("training_dataset: ", training_dataset)

        model = (
            self._model
            if model_args is None
            else PhysicsInformedRegressor(
                cp=cp,
                model=model_args.model,
                diff_eq_loss_weight=model_args.diff_eq_loss_weight,
                ic_loss_weight=model_args.ic_loss_weight,
                bc_loss_weight=model_args.bc_loss_weight,
                vertex_oriented=self._vertex_oriented,
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.get(optimization_args.optimizer)
        )
        history = model.fit(
            training_dataset,
            epochs=optimization_args.epochs,
            steps_per_epoch=training_data_args.n_batches,
            validation_data=validation_dataset,
            validation_steps=validation_data_args.n_batches
            if validation_data_args
            else None,
            validation_freq=optimization_args.validation_frequency,
            callbacks=optimization_args.callbacks,
            verbose=optimization_args.verbose,
        )

        test_loss = (
            model.evaluate(
                test_dataset,
                steps=test_data_args.n_batches,
                verbose=optimization_args.verbose,
            )
            if test_dataset
            else None
        )

        self._model = model

        return history, test_loss

    @tf.function
    def _infer(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
    ) -> tf.Tensor:
        return self.model.__call__(inputs)

    def _create_dataset(
            self,
            cp: ConstrainedProblem,
            t_interval: Tuple[float, float],
            data_args: Optional[DataArgs],
    ) -> Optional[Generator[Sequence[Sequence[tf.Tensor]], None, None]]:

        if not data_args:
            return None

        dataset = Dataset(
            cp=cp,
            t_interval=t_interval,
            y_0_functions=data_args.y_0_functions,
            point_sampler=self._sampler,
            n_domain_points=data_args.n_domain_points,
            n_boundary_points=data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented,
        )
        iterator = dataset.get_iterator(
            n_batches=data_args.n_batches,
            n_ic_repeats=data_args.n_ic_repeats,
            shuffle=data_args.shuffle,
        )
        return iterator.to_infinite_generator()

    def get_config(self):
        config = {
            'model': tf.keras.utils.serialize_keras_object(self._model),
            'cp': tf.keras.utils.serialize_keras_object(self._cp),
            'diff_eq_loss_weight': self._diff_eq_loss_weights,
            'ic_loss_weight': self._ic_loss_weights,
            'bc_loss_weight': self._bc_loss_weights,
        }
        base_config = super(PhysicsInformedMLOperator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config['model'])
        cp = tf.keras.utils.deserialize_keras_object(config['cp'])
        diff_eq_loss_weight = config['diff_eq_loss_weight']
        ic_loss_weight = config['ic_loss_weight']
        bc_loss_weight = config['bc_loss_weight']

        return cls(
            model=model,
            cp=cp,
            diff_eq_loss_weight=diff_eq_loss_weight,
            ic_loss_weight=ic_loss_weight,
            bc_loss_weight=bc_loss_weight,
        )







class DataArgs(NamedTuple):
    y_0_functions: Iterable[VectorizedInitialConditionFunction]
    n_domain_points: int
    n_batches: int
    n_boundary_points: int = 0
    n_ic_repeats: int = 1
    shuffle: bool = True


class ModelArgs(NamedTuple):
    model: tf.keras.Model
    diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0
    ic_loss_weight: Union[float, Sequence[float]] = 1.0
    bc_loss_weight: Union[float, Sequence[float]] = 1.0


class OptimizationArgs(NamedTuple):
    optimizer: Union[str, Dict[str, Any], tf.optimizers.Optimizer]
    epochs: int
    validation_frequency: int = 1
    callbacks: Sequence[tf.keras.callbacks.Callback] = ()
    verbose: Union[str, int] = "auto"


class PhysicsInformedMLOperator(Operator):

    def __init__(
            self,
            sampler: CollocationPointSampler,
            d_t: float,
            vertex_oriented: bool,
            auto_regressive: bool = False,
    ):
        super(PhysicsInformedMLOperator, self).__init__(d_t, vertex_oriented)
        self._sampler = sampler
        self._auto_regressive = auto_regressive
        self._model: Optional[PhysicsInformedRegressor] = None

    @property
    def auto_regressive(self) -> bool:
        return self._auto_regressive

    @property
    def model(self) -> Optional[PhysicsInformedRegressor]:
        return self._model

    @model.setter
    def model(self, model: Optional[PhysicsInformedRegressor]):
        self._model = model

    def solve(
            self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            x_tensor = tf.convert_to_tensor(x, tf.float32)
            u = ivp.initial_condition.y_0(x).reshape((1, -1))
            u_tensor = tf.tile(
                tf.convert_to_tensor(u, tf.float32), (x.shape[0], 1)
            )
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, tf.float32)

        t_tensor = tf.constant(
            self._d_t if self._auto_regressive else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1),
        )

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_i_tensor = self._infer((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_i_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regressive:
                    u_tensor = (
                        tf.tile(
                            tf.reshape(y_i_tensor, (1, -1)),
                            (x_tensor.shape[0], 1),
                        )
                        if diff_eq.x_dimension
                        else tf.reshape(y_i_tensor, u_tensor.shape)
                    )
                else:
                    t_tensor = tf.constant(
                        t[i + 1],
                        dtype=tf.float32,
                        shape=(u_tensor.shape[0], 1),
                    )

        return Solution(
            ivp, t, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t
        )

    def train(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            training_data_args: DataArgs,
            optimization_args: OptimizationArgs,
            model_args: Optional[ModelArgs] = None,
            validation_data_args: Optional[DataArgs] = None,
            test_data_args: Optional[DataArgs] = None,
    ) -> Tuple[tf.keras.callbacks.History, Optional[Sequence[float]]]:

        if model_args is None and self._model is None:
            raise ValueError(
                "the model arguments cannot be None if operator's model "
                "is None"
            )

        # print("training_data_args: ", training_data_args)

        if self._auto_regressive:
            if t_interval != (0.0, self._d_t):
                raise ValueError(
                    "in auto-regressive mode, training time interval "
                    f"{t_interval} must range from 0 to time step size of "
                    f"the operator ({self._d_t})"
                )

            diff_eq = cp.differential_equation
            t_symbol = diff_eq.symbols.t
            eq_sys = diff_eq.symbolic_equation_system
            if any([t_symbol in rhs.free_symbols for rhs in eq_sys.rhs]):
                raise ValueError(
                    "auto-regressive mode is not compatible with differential "
                    "equations whose right-hand sides contain any t terms"
                )

            if (
                    diff_eq.x_dimension
                    and not cp.are_all_boundary_conditions_static
            ):
                raise ValueError(
                    "auto-regressive mode is not compatible with dynamic "
                    "boundary conditions"
                )

        training_dataset = self._create_dataset(
            cp, t_interval, training_data_args
        )
        validation_dataset = self._create_dataset(
            cp, t_interval, validation_data_args
        )
        test_dataset = self._create_dataset(cp, t_interval, test_data_args)
        # print("training_dataset: ", training_dataset)

        model = (
            self._model
            if model_args is None
            else PhysicsInformedRegressor(
                cp=cp,
                model=model_args.model,
                diff_eq_loss_weight=model_args.diff_eq_loss_weight,
                ic_loss_weight=model_args.ic_loss_weight,
                bc_loss_weight=model_args.bc_loss_weight,
                vertex_oriented=self._vertex_oriented,
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.get(optimization_args.optimizer)
        )
        history = model.fit(
            training_dataset,
            epochs=optimization_args.epochs,
            steps_per_epoch=training_data_args.n_batches,
            validation_data=validation_dataset,
            validation_steps=validation_data_args.n_batches
            if validation_data_args
            else None,
            validation_freq=optimization_args.validation_frequency,
            callbacks=optimization_args.callbacks,
            verbose=optimization_args.verbose,
        )

        test_loss = (
            model.evaluate(
                test_dataset,
                steps=test_data_args.n_batches,
                verbose=optimization_args.verbose,
            )
            if test_dataset
            else None
        )

        self._model = model

        return history, test_loss

    @tf.function
    def _infer(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
    ) -> tf.Tensor:
        return self.model.__call__(inputs)

    def _create_dataset(
            self,
            cp: ConstrainedProblem,
            t_interval: Tuple[float, float],
            data_args: Optional[DataArgs],
    ) -> Optional[Generator[Sequence[Sequence[tf.Tensor]], None, None]]:

        if not data_args:
            return None

        dataset = Dataset(
            cp=cp,
            t_interval=t_interval,
            y_0_functions=data_args.y_0_functions,
            point_sampler=self._sampler,
            n_domain_points=data_args.n_domain_points,
            n_boundary_points=data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented,
        )
        iterator = dataset.get_iterator(
            n_batches=data_args.n_batches,
            n_ic_repeats=data_args.n_ic_repeats,
            shuffle=data_args.shuffle,
        )
        return iterator.to_infinite_generator()

    def get_config(self):
        config = {
            'model': tf.keras.utils.serialize_keras_object(self._model),
            'cp': tf.keras.utils.serialize_keras_object(self._cp),
            'diff_eq_loss_weight': self._diff_eq_loss_weights,
            'ic_loss_weight': self._ic_loss_weights,
            'bc_loss_weight': self._bc_loss_weights,
        }
        base_config = super(PhysicsInformedMLOperator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config['model'])
        cp = tf.keras.utils.deserialize_keras_object(config['cp'])
        diff_eq_loss_weight = config['diff_eq_loss_weight']
        ic_loss_weight = config['ic_loss_weight']
        bc_loss_weight = config['bc_loss_weight']

        return cls(
            model=model,
            cp=cp,
            diff_eq_loss_weight=diff_eq_loss_weight,
            ic_loss_weight=ic_loss_weight,
            bc_loss_weight=bc_loss_weight,
        )

class SupervisedMLOperator:
    """
    A supervised ML operator modeling a high-fidelity operator for IVPs.
    Trains a regression model on input-output pairs (initial conditions to next-step solution).
    """
    def __init__(self, d_t: float, vertex_oriented: bool, auto_regressive: bool = True, time_variant: bool = False, input_d_t: bool = False):
        if not auto_regressive and not time_variant:
            raise ValueError("operator must be time_variant if auto_regression is disabled")
        if time_variant and input_d_t:
            raise ValueError("operator must be time invariant to use d_t as input")
        self._d_t = d_t
        self._vertex_oriented = vertex_oriented
        self._auto_regressive = auto_regressive
        self._time_variant = time_variant
        self._input_d_t = input_d_t
        self._model: Optional[Any] = None

    @property
    def auto_regressive(self) -> bool:
        return self._auto_regressive

    @property
    def time_variant(self) -> bool:
        return self._time_variant

    @property
    def input_d_t(self) -> bool:
        return self._input_d_t

    @property
    def model(self) -> Optional[Any]:
        return self._model

    @model.setter
    def model(self, model: Optional[Any]):
        self._model = model

    def solve(self, ivp: Any, parallel_enabled: bool = True) -> Any:
        if self._model is None:
            raise ValueError("operator has no model")
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        y_shape = cp.y_shape(self._vertex_oriented)
        #Create input placeholder for model
        if diff_eq.x_dimension == 0:
            #ODE: input dimension = y_dim + (time or dt)
            inputs = np.empty((1, diff_eq.y_dimension + (1 if self._time_variant else 0)))
        else:
            #PDE: spatial coordinates will be appended; create grid input
            x_coords = cp.mesh.all_index_coordinates(self._vertex_oriented, flatten=True)
            #For PDE, initial state flattened and maybe time
            y_dim = diff_eq.y_dimension
            n_points = x_coords.shape[0]
            if self._time_variant or self._input_d_t:
                inputs = np.hstack([np.empty((n_points, y_dim * n_points)), np.empty((n_points, 1)), x_coords])
            else:
                inputs = np.hstack([np.empty((n_points, y_dim * n_points)), x_coords])
        t_vals = np.arange(ivp.t_interval[0] + self._d_t, ivp.t_interval[1] + 1e-9, self._d_t)
        y = np.empty((len(t_vals),) + y_shape)
        #Initial condition
        y_current = ivp.initial_condition.discrete_y_0(self._vertex_oriented)
        for i, t_i in enumerate(t_vals):
            #Fill input array
            if diff_eq.x_dimension:
                #Flatten current state and replicate for all spatial points
                inputs[:, : inputs.shape[1] - diff_eq.x_dimension - (1 if (self._time_variant or self._input_d_t) else 0)] = y_current.reshape((1, -1))
                if self._time_variant:
                    inputs[:, -diff_eq.x_dimension-1] = t_i
                elif self._input_d_t:
                    inputs[:, -diff_eq.x_dimension-1] = self._d_t
            else:
                inputs[0, : diff_eq.y_dimension] = y_current
                if self._time_variant:
                    inputs[0, -1] = t_i
                elif self._input_d_t:
                    inputs[0, -1] = self._d_t
            #Predict next state
            if isinstance(self._model, nn.Module):
                #If model is a PyTorch model, use forward pass
                inp_tensor = torch.from_numpy(inputs.astype(np.float32))
                out_tensor = self._model(inp_tensor)
                y_next = out_tensor.detach().cpu().numpy()
            else:
                #If model has predict (sklearn/keras), use it
                y_next = self._model.predict(inputs)
            y[i, ...] = y_next.reshape(y_shape)
            #Update current state for next iteration if auto-regressive
            if self._auto_regressive:
                y_current = y_next
        SolutionClass = globals().get("Solution", None)
        return SolutionClass(ivp, t_vals, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t) if SolutionClass else (t_vals, y)


class PhysicsInformedMLSymbolMapArg:
    """
    argument tuple for physics-informed symbol mapping functions.
    """
    def __init__(self, auto_diff, t: torch.Tensor, x: Optional[torch.Tensor], y_hat: torch.Tensor):
        self.auto_diff = auto_diff
        self.t = t
        self.x = x
        self.y_hat = y_hat

PhysicsInformedMLSymbolMapFunction = Callable[[PhysicsInformedMLSymbolMapArg], torch.Tensor]

class PhysicsInformedMLSymbolMapper(SymbolMapper[PhysicsInformedMLSymbolMapArg, torch.Tensor]):
    """
    Symbol mapper for physics-informed ML using auto-differentiation (PyTorch).
    """
    def __init__(self, cp):
        #cp is ConstrainedProblem containing differential_equation, mesh, etc.
        diff_eq = cp.differential_equation
        super().__init__(diff_eq)
        self._coordinate_system_type = cp.mesh.coordinate_system_type if diff_eq.x_dimension else None

    def t_map_function(self) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.t

    def y_map_function(self, y_ind: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.y_hat[:, y_ind:y_ind+1]

    def x_map_function(self, x_axis: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.x[:, x_axis:x_axis+1]

    def y_gradient_map_function(self, y_ind: int, x_axis: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_gradient(arg.x, arg.y_hat[:, y_ind:y_ind+1], x_axis, self._coordinate_system_type)

    def y_hessian_map_function(self, y_ind: int, x_axis1: int, x_axis2: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_hessian(arg.x, arg.y_hat[:, y_ind:y_ind+1], x_axis1, x_axis2, self._coordinate_system_type)

    def y_divergence_map_function(self, y_indices: Sequence[int], indices_contiguous: bool) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_divergence(
            arg.x,
            arg.y_hat[:, y_indices[0]: y_indices[-1]+1] if indices_contiguous else arg.y_hat[:, y_indices],
            self._coordinate_system_type
        )

    def y_curl_map_function(self, y_indices: Sequence[int], indices_contiguous: bool, curl_ind: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_curl(
            arg.x,
            arg.y_hat[:, y_indices[0]: y_indices[-1]+1] if indices_contiguous else arg.y_hat[:, y_indices],
            curl_ind,
            self._coordinate_system_type
        )

    def y_laplacian_map_function(self, y_ind: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_laplacian(arg.x, arg.y_hat[:, y_ind:y_ind+1], self._coordinate_system_type)

    def y_vector_laplacian_map_function(self, y_indices: Sequence[int], indices_contiguous: bool, vector_laplacian_ind: int) -> PhysicsInformedMLSymbolMapFunction:
        return lambda arg: arg.auto_diff.batch_vector_laplacian(
            arg.x,
            arg.y_hat[:, y_indices[0]: y_indices[-1]+1] if indices_contiguous else arg.y_hat[:, y_indices],
            vector_laplacian_ind,
            self._coordinate_system_type
        )

class AutoDifferentiator:
    """
    Provides differential operators using PyTorch autograd.
    Acts as a context manager for recording gradients.
    """
    def __init__(self, persistent: bool = False):
        #persistent flag (not strictly needed in PyTorch, but retained for interface)
        self.persistent = persistent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #Nothing special to release (PyTorch autograd cleans up automatically)
        return False

    def watch(self, tensor: Optional[torch.Tensor]):
        """
        Ensure given tensor requires grad for differentiation.
        """
        if tensor is not None and not tensor.requires_grad:
            tensor.requires_grad_(True)

    def _compute_gradient(self, x: torch.Tensor, y: torch.Tensor, x_axis: Union[int, torch.Tensor],
                          create_graph: bool = False) -> torch.Tensor:
        """
        Helper to compute first derivative(s) of y w.rt x-axis.
        If x_axis is int, returns gradient of y w.rt x[:, x_axis].
        If x_axis is a tensor of indices per sample, returns stacked gradient values per sample.
        """
        #Ensure x requires grad
        if not x.requires_grad:
            x.requires_grad_(True)
        #If y has more than one element, we handle as sum-of-partials approach
        if isinstance(x_axis, int):
            #Compute gradient of each output component w.rt x[:, x_axis]
            #Use torch.autograd.grad on sum of outputs to get batched gradients
            grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
            #grad shape equals x shape
            if grad is None:
                return torch.zeros_like(y)
            return grad[:, x_axis:x_axis+1]  #return gradient for specified axis
        else:
            #x_axis is a tensor of shape [batch] specifying axis per sample
            axis_indices = x_axis.long()
            batch_size = x.shape[0]
            grads_list = []
            for i in range(batch_size):
                axis_i = int(axis_indices[i].item())
                #Compute grad of y[i] w.rt x[i, axis_i] by differentiating scalar output y[i]
                yi = y[i:i+1].sum()  #sum (if y has multiple outputs, sum them to get scalar or assume scalar output per sample)
                grad_full = torch.autograd.grad(yi, x, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
                if grad_full is None:
                    grads_list.append(torch.tensor([0.0], device=x.device))
                else:
                    grads_list.append(grad_full[i, axis_i].unsqueeze(0))
            return torch.stack(grads_list, dim=0)

    def batch_gradient(self, x: torch.Tensor, y: torch.Tensor, x_axis: Union[int, torch.Tensor],
                       coordinate_system_type=None) -> torch.Tensor:
        """
        Returns gradient of y with respect to x's element defined by x_axis.
        Handles coordinate system adjustments for spherical/cylindrical coordinates.
        """
        derivative = self._compute_gradient(x, y, x_axis, create_graph=False)
        if coordinate_system_type is None or coordinate_system_type == 0:  #CoordinateSystem.CARTESIAN
            return derivative
        #If spherical coordinates (assuming CoordinateSystem.SPHERICAL == 3)
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]
            if isinstance(x_axis, int):
                if x_axis == 0:
                    return derivative
                elif x_axis == 1:
                    phi = x[:, 2:3]
                    return derivative / (r * torch.sin(phi))
                else:  #x_axis == 2
                    return derivative / r
            else:
                #For per-sample axis, adjust each element based on axis value
                adj = []
                phi = x[:, 2:3]
                for i, ax in enumerate(x_axis):
                    ax = int(ax.item())
                    if ax == 0:
                        adj.append(derivative[i:i+1])
                    elif ax == 1:
                        adj.append(derivative[i:i+1] / (r[i:i+1] * torch.sin(phi[i:i+1])))
                    else:
                        adj.append(derivative[i:i+1] / r[i:i+1])
                return torch.vstack(adj)
        else:
            #Cylindrical or polar (CoordinateSystem.CYLINDRICAL=2 or POLAR=1)
            if isinstance(x_axis, int):
                if x_axis == 1:
                    r = x[:, :1]
                    return derivative / r
                else:
                    return derivative
            else:
                #Per-sample axis adjustment for cylindrical/polar
                r = x[:, :1]
                adj = []
                for i, ax in enumerate(x_axis):
                    ax = int(ax.item())
                    if ax == 1:
                        adj.append(derivative[i:i+1] / r[i:i+1])
                    else:
                        adj.append(derivative[i:i+1])
                return torch.vstack(adj)

    def batch_hessian(self, x: torch.Tensor, y: torch.Tensor, x_axis1: int, x_axis2: int,
                      coordinate_system_type=None) -> torch.Tensor:
        """
        Returns second partial derivative of y with respect to x_axis1 and x_axis2.
        """
        #First derivative with create_graph to enable second derivative
        grad1 = self._compute_gradient(x, y, x_axis1, create_graph=True)
        second_derivative = self._compute_gradient(x, grad1, x_axis2, create_graph=False)
        if coordinate_system_type is None or coordinate_system_type == 0:
            return second_derivative
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]
            phi = x[:, 2:3]
            #Spherical coordinate Hessian adjustments (for typical radial/angular second derivatives)
            if x_axis1 == 0 and x_axis2 == 0:
                return second_derivative
            elif x_axis1 == 1 and x_axis2 == 1:
                d_y_dr = self._compute_gradient(x, y, 0, create_graph=False)
                d_y_dphi = self._compute_gradient(x, y, 2, create_graph=False)
                return (d_y_dr + ((second_derivative / torch.sin(phi)) + torch.cos(phi) * d_y_dphi) / (r * torch.sin(phi))) / r
            elif x_axis1 == 2 and x_axis2 == 2:
                d_y_dr = self._compute_gradient(x, y, 0, create_graph=False)
                return (second_derivative / r + d_y_dr) / r
            elif (x_axis1 == 0 and x_axis2 == 1) or (x_axis1 == 1 and x_axis2 == 0):
                d_y_dtheta = self._compute_gradient(x, y, 1, create_graph=False)
                return (second_derivative - d_y_dtheta / r) / (r * torch.sin(phi))
            elif (x_axis1 == 0 and x_axis2 == 2) or (x_axis1 == 2 and x_axis2 == 0):
                d_y_dphi = self._compute_gradient(x, y, 2, create_graph=False)
                return (second_derivative - d_y_dphi / r) / r
            else:  #(x_axis1 == 1 and x_axis2 == 2) or vice versa
                d_y_dtheta = self._compute_gradient(x, y, 1, create_graph=False)
                return (torch.sin(phi) * second_derivative - torch.cos(phi) * d_y_dtheta) / (r * (torch.sin(phi)**2))
        else:
            #Cylindrical/Polar Hessian adjustments
            r = x[:, :1]
            if (x_axis1 in (0,2) and x_axis2 in (0,2)):
                return second_derivative
            elif x_axis1 == 1 and x_axis2 == 1:
                d_y_dr = self._compute_gradient(x, y, 0, create_graph=False)
                return (second_derivative / r + d_y_dr) / r
            elif (x_axis1 == 1 and x_axis2 == 0) or (x_axis1 == 0 and x_axis2 == 1):
                d_y_dtheta = self._compute_gradient(x, y, 1, create_graph=False)
                return (second_derivative - d_y_dtheta / r) / r
            else:
                return second_derivative / r

    def batch_divergence(self, x: torch.Tensor, y: torch.Tensor, coordinate_system_type=None) -> torch.Tensor:
        """
        Returns divergence of vector field y.
        """
        if y.shape[1] != x.shape[1]:
            raise ValueError(f"number of y components ({y.shape[1]}) must equal number of x dimensions ({x.shape[1]})")
        x_dim = x.shape[1]
        #Sum of partial derivatives along each axis
        grads = [self._compute_gradient(x, y[..., i:i+1], i, create_graph=False) for i in range(x_dim)]
        divergence = torch.stack(grads, dim=0).sum(dim=0)
        if coordinate_system_type is None or coordinate_system_type == 0:
            return divergence
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]; phi = x[:, 2:3]
            y_r = y[..., :1]; y_theta = y[..., 1:2]; y_phi = y[..., 2:3]
            d_y_r_dr = self._compute_gradient(x, y_r, 0, create_graph=False)
            d_y_theta_dtheta = self._compute_gradient(x, y_theta, 1, create_graph=False)
            d_y_phi_dphi = self._compute_gradient(x, y_phi, 2, create_graph=False)
            #Spherical divergence formula
            return d_y_r_dr + (d_y_phi_dphi + 2*y_r + (d_y_theta_dtheta + torch.cos(phi)*y_phi)/torch.sin(phi)) / r
        else:
            #Cylindrical/Polar
            r = x[:, :1]
            y_r = y[..., :1]; y_theta = y[..., 1:2]
            d_y_r_dr = self._compute_gradient(x, y_r, 0, create_graph=False)
            d_y_theta_dtheta = self._compute_gradient(x, y_theta, 1, create_graph=False)
            div = d_y_r_dr + (y_r + d_y_theta_dtheta) / r
            if int(coordinate_system_type) == 1:  #POLAR
                return div
            else:  #CYLINDRICAL (3D with (r, theta, z))
                y_z = y[..., 2:3]
                d_y_z_dz = self._compute_gradient(x, y_z, 2, create_graph=False)
                return div + d_y_z_dz

    def batch_curl(self, x: torch.Tensor, y: torch.Tensor, curl_ind: int = 0, coordinate_system_type=None) -> torch.Tensor:
        """
        Returns curl (specified component) of vector field y.
        """
        x_dim = x.shape[1]
        if y.shape[1] != x_dim:
            raise ValueError(f"number of y components ({y.shape[1]}) must equal number of x dimensions ({x_dim})")
        if x_dim not in (2, 3):
            raise ValueError(f"curl defined only for 2D or 3D fields (got x_dim={x_dim})")
        if x_dim == 2 and curl_ind != 0:
            raise ValueError("curl index must be 0 for 2D fields")
        if not (0 <= curl_ind < x_dim):
            raise ValueError("invalid curl index")
        if coordinate_system_type is None or coordinate_system_type == 0:
            #Cartesian: use standard formulas
            if x_dim == 2 or curl_ind == 2:
                #For 2D or z-component of 3D curl: ∂y/∂x - ∂x/∂y
                return self._compute_gradient(x, y[..., 1:2], 0, create_graph=False) - self._compute_gradient(x, y[..., 0:1], 1, create_graph=False)
            elif curl_ind == 0:
                #x-component of 3D curl
                return self._compute_gradient(x, y[..., 2:3], 1, create_graph=False) - self._compute_gradient(x, y[..., 1:2], 2, create_graph=False)
            else:  #curl_ind == 1 (y-component)
                return self._compute_gradient(x, y[..., 0:1], 2, create_graph=False) - self._compute_gradient(x, y[..., 2:3], 0, create_graph=False)
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]; phi = x[:, 2:3]
            y_r = y[..., :1]; y_theta = y[..., 1:2]; y_phi = y[..., 2:3]
            if curl_ind == 0:
                d_y_theta_dphi = self._compute_gradient(x, y_theta, 2, create_graph=False)
                d_y_phi_dtheta = self._compute_gradient(x, y_phi, 1, create_graph=False)
                return (d_y_theta_dphi + (torch.cos(phi)*y_theta - d_y_phi_dtheta)/torch.sin(phi)) / r
            elif curl_ind == 1:
                d_y_r_dphi = self._compute_gradient(x, y_r, 2, create_graph=False)
                d_y_phi_dr = self._compute_gradient(x, y_phi, 0, create_graph=False)
                return d_y_phi_dr + (y_phi - d_y_r_dphi) / r
            else:  #curl_ind == 2
                d_y_r_dtheta = self._compute_gradient(x, y_r, 1, create_graph=False)
                d_y_theta_dr = self._compute_gradient(x, y_theta, 0, create_graph=False)
                return -d_y_theta_dr + (d_y_r_dtheta/torch.sin(phi) - y_theta) / r
        else:
            #Cylindrical/Polar
            r = x[:, :1]
            y_r = y[..., :1]; y_theta = y[..., 1:2]
            if int(coordinate_system_type) == 1 or curl_ind == 2:  #Polar or z-component in cylindrical
                d_y_r_dtheta = self._compute_gradient(x, y_r, 1, create_graph=False)
                d_y_theta_dr = self._compute_gradient(x, y_theta, 0, create_graph=False)
                return d_y_theta_dr + (y_theta - d_y_r_dtheta) / r
            elif curl_ind == 0:
                y_z = y[..., 2:3]
                d_y_theta_dz = self._compute_gradient(x, y_theta, 2, create_graph=False)
                d_y_z_dtheta = self._compute_gradient(x, y_z, 1, create_graph=False)
                return d_y_z_dtheta / r - d_y_theta_dz
            else:  #curl_ind == 1
                y_z = y[..., 2:3]
                d_y_r_dz = self._compute_gradient(x, y_r, 2, create_graph=False)
                d_y_z_dr = self._compute_gradient(x, y_z, 0, create_graph=False)
                return d_y_r_dz - d_y_z_dr

    def batch_laplacian(self, x: torch.Tensor, y: torch.Tensor, coordinate_system_type=None) -> torch.Tensor:
        """
        Returns scalar Laplacian of y.
        """
        if coordinate_system_type is None or coordinate_system_type == 0:
            #Sum of second partials along each axis (Cartesian)
            return torch.stack([self._compute_gradient(x, self._compute_gradient(x, y, i, create_graph=True), i, create_graph=False)
                                 for i in range(x.shape[-1])], dim=0).sum(dim=0)
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]; phi = x[:, 2:3]
            d_y_dr = self._compute_gradient(x, y, 0, create_graph=False)
            d_y_dtheta = self._compute_gradient(x, y, 1, create_graph=False)
            d_y_dphi = self._compute_gradient(x, y, 2, create_graph=False)
            d2y_dr2 = self._compute_gradient(x, d_y_dr, 0, create_graph=False)
            d2y_dtheta2 = self._compute_gradient(x, d_y_dtheta, 1, create_graph=False)
            d2y_dphi2 = self._compute_gradient(x, d_y_dphi, 2, create_graph=False)
            return d2y_dr2 + ((2.0 * d_y_dr) + ((d2y_dphi2 + (torch.cos(phi)*d_y_dphi + d2y_dtheta2/torch.sin(phi)))/torch.sin(phi)))/r / r
        else:
            #Polar or cylindrical
            r = x[:, :1]
            d_y_dr = self._compute_gradient(x, y, 0, create_graph=False)
            d_y_dtheta = self._compute_gradient(x, y, 1, create_graph=False)
            d2y_dr2 = self._compute_gradient(x, d_y_dr, 0, create_graph=False)
            d2y_dtheta2 = self._compute_gradient(x, d_y_dtheta, 1, create_graph=False)
            lap = d2y_dr2 + (d2y_dtheta2 / r + d_y_dr) / r
            if int(coordinate_system_type) == 1:  #POLAR (2D)
                return lap
            else:  #CYLINDRICAL (3D)
                d_y_dz = self._compute_gradient(x, y, 2, create_graph=False)
                d2y_dz2 = self._compute_gradient(x, d_y_dz, 2, create_graph=False)
                return lap + d2y_dz2

    def batch_vector_laplacian(self, x: torch.Tensor, y: torch.Tensor, vector_laplacian_ind: int,
                               coordinate_system_type=None) -> torch.Tensor:
        """
        Returns specified component of vector Laplacian of vector field y.
        """
        x_dim = x.shape[1]
        if y.shape[1] != x_dim:
            raise ValueError(f"number of y components ({y.shape[1]}) must match x dimensions ({x_dim})")
        if not (0 <= vector_laplacian_ind < x_dim):
            raise ValueError("vector Laplacian index out of range")
        lap = self.batch_laplacian(x, y[:, vector_laplacian_ind:vector_laplacian_ind+1], coordinate_system_type)
        if coordinate_system_type is None or coordinate_system_type == 0:
            return lap
        if int(coordinate_system_type) == 3:  #SPHERICAL
            r = x[:, :1]; phi = x[:, 2:3]
            y_r = y[:, :1]; y_theta = y[:, 1:2]; y_phi = y[:, 2:3]
            #Adjust vector Laplacian for spherical coordinates
            if vector_laplacian_ind == 1:  #theta component
                d_y_theta_d_theta = self._compute_gradient(x, y_theta, 1, create_graph=False)
                d_y_phi_d_phi = self._compute_gradient(x, y_phi, 2, create_graph=False)
                return lap - 2.0 * (y_r + d_y_phi_d_phi + (torch.cos(phi)*y_phi + d_y_theta_d_theta)/torch.sin(phi)) / (r**2)
            elif vector_laplacian_ind == 2:  #phi component
                d_y_r_d_theta = self._compute_gradient(x, y_r, 1, create_graph=False)
                d_y_phi_d_theta = self._compute_gradient(x, y_phi, 1, create_graph=False)
                return lap + 2.0 * (d_y_r_d_theta + (torch.cos(phi)*d_y_phi_d_theta - 0.5*y_theta)/torch.sin(phi)) / (torch.sin(phi)*(r**2))
            else:  #r component
                d_y_r_d_phi = self._compute_gradient(x, y_r, 2, create_graph=False)
                d_y_theta_d_theta = self._compute_gradient(x, y_theta, 1, create_graph=False)
                return lap + 2.0 * (d_y_r_d_phi - ((0.5*y_phi) + torch.cos(phi)*d_y_theta_d_theta)/(torch.sin(phi)**2)) / (r**2)
        else:
            #Cylindrical/polar vector Laplacian adjustments
            r = x[:, :1]; y_r = y[:, :1]; y_theta = y[:, 1:2]
            if vector_laplacian_ind == 0:  #r component
                d_y_theta_d_theta = self._compute_gradient(x, y_theta, 1, create_graph=False)
                return lap
            elif vector_laplacian_ind == 1:  #theta component
                d_y_r_d_r = self._compute_gradient(x, y_r, 0, create_graph=False)
                return lap - 2.0 * (d_y_r_d_r / r + y_r / (r**2))
            else:  #z component (in cylindrical)
                return lap  #in cylindrical, z-component vector Laplacian is just scalar laplacian of that component
#```````

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

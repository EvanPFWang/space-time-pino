"""helper functions for
seeding random number generators across Python, NumPy and PyTorch,
configuring visible GPU devices in an MPI context and enforcing
deterministic behaviour in PyTorch.  Timing decorators are included to
measure function execution time, both in serial and parallel (MPI)
contexts, while using numerically stable difference operations.


Example usage
-------------

code-block: python

    from utils_unified import set_random_seed, use_cpu, limit_visible_gpus,
    use_deterministic_ops, time, mpi_time

    #Set all RNG seeds for reproducible results
    set_random_seed(42)

    #Force computation on CPU (disable GPU)
    use_cpu()

    #Limit each MPI process to a single GPU based on its rank
    limit_visible_gpus()

    #Enforce deterministic algorithms in PyTorch
    use_deterministic_ops()

    #Decorate a function to measure execution time
    @time("my_function")
    def my_function(x):
        #expensive computation
        return x * x

    result, duration = my_function(10)

    #Decorate a function for MPI timing
    @mpi_time("parallel_function")
    def parallel_function(x):
        #distributed computation
        return x
    value, parallel_duration = parallel_function(123)

"""

from __future__ import annotations

import functools
import os
import random
from timeit import default_timer as timer
from typing import Any, Callable, Optional, Tuple

import numpy as np

from error_minimization import safe_subtract

#Try to import PyTorch for seeding and GPU management.
try:
    import torch  #type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  #pragma: no cover
    torch = None  #type: ignore
    _TORCH_AVAILABLE = False

#Import MPI for synchronisation across ranks.  MPI is optional for
#serial timing; timing functions will still work without it but the
#"mpi_time" decorator requires mpi4py.
try:
    from mpi4py import MPI  #type: ignore
    _MPI_AVAILABLE = True
except ImportError:  #pragma: no cover
    MPI = None  #type: ignore
    _MPI_AVAILABLE = False


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators.

    Reproducibility is critical for numerical experiments and machine
    learning.  This function sets "PYTHONHASHSEED" environment
    variable, seeds built‑in "random" module and NumPy, and, if
    available, PyTorch.  For PyTorch function seeds both CPU and
    CUDA RNGs and configures CuDNN for deterministic behaviour when
    possible.

    Args:
        seed: integer seed to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        #Seed all GPUs if visible.  "cuda.manual_seed_all" is a no‑op
        #when no CUDA devices are present.
        torch.cuda.manual_seed_all(seed)  #type: ignore[operator]
        #Recommended settings for deterministic behaviour.  See
        #https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        #Warn user that PyTorch seeding could not be performed.
        pass


def use_cpu() -> None:
    """Force computations to run on CPU by hiding all GPUs.

    This function sets "CUDA_VISIBLE_DEVICES" to an empty string,
    preventing PyTorch from enumerating any CUDA devices.  Subsequent
    operations will therefore run on CPU.  If PyTorch is not
    available function simply sets environment variable.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #Clearing cache ensures no stray CUDA contexts remain.
    if _TORCH_AVAILABLE and torch.cuda.is_available():  #type: ignore[union-attr]
        torch.cuda.empty_cache()  #type: ignore[func-returns-value]


def limit_visible_gpus() -> None:
    """Restrict each MPI process to a single GPU based on its rank.

    When running under MPI with multiple processes and multiple GPUs,
    it is often desirable to assign one GPU per rank.  This function
    queries number of CUDA devices via PyTorch and size and
    rank of "MPI.COMM_WORLD" communicator.  It then sets
    "CUDA_VISIBLE_DEVICES" to expose only GPU corresponding to
    local rank.  If number of GPUs does not match number
    of MPI processes a "ValueError" is raised.  If PyTorch or
    mpi4py are not available function raises an informative
    "ImportError".
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to query available GPUs; install torch>=2.4.1"
        )
    if not _MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for limit_visible_gpus in an MPI environment"
        )
    n_gpus = torch.cuda.device_count()  #type: ignore[call-arg]
    comm = MPI.COMM_WORLD  #type: ignore[attr-defined]
    if n_gpus == 0:
        return  #nothing to do
    if n_gpus != comm.size:
        raise ValueError(
            f"number of GPUs ({n_gpus}) must match MPI communicator size ({comm.size})"
        )
    rank = comm.rank
    #Expose only GPU corresponding to this rank.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)  #after masking, device 0 maps to selected GPU


def use_deterministic_ops() -> None:
    """Enforce deterministic algorithms in PyTorch.

    Sets environment variables and PyTorch flags that encourage
    deterministic operation, which is important for reproducible
    results.  When CuDNN is used, setting "CUBLAS_WORKSPACE_CONFIG"
    ensures deterministic behaviour in certain operations.  The
    function silently returns if PyTorch is not available.
    """
    #Configure CuBLAS workspace for deterministic behaviour.  See
    #https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if _TORCH_AVAILABLE:
        torch.use_deterministic_algorithms(True, warn_only=True)  #type: ignore[call-arg]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def time(function_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to measure wall‑clock time of a function.

    decorated function is called and its execution time is computed
    using function "timeit.default_timer".  difference between end and
    start times is computed using function "safe_subtract" to guard
    against catastrophic cancellation when run time is very short.
    function returns a tuple "(result, run_time)".  timing
    information is printed regardless of MPI rank.

    Args:
        function_name: Optional explicit name to print; otherwise the
            wrapped function's "__name__" is used.

    Returns:
        A decorator that wraps target function.
    """
    def _time_wrapper_provider(function: Callable, name: Optional[str]) -> Callable:
        disp_name = name or function.__name__

        @functools.wraps(function)
        def _time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            start_time = timer()
            value = function(*args, **kwargs)
            end_time = timer()
            run_time = safe_subtract(end_time, start_time)
            print(f"{disp_name} completed in {run_time}s")
            return value, run_time

        return _time_wrapper

    return lambda function: _time_wrapper_provider(function, function_name)


def mpi_time(function_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to measure execution time across MPI processes.

    Synchronises all ranks using barriers before and after call and
    measures elapsed time with "MPI.Wtime".  difference is
    computed with function "safe_subtract".  Only rank 0 prints the
    timing information.  If mpi4py is not available decorator
    falls back to serial timing version.

    Args:
        function_name: Optional explicit name to print.

    Returns:
        A decorator that wraps target function.
    """
    if not _MPI_AVAILABLE:
        #Fallback to serial timing if MPI is unavailable.
        return time(function_name)

    def _mpi_time_wrapper_provider(function: Callable, name: Optional[str]) -> Callable:
        disp_name = name or function.__name__

        @functools.wraps(function)
        def _mpi_time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            comm = MPI.COMM_WORLD  #type: ignore[attr-defined]
            comm.barrier()
            start_time = MPI.Wtime()  #type: ignore[attr-defined]
            value = function(*args, **kwargs)
            comm.barrier()
            end_time = MPI.Wtime()  #type: ignore[attr-defined]
            run_time = safe_subtract(end_time, start_time)
            if comm.rank == 0:
                print(f"{disp_name} completed in {run_time}s")
            return value, run_time

        return _mpi_time_wrapper

    return lambda function: _mpi_time_wrapper_provider(function, function_name)


__all__ = [
    "set_random_seed",
    "use_cpu",
    "limit_visible_gpus",
    "use_deterministic_ops",
    "time",
    "mpi_time",
]
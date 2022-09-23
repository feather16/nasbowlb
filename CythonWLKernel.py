from importlib import import_module
from typing import Protocol

class NATSBenchCellToWLVector(Protocol):
    def __call__(self, cell: list[int], H: int = 2) -> list[bytes]:
        ...
class NATSBenchCellToWLCounter(Protocol):
    def __call__(self, cell: list[int], H: int = 2) -> dict[bytes, int]:
        ...
class NATSBenchWLKernelFromCounters(Protocol):
    def __call__(self, counter1: dict[bytes, int], counter2: dict[bytes, int]) -> float:
        ...
class NATSBenchWLKernel(Protocol):
    def __call__(self, cell1: list[int], cell2: list[int], H: int = 2) -> float:
        ...
#class NATSBenchWLKernelVector(Protocol):
#    def __call__(self, cell1: list[int], cells: list[list[int]], H: int = 2) -> list[float]:
        ...
natsbench_cell_to_wl_vector: NATSBenchCellToWLVector = \
    import_module('cython_wl_kernel').cython_natsbench_cell_to_wl_kernel_vector
natsbench_cell_to_wl_counter: NATSBenchCellToWLCounter = \
    import_module('cython_wl_kernel').cython_natsbench_cell_to_wl_kernel_counter
natsbench_wl_kernel_from_wl_counters: NATSBenchWLKernelFromCounters = \
    import_module('cython_wl_kernel').cython_natsbench_wl_kernel_from_wl_counters
natsbench_wl_kernel: NATSBenchWLKernel = \
    import_module('cython_wl_kernel').cython_natsbench_wl_kernel
#natsbench_wl_kernel_vector: NATSBenchWLKernelVector = \
#    import_module('cython_wl_kernel').cython_natsbench_wl_kernel_vector
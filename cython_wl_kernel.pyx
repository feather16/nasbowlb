from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cdef extern from "wl_kernel_impl.hpp":
    cdef vector[string] cell_to_wl_kernel_vector(vector[int] cell, int H)
    cdef unordered_map[string, int] cell_to_wl_kernel_counter(vector[int] cell, int H)
    cdef int natsbench_wl_kernel_from_wl_counters(unordered_map[string, int] counter1, unordered_map[string, int] counter2)
    cdef int natsbench_wl_kernel(vector[int] cell1, vector[int] cell2, int H)
    #cdef vector[int] natsbench_wl_kernel_vector(vector[int] cell1, vector[vector[int]] cells, int H)

def cython_natsbench_cell_to_wl_kernel_vector(list cell, int H = 2) -> list:
    return cell_to_wl_kernel_vector(cell, H)
def cython_natsbench_cell_to_wl_kernel_counter(list cell, int H = 2) -> dict:
    return cell_to_wl_kernel_counter(cell, H)
def cython_natsbench_wl_kernel_from_wl_counters(dict counter1, dict counter2) -> float:
    return float(natsbench_wl_kernel_from_wl_counters(counter1, counter2))
def cython_natsbench_wl_kernel(list cell1, list cell2, int H = 2) -> float:
    return float(natsbench_wl_kernel(cell1, cell2, H))
#def cython_natsbench_wl_kernel_vector(list cell1, list cells, int H = 2) -> list:
#    ret = natsbench_wl_kernel_vector(cell1, cells, H)
#    return [float(k) for k in ret]
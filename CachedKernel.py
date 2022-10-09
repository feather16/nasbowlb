import os
import pickle
import numpy as np
from tqdm import tqdm
from typing import Any, Callable, TypeVar, Generic

from NATSBenchWrapper import NATSBenchWrapper

T = TypeVar('T')
class CachedKernel(Generic[T]):
    def __init__(self, kernel: Callable[[Any, Any], float]):
        self.cache: np.ndarray | None = None
        self.kernel = kernel

    @property
    def is_none(self) -> bool:
        return self.cache is None

    def init_empty(self, size: int):
        self.cache = np.full((size, size), -1)
        self.cached = False

    def load_pickle(
            self, 
            kernel_cache_path: str, 
            wrapper: NATSBenchWrapper, 
            verbose: bool = False
            ) -> None:
        if os.path.exists(kernel_cache_path):
            self.cache: np.ndarray = pickle.load(open(kernel_cache_path, mode='rb'))
        else:
            self.cache = self.create_kernel_cache(kernel_cache_path, wrapper, verbose)
        self.cached = True
    
    def __call__(self, index1: int, index2: int, arg1: T, arg2: T) -> float:
        kernel_value = self.cache[index1, index2]
        if kernel_value == -1:
            kernel_value = self.kernel(arg1, arg2)
            self.cache[index1, index2] = kernel_value
            self.cache[index2, index1] = kernel_value
        return kernel_value
    
    def create_kernel_cache(
            self, 
            path: str, 
            wrapper: NATSBenchWrapper, 
            verbose: bool = False
            ) -> np.ndarray:
        N = len(wrapper)
        data = np.empty((N, N))
        indices_itr = []
        for i in range(N):
            indices_itr.extend([(i, j) for j in range(i, N)])
        if verbose:
            print('Creating kernel cache ...')
            indices_itr = tqdm(indices_itr)
        for i, j in indices_itr:
            c1 = wrapper[i].wl_counter
            c2 = wrapper[j].wl_counter
            k = self.kernel(c1, c2)
            data[i, j] = data[j, i] = k
        pickle.dump(data, open(path, mode='wb'))
        return data
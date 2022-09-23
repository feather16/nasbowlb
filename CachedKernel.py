import pickle
import numpy as np
from typing import Any, Callable, TypeVar, Generic

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

    def load_pickle(self, kernel_cache_path: str) -> None:
        self.cache = pickle.load(open(kernel_cache_path, mode='rb'))
        self.cached = True
    
    def __call__(self, index1: int, index2: int, arg1: T, arg2: T) -> float:
        kernel_value = self.cache[index1, index2]
        if kernel_value == -1:
            kernel_value = self.kernel(arg1, arg2)
            self.cache[index1, index2] = kernel_value
            self.cache[index2, index1] = kernel_value
        return kernel_value
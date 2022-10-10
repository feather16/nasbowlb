import numpy as np
from nats_bench.api_topology import NATStopology

from CythonWLKernel import natsbench_cell_to_wl_counter

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OP_TO_INDEX: dict[str, int] = dict(map(lambda kv: kv[::-1], enumerate(OPS)))

class NATSBenchCell:
    # (i, j) <=> (Node_i -> Node_j)    
    def __init__(
            self, 
            arch_str: str, 
            accuracy_dict: dict[str, float], 
            flops: dict[str, float], 
            index: int,
            dataset: str,
            ):
        assert dataset in ['CIFAR-10', 'CIFAR-100', 'ImageNet']
        self.arch_str = arch_str
        self.arch_matrix: np.ndarray = NATStopology.str2matrix(self.arch_str).astype('u1')
        self.accuracy_dict = accuracy_dict
        self._accuracy: float | None = None
        self.flops = flops
        self.index = index
        self.dataset = dataset
        self.label_list: list[int] = self.to_label_list()
        
    def eval(self) -> None:
        self._accuracy = self.accuracy_dict[self.dataset]
        
    @property
    def evaluated(self) -> bool:
        return self._accuracy is not None
        
    @property
    def accuracy(self) -> float:
        assert self._accuracy is not None, 'Cell is not evaluated.'
        return self._accuracy
    
    def to_label_list(self) -> list[int]:
        ret = [-1] * 8
        ret[0] = 0
        ret[1] = self.arch_matrix[1, 0] + 1
        ret[2] = self.arch_matrix[2, 0] + 1
        ret[3] = self.arch_matrix[2, 1] + 1
        ret[4] = self.arch_matrix[3, 0] + 1
        ret[5] = self.arch_matrix[3, 1] + 1
        ret[6] = self.arch_matrix[3, 2] + 1
        ret[7] = 6
        return ret
    
    def init_wl_counter(self, H: int) -> None:
        self.wl_counter = natsbench_cell_to_wl_counter(self.label_list, H)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.arch_str}, {self.arch_matrix}, {self.accuracy_dict}, {self.flops}, {self.index}, {self.dataset})'
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\'{self.arch_str}\', {self.arch_matrix}, {self.accuracy_dict}, {self.flops}, {self.index}, \'{self.dataset}\')' 
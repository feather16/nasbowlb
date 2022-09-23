import os
from typing import Any
from importlib import import_module
import pickle
import numpy as np
from tqdm import tqdm

from Timer import Timer
from NATSBenchWrapper import NATSBenchWrapper
from CythonWLKernel import *

wrapper = NATSBenchWrapper()
wrapper.load_from_csv(f'{os.path.dirname(__file__)}/data/NATS-Bench.csv', 'ImageNet')

CACHE_DIR = 'data'

def save(path: str, H: int = 2) -> None:
    N = len(wrapper)
    data = np.empty((N, N))
    for i in tqdm(range(N)):
        for j in range(i, N):
            c1 = wrapper[i].label_list
            c2 = wrapper[j].label_list
            k = natsbench_wl_kernel(c1, c2, H)
            data[i, j] = data[j, i] = k
    pickle.dump(data, open(path, mode='wb'))

def load(path) -> Any:
    data = pickle.load(open(path, mode='rb'))
    return data

timer = Timer()
timer.start('save')
save(f'{CACHE_DIR}/NATS-Bench_WLKernel_H=2.pkl', 2)
timer.stop('save')

print(timer['save'], 'sec.')
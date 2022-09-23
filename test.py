import os
import pickle
import time
from tracemalloc import start

from NATSBenchWrapper import NATSBenchWrapper
from CythonWLKernel import *

wrapper = NATSBenchWrapper()
wrapper.load_from_csv(f'{os.path.dirname(__file__)}/data/NATS-Bench.csv', 'ImageNet')

true_kernel_values = pickle.load(open(f'{os.path.dirname(__file__)}/data/NATS-Bench_WLKernel_H=2.pkl', mode='rb'))

def wl_kernel(cell1: list[int], cell2: list[int]) -> float:
    return natsbench_wl_kernel(cell1, cell2)

def wl_kernel_from_wl_counters(counter1: dict[bytes, int], counter2: dict[bytes, int]) -> float:
    return natsbench_wl_kernel_from_wl_counters(counter1, counter2)

N = 1000
wl_counters = [natsbench_cell_to_wl_counter(wrapper[i].label_list) for i in range(15625)]
start_t = time.time()
for i in range(N):
    for j in range(i + 1, N):
        kernel_value = wl_kernel_from_wl_counters(wl_counters[i], wl_counters[j])
        true_kernel_value = true_kernel_values[i, j]
        if kernel_value != true_kernel_value:
            print(f'Error: i = {i}, j = {j} kernel_value = {kernel_value}, true_kernel_value = {true_kernel_value}')
            exit()
print('WL Kernel Test: No Errors!')
print('time:', time.time() - start_t)

N = 1000
start_t = time.time()
for i in range(N):
    for j in range(i + 1, N):
        cell1 = wrapper[i].label_list
        cell2 = wrapper[j].label_list
        kernel_value = wl_kernel(cell1, cell2)
        true_kernel_value = true_kernel_values[i, j]
        if kernel_value != true_kernel_value:
            print(f'Error: i = {i}, j = {j} kernel_value = {kernel_value}, true_kernel_value = {true_kernel_value}')
            exit()
print('WL Kernel Test: No Errors!')
print('time:', time.time() - start_t)

#start_t = time.time()
#N = 15625
#for i in range(N):
#    cell = wrapper[i].label_list
#    counter = natsbench_cell_to_wl_counter(cell)
#print(time.time() - start_t)
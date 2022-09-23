import sys
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from argparse import ArgumentParser

from NATSBenchWrapper import NATSBenchWrapper
from GPWithWLKernel import GPWithWLKernel
from Config import Config, IGNORED_KEYS

print('python version:', sys.version)

parser = ArgumentParser()
parser.add_argument('objective', type=str, choices=['acc', 'srcc', 'time'])
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('-T', type=int, required=True, help='loop count')
parser.add_argument('-P', type=int, default=16, help='pool size')
parser.add_argument('-B', type=int, default=2, help='batch size')
parser.add_argument('-D', type=int, default=100, help='initial size of data')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--strategy', type=str, default='random', choices=['random'])
parser.add_argument('--d_max', type=int)
parser.add_argument('--bagging_rate', type=float, default=8)
parser.add_argument('--acc_tops', type=int, default=5, help='for \'acc\'')
parser.add_argument('--name', type=str, help='name')
parser.add_argument('--id', type=int, help='id')
parser.add_argument('--srcc_eval_freq', type=int, default=20, help='evaluation frequency for \'srcc\'')
parser.add_argument('--srcc_eval_archs', type=int, default=100, help='evaluated architectures for \'srcc\'')
parser.add_argument('--load_kernel_cache', action='store_true')
parser.add_argument('--kernel_cache_path', type=str, default=f'{os.path.dirname(__file__)}/data/NATS-Bench_WLKernel_H=2.pkl')

args = parser.parse_args()
print('args:')
for k, v in vars(args).items():
    print(f'  {k}: {v}')

id = args.id if args.id != None else "unknownID"
objective: str = args.objective

wrapper = NATSBenchWrapper()
wrapper.load_from_csv(f'{os.path.dirname(__file__)}/data/NATS-Bench.csv', 'ImageNet')

# configからsearcherを生成
arg_dict = vars(args).copy()
for ignored_key in IGNORED_KEYS:
    del arg_dict[ignored_key]
config = Config(**arg_dict)
searcher = GPWithWLKernel(config)

# アーキテクチャの精度を計測
def acc_task():
    global searcher
    results: np.ndarray | None = None
    stat_results: list[np.ndarray] = [None] * config.trials
    print('timeTrial:')
    for trial in range(config.trials):
        start_t = time.time()
        result = searcher.accuracy_compare(wrapper)
        if results is None:
            results = np.zeros((len(result),))
        results += np.array(result)
        stat_results[trial] = np.array(result)
        print(f'  - {time.time() - start_t} # {trial + 1}')
    results /= config.trials

    print('result_mean:', np.mean(stat_results, 0).tolist())
    print('result_std:', np.std(stat_results, 0).tolist())
    print('result:', results.tolist())

# スピアマンの順位相関係数を計測
def srcc_task():
    global searcher
    results_srcc: np.ndarray = np.zeros((config.T // config.srcc_eval_freq,))
    results_acc: np.ndarray = np.zeros((config.T // config.srcc_eval_freq,))
    print('timeTrial:')
    for trial in range(config.trials):
        start_t = time.time()
        result = searcher.srcc_eval(wrapper)
        results_srcc += result['srcc']
        results_acc += result['acc']
        print(f'  - {time.time() - start_t} # {trial + 1}')
    results_srcc /= config.trials
    results_acc /= config.trials
    print('result:', {'srcc': results_srcc.tolist(), 'acc': results_acc.tolist()})

# 実行時間を計測
def time_task():
    global searcher
    result: dict[str, np.ndarray] = searcher.time_compare(wrapper)
    print('time:')
    for key, arr in result.items():
        print(f'  {key}: {arr[-1]}')
    print('result:')
    for key, arr in result.items():
        print(f'  {key}: {arr.tolist()}')
    
start_t = time.time()

if objective == 'acc':
    acc_task()
elif objective == 'srcc':
    srcc_task()
elif objective == 'time':
    time_task()

print(f'execution time: {time.time() - start_t}')
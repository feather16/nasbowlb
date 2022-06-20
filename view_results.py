from concurrent.futures import thread
import os
import yaml
from typing import Callable, Any, Optional
from matplotlib import pyplot as plt
#import threading
#import multiprocessing
import concurrent.futures
import time
import numpy as np
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

def _get_results_yaml_load(path: str) -> dict[Any, Any]:
    return yaml.safe_load(open(path))

def get_results(id_condition: Callable[[int], bool]) -> list[dict[Any, Any]]:
    results: list[dict[Any, Any]] = []
    paths: list[str] = []
    t = time.time()
    for path in os.listdir('result/log'):
        id = int(path[4:-5])
        if path.startswith('out_') and id_condition(id):
            paths.append(f'result/log/{path}')
            
    #for path in paths:
    #    result = yaml.safe_load(open(path))
    #    results.append(result)
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_get_results_yaml_load, paths))

    return sorted(results, key=lambda r: r['options']['id'])

def get_repeats(result: dict[Any, Any]) -> int:
    max_iters: int = result['options']['max_iters']
    repeats = 0
    while repeats in result['result'] and (max_iters - 1) in result['result'][repeats]:
        repeats += 1
    return repeats

def get_result_by_id(results: list[dict[Any, Any]], id: int) -> Optional[dict[Any, Any]]:
    for result in results:
        if result['options']['id'] == id:
            return result
    return None

def get_results_by_ids(results: list[dict[Any, Any]], ids: list[int]) -> list[dict[Any, Any]]:
    ret = []
    for result in results:
        if result['options']['id'] in ids:
            ret.append(result)
    return ret

def get_stat(result: dict[Any, Any], key: str, np_stat_func: Callable[[np.ndarray], np.ndarray]) -> float:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    if repeats == 0: return -1.
    sum_value_list: list[float] = []
    for r in range(repeats):
        sum_value_list.append(result['result'][r][max_iters - 1][key])
    return float(np_stat_func(sum_value_list))

def get_stats(result: dict[Any, Any], key: str, np_stat_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    sum_values_list: list[np.ndarray] = []
    for r in range(repeats):
        sum_values = np.array([result['result'][r][itr][key] for itr in range(max_iters)])
        sum_values_list.append(sum_values)
    return np_stat_func(sum_values_list, axis=0) if repeats > 0 else np.full((max_iters,), -1)

def get_average(result: dict[Any, Any], key: str) -> float:
    return get_stat(result, key, np.average)

def get_averages(result: dict[Any, Any], key: str) -> np.ndarray:
    return get_stats(result, key, np.average)

def get_stdev(result: dict[Any, Any], key: str) -> float:
    return get_stat(result, key, np.std)

def get_stdevs(result: dict[Any, Any], key: str) -> np.ndarray:
    return get_stats(result, key, np.std)

def print_results(results: list[dict[Any, Any]]) -> None:
    os.system('clear')
    dics: list[dict] = []
    for result in results:
        #if not (get_average(result, 'Best func test') <= 0.0597 and \
        #    result['options']['bagging'] != 'random_exclusive'): continue
        dic = {}
        dic['id'] = result['options']['id']
        dic['bagging'] = result['options']['bagging']
        if result['options']['bagging'] not in ['no', False]:
            dic['bagging_max_train_size'] = result['options']['bagging_max_train_size']
        dic['dataset'] = result['options']['dataset']
        if result['options']['comment'] not in ['', None]:
            dic['comment'] = result['options']['comment']
        dic['repeats'] = get_repeats(result)
        dic['max iters'] = result['options']['max_iters']
        dic['last loss'] = '%.6f ± %.6f' % (get_average(result, 'Last func test'), get_stdev(result, 'Last func test'))
        dic['best loss'] = '%.6f ± %.6f' % (get_average(result, 'Best func test'), get_stdev(result, 'Best func test'))
        dic['search time'] = '%.2f' % get_average(result, 'Time')
        dic['train time'] = '%.2f' % get_average(result, 'TrainTime')
        dic['total time'] = '%.2f' % (get_average(result, 'Time') + get_average(result, 'TrainTime'))
        dics.append(dic)
        
    key_maxlen = max(max(len(str(key)) for key in dic.keys()) for dic in dics)
    value_maxlen = max(max(len(str(value)) for value in dic.values()) for dic in dics)
        
    print('+' + '-' * (key_maxlen + 2) + '+' + '-' * (value_maxlen + 2) + '+')
    for dic in dics:
        for key, value in dic.items():
            key_spaces = (key_maxlen - len(str(key))) * ' '
            value_spaces = (value_maxlen - len(str(value))) * ' '
            print(f'| {key}{key_spaces} | {value}{value_spaces} |')
        print('+' + '-' * (key_maxlen + 2) + '+' + '-' * (value_maxlen + 2) + '+')
        
def plot_iters(
        results: list[dict[Any, Any]], 
        id_to_label: dict[int, str], 
        *,
        file_name: str = "tmp",
        title: str = "title",
        pdf: bool = False,
        x_label: str,
        y_label: str,
        keys,
        coefficient_y: float=1,
        ) -> None:
    plt.clf()
    for result in results:
        max_iters: int = result['options']['max_iters']
        if not isinstance(keys, list): keys = [keys]
        y: np.ndarray = sum(get_averages(result, key) for key in keys)
        
        id: int = result['options']['id']
        if y[0] == -1: continue
        if id in id_to_label:
            label: str = id_to_label[id]
            plt.plot(range(max_iters), y * coefficient_y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    format = 'pdf' if pdf else 'png'
    plt.savefig(f'tmp/{file_name}.{format}', format=format)
    plt.clf()     
        
id_condition: Callable[[int], bool] = lambda id: id in [839, 848]
results = get_results(id_condition)
print_results(results)

id_to_label_101 = {
    839: '既存手法',
    #840: 'ov 40',
    #841: 'ex 40',
    #842: 'ov 60',
    #843: 'ex 60',
    #844: 'ov 80',
    #845: 'ex 80',
    #846: 'ov 100',
    #847: 'ex 100',
    848: '提案手法',
    #849: 'ov 140',
    #850: 'ex 120',
    #853: 'ex 140',
    #854: 'ov 130',
    #855: 'ex 130',
}

plot_iters(
    results, 
    id_to_label_101, 
    file_name='n101_last_loss', 
    title='nasbench101 イテレーションごとの損失',
    x_label='イテレーション回数', 
    y_label='損失', 
    keys='Last func test',
    pdf=True,
)
plot_iters(
    results, 
    id_to_label_101, 
    file_name='n101_best_loss', 
    title='nasbench101 損失の最小値',
    x_label='イテレーション回数', 
    y_label='損失', 
    keys='Best func test',
    pdf=True,
)
plot_iters(
    results, 
    id_to_label_101, 
    file_name='n101_search_time', 
    title='nasbench101 探索時間（訓練を除く）',
    x_label='イテレーション回数', 
    y_label='時間（s）', 
    keys='Time',
    pdf=True,
)
plot_iters(
    results, 
    id_to_label_101, 
    file_name='n101_total_time', 
    title='nasbench101 実行時間（訓練を含む）',
    x_label='イテレーション回数', 
    y_label='時間（h）', 
    keys=['Time', 'TrainTime'],
    coefficient_y=1/3600,
    pdf=True,
)
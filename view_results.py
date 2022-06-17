import os
import yaml
from typing import Callable, Any, Optional
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

def get_results(id_condition: Callable[[int], bool]) -> list[dict[Any, Any]]:
    results: list[dict[Any, Any]] = []
    for path in os.listdir('result/log'):
        id = int(path[4:-5])
        if path.startswith('out_') and id_condition(id):
            results.append(yaml.safe_load(open(f'result/log/{path}')))
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

def get_average(result: dict[Any, Any], key: str) -> float:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    if repeats == 0: return -1.
    sum_value: float = 0.
    for r in range(repeats):
        sum_value += result['result'][r][max_iters - 1][key]
    return sum_value / repeats

def get_averages(result: dict[Any, Any], key: str) -> np.ndarray:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    sum_values: np.ndarray = np.zeros((max_iters,))
    for r in range(repeats):
        for itr in range(max_iters):
            sum_values[itr] += result['result'][r][itr][key]
    return sum_values / repeats if repeats > 0 else np.full((max_iters,), -1)

def get_average_last_loss(result: dict[Any, Any]) -> float:
    return get_average(result, 'Last func test')

def get_average_last_losses(result: dict[Any, Any]) -> np.ndarray:
    return get_averages(result, 'Last func test')

def get_average_last_time(result: dict[Any, Any]) -> float:
    return get_average(result, 'Time')

def get_average_last_times(result: dict[Any, Any]) -> np.ndarray:
    return get_averages(result, 'Time')

def print_results(results: list[dict[Any, Any]]) -> None:
    os.system('clear')
    dics: list[dict] = []
    for result in results:
        dic = {}
        dic['id'] = result['options']['id']
        dic['bagging'] = result['options']['bagging']
        if result['options']['bagging'] not in ['no', False]:
            dic['bagging_max_train_size'] = result['options']['bagging_max_train_size']
        dic['dataset'] = result['options']['dataset']
        if result['options']['comment'] not in ['', None]:
            dic['comment'] = result['options']['comment']
        dic['repeats'] = get_repeats(result)
        dic['max_iters'] = result['options']['max_iters']
        dic['last loss'] = '%.6f' % get_average_last_loss(result)
        dic['time'] = '%.2f' % get_average_last_time(result)
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
        
def plot_losses(
        results: list[dict[Any, Any]], 
        id_to_label: dict[int, str], 
        file_name: str = "plot1",
        title: str = "",
        pdf: bool = False,
        ) -> None:
    plt.clf()
    for result in results:
        max_iters: int = result['options']['max_iters']
        average_losses: np.ndarray = get_average_last_losses(result)
        id: int = result['options']['id']
        if average_losses[0] == -1: continue
        if id in id_to_label:
            label: str = id_to_label[id]
            plt.plot(range(max_iters), average_losses, label=label)
    plt.xlabel('イテレーション回数')
    plt.ylabel('損失')
    #plt.ylim(top=0.1)
    plt.legend()
    plt.title(title)
    format = 'pdf' if pdf else 'png'
    plt.savefig(f'tmp/{file_name}.{format}', format=format)
    plt.clf()
    
def plot_times(
        results: list[dict[Any, Any]], 
        id_to_label: dict[int, str], 
        file_name: str = "plot1",
        title: str = "",
        pdf: bool = False,
        ) -> None:
    plt.clf()
    for result in results:
        max_iters: int = result['options']['max_iters']
        average_times: np.ndarray = get_average_last_times(result)
        id: int = result['options']['id']
        if average_times[0] == -1: continue
        if id in id_to_label:
            label: str = id_to_label[id]
            plt.plot(range(max_iters), average_times, label=label)
    plt.xlabel('イテレーション回数')
    plt.ylabel('時間(s)')
    plt.legend()
    plt.title(title)
    format = 'pdf' if pdf else 'png'
    plt.savefig(f'tmp/{file_name}.{format}', format=format)
    plt.clf()

id_condition: Callable[[int], bool] = lambda id: id >= 839
results = get_results(id_condition)
print_results(results)

id_to_label_101 = {
    839: 'no',
    840: 'ov 40',
    841: 'ex 40',
    842: 'ov 60',
    843: 'ex 60',
    844: 'ov 80',
    845: 'ex 80',
    846: 'ov 100',
    847: 'ex 100',
}
'''
{
    771: 'no',
    772: 'ov 40',
    773: 'ex 40',
    774: 'ov 60',
    775: 'ex 60',
    776: 'ov 80',
    777: 'ex 80',
    778: 'ov 100',
    779: 'ex 100',
}
'''

plot_losses(results, id_to_label_101, 'nasbench101', 'nasbench101')
plot_times(results, id_to_label_101, 'nasbench101_time', 'nasbench101')
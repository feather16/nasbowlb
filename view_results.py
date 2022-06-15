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

def get_average_loss(result: dict[Any, Any]) -> float:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    if repeats == 0: return 1
    sum_loss: float = 0.
    for r in range(repeats):
        sum_loss += result['result'][r][max_iters - 1]['Last func test']
    return sum_loss / repeats

def get_average_losses(result: dict[Any, Any]) -> np.ndarray:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    sum_losses: np.ndarray = np.zeros((max_iters,))
    for r in range(repeats):
        for itr in range(max_iters):
            sum_losses[itr] += result['result'][r][itr]['Last func test']
    return sum_losses / repeats

def print_results(results: list[dict[Any, Any]]) -> None:
    os.system('clear')
    print('-' * 16)
    for result in results:
        print('id:', result['options']['id'])
        node_number = int(result['runningHost'][1:])
        print('gpu:', 'NVIDIA RTX A4500' if node_number <= 3 else 'GeForce GTX 1080')
        #print('cuda:', result['options']['cuda'])
        print('bagging:', result['options']['bagging'])
        print('dataset:', result['options']['dataset'])
        if result['options']['comment'] != "":
            print('comment:', result['options']['comment'])
        print('repeats:', get_repeats(result))
        print('max_iters:', result['options']['max_iters'])
        print('avg time:', get_average_loss(result))
        print('-' * 16)
        
def plot_losses(
        results: list[dict[Any, Any]], 
        id_to_label: dict[int, str] = {}, 
        file_name: str = "plot1",
        title: str = ""
        ) -> None:
    plt.clf()
    for result in results:
        max_iters: int = result['options']['max_iters']
        average_losses: np.ndarray = get_average_losses(result)
        id: int = result['options']['id']
        label: str = id_to_label[id] if id in id_to_label else str(id)
        plt.plot(range(max_iters), average_losses, label=label)
    plt.xlabel('イテレーション回数')
    plt.ylabel('損失')
    plt.legend()
    plt.title(title)
    plt.savefig(f'tmp/{file_name}.pdf', format='pdf')

id_condition: Callable[[int], bool] = lambda id: id >= 706
results = get_results(id_condition)
print_results(results)

plot_losses(get_results_by_ids(results, [712,731,732,733,734]), {
    712: '既存手法',
    731: '新提案手法(init)',
    732: '提案手法(init)',
    733: '新提案手法(fit)',
    734: '提案手法(fit)',
}, 'nasbench101', 'nasbench101')
import os
import yaml
from typing import Callable, Any

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

def get_average_time(result: dict[Any, Any]) -> float:
    max_iters: int = result['options']['max_iters']
    repeats: int = get_repeats(result)
    sum_time: float = 0.
    for r in range(repeats):
        sum_time += result['result'][r][max_iters - 1]['Last func test']
    return sum_time / repeats

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
        print('avg time:', get_average_time(result))
        #if 29 in result['result'][0]:
        #    print('time:', '{:.2f}'.format(result['result'][0][29]['Time']))
        #    print('Last func test:', '{:.5f}'.format(result['result'][0][29]['Last func test']))
        print('-' * 16)

id_condition: Callable[[int], bool] = lambda id: id not in [618]
results = get_results(id_condition)
print_results(results)
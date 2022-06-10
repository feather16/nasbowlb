import os
import yaml

target_logs = []

for path in os.listdir('result/log'):
    if path.startswith('out_') and 553 <= int(path[4:-5]):
        target_logs.append(f'result/log/{path}')
target_logs = sorted(target_logs)

for log in target_logs:
    dic = yaml.safe_load(open(log))
    print('id:', dic['options']['id'])
    node_number = int(dic['runningHost'][1:])
    print('node:', 'NVIDIA RTX A4500' if node_number <= 3 else 'GeForce GTX 1080')
    print('cuda:', dic['options']['cuda'])
    print('bagging:', dic['options']['bagging'])
    print('dataset:', dic['options']['dataset'])
    if dic['options']['comment'] != "":
        print('comment:', dic['options']['comment'])
    if 29 in dic['result'][0]:
        print('time:', dic['result'][0][29]['Time'])
    print('\n' + '-' * 16)
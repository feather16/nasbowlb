import os

from Log import *

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'
IMAGE_DIR = f'{os.path.dirname(__file__)}/result/image'

logs = LogSet(LOG_DIR, IMAGE_DIR)
all_ids = {id for id in range(3191, 5000) if logs.exists(id)}
logs.load_logs(all_ids)

for id in sorted(all_ids):
    print(f'{id}: ({logs[id].execution_time: >4.0f} s) {logs[id].command}')
import os

from Log import *

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'
IMAGE_DIR = f'{os.path.dirname(__file__)}/result/image'

logs = LogSet(LOG_DIR, IMAGE_DIR)

name_maps = [
    {
        3191: 'n',
        3192: 'n',
        3193: 'b800',
        3194: 'b800',
        3195: 'n',
        3196: 'b800',
        3197: 'n',
        3198: 'b800',
    },
    {
        3199: 'n',
        3200: 'b800',
        3201: 'n',
        3202: 'b800',
        3203: 'n',
        3204: 'b800',
    },
]

name_map_set: dict[int, str] = {}
for name_map in name_maps:
    name_map_set |= name_map
all_ids = name_map_set.keys()
logs.load_logs(all_ids)

for name_map in name_maps:
    ids = sorted(name_map.keys())
    logs.plot(ids, name_map, acc_bottom=44, acc_top=46.5)

time_ids = [key for key in name_map_set.keys() if logs[key].objective == 'time']
for time_id in time_ids:
    logs.plot_time_details(time_id, name_map_set)
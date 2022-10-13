import os

from Log import *

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'
IMAGE_DIR = f'{os.path.dirname(__file__)}/result/image'

logs = LogSet(LOG_DIR, IMAGE_DIR)

name_maps = [
    {
        4188: 'n',
        4189: 'b800',
        4190: 'n',
        4191: 'b800',
        4192: 'n',
        4193: 'b800',
    },
    {
        4194: 'n',
        4195: 'b800',
        4196: 'n',
        4197: 'b800',
        4198: 'n',
        4199: 'b800',
    },
]

name_map_set: dict[int, str] = {}
for name_map in name_maps:
    name_map_set |= name_map
all_ids = name_map_set.keys()
logs.load_logs(all_ids)

for name_map in name_maps:
    ids = sorted(name_map.keys())
    logs.plot(ids, name_map, acc_bottom=45.5, acc_top=46.8)

time_ids = [key for key in name_map_set.keys() if logs[key].objective == 'time']
for time_id in time_ids:
    logs.plot_time_details(time_id, name_map_set)
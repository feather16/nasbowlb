import os

from Log import *

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'
IMAGE_DIR = f'{os.path.dirname(__file__)}/result/image'

logs = LogSet(LOG_DIR, IMAGE_DIR)

name_maps = [
    {
        4200: 'n',
        4201: 'b800',
        4202: 'n',
        4203: 'b800',
        4204: 'n',
        4205: 'b800',
    },
    {
        4206: 'n',
        4207: 'b800',
        4208: 'n',
        4209: 'b800',
        4210: 'n',
        4211: 'b800',
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
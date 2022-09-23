import os
import re
from argparse import ArgumentParser
import yaml

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'

parser = ArgumentParser()
parser.add_argument('--noid', action='store_true', help='not print jobID')
args = parser.parse_args()

def get_commands() -> list[tuple[int, str]]:
    ret = []
    files = sorted(os.listdir(LOG_DIR))
    for file in files:
        if re.fullmatch(r'out_\d+.yaml', file):
            cmd = 'Unknown'
            with open(f'{LOG_DIR}/{file}') as f:
                cmd = yaml.safe_load(f)['command']
            jobId = int(file[len('out_'):-len('.yaml')])
            ret.append((jobId, cmd))
    return ret
    
def write_commands_to_csv(path: str = 'tmp.csv'):
    with open(path, 'w') as f:
        for jobid, cmd in get_commands():
            f.write(f'{jobid},{cmd}\n')

def print_commands(print_id: bool = True):
    for jobid, cmd in get_commands():
        if print_id:
            print(f'{jobid}: {cmd}')
        else:
            print(f'{cmd}')

print_commands(print_id=not args.noid)
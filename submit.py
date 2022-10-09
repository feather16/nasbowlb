from itertools import starmap
from json import load
import time
import subprocess

class Job:
    def __init__(self, 
            jobid: int = -1, partition: str = '', name: str = '', user: str = '', 
            status: str = '', time: str = '', nodes: int = -1, nodelist: str = '') -> None:
        self.jobid: int = jobid
        self.partition: str = partition
        self.name: str = name
        self.user: str = user
        self.status: str = status
        self.time: str = time
        self.nodes: int = nodes
        self.nodelist: str = nodelist
    
    @property
    def node_ids(self) -> list[int]:
        deleted_words = ['c', '[', ']']
        text = self.nodelist
        for deleted_word in deleted_words:
            text = text.replace(deleted_word, '')
        ids: set[int] = set()
        for token in text.split(','):
            if '-' in token:
                r = token.split('-')
                ids.update(range(int(r[0]), int(r[1]) + 1))
            else:
                ids.add(int(token))
        return sorted(list(ids))
    
    def __str__(self) -> str:
        return str(
            {
                'jobid': self.jobid,
                'partition': self.partition,
                'name': self.name,
                'user': self.user,
                'status': self.status,
                'time': self.time,
                'nodes': self.nodes,
                'nodelist': self.nodelist,
            }
        )
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + \
            f'{self.jobid}, ' + \
            f'"{self.partition}", ' + \
            f'"{self.name}", ' + \
            f'"{self.user}", ' + \
            f'"{self.status}", ' + \
            f'"{self.time}", ' + \
            f'{self.nodes}, ' + \
            f'"{self.nodelist}")'

def find_first_back(target: str, word: str) -> int:
    pos = target.find(word)
    if pos != -1:
        return pos + len(word) - 1
    else:
        return -1
    
def delete_spaces(text: str) -> str:
    while text[-1] == ' ': text = text[:-1]
    while text[0] == ' ': text = text[1:]
    return text

def load_squeue(squeue: str) -> list[Job]:
    lines = list(filter(lambda l: len(l) > 0, squeue.split('\n')))
    jobs = [Job() for line in lines[1:]]
    jobid_pos = find_first_back(lines[0], 'JOBID')
    partition_pos = find_first_back(lines[0], 'PARTITION')
    name_pos = find_first_back(lines[0], 'NAME')
    user_pos = find_first_back(lines[0], 'USER')
    status_pos = find_first_back(lines[0], 'ST')
    time_pos = find_first_back(lines[0], 'TIME')
    nodes_pos = find_first_back(lines[0], 'NODES')
    for i, line in enumerate(lines[1:]):
        jobs[i].jobid = int(delete_spaces(line[: jobid_pos + 1]))
        jobs[i].partition = delete_spaces(line[jobid_pos + 1: partition_pos + 1])
        jobs[i].name = delete_spaces(line[partition_pos + 1: name_pos + 1])
        jobs[i].user = delete_spaces(line[name_pos + 1: user_pos + 1])
        jobs[i].status = delete_spaces(line[user_pos + 1: status_pos + 1])
        jobs[i].time = delete_spaces(line[status_pos + 1: time_pos + 1])
        jobs[i].nodes = int(delete_spaces(line[time_pos + 1: nodes_pos + 1]))
        jobs[i].nodelist = delete_spaces(line[nodes_pos + 1:])
    return jobs

def get_command(
        objective: str,
        *,
        trials: int | None = None,
        T: int,
        P: int | None = None,
        B: int | None = None,
        D: int | None = None,
        device: str | None = None,
        strategy: str | None = None,
        d_max: int | None = None,
        bagging_rate: float | None = None,
        acc_tops: int | None = None,
        name: str | None = None,
        id: int | None = None,
        srcc_eval_freq: int | None = None,
        srcc_eval_archs: int | None = None,
        load_kernel_cache: bool | None = None,
        kernel_cache_path: str | None = None,
        ) -> str:
    
    assert objective in ['srcc', 'acc', 'time']
    assert trials is None or 1 <= trials
    assert 1 <= T
    assert B is None or 1 <= B
    assert P is None or 2 <= P
    if B is not None and P is not None:
        assert B < P
    assert D is None or 1 <= D
    assert device is None or device == 'cpu' or 'cuda' in device
    assert strategy is None or strategy in ['random']
    assert d_max is None or 1 <= d_max
    assert bagging_rate is None or 0 < bagging_rate
    assert acc_tops is None or 1 <= acc_tops
    assert id is None or 0 <= id
    assert srcc_eval_freq is None or 1 <= srcc_eval_freq
    assert srcc_eval_archs is None or 1 <= srcc_eval_archs

    keys = [
        'trials',
        'T',
        'P',
        'B',
        'D',
        'device',
        'strategy',
        'd_max',
        'bagging_rate',
        'acc_tops',
        'name',
        'id',
        'srcc_eval_freq',
        'srcc_eval_archs',
        'load_kernel_cache',
        'kernel_cache_path',
    ]

    command_tokens = [
        'sbatch',
        'run',
        'nasbowl.py',
        objective
    ]
    for key in keys:
        value = eval(key)
        if value is not None:
            hyphens = '--' if len(key) >= 2 else '-'
            command_tokens.append(f'{hyphens}{key}')
            if not isinstance(value, bool):
                command_tokens.append(str(value))

    command = ' '.join(command_tokens)
    
    return command

def is_cuda(command: str) -> bool:
    return '--device cuda' in command
    
'''

'''

COMMANDS = [
    get_command('srcc', trials=10, T=750),
    get_command('srcc', trials=10, T=750, d_max=800, bagging_rate=8),
    get_command('acc', trials=10, T=750),
    get_command('acc', trials=10, T=750, d_max=800, bagging_rate=8),
    get_command('time', T=1500),
    get_command('time', T=1500, d_max=800, bagging_rate=8),
    get_command('srcc', trials=10, T=750, load_kernel_cache=True),
    get_command('srcc', trials=10, T=750, d_max=800, bagging_rate=8, load_kernel_cache=True),
    get_command('acc', trials=10, T=750, load_kernel_cache=True),
    get_command('acc', trials=10, T=750, d_max=800, bagging_rate=8, load_kernel_cache=True),
    get_command('time', T=1500, load_kernel_cache=True),
    get_command('time', T=1500, d_max=800, bagging_rate=8, load_kernel_cache=True),
]

MAX_NODE_USES = 12
CUDA_NODES = set(range(4, 17))

remaining_commands = COMMANDS.copy()
while True:
    
    # 実行中のジョブを調べる
    jobs = load_squeue(subprocess.check_output('squeue').decode())
    current_node_used: set[int] = set()
    for job in jobs:
        current_node_used |= set(job.node_ids)
    num_current_node_used = sum(job.nodes for job in jobs)
    
    if num_current_node_used < MAX_NODE_USES:
        num_submit = min(MAX_NODE_USES - num_current_node_used, len(remaining_commands))
        for i in range(num_submit):
            if is_cuda(remaining_commands[0]): # GPUで実行する場合
                available_nodes = sorted(list(CUDA_NODES - current_node_used))
                if len(available_nodes) > 0: # 使用可能なノードがある場合
                    node_id = available_nodes[0]
                    current_node_used.add(node_id)
                    command = remaining_commands.pop(0)
                    command = command.replace('sbatch ', f'sbatch --nodelist=c{node_id} ')
                    subprocess.call(command.split())
                else: # 使用可能なノードがない場合
                    break
            else: # CPUで実行する場合
                command = remaining_commands.pop(0)
                subprocess.call(command.split())
        print(f'len(remaining_commands) = {len(remaining_commands)}')
        if len(remaining_commands) == 0:
            break

    time.sleep(5)
import os
from typing import Any, Iterable
import yaml
import concurrent.futures
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

from Config import Config, IGNORED_KEYS

class LogSet:
    def __init__(self, log_dir: str, image_dir: str):
        self.logs: dict[int, 'Log'] = {}
        self.log_dir = log_dir
        self.image_dir = image_dir

    def plot(
            self, 
            ids: Iterable[int], 
            name_map: dict[int, str] = {}, 
            *, 
            format: str = 'png',
            acc_bottom: float | None = None,
            acc_top: float | None = None,
            plot_title: bool = True,
            ) -> None:

        full_id_set = {id for id in ids if self.exists(id)}
        cond_set: set[int] = {self[id].config.T for id in full_id_set}
        
        for cond in cond_set:
            id_set = sorted({id for id in full_id_set if self[id].config.T == cond})
        
            id_to_label: dict[int, str] = {}
            srcc_ids: list[int] = []
            acc_ids: list[int] = []
            time_ids: list[int] = []
            for id in id_set:
                if self.exists(id):
                    id_to_label[id] = name_map[id] if id in name_map else str(id)
                    objective = self[id].objective
                    if objective == 'srcc': srcc_ids.append(id)
                    if objective == 'acc': acc_ids.append(id)
                    if objective == 'time': time_ids.append(id)

            # srcc
            if len(srcc_ids) > 0:
                label_to_ids = self.generate_label_to_ids(srcc_ids, id_to_label)
                max_id = max(srcc_ids)
                if plot_title: plt.title(f'id ≤ {max_id}')
                plt.xlabel('訓練アーキテクチャ数')
                plt.ylabel('Spearman\'s rank correlation coefficient')
                for label in label_to_ids:
                    id_set = label_to_ids[label]
                    assert len({self[id].config.srcc_eval_freq for id in id_set}) == 1
                    assert len({self[id].config.B for id in id_set}) == 1
                    assert len({self[id].config.T for id in id_set}) == 1
                    config = self[id_set[0]].config
                    freq = config.srcc_eval_freq
                    B = config.B
                    T = config.T
                    srcc = self.concat(id_set)
                    x = range(freq * B, (len(srcc) + 1) * freq * B, freq * B)
                    y = srcc
                    plt.plot(x, y, label=label)
                plt.legend()
                plt.savefig(f'{self.image_dir}/srcc_id<={max_id}.{format}')
                plt.clf()

            # acc
            if len(acc_ids) > 0:
                label_to_ids = self.generate_label_to_ids(acc_ids, id_to_label)
                max_id = max(acc_ids)
                if plot_title: plt.title(f'id ≤ {max_id}')
                plt.xlabel('訓練アーキテクチャ数')
                plt.ylabel('性能(%)')
                for label in label_to_ids:
                    id_set = label_to_ids[label]
                    assert len({self[id].config.B for id in id_set}) == 1
                    assert len({self[id].config.T for id in id_set}) == 1
                    config = self[id_set[0]].config
                    B = config.B
                    T = config.T
                    x = range(B, (T + 1) * B, B)
                    y = self.concat(id_set)
                    plt.plot(x, y, label=label)
                plt.legend()
                if acc_bottom is not None:
                    plt.ylim(bottom=acc_bottom)
                if acc_top is not None:
                    plt.ylim(top=acc_top)
                plt.savefig(f'{self.image_dir}/acc_id<={max_id}.{format}')
                plt.clf()

            # time
            if len(time_ids) > 0:
                label_to_ids = self.generate_label_to_ids(time_ids, id_to_label)
                max_id = max(time_ids)
                if plot_title: plt.title(f'id ≤ {max_id}')
                plt.xlabel('訓練アーキテクチャ数')
                plt.ylabel('時間(秒)')
                for label in label_to_ids:
                    id_set = label_to_ids[label]
                    assert len({self[id].config.B for id in id_set}) == 1
                    assert len({self[id].config.T for id in id_set}) == 1
                    config = self[id].config
                    B = config.B
                    T = config.T
                    x = range(B, (T + 1) * B, B)
                    y = self.concat(id_set)
                    plt.plot(x, y, label=label)
                plt.legend()
                plt.savefig(f'{self.image_dir}/time_id<={max_id}.{format}')
                plt.clf()

    def plot_time_details(
            self, 
            id: int,
            name_map: dict[int, str] = {},
            *, 
            format: str = 'png',
            show_id: bool = True,
            plot_title: bool = True,
            ) -> None:
        assert self[id].objective == 'time'
        if id in name_map:
            name = title = name_map[id]
            if show_id:
                name += f'_id={id}'
                title += f' (id = {id})'
        else:
            name = f'id={id}'
            title = f'id = {id}'
        if plot_title: plt.title(title)
        plt.xlabel('訓練アーキテクチャ数')
        plt.ylabel('時間(秒)')
        config = self[id].config
        B = config.B
        T = config.T
        x = range(B, (T + 1) * B, B)
        for key, value in self[id].time.items():
            y = value
            plt.plot(x, y, label=key)
        plt.legend()
        plt.savefig(f'{self.image_dir}/time_{name}.{format}')
        plt.clf()

    def __getitem__(self, id: int) -> 'Log':
        if id not in self.logs:
            log_path = self.get_log_path(id)
            self.logs[id] = Log(yaml.safe_load(open(log_path)))
        return self.logs[id]

    def exists(self, id: int) -> bool:
        ''' ログファイルが存在するか '''
        log_path = self.get_log_path(id)
        return os.path.exists(log_path)

    def get_log_path(self, id: int) -> str:
        ''' ログファイルのパス '''
        return f'{self.log_dir}/out_{id}.yaml'

    def load_logs(self, ids: Iterable[int]) -> None:
        ''' ログファイルの内容をキャッシュに格納 '''
        exist_ids: list[int] = [id for id in ids if self.exists(id)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.__getitem__, exist_ids))

    def generate_label_to_ids(self, ids: int, id_to_label: dict[int, str]) -> dict[str, list[int]]:
        label_to_ids: dict[str, list[int]] = {}
        for id in ids:
            label = id_to_label[id]
            if label not in label_to_ids:
                label_to_ids[label] = []
            label_to_ids[label].append(id)
        return label_to_ids
        
    def concat(self, ids: list[int]) -> list[float]:
        ''' 2つ以上の結果を合成 '''
        objectives = {self[id].objective for id in ids}
        assert len(objectives) == 1
        objective = objectives.pop()
        assert objective in ['srcc', 'acc', 'time']
        trials = sum(self[id].trials for id in ids)
        if objective == 'srcc':
            id_to_values = lambda id: np.array(self[id].srcc)
        elif objective == 'acc':
            id_to_values = lambda id: np.array(self[id].acc)
        else:
            id_to_values = lambda id: np.array(self[id].total_time)
        return (sum(np.array(id_to_values(id) * self[id].trials) for id in ids) / trials).tolist()

class Log(dict):
    @property
    def command(self) -> str:
        return self['command']

    @property
    def config(self) -> Config:
        args: dict[str, Any] = self['args'].copy()
        for ignored_key in IGNORED_KEYS:
            del args[ignored_key]
        return Config(**args)
    
    @property
    def trials(self) -> int:
        return self.config.trials

    @property
    def srcc(self) -> list[float]:
        assert self.objective == 'srcc'
        return self['result']['srcc']

    @property
    def final_srcc(self) -> float:
        assert self.objective == 'srcc'
        return self.srcc[-1]

    @property
    def acc(self) -> list[float]:
        assert self.objective == 'acc'
        return self['result']

    @property
    def time(self) -> dict[str, list[float]]:
        assert self.objective == 'time'
        return self['result']

    @property
    def total_time(self) -> list[float]:
        assert self.objective == 'time'
        return self.time['Total']
    
    @property
    def final_acc(self) -> dict[str, float]:
        assert self.objective == 'acc'
        return {k: v[-1] for k, v in self.acc.items()}

    @property
    def objective(self) -> str:
        return self['args']['objective'] if 'args' in self else ''

    @property
    def execution_time(self) -> float:
        return self['execution time']
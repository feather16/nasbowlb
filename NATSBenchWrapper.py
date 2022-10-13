import csv
import time
from tqdm import tqdm

from nats_bench import create
from nats_bench.api_topology import NATStopology
from nats_bench.api_utils import ArchResults
from NATSBenchCell import NATSBenchCell

class NATSBenchWrapper:
    def __init__(self):
        self.archs: list[NATSBenchCell] = []
            
    # アーカイブファイルからアーキテクチャの精度などを読み込む(低速)
    def load_from_archive(self, data_path: str) -> None:
        nats_bench: NATStopology = create(data_path, search_space='topology', fast_mode=True, verbose=False)
        self.num_archs: int = len(nats_bench)
        
        for i in tqdm(range(self.num_archs)):
            arch_results: ArchResults = nats_bench.query_by_index(i, hp='200')
            arch_str: str = arch_results.arch_str
            accuracy_dict = {}
            flops_dict = {} # 今は使っていない
            for dataset_key, dataset_name in [('cifar10-valid', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('ImageNet16-120', 'ImageNet')]:
                # 精度(%)
                more_info = nats_bench.get_more_info(i, dataset_key, hp='200', is_random=False)
                accuracy: float = more_info['valid-accuracy']
                flops: float = arch_results.get_compute_costs(dataset_key)['flops']
                accuracy_dict[dataset_name] = accuracy
                flops_dict[dataset_name] = flops
            arch = NATSBenchCell(arch_str, accuracy_dict, flops_dict, i, dataset_name)
            self.archs.append(arch)
            
    # csvファイルからアーキテクチャの精度などを読み込む(高速)
    def load_from_csv(self, csv_path: str, dataset: str) -> None:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            i = 0
            for dic in reader:
                dataset_keys = ['cifar10', 'cifar100', 'ImageNet']
                arch_str = dic['arch_str']
                accuracy, flops = {}, {}
                for dataset in dataset_keys:
                    accuracy[dataset] = float(dic[f'acc-{dataset}'])
                    flops[dataset] = float(dic[f'flops-{dataset}'])
                arch = NATSBenchCell(arch_str, accuracy, flops, i, dataset)
                self.archs.append(arch)
                i += 1
        self.num_archs: int = len(self.archs)

    # アーキテクチャの精度をcsvファイルに保存
    def save_to_csv(self, csv_path: str) -> None:
        with open(csv_path, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'arch_str', 
                'acc-cifar10', 'acc-cifar100', 'acc-ImageNet', 
                'flops-cifar10', 'flops-cifar100', 'flops-ImageNet'])
            for i, arch in enumerate(self.archs):
                writer.writerow([
                    arch.arch_str, 
                    arch.accuracy_dict['CIFAR-10'], 
                    arch.accuracy_dict['CIFAR-100'],
                    arch.accuracy_dict['ImageNet'],
                    arch.flops['CIFAR-10'], 
                    arch.flops['CIFAR-100'],
                    arch.flops['ImageNet'],
                ])
     
    def init_wl_counters(self, H: int) -> None:
        for i in range(len(self)):
            self[i].init_wl_counter(H)

    def __getitem__(self, key):
        return self.archs[key]
    
    def __len__(self) -> int:
        return self.num_archs
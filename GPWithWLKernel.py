import time
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Any
import random
import statistics
import copy
import math
import sys

from Config import Config
from NATSBenchCell import NATSBenchCell
from NATSBenchWrapper import NATSBenchWrapper
from CachedKernel import CachedKernel
from Timer import Timer
from util import spearman_rcc
from CythonWLKernel import *

class GPWithWLKernel:
    def __init__(
            self, 
            config: Config,
            wrapper: NATSBenchWrapper,
            ):
        self.config: Config = copy.copy(config)
        self.timer: Timer = Timer()
        
        self.wl_kernel: CachedKernel = CachedKernel(natsbench_wl_kernel_from_wl_counters)
        
        # キャッシュ
        if self.config.use_kernel_cache:
            assert self.config.kernel_cache_path is not None
            self.wl_kernel.load_pickle(self.config.kernel_cache_path, wrapper, config.verbose)
        self.K_cache: torch.Tensor | None = None
        self.K_inv_cache: torch.Tensor | None = None

    def random_sampler(
            self,
            sample_cells: list[NATSBenchCell],
            data: list[NATSBenchCell], 
            ) -> list[NATSBenchCell]:
        return random.sample(sample_cells, self.config.B)
    
    def acquisition_gp_with_wl_kernel(
            self,
            x_list: list[NATSBenchCell], 
            data: list[NATSBenchCell], 
            k: torch.Tensor,
            K_inv: torch.Tensor, # K^-1
            K_inv_y: torch.Tensor, # K^-1 * y
            mean_acc: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        ガウス過程回帰により平均と分散を推定
        
        `mu = k.T * K^-1 * y`
        `sigma^2 = kernel(x, x) - k.T * K_inv * k`
        '''
        
        t = len(data)
        n = len(x_list)
        
        # kernel(x, x)
        self.timer.start('WLKernel')
        xx_kernel: torch.Tensor = torch.arange(0, n, dtype=torch.float32)
        xx_kernel.apply_(lambda i: self.wl_kernel(x_list[int(i)].index, x_list[int(i)].index, x_list[int(i)].wl_counter, x_list[int(i)].wl_counter))
        xx_kernel = xx_kernel.to(k.device)
        self.timer.stop('WLKernel')
        
        # 行列演算
        self.timer.start('MatrixMult')
        k_T: torch.Tensor = k.transpose(1, 2)
        mu: torch.Tensor = mean_acc + torch.matmul(k_T, K_inv_y).reshape((n,))
        k_K_inv: torch.Tensor = torch.matmul(k_T, K_inv)
        var: torch.Tensor = xx_kernel - torch.matmul(k_K_inv, k).reshape((n,))
        self.timer.stop('MatrixMult')

        var = nn.ReLU()(var)
        return mu, torch.sqrt(var)
    
    def compose_k_vectors(
            self,
            x_list: list[NATSBenchCell], 
            data: list[NATSBenchCell], 
            device: torch.device,
            ) -> torch.Tensor:
        '''
        ガウス過程回帰で用いるベクトル `k` を構成
        '''
        t = len(data)
        n = len(x_list)
        self.timer.start('WLKernel')
        k: torch.Tensor
        if self.wl_kernel.cached:
            assert self.wl_kernel.cache is not None
            kernel_values: np.ndarray = self.wl_kernel.cache[np.ix_([x.index for x in x_list], [data[i].index for i in range(t)])]
            k = torch.from_numpy(kernel_values).to(device).float()
            k = k.reshape((n, t, 1))
        else:
            k = torch.arange(0, n * t, dtype=torch.float32).reshape((n, t, 1))
            k.apply_(lambda i: self.wl_kernel(x_list[int(i / t)].index, data[int(i % t)].index, x_list[int(i / t)].wl_counter, data[int(i % t)].wl_counter))
            k = k.to(device)
        self.timer.stop('WLKernel')
        return k

    def compose_K(
            self,
            data: list[NATSBenchCell],
            t: int,
            ) -> torch.Tensor:
        '''
        ガウス過程回帰で用いる行列 `K` を構成
        '''
        B = self.config.B
        cached = False
        K: torch.Tensor
        if self.K_cache is not None and self.K_cache.shape[0] == t - B:
            K = self.K_cache
            L = torch.empty((t - B, B), device=self.config.device)
            LM = torch.empty((B, t), device=self.config.device)
            K = torch.concat([K, L], axis=1)
            K = torch.concat([K, LM], axis=0)
            cached = True
        else:
            K = torch.empty((t, t), device=self.config.device)

        # ここのカーネル計算が時間的にネック
        self.timer.start('WLKernel')
        if self.wl_kernel.cached:
            assert self.wl_kernel.cache is not None
            if cached:
                i_indices = [data[i].index for i in range(t)]
                j_indices = [data[i].index for i in range(t - B, t)]
                K[:, t - B: t] = \
                    torch.from_numpy(self.wl_kernel.cache[np.ix_(i_indices, j_indices)]).to(K.device).float()
                i_indices = [data[i].index for i in range(t - B, t)]
                j_indices = [data[i].index for i in range(t - B)]
                K[t - B: t, :t - B] = \
                    torch.from_numpy(self.wl_kernel.cache[np.ix_(i_indices, j_indices)]).to(K.device).float()
            else:
                i_indices = [data[i].index for i in range(t)]
                j_indices = [data[i].index for i in range(t)]
                K = torch.from_numpy(self.wl_kernel.cache[np.ix_(i_indices, j_indices)]).to(K.device).float()
        else:
            if cached:
                for b in range(B):
                    c = data[t - (b + 1)]
                    tensor = torch.arange(0, t, dtype=torch.float32)
                    tensor.apply_(lambda i: self.wl_kernel(c.index, data[int(i)].index, c.wl_counter, data[int(i)].wl_counter))
                    K[:, t - (b + 1)] = tensor.to(K.device)
                    tensor = torch.arange(0, t - B, dtype=torch.float32)
                    tensor.apply_(lambda i: self.wl_kernel(c.index, data[int(i)].index, c.wl_counter, data[int(i)].wl_counter))
                    K[t - (b + 1), :t - B] = tensor.to(K.device)
            else:
                for j in range(t):
                    c = data[j]
                    tensor = torch.arange(0, t, dtype=torch.float32)
                    tensor.apply_(lambda i: self.wl_kernel(c.index, data[int(i)].index, c.wl_counter, data[int(i)].wl_counter))
                    K[j] = tensor.to(K.device)
        self.timer.stop('WLKernel')
        self.K_cache = K.clone()

        return K

    def compose_K_inv(
            self,
            K: torch.Tensor, 
            t: int, 
            is_dropped: bool
            ) -> torch.Tensor:
        '''
        ガウス過程回帰で用いる行列 `K^-1` を構成
        '''
        K_inv: torch.Tensor
        cached = False
        self.timer.start('MatrixInv')
        
        if not is_dropped and self.K_inv_cache is not None and self.K_inv_cache.shape[0] == t:
            K_inv = self.K_inv_cache
            cached = True
        if not cached:
            try:
                K_inv = torch.linalg.inv(K)
            except:
                print(f'# pinv: t = {t}', file=sys.stderr)
                K_inv = torch.linalg.pinv(K)
        self.K_inv_cache = K_inv
        self.timer.stop('MatrixInv')
        return K_inv

    def gp_with_wl_kernel_sampler(
            self,
            sample_cells: list[NATSBenchCell],
            data: list[NATSBenchCell],
            ) -> list[NATSBenchCell]:
        '''
        ガウス過程回帰に基づき、
        探索空間から`self.config.B`個のアーキテクチャをサンプリングする
        '''
        itr = (len(data) - self.config.D) // self.config.B # イテレーション回数
        gamma = 3 * math.sqrt(1/2 * math.log(2 * (itr + 1)))
        
        musigma_tuples = self.gp_with_wl_kernel(sample_cells, data)
        index_musigma_tuples = list(zip(sample_cells, musigma_tuples))
        index_musigma_tuples = sorted(index_musigma_tuples, key=lambda x: x[1][0] + gamma * x[1][1], reverse=True)[:self.config.B]
        ret = [t[0] for t in index_musigma_tuples]
        return ret

    def gp_with_wl_kernel(
            self,
            sample_cells: list[NATSBenchCell],
            data: list[NATSBenchCell], 
            ) -> list[tuple[float, float]]:
        '''
        ガウス過程回帰により、
        未知のアーキテクチャの性能の平均と標準偏差を推定
        '''
        
        t = len(data) # Kのサイズ
        B = self.config.B
        
        d_max: int = self.config.d_max
        
        musigma_tuples_list: list[list[tuple[float, float]]] = []
        
        n_samples: int
        if t > d_max and self.config.strategy == 'random':
            n_samples = math.ceil(self.config.bagging_rate * (t / d_max - 1)) + 1
        else:
            n_samples = 1
        
        K_base: torch.Tensor = self.compose_K(data, t)
        y_base: torch.Tensor = torch.tensor([data[i].accuracy for i in range(t)], device=K_base.device)
        mean_acc_tensor: torch.Tensor = torch.mean(y_base)
        y_base -= mean_acc_tensor
        mean_acc: float = float(mean_acc_tensor)
        
        for n in range(n_samples):
            # Kの構成とキャッシュ化
            K = K_base # ファンシーインデックスはコピーが作成されるので、ビューの代入でOK
            y = y_base # ファンシーインデックスはコピーが作成されるので、ビューの代入でOK
            sub_data: list[NATSBenchCell] = copy.copy(data)
            
            # バギング
            if t > d_max and self.config.strategy == 'random':
                self.timer.start('Bagging')
                sorted_remaining_indices: np.ndarray = np.sort(np.random.choice(range(t), d_max, replace=False))
                K = K[np.ix_(sorted_remaining_indices, sorted_remaining_indices)]
                y = y[sorted_remaining_indices]
                sub_data = [sub_data[i] for i in sorted_remaining_indices]
                self.timer.stop('Bagging')
                    
            # 逆行列
            K_inv: torch.Tensor = self.compose_K_inv(K, t, t >= d_max)
                                
            # 行列演算
            self.timer.start('MatrixMult')
            K_inv_y: torch.Tensor = K_inv @ y
            self.timer.stop('MatrixMult')

            k_vectors = self.compose_k_vectors(sample_cells, sub_data, K_inv.device)
            mus, sigmas = self.acquisition_gp_with_wl_kernel(
                sample_cells, 
                sub_data, 
                k_vectors, 
                K_inv, 
                K_inv_y, 
                mean_acc
            )
            musigma_tuples = [(float(mu), float(sigma)) for mu, sigma in zip(mus, sigmas)]
            musigma_tuples_list.append(musigma_tuples)
        
        ret: list[tuple[float, float]] = []
        for i in range(len(sample_cells)):
            mu = statistics.median([musigma_tuples_list[j][i][0] for j in range(n_samples)])
            sigma = statistics.median([musigma_tuples_list[j][i][1] for j in range(n_samples)])
            ret.append((mu, sigma))

        return ret

    def search(
            self,
            sampler: Callable[[list[NATSBenchCell], list[NATSBenchCell]], list[NATSBenchCell]],
            wrapper: NATSBenchWrapper, 
            data: list[NATSBenchCell], 
            search_space: list[NATSBenchCell],
            ) -> list[float]:
        '''
        `sampler`に基づいて探索
        '''

        for t in range(self.config.T):
            sample_indices = random.sample(range(len(search_space)), self.config.P)
            sample_cells = [search_space[i] for i in sample_indices]
            cell_to_index = {search_space[i]: i for i in sample_indices}
            trained_cells: list[NATSBenchCell] = sampler(sample_cells, data)
            
            # データに追加
            for cell in trained_cells:
                if not cell.evaluated:
                    cell.eval()
                data.append(cell)

            # search_spaceから学習したものを取り除く
            trained_indices = [cell_to_index[cell] for cell in trained_cells]
            trained_indices.sort(reverse=True)
            for index in trained_indices:
                search_space.pop(index)

        ret = sorted([cell.accuracy for cell in data[self.config.D:]], reverse=True) # これの計算時間は問題にならない
        return ret

    def accuracy_compare(
            self,
            wrapper: NATSBenchWrapper,
            ) -> list[float]:
        '''
        精度(画像分類)を計測
        '''

        if self.wl_kernel.is_none:
            self.wl_kernel.init_empty(len(wrapper))
        
        config_original = copy.copy(self.config) 
        
        num_loops = self.config.T
        self.config.T = 1

        gpwl_results = []
            
        random.shuffle(wrapper.cells)
        data: list[NATSBenchCell] = wrapper[:self.config.D]
        for cell in data:
            cell.eval()
        search_space = wrapper[self.config.D:]

        for t in range(num_loops):
            r = self.search(self.gp_with_wl_kernel_sampler, wrapper, data, search_space)

            # 以下は、上位config.eval_length番目のアーキテクチャの精度を記録する場合のコード
            if self.config.acc_tops is not None and len(r) >= self.config.acc_tops:
                gpwl_results.append(r[self.config.acc_tops - 1])
            else:
                gpwl_results.append(0)
            
        self.config = config_original

        self.timer.reset_all()
            
        return gpwl_results

    def time_compare(
            self,
            wrapper: NATSBenchWrapper,
            ) -> dict[str, np.ndarray]:
        '''
        実行時間を計測
        '''
        
        if self.wl_kernel.is_none:
            self.wl_kernel.init_empty(len(wrapper))

        config_original = copy.copy(self.config) 
        
        num_loops = self.config.T
        self.config.T = 1
        
        ret_arr: dict[str, list[float]] = {}
        keys = ['Total', 'WLKernel', 'MatrixMult', 'MatrixInv']
        if self.config.d_max < 1e8:
            keys.append('Bagging')
        keys.append('Others')
        for key in keys:
            ret_arr[key] = []

        random.shuffle(wrapper.cells)
        data: list[NATSBenchCell] = wrapper[:self.config.D]
        for cell in data:
            cell.eval()
        search_space = wrapper[self.config.D:]
        
        for t in range(num_loops):
            self.timer.start('Total')
            self.search(self.gp_with_wl_kernel_sampler, wrapper, data, search_space)
            self.timer.stop('Total')
            ret_arr['Total'].append(self.timer['Total'])
            self.timer.reset('Total')
            for key in filter(lambda k: k != 'Total', keys):
                ret_arr[key].append(self.timer[key])
        
        ret_np: dict[str, np.ndarray] = {}
        for key in filter(lambda k: k != 'Others', keys):
            ret_np[key] = np.array(ret_arr[key])
            
        ret_np['Total'] = np.cumsum(ret_np['Total'])
        
        ret_np['Others'] = ret_np['Total'].copy()
        for key in filter(lambda k: k != 'Total' and k != 'Others', keys):
            ret_np['Others'] -= ret_np[key]

        self.config = config_original

        self.timer.reset_all()

        return ret_np

    def srcc_eval(
            self,
            wrapper: NATSBenchWrapper, 
            ) -> dict[str, np.ndarray]:
        '''
        100個のアーキテクチャに関して、
        真の性能と
        推定された性能の
        ランキングを比較し、
        スピアマンの順位相関係数を計測
        '''

        if self.wl_kernel.is_none:
            self.wl_kernel.init_empty(len(wrapper))
        
        config_original = copy.copy(self.config) 
        
        num_loops = self.config.T
        eval_freq: int = self.config.srcc_eval_freq
        eval_archs: int = self.config.srcc_eval_archs
        search_loops = num_loops // eval_freq
        self.config.T = eval_freq
        
        srcc_list: np.ndarray = np.zeros((search_loops,))
        top_acc: np.ndarray = np.zeros((search_loops,))
        
        random.shuffle(wrapper.cells)
        data: list[NATSBenchCell] = wrapper[:self.config.D]
        for cell in data:
            cell.eval()
        search_space: list[NATSBenchCell] = wrapper[self.config.D:]
        
        for t in range(search_loops):
            self.search(self.gp_with_wl_kernel_sampler, wrapper, data, search_space)
            
            # 探索空間からeval_archs個取り出す
            sample_cells = random.sample(search_space, eval_archs)
            musigma_tuples = self.gp_with_wl_kernel(sample_cells, data)
            for sample_cell in sample_cells:
                if not sample_cell.evaluated:
                    sample_cell.eval()
            true_accs = [cell.accuracy for cell in sample_cells]
            pred_accs = [float(tp[0]) for tp in musigma_tuples]
            srcc_list[t] = spearman_rcc(true_accs, pred_accs) # これの実行時間は問題とならない
            
            list_of_tuple = sorted(zip(pred_accs, true_accs), reverse=True) # 精度が高そうな順に並び変え
            expected_accs = list(list(zip(*list_of_tuple))[1]) # 精度が高そうなもの順に，真の精度を並び替え
            acc = statistics.mean(expected_accs[:10]) #  精度が高そうなアーキテクチャ上位10個の真の精度の平均
            top_acc[t] = acc

        self.config = config_original

        self.timer.reset_all()
        
        return {'srcc': srcc_list, 'acc': top_acc}
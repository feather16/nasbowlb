import numpy as np

def get_ranks(array: list[float]) -> list[int]:
    tmp = np.array(array).argsort()
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(array))
    ranks = len(array) - ranks
    return ranks.tolist()

def spearman_rcc(values1: list[float], values2: list[float]) -> float:
    '''
    スピアマンの順位相関係数
    '''
    ranks1 = get_ranks(values1)
    ranks2 = get_ranks(values2)
    d2 = 0
    for rank1, rank2 in zip(ranks1, ranks2):
        d2 += (rank1 - rank2) ** 2
    N = len(values1)
    
    return 1 - 6 * d2 / (N * N * N - N)
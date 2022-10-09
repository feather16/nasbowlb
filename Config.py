IGNORED_KEYS = ['objective', 'id']

class Config:
    def __init__(
            self,
            *,
            trials: int,
            T: int,
            P: int,
            B: int,
            D: int,
            device: str | None = None,
            strategy: str,
            d_max: int | None = None,
            bagging_rate: float | None = None,
            acc_tops: int | None = None,
            srcc_eval_freq: int | None = None,
            srcc_eval_archs: int | None = None,
            load_kernel_cache: bool = False,
            kernel_cache_path: str | None = None,
            name: str | None = None,
            verbose: bool = False,
            ):
        self.trials = trials
        self.T = T
        self.P = P
        self.B = B
        self.D = D
        self.device: str = device if device is not None else 'cpu'
        self.d_max: int = d_max if d_max is not None else int(1e16)
        self.bagging_rate: float = bagging_rate if bagging_rate is not None else 8.
        self.strategy = strategy
        self.acc_tops = acc_tops
        self.srcc_eval_freq = srcc_eval_freq
        self.srcc_eval_archs = srcc_eval_archs
        self.load_kernel_cache = load_kernel_cache
        self.kernel_cache_path = kernel_cache_path
        if self.load_kernel_cache:
            assert self.kernel_cache_path is not None
        self.name = name
        self.verbose = verbose
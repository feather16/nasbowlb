import time
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.time: dict[str, float] = {}
        self.start_t: dict[str, float] = {}

    def start(self, name: str) -> None:
        self.start_t[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        assert name in self.start_t
        elapsed_time = time.perf_counter() - self.start_t[name]
        if name not in self.time:
            self.time[name] = 0.
        self.time[name] += elapsed_time
        return elapsed_time
    
    @contextmanager
    def measure(self, name: str) -> None:
        self.start(name)
        yield
        self.stop(name)

    def reset(self, name: str) -> None:
        if name in self.time:
            self.time[name] = 0.
            
    def reset_all(self) -> None:
        self.time = {}
        
    def __getitem__(self, name: str) -> float:
        assert isinstance(name, str)
        return self.time[name] if name in self.time else 0.
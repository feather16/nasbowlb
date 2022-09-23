import time

class Timer:
    def __init__(self):
        self.time: dict[str, float] = {}
        self.start_t: dict[str, float] = {}

    def start(self, name: str) -> None:
        self.start_t[name] = time.time()

    def stop(self, name: str) -> float:
        assert name in self.start_t
        elapsed_time = time.time() - self.start_t[name]
        if name not in self.time:
            self.time[name] = 0.
        self.time[name] += elapsed_time
        return elapsed_time

    def reset(self, name: str) -> None:
        if name in self.time:
            self.time[name] = 0.
            
    def reset_all(self) -> None:
        self.time = {}
        
    def __getitem__(self, name: str) -> float:
        return self.time[name] if name in self.time else 0.
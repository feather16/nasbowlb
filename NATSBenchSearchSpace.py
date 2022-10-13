from curses import wrapper
import random
from NATSBenchCell import NATSBenchCell
from NATSBenchWrapper import NATSBenchWrapper

class NATSBenchSearchSpace:
    def __init__(self, wrapper: NATSBenchWrapper):
        self.searched_archs: set[NATSBenchCell] = set()
        self.wrapper = wrapper
    def random_sample(self, n: int) -> list[NATSBenchCell]:
        archs = set()
        count = 0
        while count < n:
            arch = random.choice(self.wrapper.archs)
            if arch not in self.searched_archs and arch not in archs:
                archs.add(arch)
                count += 1
        return list(archs)
    def remove_archs(self, archs: list[NATSBenchCell]) -> None:
        for arch in archs:
            self.searched_archs.add(arch)
    def reset(self) -> None:
        self.searched_archs = set()
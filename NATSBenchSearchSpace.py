from curses import wrapper
import random
from NATSBenchCell import NATSBenchCell
from NATSBenchWrapper import NATSBenchWrapper

class NATSBenchSearchSpace:
    def __init__(self, wrapper: NATSBenchWrapper):
        self.searched_archs: set[NATSBenchCell] = set()
        self.wrapper = wrapper
    def random_sample(self, n: int) -> list[NATSBenchCell]:
        cells = set()
        count = 0
        while count < n:
            cell = random.choice(self.wrapper.cells)
            if cell not in self.searched_archs and cell not in cells:
                cells.add(cell)
                count += 1
        return list(cells)
    def remove_cells(self, cells: list[NATSBenchCell]) -> None:
        for cell in cells:
            self.searched_archs.add(cell)
    def reset(self) -> None:
        self.searched_archs = set()
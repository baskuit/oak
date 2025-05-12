# import torch
# import torch.nn as nn
# import torch.optim as optim

from random import randint, sample, choices
from math import ceil

from moves import Move, legal_moves

class Pokemon:
    def __init__(self, fainted = False):
        self.moves = {Move.NightShade}
        for _ in range(3):
            m = choices(legal_moves, k=1)
            self.moves.add(m[0])
        if not fainted:
            self.hp = randint(1, 353)
        else:
            self.hp = 0
    
    def hits(self):
        return ceil(self.hp / 100)

class Side:
    def __init__(self, switch = False, b = 5):
        self.active = Pokemon(switch)
        self.bench = []
        for _ in range(b):
            self.bench.append(Pokemon())

    def hits(self):
        s = self.active.hits()
        for mon in self.bench:
            s += mon.hits()
        return s

class Battle:
    def __init__(self, request=0):
        self.p1 = Side(request == 1)
        self.p2 = Side(request == 2)

    def score(self):
        diff = self.p1.hits() - self.p2.hits()
        if diff < 0:
            return 0.0
        elif diff == 0:
            return 0.5
        else:
            return 1.0

# Net Constants
ACTIVE_INPUT_SIZE = 3 + 3 + 151 + 161  # result, hp, species, moves
BENCH_INPUT_SIZE = 3 + 151 + 161  # hp, species, moves
HIDDEN_SIZE = 512
OUTPUT_SIZE = 256


if __name__ == "__main__":
    b = Battle()
    print(b.p1.bench)
    print(b.p2.bench)
    print(b.score())
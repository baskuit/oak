import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum

# Pokemon Constants
GHASTLY_HP = 324
HAUNTER_HP = 294
GENGAR_HP = 264

HP_ARRAY = [GHASTLY_HP, HAUNTER_HP, GENGAR_HP]

class Species(Enum):
    GHASTLY = 0
    HAUNTER = 1
    GENGAR = 2

class Request(Enum):
    PASS = 0
    MOVE = 1
    SWITCH = 2

class Pokemon(Enum):
    def __init__(self):
        self.species : Species
        self.hp : int

    def to_tensor(self) -> torch.tensor:
        return torch.zeros((1, 256))

class Side:
    def __init__(self):
        self.active : Pokemon
        self.bench : Pokemon[5]

    def to_tensor(self):
        t = self.active.to_tensor()
        for p in self.bench:
            t += p.to_tensor()
        return t
    
    def hits(self) -> 

class Battle:
    def __init__(self):
        self.p1 : Side
        self.p2 : Side

    def to_tensor(self) -> torch.tensor:
        return torch.concat([self.p1.to_tensor(), self.p2.to_tensor()], dim=1)

def generate_battle() -> Battle:
    return Battle()

# Net Constants
ACTIVE_INPUT_SIZE = 3 + 3 + 151 + 161  # result, hp, species, moves
BENCH_INPUT_SIZE = 3 + 151 + 161  # hp, species, moves
HIDDEN_SIZE = 512
OUTPUT_SIZE = 256

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.clamp(x, max=1.0)
        x = self.fc2(x)
        return x

class ActiveNet(TwoLayerMLP):
    def __init__(self):
        super().__init__(ACTIVE_INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

class BenchNet(TwoLayerMLP):
    def __init__(self):
        super().__init__(BENCH_INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

if __name__ == "__main__":
    pass

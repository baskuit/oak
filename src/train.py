import torch
import torch.nn as nn
import torch.nn.functional as F

n_moves = 166
n_species = 151
n_status = 16
n_volatiles = 32

class Bench:
    def __init__(self,):
        pass

class Active:
    def __init__(self,):
        pass

class Side:
    def __init__(self,):
        pass

class Battle:
    def __init__(self,):
        self.p1 = None
        self.p2 = None

class ClampedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(F.relu(x), 0.0, 1.0)

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = ClampedReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)


x = torch.rand(512)


if __name__ == "__main__":
    pass
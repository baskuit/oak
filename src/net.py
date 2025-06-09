import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
        
def raw_save(tensor, path):
    with open(path, "wb") as f:
        f.write(tensor.numpy().tobytes())

class ClampedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(F.relu(x), 0.0, 1.0)

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.relu = ClampedReLU()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def clamp_weights(self):
        with torch.no_grad():
            self.fc0.weight.clamp_(-2, 2)
            self.fc1.weight.clamp_(-2, 2)
            self.fc2.weight.clamp_(-2, 2)

    def save_quantized(self, path, str):
        raw_save((self.fc0.weight * (64)).to(torch.int8), os.path.join(path, str + "w0"))
        raw_save((self.fc1.weight * (64)).to(torch.int8), os.path.join(path, str + "w1"))
        raw_save((self.fc2.weight * (64)).to(torch.int8), os.path.join(path, str + "w2"))
        raw_save((self.fc0.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "b0"))
        raw_save((self.fc1.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "b1"))
        raw_save((self.fc2.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "b2"))

def count_out_of_bounds_params(model, lower=-2.0, upper=2.0):
    count = 0
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            values = param.data
            count += ((values < lower) | (values > upper)).sum().item()
            total += values.numel()
    return count, total

def quantize_in_dir(path):
    pokemon_net = TwoLayerMLP(198, 64, 39)
    pokemon_net.load(os.path.join(path, "p.pt"))
    active_net = TwoLayerMLP(198 + 14, 64, 55)
    active_net.load(os.path.join(path, "a.pt"))
    main_net = TwoLayerMLP(512, 32, 1)
    main_net.load(os.path.join(path, "nn.pt"))

    cp, tp = count_out_of_bounds_params(pokemon_net)
    ca, ta = count_out_of_bounds_params(active_net)

    if cp > 0 or ca > 0:
        print("one of the nets cant be quantized cus |x| > 2")
        return

    pokemon_net.save_quantized(path, "p")
    active_net.save_quantized(path, "a")
    main_net.save_quantized(path, "nn")

def train():
    # Dummy data
    batch_size = 64
    inputs = torch.randn(batch_size, 512)
    targets = torch.rand(batch_size, 1)  # Target in [0, 1]

    # Model
    model = TwoLayerMLP(input_dim=512, hidden_dim=32, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training step
    model.train()
    optimizer.zero_grad()            # Clear previous gradients
    outputs = model(inputs)          # Forward pass
    loss = criterion(outputs, targets)  # Compute loss
    loss.backward()                  # Backpropagation
    optimizer.step()                 # Update weights

    print("Loss:", loss.item())

if __name__ == '__main__':
    quantize_in_dir("./weights/9500")
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
        
def raw_save(tensor, path):
    with open(path, "wb") as f:
        f.write(tensor.detach().numpy().tobytes())

class ClampedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(F.relu(x), 0.0, 1.0)

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.relu = ClampedReLU()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, print_buffer=False):
        if print_buffer: print("input\n",x)
        x = self.relu(self.fc0(x))
        if print_buffer: print("fc0 out\n",x)
        x = self.relu(self.fc1(x))
        if print_buffer: print("fc1 out\n",x)
        x = self.fc2(x)
        if print_buffer: print("fc2 out\n",x)
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

    def save_float(self, path, str):
        raw_save(self.fc0.weight, os.path.join(path, str + "w0"))
        raw_save(self.fc1.weight, os.path.join(path, str + "w1"))
        raw_save(self.fc2.weight, os.path.join(path, str + "w2"))
        raw_save(self.fc0.bias, os.path.join(path, str + "b0"))
        raw_save(self.fc1.bias, os.path.join(path, str + "b1"))
        raw_save(self.fc2.bias, os.path.join(path, str + "b2"))


class MainNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, policy_output_dim, use_policy=False):
        super(MainNet, self).__init__()
        self.relu = ClampedReLU()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.vfc1 = nn.Linear(hidden_dim, hidden_dim)
        self.vfc2 = nn.Linear(hidden_dim, 1)
        self.use_policy = use_policy
        if use_policy:
            self.pfc1 = nn.Linear(hidden_dim, hidden_dim)
            self.pfc2 = nn.Linear(hidden_dim, policy_output_dim)

    def forward(self, acc, print_buffer=False):
        if print_buffer: print("input\n", acc)
        x = self.relu(self.fc0(acc))
        if print_buffer: print("fc0 out\n", x)
        v = self.relu(self.vfc1(x))
        p = None
        # p = self.relu(self.pfc1(x))
        if print_buffer: print("v/p 1 out\n", v, p)
        v = self.vfc2(v)
        # p = self.pfc2(p)
        if print_buffer: print("v/p 2 out\n", v, p)
        return v, None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def clamp_weights(self):
        with torch.no_grad():
            self.fc0.weight.clamp_(-2, 2)
            self.vfc1.weight.clamp_(-2, 2)
            self.vfc2.weight.clamp_(-2, 2)
            if self.use_policy:
                self.pfc1.weight.clamp_(-2, 2)
                self.pfc2.weight.clamp_(-2, 2)

    def save_quantized(self, path, str):
        raw_save((self.fc0.weight * (64)).to(torch.int8), os.path.join(path, str + "w0"))
        raw_save((self.fc0.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "b0"))

        raw_save((self.vfc1.weight * (64)).to(torch.int8), os.path.join(path, str + "vw1"))
        raw_save((self.vfc2.weight * (64)).to(torch.int8), os.path.join(path, str + "vw2"))
        raw_save((self.vfc1.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "vb1"))
        raw_save((self.vfc2.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "vb2"))

        if self.use_policy:
            raw_save((self.pfc1.weight * (64)).to(torch.int8), os.path.join(path, str + "pw1"))
            raw_save((self.pfc2.weight * (64)).to(torch.int8), os.path.join(path, str + "pw2"))
            raw_save((self.pfc1.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "pb1"))
            raw_save((self.pfc2.bias * (127 * 64)).to(torch.int32), os.path.join(path, str + "pb2"))

    def save_float(self, path, str):
        raw_save(self.fc0.weight, os.path.join(path, str + "w0"))
        raw_save(self.fc0.bias, os.path.join(path, str + "b0"))

        raw_save(self.vfc1.weight, os.path.join(path, str + "vw1"))
        raw_save(self.vfc2.weight, os.path.join(path, str + "vw2"))
        raw_save(self.vfc1.bias, os.path.join(path, str + "vb1"))
        raw_save(self.vfc2.bias, os.path.join(path, str + "vb2"))

        if self.use_policy:
            raw_save(self.pfc1.weight, os.path.join(path, str + "pw1"))
            raw_save(self.pfc2.weight, os.path.join(path, str + "pw2"))
            raw_save(self.pfc1.bias, os.path.join(path, str + "pb1"))
            raw_save(self.pfc2.bias, os.path.join(path, str + "pb2"))

def save_raw_in_dir(path=None, quantize=False):

    if (path is None and len(sys.argv) < 2):
        print("Input: net path")
        exit()

    from frame import Pokemon, Active

    if path is None:
        path = sys.argv[1]

    n_logits = 151 + 165 # all species, all moves (no none, yes struggle)

    pokemon_net = EmbeddingNet(Pokemon.n_dim, 32, 39)
    pokemon_net.load(os.path.join(path, "p.pt"))
    active_net = EmbeddingNet(Active.n_dim, 32, 55)
    active_net.load(os.path.join(path, "a.pt"))
    main_net = MainNet(512, 32, n_logits, False)
    main_net.load(os.path.join(path, "nn.pt"))
    main_net.save_quantized(path, "nn")
    
    if quantize:
        pokemon_net.save_quantized(path, "p")
        active_net.save_quantized(path, "a")
    else:
        pokemon_net.save_float(path, "p")
        active_net.save_float(path, "a")

def read_frame_and_inference():
    if len(sys.argv) < 3:
        print("Input: path to buffer, path to nets.")
        exit()

    from frame import Frame, Pokemon, Active
    FRAME_SIZE = 442

    buffer_path = sys.argv[1]
    net_path = sys.argv[2]

    pokemon_net = EmbeddingNet(Pokemon.n_dim, 32, 39)
    pokemon_net.load(os.path.join(net_path, "p.pt"))
    active_net = EmbeddingNet(Active.n_dim, 32, 55)
    active_net.load(os.path.join(net_path, "a.pt"))

    pokemon_input = torch.zeros((Pokemon.n_dim,))
    active_input = torch.zeros((Active.n_dim,))
    
    # with open(buffer_path, 'rb') as f:
    #     f.seek(FRAME_SIZE)
    #     slice_bytes = f.read(FRAME_SIZE)
    #     frame = Frame(slice_bytes)

    #     frame.battle.p1.pokemon[0].to_tensor(pokemon_input)
    #     active_pokemon = frame.battle.p1.pokemon[frame.battle.p1.order[0] - 1]
    #     active_pokemon.to_tensor(active_input[:198], write_stats=False)
    #     frame.battle.p1.active.to_tensor(active_input)
    pokemon_input[0] = 1.0
    pokemon_output = pokemon_net.relu(pokemon_net.forward(pokemon_input))
    # active_output = active_net.relu(active_net.forward(active_input))
    # print(pokemon_input)
    # print(active_input)
    # print(pokemon_output)
    # print((pokemon_output * 127).to(torch.uint8))
    # print(active_output)
    # print((active_output * 127).to(torch.uint8))

    # print("p net w1 (32x32)")
    # print(pokemon_net.fc1.weight)

if __name__ == '__main__':
    save_raw_in_dir()
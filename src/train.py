# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.multiprocessing as mp

# class ClampedReLU(nn.Module):
#     def forward(self, x):
#         return torch.clamp(F.relu(x), 0.0, 1.0)

# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(TwoLayerMLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.activation = ClampedReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return x

#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path, map_location=None):
#         state_dict = torch.load(path, map_location=map_location)
#         self.load_state_dict(state_dict)


# x = torch.rand(512)
# torch.save(x, "tensor.pt")
# y = torch.load('tensor.pt')

# x.numpy().tofile("tensor.raw")

n_moves = 166
n_species = 151
n_status = 16
n_volatiles = 32


class Bench:
    def __init__(self, buffer : bytes):
        self.species = None
        self.moves = None
        self.status = None
        self.hp = None

class Volatiles:
    def __init__(self, buffer : bytes):
        pass

class Active:
    def __init__(self, buffer : bytes):
        self.species = None
        self.moves = None
        self.status = None
        self.stats = None
        self.volatiles = Volatiles()

class Side:
    def __init__(self, buffer : bytes):
        pass

class Battle:
    def __init__(self, buffer : bytes):
        self.p1 = None
        self.p2 = None

        self.result = None


class Frame:
    def __init__(self, buffer : bytes):
        self.battle = Battle(bytes)
        self.score = float(bytes[393 : 397])
        self.eval = float(bytes[397 : 401])
        bytes.__iter__()

    def input(self):
        pass

import os
import random

FILENAME = 'buffer'
RECORD_SIZE = 405
NUM_READS = 1000

def main():
    # Get the file size to determine how many records fit
    filesize = os.path.getsize(FILENAME)
    max_records = filesize // RECORD_SIZE

    if max_records == 0:
        print("File too small for even one 405-byte record.")
        return

    with open(FILENAME, 'rb') as f:
        for _ in range(NUM_READS):
            # Pick a random record index
            n = random.randint(0, max_records - 1)
            offset = RECORD_SIZE * n

            # Seek to the offset and read the 405-byte slice
            f.seek(offset)
            slice_bytes = f.read(RECORD_SIZE)

            if len(slice_bytes) < 3:
                print(f"Slice at record {n} too small.")
                continue

            # Print the first 3 bytes of the slice
            # print(f"Record {n}, First 3 bytes: {slice_bytes[:3]}")            print(int(slice_bytes[0]))
            print(int(slice_bytes[0]))
            print(int(slice_bytes[1]))
            print(int(slice_bytes[2]))

if __name__ == '__main__':
    main()
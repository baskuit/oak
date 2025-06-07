import sys

if (len(sys.argv) < 5):
    print("Input: buffer_path, batch_size, validation_size, n_procs")
    exit()

import os
import random
import multiprocessing as mp
import time

import torch
import torch.optim as optim

import frame
import net

def get_validation_buffers(validation_buffers, buffer_path, global_buffer_size, max_samples):
    index = random.randint(0, 50)
    i = 0
    while index < global_buffer_size and i < max_samples:

        with open(buffer_path, "rb") as file:
            file.seek(index * frame.FRAME_SIZE)
            data = file.read(frame.FRAME_SIZE)
        assert(len(data) == frame.FRAME_SIZE)

        f = frame.Frame(data)
        f.write_to_buffers(validation_buffers, i)
        index += random.randint(0, 50)
        i += 1

def worker(shared_buffers, buffer_path, global_buffer_size, lock, index_queue, ready_counter):
    while True:
        try:
            index = index_queue.get(timeout=1)
        except:
            continue
        
        # with lock:
        with open(buffer_path, "rb") as file:
            file.seek(random.randint(0, global_buffer_size - 1) * frame.FRAME_SIZE)
            data = file.read(frame.FRAME_SIZE)
        assert(len(data) == frame.FRAME_SIZE)

        f = frame.Frame(data)
        f.write_to_buffers(shared_buffers, index)

        with ready_counter.get_lock():
            ready_counter.value += 1

class SharedBuffers:
    def __init__(self, batch_size):
        self.pokemon_buffer = mp.Array('f', batch_size * 2 * 5 * frame.Frame.pokemon_dim, lock=False)
        self.active_buffer = mp.Array('f', batch_size * 2 * 1 * frame.Frame.active_dim, lock=False)
        self.score_buffer = mp.Array('f', batch_size * 1, lock=False)
        self.eval_buffer = mp.Array('f', batch_size * 1, lock=False)
        self.acc_buffer = mp.Array('f', batch_size * 512, lock = False)

    def to_tensor(self, i = None):
        if i is None:
            return [
                torch.frombuffer(self.pokemon_buffer, dtype=torch.float).view(-1, 2, 5, frame.Frame.pokemon_dim,),
                torch.frombuffer(self.active_buffer, dtype=torch.float).view(-1, 2, 1, frame.Frame.active_dim,),
                torch.frombuffer(self.score_buffer, dtype=torch.float).view(-1, 1),
                torch.frombuffer(self.eval_buffer, dtype=torch.float).view(-1, 1),
                torch.frombuffer(self.acc_buffer, dtype=torch.float).view(-1, 2, 1, 256)]
        p, a, s, e, acc = self.to_tensor()
        return p[i], a[i], s[i], e[i], acc[i]     

def main():

    pokemon_net = net.TwoLayerMLP(frame.Frame.pokemon_dim, 128, frame.POKEMON_OUT - 1)
    pokemon_net.load("weights/p.pt")
    active_net = net.TwoLayerMLP(frame.Frame.active_dim, 128, frame.ACTIVE_OUT - 1)
    active_net.load("weights/a.pt")
    main_net = net.TwoLayerMLP(512, 32, 1)
    main_net.load("weights/nn.pt")

    buffer_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    validation_size = int(sys.argv[3])
    n_procs = int(sys.argv[4])

    print(f"buffer_path={buffer_path}, batch_size={batch_size}, validation_size={validation_size}, n_procs={n_procs}")

    global_buffer_size = os.path.getsize(buffer_path) // frame.FRAME_SIZE
    assert((os.path.getsize(buffer_path) % frame.FRAME_SIZE) == 0)

    validation_buffers = SharedBuffers(validation_size)
    get_validation_buffers(validation_buffers, buffer_path, global_buffer_size, validation_size)

    shared_buffers = SharedBuffers(batch_size)
    lock = mp.Lock()
    ready_counter = mp.Value('i', 0)
    manager = mp.Manager()
    index_queue = manager.Queue()
    for i in range(batch_size):
        index_queue.put(i)

    processes = []
    for _ in range(n_procs):
        p = mp.Process(target=worker, args=(shared_buffers, buffer_path, global_buffer_size, lock, index_queue, ready_counter))
        p.start()
        processes.append(p)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(main_net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(
        list(pokemon_net.parameters()) +
        list(active_net.parameters()) +
        list(main_net.parameters()),
        lr=1e-2
    )

    start = time.perf_counter()
    total_steps = 0
    try:
        while True:
            while True:
                with ready_counter.get_lock():
                    if ready_counter.value >= batch_size:
                        total_steps += 1
                        ready_counter.value = 0
                        break

            p, a, s, e, acc = shared_buffers.to_tensor()

            pokemon_out = pokemon_net.forward(p)
            active_out = active_net.forward(a)

            for player in range(2):
                a_ = active_out[:, player, 0]
                acc[:, player, 0, 1 : frame.ACTIVE_OUT] = a_
                for _ in range(5):
                    start_index = frame.ACTIVE_OUT + _ * frame.POKEMON_OUT
                    acc[:, player, 0, (start_index + 1) : start_index + frame.POKEMON_OUT] = pokemon_out[:, player, _]

            main_net_input = torch.concat([acc[:, 0, 0, :], acc[:, 1, 0, :]], dim=1)
            value_output = main_net.forward(main_net_input)

            loss = criterion(value_output, s)
            # print(value_output.shape, s.shape)
            print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            now = time.perf_counter()
            elapsed = now - start
            rate = total_steps / elapsed
            print(f"rate: {rate}")

            for i in range(batch_size):
                index_queue.put(i)

    finally:
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

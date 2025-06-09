import sys

if (len(sys.argv) < 6):
    print("Input: buffer_path, batch_size, validation_size, n_procs, out_dir")
    exit()

class TrainingParameters:

    pokemon_hidden_dim = 32
    active_hidden_dim = 32
    batch_size = int(sys.argv[2])

    pokemon_lr = .001
    pokemon_momentum = .01
    active_lr = .001
    active_momentum = .01
    main_lr = .001
    main_momentum = .01

class Options:

    buffer_path = sys.argv[1]
    validation_size = int(sys.argv[3])
    n_procs = int(sys.argv[4])
    dir_name = sys.argv[5]

    save_interval = 500
    stats_interval = 50
    output_window_size = 10


import os
import random
import multiprocessing as mp
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

import frame
import net

def get_validation_buffers(validation_buffers, buffer_path, global_buffer_size, max_samples):
    index = random.randint(0, 50)
    i = 0
    while i < max_samples:
        with open(buffer_path, "rb") as file:
            file.seek(index * frame.FRAME_SIZE)
            data = file.read(frame.FRAME_SIZE)
        assert(len(data) == frame.FRAME_SIZE)

        f = frame.Frame(data)
        f.write_to_buffers(validation_buffers, i)
        index += 25 + random.randint(0, 50)
        index %= global_buffer_size
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

    def forward(self, pokemon_net, active_net, main_net):
        p, a, s, e, acc = self.to_tensor() 
        pokemon_out = F.relu(pokemon_net.forward(p))
        active_out = F.relu(active_net.forward(a))

        # write words to acc layer, offset by 1 for the hp entry
        for player in range(2):
            a_ = active_out[:, player, 0]
            acc[:, player, 0, 1 : frame.ACTIVE_OUT] = a_
            for _ in range(5):
                start_index = frame.ACTIVE_OUT + _ * frame.POKEMON_OUT
                acc[:, player, 0, (start_index + 1) : start_index + frame.POKEMON_OUT] = pokemon_out[:, player, _]

        main_net_input = torch.concat([acc[:, 0, 0, :], acc[:, 1, 0, :]], dim=1)
        return torch.sigmoid(main_net.forward(main_net_input))

def main():

    # add iterating through Options, etc params

    for name, value in vars(Options).items():
        print(name, value)

    for name, value in vars(TrainingParameters).items():
        print(name, value)

    if not os.path.exists(Options.dir_name):
        os.mkdir(Options.dir_name)

    pokemon_net = net.TwoLayerMLP(frame.Frame.pokemon_dim, TrainingParameters.pokemon_hidden_dim, frame.POKEMON_OUT - 1)
    active_net = net.TwoLayerMLP(frame.Frame.active_dim, TrainingParameters.active_hidden_dim, frame.ACTIVE_OUT - 1)
    main_net = net.TwoLayerMLP(512, 32, 1)
    criterion = torch.nn.MSELoss()

    global_buffer_size = os.path.getsize(Options.buffer_path) // frame.FRAME_SIZE
    assert((os.path.getsize(Options.buffer_path) % frame.FRAME_SIZE) == 0)
    validation_buffers = SharedBuffers(Options.validation_size)
    get_validation_buffers(validation_buffers, Options.buffer_path, global_buffer_size, Options.validation_size)

    shared_buffers = SharedBuffers(TrainingParameters.batch_size)
    lock = mp.Lock()
    ready_counter = mp.Value('i', 0)
    manager = mp.Manager()
    index_queue = manager.Queue()
    for i in range(TrainingParameters.batch_size):
        index_queue.put(i)
    processes = []
    for _ in range(Options.n_procs):
        proc = mp.Process(target=worker, args=(shared_buffers, Options.buffer_path, global_buffer_size, lock, index_queue, ready_counter))
        proc.start()
        processes.append(proc)

    pokemon_optimizer = torch.optim.Adam(
        pokemon_net.parameters(),
        lr=TrainingParameters.pokemon_lr
    )
    active_optimizer = torch.optim.Adam(
        pokemon_net.parameters(),
        lr=TrainingParameters.active_lr
    )
    main_optimizer = torch.optim.Adam(
        main_net.parameters(),
        lr=TrainingParameters.main_lr
    )

    start = time.perf_counter()
    total_steps = 0
    try:
        while True:
            while True:
                with ready_counter.get_lock():
                    if ready_counter.value >= TrainingParameters.batch_size:
                        break

            ready_counter.value = 0

            p, a, s, e, acc = shared_buffers.to_tensor()
            value_output = shared_buffers.forward(pokemon_net, active_net, main_net)

            if total_steps % Options.stats_interval == 0:
                print(f"stats for step {total_steps}")
                now = time.perf_counter()
                elapsed = now - start
                rate = (total_steps + 1) / elapsed
                print(f"rate: {rate} steps/sec")
                window = torch.cat(
                    [value_output, e, s],
                    dim=1
                )
                print("output; eval; score:")
                print(window[:Options.output_window_size])
                p_, a_, s_, e_, acc_ = validation_buffers.to_tensor()
                validation_output = validation_buffers.forward(pokemon_net, active_net, main_net)
                validation_loss_score = criterion(validation_output, s_)
                validation_loss_eval = criterion(validation_output, e_)
                print(f"validation loss; score: {validation_loss_score}, eval: {validation_loss_eval}")

            loss = criterion(value_output, s)        
            loss.backward()
            for optim in [pokemon_optimizer, active_optimizer, main_optimizer]:
                optim.step()
                optim.zero_grad()
            main_net.clamp_weights()

            if total_steps % Options.save_interval == 0:
                print(f"saving at step {total_steps}.")
                save_dir = os.path.join(Options.dir_name, str(total_steps))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                pokemon_net.save(os.path.join(save_dir, "p.pt"))
                active_net.save(os.path.join(save_dir, "a.pt"))
                main_net.save(os.path.join(save_dir, "nn.pt"))

            for i in range(TrainingParameters.batch_size):
                index_queue.put(i)

            total_steps += 1
    finally:
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

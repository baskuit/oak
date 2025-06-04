import os
import random
import torch
import multiprocessing as mp
import sys

import net
import frame

FILE_PATH = "data.bin"
FRAME_SIZE = 405

global_buffer_size = int(sys.argv[1])
batch_size = int(sys.argv[2])
n_procs = int(sys.argv[3])

with open(FILE_PATH, "wb") as f:
    f.write(os.urandom(global_buffer_size * FRAME_SIZE))

TENSOR_SHAPE = (16, 100)  # (batch size, feature size)
NUM_PROCS = 4

def worker(shared_tensor, shape, lock, index_queue, ready_counter):
    tensor = torch.frombuffer(shared_tensor, dtype=torch.uint8).view(*shape)
    while True:
        try:
            index = index_queue.get(timeout=1)
        except:
            continue

        i = random.randint(0, global_buffer_size - 1)

        with lock:
            with open(FILE_PATH, "rb") as f:
                f.seek(offset * FRAME_SIZE)
                data = f.read(FRAME_SIZE)

        with ready_counter.get_lock():
            ready_counter.value += 1

class SharedBuffers:
    def __init__(self,):
        pokemon_size = 24
        active_size = 64
        self.pokemon_tensor = mp.Array('B', (batch_size, 10, pokemon_size), lock=False)
        self.active_tensor = mp.Array('B', (batch_size, 2, active_size), lock=False)
        self.score_tensor = mp.Array('B', (batch_size, 1), lock=False)
        self.eval_tensor = mp.Array('B', (batch_size, 1), lock=False)

    def to_tensor(self):
        return [None, None, None, None]

def main():

    shape = TENSOR_SHAPE
    size = shape[0] * shape[1]

    shared_buffers = SharedBuffers()


    # Shared data
    shared_tensor = mp.Array('B', size, lock=False)
    lock = mp.Lock()
    ready_counter = mp.Value('i', 0)

    manager = mp.Manager()
    index_queue = manager.Queue()
    for i in range(batch_size):
        index_queue.put(i)

    processes = []
    for _ in range(n_procs):
        p = mp.Process(target=worker, args=(shared_tensor, shape, lock, index_queue, ready_counter))
        p.start()
        processes.append(p)

    total_steps = 0
    try:
        while True:
            while True:
                with ready_counter.get_lock():
                    if ready_counter.value >= shape[0]:
                        total_steps += 1
                        ready_counter.value = 0
                        break

            [p, a, s, e] = shared_buffers.to_tensor()
            # Refill index queue for next batch
            for i in range(shape[0]):
                index_queue.put(i)

    finally:
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

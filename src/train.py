import os
import random
import multiprocessing as mp
import sys
import time

if (len(sys.argv) < 4):
    print("Input: buffer_path, batch_size, n_procs")
    exit()

import torch
import frame
import net

start = time.perf_counter()

FRAME_SIZE = 405

buffer_path = sys.argv[1]
batch_size = int(sys.argv[2])
n_procs = int(sys.argv[3])

global_buffer_size = os.path.getsize(buffer_path) // FRAME_SIZE

assert((os.path.getsize(buffer_path) % FRAME_SIZE) == 0)

TENSOR_SHAPE = (16, 100)  # (batch size, feature size)
NUM_PROCS = 4

def worker(shared_buffers, shape, lock, index_queue, ready_counter):
    while True:
        try:
            index = index_queue.get(timeout=1)
        except:
            continue

        i = random.randint(0, global_buffer_size - 1)

        with lock:
            with open(buffer_path, "rb") as f:
                f.seek(i * FRAME_SIZE)
                data = f.read(FRAME_SIZE)
        assert(len(data) == FRAME_SIZE)

        f = frame.Frame(data)

        with ready_counter.get_lock():
            ready_counter.value += 1

class SharedBuffers:
    def __init__(self, size):
        pokemon_size = 24
        active_size = 64
        self.pokemon_buffer = mp.Array('f', (batch_size, 10, pokemon_size), lock=False)
        self.active_buffer = mp.Array('f', (batch_size, 2, active_size), lock=False)
        self.score_buffer = mp.Array('f', (batch_size, 1), lock=False)
        self.eval_buffer = mp.Array('f', (batch_size, 1), lock=False)
        self.shared_tensor = mp.Array('B', size, lock=False)


    def to_tensor(self):
            return [
                torch.frombuffer(self.pokemon_buffer, dtype=torch.float),
                torch.frombuffer(self.active_buffer, dtype=torch.float),
                torch.frombuffer(self.score_buffer, dtype=torch.float),
                torch.frombuffer(self.eval_buffer, dtype=torch.float)]

def main():

    shape = TENSOR_SHAPE
    size = shape[0] * shape[1]

    shared_buffers = SharedBuffers(size)
    lock = mp.Lock()
    ready_counter = mp.Value('i', 0)

    manager = mp.Manager()
    index_queue = manager.Queue()
    for i in range(batch_size):
        index_queue.put(i)

    processes = []
    for _ in range(n_procs):
        p = mp.Process(target=worker, args=(shared_buffers, shape, lock, index_queue, ready_counter))
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

            now = time.perf_counter()
            elapsed = now - start
            rate = total_steps / elapsed
            print(f"rate: {rate}")
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

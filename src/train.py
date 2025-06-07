import sys

if (len(sys.argv) < 4):
    print("Input: buffer_path, batch_size, n_procs")
    exit()

import os
import random
import multiprocessing as mp
import time

import torch
import frame
import net

def worker(shared_buffers, buffer_path, global_buffer_size, lock, index_queue, ready_counter):
    while True:
        try:
            index = index_queue.get(timeout=1)
        except:
            continue
        
        # with lock:
        with open(buffer_path, "rb") as f:
            f.seek(random.randint(0, global_buffer_size - 1) * frame.FRAME_SIZE)
            data = f.read(frame.FRAME_SIZE)
        assert(len(data) == frame.FRAME_SIZE)

        f = frame.Frame(data)
        f.write_to_buffers(shared_buffers, index)

        with ready_counter.get_lock():
            ready_counter.value += 1

class SharedBuffers:
    def __init__(self, batch_size):
        self.pokemon_buffer = mp.Array('f', batch_size * 2 * 5 * frame.Frame.pokemon_dim, lock=False)
        self.active_buffer = mp.Array('f', batch_size * 2 * 1 * frame.Frame.active_dim, lock=False)
        self.score_buffer = mp.Array('f', batch_size, lock=False)
        self.eval_buffer = mp.Array('f', batch_size, lock=False)


    def to_tensor(self, i = None):
        if i is None:
            return [
                torch.frombuffer(self.pokemon_buffer, dtype=torch.float).view(-1, 2, 5, frame.Frame.pokemon_dim,),
                torch.frombuffer(self.active_buffer, dtype=torch.float).view(-1, 2, 1, frame.Frame.active_dim,),
                torch.frombuffer(self.score_buffer, dtype=torch.float).view(-1, 1),
                torch.frombuffer(self.eval_buffer, dtype=torch.float).view(-1, 1)]
        p, a, s, e = self.to_tensor()
        return p[i], a[i], s[i], e[i]        

def main():

    buffer_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    n_procs = int(sys.argv[3])

    print(f"buffer_path={buffer_path}, batch_size={batch_size}, n_procs={n_procs}")

    global_buffer_size = os.path.getsize(buffer_path) // frame.FRAME_SIZE

    assert((os.path.getsize(buffer_path) % frame.FRAME_SIZE) == 0)

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

            [p, a, s, e] = shared_buffers.to_tensor()

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

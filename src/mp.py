import os
import random
import torch
import multiprocessing as mp

FILE_PATH = "data.bin"
TENSOR_SHAPE = (16, 100)  # (batch size, feature size)
NUM_PROCS = 4

def worker(shared_tensor, shape, lock, index_queue, ready_counter):
    tensor = torch.frombuffer(shared_tensor, dtype=torch.uint8).view(*shape)
    while True:
        try:
            index = index_queue.get(timeout=1)
        except:
            continue

        offset = random.randint(0, os.path.getsize(FILE_PATH) - shape[1])
        with lock:
            with open(FILE_PATH, "rb") as f:
                f.seek(offset)
                data = f.read(shape[1])

        tensor[index] = torch.tensor(list(data), dtype=torch.uint8)

        with ready_counter.get_lock():
            ready_counter.value += 1

def main():
    # Create dummy file
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "wb") as f:
            f.write(os.urandom(10_000))

    shape = TENSOR_SHAPE
    size = shape[0] * shape[1]

    # Shared data
    shared_tensor = mp.Array('B', size, lock=False)
    lock = mp.Lock()
    ready_counter = mp.Value('i', 0)

    # Use Manager to make a shared queue
    manager = mp.Manager()
    index_queue = manager.Queue()

    # Fill initial indices
    for i in range(shape[0]):
        index_queue.put(i)

    processes = []
    for _ in range(NUM_PROCS):
        p = mp.Process(target=worker, args=(shared_tensor, shape, lock, index_queue, ready_counter))
        p.start()
        processes.append(p)

    try:
        while True:
            while True:
                with ready_counter.get_lock():
                    if ready_counter.value >= shape[0]:
                        print("Batch ready!")
                        ready_counter.value = 0
                        break

            # You could now train on the tensor here
            batch_tensor = torch.frombuffer(shared_tensor, dtype=torch.uint8).view(*shape)
            print("Sample from tensor[0]:", batch_tensor[0][:10])

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

#include "../include/actor-pool.hh"

using T = Battle<0, 0, ChanceObs, float, float>;
constexpr size_t pool_size = 1 << 14;
constexpr size_t n_threads = 8;
using AP = ActorPool<T::State, pool_size>;

void thread_fn(AP *actor_pool, uint64_t seed, bool *flag) {
  prng thread_device{seed};
  while (*flag) {
    actor_pool->act(thread_device);
  }
}

size_t time_test(size_t pool_size, size_t n_threads) {
  bool flag{true};
  prng device{123092385034};
  AP actor_pool{device};
  std::thread threads[n_threads];
  for (int i{}; i < n_threads; ++i) {
    threads[i] =
        std::thread{&thread_fn, &actor_pool, device.uniform_64(), &flag};
  }

  sleep(10);
  flag = false;

  for (int i{}; i < n_threads; ++i) {
    threads[i].join();
  }

//   std::cout << actor_pool.total.load() << std::endl;
  return actor_pool.total.load();
}

int main() {
  size_t max_threads = 8;
  for (int i{1}; i <= max_threads; ++i) {
    size_t total = time_test(pool_size, i);
    std::cout << "per thread: " << total / (float)i << std::endl;
  }
  return 0;
}
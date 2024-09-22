#pragma once

#include <atomic>
#include <chrono>
#include <thread>

// define current Helpers::Battle based obs. Set workers to generate libpkmn
// battles

struct Manager {

  int thread_count = 32;

  std::atomic<uint64_t> sampled_histories;

  void start_timer() {
    // start = std::chrono::high_resolution_clock::now();
    sampled_histories.store(0);
  }

  float sample_rate() {
    // const auto duration = start = std::chrono::high_resolution_clock::now() -
    // start;
  }

  static void thread_function();
};
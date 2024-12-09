#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

class ThreadManager {
private:
  std::vector<std::thread> threads;   // Array of threads
  std::vector<bool> thread_available; // Availability of threads
  std::mutex mtx;                     // Mutex for synchronization

public:
  // Constructor to initialize with a given number of threads
  ThreadManager(size_t thread_count = std::thread::hardware_concurrency())
      : threads(thread_count), thread_available(thread_count, true) {}

  // Destructor to join all threads
  ~ThreadManager() {
    for (auto &t : threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  // Templated function to run a callable object
  template <typename Callable, typename... Args>
  std::optional<bool> run(Callable &&task, Args &&...args) {
    std::lock_guard<std::mutex> lock(mtx);

    // Check if a thread is available
    auto it = std::find(thread_available.begin(), thread_available.end(), true);
    if (it == thread_available.end()) {
      return std::nullopt; // No thread available
    }

    // Get the index of the available thread
    size_t index = std::distance(thread_available.begin(), it);
    thread_available[index] = false;

    // Launch the thread
    if (threads[index].joinable()) {
      threads[index].join(); // Ensure the thread is clean before reuse
    }
    std::optional<bool> result;
    threads[index] =
        std::thread([this, index, &result, task = std::forward<Callable>(task),
                     ... args = std::forward<Args>(args)]() mutable {
          result = task(std::forward<Args>(args)...); // Execute the task
          {
            std::lock_guard<std::mutex> lg(mtx);
            thread_available[index] = true; // Mark thread as available
          }
        });

    return result; // Return the forwarded task's return value
  }
};
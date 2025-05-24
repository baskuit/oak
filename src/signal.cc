#include <atomic>
#include <csignal>
#include <iostream>
#include <unistd.h>

std::atomic<bool> running(true);

void handle_signal(int signal) {
  std::cout << "\nCaught signal: " << signal << std::endl;
  running = false;
}

int main() {
  // Catch SIGINT (Ctrl+C)
  std::signal(SIGINT, handle_signal);

  std::cout << "Press Ctrl+C to exit...\n";

  while (running) {
    std::cout << ".";
    std::cout.flush();
    sleep(1);
  }

  std::cout << "\nExiting gracefully.\n";
  return 0;
}

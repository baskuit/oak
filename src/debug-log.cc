#include <iostream>
#include <mutex>

#include <types/random.h>
#include <types/vector.h>

#include <battle/chance.h>
#include <battle/data/data.h>
#include <battle/debug-log.h>

int rollout_sample_teams_and_save_log(int argc, char **argv) {
  constexpr size_t log_size{64};
  if (argc != 4) {
    std::cout
        << "Usage: provide two sample team indices [0 - 99] and a u64 seed."
        << std::endl;
    return 1;
  }
  int p1 = std::atoi(argv[1]);
  int p2 = std::atoi(argv[2]);

  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_save_log(argc, argv);
}

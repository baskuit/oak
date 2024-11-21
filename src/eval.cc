#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>

#include <battle/init.h>

#include <pi/abstract.h>

#include <util/random.h>

#include <iostream>
#include <numeric>
#include <sstream>

static_assert(Options::calc && Options::chance && !Options::log);

int all_1v1(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Usage: provide seed, two set indices, and mcts iterations"
              << std::endl;
    return 1;
  }
  const uint64_t seed = std::atoi(argv[1]);
  const int i = std::atoi(argv[2]);
  const int j = std::atoi(argv[3]);
  const size_t iterations = std::atoi(argv[4]);

  prng device{seed};

  auto battle = Init::battle(SampleTeams::teams[0], SampleTeams::teams[1]);

  return 0;
}

int main(int argc, char **argv) { return all_1v1(argc, argv); }

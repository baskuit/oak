#include <battle/init.h>
#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>
#include <pi/abstract.h>
#include <util/random.h>

#include <iostream>

static_assert(Options::calc && Options::chance && !Options::log);

int abstract_test(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: provide two team indices to start Abstract test"
              << std::endl;
    return 1;
  }
  const uint64_t seed = 192380129381;
  const int i = std::atoi(argv[1]);
  const int j = std::atoi(argv[2]);

  prng device{seed};

  auto battle = Init::battle(SampleTeams::teams[0], SampleTeams::teams[1]);
  pkmn_gen1_battle_options options;
  auto result = Init::update(battle, 0, 0, options);

  Abstract::Battle abstract{battle};

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }

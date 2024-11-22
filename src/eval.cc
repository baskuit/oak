#include <battle/debug-log.h>
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

  auto battle = Init::battle(SampleTeams::teams[i], SampleTeams::teams[j]);
  pkmn_gen1_battle_options options;
  auto result = Init::update(battle, 0, 0, options);
  Abstract::Battle abstract{battle};

  int t = 0;
  while (!pkmn_result_type(result)) {
    const auto [choices1, choices2] = Init::choices(battle, result);
    auto c1 = choices1[device.random_int(choices1.size())];
    auto c2 = choices2[device.random_int(choices2.size())];
    result = Init::update(battle, c1, c2, options);
    ++t;
  }

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }

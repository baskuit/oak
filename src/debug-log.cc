#include <iostream>

#include <types/random.h>

#include <battle/data/data.h>
#include <battle/debug-log.h>

int rollout_sample_teams_and_save_log(int argc, char **argv) {
  constexpr size_t log_size{128};
  if (argc != 4) {
    std::cout
        << "Usage: provide two sample team indices [0 - 99] and a u64 seed."
        << std::endl;
    return 1;
  }

  int p1 = std::atoi(argv[1]);
  int p2 = std::atoi(argv[2]);
  uint64_t seed = std::atoi(argv[3]);

  auto battle =
      Data::init_battle(SampleTeams::teams[p1], SampleTeams::teams[p2], seed);
  pkmn_gen1_battle_options options{};

  DebugLog<log_size> debug_log{};
  debug_log.set_header(battle);
  prng device{seed};

  int turns = 0;
  pkmn_choice c1{0};
  pkmn_choice c2{0};
  auto result = debug_log.update_battle(&battle, &options, c1, c2);
  std::array<pkmn_choice, 9> choices{};
  while (!pkmn_result_type(result)) {
    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto i = device.random_int(m);
    c1 = choices[i];
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto j = device.random_int(n);
    c2 = choices[j];

    result = debug_log.update_battle(&battle, &options, c1, c2);
    ++turns;
  }

  std::cout << "Played " << turns << " turns." << std::endl;

  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_save_log(argc, argv);
}

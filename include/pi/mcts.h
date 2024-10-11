#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <pi/exp3.h>

#include <pkmn.h>

#include <types/random.h>

#define size_t int

namespace offset {
constexpr auto seed = 376;
};

class MCTS {
private:
  pkmn_gen1_battle_options options;
  size_t total_nodes;
  size_t total_depth;

public:
  auto run(auto iterations, auto &prng, auto &eval, auto &node,
           const pkmn_gen1_battle const *battle, pkmn_result result,
           const pkmn_gen1_chance_durations const *durations) {

    for (auto iteration = 0; iteration < iterations; ++iteration) {
      auto battle_copy = battle;
      auto *battle_seed =
          std::bit_cast<uint64_t *>(battle_copy.bytes + offset::seed);
      *battle_seed = prng.uniform_64();

      pkmn_gen1_chance_options chance_options{};
      chance_options.durations = &durations;
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

      run_iteration(prng, eval, node, battle_copy, result);
    }
  }

private:
  float run_iteration(auto &prng, auto &eval, auto &node,
                      pkmn_gen1_battle const *battle,
                      pkmn_result result) {
    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] { break; }
    case PKMN_RESULT_WIN: {
      return 1.0;
    }
    case PKMN_RESULT_LOSE: {
      return 0.0;
    }
    case PKMN_RESULT_TIE: {
      return 0.5;
    }
    default: {
      assert(false);
      return 0.5;
    }
    };

    if (!node.is_init()) {
      node.init();
      return eval(battle);
    }

    // do bandit
    std::array<pkmn_choice, 9> choices;


    return 0;
  }

  void rollout(auto& prng, pkmn_gen1_battle *battle, pkmn_result result) {

  }

};

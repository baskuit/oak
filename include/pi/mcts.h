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
  std::array<pkmn_choice, 9> choices;

public:
  auto run(const auto iterations, auto &prng, auto &eval, auto &node,
           const pkmn_gen1_battle const *battle, pkmn_result result,
           const pkmn_gen1_chance_durations const *durations) {
    total_nodes = 0;
    total_depth = 0;
    for (auto iteration = 0; iteration < iterations; ++iteration) {
      auto battle_copy = battle;
      auto *battle_seed =
          std::bit_cast<uint64_t *>(battle_copy.bytes + offset::seed);
      *battle_seed = prng.uniform_64();

      pkmn_gen1_chance_options chance_options{};
      chance_options.durations = &durations;
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

      run_iteration(prng, eval, &node, battle_copy, result);
    }

    struct Output {

    };
  }

private:
  float run_iteration(auto &prng, auto &eval, auto *node,
                      pkmn_gen1_battle const *battle,
                      pkmn_result result, auto depth) {
    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] { break; }
    case PKMN_RESULT_WIN: {
      total_depth += depth;
      return 1.0;
    }
    case PKMN_RESULT_LOSE: {
      total_depth += depth;
      return 0.0;
    }
    case PKMN_RESULT_TIE: {
      total_depth += depth;
      return 0.5;
    }
    default: {
      assert(false);
      total_depth += depth;
      return 0.5;
    }
    };

    if (!node.is_init()) {
      ++total_nodes;
      total_depth += depth;
      node.init(battle, result);
      // return eval(battle, result);
      return rollout(prng, battle, result);
    }

    using Bandit = decltype(node.stats());
    using Outcome = typename Bandit::Outcome;
    Outcome outcome;

    // do bandit
    pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(), PKMN_GEN1_MAX_CHOICES);
    node.stats.select(outcome, rows);
    const auto c1 = choices[outcome.p1_index];
    pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(), PKMN_GEN1_MAX_CHOICES);
    node.stats.select(outcome, cols);
    const auto c2 = choices[outcome.p2_index];

    pkmn_gen1_battle_update(battle, c1, c2, &options);
    const auto* chance_actions = pkmn_gen1_battle_options_chance_durations(&options);
    const auto& obs = *std::bit_cast<std::array<uint8_t, 16> *>(chance_actions->bytes);

    const auto *child = node(outcome.p1_index, outcome.p2_index, obs);
    const auto value = run_iteration(prng, eval, child, battle, result, depth + 1);

    return value;
  }

  void rollout(auto& prng, pkmn_gen1_battle *battle, pkmn_result result) {
    while (!pkmn_result_type(result)) {
      const uint64_t seed = prng.uniform_64();
      const auto rows = pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(), PKMN_GEN1_MAX_CHOICES);
      const pkmn_choice c1 = choices[seed % rows];
      seed >>= 32;
      const auto cols = pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[seed % cols];
      result = pkmn_gen1_battle_update(battle, c1, c2);
    }
    switch (pkmn_result_type(result)) {
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
  }

};

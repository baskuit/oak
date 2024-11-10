#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <data/strings.h>
#include <pi/exp3.h>

#include <pkmn.h>

#include <types/random.h>

namespace offset {
constexpr auto seed = 376;
};

template <bool debug_print = false> class MCTS {
public:
  pkmn_gen1_battle_options options;
  size_t total_nodes;
  size_t total_depth;
  std::array<pkmn_choice, 9> choices;

  auto run(const auto iterations, auto &prng, auto &node,
           const pkmn_gen1_battle *const battle, pkmn_result result,
           const pkmn_gen1_chance_durations *const durations) {
    total_nodes = 0;
    total_depth = 0;
    for (auto iteration = 0; iteration < iterations; ++iteration) {
      auto battle_copy = *battle;
      auto *battle_seed =
          std::bit_cast<uint64_t *>(battle_copy.bytes + offset::seed);
      *battle_seed = prng.uniform_64();

      pkmn_gen1_chance_options chance_options{};
      chance_options.durations = *durations;
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

      run_iteration(prng, &node, &battle_copy, result);
    }

    struct Output {};
  }

private:
  void print(const auto data, size_t depth) {
    if constexpr (!debug_print) {
      return;
    }

    std::cout << std::string('\t', depth);
    std::cout << data << std::endl;
  }

  float run_iteration(auto &prng, auto *node, pkmn_gen1_battle *battle,
                      pkmn_result result, size_t depth = 0) {

    if (node->stats().is_init()) {
      using Bandit = std::remove_reference<decltype(node->stats())>::type;
      using Outcome = typename Bandit::Outcome;
      Outcome outcome;

      // do bandit
      node->stats().select(prng, outcome);

      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = choices[outcome.p1_index];
      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[outcome.p2_index];

      print("___", depth);

      pkmn_gen1_battle_update(battle, c1, c2, &options);
      const auto *chance_actions =
          pkmn_gen1_battle_options_chance_actions(&options);
      const auto &obs =
          std::bit_cast<std::array<uint8_t, 16>>(chance_actions->bytes);

      std::cout << "Obs: " << buffer_to_string(obs.data(), 16) << std::endl;

      auto *child = (*node)(outcome.p1_index, outcome.p2_index, obs);
      const auto value = run_iteration(prng, child, battle, result, depth + 1);

      return value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        ++total_nodes;
        return init_stats_and_rollout(node->stats(), prng, battle, result);
      }
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

  float init_stats_and_rollout(auto &stats, auto &prng,
                               pkmn_gen1_battle *battle, pkmn_result result) {

    const auto p1_choices =
        pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                                 choices.data(), PKMN_GEN1_MAX_CHOICES);
    const auto p2_choices =
        pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                                 choices.data(), PKMN_GEN1_MAX_CHOICES);
    stats.init(p1_choices, p2_choices);
    do {
      uint64_t seed = prng.uniform_64();
      const auto rows = pkmn_gen1_battle_choices(
          battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      const pkmn_choice c1 = choices[seed % rows];
      seed >>= 32;
      const auto cols = pkmn_gen1_battle_choices(
          battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[seed % cols];
      result = pkmn_gen1_battle_update(battle, c1, c2, &options);
    } while (!pkmn_result_type(result));

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

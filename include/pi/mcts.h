#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <data/strings.h>
#include <pi/exp3.h>
#include <pi/tree.h>

#include <pkmn.h>

#include <data/offsets.h>
#include <data/options.h>

static_assert(Options::chance && Options::calc && !Options::log);

template <bool debug_print = false, bool clamp_rolls = true> class MCTS {
public:
  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  size_t total_nodes;
  size_t total_depth;
  std::array<pkmn_choice, 9> choices;
  std::array<std::array<uint32_t, 9>, 9> visit_matrix;

  struct Output {
    size_t iterations;
    float total_value;
    float average_value;
    std::vector<float> p1;
    std::vector<float> p2;
    std::array<std::array<uint32_t, 9>, 9> visit_matrix;
  };

  auto run(const auto iterations, auto &prng, auto &node,
           const pkmn_gen1_battle *const battle, const pkmn_result result,
           const pkmn_gen1_chance_durations *const durations) {

    chance_options = {};
    calc_options = {};
    total_nodes = 0;
    total_depth = 0;
    constexpr bool enable_visit_matrix = true;
    if constexpr (enable_visit_matrix) {
      visit_matrix = {};
    }
    float total_value = 0;
    for (auto iteration = 0; iteration < iterations; ++iteration) {
      auto battle_copy = *battle;
      auto *battle_seed =
          std::bit_cast<uint64_t *>(battle_copy.bytes + Offsets::seed);
      *battle_seed = prng.uniform_64();

      chance_options.durations = *durations;
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);

      total_value +=
          run_iteration<enable_visit_matrix>(prng, &node, &battle_copy, result);
    }

    Output output;
    output.iterations = iterations;
    output.total_value = total_value;
    output.average_value = total_value / iterations;
    const auto [c1, c2] = Helper::get_choices(battle, result);
    output.p1.resize(c1.size());
    output.p2.resize(c2.size());
    if constexpr (enable_visit_matrix) {
      for (int i = 0; i < c1.size(); ++i) {
        for (int j = 0; j < c2.size(); ++j) {
          output.p1[i] += visit_matrix[i][j];
          output.p2[j] += visit_matrix[i][j];
        }
      }
      for (int i = 0; i < c1.size(); ++i) {
        output.p1[i] /= (float)iterations;
      }
      for (int j = 0; j < c2.size(); ++j) {
        output.p2[j] /= (float)iterations;
      }
      output.visit_matrix = visit_matrix;
    }
    return output;
  }

private:
  template <bool enable_visit_matrix>
  float run_iteration(auto &prng, auto *node, pkmn_gen1_battle *battle,
                      pkmn_result result, size_t depth = 0) {
    const auto print = [depth](const auto data, bool endl = true) {
      if constexpr (!debug_print) {
        return;
      }
      for (auto i = 0; i < depth; ++i) {
        std::cout << "  ";
      }
      std::cout << data;
      if (endl) {
        std::cout << std::endl;
      }
    };

    const auto battle_options_set = [this, battle]() {
      if constexpr (clamp_rolls) {
        constexpr auto rolls = 3;
        constexpr int step = 38 / (rolls - 1);
        this->calc_options.overrides.bytes[0] =
            217 + step * (battle->bytes[Offsets::seed + 7] % rolls);
        this->calc_options.overrides.bytes[8] =
            217 + step * (battle->bytes[Offsets::seed + 8] % rolls);
        pkmn_gen1_battle_options_set(&this->options, NULL, NULL,
                                     &this->calc_options);
      } else {
        pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      }
    };

    if (node->stats().is_init()) {
      using Bandit = std::remove_reference_t<decltype(node->stats())>;
      using Outcome = typename Bandit::Outcome;

      // do bandit
      Outcome outcome;
      node->stats().select(prng, outcome);

      if constexpr (enable_visit_matrix) {
        if (depth == 0) {
          ++visit_matrix[outcome.p1_index][outcome.p2_index];
        }
      }

      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = choices[outcome.p1_index];
      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[outcome.p2_index];

      print("P1: " + side_choice_string(battle->bytes, c1) +
            " P2: " + side_choice_string(battle->bytes + Offsets::side, c2));
      print(node->stats().visit_string());

      battle_options_set();
      result = pkmn_gen1_battle_update(battle, c1, c2, &options);
      const auto &obs = std::bit_cast<const std::array<uint8_t, 16>>(
          *pkmn_gen1_battle_options_chance_actions(&options));

      print("Obs: " + buffer_to_string(obs.data(), 16));
      auto *child = (*node)(outcome.p1_index, outcome.p2_index, obs);
      outcome.value = run_iteration<enable_visit_matrix>(prng, child, battle,
                                                         result, depth + 1);
      node->stats().update(outcome);

      print("value: " + std::to_string(outcome.value));

      return outcome.value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        ++total_nodes;
        print("Initializing node");
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

    auto seed = prng.uniform_64();
    auto m =
        pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                                 choices.data(), PKMN_GEN1_MAX_CHOICES);
    auto c1 = choices[seed % m];
    auto n =
        pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                                 choices.data(), PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    auto c2 = choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(battle, c1, c2, &options);
    stats.init(m, n);
    while (!pkmn_result_type(result)) {
      seed = prng.uniform_64();
      m = pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1,
                                   pkmn_result_p1(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      c1 = choices[seed % m];
      n = pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2,
                                   pkmn_result_p2(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      c2 = choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(battle, c1, c2, &options);
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

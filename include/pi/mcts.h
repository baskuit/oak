#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <data/strings.h>
#include <pi/exp3.h>
#include <pi/tree.h>
#include <util/random.h>

#include <pkmn.h>

#include <data/durations.h>
#include <data/offsets.h>
#include <data/options.h>

static_assert(Options::chance && Options::calc && !Options::log);

namespace MonteCarlo {
struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

struct Model {
  prng device;
};
} // namespace MonteCarlo

struct MCTS {

  template <bool _root_matrix = true, size_t _root_rolls = 3,
            size_t _other_rolls = 1, size_t _min_mon = 1>
  struct Options {
    static constexpr bool root_matrix = _root_matrix;
    static constexpr size_t root_rolls = _root_rolls;
    static constexpr size_t other_rolls = _other_rolls;
    static constexpr bool debug_print = false;
    static constexpr size_t min_mon = _min_mon;

    static constexpr size_t rolls_same = (root_rolls == other_rolls);
    static constexpr size_t can_defer = (_min_mon > 1);
    static constexpr bool clamping = (root_rolls != 39) || (other_rolls != 39);
  };

  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  pkmn_gen1_calc_options calc_options;
  size_t total_nodes;
  size_t total_depth;
  std::array<pkmn_choice, 9> choices;
  std::array<std::array<uint32_t, 9>, 9> visit_matrix;
  std::array<std::array<float, 9>, 9> value_matrix;

  struct Output {
    size_t iterations;
    float total_value;
    float average_value;
    std::vector<pkmn_choice> choices1;
    std::vector<pkmn_choice> choices2;
    std::vector<float> p1;
    std::vector<float> p2;
    std::array<std::array<uint32_t, 9>, 9> visit_matrix;
    std::array<std::array<float, 9>, 9> value_matrix;
    std::chrono::milliseconds duration;
  };

  void set_sleep(auto &device, auto &battle, const auto &durations) const {
    for (auto s = 0; s < 2; ++s) {
      const auto side = battle.bytes + s * Offsets::side;
      const auto order = side + Offsets::order;
      for (auto p = 0; p < 6; ++p) {
        if (const auto d = Durations::sleep(durations, s, p)) {
          const auto slot = order[p] - 1;
          const auto pokemon = side + Offsets::pokemon * slot;

          if (0b10000000 & pokemon[Offsets::status]) {
            continue;
          }

          const uint8_t max = 8 - d;
          pokemon[Offsets::status] &= 0b11111000;
          pokemon[Offsets::status] |=
              static_cast<uint8_t>(device.random_int(max) + 1);
        }
      }
    }
  }

  template <typename Options = Options<>>
  auto run(const auto dur, auto &node, auto &input, auto &model) {

    const auto iter = [this, &input, &model, &node]() -> float {
      auto copy = input;
      std::bit_cast<uint64_t *>(copy.battle.bytes + Offsets::seed)[0] =
          model.device.uniform_64();
      chance_options.durations = copy.durations;
      set_sleep(model.device, copy.battle, copy.durations); // TODO
      pkmn_gen1_battle_options_set(&options, nullptr, &chance_options, nullptr);
      const auto value = run_iteration<Options>(&node, copy, model);
      return value;
    };

    *this = {};
    Output output{};

    // passing chrono duration for iteration count
    if constexpr (requires {
                    std::chrono::duration_cast<std::chrono::milliseconds>(dur);
                  }) {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
      std::chrono::milliseconds elapsed{};
      while (elapsed < duration) {
        output.total_value += iter();
        ++output.iterations;
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
      }
      output.duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    } else {
      const auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < dur; ++i) {
        output.total_value += iter();
      }
      output.iterations = dur;
      const auto end = std::chrono::high_resolution_clock::now();
      output.duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    output.average_value = output.total_value / output.iterations;
    const auto [c1, c2] = Init::choices(input.battle, input.result);
    output.choices1 = c1;
    output.choices2 = c2;
    output.p1.resize(c1.size());
    output.p2.resize(c2.size());

    if constexpr (Options::root_matrix) {
      for (int i = 0; i < c1.size(); ++i) {
        for (int j = 0; j < c2.size(); ++j) {
          output.p1[i] += visit_matrix[i][j];
          output.p2[j] += visit_matrix[i][j];
        }
      }
      for (int i = 0; i < c1.size(); ++i) {
        output.p1[i] /= (float)output.iterations;
      }
      for (int j = 0; j < c2.size(); ++j) {
        output.p2[j] /= (float)output.iterations;
      }
      output.visit_matrix = visit_matrix;
      output.value_matrix = value_matrix;
      // for (const auto &row : value_matrix) {

      // }
    }

    return output;
  }

  template <size_t rolls> constexpr uint8_t roll_byte(const uint8_t seed) {
    if constexpr (rolls == 1) {
      // mid
      return 236;
    } else {
      static_assert((rolls == 2) || (rolls == 3) || (rolls == 20));
      constexpr int step = 38 / (rolls - 1);
      return 217 + step * (seed % rolls);
    }
  }

  template <typename Options>
  float run_iteration(auto *node, auto &input, auto &model, size_t depth = 0) {

    auto &battle = input.battle;
    auto &durations = input.durations;
    auto &result = input.result;
    auto &device = model.device;

    const auto print = [depth](const auto &data, bool endl = true) -> void {
      if constexpr (!Options::debug_print) {
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

    const auto battle_options_set = [this, &battle, depth]() {
      if constexpr (!Options::clamping) {
        pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      } else {
        const auto *rand = battle.bytes + Offsets::seed + 6;
        auto *over = this->calc_options.overrides.bytes;
        if constexpr (Options::rolls_same) {
          over[0] = roll_byte<Options::root_rolls>(rand[0]);
          over[8] = roll_byte<Options::root_rolls>(rand[1]);
        } else {

          if (depth == 0) {
            over[0] = roll_byte<Options::root_rolls>(rand[0]);
            over[8] = roll_byte<Options::root_rolls>(rand[1]);
          } else {
            over[0] = roll_byte<Options::other_rolls>(rand[0]);
            over[8] = roll_byte<Options::other_rolls>(rand[1]);
          }
        }
        pkmn_gen1_battle_options_set(&this->options, nullptr, nullptr,
                                     &this->calc_options);
      }
    };

    if (node->stats().is_init()) {
      using Bandit = std::remove_reference_t<decltype(node->stats())>;
      using Outcome = typename Bandit::Outcome;

      // do bandit
      Outcome outcome;

      node->stats().select(device, outcome);
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c1 = choices[outcome.p1_index];
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               choices.data(), PKMN_GEN1_MAX_CHOICES);
      const auto c2 = choices[outcome.p2_index];

      print("P1: " + side_choice_string(battle.bytes, c1) +
            " P2: " + side_choice_string(battle.bytes + Offsets::side, c2));
      print(node->stats().visit_string());

      battle_options_set();
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
      // if constexpr (requires { input.abstract; }) {
      // input.abstract.update(battle, model.eval.ovo_matrix);
      // auto &abstract = input.abstract;
      // decltype(input.abstract) copy{battle, model.eval.ovo_matrix};
      // assert(copy.hp1 == abstract.hp1);
      // assert(copy.hp2 == abstract.hp2);
      // assert(copy.status1 == abstract.status1);
      // assert(copy.status2 == abstract.status2);
      // }
      const auto &obs = std::bit_cast<const std::array<uint8_t, 16>>(
          *pkmn_gen1_battle_options_chance_actions(&options));

      print("Obs: " + buffer_to_string(obs.data(), 16));
      auto *child = (*node)(outcome.p1_index, outcome.p2_index, obs);
      const auto value = run_iteration<Options>(child, input, model, depth + 1);
      outcome.value = value;
      node->stats().update(outcome);

      print("value: " + std::to_string(value));

      if constexpr (Options::root_matrix) {
        if (depth == 0) {
          ++visit_matrix[outcome.p1_index][outcome.p2_index];
          value_matrix[outcome.p1_index][outcome.p2_index] += value;
        }
      }

      return outcome.value;
    }

    total_depth += depth;

    switch (pkmn_result_type(result)) {
    case PKMN_RESULT_NONE:
      [[likely]] {
        print("Initializing node");
        ++total_nodes;
        if constexpr (requires { model.eval; }) {
          init_stats(node->stats(), battle, result);
          decltype(input.abstract) abstract{input.battle,
                                            model.eval.ovo_matrix};
          if constexpr (Options::can_defer) {
            if (abstract.m < Options::min_mon ||
                abstract.n < Options::min_mon) {
              return rollout(node->stats(), device, battle, result);
            } else {
              return model.eval.value(abstract);
            }
          } else {
            return model.eval.value(abstract);
          }
        } else {
          return init_stats_and_rollout(node->stats(), device, battle, result);
        }
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
                               pkmn_gen1_battle &battle, pkmn_result result) {

    auto seed = prng.uniform_64();
    auto m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                      pkmn_result_p1(result), choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    auto c1 = choices[seed % m];
    auto n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                      pkmn_result_p2(result), choices.data(),
                                      PKMN_GEN1_MAX_CHOICES);
    seed >>= 32;
    auto c2 = choices[seed % n];
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
    result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
    stats.init(m, n);
    while (!pkmn_result_type(result)) {
      seed = prng.uniform_64();
      m = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1,
                                   pkmn_result_p1(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      c1 = choices[seed % m];
      n = pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2,
                                   pkmn_result_p2(result), choices.data(),
                                   PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      c2 = choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
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

  void init_stats(auto &stats, pkmn_gen1_battle &battle, pkmn_result result) {

    const auto m = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    const auto n = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
        PKMN_GEN1_MAX_CHOICES);
    stats.init(m, n);
  }

  float rollout(auto &stats, auto &prng, pkmn_gen1_battle &battle,
                pkmn_result result) {
    while (!pkmn_result_type(result)) {
      auto seed = prng.uniform_64();
      const auto m = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      const auto c1 = choices[seed % m];
      const auto n = pkmn_gen1_battle_choices(
          &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), choices.data(),
          PKMN_GEN1_MAX_CHOICES);
      seed >>= 32;
      const auto c2 = choices[seed % n];
      pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options);
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
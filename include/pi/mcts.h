#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <pi/exp3.h>

#include <types/random.h>

struct MCTS {

  size_t total_nodes;
  size_t total_depth;

  template <typename PRNG, typename Node, typename Battle, typename Model>
  float run_iteration(PRNG &device, Node *node, Battle &battle, Model &model,
                      int depth = 0) {
    if (battle.terminal()) {
      return battle.payoff();
    }

    if (!node->is_init()) {
      ++total_nodes;
      total_depth += depth;
      node->init(battle.rows(), battle.cols());
      return model.inference(battle);
    }

    auto &data = node->data();
    using Outcome = std::remove_reference_t<decltype(data)>::Outcome;
    Outcome outcome;
    data.select(device, outcome);
    battle.apply_actions(battle.row_actions[outcome.row_idx],
                         battle.col_actions[outcome.col_idx]);
    battle.get_actions();
    Node *next_node = (*node)(outcome.row_idx, outcome.col_idx, battle.obs());

    outcome.value = run_iteration(device, next_node, battle, model, depth + 1);
    data.update(outcome);
    return outcome.value;
  }

  template <typename PRNG, typename Node, typename Battle, typename Model>
  auto run(size_t iterations, PRNG &device, Battle &battle, Model &model,
           Node *node) {
    double total_value = 0;
    const size_t window_size = 1 << 8;
    std::array<float, window_size> window{};

    for (auto i = 0; i < iterations; ++i) {
      auto battle_copy{battle};
      battle_copy.randomize_transition(device);
      const float value = MCTS::run_iteration(device, node, battle_copy, model);
      total_value += value;
      window[i % window_size] = value;
    }

    struct Output {
      double average_value;
      double rolling_average_value;
      double average_depth;
      size_t window_size;
      std::vector<float> row_strategy;
      std::vector<float> col_strategy;
    };
    Output output;
    output.window_size = window_size;
    output.average_value = total_value / iterations;
    output.rolling_average_value =
        std::accumulate(window.begin(), window.end(), 0.0) /
        (double)window_size;
    output.average_depth = total_depth / (double)iterations;
    output.row_strategy = Exp3::empirical_strategies(node->data().row_visits);
    output.col_strategy = Exp3::empirical_strategies(node->data().col_visits);
    return output;
  }
};

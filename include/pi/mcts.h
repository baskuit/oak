#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include <pi/exp3.h>

#include <types/random.h>

namespace MCTS {

static size_t total_nodes = 0;

template <typename PRNG, typename Node, typename Battle, typename Model>
float run_iteration(PRNG &device, Node *node, Battle &battle, Model &model) {
  if (battle.is_terminal()) {
    // std::cout << "terminal." << std::endl;
    return battle.payoff();
  }

  if (!node->is_init()) {
    ++total_nodes;
    node->init(battle.rows(), battle.cols());
    // std::cout << "terminal." << std::endl;

    return model.inference(std::move(battle));
  }

  auto &data = node->data();
  using Outcome = std::remove_reference_t<decltype(data)>::Outcome;
  Outcome outcome;
  data.select(device, outcome);
  // std::cout << "select: " << outcome.row_idx << ' ' << outcome.col_idx << ' '
  // << outcome.row_mu << ' ' << outcome.col_mu;
  battle.apply_actions(battle.row_actions[outcome.row_idx],
                       battle.col_actions[outcome.col_idx]);
  battle.get_actions();
  Node *next_node = (*node)(outcome.row_idx, outcome.col_idx, battle.obs());

  outcome.value = run_iteration(device, next_node, battle, model);
  data.update(outcome);
  return outcome.value;
}

}; // namespace MCTS

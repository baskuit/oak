#pragma once

#include <cstdint>
#include <utility>

namespace MCTS {

struct Outcome {
    uint8_t row_idx;
    uint8_t col_idx;
    float value;
};

template <typename Node, typename Battle, typename Model>
float run_iteration (Node *node, Battle& battle, Model &model) {
    if (battle.terminal()) {
        return battle.payoff();
    }

    if (!node->is_init()) {
        node->init(battle.rows(), battle.cols());
        return model.inference(std::move(battle));
    }

    Outcome outcome;

    auto& data = node->data();
    data.select(outcome);
    battle.apply_actions(
        battle.row_actions[outcome.row_idx],
        battle.col_actions[outcome.col_idx]
    );
    battle.get_actions();
    Node* next_node = (*node)(outcome.row_idx, outcome.col_idx, battle.obs());

    outcome.value = run_iteration(next_node, battle, model);

    data.update(outcome);

    return outcome.value;
}

};
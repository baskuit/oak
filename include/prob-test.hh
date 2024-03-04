#pragma once

#include <pkmn.h>

/*

DefaultNodes<...>::ChanceNode happens to be a generic linked-list backed set impl.
we don't have to worry about hash collisions with std::unordered_map

*/

struct ProbTestMStats {
    bool seen = false;
};
struct ProbTestCStats {};

template <typename Types>
typename Types::Prob prob_test(typename Types::PRNG &device, size_t tries, const typename Types::State &state,
                               pkmn_choice row_action, pkmn_choice col_action) {
    using Nodes = DefaultNodes<Types, ProbTestMStats, ProbTestCStats>;
    using ChanceNode = Nodes::ChanceNode;

    typename Types::Prob total_prob{0};
    ChanceNode chance_node{};

    for (size_t t{}; t < tries; ++t) {
        auto state_copy{state};
        state_copy.randomize_transition(device);
        state_copy.apply_actions(row_action, col_action);

        auto *matrix_node = chance_node.access(state_copy.get_obs());
        if (matrix_node->stats.seen) {
            // do nothing
        } else {
            total_prob += state_copy.prob;
            canonicalize(total_prob);
            matrix_node->stats.seen = true;
        }
    }

    std::cout << "prob test count: " << chance_node.count_matrix_nodes() << std::endl;

    return total_prob;
}
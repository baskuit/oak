#pragma once

#include <pkmn.h>

#include <map.h>

// This is the tree structure that will be used from computing the 1v1 evals

// Instead of hashing bucketed states like in the TT, we have a 'perfect hashing'
// scheme using trees. This is because this search needs correctness over speed

namespace Tree {

template <typename NodeData>
struct Node {
    using Chance = std::map<pkmn_gen1_chance_actions, std::unique_ptr<Node>>;

    NodeData data;
    std::vector<Chance> chance_nodes;

    Node *at(auto p1_index, auto p2_index, auto obs) const {
        return nullptr;
    };

    Node *operator()(auto p1_index, auto p2_index, auto obs) {

    };
};
};
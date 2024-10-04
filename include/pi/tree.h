#pragma once

#include <assert.h>

#include <map>
#include <memory>

// This is the tree structure that will be used from computing the 1v1 evals

// Instead of hashing bucketed states like in the TT, we have a 'perfect
// hashing' scheme using trees. This is because this search needs correctness
// over speed

namespace Tree {

template <typename NodeData, typename Obs> class Node {
private:
  using Chance = std::map<std::tuple<uint8_t, uint8_t, Obs>,
                          std::unique_ptr<Node<NodeData, Obs>>>;
  NodeData _data;
  Chance _map;

public:
  const auto &data() const noexcept { return _data; }
  auto &data() noexcept { return _data; }

  void init(auto rows, auto cols) noexcept { _data.init(rows, cols); }

  bool is_init() const noexcept { return _data.is_init(); }

  Node *at(auto p1_index, auto p2_index, auto obs) const {
    return _map.at({p1_index, p2_index, obs}).get();
  };

  Node *operator()(auto p1_index, auto p2_index, auto obs) {
    auto &node = _map[{p1_index, p2_index, obs}];
    if (!node) {
      node = std::make_unique<Node<NodeData, Obs>>();
    }
    return node.get();
  };
};

}; // namespace Tree
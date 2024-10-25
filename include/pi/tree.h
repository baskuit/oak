#pragma once

#include <pkmn.h>

#include <assert.h>

#include <map>
#include <memory>

// This is the tree structure that will be used from computing the 1v1 evals

// Instead of hashing bucketed states like in the TT, we have a 'perfect
// hashing' scheme using trees. This is because this search needs correctness
// over speed

namespace Tree {

template <typename BanditData, typename Obs> class Node {
private:
  using ChanceMap = std::map<std::tuple<uint8_t, uint8_t, Obs>,
                          std::unique_ptr<Node<BanditData, Obs>>>;
  BanditData _data;
  ChanceMap _map;

public:
  const auto &data() const noexcept { return _data; }
  auto &data() noexcept { return _data; }

  void init(auto p1_choices, auto p2_choices) noexcept { _data.init(p1_choices, p2_choices); }

  bool is_init() const noexcept { return _data.is_init(); }

  Node *at(auto p1_index, auto p2_index, auto obs) const {
    return _map.at({p1_index, p2_index, obs}).get();
  };

  Node *operator()(auto p1_index, auto p2_index, auto obs) {
    auto &node = *_map[{p1_index, p2_index, obs}];
    if (!node) {
      node = std::make_unique<Node<BanditData, Obs>>();
    }
    return node.get();
  };
};

}; // namespace Tree
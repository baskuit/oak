#pragma once

#include <pkmn.h>

#include <assert.h>

#include <map>
#include <memory>

namespace Tree {

template <typename BanditData, typename Obs> class Node {
private:
  using ChanceMap = std::map<std::tuple<uint8_t, uint8_t, Obs>,
                             std::unique_ptr<Node<BanditData, Obs>>>;
  BanditData _data;
  ChanceMap _map;

public:
  const auto &stats() const noexcept { return _data; }
  auto &stats() noexcept { return _data; }

  void init(auto p1_choices, auto p2_choices) noexcept {
    _data.init(p1_choices, p2_choices);
  }

  bool is_init() const noexcept { return _data.is_init(); }


  Node *operator()(auto p1_index, auto p2_index, auto obs) {
    auto &node = _map[{p1_index, p2_index, obs}];
    if (!node) {
      node = std::make_unique<Node<BanditData, Obs>>();
    }
    return node.get();
  }

  Node *operator[](auto p1_index, auto p2_index, auto obs) {
    return _map[{p1_index, p2_index, obs}].get();
  }

  std::unique_ptr<Node> release_child(auto p1_index, auto p2_index, auto obs) {
    return std::move(_map[{p1_index, p2_index, obs}]);
  }
};

}; // namespace Tree
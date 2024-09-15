#pragma once

#include <pkmn.h>

#include "chance.h"
#include "util.h"

#include <vector>

template <typename Key, typename Value> struct LinearScanSet {
  std::vector<std::pair<Key, Value>> data;

  Value &operator[](const Key &key) {
    for (auto &pair : data) {
      if (pair.first == key) {
        return pair.second;
      }
    }
    data.emplace_back(key, Value{});
    return data.back().second;
  }

  const Value *at(const Key &key) const noexcept {
    for (const auto &pair : data) {
      if (pair.first == key) {
        return &pair.second;
      }
    }
    return nullptr;
  }
};

template <typename State, typename PRNG>
auto actions_test(PRNG &device, const State &state, pkmn_choice p1_action,
                  pkmn_choice p2_action, int tries) {
  using Data = std::tuple<size_t, float, float>;
  LinearScanSet<std::array<uint8_t, 16>, Data> map{};
  for (int i = 0; i < tries; ++i) {
    State state_copy{state};
    state_copy.randomize_transition(device);
    state_copy.apply_actions(p1_action, p2_action);
    const auto *bytes =
        pkmn_gen1_battle_options_chance_actions(&state_copy.options())->bytes;
    const auto &array =
        *reinterpret_cast<const std::array<uint8_t, 16> *>(bytes);

    auto &data = map[array];
    const auto p = state_copy.prob();

    if (std::get<0>(data) == 0) {
      std::get<1>(data) = p.first;
      std::get<2>(data) = p.second;
    } else {
      assert(std::get<1>(data) == p.first);
      assert(std::get<2>(data) == p.second);
    }
    ++std::get<0>(data);
  }
  return map;
}

void display_actions_test_map(const auto &map) {
  double total_prob = 0;
  for (const auto &pair : map.data) {
    const auto *buf = pair.first.data();
    const auto &data = pair.second;
    const size_t n = std::get<0>(data);
    const size_t p = std::get<1>(data);
    const size_t q = std::get<2>(data);
    const double x = p / (double)q;
    std::cout << buffer_to_string(buf, 16) << " : " << n << ", " << p << " / "
              << q << " = " << x << std::endl;
    total_prob += x;
  }
  std::cout << "number of branches: " << map.data.size() << std::endl;
  std::cout << "total observed prob: " << total_prob << std::endl;
}

template <typename PRNG, typename State>
void copy_rollout_test(PRNG &device, State &state) {
  State state_copy{state};
  while (!state.is_terminal()) {
    assert(!state_copy.is_terminal());
    const auto p1_action = state.row_actions[device.random_int(state.rows())];
    const auto p2_action = state.col_actions[device.random_int(state.cols())];
    state.apply_actions(p1_action, p2_action);
    state_copy.apply_actions(p1_action, p2_action);
    state.get_actions();
    state_copy.get_actions();
    assert(state.rows() == state_copy.rows());
    assert(state.cols() == state_copy.cols());
    for (int i = 0; i < state.rows(); ++i) {
      assert(state.row_actions[i] == state_copy.row_actions[i]);
    }
    for (int i = 0; i < state.cols(); ++i) {
      assert(state.col_actions[i] == state_copy.col_actions[i]);
    }
    assert(state.obs() == state_copy.obs());
  }
  assert(state_copy.is_terminal());
  assert(state.payoff() == state_copy.payoff());
}

#pragma once

#include <array>
#include <numeric>
#include <cmath>

float sigmoid (auto x) {
  return 1 / (1 + std::exp(-static_cast<float>(x)));
}

class PGame {
public:
  int score = 0;
  unsigned int depth;
  std::array<int, 4> row_actions;
  std::array<int, 4> col_actions;

  PGame() = default;

  PGame(auto depth) : depth{static_cast<unsigned int>(depth)} {
    std::iota(row_actions.begin(), row_actions.end(), 0);
    std::iota(col_actions.begin(), col_actions.end(), 0);
  }
  void apply_actions(int row_action, int col_action) noexcept {
    score += row_action - col_action + 1;
    --depth;
  }
  void get_actions() const noexcept {}
  auto rows() const noexcept { return 4; }
  auto cols() const noexcept { return 4; }
  int obs() const noexcept { return 1; }
  bool terminal() const noexcept { return depth == 0; }
  float payoff() const noexcept { return sigmoid(score); }
};

class PGameModel {
public:
  float inference(PGame &&state) { return .5; }
};
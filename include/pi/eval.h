#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <pi/abstract.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <numeric>
#include <stddef.h>
#include <vector>

namespace Eval {

// The eval currently only uses hp and status information to bucket states.
// pp,
constexpr size_t n_hp = 4;
constexpr size_t n_status = 16;

struct Pokemon {
  Data::Species species;
  std::array<Data::Moves, 4> moves;
  float hp = 1.0;
};

// basic mcts util that conducts a search to populate the Mono E Mono data
float get_value(const auto &p1, const auto &p2) { return 0; }

// expected values is [0, 1]
float from_matrix(const auto &expected_values, const auto m, const auto n) {
  std::vector<float> p1_material;
  std::vector<float> p2_material;
  p1_material.resize(m);
  p2_material.resize(n);

  const auto sigmoid = [](float x) { return 1 / (1 + std::exp(-x)); };
  const auto inv_sigmoid = [](float y) { return -std::log((1 / y) - 1); };

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto p = expected_values[i][j];
      const auto logit = inv_sigmoid(p);
      constexpr auto bound = 1000;
      // for stability
      const auto clamped = std::max(std::min(logit, bound), -bound);

      p1_material[i] += clamped / n;
      p2_material[j] -= clamped / m;
    }
  }

  // we could also e.g. give extra weight to the actives
  const float p1_sum =
      std::accumulate(p1_material.begin(), p1_material.end(), 0);
  const float p2_sum =
      std::accumulate(p2_material.begin(), p2_material.end(), 0);
  const float material_difference = (p1_sum - p2_sum) / 2;
  return sigmoid(material_difference);
}

void from_matrix_test() {}

using MEM = float[n_hp][n_status][n_hp][n_status];

class CachedEval {
public:
  static constexpr auto status_index(Status status) noexcept { return 0; }

  std::array<std::array<MEM, 6>, 6> mem_matrix;
  std::array<std::array<float, 6>, 6> value_matrix;

  float value() const {}

  CachedEval() = default;

  CachedEval(const pkmn_gen1_battle &battle) {}

  CachedEval(const pkmn_gen1_battle &battle, const auto &global) {}
};

struct GlobalMEM {};

}; // namespace Eval

#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <pi/abstract.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <vector>

namespace Eval {

// The eval currently only uses hp and status information to bucket states.
// pp,
constexpr size_t n_hp = 3;
constexpr size_t n_status = 1;

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

using MEM = std::array<std::array<float, n_hp>, n_hp>;

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

class GlobalMEM {
public:
  struct Set {
    Data::Species species;
    std::array<Data::Moves, 4> moves;
    int level;
  };

  using SetID = uint32_t;

  constexpr MEM switch_sides(const MEM &mem) const noexcept {
    MEM switched{};
    for (int i = 0; i < n_hp; ++i) {
      for (int j = 0; j < n_hp; ++j) {
        switched[i][j] = 1 - mem[j][i];
      }
    }
    return switched;
  }

  constexpr SetID toID(const auto &set) const noexcept {
    assert(set.species != Data::Species::None);
    std::array<Data::Moves, 4> ordered_moves{};
    std::copy(set.moves.begin(), set.moves.begin() + 4, ordered_moves.begin());
    std::sort(ordered_moves.begin(), ordered_moves.end(),
              std::greater<Data::Moves>());
    uint32_t id = (static_cast<uint32_t>(set.species) - 1);
    for (int i = 0; i < 4; ++i) {
      id *= 166;
      id += static_cast<uint32_t>(ordered_moves[0]);
    }
    id *= 100;
    return id;
  }

  const MEM &operator()(const auto &p1_set, const auto &p2_set) const {
    SetID id1 = toID(p1_set);
    SetID id2 = toID(p2_set);
    // if (id1 > id2) {

    // } else {

    // }
    // return MEMData.at({});
  }

  bool save(const std::filesystem::path path) const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return false;
    }

    for (const auto &[key, value] : MEMData) {
      file.write(std::bit_cast<const char *>(&key), sizeof(key));
      file.write(std::bit_cast<const char *>(value.data()), sizeof(value));
    }
    file.close();

    return true;
  }

  bool load(const std::filesystem::path path) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    file.seekg(0, std::ios::beg);
    while (file) {
      std::pair<SetID, SetID> key;
      MEM value;

      file.read(std::bit_cast<char *>(&key), sizeof(key));
      file.read(std::bit_cast<char *>(value.data()), sizeof(value));

      if (file.gcount() == sizeof(key) + sizeof(value)) {
        MEMData[key] = value;
      } else {
        return false;
      }
    }
    file.close();
    return true;
  }

private:
  std::map<std::pair<SetID, SetID>, MEM> MEMData;
};

}; // namespace Eval

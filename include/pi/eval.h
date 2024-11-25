#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <battle/init.h>
#include <pi/abstract.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>
#include <util/random.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <type_traits>
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

struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  Abstract::Battle abstract;
  pkmn_result result;
};

// basic mcts util that conducts a search to populate the Mono E Mono data
float get_value(const auto &set1, const auto &set2, size_t iterations,
                auto seed) {
  std::array<std::remove_reference_t<decltype(set1)>, 1> p1{set1};
  std::array<std::remove_reference_t<decltype(set2)>, 1> p2{set2};
  prng device{seed};
  MonteCarlo::Model model{device.uniform_64()};
  MonteCarlo::Input input;
  input.battle = Init::battle(p1, p2, device.uniform_64());
  using Node =
      Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;
  Node node{};
  MCTS search;
  input.result = Init::update(input.battle, 0, 0, search.options);
  pkmn_gen1_chance_durations durations{};
  const auto output = search.run(iterations, node, input, model);
  return output.average_value;
}

using MEM = std::array<std::array<float, n_hp>, n_hp>;

void print_mem(MEM &mem) {
  for (int i = 0; i < n_hp; ++i) {
    for (int j = 0; j < n_hp; ++j) {
      std::cout << mem[i][j] << '\t';
    }
    std::cout << std::endl;
  }
}

MEM compute_table(auto set1, auto set2, const auto seed,
                  const auto iterations) {
  MEM result;
  for (int hp1 = 1; hp1 <= 3; ++hp1) {
    for (int hp2 = 1; hp2 <= 3; ++hp2) {
      set1.hp = hp1 / 3.0;
      set2.hp = hp2 / 3.0;
      result[hp1 - 1][hp2 - 1] = get_value(set1, set2, iterations, seed);
    }
  }
  return result;
}

class GlobalMEM {
public:
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

  void add_matchups(const auto &p1, const auto &p2) {
    for (const auto &set1 : p1) {
      for (const auto &set2 : p2) {
        (*this)(set1, set2);
      }
    }
  }

  MEM operator()(const auto &p1_set, const auto &p2_set) {
    SetID id1 = toID(p1_set);
    SetID id2 = toID(p2_set);

    if (MEMData.contains({id1, id2})) {
      return MEMData[{id1, id2}];
    } else if (MEMData.contains({id2, id1})) {
      return switch_sides(MEMData[{id2, id1}]);
    } else {
      return MEMData[{id1, id2}] =
                 compute_table(p1_set, p2_set, device.uniform_64(), 1 << 16);
    }
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
      std::cout << "cant open file" << std::endl;
      return false;
    }

    file.seekg(0, std::ios::beg);
    while (file.peek() != EOF) {
      std::pair<SetID, SetID> key;
      MEM value;

      file.read(std::bit_cast<char *>(&key), sizeof(key));
      if (const auto g = file.gcount(); g != sizeof(key)) {
        std::cout << "cant read key: " << gamma << std::endl;
        return false;
      }
      file.read(std::bit_cast<char *>(value.data()), sizeof(value));
      if (const auto g = file.gcount(); g != sizeof(value)) {
        std::cout << "cant read value: " << g << std::endl;
        return false;
      }
      MEMData[key] = value;
    }

    file.close();
    return true;
  }

private:
  std::map<std::pair<SetID, SetID>, MEM> MEMData{};
  prng device{9348509345830};
  std::mutex mutex{};
};

class CachedEval {
public:
  static constexpr auto status_index(Abstract::Status status) noexcept {
    return 0;
  }

  std::array<std::array<MEM, 6>, 6> mem_matrix;

  float value() const { return 0; }

  CachedEval(const auto &p1, const auto &p2, GlobalMEM &global) {
    global.add_matchups(p1, p2);
    const auto m = p1.size();
    const auto n = p2.size();
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        mem_matrix[i][j] = global(p1[i], p2[j]);
      }
    }
  }

  float value(const Abstract::Battle &battle) {

    static constexpr int HP_Numerators[] = {
        0, // KO (not used, but added for completeness)
        1, // ONE
        1, // TWO
        2, // THREE
        2, // FOUR
        2, // FIVE
        3, // SIX
        3  // SEVEN
    };

    const auto sigmoid = [](float x) { return 1 / (1 + std::exp(-x)); };
    const auto inv_sigmoid = [](float y) { return -std::log((1 / y) - 1); };
    const auto not_ko = [](const auto &elem) {
      return elem.hp != Abstract::HP::KO;
    };

    float m1 = 0;
    float m2 = 0;
    const auto b1 = battle.sides[0].bench;
    const auto b2 = battle.sides[1].bench;

    size_t m = std::count_if(b1.begin(), b1.end(), not_ko);
    size_t n = std::count_if(b2.begin(), b2.end(), not_ko);

    for (int i = 0; i < 6; ++i) {
      if (b1[i].hp == Abstract::HP::KO) {
        continue;
      }
      for (int j = 0; j < 6; ++j) {
        if (b2[j].hp == Abstract::HP::KO) {
          continue;
        }
        const auto &mem = mem_matrix[i][j];
        const auto v = mem[HP_Numerators[static_cast<uint8_t>(b1[i].hp)]]
                          [HP_Numerators[static_cast<uint8_t>(b2[j].hp)]];
        // const auto logit = inv_sigmoid(v);
        // constexpr float bound = 3;
        // // for stability
        // const auto clamped = std::max(std::min(logit, bound), -bound);

        // std::cout << v << '/' << clamped << ' ';

        m1 += v / n;
        m2 += (1 - v) / m;
      }
      // std::cout << std::endl;
      // std::cout << "m1: " << m1 << " m2: " << m2 << std::endl;
    }
    return sigmoid((m1 - m2) / (2));
  }

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
};

struct Model {
  prng device;
  Eval::CachedEval eval;
};

} // namespace Eval

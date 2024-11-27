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

const auto sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

namespace Eval {

constexpr size_t n_hp = 3;
constexpr size_t n_status = 5;

struct Input {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  Abstract::Battle abstract;
  pkmn_result result;
};

float get_value(const auto &set1, const auto &set2, size_t iterations,
                auto seed) {
  std::array<std::remove_reference_t<decltype(set1)>, 1> p1{set1};
  std::array<std::remove_reference_t<decltype(set2)>, 1> p2{set2};
  prng device{seed};
  MonteCarlo::Model model{device.uniform_64()};
  MonteCarlo::Input input{};
  input.battle = Init::battle(p1, p2, device.uniform_64());
  using Node =
      Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;
  Node node{};
  MCTS search;
  input.result = Init::update(input.battle, 0, 0, search.options);
  // This rolls for sleep turns at the start of each MCTS iteration
  if (set1.status & 7) {
    input.durations.bytes[0] = 1;
  }
  if (set2.status & 7) {
    input.durations.bytes[4] = 1;
  }
  const auto output = search.run(iterations, node, input, model);
  return output.average_value;
}

using OVO = std::array<
    std::array<std::array<std::array<float, n_status>, n_hp>, n_status>, n_hp>;

struct OVO2 {
  std::array<
      std::array<std::array<std::array<float, n_status>, n_hp>, n_status>, n_hp>
      data;

  const float &operator()(auto h1, auto s1, auto h2, auto s2) const {
    return data[0][0][0][0];
  }
  float &operator()(auto h1, auto s1, auto h2, auto s2) {
    return data[0][0][0][0];
  }
};

OVO compute_table(auto set1, auto set2, const auto iterations,
                  const auto seed) {
  OVO result;
  // clr, slp, psn, brn, par
  std::array<uint8_t, 6> STATUS{0b00000000, 0b00000100, 0b00001000, 0b00010000,
                                0b01000000};
  for (int h1 = 0; h1 < n_hp; ++h1) {
    for (int s1 = 0; s1 < n_status; ++s1) {
      for (int h2 = 0; h2 < n_hp; ++h2) {
        for (int s2 = 0; s2 < n_status; ++s2) {
          set1.hp = (h1 + 1) / 3.0;
          set2.hp = (h2 + 1) / 3.0;
          set1.status = STATUS[s1];
          set2.status = STATUS[s2];
          result[h1][s1][h2][s2] = get_value(set1, set2, iterations, seed);
        }
      }
    }
  }
  return result;
}

constexpr OVO ovo_mirror(const OVO &ovo) noexcept {
  OVO mirrored{};
  for (int h1 = 0; h1 < n_hp; ++h1) {
    for (int s1 = 0; s1 < n_status; ++s1) {
      for (int h2 = 0; h2 < n_hp; ++h2) {
        for (int s2 = 0; s2 < n_status; ++s2) {
          mirrored[h1][s1][h2][s2] = 1 - ovo[h2][s2][h1][s1];
        }
      }
    }
  }
  return mirrored;
}

class OVODict {
public:
  using SetID = uint64_t;
  static constexpr auto n_moves_with_none = 166;

  static constexpr SetID toID(const auto &set) noexcept {
    assert(set.species != Data::Species::None);
    std::array<Data::Moves, 4> ordered_moves{};
    std::copy(set.moves.begin(), set.moves.begin() + 4, ordered_moves.begin());
    std::sort(ordered_moves.begin(), ordered_moves.end(),
              std::greater<Data::Moves>());
    SetID id = (static_cast<SetID>(set.species) - 1);
    for (int i = 0; i < 4; ++i) {
      id *= n_moves_with_none;
      id += static_cast<SetID>(ordered_moves[i]);
    }
    return id;
  }

  static constexpr SampleTeams::Set fromID(SetID id) noexcept {
    SampleTeams::Set set{};
    for (int i = 0; i < 4; ++i) {
      const uint8_t moveid = id % n_moves_with_none;
      id -= moveid;
      id /= n_moves_with_none;
      set.moves[3 - i] = static_cast<Data::Moves>(moveid);
    }
    set.species = static_cast<Data::Species>(id + 1);
    assert(id < 151);
    return set;
  }

  void add_matchups(const auto &p1, const auto &p2) {
    for (const auto &set1 : p1) {
      for (const auto &set2 : p2) {
        get(set1, set2);
      }
    }
  }

  OVO get(const auto &p1_set, const auto &p2_set) {
    SetID id1 = toID(p1_set);
    SetID id2 = toID(p2_set);

    if (OVOData.contains({id1, id2})) {
      return OVOData[{id1, id2}];
    } else if (OVOData.contains({id2, id1})) {
      return ovo_mirror(OVOData[{id2, id1}]);
    } else {
      return OVOData[{id1, id2}] =
                 compute_table(p1_set, p2_set, iterations, device.uniform_64());
    }
  }

  bool save(const std::filesystem::path path) {
    std::unique_lock lock{mutex};
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return false;
    }

    for (const auto &[key, value] : OVOData) {
      file.write(std::bit_cast<const char *>(&key), sizeof(key));
      file.write(std::bit_cast<const char *>(value.data()), sizeof(value));
      static_assert(sizeof(value) == 9 * 25 * 4);
      static_assert(sizeof(key) == 16);
    }
    file.close();

    return true;
  }

  bool load(const std::filesystem::path path) {
    std::unique_lock lock{mutex};
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::cout << "cant open file" << std::endl;
      return false;
    }

    file.seekg(0, std::ios::beg);
    while (file.peek() != EOF) {
      std::pair<SetID, SetID> key;
      OVO value;

      file.read(std::bit_cast<char *>(&key), sizeof(key));
      if (const auto g = file.gcount(); g != sizeof(key)) {
        std::cout << "cant read key: " << g << std::endl;
        return false;
      }
      file.read(std::bit_cast<char *>(value.data()), sizeof(value));
      if (const auto g = file.gcount(); g != sizeof(value)) {
        std::cout << "cant read value: " << g << std::endl;
        return false;
      }
      OVOData[key] = value;
    }

    file.close();
    return true;
  }

  void print() const {
    for (const auto pair : OVOData) {
      const auto set1 = fromID(pair.first.first);
      const auto set2 = fromID(pair.first.second);
      std::cout << set_string(set1) << " : " << set_string(set2) << std::endl;
      std::cout << pair.second[2][0][2][0] << std::endl;
    }
  }

  size_t iterations = 1 << 18;
  std::map<std::pair<SetID, SetID>, OVO> OVOData{};

private:
  prng device{9348509345830};
  std::mutex mutex{};

  static_assert(sizeof(decltype(*OVOData.begin())) == 920);
};

static_assert(
    OVODict::toID(SampleTeams::teams[0][0]) ==
    OVODict::toID(OVODict::fromID(OVODict::toID(SampleTeams::teams[0][0]))));

class CachedEval {
public:
  std::array<std::array<OVO, 6>, 6> ovo_matrix;
  std::array<std::array<float, 6>, 6> value_matrix;
  std::array<float, 6> pieces1{};
  std::array<float, 6> pieces2{};

  CachedEval(const auto &p1, const auto &p2, OVODict &global) {
    global.add_matchups(p1, p2);
    const auto m = p1.size();
    const auto n = p2.size();
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        ovo_matrix[i][j] = global.get(p1[i], p2[j]);
        value_matrix[i][j] = ovo_matrix[i][j][2][0][2][0];
      }
    }
  }

  float value(const pkmn_gen1_battle &battle) {
    const auto &b = View::ref(battle);

    const auto alive_and_unfrozen = [](const View::Pokemon &p) -> bool {
      return !(p.status() == Data::Status::Freeze && p.hp() == 0);
    };

    int m = 0;
    int n = 0;
    for (int i = 0; i < 6; ++i) {
      m += (b.side(0).pokemon(i).hp() != 0);
      n += (b.side(1).pokemon(i).hp() != 0);
    }

    float m1 = 0;
    float m2 = 0;

    for (int i = 0; i < 6; ++i) {
      const auto &p1 = b.side(0).pokemon(i);
      if (!p1.hp()) {
        continue;
      }
      for (int j = 0; j < 6; ++j) {
        const auto &p2 = b.side(1).pokemon(j);
        if (!p2.hp()) {
          continue;
        }

        if (p1.status() == Data::Status::Freeze) {
          if (p2.status() != Data::Status::Freeze) {
            m2 += 1;
            continue;
          }
        }
        if (p2.status() == Data::Status::Freeze) {
          m1 += 1;
          continue;
        }

        const int h1 = std::ceil(3.0f * p1.hp() / p1.stats().hp()) - 1;
        const int h2 = std::ceil(3.0f * p2.hp() / p2.stats().hp()) - 1;
        const auto s1 =
            static_cast<uint8_t>(Abstract::simplify_status(p1.status()));
        const auto s2 =
            static_cast<uint8_t>(Abstract::simplify_status(p2.status()));

        assert((h1 >= 0) && (h1 <= 2));
        assert((h2 >= 0) && (h2 <= 2));
        assert((s1 >= 0) && (s1 <= 4));
        assert((s2 >= 0) && (s2 <= 4));
        const auto v = ovo_matrix[i][j][h1][s1][h2][s2];
        m1 += v;
        m2 += 1 - v;
      }
    }

    m1 /= n;
    m2 /= m;
    float x = sigmoid(2 * (m1 - m2));
    return x;
  }

  float value(const Abstract::Battle &battle) {

    pieces1 = {};
    pieces2 = {};
    // for (int i = 0; i < 6; ++i) {
    //   for (int j = 0; j < 6; ++j) {
    //     value_matrix[i][j] = -1;
    //   }
    // }

    for (int i = 0; i < 6; ++i) {
      const auto &p1 = battle.sides[0].bench[i];
      if (p1.hp == Abstract::HP::HP0) {
        continue;
      }
      for (int j = 0; j < 6; ++j) {
        const auto &p2 = battle.sides[1].bench[j];
        if (p2.hp == Abstract::HP::HP0) {
          continue;
        }

        if (p1.status == Abstract::Status::Freeze) {
          if (p2.status != Abstract::Status::Freeze) {
            pieces1[i] += .9;
            pieces2[j] += .1;
            continue;
          } else {
            pieces1[i] += .5;
            pieces2[j] += .5;
          }
        }
        if (p2.status == Abstract::Status::Freeze) {
          pieces1[i] += .1;
          pieces2[j] += .9;
          continue;
        }

        const auto &ovo = ovo_matrix[i][j];
        const auto value =
            ovo[static_cast<uint8_t>(p1.hp) - 1][static_cast<uint8_t>(
                p1.status)][static_cast<uint8_t>(p2.hp) - 1]
               [static_cast<uint8_t>(p2.status)];
        // value_matrix[i][j] = value;
        pieces1[i] += value;
        pieces2[j] += 1 - value;
      }
    }

    // for (int i = 0; i < 6; ++i) {
    //   for (int j = 0; j < 6; ++j) {
    //     std::cout << value_matrix[i][j] << ' ';
    //   }
    //   std::cout << std::endl;
    // }

    const auto a = battle.sides[0].active.n_alive;
    const auto b = battle.sides[1].active.n_alive;

    const auto m1 = std::accumulate(pieces1.begin(), pieces1.end(), 0.0f);
    const auto m2 = std::accumulate(pieces2.begin(), pieces2.end(), 0.0f);
    const auto v = sigmoid(2 * (m1 / b - m2 / a));
    // std::cout << a << " " << b << std::endl;
    // std::cout << m1 << " : " << m2 << " = " << v << std::endl;
    return v;
  }

  void update(const Abstract::Battle &battle) {
    {
      const auto slot = battle.sides[0].active.slot;
      const auto &p1 = battle.sides[0].bench[slot];
      if (p1.alive_and_unfrozen()) {
        pieces1[slot] = 0;
        for (auto i = 0; i < 6; ++i) {
          const auto &p2 = battle.sides[1].bench[i];
          if (p2.alive_and_unfrozen()) {
            pieces2[i] -= 1 - value_matrix[slot][i];
            value_matrix[slot][i] =
                ovo_matrix[slot][i][static_cast<uint8_t>(p1.hp) - 1]
                          [static_cast<uint8_t>(p1.status)]
                          [static_cast<uint8_t>(p2.hp) - 1]
                          [static_cast<uint8_t>(p2.status)];
            pieces1[slot] += value_matrix[slot][i];
            pieces2[i] += 1 - value_matrix[slot][i];
          }
        }
      }
    }
    {
      const auto slot = battle.sides[1].active.slot;
      const auto &p2 = battle.sides[1].bench[slot];
      if (p2.alive_and_unfrozen()) {
        pieces2[slot] = 0;
        for (auto i = 0; i < 6; ++i) {
          const auto &p1 = battle.sides[0].bench[i];
          if (p1.alive_and_unfrozen()) {
            pieces1[i] -= value_matrix[slot][i];
            value_matrix[i][slot] =
                ovo_matrix[i][slot][static_cast<uint8_t>(p1.hp) - 1]
                          [static_cast<uint8_t>(p1.status)]
                          [static_cast<uint8_t>(p2.hp) - 1]
                          [static_cast<uint8_t>(p2.status)];
            pieces2[slot] += 1 - value_matrix[i][slot];
            pieces1[i] += value_matrix[slot][i];
          }
        }
      }
    }
  }
};

// float value(const Abstract::Battle &battle) {
//   float m1 = 0;
//   float m2 = 0;

//   for (auto i = 0; i < 6; ++i) {
//     const auto &p1 = battle.sides[0].bench[i];
//     if (p1.alive_and_unfrozen()) {
//       m1 += pieces1[i];
//     }
//     const auto &p2 = battle.sides[1].bench[i];
//     if (p2.alive_and_unfrozen()) {
//       m2 += pieces2[i];
//     }
//   }

//   const auto a = battle.sides[0].active.n_alive;
//   const auto b = battle.sides[1].active.n_alive;
//   const auto v = sigmoid(2 * (m1 / b - m2 / a));
//   // std::cout << a << " " << b << std::endl;
//   // std::cout << m1 << " : " << m2 << " = " << v << std::endl;
//   return v;
// }

struct Model {
  prng device;
  Eval::CachedEval eval;
};

} // namespace Eval

namespace PokeEngine {
constexpr float POKEMON_ALIVE = 30.0;
constexpr float POKEMON_HP = 100.0;

constexpr float POKEMON_ATTACK_BOOST = 30.0;
constexpr float POKEMON_DEFENSE_BOOST = 15.0;
constexpr float POKEMON_SPECIAL_ATTACK_BOOST = 30.0;
constexpr float POKEMON_SPECIAL_DEFENSE_BOOST = 15.0;
constexpr float POKEMON_SPEED_BOOST = 30.0;

constexpr float POKEMON_BOOST_MULTIPLIER_6 = 3.3;
constexpr float POKEMON_BOOST_MULTIPLIER_5 = 3.15;
constexpr float POKEMON_BOOST_MULTIPLIER_4 = 3.0;
constexpr float POKEMON_BOOST_MULTIPLIER_3 = 2.5;
constexpr float POKEMON_BOOST_MULTIPLIER_2 = 2.0;
constexpr float POKEMON_BOOST_MULTIPLIER_1 = 1.0;
constexpr float POKEMON_BOOST_MULTIPLIER_0 = 0.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15;
constexpr float POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3;

constexpr float POKEMON_FROZEN = -40.0;
constexpr float POKEMON_ASLEEP = -25.0;
constexpr float POKEMON_PARALYZED = -25.0;
constexpr float POKEMON_TOXIC = -30.0;
constexpr float POKEMON_POISONED = -10.0;
constexpr float POKEMON_BURNED = -25.0;

constexpr float LEECH_SEED = -30.0;
constexpr float SUBSTITUTE = 40.0;
constexpr float CONFUSION = -20.0;

constexpr float REFLECT = 20.0;
constexpr float LIGHT_SCREEN = 20.0;
constexpr float STICKY_WEB = -25.0;
constexpr float AURORA_VEIL = 40.0;
constexpr float SAFE_GUARD = 5.0;
constexpr float TAILWIND = 7.0;

constexpr float STEALTH_ROCK = -10.0;
constexpr float SPIKES = -7.0;
constexpr float TOXIC_SPIKES = -7.0;

float eval_burned(const uint8_t *pokemon) {
  float multiplier = 0;
  for (int m = 0; m < 4; ++m) {
    auto moveid = pokemon[2 * m + Offsets::moves];
    const auto &move_data = Data::get_move_data(moveid);
    if (move_data.bp > 0 && Data::is_physical(move_data.type)) {
      ++multiplier;
    }
  }
  // don't make burn as punishing for special attackers
  return multiplier * POKEMON_BURNED;
}

float get_boost_multiplier(const uint8_t stage) {
  switch (stage) {
  case 0b00000000:
    return POKEMON_BOOST_MULTIPLIER_0;
  case 0b00001001:
    return POKEMON_BOOST_MULTIPLIER_NEG_1;
  case 0b00000001:
    return POKEMON_BOOST_MULTIPLIER_1;
  case 0b00001010:
    return POKEMON_BOOST_MULTIPLIER_NEG_2;
  case 0b00000010:
    return POKEMON_BOOST_MULTIPLIER_2;
  case 0b00001011:
    return POKEMON_BOOST_MULTIPLIER_NEG_3;
  case 0b10000011:
    return POKEMON_BOOST_MULTIPLIER_3;
  case 0b00001100:
    return POKEMON_BOOST_MULTIPLIER_NEG_4;
  case 0b00000100:
    return POKEMON_BOOST_MULTIPLIER_4;
  case 0b00001101:
    return POKEMON_BOOST_MULTIPLIER_NEG_5;
  case 0b00000101:
    return POKEMON_BOOST_MULTIPLIER_5;
  case 0b00001110:
    return POKEMON_BOOST_MULTIPLIER_NEG_6;
  case 0b00000110:
    return POKEMON_BOOST_MULTIPLIER_6;
  }
  return 0;
}

float evaluate_status(const uint8_t *pokemon, const uint8_t byte) {
  if (byte & 7) {
    return POKEMON_ASLEEP;
  }
  switch (byte) {
  case 0b00000000:
    return 0;
  case 0b00001000:
    return POKEMON_POISONED;
  case 0b00010000:
    return eval_burned(pokemon);
  case 0b00100000:
    return POKEMON_FROZEN;
  case 0b01000000:
    return POKEMON_PARALYZED;
  case 0b10001000:
    return POKEMON_TOXIC;
  }
  return 0;
}

float evaluate_pokemon(const uint8_t *data) {
  float score = POKEMON_ALIVE;
  const auto u16 = std::bit_cast<const uint16_t *>(data);
  score += POKEMON_HP * (float)u16[9] / u16[0];
  score += evaluate_status(data, data[Offsets::status]);
  return score;
}

float evaluate_side(const uint8_t *data) {
  float score;
  for (int p = 0; p < 6; ++p) {
    const uint8_t *pokemon = data + Offsets::pokemon * p;
    const auto u16 = std::bit_cast<const uint16_t *>(pokemon);
    if (u16[9]) {
      score += evaluate_pokemon(pokemon);
    }
  }

  // TODO bad casts
  const auto active = data + Offsets::active;
  score += get_boost_multiplier(active[12] & 15) * POKEMON_ATTACK_BOOST;
  score += get_boost_multiplier(active[12] >> 4) * POKEMON_DEFENSE_BOOST;
  score += get_boost_multiplier(active[13] & 15) * POKEMON_SPEED_BOOST;
  score += get_boost_multiplier(active[13] >> 4) * POKEMON_SPECIAL_ATTACK_BOOST;

  const auto volatiles = active + 16;
  const bool sub = volatiles[5] > 0;
  const bool light_screen = active[1] & 128;
  const bool reflect = active[2] & 1;

  score += sub * SUBSTITUTE;
  score += light_screen * LIGHT_SCREEN;
  score += reflect * REFLECT;
  return score;
}

float evaluate_battle(const pkmn_gen1_battle &battle) {
  float p1_score = evaluate_side(battle.bytes);
  float p2_score = evaluate_side(battle.bytes + Offsets::side);
  return sigmoid(p1_score - p2_score);
}

} // namespace PokeEngine

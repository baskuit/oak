#pragma once

#include <array>

#include "util.h"
#include <bit>

#include <pkmn.h>

// turn 0 base stats. We bucket the active pokemons' stats using the log of the
// ration between the current and base stats
struct BaseHashData {
  std::array<std::array<uint16_t, 4>, 6> side0_stats;
  std::array<std::array<uint16_t, 4>, 6> side1_stats;

  BaseHashData(const pkmn_gen1_battle *const battle) noexcept {
    const auto set_poke = [](const uint8_t *pokemon,
                             std::array<uint16_t, 4> &p) {
      const auto stats = std::bit_cast<const uint16_t *>(pokemon + 2);
      std::memcpy(p.data(), stats, 4 * 2);
    };
    for (int i = 0; i < 6; ++i) {
      set_poke(battle->bytes + (24 * i), side0_stats[i]);
      set_poke(battle->bytes + (24 * i + 184), side1_stats[i]);
    }
  }

  void print() const noexcept {
    for (const auto &stats : side1_stats) {
      for (int i = 0; i < 4; ++i) {
        std::cout << stats[i] << ' ';
      }
      std::cout << std::endl;
    }
  }
};

uint64_t compute_hash(const pkmn_gen1_battle *const battle,
                      const pkmn_gen1_chance_actions *const actions,
                      const BaseHashData &base_hash_data,
                      const uint64_t *const z) {
  // missing durations
  struct P {
    float hp;
    uint8_t hp_discrete;
    uint8_t status_id;
    uint8_t sleep_duration;
    std::array<bool, 4> no_pp;

    // 30
    uint64_t get_hash(const uint64_t *const z) const {
      auto index = 0;
      auto result = 0;
      result ^= z[index + hp_discrete];
      index += 16;
      result ^= z[index + status_id - 2];
      index += 6;
      result ^= z[index + no_pp[0]];
      index += 2;
      result ^= z[index + no_pp[1]];
      index += 2;
      result ^= z[index + no_pp[2]];
      index += 2;
      result ^= z[index + no_pp[3]];
      return result;
    }
  };

  struct A {
    std::array<uint16_t, 4> stats;
    std::array<int8_t, 4> boosts;
    struct Durations {};
    Durations durations;

    uint64_t get_hash(const uint64_t *const z) const {
      auto index = 0;
      auto result = 0;

      result ^= z[index + boosts[0]];
      index += 2;
      result ^= z[index + boosts[1]];
      index += 2;
      result ^= z[index + boosts[2]];
      index += 2;
      result ^= z[index + boosts[3]];
      return result;
    }
  };

  A a0;
  A a1;
  std::array<P, 6> side0{};
  std::array<P, 6> side1{};

  const auto &base_stats0 =
      base_hash_data.side0_stats.at(order_bits(battle->bytes)[0] - 1);
  const auto &base_stats1 =
      base_hash_data.side1_stats.at(order_bits(battle->bytes + 184)[0] - 1);

  const auto set_active = [](const auto &stats, const auto *active, A &a) {
    const auto active_stats = std::bit_cast<const uint16_t *>(active + 2);

    for (int i = 0; i < 4; ++i) {
      a.stats[i] = active_stats[i];
      float log_ratio = log(a.stats[i] / (float)stats[i]) / 1.35 * 2;
      a.boosts[i] = 4 - static_cast<int>(log_ratio);
      // std::cout << "( " << log_ratio << ", " << (int)a.boosts[i] << ")\t";
      // std::cout << "( " << log_ratio << ")\t";
    }
    // std::cout << std::endl;
  };

  set_active(base_stats0, battle->bytes + 144, a0);
  set_active(base_stats1, battle->bytes + 144 + 184, a1);

  uint64_t hash = 0;
  auto index = 0;

  for (int i = 0; i < 6; ++i) {
    P &p0 = side0[i];
    P &p1 = side1[i];

    const uint8_t *pokemon0 = battle->bytes + (24 * i);
    const uint8_t *pokemon1 = battle->bytes + (24 * i + 184);

    const auto set_poke = [&base_hash_data, i](const uint8_t *pokemon, P &p) {
      float hp = 255 * pokemon[19] + pokemon[18];

      float max_hp = 255 * pokemon[1] + pokemon[0];

      p.hp = hp / max_hp;
      p.hp_discrete = static_cast<uint8_t>(p.hp * 15);
      p.status_id = pokemon[20] >> 3;
      p.sleep_duration = 0;
      for (int m = 0; m < 4; ++m) {
        p.no_pp[m] = (pokemon[11 + 2 * m] == 0);
      }

      // std::cout << "hp: " << p.hp << " disc: " << (int)p.hp_discrete
      //           << std::endl;
      // std::cout << "status: " << (int)p.status_id << std::endl;
      // std::cout << p.no_pp[0] << ' ' << p.no_pp[1] << ' ' << p.no_pp[2] << '
      // '
      //           << p.no_pp[3] << std::endl;
    };
    set_poke(pokemon0, p0);
    set_poke(pokemon1, p1);

    hash ^= p0.get_hash(z + index);
    index += 30;
    hash ^= p1.get_hash(z + index);
    index += 30;

    hash ^= z[index + order_bits(battle->bytes)[0] - 1];
    index += 6;
    hash ^= z[index + order_bits(battle->bytes + 184)[0] - 1];
    index += 6;
  }

  return hash;
}
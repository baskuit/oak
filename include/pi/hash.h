#pragma once

#include <array>
#include <bits>

#include <pkmn.h>

// turn 0 base stats. We bucket the active pokemons' stats using the log of the
// ratio between the current and base stats
struct BaseStatData {
  std::array<std::array<uint16_t, 4>, 6> p1;
  std::array<std::array<uint16_t, 4>, 6> p2;

  constexpr BaseStatData(const pkmn_gen1_battle *const battle) noexcept {
    const auto set_poke = [](const uint8_t *pokemon,
                             std::array<uint16_t, 4> &p) {
      const auto stats = std::bit_cast<const uint16_t *>(pokemon + 2);
      std::memcpy(p.data(), stats, 4 * 2);
    };
    for (int p = 0; p < 6; ++p) {
      set_poke(battle->bytes + (24 * p), p1[p]);
      set_poke(battle->bytes + (24 * p + 184), p2[p]);
    }
  }
};
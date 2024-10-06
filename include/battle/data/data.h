#pragma once

#include <battle/data/moves.h>
#include <battle/data/species.h>
#include <battle/data/types.h>

#include <cstddef>
#include <cstring>

namespace Data {

constexpr auto get_types(const Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1].types;
}

constexpr auto get_move_data(const Moves move) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(move) - 1];
}

constexpr uint8_t get_max_pp(const Moves move) noexcept {
  return std::min(PP[static_cast<uint8_t>(move) - 1] / 5 * 8, 61);
}

constexpr auto get_species_data(const Species species) noexcept {
  return SPECIES_DATA[static_cast<uint8_t>(species) - 1];
}

constexpr uint16_t compute_stat(uint8_t base, bool hp = false) noexcept {
  const uint16_t evs = 255;
  const uint32_t core = (2 * (base + 15)) + 63;
  return hp ? core + 110 : core + 5;
}

// TODO don't assume everything
template <typename Side>
constexpr void init_side(const Side &side, uint8_t *bytes) {
  assert(side.size() > 0 && side.size() <= 6);
  std::memset(bytes, 0, 24 * 6); // TODO this is overkill
  std::memset(bytes + 184, 0, 24 * 6);
  for (int i = 0; i < side.size(); ++i) {
    bytes[176 + i] = static_cast<uint8_t>(i + 1);
  }
  for (int i = side.size(); i < 6; ++i) {
    bytes[176 + i] = 0;
  }
  for (const auto &set : side) {
    assert(set.moves.size() <= 4);
    const auto species = set.species;
    if (species != Species::None) {
      // stats
      const auto base = get_species_data(species).base_stats;
      auto *stat_bytes = reinterpret_cast<uint16_t *>(bytes);
      for (int s = 0; s < 5; ++s) {
        stat_bytes[s] = compute_stat(base[s], s == 0);
      }
      // moves
      auto *move_bytes = bytes + 10;
      for (const auto move : set.moves) {
        move_bytes[0] = static_cast<uint8_t>(move);
        move_bytes[1] = Data::get_max_pp(move);
        move_bytes += 2;
      }
      // tail
      stat_bytes[9] = stat_bytes[0];
      bytes[20] = 0;
      bytes[21] = static_cast<uint8_t>(species);
      const auto types = get_types(species);
      bytes[22] = (static_cast<uint8_t>(types[1]) << 4) |
                  static_cast<uint8_t>(types[0]);
      bytes[23] = 100;
    }
    bytes += 24;
  }
}

static_assert(sizeof(Data::Species) == 1);
static_assert(sizeof(Data::Moves) == 1);
static_assert(sizeof(Data::Types) == 1);
static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);

}; // namespace Data
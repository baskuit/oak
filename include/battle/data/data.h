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

template <typename Side> void init_side(const Side &side, uint8_t *bytes) {
  assert(side.size() > 0 && side.size() <= 6);
  std::memset(bytes, 0, 24 * 6); // TODO this is overkill
  std::memset(bytes + 184, 0, 24 * 6);
  for (int i = 0; i < side.size(); ++i) {
    bytes[100 + i] = static_cast<uint8_t>(i + 1);
  }
  for (const auto &set : side) {
    assert(side.moves.size() <= 4);
    if (set.species != Data::Species::None) {
      uint8_t *move_bytes = bytes + 10;
      for (const auto move : set.moves) {
        move_bytes[0] = static_cast<uint8_t>(move);
        move_bytes[1] = Data::get_max_pp(move);
        move_bytes += 2;
      }
      bytes[23] = static_cast<uint8_t>(set.species);
    }
    bytes += 24;
  }
}

static_assert(sizeof(Data::Species) == 1);
static_assert(sizeof(Data::Moves) == 1);
static_assert(sizeof(Data::Types) == 1);

}; // namespace Data
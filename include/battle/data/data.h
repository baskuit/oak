#pragma once

#include <battle/data/moves.h>
#include <battle/data/sample-teams.h>
#include <battle/data/species.h>
#include <battle/data/types.h>

#include <assert.h>
#include <cstddef>
#include <cstring>
#include <type_traits>

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

constexpr std::array<uint16_t, 5> compute_stats(const auto &pokemon) noexcept {
  std::array<uint16_t, 5> stats;
  const auto base = get_species_data(pokemon.species).base_stats;
  const auto ev = [&pokemon]() {
    if constexpr (requires { pokemon.ev; }) {
      return pokemon.ev;
    } else {
      return std::array<uint8_t, 5>{63, 63, 63, 63, 63};
    }
  }();
  const auto dv = [&pokemon]() {
    if constexpr (requires { pokemon.dv; }) {
      return pokemon.dv;
    } else {
      return std::array<uint8_t, 5>{15, 15, 15, 15, 15};
    }
  }();
  for (int s = 0; s < 5; ++s) {
    const uint32_t core = 2 * (base[s] + dv[s]) + ev[s];
    stats[s] = (s == 0) ? core + 110 : core + 5;
  }
  return stats;
}

constexpr void init_pokemon(const auto &pokemon, uint8_t *bytes) noexcept {
  const auto species = pokemon.species;
  if (species == Species::None) {
    return;
  }

  const auto stats = compute_stats(pokemon);
  auto *u16_ptr = std::bit_cast<uint16_t *>(bytes);
  for (int s = 0; s < 5; ++s) {
    u16_ptr[s] = stats[s];
  }

  auto *move_bytes = bytes + 10;
  std::array<uint8_t, 5> dv, ev;
  for (const auto move : pokemon.moves) {
    move_bytes[0] = static_cast<uint8_t>(move);
    move_bytes[1] = Data::get_max_pp(move);
    move_bytes += 2;
  }

  if constexpr (requires { pokemon.hp; }) {
    u16_ptr[9] = pokemon.hp;
  } else {
    u16_ptr[9] = u16_ptr[0];
  }
  if constexpr (requires { pokemon.status; }) {
    bytes[20] = static_cast<uint8_t>(pokemon.status);
  } else {
    bytes[20] = 0;
  }
  bytes[21] = static_cast<uint8_t>(species);
  const auto types = get_types(species);
  bytes[22] =
      (static_cast<uint8_t>(types[1]) << 4) | static_cast<uint8_t>(types[0]);
  if constexpr (requires { pokemon.level; }) {
    bytes[23] = pokemon.level;
  } else {
    bytes[23] = 100;
  }
}

constexpr void init_side(const auto &side, uint8_t *bytes) noexcept {
  assert(side.size() > 0 && side.size() <= 6);
  std::memset(bytes, 0, 24 * 6);
  for (const auto &set : side) {
    assert(set.moves.size() <= 4);
    init_pokemon(set, bytes);
    bytes += 24;
  }
  std::memset(bytes + 176, 0, 6);
  // TODO this breaks if you have None mon followed by a real one... prolly
  for (auto i = 0; i < side.size(); ++i) {
    bytes[176 + i] = i + 1;
  }
}

static_assert(sizeof(Data::Species) == 1);
static_assert(sizeof(Data::Moves) == 1);
static_assert(sizeof(Data::Types) == 1);
// static_assert(compute_stat(100, false) == 298);
// static_assert(compute_stat(250, true) == 703);
// static_assert(compute_stat(5, false) == 108);

}; // namespace Data
#pragma once

#include <data/moves.h>
#include <data/offsets.h>
#include <data/sample-teams.h>
#include <data/species.h>
#include <data/types.h>

#include <assert.h>
#include <bit>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include <pkmn.h>

namespace {
using Data::get_species_data;
using Data::get_types;
using Data::Moves;
using Data::Species;

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

constexpr void init_pokemon(const auto &pokemon,
                            uint8_t *const bytes) noexcept {
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
    move_bytes[1] = get_max_pp(move);
    move_bytes += 2;
  }

  if constexpr (requires { pokemon.hp; }) {
    if constexpr (std::is_floating_point_v<decltype(pokemon.hp)>) {
      u16_ptr[9] = std::min(std::max(pokemon.hp, 0.0), 1.0) * u16_ptr[0];
    } else {
      u16_ptr[9] = pokemon.hp;
    }
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

constexpr void init_side(const auto &side, uint8_t *const bytes) noexcept {
  assert(side.size() > 0 && side.size() <= 6);
  std::memset(bytes, 0, 24 * 6);
  auto i = 0;
  for (const auto &set : side) {
    assert(set.moves.size() <= 4);
    init_pokemon(set, bytes + i * Offsets::pokemon);
    ++i;
  }
  std::memset(bytes + Offsets::order, 0, 6);
  for (uint8_t i = 0; i < side.size(); ++i) {
    bytes[Offsets::order + i] = i + 1;
  }
}
} // end anonymous namespace

namespace Init {
constexpr auto battle(const auto &p1, const auto &p2,
                      uint64_t seed = 0x123445) {
  pkmn_gen1_battle battle{};
  init_side(p1, battle.bytes);
  init_side(p2, battle.bytes + Offsets::side);
  auto *ptr_64 = std::bit_cast<uint64_t *>(battle.bytes + 2 * Offsets::side);
  ptr_64[0] = 0; // turn, last used, etc
  ptr_64[1] = seed;
  return battle;
}

pkmn_result update(pkmn_gen1_battle &battle, const auto c1, const auto c2,
                   pkmn_gen1_battle_options &options) {
  const auto get_choice = [](const auto c, const uint8_t *side) -> pkmn_choice {
    if constexpr (std::is_same_v<typeof(c), Species>) {
      for (int i = 0; i < 6; ++i) {
        if (static_cast<uint8_t>(c) == side[24 * i + 23]) {
          return {};
        }
      }
      throw std::runtime_error{"Init::update - invalid switch"};
    } else if constexpr (std::is_same_v<typeof(c), Moves>) {
      for (int i = 0; i < 4; ++i) {
      }
      throw std::runtime_error{"Init::update - invalid move"};
    } else {
      static_assert(false, "Invalid type for Init::update");
    }
    return {};
  };

  pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  return pkmn_gen1_battle_update(&battle, get_choice(c1, battle.bytes),
                                 get_choice(c1, battle.bytes + Offsets::side),
                                 &options);
}

} // namespace Init

static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);

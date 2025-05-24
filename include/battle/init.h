#pragma once

#include <data/moves.h>
#include <data/offsets.h>
#include <data/species.h>
#include <data/types.h>

#include <assert.h>
#include <bit>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    using HP = std::decay_t<decltype(pokemon.hp)>;
    if constexpr (std::is_floating_point_v<HP>) {
      u16_ptr[9] = std::min(std::max(pokemon.hp, HP{0}), HP{1}) * u16_ptr[0];
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

constexpr std::array<std::array<uint8_t, 2>, 13> boosts{
    std::array<uint8_t, 2>{25, 100}, // -6
    {28, 100},                       // -5
    {33, 100},                       // -4
    {40, 100},                       // -3
    {50, 100},                       // -2
    {66, 100},                       // -1
    {1, 1},                          //  0
    {15, 10},                        // +1
    {2, 1},                          // +2
    {25, 10},                        // +3
    {3, 1},                          // +4
    {35, 10},                        // +5
    {4, 1}                           // +6
};

constexpr uint16_t boost(uint16_t stat, auto b) {
  const auto &pair = boosts[b + 6];
  return std::min(999, stat * pair[0] / pair[1]);
}

constexpr void init_active(const auto &active, uint8_t *const bytes) noexcept {
  if constexpr (requires { active.volatiles; }) {
  }

  if constexpr (requires { active.boosts.atk; }) {
    bytes[0] = boost(100, 2);
  }
}

constexpr void init_party(const auto &party, uint8_t *const bytes) noexcept {
  const uint8_t n = party.size();
  assert(n > 0 && n <= 6);
  std::memset(bytes, 0, 24 * 6);
  std::memset(bytes + Offsets::order, 0, 6);

  uint8_t n_alive = 0;

  for (uint8_t i = 0; i < n; ++i) {
    const auto &set = party[i];
    assert(set.moves.size() <= 4);
    init_pokemon(set, bytes + i * Offsets::pokemon);

    if (set.species != Data::Species::None) {
      if constexpr (requires { set.hp; }) {
        if (set.hp == 0) {
          continue;
        }
      }
      bytes[Offsets::order + n_alive] = i + 1;
      ++n_alive;
    }
  }
}

constexpr void init_side(const auto &side, uint8_t *const bytes) noexcept {
  if constexpr (requires { side.pokemon; }) {
    init_party(side.pokemon, bytes);
    if constexpr (requires { side.active; }) {
      init_active(side.active, bytes + Offsets::active);
    }
  } else {
    init_party(side, bytes);
  }
}
} // end anonymous namespace

namespace Init {

struct Set {
  Species species;
  std::array<Moves, 4> moves;
  float hp = 1;
  uint8_t status = 0;
  uint8_t sleep = 0;
  constexpr bool operator==(const Set &) const = default;
};

struct Config {
  std::array<Set, 6> pokemon;
};

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

constexpr pkmn_gen1_battle_options options() { return {}; }

[[nodiscard]] pkmn_result update(pkmn_gen1_battle &battle, const auto c1,
                                 const auto c2,
                                 pkmn_gen1_battle_options &options) {
  const auto get_choice = [](const auto c, const uint8_t *side) -> pkmn_choice {
    using Choice = decltype(c);
    if constexpr (std::is_same_v<Choice, Species>) {
      for (uint8_t i = 1; i < 6; ++i) {
        const auto slot = side[Offsets::order + i] - 1;
        if (static_cast<uint8_t>(c) == side[24 * slot + Offsets::species]) {
          return ((i + 1) << 2) | 2;
        }
      }
      throw std::runtime_error{"Init::update - invalid switch"};
    } else if constexpr (std::is_same_v<Choice, Moves>) {
      for (uint8_t i = 0; i < 4; ++i) {
        if (static_cast<uint8_t>(c) == side[Offsets::active_moves + 2 * i]) {
          return ((i + 1) << 2) | 1;
        }
      }
      throw std::runtime_error{"Init::update - invalid move"};
    } else if constexpr (std::is_integral_v<Choice>) {
      return c;
    } else {
      static_assert(false);
    }
  };
  pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  return pkmn_gen1_battle_update(&battle, get_choice(c1, battle.bytes),
                                 get_choice(c2, battle.bytes + Offsets::side),
                                 &options);
}

static auto choices(const pkmn_gen1_battle &battle, const pkmn_result result)
    -> std::pair<std::vector<pkmn_choice>, std::vector<pkmn_choice>> {
  std::vector<pkmn_choice> p1_choices;
  std::vector<pkmn_choice> p2_choices;
  p1_choices.resize(PKMN_GEN1_MAX_CHOICES);
  p2_choices.resize(PKMN_GEN1_MAX_CHOICES);
  const auto m =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
  const auto n =
      pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
  p1_choices.resize(m);
  p2_choices.resize(n);
  return {p1_choices, p2_choices};
}

static auto score(const pkmn_result result) {
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_NONE: {
    return .5;
  }
  case PKMN_RESULT_WIN: {
    return 1.0;
  }
  case PKMN_RESULT_LOSE: {
    return 0.0;
  }
  case PKMN_RESULT_TIE: {
    return 0.5;
  }
  default: {
    assert(false);
    return 0.5;
  }
  }
}

} // namespace Init

static_assert(compute_stat(100, false) == 298);
static_assert(compute_stat(250, true) == 703);
static_assert(compute_stat(5, false) == 108);

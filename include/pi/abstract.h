#pragma once

#include <array>

#include <battle/view.h>
#include <data/offsets.h>

#include <bit>
#include <cmath>

#include <pkmn.h>

namespace Abstract {

enum class HP : std::underlying_type_t<std::byte> {
  HP0,
  HP1,
  HP2,
  HP3,
};

enum class Status : std::underlying_type_t<std::byte> {
  None,
  Sleep,
  Poison,
  Burn,
  Paralysis,
  Freeze,
};

constexpr Status simplify_status(const auto status) {
  if (static_cast<uint8_t>(status) & 7) {
    return Status::Sleep;
  }
  switch (static_cast<uint8_t>(status)) {
  case 0b00000000:
    return Status::None;
  case 0b00001000:
    return Status::Poison;
  case 0b00010000:
    return Status::Burn;
  case 0b00100000:
    return Status::Freeze;
  case 0b01000000:
    return Status::Paralysis;
  case 0b10001000:
    return Status::Poison;
  default:
    assert(false);
    return Status::None;
  }
}

#pragma pack(push, 1)
struct Pokemon {
  HP hp;
  Status status;

  static constexpr HP get_hp(const View::Pokemon &pokemon) {
    const float x = 3.0f * pokemon.hp() / pokemon.stats().hp();
    return static_cast<HP>(std::ceil(x));
  }

  bool operator==(const Pokemon &) const = default;

  constexpr Pokemon() : hp{HP::HP3}, status{Status::None} {}
  constexpr Pokemon(const View::Pokemon &pokemon)
      : hp{get_hp(pokemon)}, status{simplify_status(pokemon.status())} {}

  constexpr bool alive_and_unfrozen() const noexcept {
    return !(hp == HP::HP0 || status == Status::Freeze);
  }
};
#pragma pack(pop)

// static_assert(std::ceil(3.0f * 0 / 300) == 0);
// static_assert(std::ceil(3.0f * 1 / 300) == 1);
// static_assert(std::ceil(3.0f * 100 / 300) == 1);
// static_assert(std::ceil(3.0f * 101 / 300) == 2);
// static_assert(std::ceil(3.0f * 200 / 300) == 2);
// static_assert(std::ceil(3.0f * 201 / 300) == 3);
// static_assert(std::ceil(3.0f * 300 / 300) == 3);

#pragma pack(push, 1)
struct Active {
  std::array<int8_t, 5> stats;
  uint8_t reflect = 0;
  uint8_t light_screen = 0;
  uint8_t slot;
  int32_t n_alive = 0;
  uint8_t padding[20 - 12];

  bool operator==(const Active &) const = default;

  Active() = default;
  constexpr Active(const View::Side &side)
      : slot{static_cast<uint8_t>(side.order()[0] - 1)} {}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Side {
  Active active;
  std::array<Pokemon, 6> bench;

  bool operator==(const Side &) const = default;

  Side() = default;
  constexpr Side(const View::Side &side)
      : active{side},
        bench{Pokemon{side.pokemon(0)}, Pokemon{side.pokemon(1)},
              Pokemon{side.pokemon(2)}, Pokemon{side.pokemon(3)},
              Pokemon{side.pokemon(4)}, Pokemon{side.pokemon(5)}} {
    for (int p = 0; p < 6; ++p) {
      const auto &pokemon = side.pokemon(p);
      if ((pokemon.hp() > 0) && (pokemon.stats().hp() > 0)) {
        ++active.n_alive;
      }
    }
  }

  void update(const uint8_t *const side) noexcept {}
};
#pragma pack(pop)

struct Battle {
  std::array<Side, 2> sides;

  Battle() = default;

  bool operator==(const Battle &) const = default;

  constexpr Battle(const pkmn_gen1_battle &battle) noexcept
      : sides{View::ref(battle).side(0), View::ref(battle).side(1)} {}

  void update(const pkmn_gen1_battle &battle) noexcept {
    const auto &b = View::ref(battle);
    for (auto s = 0; s < 2; ++s) {
      const auto &side = b.side(s);
      const auto slot = side.order()[0] - 1;
      sides[s].active.slot = slot;
      sides[s].bench[slot] = Pokemon{side.pokemon(slot)};
      sides[s].active.n_alive = 0;
      for (int p = 0; p < 6; ++p) {
        const auto &pokemon = side.pokemon(p);
        if ((pokemon.hp() > 0) && (pokemon.stats().hp() > 0)) {
          ++sides[s].active.n_alive;
        }
      }
    }
  }
};

static_assert(sizeof(Active) == 20);
static_assert(sizeof(Pokemon) == 2);
static_assert(sizeof(Side) == 32);
static_assert(sizeof(Battle) == 64);

}; // namespace Abstract
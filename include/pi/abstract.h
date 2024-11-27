#pragma once

#include <array>

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
  case 0b01000000:
    return Status::Paralysis;
  case 0b10001000:
    return Status::Poison;
  default:
    // assert(false);
    return Status::None;
  }
}

#pragma pack(push, 1)
struct Bench {
  HP hp;
  Status status;

  static constexpr HP get_hp(const uint8_t *const bytes) {
    const auto cur = std::bit_cast<const uint16_t *const>(bytes)[9];
    const auto max = std::bit_cast<const uint16_t *const>(bytes)[0];
    return static_cast<HP>(std::ceil(3 * cur / max));
  }

  bool operator==(const Bench &) const = default;

  constexpr Bench() : hp{HP::HP3}, status{Status::None} {}
  constexpr Bench(const uint8_t *const bytes)
      : hp{get_hp(bytes)}, status{simplify_status(bytes[Offsets::status])} {}
};
#pragma pack(pop)

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
  constexpr Active(const uint8_t *bytes) : slot{bytes[176 - 144] - 1} {}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Side {
  Active active;
  std::array<Bench, 6> bench;

  bool operator==(const Side &) const = default;

  Side() = default;
  constexpr Side(const uint8_t *const bytes)
      : active{bytes + Offsets::active},
        bench{Bench{bytes + 0 * Offsets::pokemon},
              Bench{bytes + 1 * Offsets::pokemon},
              Bench{bytes + 2 * Offsets::pokemon},
              Bench{bytes + 3 * Offsets::pokemon},
              Bench{bytes + 4 * Offsets::pokemon},
              Bench{bytes + 5 * Offsets::pokemon}} {
    for (int p = 0; p < 6; ++p) {
      const auto pokemon =
          std::bit_cast<const uint16_t *>(bytes + p * Offsets::pokemon);
      if (pokemon[0] && pokemon[9]) {
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
      : sides{battle.bytes, battle.bytes + Offsets::side} {}

  void update(const pkmn_gen1_battle &battle) noexcept {
    sides[0].active.slot = battle.bytes[Offsets::order] - 1;
    auto &prior_active_1 = sides[0].bench[sides[0].active.slot];
    prior_active_1 =
        Bench{battle.bytes + Offsets::pokemon * (sides[0].active.slot)};
    sides[0].active.n_alive = 0;
    for (int p = 0; p < 6; ++p) {
      const auto pokemon =
          std::bit_cast<const uint16_t *>(battle.bytes + p * Offsets::pokemon);
      if (pokemon[0] && pokemon[9]) {
        ++sides[0].active.n_alive;
      }
    }

    sides[1].active.slot = battle.bytes[Offsets::side + Offsets::order] - 1;
    auto &prior_active_2 = sides[1].bench[sides[1].active.slot];
    prior_active_2 = Bench{battle.bytes + Offsets::side +
                           Offsets::pokemon * (sides[1].active.slot)};
    sides[1].active.n_alive = 0;
    for (int p = 0; p < 6; ++p) {
      const auto pokemon = std::bit_cast<const uint16_t *>(
          battle.bytes + Offsets::side + p * Offsets::pokemon);
      if (pokemon[0] && pokemon[9]) {
        ++sides[1].active.n_alive;
      }
    }
  }
};

static_assert(sizeof(Active) == 20);
static_assert(sizeof(Bench) == 2);
static_assert(sizeof(Side) == 32);
static_assert(sizeof(Battle) == 64);

}; // namespace Abstract
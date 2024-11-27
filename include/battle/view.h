#pragma once

#include <pkmn.h>

#include <data/moves.h>
#include <data/offsets.h>
#include <data/species.h>
#include <data/status.h>

#include <array>
#include <bit>

namespace View {

struct Volatiles {
  uint8_t bytes[8];
};

struct ActivePokemon {
  uint8_t bytes[32];
};

struct Stats {
  uint8_t bytes[10];

  constexpr uint16_t &hp() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 0);
  }

  constexpr uint16_t &atk() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 2);
  }

  constexpr uint16_t &def() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 4);
  }

  constexpr uint16_t &spe() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 6);
  }

  constexpr uint16_t &spc() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 8);
  }
};

struct MoveSlot {
  Data::Moves id;
  uint8_t pp;
};

struct Pokemon {
  uint8_t bytes[24];

  constexpr Stats &stats() noexcept { return *std::bit_cast<Stats *>(bytes); }

  constexpr std::array<MoveSlot, 4> &moves() noexcept {
    return *std::bit_cast<std::array<MoveSlot, 4> *>(bytes + 10);
  }

  constexpr uint16_t &hp() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 18);
  }

  constexpr Data::Status &hp() noexcept {
    return *std::bit_cast<Data::Status *>(bytes + 20);
  }

  constexpr Data::Species &species() noexcept {
    return *std::bit_cast<Data::Species *>(bytes + 21);
  }

  constexpr uint8_t &types() noexcept { return bytes[22]; }

  constexpr uint8_t &level() noexcept { return bytes[23]; }
};

struct Side {

  uint8_t bytes[Offsets::side];

  constexpr Pokemon &pokemon(const auto slot) noexcept {
    return *std::bit_cast<Pokemon *>(bytes + slot * Offsets::pokemon);
  }

  constexpr ActivePokemon &active() noexcept {
    return *std::bit_cast<ActivePokemon *>(bytes + Offsets::active);
  }

  constexpr std::array<uint8_t, 6> &order() noexcept {
    return *std::bit_cast<std::array<uint8_t, 6> *>(bytes + Offsets::order);
  }
};

struct Battle {
  uint8_t bytes[PKMN_GEN1_BATTLE_SIZE];

  constexpr Side &side(const auto side) noexcept {
    return *std::bit_cast<Side *>(bytes + side * Offsets::side);
  }
};

constexpr Battle& ref(pkmn_gen1_battle &battle) noexcept {
  return *std::bit_cast<Battle*>(&battle);
}

}; // namespace View
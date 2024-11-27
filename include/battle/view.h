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

  constexpr const uint16_t &hp() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 0);
  }

  constexpr uint16_t &atk() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 2);
  }

  constexpr const uint16_t &atk() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 2);
  }

  constexpr uint16_t &def() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 4);
  }

  constexpr const uint16_t &def() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 4);
  }

  constexpr uint16_t &spe() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 6);
  }

  constexpr const uint16_t &spe() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 6);
  }

  constexpr uint16_t &spc() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 8);
  }

  constexpr const uint16_t &spc() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 8);
  }
};

struct MoveSlot {
  Data::Moves id;
  uint8_t pp;
};

struct Pokemon {
  uint8_t bytes[24];

  constexpr Stats &stats() noexcept {
    return *std::bit_cast<Stats *>(bytes + 0);
  }

  constexpr const Stats &stats() const noexcept {
    return *std::bit_cast<const Stats *>(bytes + 0);
  }

  constexpr std::array<MoveSlot, 4> &moves() noexcept {
    return *std::bit_cast<std::array<MoveSlot, 4> *>(bytes + 10);
  }

  constexpr const std::array<MoveSlot, 4> &moves() const noexcept {
    return *std::bit_cast<const std::array<MoveSlot, 4> *>(bytes + 10);
  }

  constexpr uint16_t &hp() noexcept {
    return *std::bit_cast<uint16_t *>(bytes + 18);
  }

  constexpr const uint16_t &hp() const noexcept {
    return *std::bit_cast<const uint16_t *>(bytes + 18);
  }

  constexpr Data::Status &status() noexcept {
    return *std::bit_cast<Data::Status *>(bytes + 20);
  }

  constexpr const Data::Status &status() const noexcept {
    return *std::bit_cast<const Data::Status *>(bytes + 20);
  }

  constexpr Data::Species &species() noexcept {
    return *std::bit_cast<Data::Species *>(bytes + 21);
  }

  constexpr const Data::Species &species() const noexcept {
    return *std::bit_cast<const Data::Species *>(bytes + 21);
  }

  constexpr uint8_t &types() noexcept { return bytes[22]; }

  constexpr const uint8_t &types() const noexcept { return bytes[22]; }

  constexpr uint8_t &level() noexcept { return bytes[23]; }

  constexpr const uint8_t &level() const noexcept { return bytes[23]; }
};

struct Side {

  uint8_t bytes[Offsets::side];

  constexpr Pokemon &pokemon(const auto slot) noexcept {
    return *std::bit_cast<Pokemon *>(bytes + slot * Offsets::pokemon);
  }

  constexpr const Pokemon &pokemon(const auto slot) const noexcept {
    return *std::bit_cast<const Pokemon *>(bytes + slot * Offsets::pokemon);
  }

  constexpr ActivePokemon &active() noexcept {
    return *std::bit_cast<ActivePokemon *>(bytes + Offsets::active);
  }

  constexpr const ActivePokemon &active() const noexcept {
    return *std::bit_cast<const ActivePokemon *>(bytes + Offsets::active);
  }

  constexpr std::array<uint8_t, 6> &order() noexcept {
    return *std::bit_cast<std::array<uint8_t, 6> *>(bytes + Offsets::order);
  }

  constexpr const std::array<uint8_t, 6> &order() const noexcept {
    return *std::bit_cast<const std::array<uint8_t, 6> *>(bytes +
                                                          Offsets::order);
  }
};

struct Battle {
  uint8_t bytes[PKMN_GEN1_BATTLE_SIZE];

  constexpr Side &side(const auto side) noexcept {
    return *std::bit_cast<Side *>(bytes + side * Offsets::side);
  }

  constexpr const Side &side(const auto side) const noexcept {
    return *std::bit_cast<const Side *>(bytes + side * Offsets::side);
  }
};

constexpr Battle &ref(pkmn_gen1_battle &battle) noexcept {
  return *std::bit_cast<Battle *>(&battle);
}

constexpr const Battle &ref(const pkmn_gen1_battle &battle) noexcept {
  return *std::bit_cast<const Battle *>(&battle);
}

struct Duration {
  uint8_t bytes[4];
};

struct Durations {
  uint8_t bytes[8];
};

}; // namespace View

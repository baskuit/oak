#pragma once

#include <pkmn.h>

#include <data/moves.h>
#include <data/offsets.h>
#include <data/species.h>
#include <data/status.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace View {

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

  auto percent() const noexcept {
    return std::ceil(100.0f * hp() / stats().hp());
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

struct Volatiles {
  uint64_t bits;
  bool bide() const { return bits & (1 << 0); }
  bool thrashing() const { return bits & (1 << 1); }
  bool multi_hit() const { return bits & (1 << 2); }
  bool flinch() const { return bits & (1 << 3); }
  bool charging() const { return bits & (1 << 4); }
  bool binding() const { return bits & (1 << 5); }
  bool invulnerable() const { return bits & (1 << 6); }
  bool confusion() const { return bits & (1 << 7); }
  bool mist() const { return bits & (1 << 8); }
  bool focus_energy() const { return bits & (1 << 9); }
  bool substitute() const { return bits & (1 << 10); }
  bool recharging() const { return bits & (1 << 11); }
  bool rage() const { return bits & (1 << 12); }
  bool leech_seed() const { return bits & (1 << 13); }
  bool toxic() const { return bits & (1 << 14); }
  bool light_screen() const { return bits & (1 << 15); }
  bool reflect() const { return bits & (1 << 16); }
  bool transform() const { return bits & (1 << 17); }
  bool confusion_left() const { return (bits >> 18) & 0b111; }
  bool attacks() const { return (bits >> 21) & 0b111; }
  bool state() const { return (bits >> 24) & 0xFFFF; }
  bool substitute_hp() const { return (bits >> 40) & 0xFF; }
  bool transform_species() const { return (bits >> 48) & 0xF; }
  bool disable_left() const { return (bits >> 52) & 0xF; }
  bool disable_move() const { return (bits >> 56) & 0b111; }
  bool toxic_counter() const { return (bits >> 59) & 0b11111; }
};

struct ActivePokemon {
  uint8_t bytes[32];

  constexpr Stats &stats() noexcept {
    return *std::bit_cast<Stats *>(bytes + 0);
  }

  constexpr const Stats &stats() const noexcept {
    return *std::bit_cast<Stats *>(bytes + 0);
  }

  const uint8_t boost_atk() const noexcept { return bytes[12] & 0b00001111; }
  const uint8_t boost_def() const noexcept { return bytes[12] & 0b11110000; }
  const uint8_t boost_spe() const noexcept { return bytes[13] & 0b00001111; }
  const uint8_t boost_spc() const noexcept { return bytes[13] & 0b11110000; }
  const uint8_t boost_acc() const noexcept { return bytes[14] & 0b00001111; }
  const uint8_t boost_eva() const noexcept { return bytes[14] & 0b11110000; }
  void set_boost_atk(auto value) noexcept {
    bytes[12] &= 0b11110000;
    bytes[12] |= static_cast<uint8_t>(value) & 0b00001111;
  }

  void set_boost_def(auto value) noexcept {
    bytes[12] &= 0b00001111;
    bytes[12] |= static_cast<uint8_t>(value) & 0b11110000;
  }

  void set_boost_spe(auto value) noexcept {
    bytes[13] &= 0b11110000;
    bytes[13] |= static_cast<uint8_t>(value) & 0b00001111;
  }

  void set_boost_spc(auto value) noexcept {
    bytes[13] &= 0b00001111;
    bytes[13] |= static_cast<uint8_t>(value) & 0b11110000;
  }

  void set_boost_acc(auto value) noexcept {
    bytes[14] &= 0b11110000;
    bytes[14] |= static_cast<uint8_t>(value) & 0b00001111;
  }

  void set_boost_eva(auto value) noexcept {
    bytes[15] &= 0b00001111; // Clear the upper 4 bits
    bytes[15] |=
        static_cast<uint8_t>(value) & 0b11110000; // Set the upper 4 bits
  }

  constexpr Volatiles &volatiles() noexcept {
    return *std::bit_cast<Volatiles *>(bytes + 16); // TODO
  }

  constexpr const Volatiles &volatiles() const noexcept {
    return *std::bit_cast<const Volatiles *>(bytes + 16); // TODO
  }
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

  auto sleep(auto slot) const {
    auto b = std::bit_cast<uint32_t>(bytes);
    return 7 & (b >> (3 * slot));
  }

  void set_sleep(auto slot, auto value) {
    auto b = std::bit_cast<uint32_t>(bytes);
    const auto offset = 3 * slot;
    const auto clear = ~(7 << offset);
    b = (b & clear) | (value << offset);
    std::memcpy(bytes, &b, sizeof(b));
  }

  auto confusion() const {
    auto b = std::bit_cast<uint32_t>(bytes);
    return 7 & (b >> 18);
  }

  auto disable() const {
    auto b = std::bit_cast<uint32_t>(bytes);
    return 15 & (b >> 21);
  }

  auto attacking() const {
    auto b = std::bit_cast<uint32_t>(bytes);
    return 7 & (b >> 25);
  }

  auto binding() const {
    auto b = std::bit_cast<uint32_t>(bytes);
    return 7 & (b >> 28);
  }
};

struct Durations {
  uint8_t bytes[8];

  Duration &duration(auto i) {
    return *std::bit_cast<Duration *>(bytes + 4 * i);
  }

  const Duration &duration(auto i) const {
    return *std::bit_cast<const Duration *>(bytes + 4 * i);
  }
};

static Durations &ref(pkmn_gen1_chance_durations &durations) {
  return *std::bit_cast<Durations *>(&durations);
}

static const Durations &ref(const pkmn_gen1_chance_durations &durations) {
  return *std::bit_cast<const Durations *>(&durations);
}

}; // namespace View

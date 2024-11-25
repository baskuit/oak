#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

namespace Data {

enum class Types : std::underlying_type_t<std::byte> {
  Normal,
  Fighting,
  Flying,
  Poison,
  Ground,
  Rock,
  Bug,
  Ghost,
  Fire,
  Water,
  Grass,
  Electric,
  Psychic,
  Ice,
  Dragon,
};

// 2x de facto multiplier
enum class Effectiveness : std::underlying_type_t<std::byte> {
  I = 0,
  R = 1,
  N = 2,
  S = 4,
};

using E = Effectiveness;
constexpr static std::array<std::array<Effectiveness, 15>, 15> TYPE_CHART{
    {{E::N, E::N, E::N, E::N, E::N, E::R, E::N, E::I, E::N, E::N, E::N, E::N,
      E::N, E::N, E::N},
     {E::S, E::N, E::R, E::R, E::N, E::S, E::R, E::I, E::N, E::N, E::N, E::N,
      E::R, E::S, E::N},
     {E::N, E::S, E::N, E::N, E::N, E::R, E::S, E::N, E::N, E::N, E::S, E::R,
      E::N, E::N, E::N},
     {E::N, E::N, E::N, E::R, E::R, E::R, E::S, E::R, E::N, E::N, E::S, E::N,
      E::N, E::N, E::N},
     {E::N, E::N, E::I, E::S, E::N, E::S, E::R, E::N, E::S, E::N, E::R, E::S,
      E::N, E::N, E::N},
     {E::N, E::R, E::S, E::N, E::R, E::N, E::S, E::N, E::S, E::N, E::N, E::N,
      E::N, E::S, E::N},
     {E::N, E::R, E::R, E::S, E::N, E::N, E::N, E::R, E::R, E::N, E::S, E::N,
      E::S, E::N, E::N},
     {E::I, E::N, E::N, E::N, E::N, E::N, E::N, E::S, E::N, E::N, E::N, E::N,
      E::I, E::N, E::N},
     {E::N, E::N, E::N, E::N, E::N, E::R, E::S, E::N, E::R, E::R, E::S, E::N,
      E::N, E::S, E::R},
     {E::N, E::N, E::N, E::N, E::S, E::S, E::N, E::N, E::S, E::R, E::R, E::N,
      E::N, E::N, E::R},
     {E::N, E::N, E::R, E::R, E::S, E::S, E::R, E::N, E::R, E::S, E::R, E::N,
      E::N, E::N, E::R},
     {E::N, E::N, E::S, E::N, E::I, E::N, E::N, E::N, E::N, E::S, E::R, E::R,
      E::N, E::N, E::R},
     {E::N, E::S, E::N, E::S, E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N,
      E::R, E::N, E::N},
     {E::N, E::N, E::S, E::N, E::S, E::N, E::N, E::N, E::N, E::R, E::S, E::N,
      E::N, E::R, E::S},
     {E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N, E::N,
      E::N, E::N, E::S}}};

consteval auto get_effectiveness(Types attacking, Types defending) noexcept {
  return TYPE_CHART[static_cast<uint8_t>(attacking)]
                   [static_cast<uint8_t>(defending)];
}

constexpr bool is_special(const auto type) {
  return static_cast<uint8_t>(type) >= 8;
}

constexpr bool is_physical(const auto type) {
  return static_cast<uint8_t>(type) < 8;
}

static_assert(sizeof(Types) == 1);

} // namespace Data

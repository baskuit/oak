#pragma once

#include "pkmn.h"
#include <array>
#include <cstdlib>
#include <memory>

#pragma pack(push, 1)
struct Frame {
  std::array<uint8_t, Sizes::battle> battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  float eval;
  float score;
  uint32_t iter;

  Frame() = default;
  Frame(const pkmn_gen1_battle *const battle,
        const pkmn_gen1_chance_durations *const durations, pkmn_result result,
        float eval, auto iter)
      : result{result}, eval{eval}, iter{static_cast<uint32_t>(iter)} {
    std::memcpy(this->battle.data(), battle, Sizes::battle);
    std::memcpy(this->durations.bytes, durations->bytes, Sizes::durations);
  }
};
#pragma pack(pop)

static_assert(sizeof(Frame) == 405);
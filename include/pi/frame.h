#pragma once

#include <array>
#include <cstdlib>
#include <memory>

#include "pkmn.h"

#include <data/offsets.h>

#pragma pack(push, 1)
struct Frame {
  std::array<uint8_t, Sizes::battle> battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  float eval;
  float score;
  uint32_t iter;
  uint8_t row_col;
  std::array<uint16_t, 9> p1_visits;
  std::array<uint16_t, 9> p2_visits;

  Frame() = default;
  Frame(const pkmn_gen1_battle *const battle,
        const pkmn_gen1_chance_durations *const durations, pkmn_result result,
        const auto &output)
      : result{result}, eval{output.average_value},
        iter{static_cast<uint32_t>(output.iterations)},
        row_col{static_cast<uint8_t>(9 * (output.m - 1) + (output.n - 1))},
        p1_visits{}, p2_visits{} {
    std::memcpy(this->battle.data(), battle->bytes, Sizes::battle);
    std::memcpy(this->durations.bytes, durations->bytes, Sizes::durations);
    for (auto i = 0; i < output.m; ++i) {
      for (auto j = 0; j < output.n; ++j) {
        p1_visits[i] += output.visit_matrix[i][j];
        p2_visits[j] += output.visit_matrix[i][j];
      }
    }
  }
};
#pragma pack(pop)

static_assert(sizeof(Frame) == 442);
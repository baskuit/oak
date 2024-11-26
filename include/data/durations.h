#pragma once

#include <bit>
#include <iostream>

#include <pkmn.h>

namespace Durations {

uint8_t duration(const pkmn_gen1_chance_durations &durations, auto side,
                 auto slot) {
  auto duration = std::bit_cast<const uint32_t *>(&durations) + side;
  return 7 & ((*duration) >> (3 * slot));
}

void print(const pkmn_gen1_chance_durations &durations) {
  for (auto i = 0; i < 2; ++i) {
    for (auto p = 0; p < 6; ++p) {
      std::cout << (int)duration(durations, i, p) << ' ';
    }
    std::cout << std::endl;
  }
}

} // namespace Durations
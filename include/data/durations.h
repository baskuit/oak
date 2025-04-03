#pragma once

#include <bit>
#include <iostream>

#include <pkmn.h>

namespace Durations {

// gets the observed number of turns slept, as opposed to the de facto sleep
// roll
static uint8_t sleep(const pkmn_gen1_chance_durations &durations, auto side,
                     auto slot) {
  auto duration = std::bit_cast<const uint32_t *>(&durations) + side;
  return 7 & ((*duration) >> (3 * slot));
}

static void print(const pkmn_gen1_chance_durations &durations) {
  for (auto i = 0; i < 2; ++i) {
    for (auto p = 0; p < 6; ++p) {
      std::cout << (int)sleep(durations, i, p) << ' ';
    }
    std::cout << std::endl;
  }
}

} // namespace Durations
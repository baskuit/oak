#pragma once

#include <pkmn.h>

namespace View {

struct Pokemon {
  uint8_t bytes[24];
};

struct Active {};

}; // namespace View
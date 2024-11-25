#pragma once

#include <pkmn.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <assert.h>
#include <data/moves.h>
#include <data/species.h>
#include <data/strings.h>

namespace Strings {

std::string status(const auto status) {
  const auto byte = static_cast<uint8_t>(status);
  if (byte == 0) {
    return "";
  }
  if (byte & 7) {
    if (byte & 128) {
      return "RST";
    } else {
      return "SLP";
    }
  }
  switch (byte) {
  case 0b00001000:
    return "PSN";
  case 0b00010000:
    return "BRN";
  case 0b00100000:
    return "";
  case 0b01000000:
    return "TOX";
  default:
    assert(false);
    return "";
  };
}

std::string pokemon_to_string(const uint8_t *const data) {
  std::stringstream sstream{};
  sstream << Names::species_string(data[21]);
  if (data[23] != 100) {
    sstream << " (lvl " << (int)data[23] << ")";
  }
  sstream << status(data[20]);
  for (int m = 0; m < 4; ++m) {
  }

  sstream.flush();
  return sstream.str();
}

} // namespace Strings
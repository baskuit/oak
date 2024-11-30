#pragma once

#include <pkmn.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <data/moves.h>
#include <data/species.h>
#include <data/strings.h>

#include <battle/view.h>

namespace Strings {

// Checks that lower cases match up to the shorter string
bool match(const auto &A, const auto &B) {
  return std::equal(
      A.begin(), A.begin() + std::min(A.size(), B.size()), B.begin(), B.end(),
      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

auto find_unique(const auto &container, const auto &value) {
  const auto matches = [&value](const auto &x) { return match(x, value); };
  auto it = std::find_if(container.begin(), container.end(), matches);
  if (it != container.end()) {
    if (auto other = std::find_if(it + 1, container.end(), matches);
        other != container.end()) {
      return container.end(); // return end if not unique
    }
  }
  return it;
}

int unique_index(const auto &container, const auto &value) {
  const auto it = find_unique(container, value);
  if (it == container.end()) {
    return -1;
  }
  return std::distance(container.begin(), it);
}

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
    return "FRZ";
  case 0b01000000:
    return "PAR";
  case 0b10001000:
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
    if (data[2 * m + 10] != 0) {
      sstream << Names::move_string(data[2 * m + 10]) << ": "
              << (int)data[2 * m + 11] << " ";
    }
  }

  sstream.flush();
  return sstream.str();
}

std::string battle_to_string(const pkmn_gen1_battle &battle) {
  std::stringstream sstream{};

  const auto &b = View::ref(battle);

  for (auto s = 0; s < 2; ++s) {
    const auto &side = b.side(s);

    for (auto slot = 0; slot < 6; ++slot) {
      auto i = side.order()[slot] - 1;

      auto &p = side.pokemon(i);

      if (slot == 0) {
        // pass for now
      }

      sstream << Names::species_string(p.species()) << ": ";
      const auto hp = p.hp();
      const bool ko = (hp == 0);
      if (!ko) {
        sstream << p.percent() << "% ";
      } else {
        sstream << "KO " << std::endl;
        continue;
      }
      const auto st = p.status();
      if (st != Data::Status::None) {
        sstream << status(st) << ' ';
      }
      for (auto m = 0; m < 4; ++m) {
        sstream << Names::move_string(p.moves()[m].id) << ' ';
      }
      sstream << std::endl;
    }
    sstream << std::endl;
  }
  return sstream.str();
}

Data::Species string_to_species(const std::string &str) {
  const int index = unique_index(Names::SPECIES_STRING, str);
  if (index < 0) {
    throw std::runtime_error{"Could not match string to Species"};
    return Data::Species::None;
  } else {
    return static_cast<Data::Species>(index);
  }
}

Data::Moves string_to_move(const std::string &str) {
  const int index = unique_index(Names::MOVE_STRING, str);
  if (index < 0) {
    throw std::runtime_error{"Could not match string to Move"};
    return Data::Moves::None;
  } else {
    return static_cast<Data::Moves>(index);
  }
}

} // namespace Strings
#pragma once

#include <array>

#include "../include/util.h"

// WIP clone of the official showdown random team generator

namespace RandomBattles {

using namespace Helpers;

Battle generate(uint64_t seed) {
    return {};
}

bool RandbatObservationMatches(const Battle &seen, const Battle &omni) {

  const auto pokemon_match_almost = [](const Pokemon &a, const Pokemon &b) {
    if (a.species != b.species) {
      return false;
    }
    // todo optimize?
    for (int i = 0; i < 4; ++i) {
      if (a.moves[i] == Moves::None) {
        continue;
      }
      bool seen = false;
      for (int j = 0; j < 4; ++j) {
        seen = seen || (a.moves[i] == b.moves[j]);
      }
      if (!seen) {
        return false;
      }
    }
    return true;
  };

  const auto sides_match_almost = [](const Side &a, const Side &b) {
    for (const auto &pokemon : a) {
      if (pokemon.species == Species::None) {
        continue;
      }
      return false;
      // for (int i)
      // if (!pokemon_match_almost)
    }
    return true;
  };

  bool observer_can_be_p1 = true;
  bool observer_can_be_p2 = true;
  for (int side = 0; side < 2; ++side) {
    for (int pokemon = 0; pokemon < 6; ++pokemon) {
    }
  }

  return seen == omni;
}

struct prng {
  uint64_t _state;

  void next() {}
};

Battle generate(prng device) { return {}; }

bool test_generate(const uint64_t seed, const Battle &observed_battle) {
  return RandbatObservationMatches(observed_battle, generate(prng{seed}));
}


};
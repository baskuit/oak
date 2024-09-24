#pragma once

#include <types/vector.h>

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>

#include "data.h"

#include "../include/util.h"

// naive implementation of these showdown functions means resizing vectors,
// which is slow uses Pinyon static vector to keep interface the same

// WIP clone of the official showdown random team generator
namespace RandomBattles {

struct PRNG {
  int64_t seed;

  PRNG(int64_t seed) : seed{seed} {}

  auto nextFrame(uint64_t seed, int framesToAdvance = 1) {
    static constexpr uint64_t a{0x5D588B656C078965};
    static constexpr uint64_t c{0x0000000000269EC3};
    for (int i = 0; i < framesToAdvance; i++) {
      seed = a * seed + c;
    }
    return seed;
  }

  double next() {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32; // Use the upper 32 bits
    return (double)top / 0x100000000;
  }

  int next(int to) {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32; // Use the upper 32 bits
    return to * ((double)top / 0x100000000);
  }

  int next(int from, int to) {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32; // Use the upper 32 bits
    return from + (to - from) * ((double)top / 0x100000000);
  }

  template <typename Container, typename T>
  void shuffle(Container &items, int start = 0, int end = -1) {
    if (end < 0) {
      end = items.size();
    }
    while (start < end - 1) {
      const auto nextIndex = next(start, end);
      if (start != nextIndex) {
        std::swap(items[start], items[nextIndex]);
      }
      ++start;
    }
  }

  template <typename Container, typename T>
  const T &sample(const Container &items) {
    assert(items.size() > 0);
    return items[next(items.size())];
  }

  bool randomChance(int numerator, int denominator) {
    return next(denominator) < numerator;
  }

  void display() {
    const auto *data = reinterpret_cast<uint16_t *>(&seed);
    std::cout << "[ ";
    for (int i = 3; i >= 1; --i) {
      std::cout << (int)data[i] << ", ";
    }
    std::cout << (int)data[0] << " ]\n";
  }
};

template <typename Container>
auto sampleNoReplace(Container& container, PRNG& prng) {
  const auto n = container.size();
  const auto index = prng.next(n);
  const auto val = container[index];
  container[index] = container[n - 1];
  container.resize(n - 1);
  return val;
}

struct Teams {
  PRNG prng;
  bool battleHasDitto = false;
  std::array<int, 15> typeCount{};
  int numMaxLevelPokemon{};

  Teams(PRNG prng) : prng{prng} {}

  Helpers::Side randomTeam() {

    Helpers::Side team{};
    auto n_pokemon = 0;

    prng.next(); // for type sample call

    // clone the pool but now as an ArrayBaseVector derived struct with fast
    // sampling
    using Arr = ArrayBasedVector<146>::Vector<Data::Species>;
    Arr pokemonPool{
        RandomBattlesData::pokemonPool};
    // std::cout << "pool size: " << pokemonPool.size() << std::endl;
    // for (int i = 0; i < 10; ++i) {
    //   std::cout << Names::species_name[static_cast<int>(pokemonPool[i])] << '
    //   ';
    // }std::cout << std::endl;

    while (n_pokemon < 6 && pokemonPool.size()) {
      Helpers::Species species = sampleNoReplace(pokemonPool, prng);

      auto name = Names::species_name[static_cast<int>(species)];
      std::cout << "sampled: " << static_cast<int>(species) << " : " << name
                << std::endl;

      if (species == Helpers::Species::Ditto && battleHasDitto) {
        continue;
      }

      bool skip = false;

      // for (const typeName of species.types) {
      // 	if (typeCount[typeName] >= 2) {
      // 		skip = true;
      // 		break;
      // 	}
      // }

      // if (skip) {
      // 	rejectedButNotInvalidPool.push(species.id);
      // 	continue;
      // }

      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        skip = true;
      }

      if (skip) {
        // rejected but not invalid pool
      }

      // accept the set
      team[n_pokemon++] = randomSet(species);

      // only set down here, TODO maybe optimize
      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        ++numMaxLevelPokemon;
      }

      if (species == Helpers::Species::Ditto) {
        battleHasDitto = true;
      }
    }

    return {};
  }

  Helpers::Pokemon randomSet(Helpers::Species species) {
    Helpers::Pokemon set{};

    return set;
  }
};

bool RandbatObservationMatches(const Helpers::Battle &seen,
                               const Helpers::Battle &omni) {

  const auto pokemon_match_almost = [](const Helpers::Pokemon &a,
                                       const Helpers::Pokemon &b) {
    if (a.species != b.species) {
      return false;
    }
    // todo optimize?
    for (int i = 0; i < 4; ++i) {
      if (a.moves[i] == Helpers::Moves::None) {
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

  const auto sides_match_almost = [](const Helpers::Side &a,
                                     const Helpers::Side &b) {
    for (const auto &pokemon : a) {
      if (pokemon.species == Helpers::Species::None) {
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

} // namespace RandomBattles
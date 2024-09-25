#pragma once

#include <types/vector.h>

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <unordered_map>

// now includes ./data.h
#include "random-set-data.h"

#include <util.h>

// WIP clone of the official showdown random team generator
namespace RandomBattles {

struct PRNG {
  int64_t seed;

  PRNG(int64_t seed) : seed{seed} {}

  int64_t nextFrame(int64_t seed) {
    static constexpr int64_t a{0x5D588B656C078965};
    static constexpr int64_t c{0x0000000000269EC3};
    seed = a * seed + c;
    return seed;
  }

  void next() noexcept { seed = nextFrame(seed); }

  int next(int to) noexcept {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32;
    return to * ((double)top / 0x100000000);
  }

  int next(int from, int to) {
    seed = nextFrame(seed);
    const uint32_t top = seed >> 32; // Use the upper 32 bits
    return from + (to - from) * ((double)top / 0x100000000);
  }

  template <typename Container>
  void shuffle(Container &items, int start = 0, int end = -1) noexcept {
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
auto sampleNoReplace(Container &container, PRNG &prng) {
  const auto n = container.size();
  const auto index = prng.next(n);
  const auto val = container[index];
  container[index] = container[n - 1];
  container.resize(n - 1);
  return val;
}

class Teams {
private:
  PRNG prng;
  bool battleHasDitto = false;

public:
  Teams(PRNG prng) : prng{prng} {}

  Helpers::Side randomTeam() {

    Helpers::Side team{};
    auto n_pokemon = 0;

    prng.next(); // for type sample call

    std::array<int, 15> typeCount{};
    std::array<int, 6> weaknessCount{};
    int numMaxLevelPokemon{};

    using Arr = ArrayBasedVector<146>::Vector<Data::Species>;

    Arr pokemonPool{RandomBattlesData::pokemonPool};
    Arr rejectedButNotInvalidPool{};

    while (n_pokemon < 6 && pokemonPool.size()) {
      auto species = sampleNoReplace(pokemonPool, prng);

      if (species == Helpers::Species::Ditto && battleHasDitto) {
        continue;
      }

      bool skip = false;

      // types
      const auto types = Data::get_types(species);
      for (const Data::Types type : types) {
        if (typeCount[static_cast<uint8_t>(type)] >= 2) {
          skip = true;
          break;
        }
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // weakness
      const auto &w =
          RandomBattlesData::RANDOM_SET_DATA[static_cast<uint8_t>(species)]
              .weaknesses;
      for (int i = 0; i < 6; ++i) {
        if (!w[i]) {
          continue;
        }
        if (weaknessCount[i] >= 2) {
          skip = true;
          break;
        }
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // lvl 100
      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        skip = true;
      }
      if (skip) {
        rejectedButNotInvalidPool.push_back(species);
        continue;
      }

      // accept the set
      team[n_pokemon++] = randomSet(species);

      // update
      typeCount[static_cast<uint8_t>(types[0])]++;
      if (types[0] != types[1]) {
        typeCount[static_cast<uint8_t>(types[1])]++;
      }

      for (int i = 0; i < 6; ++i) {
        weaknessCount[i] += w[i];
      }
      if (RandomBattlesData::isLevel100(species) && numMaxLevelPokemon > 1) {
        ++numMaxLevelPokemon;
      }
      if (species == Helpers::Species::Ditto) {
        battleHasDitto = true;
      }
    }

    while (n_pokemon < 6 && rejectedButNotInvalidPool.size()) {
      const auto species = sampleNoReplace(rejectedButNotInvalidPool, prng);
      team[n_pokemon++] = randomSet(species);
    }

    return team;
  }

  Helpers::Pokemon randomSet(Helpers::Species species) {
    Helpers::Pokemon set{};

    const auto print = [](const auto &x) { std::cout << x << std::endl; };

    const auto print_move = [](const auto &x) {
      std::cout << Names::move_name[static_cast<int>(x)] << std::endl;
    };

    const auto data{
        RandomBattlesData::RANDOM_SET_DATA[static_cast<int>(species)]};
    const auto maxMoveCount = 4;

    // todo use something faster
    using Map = std::unordered_map<Data::Moves, bool>;
    Map moves{};

    // combo moves
    if (data.n_combo_moves && (data.n_combo_moves <= maxMoveCount) &&
        prng.randomChance(1, 2)) {
      for (int m = 0; m < data.n_combo_moves; ++m) {
        moves[data.combo_moves[m]] = true;
      }
    }

    // exclusive moves
    if ((moves.size() < maxMoveCount) && data.n_exclusive_moves) {
      moves[data.exclusive_moves[prng.next(data.n_exclusive_moves)]] = true;
    }

    // essential moves
    if ((moves.size() < maxMoveCount) && data.n_essential_moves) {
      for (int m = 0; m < data.n_essential_moves; ++m) {
        moves[data.essential_moves[m]] = true;
        if (moves.size() == maxMoveCount) {
          break;
        }
      }
    }

    // TODO this can be done without copying/using pinyon
    // moves
    ArrayBasedVector<RandomBattlesData::RandomSetEntry::max_moves>::Vector<
        Data::Moves>
        movePool{data.moves};
    movePool.resize(data.n_moves);
    while ((moves.size() < maxMoveCount) && movePool.size()) {
      const auto move = sampleNoReplace(movePool, prng);
      moves[move] = true;
    }

    int m = 0;
    for (const auto &[key, value] : moves) {
      set.moves[m] = key;
      ++m;
      if (m >= 4) {
        break;
      }
    }
    prng.shuffle(set.moves);
    set.species = species;

    // std::cout << Names::species_name[static_cast<uint8_t>(set.species)] <<
    // std::endl; std::cout << '\t' <<
    // Names::move_name[static_cast<uint8_t>(set.moves[0])] << std::endl;
    // std::cout << '\t' << Names::move_name[static_cast<uint8_t>(set.moves[1])]
    // << std::endl; std::cout << '\t' <<
    // Names::move_name[static_cast<uint8_t>(set.moves[2])] << std::endl;
    // std::cout << '\t' << Names::move_name[static_cast<uint8_t>(set.moves[3])]
    // << std::endl; std::cout << std::endl;
    return set;
  }
};

} // namespace RandomBattles
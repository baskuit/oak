#pragma once

#include <pkmn.h>

#include <array>
#include <map>

#include <battle/view.h>

namespace NNUE {

using StatusIndex = uint8_t;
using MoveKey = uint8_t;
using PokemonKey = std::pair<StatusIndex, MoveKey>;

constexpr auto process_status(auto status, auto sleeps) {
  // TODO
  return std::bit_cast<StatusIndex>(status);
}

using Volatiles = uint64_t;
using Stats = std::array<uint16_t, 5>;

struct ActiveKey {
  Volatiles volatiles;
  Stats stats;
  PokemonKey pokemon_key;

  bool operator<(const ActiveKey &other) const {
    return (volatiles < other.volatiles) || (stats < other.stats) ||
           (pokemon_key < other.pokemon_key);
  }
};

PokemonKey get_pokemon_key(const View::Side &side, const View::Duration &d,
                           auto i) {
  const auto &pokemon = side.pokemon(side.order()[i] - 1);
  const auto sleeps = d.sleep(i);
  MoveKey move_key{};
  for (auto i = 0; i < 4; ++i) {
    move_key |= (pokemon.moves()[i].pp > 0);
    move_key <<= 1;
  }
  process_status(std::bit_cast<uint8_t>(pokemon.status()), sleeps);
  return PokemonKey{move_key, move_key};
}

ActiveKey get_active_key(const View::Side &side, const View::Duration &d) {
  return {*std::bit_cast<Volatiles *>(&side.active().volatiles()),
          *std::bit_cast<Stats *>(&side.active().stats()),
          get_pokemon_key(side, d, 0)};
}

struct Abstract {
  struct Side {
    std::array<PokemonKey, 6> pokemon_keys;
    ActiveKey active_key;
    std::array<uint8_t, 6> order;
    std::array<uint8_t, 6> hp;
  };

  Side p1;
  Side p2;

  Abstract(const pkmn_gen1_battle &battle,
           const pkmn_gen1_chance_durations &durations) {
    const auto &b = View::ref(battle);
    const auto &d = View::ref(durations);
    p1.active_key = get_active_key(b.side(0), d.duration(0));
    p2.active_key = get_active_key(b.side(1), d.duration(1));

    for (auto i = 0; i < 6; ++i) {
      p1.pokemon_keys[i] = get_pokemon_key(b.side(0), d.duration(0), i);
      p2.pokemon_keys[i] = get_pokemon_key(b.side(1), d.duration(1), i);
    }
    p1.order = b.side(0).order();
    p2.order = b.side(1).order();
  }

  void update(const View::Battle &battle, pkmn_choice c1, pkmn_choice c2) {
    return;
  }
};

constexpr auto pokemon_in_dim = 198;
constexpr auto active_in_dim = 212;

using PokemonInput = std::array<float, pokemon_in_dim>;
using ActiveInput = std::array<float, active_in_dim>;

template <bool read_stats = true>
PokemonInput get_pokemon_input(const PokemonKey key, const View::Side &side,
                               auto slot) {
  PokemonInput output;

  const auto &pokemon = side.pokemon(slot);
  auto c = 0;
  if constexpr (read_stats) {
    const auto &stats = pokemon.stats();
    output[0] = stats.hp();
    output[1] = stats.atk();
    output[2] = stats.def();
    output[3] = stats.spc();
    output[4] = stats.spe();
  }
  c += 5;
  const auto &moves = pokemon.moves();
  for (auto i = 0; i < 4; ++i) {
    if (moves[i].id && moves[i].pp) {
      output[c + moves[i].id - 1] = 1.0;
    }
  }
  c += 4;
  if (key.first) {
    output[c + key.first] = 1.0;
  }
  c += 14; // TODO
  output[c + static_cast<int>(pokemon.types() % 16)] = 1.0;
  output[c + static_cast<int>(pokemon.types() / 16)] = 1.0;
}

ActiveInput get_active_input(const ActiveKey &key,
                             const View::ActivePokemon &active) {
  ActiveInput output;
  // const auto p = get_pokemon_input()
  return output;
}

constexpr auto pokemon_out_dim = 39;
constexpr auto active_out_dim = 55;

using PokemonWord = std::array<uint8_t, 39>;
using ActiveWord = std::array<uint8_t, 55>;

struct WordCaches {

  template <typename Net, typename Key, typename Value, typename Data>
  struct NNCache {
  private:
    std::map<Key, Value> _cache;
    Data data;

  public:
    NNCache(const auto &nn) : _cache{} {}

    const Value &operator[](const Key &key) {
      const auto it = _cache.find(key);
      if (it == _cache.end()) {
        // inference
      } else {
        return *it;
      }
    }
  };

  struct Slot {
    std::map<PokemonKey, PokemonWord> p_cache;
    std::map<ActiveKey, ActiveWord> a_cache;

    void print_sizes() const {
      std::cout << p_cache.size() << ' ' << a_cache.size() << std::endl;
    }
  };

  struct Side {
    std::array<Slot, 6> slots;

    void print_sizes() const {
      for (const auto &s : slots) {
        s.print_sizes();
      }
    }
  };

  Side p1;
  Side p2;

  void write_acc(const Abstract &a, uint8_t *const acc) {
    auto *const acc2 = acc + 256;
    std::memcpy(p1.slots[a.p1.order[0] - 1].a_cache[a.p1.active_key].data(),
                acc + 1, active_out_dim);
    std::memcpy(p2.slots[a.p2.order[0] - 1].a_cache[a.p2.active_key].data(),
                acc2 + 1, active_out_dim);
    for (auto i = 0; i < 5; ++i) {
      std::memcpy(p1.slots[a.p1.order[i + 1] - 1]
                      .p_cache[a.p1.pokemon_keys[a.p1.order[i + 1] - 1]]
                      .data(),
                  acc + active_out_dim + i * pokemon_out_dim + 1,
                  pokemon_out_dim);
      std::memcpy(p2.slots[a.p2.order[i + 1] - 1]
                      .p_cache[a.p2.pokemon_keys[a.p2.order[i + 1] - 1]]
                      .data(),
                  acc2 + active_out_dim + i * pokemon_out_dim + 1,
                  pokemon_out_dim);
    }
  }
};

}; // namespace NNUE
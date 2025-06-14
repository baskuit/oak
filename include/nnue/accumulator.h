#pragma once

#include <pkmn.h>

#include <array>
#include <bit>
#include <map>

#include <battle/view.h>
#include <data/status.h>
#include <nnue/nnue_architecture.h>

// The data in a pokemon slot that is used by the network can be split into two
// categories: rhe data that stays the same (base stats, move id), and the data
// the changes with battle updates (pp, status). Note that hp is missing, since
// that is added at the accumulator layer to avoid bucketing. The data that
// changes is used as the key to lookup the pokemon embedding. The data that
// does not change is just stored as a const battle object and the individual
// caches (std::maps) store offsets of that battle as pointers.The same applies
// to the active slots.

namespace NNUE {

using StatusIndex = uint8_t;
using MoveKey = uint8_t;
using PokemonKey = std::pair<StatusIndex, MoveKey>;

constexpr StatusIndex get_status_index(auto status, auto sleeps) {
  if (!status) {
    return 0;
  }
  auto index = 0;
  if (!Data::is_sleep(status)) {
    index = std::countr_zero(status) - 4;
    assert((index >= 0) && (index < 4));
  } else {
    if (!Data::self(status)) {
      index = 4 + sleeps;
      assert((index >= 4) && (index < 12));
    } else {
      const auto s = status & 7;
      index = 12 + (s - 1);
      assert((index >= 12) && (index < 14));
    }
  }
  return index + 1;
}

using Volatiles = uint64_t;
using Stats = std::array<uint16_t, 5>;

struct ActiveKey {
  Volatiles volatiles;
  Stats stats;
  PokemonKey pokemon_key;

  bool operator<(const ActiveKey &other) const {
    return (pokemon_key < other.pokemon_key) || (volatiles < other.volatiles) ||
           (stats < other.stats);
  }
};

PokemonKey get_pokemon_key(const View::Side &side, const View::Duration &d,
                           auto i) {
  const auto &pokemon = side.pokemon(side.order(i) - 1);
  const auto sleeps = d.sleep(i);
  MoveKey move_key{};
  for (const auto ms : pokemon.moves()) {
    move_key |= (ms.pp > 0);
    move_key <<= 1;
  }
  const auto si =
      get_status_index(std::bit_cast<uint8_t>(pokemon.status()), sleeps);
  return PokemonKey{si, move_key};
}

ActiveKey get_active_key(const View::Side &side, const View::Duration &d) {
  return {*std::bit_cast<Volatiles *>(&side.active().volatiles()),
          *std::bit_cast<Stats *>(&side.active().stats()),
          get_pokemon_key(side, d, 0)}; // TODO this can probably be retrieved.
}

// Keys to use in the various pokemon/active caches for computing accumulator
// layer. Used for both cache lookup and tensor input (pokemon/active word net)
// construction in the event of a cache miss
struct BattleKeys {
  struct Side {
    // This array should reflect the raw layout of the battle object, ignoring
    // order
    std::array<PokemonKey, 6> pokemon_keys;
    ActiveKey active_key;
    std::array<uint8_t, 6> order;
    std::array<uint8_t, 6> hp;
  };

  Side p1;
  Side p2;

  BattleKeys(const pkmn_gen1_battle &battle,
             const pkmn_gen1_chance_durations &durations) {
    const auto &b = View::ref(battle);
    const auto &d = View::ref(durations);
    p1.active_key = get_active_key(b.side(0), d.duration(0));
    p2.active_key = get_active_key(b.side(1), d.duration(1));

    for (auto i = 0; i < 6; ++i) {
      p1.pokemon_keys[b.side(0).order(i) - 1] =
          get_pokemon_key(b.side(0), d.duration(0), i);
      p2.pokemon_keys[b.side(1).order(i) - 1] =
          get_pokemon_key(b.side(1), d.duration(1), i);
      const auto &pokemon1 = b.side(0).pokemon(b.side(0).order(i) - 1);
      const auto &pokemon2 = b.side(1).pokemon(b.side(1).order(i) - 1);
      p1.hp[i] = (255 * (uint32_t)pokemon1.hp()) / pokemon1.stats().hp();
      p2.hp[i] = (255 * (uint32_t)pokemon2.hp()) / pokemon2.stats().hp();
    }
    p1.order = b.side(0).order();
    p2.order = b.side(1).order();
  }

  void update(const View::Battle &battle, pkmn_choice c1, pkmn_choice c2) {
    // TODO?
    return;
  }
};

constexpr auto pokemon_in_dim = 198;
constexpr auto active_in_dim = 212;

using PokemonInput = std::array<float, pokemon_in_dim>;
using ActiveInput = std::array<float, active_in_dim>;

template <bool read_stats = true>
PokemonInput get_pokemon_input(const PokemonKey key,
                               const View::Pokemon &pokemon) {
  PokemonInput output;
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
    if ((bool)moves[i].id && (bool)moves[i].pp) {
      output[c + (uint8_t)moves[i].id - 1] = 1.0;
    }
  }
  c += 4;
  if (key.first) {
    output[c + (key.first - 1)] = 1.0;
  }
  c += 14;
  output[c + static_cast<int>(pokemon.types() % 16)] = 1.0;
  output[c + static_cast<int>(pokemon.types() / 16)] = 1.0;
  return output;
}

ActiveInput get_active_input(const ActiveKey &key,
                             const View::ActivePokemon &active,
                             const View::Pokemon &pokemon) {
  ActiveInput output;
  const auto p = get_pokemon_input(key.pokemon_key, pokemon);
  std::memcpy(output.data(), p.data(), sizeof(float) * pokemon_in_dim);
  const auto &volatiles = active.volatiles();
  auto c = pokemon_in_dim;
  output[c + 0] = volatiles.binding();
  output[c + 1] = volatiles.substitute();
  output[c + 2] = volatiles.recharging();
  output[c + 3] = volatiles.leech_seed();
  output[c + 4] = volatiles.toxic();
  output[c + 5] = volatiles.light_screen();
  output[c + 6] = volatiles.reflect();
  output[c + 7] = volatiles.substitute_hp();
  output[c + 8] = volatiles.toxic_counter();
  return output;
}

constexpr auto pokemon_out_dim = 39;
constexpr auto active_out_dim = 55;

using PokemonOutput = std::array<uint8_t, pokemon_out_dim>;
using ActiveOutput = std::array<uint8_t, active_out_dim>;

using PokemonNet = WordNet<pokemon_in_dim, 32, pokemon_out_dim>;
using ActiveNet = WordNet<active_in_dim, 32, active_out_dim>;

struct NNUECache {

  template <typename Net, typename Key, typename Value> struct WordCache {
    static constexpr bool is_active = std::is_same_v<Key, ActiveKey>;

    std::map<Key, Value> _cache;
    View::Pokemon *p;
    View::ActivePokemon *a;
    Net *net;

  public:
    WordCache() = default;
    WordCache(const auto &nn) : _cache{} {}

    const Value &operator[](const Key &key) {
      const auto it = _cache.find(key);
      if (it == _cache.end()) {
        const auto get_input = [this, key]() {
          if constexpr (is_active) {
            return get_active_input(key, *a, *p);
          } else {
            return get_pokemon_input(key, *p);
          }
        };
        const auto input = get_input();
        const auto output = net->propagate(input.data());
        _cache[key] = output;
      }
      return _cache[key];
    }

    size_t size() const { return _cache.size(); }
  };

  using PokemonCache = WordCache<PokemonNet, PokemonKey, PokemonOutput>;
  using ActiveCache = WordCache<ActiveNet, ActiveKey, ActiveOutput>;

  struct Slot {
    PokemonCache p_cache;
    ActiveCache a_cache;

    Slot() = default;

    Slot(auto &pokemon_net, auto &active_net)
        : p_cache{pokemon_net}, a_cache{active_net} {}

    void print_sizes() const {
      std::cout << p_cache.size() << ' ' << a_cache.size() << std::endl;
    }
  };

  struct SideCache {
    std::array<Slot, 6> slots;

    SideCache(auto &pokemon_net, auto &active_net) {
      slots.fill(Slot{pokemon_net, active_net});
    }

    void print_sizes() const {
      for (const auto &s : slots) {
        s.print_sizes();
      }
    }
  };

  // BattleKeys has the current info, we only need this for the 'base'
  View::Battle b;

  SideCache p1;
  SideCache p2;

  NNUECache(const pkmn_gen1_battle *battle, PokemonNet &pokemon_net,
            ActiveNet &active_net)
      : b{View::ref(*battle)}, p1{pokemon_net, active_net},
        p2{pokemon_net, active_net} {

    SideCache *side_ptr[2]{&p1, &p2};
    for (auto s = 0; s < 2; ++s) {
      const auto &side = b.side(s);
      for (int i = 0; i < 6; ++i) {
        auto &slot = side_ptr[s]->slots[i];
        slot.p_cache.p = &b.side(0).pokemon(i); // TODO order
        slot.p_cache.a = &b.side(0).active();
        slot.p_cache.net = &pokemon_net;
        slot.a_cache.p = &b.side(0).pokemon(i);
        slot.a_cache.a = &b.side(0).active();
        slot.a_cache.net = &active_net;
      }
    }
  }

  void accumulate(const BattleKeys &a, uint8_t *const output) {
    auto *const acc1 = output;
    auto *const acc2 = output + 256;
    acc1[0] = a.p1.hp[0];
    acc2[0] = a.p2.hp[0];
    std::memcpy(acc1 + 1,
                p1.slots[a.p1.order[0] - 1].a_cache[a.p1.active_key].data(),
                active_out_dim);
    std::memcpy(acc2 + 1,
                p2.slots[a.p2.order[0] - 1].a_cache[a.p2.active_key].data(),
                active_out_dim);
    for (auto i = 0; i < 5; ++i) {
      auto *const poke1 = acc1 + (active_out_dim + i * pokemon_out_dim);
      auto *const poke2 = acc2 + (active_out_dim + i * pokemon_out_dim);
      poke1[0] = a.p1.hp[i + 1];
      poke2[0] = a.p2.hp[i + 1];
      std::memcpy(poke1 + 1,
                  p1.slots[a.p1.order[i + 1] - 1]
                      .p_cache[a.p1.pokemon_keys[a.p1.order[i + 1] - 1]]
                      .data(),
                  pokemon_out_dim);
      std::memcpy(poke2 + 1,
                  p2.slots[a.p2.order[i + 1] - 1]
                      .p_cache[a.p2.pokemon_keys[a.p2.order[i + 1] - 1]]
                      .data(),
                  pokemon_out_dim);
    } // TODO all this order shit
  }
};

}; // namespace NNUE
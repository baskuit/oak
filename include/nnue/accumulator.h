#pragma once

#include <array>
#include <unordered_map>

#include <battle/view.h>

namespace NNUE {

using ProcessedStatus = uint8_t;
using MoveKey = uint8_t;
using PokemonKey = std::pair<ProcessedStatus, MoveKey>;

constexpr auto process_status(auto status, auto sleeps) {
    return std::bit_cast<ProcessedStatus>(status);
}

using Volatiles = uint64_t;
using Stats = std::array<uint16_t, 5>;

struct ActiveKey {
    Volatiles volatiles;
    Stats stats;
    PokemonKey pokemon_key;
};

using PokemonWord = std::array<uint8_t, 39>;
using ActiveWord = std::array<uint8_t, 53>;

struct WordCaches {

    struct Slot {
        std::unordered_map<PokemonKey, PokemonWord> p_cache;
        std::unordered_map<ActiveKey, ActiveWord> a_cache;
    };

    struct Side {
        std::array<Slot, 6> slots;
    };

    Side p1;
    Side p2;
};

PokemonKey get_pokemon_key(const View::Side& side, const View::Duration& d, auto i) {
    const auto& pokemon = side.pokemon(side.order()[i] - 1);
    const auto sleeps = d.sleep(i);
    MoveKey move_key{};
    for (auto i = 0; i < 4; ++i) {
        move_key |= (pokemon.moves()[i].pp > 0);
        move_key <<= 1;
    }
    process_status(std::bit_cast<uint8_t>(pokemon.status()), sleeps);
    return PokemonKey{move_key, move_key};
}

ActiveKey get_active_key(const View::Side& side, const View::Duration& d) {
    return {
        *std::bit_cast<Volatiles*>(&side.active().volatiles()),
        *std::bit_cast<Stats*>(&side.active().stats()),
        get_pokemon_key(side, d, 0)};
}

struct Abstract {
    struct Side {
        std::array<PokemonKey, 6> pokemon_keys;
        ActiveKey active_key;
        uint8_t active_slot;
    };
        
    Side p1;
    Side p2;

    void update(const View::Battle &battle, pkmn_choice c1, pkmn_choice c2) {
        return;
    }
};

};
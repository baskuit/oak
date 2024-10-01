#pragma once

// #include <pkmn.h>
#include <assert.h>

// #include ""

namespace Data{using Species = unsigned int; using Moves = unsigned int;};

using pkmn_gen1_battle = uint8_t[384];

namespace Eval {

    // We may just use Randbat data for this
    struct FullSet {
        Data::Species species;
        std::array<Data::Moves, 4> moves;

        void assert() const {
            assert(
                (moves[0] > moves[1] || moves[1] == 0)  &&
                (moves[1] > moves[2] || moves[2] == 0)  &&
                (moves[2] > moves[3] || moves[3] == 0);
            )
        }
    };

    using SpeciesMoveSetID = uint64_t;
    static constexpr uint64_t MAX_SPECIES_MOVE_SETS{115418636672}; // 166^4 * 152

    template <typename Container>
    constexpr SpeciesMoveSetID getID(const Data::Species species, const Container& container) {
        auto id = species;
        for (int i = 0; i < 4; ++i) {
            id *= 166;
            id += static_cast<SpeciesMoveSetID>(container[i]);
        }
        return id;
    }

    struct BasicMatchupEntry {
        float value;
        float variance;
    };

    // Holds values/variance for say any two species
    struct SpeciesMatchupTable {
        BasicMatchupEntry data[3][3]; 
    };


};
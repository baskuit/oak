#pragma once

#include "data.h"

namespace Client {
    
    struct Battle;

    struct Durations {
        uint8_t sleep;
    };

    struct OurSide {
        struct Active{
            uint16_t stats{};
            
        };

        Active active;
    };

    struct OppSide {
        struct Active {

        };
        struct Bench {
            Data::Species species;
            std::array<Data::Moves, 4> moves;
        };

        Active active;
        std::array<Bench, 6> bench;
    };

    struct Battle {
        OurSide our_side;
        OppSide opp_side;

        // get libpkmn battle
        template <typename State>
        State reify() const {
            return {};
        }

        void update (pkmn_choice action, auto turn_log) {}

        // one ctor to read from present info only. called at the start but if we detect our info is no good somehow, then we can construct a new client representation
        Battle() = default;
    };
};
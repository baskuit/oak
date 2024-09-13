#include <mutex>

#include <pkmn.h>

#include <types/array.h>
#include <types/random.h>

struct Types {
    using Real = float;
    template <typename T>
    using Vector = ArrayBasedVector<9>::Vector<T, uint32_t>;
    using PRNG = prng;
    using Mutex = std::mutex;

    struct State {
        pkmn_gen1_battle battle;
        pkmn_result result;
        pkmn_gen1_battle_options options;
    };

    struct Model {
        float inference(State&& state) {
            return 0;
        }
    }
};

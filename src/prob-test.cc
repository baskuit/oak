#include "../include/prob-test.hh"

#include "../include/battle.hh"
#include "../include/sides.hh"

// normal rollout but also performs prob_test on each update
template <typename Types>
bool rollout_battle_with_prob_test(const size_t tries, typename Types::PRNG &device, typename Types::State &state,
                                   const typename Types::Prob min_explored) {
    while (!state.is_terminal()) {
        const int row_idx = device.random_int(state.row_actions.size());
        const int col_idx = device.random_int(state.col_actions.size());
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];
        typename Types::Prob total_prob = prob_test<Types>(device, tries, state, row_action, col_action);
        if (total_prob < min_explored) {
            return false;
        }
        state.apply_actions(row_action, col_action);
        state.get_actions();
    }
    return true;
}

// generates random 6v6's from secret sides.hh; clamps if possible
template <typename Types>
typename Types::State generator(typename Types::PRNG &device) {
    // typename Types::State state{sides[1 + device.random_int(20)], sides[1 + device.random_int(20)]};
    typename Types::State state{sides[0], sides[0]};
    state.apply_actions(0, 0);
    state.get_actions();
    if constexpr (Types::State::dcalc) {
        state.clamped = true;
    }
    return state;
}

// simply loops the rollout test
template <typename Types>
bool large_rollout_test(const size_t n_battles, const size_t chance_tries, typename Types::PRNG &device,
                        typename Types::Prob min_explored) {
    for (size_t i{}; i < n_battles; ++i) {
        typename Types::State state = generator<Types>(device);
        const bool success = rollout_battle_with_prob_test<Types>(chance_tries, device, state, min_explored);
        if (!success) {
            return false;
        }
    }
    return true;
}

int main() {
    using Types = Battle<64, 3, ChanceObs, mpq_class, mpq_class>;
    const size_t n_battles{1 << 10};
    const size_t chance_tries{1 << 8};
    prng device{0};
    typename Types::Prob min_explored{typename Types::Q{1, 100}};
    large_rollout_test<Types>(n_battles, chance_tries, device, min_explored);
    return 0;
}

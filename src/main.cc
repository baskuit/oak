#include "../include/battle.hh"
#include "../include/prob-test.hh"
#include "../include/sides.hh"

int main() {
    prng device{0};
    using T = Battle<64, 3, ChanceObs, mpq_class, mpq_class>;

    T::State state{sides[0], sides[0]};

    state.clamped = true;
    state.apply_actions(0, 0);
    state.get_actions();
    pkmn_choice action = state.row_actions[0];
    mpq_class total_prob = prob_test<T>(device, 1 << 5, state, action, action);
    std::cout << total_prob.get_str() << " ~ " << total_prob.get_d() << std::endl;
    return 0;
}
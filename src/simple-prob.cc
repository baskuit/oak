#include "../include/battle.hh"
#include "../include/sides.hh"

int main() {
    using Types = Battle<64, 3, ChanceObs, mpq_class, mpq_class>;
    prng device{0};
    Types::State state{sides[0], sides[0]};
    state.clamped = true;
    state.apply_actions(0, 0);
    state.get_actions();
    state.randomize_transition(device);
    state.apply_actions(state.row_actions[0], state.col_actions[0]);
    std::cout << state.prob.get_str() << std::endl;
    std::cout << state.prob.get_d() << std::endl;
}
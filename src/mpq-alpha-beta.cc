#include "../include/battle.hh"
#include "../include/debug-log.hh"
#include "../include/old-battle.hh"
#include "../include/sides.hh"

int main() {
    prng device{1121256};
    using T = MonteCarloModel<Battle<0, 3, ChanceObs, float, float>>;

    using U = AlphaBetaForce<T>;
    T::State state{sides[0], sides[0]};
    state.clamped = true;
    DebugLog<T::State> debug_log{state};

    state.apply_actions(0, 0);
    state.get_actions();

    std::cout << "rows: " << state.row_actions.size() << " cols: " << state.col_actions.size() << std::endl;

    U::Model model{0};
    U::ModelOutput output{};

    U::Search search{0, 1 << 10, 0.0f};
    U::MatrixNode node{};
    search.run(1, device, state, model, node);

    std::cout << math::to_float(node.alpha) << ' ' << math::to_float(node.beta) << std::endl;
    node.chance_data_matrix.print();
    return 0;
}
#include "../include/battle.hh"
#include "../include/debug-log.hh"
#include "../include/old-battle.hh"

#include "../include/sides.hh"

int main()
{
    prng device{1121256};
    using T = MonteCarloModel<
        Battle<64, 3, ChanceObs, mpq_class, mpq_class>>;
    using U = AlphaBetaForce<T>;
    T::State
        state{sides[0], sides[0]};
    DebugLog<T::State> debug_log{state};

    state.apply_actions(0, 0);
    state.get_actions();
    state.clamped = true;

    std::cout << "rows: " << state.row_actions.size() << " cols: " << state.col_actions.size() << std::endl;

    U::Model model{0};
    U::ModelOutput output{};

    U::Search search{0, 1 << 10 , mpq_class{0}};
    U::MatrixNode node{};
    search.run(1, device, state, model, node);

    std::cout << node.alpha.get_d() << ' ' << node.beta.get_d() << std::endl;
    node.chance_data_matrix.print();
    return 0;
}
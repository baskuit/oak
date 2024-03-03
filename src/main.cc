#include "../include/battle.hh"
#include "../include/debug-log.hh"
#include "../include/old-battle.hh"

#include "../include/sides.hh"


int main()
{
    prng device{0};
    using T = MonteCarloModel<
        Battle<64, 0, ChanceObs, mpq_class, mpq_class>>;
    using U = AlphaBetaIter<T>;
    T::State
        state{sides[3], sides[4]};
    DebugLog<T::State> debug_log{state};

    state.apply_actions(0, 0);
    state.get_actions();

    U::Model model{0};
    U::ModelOutput output{};

    U::Search search{0, 1 << 7, mpq_class{0}};
    U::MatrixNode node{};
    search.run(1, device, state, model, node);

    return 0;
}
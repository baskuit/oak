#include "../include/battle.hh"
#include "../include/debug-log.hh"
#include "../include/old-battle.hh"

#include "../include/sides.hh"

/*

Rolls out a random battle using the eval log encoding

*/

int main()
{
    prng device{1121256};
    using T = MonteCarloModel<
        Battle<64, 3, ChanceObs, float, float>>;
    using U = AlphaBetaForce<T>;
    T::State
        state{sides[1], sides[0]};
    state.clamped = true;
    DebugLog<T::State> debug_log{state};

    using RowModelTypes = SearchModel<TreeBandit<Exp3<T>>>;
    using ColModelTypes = SearchModel<TreeBandit<Exp3<T>>>;
    RowModelTypes::Model row_model{1 << 12, prng{123}, {prng{123}}, {}};
    ColModelTypes::Model col_model{1 << 12, prng{321}, {prng{321}}, {}};
    
    rollout_with_eval_debug<T::State, RowModelTypes, ColModelTypes>(state, row_model, col_model, debug_log);

    debug_log.print();
}
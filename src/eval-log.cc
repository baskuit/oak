#include "../include/eval-log.hh"

#include "../include/battle.hh"
#include "../include/mc-average.hh"
#include "../include/old-battle.hh"
#include "../include/sides.hh"

/*

Rolls out a random battle using the eval log encoding

*/

int main() {
    prng device{1121256};
    using T = MonteCarloModelAverage<Battle<0, 3, ChanceObs, float, float>>;
    using U = AlphaBetaForce<T>;
    T::State state{sides[1], sides[2]};
    state.clamped = true;
    EvalLog<T::State> debug_log{state};

    using RowModelTypes = SearchModel<TreeBandit<MatrixUCB<T>>, false, true, true, false>;
    using ColModelTypes = SearchModel<TreeBandit<MatrixUCB<T>>, false, true, true, false>;
    RowModelTypes::Model row_model{1000, prng{123}, {prng{123}, 1 << 3}, {2.0, .1}};
    ColModelTypes::Model col_model{1000, prng{321}, {prng{321}, 1 << 3}, {2.0, .1}};

    rollout_with_eval_debug<T::State, RowModelTypes, ColModelTypes>(state, row_model, col_model, debug_log);

    debug_log.print();
    debug_log.save(state);
}
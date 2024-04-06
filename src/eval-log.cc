#include "../include/eval-log.hh"

#include "../include/battle.hh"
#include "../include/clamp.hh"
#include "../include/mc-average.hh"
#include "../include/old-battle.hh"
#include "../include/sides.hh"
/*

Rolls out a random battle using the eval log encoding

*/

int main() {
    prng device{};

    // iter, tree bandit, policy ,value, empirical
    using M = MonteCarloModel<Battle<0, 3, ChanceObs, float, float>>;
    using UCBEmpirical = Clamped<SearchModel<TreeBandit<UCB<M>>, true, true, true, true, true>>;
    using UCBArgMax = Clamped<SearchModel<TreeBandit<UCB<M>>, true, true, true, true, false>>;

    M::State state{sides[1], sides[2]};
    state.randomize_transition(device);
    EvalLog<M::State> debug_log{state};

    const size_t iterations = 1 << 20;
    UCBEmpirical::Model row_model{iterations, prng{device.uniform_64()}, prng{device.uniform_64()}, {2.0}};
    UCBArgMax::Model col_model{iterations, prng{device.uniform_64()}, prng{device.uniform_64()}, {2.0}};

    rollout_with_eval_debug<M::State, UCBEmpirical, UCBArgMax>(state, row_model, col_model, debug_log);

    debug_log.print();
    debug_log.save(state);
}
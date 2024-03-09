#include "../include/battle.hh"
#include "../include/clamp.hh"
#include "../include/debug-log.hh"
#include "../include/mc-average.hh"
#include "../include/sides.hh"

template <typename Types>
void self_play_loop(typename Types::PRNG *device_, typename Types::Model *model_) {
    typename Types::PRNG device{device_->uniform_64()};
    typename Types::Model model{*model_};

    const int total = 250000;

    for (int i = 0; i < total; ++i) {
        // don't use mono tauros
        const int r = device.random_int(n_sides - 1);
        const int c = device.random_int(n_sides - 1);
        typename Types::State state{sides[r + 1], sides[c + 1]};
        state.randomize_transition(device);
        DebugLog<typename Types::State> debug_log{state};
        self_play_rollout_with_eval_debug<typename Types::State, Types>(device, state, model, debug_log);
        debug_log.save(state);
    }
}

int main() {
    prng device{1030824};
    using T = AlphaBetaForce<MonteCarloModelAverage<Battle<0, 3, ChanceObs, float, float>>>;
    using U = Clamped<SearchModel<T, false, true, false>>;

    U::Model model{1, prng{234723875}, {prng{12312}, 1 << 3}, {0, 1 << 8, 0.0f}};

    constexpr int n_threads = 8;
    std::thread thread_pool[n_threads];
    for (int t{}; t < n_threads; ++t) {
        thread_pool[t] = std::thread(&self_play_loop<U>, &device, &model);
    }
    for (int t{}; t < n_threads; ++t) {
        thread_pool[t].join();
    }

    return 0;
}
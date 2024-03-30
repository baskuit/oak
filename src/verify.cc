#include "../include/battle.hh"
#include "../include/clamp.hh"
#include "../include/eval-log.hh"
#include "../include/mc-average.hh"
#include "../include/sides.hh"
#include "../include/analysis.hh"

/*

Does basically the same things as ab-self-play but with deeper search

*/

template <typename State>
State generator(prng &device, const int max_alive_side, const float use_prob = .33) {
    /*
    uses n_alive to create search-tractable states
    */
    const int r = device.random_int(n_sides - 1);
    const int c = device.random_int(n_sides - 1);
    State state{sides[r + 1], sides[c + 1]};
    state.randomize_transition(device);

    int m = 6;
    int n = 6;
    int rows = 1;
    int cols = 1;
    while (!state.is_terminal()) {
        const int row_idx = device.random_int(rows);
        const int col_idx = device.random_int(cols);
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];
        state.apply_actions(row_action, col_action);
        state.get_actions();
        rows = state.row_actions.size();
        cols = state.col_actions.size();
        m = n_alive_side(state.battle.bytes, 0);
        n = n_alive_side(state.battle.bytes, 1);

        if (m <= max_alive_side && n <= max_alive_side && (rows > 1 || cols > 1) && device.uniform() < use_prob) {
            state.clamped = true;
            return state;
        }
    }

    return generator<State>(device, max_alive_side);
}

template <typename Types>
void self_play_loop(typename Types::PRNG *device_, typename Types::Model *model_, const int max_positions) {
    typename Types::PRNG device{device_->uniform_64()};
    typename Types::Model model{*model_};
    AnalysisData analysis_data{};

    for (int i = 0; i < max_positions; ++i) {
        typename Types::State state = generator<typename Types::State>(device, 3);
        std::cout << "generated!" << std::endl;
        std::cout << state.row_actions.size() << ' ' << state.col_actions.size() << std::endl;

        analysis_data.push<Types>(std::move(state), model);
    }

    analysis_data.print();
}

int main() {
    prng device{1030824};
    using T = AlphaBetaForce<MonteCarloModelAverage<Battle<0, 3, ChanceObs, float, float>>>;
    using U = Clamped<SearchModel<T, false, true, false>>;

    // depth 3 alpha beta
    const size_t search_depth = 3;
    U::Model model{search_depth, prng{234723875}, {prng{12312}, 1 << 5}, {0, 1 << 10, 0.0f}};

    constexpr int n_threads = 1;
    constexpr int max_positions = 1;
    std::array<std::thread, n_threads> threads{};
    for (int t{}; t < n_threads; ++t) {
        threads[t] = std::thread(&self_play_loop<U>, &device, &model, max_positions);
    }
    for (int t{}; t < n_threads; ++t) {
        threads[t].join();
    }

    return 0;
}
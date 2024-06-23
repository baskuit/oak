#include "../include/battle.hh"
#include "../include/sides.hh"

/*

Is depth 4 alpha beta possible?

*/

template <typename State>
State generator(prng &device, const int max_alive_side, const float use_prob = .33) {
    /*
    uses n_alive to create search-tractable states
    */

    const int r = device.random_int(n_sides);
    const int c = device.random_int(n_sides);
    State state{sides[r], sides[c]};
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

int foo(prng &device) {
    constexpr bool debug_print = false;
    using Types = AlphaBetaRefactor<MonteCarloModel<Battle<0, 3, ChanceObs, mpq_class, mpq_class>>, debug_print>;

    Types::State state = generator<Types::State>(device, 3);
    Types::Model model{device.uniform_64()};
    Types::MatrixNode node{};
    Types::Search search{1, 1 << 3};

    const size_t depth = 4;
    const auto output = search.run(depth, device, state, model, node);

    std::cout << "alpha: " << output.alpha.get_d() << " beta: " << output.beta.get_d() << std::endl;
    std::cout << "counts: ";
    for (const auto c : output.counts) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;
    std::cout << "times: ";
    for (const auto c : output.times) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;

    return 0;
}

int main () {
    prng device{11213409256};

    std::vector<char> x(1);
    for (const auto y : x) {
        foo(device);
    }
}
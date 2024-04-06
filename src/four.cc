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

int main() {
    using AB = AlphaBetaForce<MonteCarloModel<Battle<0, 3, ChanceObs, float, float>>>;

    prng device{1121256};

    AB::State state = generator<AB::State>(device, 3);

    std::cout << "rows: " << state.row_actions.size() << " cols: " << state.col_actions.size() << std::endl;

    AB::Model model{device.uniform_64()};
    AB::Search search{1, 1 << 8, 0.1f};
    AB::MatrixNode node{};
    search.run(4, device, state, model, node);

    std::cout << node.count_matrix_nodes() << std::endl;
    return 0;
}
#include "../include/battle.hh"
#include "../include/sides.hh"
#include "../include/mc-average.hh"

/*

Is depth 4 alpha beta possible?

*/

template <typename State, typename Output>
void print_output (const State& state, const Output& output) {
    std::cout << "alpha: " << output.alpha << " beta: " << output.beta << std::endl;
    std::cout << "terminal: ";
    for (const auto c : output.terminal_count) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;
    std::cout << "inference: ";
    for (const auto c : output.inference_count) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;
    std::cout << "alpha_beta: ";
    for (const auto c : output.alpha_beta_count) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;
    std::cout << "counts: ";
    for (const auto c : output.matrix_node_count) {
        std::cout << c << ", ";
    } 
    std::cout << std::endl;
    std::cout << "times: ";
    for (const auto c : output.times) {
        std::cout << (double)c/1000 << ", ";
    } 
    std::cout << std::endl;
    std::cout << "times w.o.i: ";
    for (const auto c : output.times_without_inference) {
        std::cout << (double)c/1000 << ", ";
    } 
    std::cout << std::endl;

    std::cout << "row strategy:";
    for (int i = 0; i < state.row_actions.size(); ++i) {
        int x = state.row_actions[i];
        std::cout << "( " << int(x /4) << "|" << x % 4 << ") = " << output.row_strategy[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "col strategy:";
    for (int i = 0; i < state.col_actions.size(); ++i) {
        int x = state.col_actions[i];
        std::cout << "( " << int(x /4) << "|" << x % 4 << ") = " << output.col_strategy[i] << ", ";
    }
    std::cout << std::endl;

    // std::cout << "total_solves: " << std::endl;
    // for (int i = 0; i < 9; ++i) {
    //     for (int j = 0; j <= i; ++j) {
    //         std::cout << i + 1 << "," << j + 1 << " = " << output.total_solves[i][j] << " | ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "total_solves_raw: " << output.total_solves_raw << std::endl;
}

template <typename Types>
struct BattleSearchModel : Types {
    struct ModelOutput {
        ConstantSum<1, 1>::Value<float> value;
    };

    class Model {
        using S = Battle<0, 3, ChanceObs, float, float>;
        using T = TreeBanditFlat<Exp3<MonteCarloModel<S>>>;
    public:

        prng device;
        T::Search search{};
        T::Model model;

        Model (const prng &device) : device{device}, model{this->device.uniform_64()} {}

        void inference (Types::State &&state, ModelOutput &output) {
            search.run_for_iterations(1 << 7, device, state, model);
            search.get_empirical_value(search.matrix_data[0].stats, output.value);
        }

    };
};

template <typename State>
State generator(prng &device, const int max_alive_side, const float use_prob = .33) {
    /*
    uses n_alive to create search-tractable states
    */

    const int r = device.random_int(n_sides);
    const int c = device.random_int(n_sides);
    State state{sides[r], sides[c]};
    state.randomize_transition(device);

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
        int m = n_alive_side(state.battle.bytes, 0);
        int n = n_alive_side(state.battle.bytes, 1);

        if (m <= max_alive_side && n <= max_alive_side && (rows > 1 && cols > 1) && device.uniform() < use_prob) {
            state.clamped = true;
            return state;
        }
    }

    return generator<State>(device, max_alive_side);
}

template <typename T>
void print_matrix_data (const T& matrix_data) {
    const int rows = matrix_data.rows;
    const int cols = matrix_data.cols;

    const auto tostring = [](int x) {
        if (x == 1000) {
            --x;
        } 
        std::string a = std::to_string(x);
        while (a.size() < 3) {
            a = "0" + a;
        }
        return a;
    };

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const auto [a, b, c, d] = matrix_data.data[i][j];
            const int x = a * 1000;
            const int y = b * 1000;
            const int z = c * 1000;
            std::cout << "(" << tostring(x) << " " << tostring(y) << "|" << tostring(z) << " " << tostring(d) << ") ";
        }
        std::cout << std::endl;
    }
}

const int max_alive = 5;
const size_t depth = 3;
const size_t min_tries = 1;
const size_t max_tries = 1 << 8;
const double max_unexplored = .01;
const double min_chance_prob = 0;

template <typename Types>
void bar (const prng& device_) {
    prng device{device_};
    typename Types::State state = generator<typename Types::State>(device, max_alive);
    typename Types::Model model{device.uniform_64()};
    typename Types::MatrixNode node{};
    typename Types::Search search{min_tries, max_tries, max_unexplored, min_chance_prob};

    const auto output = search.run(depth, device, state, model, node);
    print_output(state, output);

    print_matrix_data(output.matrix_print_data);
}

template <typename Types>
void bar2 (const prng& device_) {
    prng device{device_};
    typename Types::State state = generator<typename Types::State>(device, max_alive);
    typename Types::Model model{device.uniform_64(), 1 << 7};
    typename Types::MatrixNode node{};
    typename Types::Search search{min_tries, max_tries, max_unexplored, min_chance_prob};

    const auto output = search.run(depth, device, state, model, node);
    print_output(state, output);

    print_matrix_data(output.matrix_print_data);
}

int state_count = 0;

int foo(prng &device) {
    constexpr bool debug_print = false;
    using T = AlphaBetaRefactor<MonteCarloModel<Battle<0, 3, ChanceObs, float, float>>, debug_print>;
    using U = AlphaBetaRefactor<BattleSearchModel<Battle<0, 3, ChanceObs, float, float>>, debug_print>;
    using V = AlphaBetaRefactor<MonteCarloModelAverage<Battle<0, 3, ChanceObs, float, float>>, debug_print>;
    std::cout << "\nPOSITION: " << (++state_count) << std::endl;
    prng foo_device{device.uniform_64()};   

    // std::cout << "Monte Carlo at leaf nodes:" << std::endl;
    // bar<T>(foo_device);
    // std::cout << "Tree Search at leaf nodes:" << std::endl;
    // bar<U>(foo_device);
    std::cout << "Monte Carlo Average at leaf nodes:" << std::endl;
    bar2<V>(foo_device);


    return 0;
}

int main () {
    prng device{950356747984600};

    std::vector<char> x(3);
    for (const auto y : x) {
        foo(device);
    }
    return 0;
}
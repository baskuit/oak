#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <pinyon.h>

#include "../include/alpha-beta-refactor.h"
#include "../include/battle.h"
#include "../include/exp3.h"
#include "../include/sides.h"

std::mutex MUTEX{};
size_t N = 0;
size_t DOUBLE_Q = 0;
bool STOP = false;

std::ofstream OUTPUT_FILE{"ab-vs-mcts.txt", std::ios::out | std::ios::trunc};

using ModelTypes = PModel<Battle<0, 3, ChanceObs, float, float>>;
using ABTypes = AlphaBetaRefactor<ModelTypes>;
using MCTSTypes = TreeBanditRootMatrix<Exp3Oak<ModelTypes>>;

constexpr int battle_size{384};
constexpr int n_sides{100};

void read_battle_bytes(std::array<uint8_t, battle_size> &bytes,
                       pkmn_result &result) {
  uint32_t byte;
  for (int i = 0; i < battle_size; ++i) {
    std::cin >> byte;
    bytes[i] = static_cast<uint8_t>(byte);
  }
}

struct DualSearchOutput {
  std::vector<float> mcts_row_policy;
  std::vector<float> mcts_col_policy;
  std::vector<float> ab_row_policy;
  std::vector<float> ab_col_policy;
};

template <typename Vec> void print(const Vec &vec) {
  for (const auto x : vec) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
}

DualSearchOutput dual_search(const ModelTypes::State &state,
                             const uint64_t seed = 13423546425745) {
  const size_t rows = state.row_actions.size();
  const size_t cols = state.col_actions.size();

  ModelTypes::PRNG device{seed};
  ModelTypes::Model model{};

  size_t ab_search_time{};
  size_t ab_search_matrix_node_count{};

  DualSearchOutput ds_output{};
  if (rows > 1) {
    const size_t depth = 3;
    const size_t min_tries = 1;
    const size_t max_tries = 1 << 7;
    const float max_unexplored = .1;
    const float min_reach_prob_initial = 1;
    const float min_reach_prob_base = 1 / 4.0;

    ABTypes::Search search{min_tries, max_tries, max_unexplored,
                           min_reach_prob_initial, min_reach_prob_base};
    ABTypes::MatrixNode node{};
    const auto output = search.run(depth, device, state, model, node);
    for (const size_t t : output.times) {
      ab_search_time += t;
    }
    for (const size_t c : output.matrix_node_count) {
      ab_search_matrix_node_count += c;
    }
    ds_output.ab_row_policy = std::move(output.row_strategy);
    ds_output.ab_col_policy = std::move(output.col_strategy);
  } else {
    ds_output.ab_row_policy = {1};
    ds_output.mcts_row_policy = {1};
  }

  if (cols > 1) {
    MCTSTypes::Search search{};
    MCTSTypes::MatrixNode node{};
    search.run(ab_search_time / 1000, device, state, model, node);

    FastInput input;
    input.rows = rows;
    input.cols = cols;
    input.den = 80;
    std::vector<int> solve_matrix_data{};
    solve_matrix_data.resize(rows * cols);

    int idx{};
    for (const auto &pair : search.empirical_matrix) {
      size_t n = pair.first;
      if (n == 0) {
        n += 1;
      }
      const float q = pair.second / n;
      const int q_disc = 80 * q;
      solve_matrix_data[idx] = q_disc;
      ++idx;
    }

    FloatOneSumOutput output;
    ds_output.mcts_row_policy.resize(rows + 2);
    ds_output.mcts_col_policy.resize(cols + 2);
    input.data = solve_matrix_data.data();
    output.row_strategy = ds_output.mcts_row_policy.data();
    output.col_strategy = ds_output.mcts_col_policy.data();

    solve_fast(&input, &output);
    ds_output.mcts_row_policy.resize(rows);
    ds_output.mcts_col_policy.resize(cols);
  } else {
    ds_output.ab_col_policy = {1};
    ds_output.mcts_col_policy = {1};
  }

  return ds_output;
}

void compare(int i, int j, uint64_t seed) {
  std::cout << i << ' ' << j << std::endl;
  ModelTypes::PRNG device{seed};

  ModelTypes::State state{sides[i], sides[j]};
  state.apply_actions(0, 0);
  state.randomize_transition(device);
  state.get_actions();
  while (!state.is_terminal()) {
    state.clamped = true;
    const auto output = dual_search(state, device.uniform_64());
    state.clamped = false;

    const int a = device.sample_pdf(output.ab_row_policy);
    const int b = device.sample_pdf(output.mcts_col_policy);
    // MUTEX.lock();
    // std::cout << "selected: " << a << ' ' << b << std::endl;
    // math::print(output.ab_row_policy);
    // math::print(output.mcts_col_policy);
    // MUTEX.unlock();
    const auto row_action = state.row_actions[a];
    const auto col_action = state.col_actions[b];
    state.apply_actions(row_action, col_action);
    state.get_actions();
  }
  MUTEX.lock();
  N += 1;
  DOUBLE_Q += (2 * state.get_payoff().get_row_value());
  MUTEX.unlock();
}

void loop_compare(const uint64_t seed) {
  ModelTypes::PRNG device{seed};

  while (!STOP) {
    const int i = device.random_int(n_sides);
    const int j = device.random_int(n_sides);
    const uint64_t seed = device.uniform_64();
    compare(i, j, seed);
    compare(j, i, seed);
  }
}

int main() {

  ModelTypes::PRNG device{32847948763498673};

  constexpr size_t threads = 24;

  std::thread thread_pool[threads];
  for (int i = 0; i < threads; ++i) {
    thread_pool[i] = std::thread(&loop_compare, device.uniform_64());
  }

  for (int i = 0; i < 60 * 24 * 2; ++i) {
    sleep(60);
    const float v = DOUBLE_Q / 2.0 / (N == 0 ? 1 : N);
    std::cout << "(v: " << v << " N: " << N << " Q: " << DOUBLE_Q / 2.0 << ")"
              << std::endl;
    OUTPUT_FILE << std::to_string(v);
    OUTPUT_FILE.flush();
  }
  OUTPUT_FILE.close();
  STOP = true;

  for (int i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return 0;
}

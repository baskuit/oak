#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <pinyon.h>

#include "../include/alpha-beta-refactor.h"
#include "../include/battle.h"

using ModelTypes = PModel<Battle<0, 3, ChanceObs, float, float>>;
using ABTypes = AlphaBetaRefactor<ModelTypes>;
using MCTSTypes = TreeBanditRootMatrix<Exp3<ModelTypes>>;

static constexpr int battle_size{384};

void read_battle_bytes(std::array<uint8_t, battle_size> &bytes, pkmn_result& result) {
  uint32_t byte;
  for (int i = 0; i < battle_size; ++i) {
    std::cin >> byte;
    bytes[i] = static_cast<uint8_t>(byte);
  }
  uint32_t result_int;
  std::cin >> result_int;
  result = static_cast<uint8_t>(result_int);
  std::cout << "!result: " << result_int << std::endl;
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

DualSearchOutput dual_search(const ModelTypes::State &state) {
  const size_t rows = state.row_actions.size();
  const size_t cols = state.col_actions.size();

  ModelTypes::PRNG device{13423546425745};
  ModelTypes::Model model{};

  size_t ab_search_time{};
  size_t ab_search_matrix_node_count{};

  DualSearchOutput ds_output{};

  {
    ABTypes::Search search{1 << 3, 1 << 7};
    ABTypes::MatrixNode node{};
    const auto output = search.run(3, device, state, model, node);
    for (const size_t t : output.times) {
      ab_search_time += t;
    }
    for (const size_t c : output.matrix_node_count) {
      ab_search_matrix_node_count += c;
    }
    ds_output.ab_row_policy = std::move(output.row_strategy);
    ds_output.ab_col_policy = std::move(output.col_strategy);
  }

  std::cout << '!' << ab_search_time / 1000000.0 << std::endl;

  {
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
      float q = pair.second / n;
      int q_disc = 80 * q;
      solve_matrix_data[idx] = q_disc;
      ++idx;
    }

    FloatOneSumOutput output;
    ds_output.mcts_row_policy.resize(rows);
    ds_output.mcts_col_policy.resize(cols);
    input.data = solve_matrix_data.data();
    output.row_strategy = ds_output.mcts_row_policy.data();
    output.col_strategy = ds_output.mcts_col_policy.data();
    solve_fast(&input, &output);
  }

  return ds_output;
}

int main() {

  std::array<uint8_t, battle_size> bytes{};
  pkmn_result result{};

  while (true) {
    std::cout << "!newloop" << std::endl;
    read_battle_bytes(bytes, result);

    ModelTypes::State state{bytes.data(), bytes.data() + 184};
    state.result = result;
    state.result_kind = pkmn_result_type(result);
    state.clamped = true;
    state.get_actions();

    const auto output = dual_search(state);

    print(output.ab_row_policy);
    print(output.ab_col_policy);
    print(output.mcts_row_policy);
    print(output.mcts_col_policy);
  }

  return 0;
}
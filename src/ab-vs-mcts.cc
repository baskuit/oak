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

void read_battle_bytes(std::array<uint8_t, battle_size> &bytes) {
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

  std::array<uint8_t, battle_size> bytes{230,0,133,0,133,0,125,0,165,0,79,24,14,48,75,40,34,24,230,0,0,1,58,100,218,0,139,0,121,0,165,0,135,0,126,8,83,24,163,32,68,32,218,0,0,4,136,100,228,0,131,0,165,0,121,0,135,0,57,24,59,8,34,24,156,16,228,0,0,7,153,100,210,0,145,0,95,0,215,0,135,0,85,24,86,32,57,24,69,32,210,0,0,25,187,100,200,0,147,0,105,0,179,0,85,0,162,16,34,24,59,8,85,24,200,0,0,19,0,100,220,0,125,0,115,0,147,0,105,0,38,24,98,48,17,56,119,32,220,0,0,16,32,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,0,0,34,1,235,0,225,0,255,0,175,0,34,24,63,8,59,8,89,16,34,1,0,128,0,100,128,2,45,0,45,0,135,0,245,0,115,32,69,32,135,16,86,32,128,2,0,113,0,100,204,1,255,0,165,0,95,0,165,0,34,24,115,32,156,16,58,16,204,1,0,143,0,100,74,1,225,0,205,0,145,0,29,1,79,24,94,16,153,8,38,24,74,1,0,103,202,100,4,1,185,0,205,0,9,1,235,0,105,32,86,32,59,8,85,24,4,1,0,121,201,100,250,0,135,0,125,0,19,1,49,1,94,16,69,32,86,32,105,32,250,0,0,65,204,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,0,0,0,0,0,0,0,0,0,0,4,0,3,0,2,0,1,0};

  while (true) {
    // read_battle_bytes(bytes);

    ModelTypes::State state{bytes.data(), bytes.data() + 184};
    state.apply_actions(0, 0);
    state.get_actions();

    const auto output = dual_search(state);

    print(output.ab_row_policy);
    print(output.ab_col_policy);
    print(output.mcts_row_policy);
    print(output.mcts_col_policy);
  break;
  }

  return 0;
}
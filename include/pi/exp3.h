#pragma once

#include <cstdint>
#include <array>

namespace Exp3 {
#pragma pack(1)
struct Exp3Entry {
  float gain;
  uint32_t visits;
};
#pragma pop()

struct Exp3EntryTest {
  static_assert(sizeof(Exp3Entry) == 8);
};

struct RootExp3Data;

class Exp3Data {
public:
  std::array<Exp3Entry, 9> row_ucb_data;
  std::array<Exp3Entry, 9> col_ucb_data;
  // top bits of hash not used for table index
  uint8_t collision[5];
  // padded handle in case of collision. the top bits here dont matter - we can
  // increment normally and just mod/and to get the real handle
  uint32_t overflow;
  // index in row_cols_map
  uint8_t row_col_index;

public:
  void init(uint8_t rows, uint8_t cols) {
    uint8_t prod = rows * cols;
    uint8_t flip = rows > cols;
    row_col_index = (prod - 1) + 81 * flip;
  }

  // naive update function that uses float conversion
  template <typename Outcome> void update(const Outcome &outcome) {
    {
      Exp3Entry &entry = row_ucb_data[outcome.row_idx];
      float value = entry.value / 255.0;
      value *= entry.visits++;
      value += outcome.value;
      assert(entry.visits != 0);
      value /= entry.visits;
      entry.value = static_cast<uint8_t>(value * 255);
    };
    {
      Exp3Entry &entry = col_ucb_data[outcome.col_idx];
      float value = entry.value / 255.0;
      value *= entry.visits++;
      value += outcome.value;
      assert(entry.visits != 0);
      value /= entry.visits;
      entry.value = static_cast<uint8_t>(value * 255);
    };
  }

  template <typename Outcome>
  void select(const RootExp3Data &root_node, Outcome &outcome) {
    const auto [rows, cols] = Precomputed::row_col_map[row_col_index];

    int N = 0;
    for (auto i = 0; i < rows; ++i) {
      N += row_ucb_data[i].visits;
    }
    const float log_N = log(N);

    float max_score = 0;
    for (auto i = 0; i < rows; ++i) {
      auto &entry = row_ucb_data[i];
      // TODO probably remove divby0 hack
      const float score =
          entry.value * 255.0 + sqrt(log_N / (entry.visits + 1));
      if (max_score < score) {
        max_score = score;
        outcome.row_idx = i;
      }
    }
    max_score = 0;
    for (auto j = 0; j < cols; ++j) {
      auto &entry = col_ucb_data[j];
      const float score =
          entry.value * 255.0 + sqrt(log_N / (entry.visits + 1));
      if (max_score < score) {
        max_score = score;
        outcome.col_idx = j;
      }
    }
  }
};

};
#include <algorithm>
#include <assert.h>
#include <bit>
#include <cstring>
#include <memory>

namespace UCB {

namespace Precomputed {

consteval auto generate_row_col_map() {
  std::array<std::array<uint8_t, 2>, 81 * 2> result{};
  for (uint8_t i = 1; i <= 9; ++i) {
    for (uint8_t j = 1; j <= 9; ++j) {
      result[(i * j - 1) + 81 * (i > j)] = {i, j};
    }
  }
  return result;
}

static constexpr std::array<std::array<uint8_t, 2>, 81 * 2> row_col_map{
    generate_row_col_map()};

}; // namespace Precomputed

#pragma pack(1)
struct UCBEntry {
  uint16_t visits;
  uint8_t value;
};
#pragma pop()

struct RootUCBEntry {
  uint64_t visits;
  double value;
};

struct UCBEntryTest {
  static_assert(sizeof(UCBEntry) == 3);
  static_assert(sizeof(RootUCBEntry) == 16);
};

struct RootUCBNode;

class UCBNode {
public:
  std::array<UCBEntry, 9> row_ucb_data;
  std::array<UCBEntry, 9> col_ucb_data;
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
      UCBEntry &entry = row_ucb_data[outcome.row_idx];
      float value = entry.value / 255.0;
      value *= entry.visits++;
      value += outcome.value;
      assert(entry.visits != 0);
      value /= entry.visits;
      entry.value = static_cast<uint8_t>(value * 255);
    };
    {
      UCBEntry &entry = col_ucb_data[outcome.col_idx];
      float value = entry.value / 255.0;
      value *= entry.visits++;
      value += outcome.value;
      assert(entry.visits != 0);
      value /= entry.visits;
      entry.value = static_cast<uint8_t>(value * 255);
    };
  }

  template <typename Outcome>
  void select(const RootUCBNode &root_node, Outcome &outcome) {
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

class UCBNodeTest {
  static_assert(sizeof(UCBNode) == 64);
};

// This does MatrixUCB instead of joint UCB, also stores tree-wide data like 'c'
struct RootUCBNode {
  float c_uct{2};
  Matrix<RootUCBEntry, std::vector> empirical_matrix;

  RootUCBNode(float c_uct) : c_uct{c_uct} {}

  template <typename Battle> void init(const Battle &battle) {
    empirical_matrix = {battle.rows(), battle.cols()};
  }

  template <typename Outcome> void update(const Outcome &outcome) {}

  template <typename Outcome> void select(Outcome &outcome) {}
};

} // namespace UCB
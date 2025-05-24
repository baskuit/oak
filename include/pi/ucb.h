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

struct FatUCBEntry {
  uint32_t visits;
  float value;
};

#pragma pack(push, 1)
template <float gamma = .1f, bool enable_visits = false>
class JointBanditData : public JointBanditDataBase<enable_visits> {
public:
  std::array<FatUCBEntry, 9> p1_ucb;
  std::array<FatUCBEntry, 9> p2_ucb;

  using JointBanditDataBase<enable_visits>::p1_gains;
  using JointBanditDataBase<enable_visits>::p2_gains;
  using JointBanditDataBase<enable_visits>::_rows;
  using JointBanditDataBase<enable_visits>::_cols;

  struct Outcome {
    float value;
    float p1_mu;
    float p2_mu;
    uint8_t p1_index;
    uint8_t p2_index;
  };

  void init(auto rows, auto cols) noexcept {
    _rows = rows;
    _cols = cols;
    std::fill(p1_gains.begin(), p1_gains.begin() + rows, 0);
    std::fill(p2_gains.begin(), p2_gains.begin() + cols, 0);
    if constexpr (enable_visits) {
      std::fill(this->p1_visits.begin(), this->p1_visits.begin() + rows,
                uint24_t{});
      std::fill(this->p2_visits.begin(), this->p2_visits.begin() + cols,
                uint24_t{});
    }
  }

  bool is_init() const noexcept { return this->_rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    if constexpr (enable_visits) {
      ++this->p1_visits[outcome.p1_index];
      ++this->p2_visits[outcome.p2_index];
    }

    if ((p1_gains[outcome.p1_index] += outcome.value / outcome.p1_mu) >= 0) {
      const auto max = p1_gains[outcome.p1_index];
      for (auto &v : p1_gains) {
        v -= max;
      }
    }
    if ((p2_gains[outcome.p2_index] += (1 - outcome.value) / outcome.p2_mu) >=
        0) {
      const auto max = p2_gains[outcome.p2_index];
      for (auto &v : p2_gains) {
        v -= max;
      }
    }
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) const noexcept {
    constexpr float one_minus_gamma = 1 - gamma;
    std::array<float, 9> forecast{};
    if (_rows == 1) {
      outcome.p1_index = 0;
      outcome.p1_mu = 1;
    } else {
      const float eta{gamma / _rows};
      softmax(forecast, p1_gains, _rows, eta);
      std::transform(
          forecast.begin(), forecast.begin() + _rows, forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p1_index = device.sample_pdf(forecast);
      outcome.p1_mu = forecast[outcome.p1_index];
    }
    if (_cols == 1) {
      outcome.p2_index = 0;
      outcome.p2_mu = 1;
    } else {
      const float eta{gamma / _cols};
      softmax(forecast, p2_gains, _cols, eta);
      std::transform(
          forecast.begin(), forecast.begin() + _cols, forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p2_index = device.sample_pdf(forecast);
      outcome.p2_mu = forecast[outcome.p2_index];
    }

    // TODO
    outcome.p1_index =
        std::min(outcome.p1_index, static_cast<uint8_t>(_rows - 1));
    outcome.p2_index =
        std::min(outcome.p2_index, static_cast<uint8_t>(_cols - 1));

    assert(outcome.p1_index < _rows);
    assert(outcome.p2_index < _cols);
  }

  std::string visit_string() const {
    std::stringstream sstream{};
    if constexpr (enable_visits) {
      sstream << "V1: ";
      for (auto i = 0; i < _rows; ++i) {
        sstream << std::to_string(this->p1_visits[i]) << " ";
      }
      sstream << "V2: ";
      for (auto i = 0; i < _cols; ++i) {
        sstream << std::to_string(this->p2_visits[i]) << " ";
      }
      sstream.flush();
    }
    return sstream.str();
  }

  std::pair<std::vector<float>, std::vector<float>>
  policies(float iterations) const {

    std::vector<float> p1{};
    std::vector<float> p2{};

    p1.resize(_rows);
    p2.resize(_cols);

    if constexpr (enable_visits) {
      for (auto i = 0; i < _rows; ++i) {
        p1[i] = this->p1_visits[i].value() / (iterations - 1);
      }
      for (auto i = 0; i < _cols; ++i) {
        p2[i] = this->p2_visits[i].value() / (iterations - 1);
      }
    }
    return {p1, p2};
  }
};
#pragma pack(pop)

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
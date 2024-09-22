#include <types/matrix.h>
#include <types/random.h>
#include <types/vector.h>

namespace RBY_UCB {

struct Outcome {
  uint16_t row_idx;
  uint16_t col_idx;
  float value;
};

struct UCBEntry {
  uint32_t visits;
  float value;
  UCBEntry() : visits{1}, value{.5} {}
};

struct JointUCBDataGlobal {
  float c_uct{2};
  Matrix<UCBEntry, std::vector, uint16_t> empirical_matrix;

  JointUCBDataGlobal(float c_uct) : c_uct{c_uct} {}

  template <typename Battle> void init(const Battle &battle) {
    empirical_matrix = {battle.rows(), battle.cols()};
  }

  template <typename Outcome> void update(const Outcome &outcome) {
    auto &entry = empirical_matrix(outcome.row_idx, outcome.col_idx);
    entry.value *= (entry.visits++);
    entry.value += outcome.value;
    entry.value /= row_entry.visits;
  }
};

class JointUCBDataLocal {
private:
  using UCBData = ArrayBasedVector<9>::Vector<UCBEntry, uint16_t>;
  uint32_t _n;
  UCBData _row_ucb_data;
  UCBData _col_ucb_data;

public:
  template <typename Outcome> void update(const Outcome &outcome) {
    UCBEntry &row_entry = _row_ucb_data[outcome.row_idx];
    row_entry.value *= (row_entry.visits++);
    row_entry.value += outcome.value;
    row_entry.value /= row_entry.visits;
    UCBEntry &col_entry = _col_ucb_data[outcome.col_idx];
    col_entry.value *= (col_entry.visits++);
    col_entry.value += outcome.value;
    col_entry.value /= col_entry.visits;
    ++_n;
  }

  template <typename Outcome>
  void select(const JointUCBDataGlobal &bandit, Outcome &outcome) {
    using Vector = ArrayBasedVector<9>::Vector<float, uint16_t>;
    Vector row_forecast{};
    Vector col_forecast{};
    const auto _rows = _row_ucb_data.size();
    const auto _cols = _col_ucb_data.size();
    row_forecast.resize(_rows);
    col_forecast.resize(_cols);

    const float c = bandit.c;
    const auto log_n = log(_n);

    if (_rows == 1) {
      row_forecast[0] = 1.0;
    } else {
      std::transform(_row_ucb_data.begin(), _row_ucb_data.begin() + _rows,
                     row_forecast.begin(), [_n, log_n](const UCBEntry &entry) {
                       return entry.value + sqrt(log_n / entry.n);
                     });
    }
    if (_cols == 1) {
      col_forecast[0] = Rational<>{1};
    } else {
      std::transform(_col_ucb_data.begin(), _col_ucb_data.begin() + _cols,
                     col_forecast.begin(), [_n, log_n](const UCBEntry &entry) {
                       return entry.q + sqrt(log_n / entry.n);
                     });
    }

    outcome.row_idx =
        std::max_element(row_forecast.begin(), row_forecast.end()) -
        row_forecast.begin();
    outcome.col_idx =
        std::max_element(col_forecast.begin(), col_forecast.end()) -
        col_forecast.begin();
  }

  void init(const Battle &state) {
    _n = 1;
    _row_ucb_data.resize(state.rows());
    _col_ucb_data.resize(state.cols());
  }
};

} // namespace RBY_UCB
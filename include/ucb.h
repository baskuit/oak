#include <algorithm/algorithm.h>
#include <libpinyon/math.h>
#include <tree/tree.h>
#include <type_traits>

template <typename Types> struct UCBOak : Types {
  using Real = typename Types::Real;

  template <template <typename...> typename Vector, typename T>
  void print(const Vector<T> &input) {
    for (int i = 0; i < input.size(); ++i) {
      std::cout << input[i] << ", ";
    }
    std::cout << std::endl;
  }

  struct UCBEntry {
    Real q{.5};
    uint32_t n{1};

    void update(Real x) {
      q = q * n + x;
      ++n;
      q = q / n;
    }
  };

  struct MatrixStats {
    Types::template Vector<UCBEntry> row_ucb;
    Types::template Vector<UCBEntry> col_ucb;
    unsigned int N{1};
  };
  struct ChanceStats {};
  struct Outcome {
    int row_idx, col_idx;
    Types::Value value;
  };

  class BanditAlgorithm {
  public:
    Real c{2};

    constexpr BanditAlgorithm() {}

    constexpr BanditAlgorithm(Real c) : c{c} {}

    friend std::ostream &operator<<(std::ostream &os,
                                    const BanditAlgorithm &search) {
      os << "UCB; c: " << search.c;
      return os;
    }

    void get_empirical_strategies(const MatrixStats &stats,
                                  Types::VectorReal &row_strategy,
                                  Types::VectorReal &col_strategy) const {}

    void get_empirical_value(const MatrixStats &stats,
                             Types::Value &value) const {}

    void get_refined_strategies(const MatrixStats &stats,
                                Types::VectorReal &row_strategy,
                                Types::VectorReal &col_strategy) const {}

    void get_refined_value(const MatrixStats &stats,
                           Types::Value &value) const {}

    // protected:
    void initialize_stats(int iterations, const Types::State &state,
                          Types::Model &model, MatrixStats &stats) const {}

    void expand(MatrixStats &stats, const size_t &rows, const size_t &cols,
                const Types::ModelOutput &output) const {
      stats.row_ucb.resize(rows);
      stats.col_ucb.resize(cols);
    }

    void select(Types::PRNG &device, const MatrixStats &stats,
                Outcome &outcome) const {
      const size_t rows = stats.row_ucb.size();
      const size_t cols = stats.col_ucb.size();
      typename Types::VectorReal row_forecast{};
      typename Types::VectorReal col_forecast{};
      row_forecast.resize(rows, 0);
      col_forecast.resize(cols, 0);

      const Real c = this->c;
      const auto N = stats.N;

      if (rows == 1) {
        row_forecast[0] = Rational<>{1};
      } else {
        std::transform(stats.row_ucb.begin(), stats.row_ucb.begin() + rows,
                       row_forecast.begin(), [N](const UCBEntry &entry) {
                         return entry.q + sqrt(log(N) / entry.n);
                       });
      }
      if (cols == 1) {
        col_forecast[0] = Rational<>{1};
      } else {
        std::transform(stats.col_ucb.begin(), stats.col_ucb.begin() + cols,
                       col_forecast.begin(), [N](const UCBEntry &entry) {
                         return entry.q + sqrt(log(N) / entry.n);
                       });
      }
      outcome.row_idx =
          std::max_element(row_forecast.begin(), row_forecast.end()) -
          row_forecast.begin();
      outcome.col_idx =
          std::max_element(col_forecast.begin(), col_forecast.end()) -
          col_forecast.begin();

      //   math::print(row_forecast);
      //   std::cout << outcome.row_idx << std::endl;
      //   math::print(col_forecast);
      //   std::cout << outcome.col_idx << std::endl << std::endl;
    }

    void update_matrix_stats(MatrixStats &stats, const Outcome &outcome) const {
      ++stats.N;
      stats.row_ucb[outcome.row_idx].update(outcome.value.get_row_value());
      stats.col_ucb[outcome.col_idx].update(outcome.value.get_col_value());
    }

    void update_chance_stats(ChanceStats &stats, const Outcome &outcome) const {
    }
  };
};
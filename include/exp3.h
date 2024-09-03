#include <algorithm/algorithm.h>
#include <libpinyon/math.h>
#include <tree/tree.h>
#include <type_traits>

template <IsValueModelTypes Types>
struct Exp3Oak : Types {
    using Real = typename Types::Real;

    struct MatrixStats {
        typename Types::VectorReal row_gains;
        typename Types::VectorReal col_gains;
        static_assert(std::is_base_of_v<std::array<Real, 9>, typename Types::VectorReal>);
    };
    struct ChanceStats {};
    struct Outcome {
        int row_idx, col_idx;
        Types::Value value;
        Real row_mu, col_mu;
    };

    class BanditAlgorithm {
       public:
        Real gamma{.01};
        Real one_minus_gamma{gamma * -1 + 1};

        constexpr BanditAlgorithm() {}

        constexpr BanditAlgorithm(Real gamma) : gamma(gamma), one_minus_gamma{gamma * -1 + 1} {}

        friend std::ostream &operator<<(std::ostream &os, const BanditAlgorithm &search) {
            os << "Exp3; gamma: " << search.gamma;
            return os;
        }

        void get_empirical_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                      Types::VectorReal &col_strategy) const {
            // row_strategy.resize(stats.row_visits.size());
            // col_strategy.resize(stats.col_visits.size());
            // math::power_norm(stats.row_visits, row_strategy.size(), 1, row_strategy);
            // math::power_norm(stats.col_visits, col_strategy.size(), 1, col_strategy);
        }

        void get_empirical_value(const MatrixStats &stats, Types::Value &value) const {
            // const Real den = typename Types::Q{1, (stats.visits + (stats.visits == 0))};
            // if constexpr (Types::Value::IS_CONSTANT_SUM) {
            //     value = typename Types::Value{Real{stats.value_total.get_row_value() * den}};
            // } else {
            //     value = typename Types::Value{stats.value_total * den};
            // }
        }

        void get_refined_strategies(const MatrixStats &stats, Types::VectorReal &row_strategy,
                                    Types::VectorReal &col_strategy) const {
            // row_strategy.resize(stats.row_visits.size());
            // col_strategy.resize(stats.col_visits.size());
            // math::power_norm(stats.row_visits, row_strategy.size(), 1, row_strategy);
            // math::power_norm(stats.col_visits, col_strategy.size(), 1, col_strategy);
            // denoise(row_strategy, col_strategy);
        }

        void get_refined_value(const MatrixStats &stats, Types::Value &value) const {
            // get_empirical_value(stats, value);
        }

        // protected:
        void initialize_stats(int iterations, const Types::State &state, Types::Model &model,
                              MatrixStats &stats) const {}

        void expand(MatrixStats &stats, const size_t &rows, const size_t &cols,
                    const Types::ModelOutput &output) const {
            // stats.row_visits.resize(rows, 0);
            // stats.col_visits.resize(cols, 0);
            // stats.row_gains.resize(rows, 0);
            // stats.col_gains.resize(cols, 0);
            std::fill(std::begin(stats.row_gains), std::begin(stats.row_gains) + rows, 0);
            std::fill(std::begin(stats.col_gains), std::begin(stats.col_gains) + cols, 0);
        }

        void select(Types::PRNG &device, const MatrixStats &stats, Outcome &outcome) const {
            const size_t rows = stats.row_gains.size();
            const size_t cols = stats.col_gains.size();
            const auto &one_minus_gamma = this->one_minus_gamma;
            typename Types::VectorReal row_forecast(rows);
            typename Types::VectorReal col_forecast(cols);
            if (rows == 1) {
                row_forecast[0] = Rational<>{1};
            } else {
                const Real eta{gamma / static_cast<Real>(rows)};
                softmax(row_forecast, stats.row_gains, rows, eta);
                std::transform(row_forecast.begin(), row_forecast.begin() + rows, row_forecast.begin(),
                               [eta, one_minus_gamma](Real value) { return one_minus_gamma * value + eta; });
            }
            if (cols == 1) {
                col_forecast[0] = Rational<>{1};
            } else {
                const Real eta{gamma / static_cast<Real>(cols)};
                softmax(col_forecast, stats.col_gains, cols, eta);
                std::transform(col_forecast.begin(), col_forecast.begin() + cols, col_forecast.begin(),
                               [eta, one_minus_gamma](Real value) { return one_minus_gamma * value + eta; });
            }
            const int row_idx = device.sample_pdf(row_forecast);
            const int col_idx = device.sample_pdf(col_forecast);
            outcome.row_idx = row_idx;
            outcome.col_idx = col_idx;
            outcome.row_mu = static_cast<Real>(row_forecast[row_idx]);
            outcome.col_mu = static_cast<Real>(col_forecast[col_idx]);
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome) const {
            // stats.value_total += PairReal<Real>{outcome.value.get_row_value(), outcome.value.get_col_value()};
            // stats.visits += 1;
            // stats.row_visits[outcome.row_idx] += 1;
            // stats.col_visits[outcome.col_idx] += 1;
            if ((stats.row_gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu) >= 0) {
                const auto max = stats.row_gains[outcome.row_idx];
                for (auto &v : stats.row_gains) {
                    v -= max;
                }
            }
            if ((stats.col_gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu) >= 0) {
                const auto max = stats.col_gains[outcome.col_idx];
                for (auto &v : stats.col_gains) {
                    v -= max;
                }
            }
        }

        void update_chance_stats(ChanceStats &stats, const Outcome &outcome) const {}

        // multithreaded

        void select(Types::PRNG &device,
                    const MatrixStats &stats,  // TODO made const, does this fix bug?
                    Outcome &outcome, Types::Mutex &mutex) const {
            mutex.lock();
            typename Types::VectorReal row_forecast(stats.row_gains);
            typename Types::VectorReal col_forecast(stats.col_gains);
            mutex.unlock();
            const size_t rows = row_forecast.size();
            const size_t cols = col_forecast.size();
            const auto &one_minus_gamma = this->one_minus_gamma;

            if (rows == 1) {
                row_forecast[0] = Rational<>{1};
            } else {
                const Real eta{gamma / static_cast<Real>(rows)};
                softmax(row_forecast, row_forecast, rows, eta);
                std::transform(row_forecast.begin(), row_forecast.begin() + rows, row_forecast.begin(),
                               [eta, one_minus_gamma](Real value) { return one_minus_gamma * value + eta; });
            }
            if (cols == 1) {
                col_forecast[0] = Rational<>{1};
            } else {
                const Real eta{gamma / static_cast<Real>(cols)};
                softmax(col_forecast, col_forecast, cols, eta);
                std::transform(col_forecast.begin(), col_forecast.begin() + cols, col_forecast.begin(),
                               [eta, one_minus_gamma](Real value) { return one_minus_gamma * value + eta; });
            }
            const int row_idx = device.sample_pdf(row_forecast, rows);
            const int col_idx = device.sample_pdf(col_forecast, cols);
            outcome.row_idx = row_idx;
            outcome.col_idx = col_idx;
            outcome.row_mu = static_cast<Real>(row_forecast[row_idx]);
            outcome.col_mu = static_cast<Real>(col_forecast[col_idx]);
        }

        void update_matrix_stats(MatrixStats &stats, const Outcome &outcome, Types::Mutex &mutex) const {
            mutex.lock();
            // stats.value_total += outcome.value;
            // stats.visits += 1;
            // stats.row_visits[outcome.row_idx] += 1;
            // stats.col_visits[outcome.col_idx] += 1;
            if ((stats.row_gains[outcome.row_idx] += outcome.value.get_row_value() / outcome.row_mu) >= 0) {
                const auto max = stats.row_gains[outcome.row_idx];
                for (auto &v : stats.row_gains) {
                    v -= max;
                }
            }
            if ((stats.col_gains[outcome.col_idx] += outcome.value.get_col_value() / outcome.col_mu) >= 0) {
                const auto max = stats.col_gains[outcome.col_idx];
                for (auto &v : stats.col_gains) {
                    v -= max;
                }
            }
            mutex.unlock();
        }

        void update_chance_stats(ChanceStats &stats, const Outcome &outcome, Types::Mutex &mutex) const {}

       private:
        template <typename GainsVector>
        inline void softmax(Types::VectorReal &forecast, const GainsVector &gains, const size_t k,
                            Real eta) const {
            Real sum = 0;
            for (size_t i = 0; i < k; ++i) {
                const Real y{std::exp(static_cast<float>(gains[i] * eta))};  // TODO maybe but T::Float back
                forecast[i] = y;
                sum += y;
            }
            for (size_t i = 0; i < k; ++i) {
                forecast[i] /= sum;
            }
        };

        inline void denoise(Types::VectorReal &row_strategy, Types::VectorReal &col_strategy) const {
            const size_t rows = row_strategy.size();
            const size_t cols = col_strategy.size();
            const auto &one_minus_gamma = this->one_minus_gamma;
            if (rows > 1) {
                const Real eta{gamma / static_cast<Real>(rows)};
                std::transform(row_strategy.begin(), row_strategy.begin() + rows, row_strategy.begin(),
                               [eta, one_minus_gamma](Real value) { return (value - eta) / one_minus_gamma; });
            }
            if (cols > 1) {
                const Real eta{gamma / static_cast<Real>(cols)};
                std::transform(col_strategy.begin(), col_strategy.begin() + cols, col_strategy.begin(),
                               [eta, one_minus_gamma](Real value) { return (value - eta) / one_minus_gamma; });
            }
        }  // TODO can produce negative values but this shouldnt cause problems.
    };
};
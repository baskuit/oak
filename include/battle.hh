#pragma once

#include <pkmn.h>

#include <limits>
#include <pinyon.hh>

/*

This is the pinyon wrapper for gen 1 libpkmn.

This wrapper is intended to be fast so the data structures and control flow depend on the features that libpkmn was
compiled with, e.g.

`-Dlog -Dchance -Dcalc`

Note:

`-Dshowdown` is assumed.

*/

constexpr size_t SIZE_BATTLE_NO_PRNG = 376;
constexpr size_t SIZE_BATTLE_WITH_PRNG = SIZE_BATTLE_NO_PRNG + 8;
constexpr size_t SIZE_SIDE = 184;

/*
These enums are used to choose between the choice of Obs data type

ChanceObs will use the chance actions (16 bytes)
LogObs will use the debug log. Even in OU, this typically requires 64 bytes (otherwise the obs are not faithful)
BattleObs will use the entire battle except the seed (376 bytes)
*/
enum BattleObsT {
    // In order of preference aka size
    ChanceObs,
    LogObs,
    BattleObs,
};

/*
This implemenents the basic pinyon "type list" for our battle wrapper. The types subject to change are:
    Real, Prob, Obs (and ObsHash)

The Action type will always be `pkmn_choice`, the Value type will always be ConstantSum<1, 1>::Value<Real>

Pinyon uses the `ObsHash` name to hash the Obs type for use with std::unordered_map
The hash function for the ChanceObs case is collision free in the OU case, at least as far as my tests can tell
I have absolutely no idea about the BattleObs hasher. Its not supposed to be used lol
Note: TreeBandit with DefaultNodes (aka the most basic possible search) will use equality with a linear scan to
identify different obs

*/
namespace BattleTypesImpl {
template <typename Real, typename Prob, BattleObsT Obs, size_t LOG_SIZE>
struct BattleTypes;

template <typename Real, typename Prob, size_t LOG_SIZE>
struct BattleTypes<Real, Prob, ChanceObs, LOG_SIZE>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, 16>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {
    struct ObsHash {
        size_t operator()(const std::array<uint8_t, 16> &obs) const {
            static const uint64_t duration_mask = 0xFFFFFFFFFF0FFFFF;
            const uint64_t *a = reinterpret_cast<const uint64_t *>(obs.data());
            const uint64_t side_1 = a[0] & duration_mask;
            const uint64_t side_2 = a[1] & duration_mask;
            return ((side_1 << 32) >> 32) | (side_2 << 32);
        }
    };
};

template <typename Real, typename Prob, size_t LOG_SIZE>
struct BattleTypes<Real, Prob, LogObs, LOG_SIZE>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, LOG_SIZE>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {
    struct ObsHash {
        size_t operator()(const std::array<uint8_t, LOG_SIZE> &obs) const {
            const uint64_t *a = reinterpret_cast<const uint64_t *>(obs.data());
            size_t hash = 0;
            for (int i = 0; i < 8; ++i) {
                hash ^= a[i];
            }
            return hash;
        }
    };
};

template <typename Real, typename Prob, size_t LOG_SIZE>
struct BattleTypes<Real, Prob, BattleObs, LOG_SIZE>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, SIZE_BATTLE_NO_PRNG>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {};
    struct ObsHash {
        size_t operator()(const std::array<uint8_t, SIZE_BATTLE_NO_PRNG> &obs) const {
            const uint64_t *a = reinterpret_cast<const uint64_t *>(obs.data());
            size_t hash = 0;
            for (int i = 0; i < 47; ++i) {
                hash ^= a[i];
            }
            return hash;
        }
    };
};

namespace BattleDataImpl {
/*

This base class template spares code by holding
the data members that are required for all libpkmn variants.

We also implement the basic `State` methods that are expected by pinyon.
In most of the library code we use the base class `PerfectInfoState` for this.
We opt not to use that here since it has an `Obs` type member and the `get_obs` method
simply returns a reference to this.
The search code for pinyon copies the state object instead of using an `unmake_move` method
In view of this, the `Obs` member merely results in extra allocation.
This is because the `pkmn_gen1_options` struct is the canonical container for the chance actions
(assuming we are using ChanceObs)
So even if our `get_obs` currectly just references this data theres still a vestigial data member
that must be allocated for with every copied state.

*/

template <size_t LOG_SIZE, size_t ROLLS = 0, typename Real = mpq_class, typename Prob = mpq_class,
          BattleObsT Obs = ChanceObs>
struct BattleDataBase {
    A<9>::Array<pkmn_choice> row_actions{};
    A<9>::Array<pkmn_choice> col_actions{};
    Prob prob{};

    pkmn_gen1_battle battle;
    pkmn_gen1_battle_options options;
    pkmn_result result{};
    pkmn_result_kind result_kind{};

    bool is_terminal() const { return result_kind; }

    ConstantSum<1, 1>::Value<Real> get_payoff() const {
        if (this->result_kind) [[likely]] {
            switch (pkmn_result_type(this->result)) {
                case PKMN_RESULT_WIN: {
                    return {Real{1}};
                }
                case PKMN_RESULT_LOSE: {
                    return {Real{0}};
                }
                case PKMN_RESULT_ERROR: {
                    std::exception();
                }
            }
        }
        // tie/not terminal
        // Rational type is used because it has a conversion operator to mpq_class and floating point
        return {Real{Rational<>(1, 2)}};
    }

    const Prob& get_prob () const {
        return prob;
    }

    void randomize_transition(prng &device) {
        uint8_t *battle_prng_bytes = battle.bytes + SIZE_BATTLE_NO_PRNG;
        *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
    }

    void randomize_transition(const uint64_t seed) {
        uint8_t *battle_prng_bytes = battle.bytes + SIZE_BATTLE_NO_PRNG;
        *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = seed;
    }
};

/*

Different data members are required for a libpkmn battle depending on what features we want
C++'s `if constexpr` allows for a lot of in-line compile time stuff
but changing data members at compile time requires templat specialization
*/

template <size_t LOG_SIZE, size_t ROLLS = 0, typename Real = mpq_class, typename Prob = mpq_class,
          BattleObsT Obs = ChanceObs>
struct BattleData;

template <size_t LOG_SIZE, size_t ROLLS, typename Real, typename Prob, BattleObsT ObsEnum>
struct BattleData : BattleDataBase<LOG_SIZE, ROLLS, Real, Prob, ObsEnum> {
    std::array<uint8_t, LOG_SIZE> log_buffer{};
    pkmn_gen1_log_options log_options{};
    pkmn_gen1_chance_options chance_options{};
    pkmn_gen1_calc_options calc_options{};
    pkmn_rational *p;
    bool clamped = false;

    inline void set() { pkmn_gen1_battle_options_set(&this->options, &log_options, &chance_options, &calc_options); }

    static constexpr bool dlog = true;
    static constexpr bool dchance = true;
    static constexpr bool dcalc = true;
    static constexpr size_t log_size = LOG_SIZE;
};

// No chance/calc needed
template <size_t LOG_SIZE, typename Real, BattleObsT Obs>
struct BattleData<LOG_SIZE, 0, Real, bool, Obs> : BattleDataBase<LOG_SIZE, 0, Real, bool, Obs> {
    static_assert(Obs == BattleObs || Obs == LogObs);
    std::array<uint8_t, LOG_SIZE> log_buffer{};
    pkmn_gen1_log_options log_options{};

    inline void set() { pkmn_gen1_battle_options_set(&this->options, &log_options, NULL, NULL); }

    static constexpr bool dlog = true;
    static constexpr bool dchance = false;
    static constexpr bool dcalc = false;
    static constexpr size_t log_size = LOG_SIZE;
};

template <typename Real>
struct BattleData<0, 0, Real, bool, BattleObs> : BattleDataBase<0, 0, Real, bool, BattleObs> {
    inline void set() { pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL); }

    static constexpr bool dlog = false;
    static constexpr bool dchance = false;
    static constexpr bool dcalc = false;
    static constexpr size_t log_size = 0;
};

template <size_t ROLLS, typename Real, typename Prob, BattleObsT ObsEnum>
struct BattleData<0, ROLLS, Real, Prob, ObsEnum> : BattleDataBase<0, ROLLS, Real, Prob, ObsEnum> {
    pkmn_gen1_chance_options chance_options{};
    pkmn_gen1_calc_options calc_options{};
    pkmn_rational *p;
    bool clamped = false;

    inline void set() { pkmn_gen1_battle_options_set(&this->options, NULL, &chance_options, &calc_options); }

    static constexpr bool dlog = false;
    static constexpr bool dchance = true;
    static constexpr bool dcalc = true;
    static constexpr size_t log_size = 0;
};
};  // namespace BattleDataImpl

template <size_t LOG_SIZE, size_t ROLLS = 0, BattleObsT Obs = ChanceObs, typename Prob = mpq_class,
          typename Real = mpq_class>
struct Battle : BattleTypesImpl::BattleTypes<Real, Prob, Obs, LOG_SIZE> {
    using TypeList = BattleTypesImpl::BattleTypes<Real, Prob, Obs, LOG_SIZE>;

    class State : public BattleDataImpl::BattleData<LOG_SIZE, ROLLS, Real, Prob, Obs> {
       public:
        State() {}

        State(const uint8_t *row_side, const uint8_t *col_side) {
            // init: copy sides onto battle and zero initialize certain bits
            // PRNG bytes are left uninitialized (zero initializing is no better, terrible seed)
            memcpy(this->battle.bytes, row_side, SIZE_SIDE);
            memcpy(this->battle.bytes + SIZE_SIDE, col_side, SIZE_SIDE);
            memset(this->battle.bytes + 2 * SIZE_SIDE, 0, SIZE_BATTLE_NO_PRNG - 2 * SIZE_SIDE);

            if constexpr (State::dlog) {
                this->log_options = {this->log_buffer.data(), LOG_SIZE};
            }
            if constexpr (State::dchance) {
                pkmn_rational_init(&this->chance_options.probability);
                this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
            }

            this->set();

            get_actions();
        }

        State(const State &other) {
            this->prob = other.prob;
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            memcpy(this->battle.bytes, other.battle.bytes, SIZE_BATTLE_NO_PRNG);
            this->options = other.options;
            this->result = other.result;
            this->result_kind = other.result_kind;
            if constexpr (State::dlog) {
                this->log_options = {this->log_buffer.data(), LOG_SIZE};
                pkmn_gen1_battle_options_set(&this->options, &this->log_options, NULL, NULL);
            } else {
                pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
            }
            if constexpr (State::dchance) {
                this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
            }
            if constexpr (State::dcalc) {
                this->clamped = other.clamped;
            }
        }

        State& operator=(const State& other) {
            this->prob = other.prob;
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            memcpy(this->battle.bytes, other.battle.bytes, SIZE_BATTLE_NO_PRNG);
            this->options = other.options;
            this->result = other.result;
            this->result_kind = other.result_kind;
            if constexpr (State::dlog) {
                this->log_options = {this->log_buffer.data(), LOG_SIZE};
                pkmn_gen1_battle_options_set(&this->options, &this->log_options, NULL, NULL);
            } else {
                pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
            }
            if constexpr (State::dchance) {
                this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
            }
            if constexpr (State::dcalc) {
                this->clamped = other.clamped;
            }
            return *this;
        }

        template <typename State_>
            requires(!std::is_same_v<State_, State>)
        State(const State_ &other) {
            this->prob = other.prob.get_d();
            this->row_actions = other.row_actions;
            this->col_actions = other.col_actions;
            memcpy(this->battle.bytes, other.battle.bytes, SIZE_BATTLE_NO_PRNG);
            this->options = other.options;
            this->result = other.result;
            this->result_kind = other.result_kind;
            if constexpr (State::dlog) {
                if constexpr (State_::dlog) {
                    // TODO
                } else {
                }
                this->log_options = {this->log_buffer.data(), LOG_SIZE};
                pkmn_gen1_battle_options_set(&this->options, &this->log_options, NULL, NULL);
            } else {
                pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
            }
            if constexpr (State::dchance) {
                this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
            }
            if constexpr (State::dcalc) {
                this->clamped = other.clamped;
            }
        }

        const auto &get_obs() const {
            if constexpr (Obs == ChanceObs) {
                auto *ptr = pkmn_gen1_battle_options_chance_actions(&this->options)->bytes;
                const std::array<uint8_t, 16> &obs_ref = *reinterpret_cast<std::array<uint8_t, 16> *>(ptr);
                return obs_ref;
            }
            if constexpr (Obs == LogObs) {
                return this->log_buffer;
            }
            if constexpr (Obs == BattleObs) {
                auto *ptr = this->battle.bytes;
                const std::array<uint8_t, SIZE_BATTLE_NO_PRNG> &obs_ref =
                    *reinterpret_cast<std::array<uint8_t, SIZE_BATTLE_NO_PRNG> *>(ptr);
                return obs_ref;
            }
        }

        void get_actions() {
            this->row_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P1,
                                                              pkmn_result_p1(this->result), this->row_actions.data(),
                                                              PKMN_MAX_CHOICES));
            this->col_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P2,
                                                              pkmn_result_p2(this->result), this->col_actions.data(),
                                                              PKMN_MAX_CHOICES));
        }

        void get_actions(TypeList::VectorAction &row_actions, TypeList::VectorAction &col_actions) const {
            row_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P1, pkmn_result_p1(this->result),
                                                        row_actions.data(), PKMN_MAX_CHOICES));
            col_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P2, pkmn_result_p2(this->result),
                                                        col_actions.data(), PKMN_MAX_CHOICES));
        }

        void get_actions_no_switch() {
            this->row_actions.resize(pkmn_gen1_battle_choices_no_switch(&this->battle, PKMN_PLAYER_P1,
                                                              pkmn_result_p1(this->result), this->row_actions.data(),
                                                              PKMN_MAX_CHOICES));
            this->col_actions.resize(pkmn_gen1_battle_choices_no_switch(&this->battle, PKMN_PLAYER_P2,
                                                              pkmn_result_p2(this->result), this->col_actions.data(),
                                                              PKMN_MAX_CHOICES));
        }

        void get_actions_no_switch(TypeList::VectorAction &row_actions, TypeList::VectorAction &col_actions) const {
            row_actions.resize(pkmn_gen1_battle_choices_no_switch(&this->battle, PKMN_PLAYER_P1, pkmn_result_p1(this->result),
                                                        row_actions.data(), PKMN_MAX_CHOICES));
            col_actions.resize(pkmn_gen1_battle_choices_no_switch(&this->battle, PKMN_PLAYER_P2, pkmn_result_p2(this->result),
                                                        col_actions.data(), PKMN_MAX_CHOICES));
        }

        void apply_actions(pkmn_choice row_action, pkmn_choice col_action) {
            // Only 2, 3, 20, and 39 are supported as Roll values
            // TODO use 'high quality' bits of the showdown seed
            if constexpr (ROLLS != 0) {
                if (this->clamped) {
                    this->calc_options.overrides.bytes[0] =
                        217 + int{38 / (ROLLS - 1)} * (this->battle.bytes[383] % ROLLS);
                    this->calc_options.overrides.bytes[8] =
                        217 + int{38 / (ROLLS - 1)} * (this->battle.bytes[382] % ROLLS);
                    pkmn_gen1_battle_options_set(&this->options, NULL, NULL, &this->calc_options);
                } else {
                    pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
                }
            } else {
                pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
            }

            // The actual update call
            this->result = pkmn_gen1_battle_update(&this->battle, row_action, col_action, &this->options);

            // bool is the type when no prob is computed i.e. dchance is false
            if constexpr (std::is_floating_point_v<Prob>) {
                this->prob = static_cast<Prob>(pkmn_rational_numerator(this->p) / pkmn_rational_denominator(this->p));
            } else if constexpr (!std::is_same_v<Prob, bool>) {
                this->prob = Prob{pkmn_rational_numerator(this->p), pkmn_rational_denominator(this->p)};
            }

            if constexpr (ROLLS != 0) {
                if (this->clamped) {
                    // TODO clean up static_cast nonsense
                    const auto &obs_ref = this->get_obs();
                    if ((obs_ref[1] & 2) && obs_ref[0]) {
                        this->prob *= static_cast<Prob>(typename TypeList::Q{static_cast<int>(39 / ROLLS), 1});
                        math::canonicalize(this->prob);
                    }
                    if ((obs_ref[9] & 2) && obs_ref[8]) {
                        this->prob *= static_cast<Prob>(typename TypeList::Q{static_cast<int>(39 / ROLLS), 1});
                        math::canonicalize(this->prob);
                    }
                }
            }

            // updates is_terminal() output
            this->result_kind = pkmn_result_type(this->result);
        }
    };
};

// Utils

// counts number of alive mons using pointer to raw battle data - no templating needed
int n_alive(const uint8_t *data) {
    int n{0};
    for (int s = 0; s < 2; ++s) {
        for (int p = 0; p < 6; ++p) {
            const int index = SIZE_SIDE * s + 24 * p + 18;
            const bool alive = data[index] || data[index + 1];
            if (alive) ++n;
        }
    }
    return n;
}

int n_alive_side(const uint8_t *data, int player = 0) {
    int n{0};
    for (int p = 0; p < 6; ++p) {
        const int index = SIZE_SIDE * player + 24 * p + 18;
        const bool alive = data[index] || data[index + 1];
        if (alive) ++n;
    }

    return n;
}

// here we need to call the `apply_actions` function so internals are updated so we have to template
// the data pointer is assumed to be a buffer(slice) big enough for that turn
template <typename State>
void apply_actions_with_log(State &state, pkmn_choice row_action, pkmn_choice col_action, uint8_t *data) {
    state.apply_actions(row_action, col_action);
    if constexpr (State::dlog) {
        memcpy(data, state.log_buffer.data(), State::log_size);
    }
    memcpy(data + State::log_size, state.battle.bytes, SIZE_BATTLE_WITH_PRNG);
    data[State::log_size + SIZE_BATTLE_WITH_PRNG] = state.result;
    data[State::log_size + SIZE_BATTLE_WITH_PRNG + 1] = row_action;
    data[State::log_size + SIZE_BATTLE_WITH_PRNG + 2] = col_action;
}

// helper for eval log function below
template <typename Real>
void write_real_as_float(const Real x, uint8_t *data, int &index) {
    float f = math::to_float(x);
    uint8_t *f_raw = reinterpret_cast<uint8_t *>(&f);
    memcpy(data + index, f_raw, 4);
    index += 4;
}

//
template <typename State, typename RowModelOutput, typename ColModelOutput>
int apply_actions_with_eval_log(State &state, pkmn_choice row_action, pkmn_choice col_action,
                                RowModelOutput *row_output, ColModelOutput *col_output, uint8_t *data) {
    // no non eval update
    const uint8_t rows = state.row_actions.size();
    const uint8_t cols = state.col_actions.size();
    apply_actions_with_log(state, row_action, col_action, data);
    // return;

    // prepare
    int index = 3 + SIZE_BATTLE_WITH_PRNG + State::log_size;
    
    constexpr float NaN_ = std::numeric_limits<float>::quiet_NaN();
    float row_eval = NaN_;
    float col_eval = NaN_;
    std::array<float, 9> rows_row_policy{NaN_};
    std::array<float, 9> rows_col_policy{NaN_};
    std::array<float, 9> cols_row_policy{NaN_};
    std::array<float, 9> cols_col_policy{NaN_};
    if (row_output != nullptr) {
        row_eval = math::to_float(row_output->value.get_row_value());
        for (uint8_t row_idx{}; row_idx < rows; ++row_idx) {
            rows_row_policy[row_idx] = math::to_float(row_output->row_policy[row_idx]);
        }
        for (uint8_t col_idx{}; col_idx < cols; ++col_idx) {
            rows_col_policy[col_idx] = math::to_float(row_output->col_policy[col_idx]);
        }
    }
    if (col_output != nullptr) {
        col_eval = math::to_float(col_output->value.get_row_value());
        for (uint8_t row_idx{}; row_idx < rows; ++row_idx) {
            cols_row_policy[row_idx] = math::to_float(col_output->row_policy[row_idx]);
        }
        for (uint8_t col_idx{}; col_idx < cols; ++col_idx) {
            cols_col_policy[col_idx] = math::to_float(col_output->col_policy[col_idx]);
        }
    }

    // writing
    data[index++] = rows;
    data[index++] = cols;
    write_real_as_float(row_eval, data, index);
    for (int row_idx{}; row_idx < rows; ++row_idx) {
        write_real_as_float(rows_row_policy[row_idx], data, index);
    }
    for (int col_idx{}; col_idx < cols; ++col_idx) {
        write_real_as_float(rows_col_policy[col_idx], data, index);
    }
    data[index++] = 0;
    write_real_as_float(col_eval, data, index);
    for (int row_idx{}; row_idx < rows; ++row_idx) {
        write_real_as_float(cols_row_policy[row_idx], data, index);
    }
    for (int col_idx{}; col_idx < cols; ++col_idx) {
        write_real_as_float(cols_col_policy[col_idx], data, index);
    }
    data[index++] = 0;

    return index;
}

template <typename State>
void get_active_hp(const State &state) {
    const int hp_0 = state.battle.bytes[18] + 256 * state.battle.bytes[19];
    const int hp_1 = state.battle.bytes[18 + SIZE_SIDE] + 256 * state.battle.bytes[19 + SIZE_SIDE];
    std::cout << "action hp: " << hp_0 << ' ' << hp_1 << std::endl;
}

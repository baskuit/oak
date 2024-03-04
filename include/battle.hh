#pragma once

#include <pkmn.h>

#include <pinyon.hh>

/*

This is the pinyon wrapper for gen 1 libpkmn.

This wrapper is intended to be fast also
and so there are template parameters that reflect libpkmn flags

`-Dlog -Dchance -Dcalc`

as well as damage roll clamping, which is crucial for search feasibility.

`-Dshowdown` is assumed.

*/

constexpr size_t SIZE_BATTLE_NO_PRNG = 376;
constexpr size_t SIZE_BATTLE_WITH_PRNG = SIZE_BATTLE_NO_PRNG + 8;
constexpr size_t SIZE_SIDE = 184;

enum BattleObsT {
    // In order of preference aka size
    ChanceObs,
    LogObs,
    BattleObs,
};

// basically a converter between ObsType enums and the actual types in the constructor for the basic type list
namespace BattleTypesImpl {
template <typename Real, typename Prob, BattleObsT Obs>
struct BattleTypes;

template <typename Real, typename Prob>
struct BattleTypes<Real, Prob, ChanceObs>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, 16>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {};

template <typename Real, typename Prob>
struct BattleTypes<Real, Prob, LogObs>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, 64>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {};

template <typename Real, typename Prob>
struct BattleTypes<Real, Prob, BattleObs>
    : DefaultTypes<Real, pkmn_choice, std::array<uint8_t, 376>, Prob, ConstantSum<1, 1>::Value, A<9>::Array> {};
};  // namespace BattleTypesImpl

namespace BattleDataImpl {
/*

This base class template spares code by holding
the data members that are required for all battles.

We also implement the basic `State` methods that are expected by pinyon.
In most of the library code we use the base class `PerfectInfoState` for this.
We opt not to use that here since it has an `Obs` type member and the `get_obs` method
simply returns a reference to this.
The search code for pinyon copies the state object instead of using an `unmake_move` method
In view of this, the `Obs` member merely results in extra allocation.
This is because the `pkmn_gen1_options` struct is the canonical container for the chance actions, for example.
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
    pkmn_result_kind result_kind;

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
    pkmn_gen1_chance_options chance_options;
    pkmn_gen1_calc_options calc_options;
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
struct Battle : BattleTypesImpl::BattleTypes<Real, Prob, Obs> {
    using TypeList = BattleTypesImpl::BattleTypes<Real, Prob, Obs>;

    class State : public BattleDataImpl::BattleData<LOG_SIZE, ROLLS, Real, Prob, Obs> {
       public:
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
            // std::cout << "normal copy constr invoked" << std::endl;
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

        template <typename State_>
        State(const State_ &other) {
            std::cout << "templated copy constr invoked" << std::endl;
            this->prob = other.prob;
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
                // std::array<uint8_t, 16> obs = obs_ref;
                // for (int i = 0; i < 16; ++i) {
                // 	std::cout << obs[i] << ' ';
                // }
                // std::cout << std::endl;
                // return obs;
                return obs_ref;
            }
            if constexpr (Obs == LogObs) {
                return this->log_buffer;
            }
            if constexpr (Obs == BattleObs) {
                auto *ptr = this->battle.bytes;
                const std::array<uint8_t, SIZE_BATTLE_WITH_PRNG> &obs_ref =
                    *reinterpret_cast<std::array<uint8_t, SIZE_BATTLE_WITH_PRNG> *>(ptr);
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

        // possibly a legacy method
        void get_actions(TypeList::VectorAction &row_actions, TypeList::VectorAction &col_actions) const {
            row_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P1, pkmn_result_p1(this->result),
                                                        row_actions.data(), PKMN_MAX_CHOICES));
            col_actions.resize(pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P2, pkmn_result_p2(this->result),
                                                        col_actions.data(), PKMN_MAX_CHOICES));
        }

        void apply_actions(pkmn_choice row_action, pkmn_choice col_action) {
            // TODO assumes ROLLS = 3
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

            this->result = pkmn_gen1_battle_update(&this->battle, row_action, col_action, &this->options);

            if constexpr (std::is_floating_point_v<Prob>) {
                this->prob = static_cast<Prob>(pkmn_rational_numerator(this->p) / pkmn_rational_denominator(this->p));
            } else if constexpr (!std::is_same_v<Prob, bool>) {
                this->prob = Prob{pkmn_rational_numerator(this->p), pkmn_rational_denominator(this->p)};
            }

            if constexpr (State::dchance) {
                pkmn_gen1_chance_actions *chance_ptr = pkmn_gen1_battle_options_chance_actions(&this->options);
            }

            if constexpr (ROLLS != 0) {
                // TODO clean up static_cast nonsense
                const auto &obs_ref = this->get_obs();
                if (obs_ref[1] & 2 && obs_ref[0]) {
                    this->prob *= static_cast<Prob>(typename TypeList::Q{static_cast<int>(39 / ROLLS), 1});
                    canonicalize(this->prob);
                }
                if (obs_ref[9] & 2 && obs_ref[8]) {
                    this->prob *= static_cast<Prob>(typename TypeList::Q{static_cast<int>(39 / ROLLS), 1});
                    canonicalize(this->prob);
                }
            }

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
    memcpy(data, state.log_buffer.data(), State::log_size);
    memcpy(data + State::log_size, state.battle.bytes, SIZE_BATTLE_WITH_PRNG);
    data[State::log_size + SIZE_BATTLE_WITH_PRNG] = state.result;
    data[State::log_size + SIZE_BATTLE_WITH_PRNG + 1] = row_action;
    data[State::log_size + SIZE_BATTLE_WITH_PRNG + 2] = col_action;
}

template <typename State>
void get_active_hp(const State &state) {
    const int hp_0 = state.battle.bytes[18] + 256 * state.battle.bytes[19];
    const int hp_1 = state.battle.bytes[18 + SIZE_SIDE] + 256 * state.battle.bytes[19 + SIZE_SIDE];
    std::cout << "action hp: " << hp_0 << ' ' << hp_1 << std::endl;
}

// TODO just write extra function in zig, dummy!
template <typename Types>
struct NoSwitch {
    class State : public Types::State {
        using Types::State::State;

        void get_actions() {
            Types::State::get_actions();
            // if (this->result_kind == move) {} // TODO skip based on request status
            // TODO code to prune non choice
        }

        void get_action(Types::VectorAction &row_actions, Types::VectorAction &col_actions) const {
            // TODO duplicate above
        }
    };
};

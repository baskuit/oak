#pragma once

#include <pinyon.hh>
#include <pkmn.h>

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

// basic type list depends on above template params / flags.
// No log or chance means we have to improvise. I'm feeling hash function tbh TODO
// if chance actions then those are used as Obs type. Otherwise use the debug log
// it's faster to use 9 element arrays than vectors for actions containers, certain vectors, etc

template <typename Obs, typename Prob, typename Real = float>
using BattleTypes = DefaultTypes<
	Real,
	pkmn_choice,
	Obs,
	Prob,
	ConstantSum<1, 1>::Value,
	A<9>::Array>;

// ad hoc hash function
struct BattleHash
{
	// TODO always returns 0
	size_t operator()(const std::array<uint8_t, SIZE_BATTLE_NO_PRNG> &data) const { return 0; }
};

// log=0 means we don't even allocate the log buffer. save_debug_log() just saves what it has
// chance determines whether chance actions are used as Obs type. otherwise log or battle hash is used
// chance=false also forces the Prob template param to be bool. using probs but not chance actions is silly
// rolls= the number of possible damage rolls. a simple flooring function determines the roll byte
// Only values: 3, 20, and 39 are aesthetic. imo 3 is best anyway since it's just min, mean, and max.
// if default is used no calc_options are passed to options, so a different specialization is used
// prob type can be floating point or rational type. there are tradeoffs for either
// prob=bool assumes -Dchance/-Dcalc are not enabled

enum BattleObsType
{
	// In order of size aka preference
	ChanceObs,
	LogObs,
	BattleObs,
};

namespace BattleDataImpl
{

	/*
	Different data members are required for a libpkmn battle depending on what features we want
	C++'s `if constexpr` allows for a lot of in-line compile time stuff
	but changing data members at compile time requires templat specialization
	*/

	// all battle wrappers will use this data, so we place it in a base class
	template <size_t LOG_SIZE, size_t ROLLS = 0, typename Prob = mpq_class, BattleObsType Obs = ChanceObs>
	struct BattleDataBase : PerfectInfoState<BattleTypes<std::array<uint8_t, 16>, float, float>>
	{
		pkmn_gen1_battle battle;
		pkmn_gen1_battle_options options;
		pkmn_result result;
		pkmn_result_kind result_kind;
	};

	template <size_t LOG_SIZE, size_t ROLLS = 0, typename Prob = mpq_class, BattleObsType Obs = ChanceObs>
	struct BattleData;

	template <size_t LOG_SIZE, size_t ROLLS, typename Prob, BattleObsType ObsEnum>
	struct BattleData
		: BattleDataBase<LOG_SIZE, ROLLS, Prob, ObsEnum>
	{
		std::array<uint8_t, LOG_SIZE> log_buffer;
		pkmn_gen1_log_options log_options;
		pkmn_gen1_chance_options chance_options;
		pkmn_gen1_calc_options calc_options;
		pkmn_rational *p;

		static constexpr bool dlog = true;
		static constexpr bool dchance = true;
		static constexpr bool dcalc = true;
		static constexpr size_t log_size = LOG_SIZE;
	};

	// No chance/calc needed
	template <size_t LOG_SIZE, BattleObsType Obs>
	struct BattleData<LOG_SIZE, 0, bool, Obs>
		: BattleDataBase<LOG_SIZE, 0, bool, Obs>
	{
		static_assert(Obs == BattleObs || Obs == LogObs);
		std::array<uint8_t, LOG_SIZE> log_buffer;
		pkmn_gen1_log_options log_options;

		static constexpr bool dlog = true;
		static constexpr bool dchance = false;
		static constexpr bool dcalc = false;
		static constexpr size_t log_size = LOG_SIZE;
	};

	template <>
	struct BattleData<0, 0, bool, BattleObs>
		: BattleDataBase<0, 0, bool, BattleObs>
	{
		static constexpr bool dlog = false;
		static constexpr bool dchance = false;
		static constexpr bool dcalc = false;
		static constexpr size_t log_size = 0;
	};

	template <size_t ROLLS, typename Prob, BattleObsType ObsEnum>
	struct BattleData<0, ROLLS, Prob, ObsEnum>
		: BattleDataBase<0, ROLLS, Prob, ObsEnum>
	{
		pkmn_gen1_chance_options chance_options;
		pkmn_gen1_calc_options calc_options;
		pkmn_rational *p;

		static constexpr bool dlog = false;
		static constexpr bool dchance = true;
		static constexpr bool dcalc = true;
		static constexpr size_t log_size = 0;
	};
};

template <
	size_t LOG_SIZE,
	size_t ROLLS = 0,
	BattleObsType Obs = ChanceObs,
	typename Prob = mpq_class,
	typename Real = mpq_class>
struct Battle : BattleTypes<std::array<uint8_t, 16>, float, float>
{
	using TypeList = BattleTypes<std::array<uint8_t, 16>, float, float>;

	class State
		: public BattleDataImpl::BattleData<LOG_SIZE, ROLLS, Prob, BattleObsType::ChanceObs>
	{
	public:
		State(const uint8_t *row_side, const uint8_t *col_side)
		{
			// init: copy sides onto battle and zero initialize certain bits
			// PRNG bytes are left uninitialized (zero initializing is no better, terrible seed)
			memcpy(this->battle.bytes, row_side, SIZE_SIDE);
			memcpy(this->battle.bytes + SIZE_SIDE, col_side, SIZE_SIDE);
			memset(this->battle.bytes + 2 * SIZE_SIDE, 0, SIZE_BATTLE_NO_PRNG - 2 * SIZE_SIDE);

			if constexpr (State::dlog)
			{
				this->log_options = {this->log_buffer.data(), LOG_SIZE};
			}
			if constexpr (State::dchance)
			{
				pkmn_rational_init(&this->chance_options.probability);
				this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
			}

			get_actions();
		}

		template <typename State_>
		State(const State_ &other)
		{
			this->prob = other.prob;
			this->row_actions = other.row_actions;
			this->col_actions = other.col_actions;
			this->terminal = other.terminal;
			memcpy(this->battle.bytes, other.battle.bytes, SIZE_BATTLE_NO_PRNG);
			this->options = other.options;
			this->result = other.result;
			// this->result_kind = other.result_kind; // not needed prolly
			if constexpr (State::dlog)
			{
				this->log_options = {this->log_buffer.data(), LOG_SIZE};
				pkmn_gen1_battle_options_set(&this->options, &this->log_options, NULL, NULL);
			}
			else
			{
				pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
			}
			if constexpr (State::dchance)
			{
				this->p = pkmn_gen1_battle_options_chance_probability(&this->options);
			}
		}

		void get_actions()
		{
			this->row_actions.resize(
				pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P1, pkmn_result_p1(this->result), this->row_actions.data(), PKMN_MAX_CHOICES));
			this->col_actions.resize(
				pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P2, pkmn_result_p2(this->result), this->col_actions.data(), PKMN_MAX_CHOICES));
		}

		void get_actions(
			TypeList::VectorAction &row_actions,
			TypeList::VectorAction &col_actions) const
		{
			row_actions.resize(
				pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P1, pkmn_result_p1(this->result), row_actions.data(), PKMN_MAX_CHOICES));
			col_actions.resize(
				pkmn_gen1_battle_choices(&this->battle, PKMN_PLAYER_P2, pkmn_result_p2(this->result), col_actions.data(), PKMN_MAX_CHOICES));
		}

		void apply_actions(
			pkmn_choice row_action,
			pkmn_choice col_action)
		{
			// TODO assumes ROLLS = 3
			if constexpr (ROLLS != 0)
			{
				this->calc_options.overrides.bytes[0] = 217 + 19 * (this->battle.bytes[383] % 3);
				this->calc_options.overrides.bytes[8] = 217 + 19 * (this->battle.bytes[382] % 3);
				pkmn_gen1_battle_options_set(&this->options, NULL, NULL, &this->calc_options);
			}
			else
			{
				pkmn_gen1_battle_options_set(&this->options, NULL, NULL, NULL);
			}

			this->result = pkmn_gen1_battle_update(&this->battle, row_action, col_action, &this->options);

			if constexpr (std::is_floating_point_v<Prob>)
			{
				this->prob = static_cast<Prob>(pkmn_rational_numerator(this->p) / pkmn_rational_denominator(this->p));
			}
			else
			{
				this->prob = Prob{pkmn_rational_numerator(this->p), pkmn_rational_denominator(this->p)};
			}

			if constexpr (State::dchance)
			{
				pkmn_gen1_chance_actions *chance_ptr = pkmn_gen1_battle_options_chance_actions(&this->options);
			}

			// TODO WARNING trying this disabled
			// memcpy(this->obs.data(), chance_ptr->bytes, 16);
			if constexpr (ROLLS != 0)
			{
				const Obs &obs_ref = this->get_obs();

				if (obs_ref[1] & 2 && obs_ref[0])
				{
					this->prob *= static_cast<typename TypeList::Prob>(
						typename TypeList::Q{13, 1});
				}
				if (obs_ref[9] & 2 && obs_ref[8])
				{
					this->prob *= static_cast<typename TypeList::Prob>(
						typename TypeList::Q{13, 1});
				}
			}

			this->result_kind = pkmn_result_type(this->result);
			if (this->result_kind) [[unlikely]]
			{
				this->terminal = true;
				switch (pkmn_result_type(this->result))
				{
				case PKMN_RESULT_WIN:
				{
					this->payoff = TypeList::Value{Real{0}};
					break;
				}
				case PKMN_RESULT_LOSE:
				{
					this->payoff = TypeList::Value{Real{1}};
					break;
				}
				case PKMN_RESULT_TIE:
				{
					// TODO mpq_class/float stuff here
					this->payoff = TypeList::Value{Real{Rational<>(1, 2)}};
					break;
				}
				case PKMN_RESULT_ERROR:
				{
					std::exception();
				}
				}
			}
		}
	};
};

// Utils

// counts number of alive mons using pointer to raw battle data - no templating needed
int n_alive(const uint8_t *data)
{
	int n{0};
	for (int s = 0; s < 2; ++s)
	{
		for (int p = 0; p < 6; ++p)
		{
			const int index = SIZE_SIDE * s + 24 * p + 18;
			const bool alive = data[index] || data[index + 1];
			if (alive)
				++n;
		}
	}
	return n;
}

// here we need to call the `apply_actions` function so internals are updated so we have to template
// the data pointer is assumed to be a buffer(slice) big enough for that turn
template <typename State>
void apply_actions_with_log(
	State &state,
	pkmn_choice row_action,
	pkmn_choice col_action,
	uint8_t *data)
{
	state.apply_actions(row_action, col_action);
	memcpy(state.log_buffer.data(), data, State::log_size);
	memcpy(state.battle.bytes, data, SIZE_BATTLE_WITH_PRNG);
	data[State::log_size + SIZE_BATTLE_WITH_PRNG] = state.result;
	data[State::log_size + SIZE_BATTLE_WITH_PRNG + 1] = row_action;
	data[State::log_size + SIZE_BATTLE_WITH_PRNG + 2] = col_action;
}

// assumes the output objects have the pinyon-esque 'row/col_strategy' members
template <typename State, typename RowModelOutput, typename ColModelOutput>
void apply_actions_with_log_and_eval(
	State &state,
	pkmn_choice row_action,
	pkmn_choice col_action,
	uint8_t *data)
{
	state.apply_actions(row_action, col_action);
	memcpy(state.log_buffer, data, State::log_size);
	memcpy(state.battle, data + State::log_size, SIZE_BATTLE_WITH_PRNG);
	data[State::log_size + SIZE_BATTLE_WITH_PRNG] = state.result;
	data[State::log_size + SIZE_BATTLE_WITH_PRNG + 1] = row_action;
	data[State::log_size + SIZE_BATTLE_WITH_PRNG + 2] = col_action;
	// TODO row_value, col_value, rows, row_model_row_policy, col_model_row_policy, etc
}

// TODO just write extra function in zig, dummy!
template <typename Types>
struct NoSwitch
{
	class State : public Types::State
	{
		using Types::State::State;

		void get_actions()
		{
			Types::State::get_actions();
			// if (this->result_kind == move) {} // TODO skip based on request status
			// TODO code to prune non choice
		}

		void get_action(
			Types::VectorAction &row_actions,
			Types::VectorAction &col_actions) const
		{
			// TODO duplicate above
		}
	};
};

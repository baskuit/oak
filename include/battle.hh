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
	ConstantSum<1, 1>::Value<Real>,
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
template <
	int log = 64,
	bool chance = true,
	int rolls = 39,
	typename Prob,
	typename Real>
struct Battle;

// log size
// prob type
// real type
// obs type
// clamping

enum BattleObs
{
	log,
	chance,
	battle,
};

struct BattleDataCommon
{
	pkmn_gen1_battle battle;
	pkmn_gen1_battle_options options;
	pkmn_result result;
	pkmn_result_kind result_kind;
};

template <size_t LOG_SIZE, size_t ROLLS = 0, typename Prob = mpq_class, BattleObs Obs = chance>
struct BattleData;

template <size_t LOG_SIZE, size_t ROLLS, typename Prob, BattleObs Obs>
struct BattleData<LOG_SIZE, ROLLS, Prob, Obs>
	: BattleDataCommon
{
	static constexpr bool dlog = true;
	static constexpr bool dchance = true;
	static constexpr bool dcalc = true;

	std::array<uint8_t, LOG_SIZE> log_buffer;
	pkmn_gen1_chance_options chance_options;
	pkmn_gen1_calc_options calc_options;
	pkmn_rational *p;
};

template <size_t LOG_SIZE, BattleObs Obs>
struct BattleData<LOG_SIZE, 0, bool, Obs>
	: BattleDataCommon
{
	static constexpr bool dlog = true;
	static constexpr bool dchance = false;
	static constexpr bool dcalc = false;

	std::array<uint8_t, LOG_SIZE> log_buffer;
};

template <>
struct BattleData<0, 0, bool, battle>
	: BattleDataCommon
{
	static constexpr bool dlog = false;
	static constexpr bool dchance = false;
	static constexpr bool dcalc = false;
};


template <
	size_t LOG_SIZE,
	size_t ROLLS = 0,
	typename Prob = mpq_class,
	typename Real = mpq_class,
	BattleObs Obs = chance>
struct Battle2 
{
	class State : BattleData<LOG_SIZE, ROLLS, Prob, Obs>
	{
	};
};

/*
we use fundamentally different logic depending on the template parameters above
while much of it can be accomplished using `if constexpr`, we can't do that with data members
*/

using Btl = Battle<>;

// this is the most complicated template specialization since it has the most features
// the comments will generally apply to the other specializations
template <int log, int rolls, typename prob>
struct Battle<log, true, rolls, prob> : BattleTypes<std::array<uint8_t, 16>, prob>
{
	class State
	{
	public:
		static constexpr int log_size = log;
		static constexpr int rolls = rolls;

		pkmn_gen1_battle battle;
		pkmn_gen1_battle_options options;
		std::array<uint8_t, log> log_buffer;

		pkmn_result result{}; // previous bugs caused by not initializing libpkmn stuff
		pkmn_result_kind result_kind;
		pkmn_gen1_chance_options chance_options{};
		pkmn_rational *p{};
		pkmn_gen1_calc_options calc_options{};

		pkmn_gen1_log_options log_options;

		std::vector<uint8_t> debug_log{};

		State(const uint8_t *row_side, const uint8_t *col_side)
		{
			memcpy(battle.bytes, row_side, SIZE_SIDE);
			memcpy(battle.bytes + SIZE_SIDE, col_side, SIZE_SIDE);
			memset(battle.bytes + 2 * SIZE_SIDE, 0, SIZE_BATTLE_NO_PRNG - 2 * SIZE_SIDE);

			log_options = {log_buffer.data(), log_size};
			pkmn_rational_init(&chance_options.probability);
			pkmn_gen1_battle_options_set(&options, &log_options, &chance_options, &calc_options);
			p = pkmn_gen1_battle_options_chance_probability(&options);
			get_actions();
		}

		State(const State &other)
		{
			this->row_actions = other.row_actions;
			this->col_actions = other.col_actions;
			this->terminal = other.terminal;
			result = other.result;
			memcpy(battle.bytes, other.battle.bytes, SIZE_BATTLE_WITH_PRNG);
			options = other.options;
			log_options = {log.data(), log_size};
			pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);
			p = pkmn_gen1_battle_options_chance_probability(&options);
			log_buffer = other.log_buffer;
		}

		void get_actions()
		{
			this->row_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result), this->row_actions.data(), PKMN_MAX_CHOICES));
			this->col_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result), this->col_actions.data(), PKMN_MAX_CHOICES));
		}

		void get_actions(
			TypeList::VectorAction &row_actions,
			TypeList::VectorAction &col_actions) const
		{
			row_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result), row_actions.data(), PKMN_MAX_CHOICES));
			col_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result), col_actions.data(), PKMN_MAX_CHOICES));
		}

		void apply_actions(
			pkmn_choice row_action,
			pkmn_choice col_action)
		{
			calc_options.overrides.bytes[0] = 217 + 19 * (battle.bytes[383] % 3);
			calc_options.overrides.bytes[8] = 217 + 19 * (battle.bytes[382] % 3);
			pkmn_gen1_battle_options_set(&options, NULL, NULL, &calc_options);

			result = pkmn_gen1_battle_update(&battle, row_action, col_action, &options);

			if constexpr (std::is_floating_point_v<prob>)
			{
				this->prob = static_cast<prob>(pkmn_rational_numerator(p) / pkmn_rational_denominator(p));
			}
			else
			{
				this->prob = prob{pkmn_rational_numerator(p), pkmn_rational_denominator(p)};
			}

			pkmn_gen1_chance_actions *chance_ptr = pkmn_gen1_battle_options_chance_actions(&options);

			// TODO WARNING trying this disabled
			// memcpy(this->obs.data(), chance_ptr->bytes, 16);
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

			result_kind = pkmn_result_type(result);
			if (result_kind) [[unlikely]]
			{
				this->terminal = true;
				switch (pkmn_result_type(result))
				{
				case PKMN_RESULT_WIN:
				{
					this->payoff = TypeList::Value{1.0f};
					break;
				}
				case PKMN_RESULT_LOSE:
				{
					this->payoff = TypeList::Value{0.0f};
					break;
				}
				case PKMN_RESULT_TIE:
				{
					this->payoff = TypeList::Value{0.5f};
					break;
				}
				case PKMN_RESULT_ERROR:
				{
					exit(1);
				}
				}
			}
		}

		const Obs &get_obs() const
		{
			auto *ptr = pkmn_gen1_battle_options_chance_actions(&options)->bytes;
			const Obs &obs_ref = *reinterpret_cast<std::array<uint8_t, 16> *>(ptr);
			return obs_ref;
		}

		void randomize_transition(TypeList::PRNG &device)
		{
			uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
			*(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
		}

		void randomize_transition(const TypeList::Seed seed)
		{
			uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
			*(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = seed;
		}
	};
};

// No logging
template <int rolls, typename prob>
struct Battle<0, true, rolls, prob> : BatleTypes<std::array<uint8_t, 16>, prob>
{
	class State
	{
	public:
		static constexpr int log_size = 0;
		static constexpr int rolls = rolls;

		pkmn_gen1_battle battle;
		pkmn_gen1_battle_options options;

		pkmn_result result{}; // previous bugs caused by not initializing libpkmn stuff
		pkmn_result_kind result_kind;
		pkmn_gen1_chance_options chance_options{};
		pkmn_rational *p{};
		pkmn_gen1_calc_options calc_options{};

		State(const uint8_t *row_side, const uint8_t *col_side)
		{
			memcpy(battle.bytes, row_side, SIZE_SIDE);
			memcpy(battle.bytes + SIZE_SIDE, col_side, SIZE_SIDE);
			memset(battle.bytes + 2 * SIZE_SIDE, 0, SIZE_BATTLE_NO_PRNG - 2 * SIZE_SIDE);

			pkmn_rational_init(&chance_options.probability);
			pkmn_gen1_battle_options_set(&options, NULL, &chance_options, &calc_options);
			p = pkmn_gen1_battle_options_chance_probability(&options);
			get_actions();
		}

		State(const State &other)
		{
			this->row_actions = other.row_actions;
			this->col_actions = other.col_actions;
			this->terminal = other.terminal;
			result = other.result;
			memcpy(battle.bytes, other.battle.bytes, SIZE_BATTLE_WITH_PRNG);
			options = other.options;
			pkmn_gen1_battle_options_set(&options, NULL, NULL, NULL);
			p = pkmn_gen1_battle_options_chance_probability(&options);
		}

		void get_actions()
		{
			this->row_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result), this->row_actions.data(), PKMN_MAX_CHOICES));
			this->col_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result), this->col_actions.data(), PKMN_MAX_CHOICES));
		}

		void get_actions(
			TypeList::VectorAction &row_actions,
			TypeList::VectorAction &col_actions) const
		{
			row_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P1, pkmn_result_p1(result), row_actions.data(), PKMN_MAX_CHOICES));
			col_actions.resize(
				pkmn_gen1_battle_choices(&battle, PKMN_PLAYER_P2, pkmn_result_p2(result), col_actions.data(), PKMN_MAX_CHOICES));
		}

		void apply_actions(
			pkmn_choice row_action,
			pkmn_choice col_action)
		{
			calc_options.overrides.bytes[0] = 217 + 19 * (battle.bytes[383] % 3);
			calc_options.overrides.bytes[8] = 217 + 19 * (battle.bytes[382] % 3);
			pkmn_gen1_battle_options_set(&options, NULL, NULL, &calc_options);
			result = pkmn_gen1_battle_update(&battle, row_action, col_action, &options);
			if constexpr (std::is_floating_point_v<prob>)
			{
				this->prob = static_cast<prob>(pkmn_rational_numerator(p) / pkmn_rational_denominator(p));
			}
			else
			{
				this->prob = prob{pkmn_rational_numerator(p), pkmn_rational_denominator(p)};
			}
			pkmn_gen1_chance_actions *chance_ptr = pkmn_gen1_battle_options_chance_actions(&options);
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

			result_kind = pkmn_result_type(result);
			if (result_kind) [[unlikely]]
			{
				this->terminal = true;
				switch (pkmn_result_type(result))
				{
				case PKMN_RESULT_WIN:
				{
					this->payoff = TypeList::Value{1.0f};
					break;
				}
				case PKMN_RESULT_LOSE:
				{
					this->payoff = TypeList::Value{0.0f};
					break;
				}
				case PKMN_RESULT_TIE:
				{
					this->payoff = TypeList::Value{0.5f};
					break;
				}
				case PKMN_RESULT_ERROR:
				{
					exit(1);
				}
				}
			}
		}

		const Obs &get_obs() const
		{
			auto *ptr = pkmn_gen1_battle_options_chance_actions(&options)->bytes;
			const Obs &obs_ref = *reinterpret_cast<std::array<uint8_t, 16> *>(ptr);
			return obs_ref;
		}

		void randomize_transition(TypeList::PRNG &device)
		{
			uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
			*(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
		}

		void randomize_transition(const TypeList::Seed seed)
		{
			uint8_t *battle_prng_bytes = battle.bytes + n_bytes_battle;
			*(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = seed;
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
	battle.apply_actions(row_action, col_action);
	memcpy(battle.log_buffer, data, State::log_size);
	memcpy(battle.battle, data, SIZE_BATTLE_WITH_PRNG);
	data[State::log_size + SIZE_BATTLE_WITH_PRNG] = battle.result;
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
	battle.apply_actions(row_action, col_action);
	memcpy(battle.log_buffer, data, State::log_size);
	memcpy(battle.battle, data + State::log_size, SIZE_BATTLE_WITH_PRNG);
	data[State::log_size + SIZE_BATTLE_WITH_PRNG] = battle.result;
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
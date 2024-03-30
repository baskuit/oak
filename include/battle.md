# Pinyon

Pinyon has many out-of-the-box search utilities for any game. These are accessilbe once it is given a C++ wrapper that complies with the pinyon interface (e.g. `get_actions`, `apply_actions`, `randomize_transition`, etc.)

# Libpkmn

Libpkmn is a high performance simulator for RBY that matches a 'patched' showdown.

At the time of writing this library, libpkmn has experimental support for identifying and controlling the transition of the state. It also supports quantifying the probability, too. These features are needed for pinyon's solvers. Otherwise only mcts (aka tree-bandit) will work.

Because this support is experimental, `oak` is pinned to a fork which begins at a fixed commit on the main repo. The fork also has a patch to the probability measuring mode and a chance to the debug log binary procol to support the encoding/decoding of model values, policies, and (TODO) data matrices.

# Template

The wrapper for the battle object has several template parameters.

```cpp
template <
	size_t LOG_SIZE,
	size_t ROLLS = 0,
	BattleObsT Obs = ChanceObs,
	typename Prob = mpq_class,
	typename Real = mpq_class>
```

These parameters control the underlying types of certain key data members. The various configurations of these parameters correspond to enabling/disabling of the `-Dlog`, `-Dchance`, `-Dcalc` flags on likpkmn.

> It is the user's reponsibility to ensure that the library was compiled with the correct flags.

## Parameters

* `LOG_SIZE`
If zero then no log buffer is used. A size of `64` is large enough for the typical OU battle. Logs are required to produce the html debug logs. These logs are modified on the oak branch to also store and display *eval* information
* `ROLLS`
The number of possible rolls to clamp to. If zero then no clamping is done and no calc actions are set. Currently the only implemented/tested options are 2, 3, 20, 39. These are the uniform sub-ranges of the 39 usual rolls.

* `Obs`
Pinyon uses a special `Obs` type to distinguish different transitions from eachother. The parameter is of the enum `BattleObsT` type. This enum has 3 values: `BattleObs`, `LogObs`, and `ChanceObs`. These correspond to use the 376 battle bytes, the `LOG_SIZE` log bytes, or the 16 chance action bytes.

* `Prob`
* `Real`
These both are just basic choices for types. `float` and `mpq_class`/`Rational<>` are the expected choices. There are valid reasons for using almost any permutation of them. In particular, using `void` as the underlying bool probability will disabled the get_prob() command.

### Building `libpkmn`

The user must be careful that the `libpkmn` library they compiled matches the features they are requesting with the wrapper.

> E.g. if `-dLog` was used then `LOG_SIZE` tempalte parameter must be non zero

Similarly if logs, chance actions are expected by the wrapper then they must be enabled in `libpkmn`.


# `Battle<...>` Implementation

This wrapper is intended to be (nearly) optimal for every valid combination of template parameters. To minimze the amount of duplicate code, this is done using template speciailization for the ancillary data members that are needed for `libpkmn` C bindings. The differing control flow information is then handled with monstly `if constexpr` in a derived class that implements the methods.

## `struct BattleTypes<...>`

This is the basic pinyon 'type list' that defines:
* fundamental arithmetic types `Real`, `Prob`
* data oriented types like `VectorReal`
	(it is faster to use an array with 9 elements for action indexes data since we avoid reallocation)
* performance tweaking classes like `Mutex`, `PRNG`
* `ObsHashType` which defines an operator that converts the three observation types into a `uint64_t` for use with `std::unordered_map<>`

The `DefaultTypes` utility is almost good enough, but we derive from it to shadow the `ObsHashType` with a correct implementation.

## `struct BattleDataBase<>`

The data members are common to all are defined here. Some pinyon functions like `is_terminal` can be implemented here as well


## `struct BattleData<... > : BattleDataBase<...>`


There is a 'maximal' specialization where all features of `libpkmn` are being used, hence all optional data members are defined. It is a useful example.

```cpp
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
```

`log_options` holds a pointer to the `log_buffer` data.
`chance_options` may be only required for construction (TODO remove if thats true). 
`calc_options` holds the damage roll bytes.
`p` is a pointer to the probability data in the `options` struct.
`clamped` is a run-time flag to enable/disable damage roll clamping.

The `set` method is defined here since in the other specialization, we may pass `NULL`.

Lastly the `d...` vars are a useful convencience for the control flow in the derived classes.

## `class Battle<...>::State : BattleData<... >`

The final `State` implementation that all pinyon utilities will ask for.

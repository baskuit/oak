# Template

The `pinyon` wrapper for the battle object has several template parameters.

```cpp
template <
	size_t LOG_SIZE,
	size_t ROLLS = 0,
	BattleObsT Obs = ChanceObs,
	typename Prob = mpq_class,
	typename Real = mpq_class>
```

where the choices for `Obs` correspond to using the chance actions, log buffer, or battle buffer.

## Parameters

* `LOG_SIZE`
If zero then no log buffer is used. A size of `64` is probably good enough to not overflow in the Overused tier.
* `ROLLS`
If zero then no clamping is done and no calc actions are set. Currently the only implemented/tested option is `3` TODO

* `Obs`
See the `get_obs` pinyon function.

* `Prob`
* `Real`
These both are just basic choices for types. `float` and `mpq_class`/`Rational<>` are the expected choices. There are valid reasons for using almost any permutation of them.

# `libpkmn`

This repo would not be possible without pkmn/engine. A fast simulator for use in compiled languages is probably a basic necessity of AI.
In fact, `pinyon` was written in *anticipation* of such a library; there were none available at the start of the project.

## Support

This repo makes heavy use of the `chance` and `calc` features of the battle library. However these features are not *officially* supported and are subject to total overhaul at any moment. Also that moment should not be expected to be soon.

## `no-commit` branch

This repo uses a patched version of `libpkmn` to make use of the `-Dchance` and `-Dcalc` features properly.

> This branch will **not be updated** with the main branch as time goes on.



### Building `libpkmn`

The user must be careful that the `libpkmn` library they compiled matches the features they are requesting with the wrapper.

> E.g. if `-dLog` was used then `LOG_SIZE` tempalte parameter must be non zero

Similarly if logs, chance actions are expected by the wrapper then they must be enabled in `libpkmn`.


# Notes

```cpp
    struct BattleDataBase<>;
```

```cpp
	struct BattleData<... > : BattleDataBase<...>;
```


```cpp
	class Battle<... >::State : BattleData<... >;
```
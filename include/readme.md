## /data

* moves.h
* offsets.h
* species.h
* types.h

This directory fleshes out the barebones libpkmn C API. It duplicates the basic enum definitions and ancillary data (base power, pp, base stats) that libpkmn does not expose.

* strings.h

String utilities: decoding/formatting byte data, move/species names, etc.

## /battle

* debug-log.h

`DebugLog<size_t log_size>` is a helper struct for creating debug logs. It provides a wrapper for `pkmn_gen1_battle_update` that stores the frame data of each update.

* init.h

`Battle::init(auto p1, auto p2, seed = 0)` returns a `pkmn_gen1_battle` with initialized stats, order bytes, etc.

## /pi

* abstract.h

Abstract:: defines structs that bucket similar states together by approximating hp, stats, etc.
This means states that are only perturbed can be treated as the same.

* exp3.h

`Exp3::JointBanditData` is a compact struct that stores Exp3 bandit data for both players, with implementations for selection and update.

* mcts.h

Basic Monte Carlo tree search with actions clamping and duration sampling (TODO)

* tree.h

This structure stores all the search stats in a way that guarantees no "collisions". No two different histories will access the same stats.

* eval.h

The previous `/pi` headers create a bootstrap for our HCE. We can take the average MCTS root value as an estimate for the value of a 1v1

These values are cached or computed at initialization and then combined into a value estimator for the full 6v6 game.

This HCE can then be used with `mcts.h` as a replacement for the slow monte-carlo rollout eval
Or it can be used in the forthcoming *graph search* that uses the abstract battle to produce a hash for a transposition table

* tt.h
* ucb.h

## /ii

Imperfect info search is limited to IS-MCTS, basically. Out best hope currently is to quickly determinize the private observations of the acting player. This is particularly fast and accurate to do in `gen1randombattles`

I don't want to write client code so I'm going to lean on @pkmn/client. Therefore there will be a simple struct clone of client observations here
Then we just need methods to fill in the blanks with valid randbats teams

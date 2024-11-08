## /data

* moves.h
* offsets.h
* species.h
* types.h

This directory fleshes out the barebones libpkmn C API. It duplicates the basic enum definitions and ancillary data (base power, pp, base stats) that libpkmn does not expose.

strings.h

String utilities.

## /battle

* debug-log.h

`DebugLog<size_t log_size>` is a helper struct for creating debug logs. It stores the binary data in standard arrays and merely wraps the battle update function

* init.h

`Battle::init(auto p1, auto p2, seed = 0)` returns a `pkmn_gen1_battle` with initialized stats, order bytes, etc.

## /pi

abstract.h

Abstract:: defines structs that bucket similar states together by approximating hp, stats, etc.
This means that states with only small continuous differences can be treated as the same.

The headers below define two different kinds of perfect info search. The first is a provably correct approach that uses chance action 'keys' to keep track of states. The second is a fast approach that uses an imperfect hash and bandit algorithm.

eval.h

A hand crafted eval that uses precomputed 1v1 values and combines them into a crude full battle estimate TODO

exp3.h

`` is a 128 byte struct that holds joint Exp3 bandit data for both players. To manage this it uses 24bit integers for

mcts.h

Supports vanilla monte-carlo and eval.h for leaf value estimation.

tree.h

This structure stores all the search stats in a way that guarantees no "collisions". No two different histories will access the same stats.


tt.h
ucb.h

## /ii

Imperfect info search is limited to IS-MCTS, basically. Out best hope currently is to quickly determinize the private observations of the acting player. This is particularly fast and accurate to do in `gen1randombattles`

I don't want to write client code so I'm going to lean on @pkmn/client. Therefore there will be a simple struct clone of client observations here
Then we just need methods to fill in the blanks with valid randbats teams

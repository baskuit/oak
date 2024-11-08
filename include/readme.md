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
eval.h
exp3.h
mcts.h
tree.h
tt.h
ucb.h

## /ii

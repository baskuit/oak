# battle

This folder contains all things that pertain to libpkmn, it's wrapper, and search in the perfect info context

* battle.h
This is the C++ wrapper for the libpkmn c interface. We construct battles using side bytes currently

* chance.h
Utilities for pkmn_gen1_chance_actions, including duration masking and printing

* eval.h
This is our hand crafted eval. We use cached 1v1 values that are computed at the start of the battle

* bucketing.h
Low resolution view of perfect info battle. 

# pi

Fast monte carlo graph search on bucketed states. Requires the opponents team be determinized. The hash and value estimation try to 'effeciently update', a la Stockfish.

# ii

System code for determinizing client observations and managing many perfect-info search workers. Combine worker policies/q-values into final policy for the bot
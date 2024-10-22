# About

This repo is firstly a combination of computer search with `libpkmn`.

# Scope

Currently, the target is a Pokemon-Showdown bot for the `gen1randombattle` format - probably the simplest format that is actively played.

The approach is to combine sound perfect info search with determinization of the opponents private information. It is essentially IS-MCTS.

# Building

Must have cmake and zig installed

```
git clone --recurse-submodules https://github.com/baskuit/oak
cd oak && git submodule update --recursive
mkdir build && cd build && cmake .. && make && cd ..
```
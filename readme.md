# About

This repo is firstly a combination of computer search with `libpkmn`.

# Scope

Currently, the target is a Pokemon-Showdown bot for the `gen1randombattle` format - probably the simplest format that is actively played.

The approach is to combine sound perfect info search with determinization of the opponents private information. It is essentially IS-MCTS.

The code is detailed in `include/readme.md`.

# Building

Must have cmake and zig installed. The bash below clones the repo, builds the lipkmn libraries

```
git clone --recurse-submodules https://github.com/baskuit/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && ./dev/libpkmn
mkdir build && cd build && cmake .. && make && cd ..
```

Typescript code and showdown dev scripts will usually require that the `pokemon-showdown` submodule is built:

```
cd extern/pokemon-showdown && node build
```

# Status

This repo is far from completion. The following is completed

* Write a usuable C++ API

* Clone the gen1randombattle team gen logic

* Implement perfect-info MCTS for accurate 1v1 estimation

Remaining items:

* Create adequately strong HCE using 1v1 values

TODO

# About

This repo is foremost an application search to `libpkmn`. Currently I am working on a fast transposition-table based monte-carlo graph search: at the root node it uses MatrixUCB and at the child nodes it uses joint UCB for speed. This is directed towards attacking the `gen1randombattle` format.

# Building

Must have cmake and zig installed

```
git clone --recurse-submodules https://github.com/baskuit/oak
cd oak && git checkout restart && git submodule update --recursive
mkdir build && cd build && cmake .. && make && cd .. 
```
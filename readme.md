# Building

Must have cmake and zig installed

```
git clone --recurse-submodules https://github.com/baskuit/oak
cd oak && git checkout restart && git submodule update --recursive
mkdir build && cd build && cmake .. && make && cd .. 
```
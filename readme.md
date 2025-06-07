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

The `pkmn-debug` utility is built via
```
cd extern/engine
npm install && npm run compile
```
and run with 
```
./extern/engine/src/bin/pkmn-debug
```

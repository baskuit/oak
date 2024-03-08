#!/bin/bash

for file in *log; do
    filename=$(basename "$file" .log)
    ../extern/engine/src/bin/pkmn-debug "$file" > "${filename}.html"
    echo "Converted $file to ${filename}.html"
done

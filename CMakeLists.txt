cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(extern/pinyon)
target_include_directories(pinyon INTERFACE extern/pinyon/src)

include_directories(extern/engine/zig-out/include)
set(libpkmn ${PROJECT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a)

add_subdirectory(extern/lrslib)
target_include_directories(lrslib64 INTERFACE extern/lrslib/src)
target_include_directories(lrslib128 INTERFACE extern/lrslib/src)

add_executable(bull ./src/bull.cc)
target_link_libraries(bull pinyon)

add_executable(bull-pinyon ./src/bull-pinyon.cc)
target_link_libraries(bull-pinyon pinyon)
target_link_libraries(bull-pinyon ${libpkmn})

add_executable(prob-test ./src/prob-test.cc)
target_link_libraries(prob-test pinyon)
target_link_libraries(prob-test ${libpkmn})

add_executable(simple-prob ./src/simple-prob.cc)
target_link_libraries(simple-prob pinyon)
target_link_libraries(simple-prob ${libpkmn})

add_executable(main ./src/main.cc)
target_link_libraries(main pinyon)
target_link_libraries(main ${libpkmn})

add_executable(tune ./src/tune.cc)
target_link_libraries(tune pinyon)
target_link_libraries(tune ${libpkmn})

add_executable(four ./src/four.cc)
target_link_libraries(four pinyon)
target_link_libraries(four lrslib128)
target_link_libraries(four ${libpkmn})

add_executable(ab-mpq ./src/ab-mpq.cc)
target_link_libraries(ab-mpq pinyon)
target_link_libraries(ab-mpq ${libpkmn})

add_executable(ab-self-play ./src/ab-self-play.cc)
target_link_libraries(ab-self-play pinyon)
target_link_libraries(ab-self-play ${libpkmn})

add_executable(verify ./src/verify.cc)
target_link_libraries(verify pinyon)
target_link_libraries(verify ${libpkmn})

add_executable(eval-log ./src/eval-log.cc)
target_link_libraries(eval-log pinyon)
target_link_libraries(eval-log ${libpkmn})

add_executable(actor-pool-test ./src/actor-pool-test.cc)
target_link_libraries(actor-pool-test pinyon)
target_link_libraries(actor-pool-test ${libpkmn})

add_executable(parse-self-play ./src/parse-self-play.cc)
target_link_libraries(parse-self-play ${libpkmn})

add_executable(a ./benchmark/a.cc)
target_link_libraries(a pinyon)
target_link_libraries(a ${libpkmn})

add_executable(b ./benchmark/b.cc)
target_link_libraries(b pinyon)
target_link_libraries(b ${libpkmn})
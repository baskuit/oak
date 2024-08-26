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

add_executable(main ./src/io.cc)
target_link_libraries(main pinyon)

add_executable(a ./benchmark/a.cc)
target_link_libraries(a pinyon)
target_link_libraries(a ${libpkmn})

add_executable(b ./benchmark/b.cc)
target_link_libraries(b pinyon)
target_link_libraries(b ${libpkmn})
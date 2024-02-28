set(CMAKE_MINIMUM_REQUIRED_VERSION 3.8)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern/pinyon)

include_directories(extern/engine/zig-out/include)
set(libpkmn ${PROJECT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a)

add_executable(main src/main.cc)
target_link_libraries(main ${libpkmn})

add_executable(bull ./src/bull.cc)
target_link_libraries(bull pinyon)

add_executable(bull-pinyon ./src/bull-pinyon.cc)
target_link_libraries(bull-pinyon pinyon)
target_link_libraries(bull-pinyon ${libpkmn})

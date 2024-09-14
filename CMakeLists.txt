cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 23)

include(CTest)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a
  COMMAND
    zig build -Dshowdown -Dpic -Dchance -Dcalc -Dlog -Doptimize=ReleaseSafe
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine
  COMMENT
    "Building libpkmn with the zig in PATH"
)

add_library(libpkmn STATIC IMPORTED)
set_target_properties(libpkmn PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a
)

add_subdirectory(extern/pinyon)

add_custom_target(build_zig_lib ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a)

add_executable(main src/main.cc)
add_dependencies(main build_zig_lib)
target_include_directories(main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include
)
target_link_libraries(main PRIVATE libpkmn)
target_link_libraries(main PRIVATE pinyon)

add_executable(libpkmn_test src/test.cc)
add_dependencies(libpkmn_test build_zig_lib)
target_include_directories(libpkmn_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include
)
target_link_libraries(libpkmn_test PRIVATE libpkmn)
target_link_libraries(libpkmn_test PRIVATE pinyon)
add_test(NAME libpkmn_test_  COMMAND libpkmn_test)
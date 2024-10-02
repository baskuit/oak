cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 23)

include(CTest)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ZIG_BUILD_MODE "Debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ZIG_BUILD_MODE "ReleaseFast")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ZIG_BUILD_MODE "ReleaseSafe")
else()
  set(ZIG_BUILD_MODE "ReleaseSafe")
endif()

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/lib/libpkmn-showdown.a
  COMMAND
    zig build -Dshowdown -Dpic -Dchance -Dcalc -Dlog -Doptimize=${ZIG_BUILD_MODE}
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine
  COMMENT
    "Building libpkmn with Zig in PATH, using ${ZIG_BUILD_MODE} mode"
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

add_executable(teamgen src/teamgen.cc)
add_dependencies(teamgen build_zig_lib)
target_include_directories(teamgen PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include
)
target_link_libraries(teamgen PRIVATE libpkmn)
target_link_libraries(teamgen PRIVATE pinyon)
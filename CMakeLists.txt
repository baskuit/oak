cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.2)

set(CMAKE_CXX_STANDARD 23)

set(ZIG_BUILD_MODE "ReleaseSafe")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ZIG_BUILD_MODE "Debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ZIG_BUILD_MODE "ReleaseFast")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ZIG_BUILD_MODE "ReleaseSafe")
endif()

set(ZIG_BASE_BUILD zig build -Dshowdown -Dpic -Doptimize=${ZIG_BUILD_MODE})
set(WORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine)

function(build_libpkmn target dir extra_flags)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${dir}/libpkmn-showdown.a
    COMMAND ${ZIG_BASE_BUILD} ${extra_flags} --prefix-lib-dir ${CMAKE_CURRENT_BINARY_DIR}/${dir}
    WORKING_DIRECTORY ${WORK_DIR}
    COMMENT "Building ${target} in ${ZIG_BUILD_MODE} mode"
  )
  add_library(${target} STATIC IMPORTED)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${dir}/libpkmn-showdown.a)
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include)
endfunction()

set(fast_flags -Doption=ebc=false -Doption=miss=false -Doption=advance=false)
set(chance_flags -Dchance -Doption=ebc=false -Doption=miss=false -Doption=advance=false -Dcalc -Doption=key=true)
set(client_flags -Dlog)

build_libpkmn(libpkmn_fast fast ${fast_flags})
build_libpkmn(libpkmn_chance chance ${chance_flags})
build_libpkmn(libpkmn_client client ${client_flags})

add_subdirectory(extern/pinyon)

add_custom_target(build_chance ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/chance/libpkmn-showdown.a)
add_custom_target(build_client ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/client/libpkmn-showdown.a)

add_executable(debug-log src/debug-log.cc)
add_dependencies(debug-log build_client)
target_include_directories(debug-log PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(debug-log PRIVATE libpkmn_client pinyon)

add_executable(mcts src/mcts.cc)
add_dependencies(mcts build_chance)
target_include_directories(mcts PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mcts PRIVATE libpkmn_chance pinyon)

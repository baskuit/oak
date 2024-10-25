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

build_libpkmn(libpkmn_fast fast "-Doption=ebc=false -Doption=miss=false -Doption=advance=false")
build_libpkmn(libpkmn_correct correct "-Doption=ebc=false" "-Doption=miss=false" "-Doption=advance=false" "-Dchance" "-Dcalc" "-Doption=key=true")
build_libpkmn(libpkmn_client client "-Doption=ebc=true" "-Doption=miss=true" "-Doption=advance=true" "-Dlog")

add_subdirectory(extern/pinyon)

add_custom_target(build_zig_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/client/libpkmn-showdown.a)

# add_executable(main src/main.cc)
# add_dependencies(main build_zig_lib)
# target_include_directories(main PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(main PRIVATE libpkmn_client pinyon)

# add_executable(teamgen src/teamgen.cc)
# add_dependencies(teamgen build_zig_lib)
# target_include_directories(teamgen PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include
# )
# target_link_libraries(teamgen PRIVATE libpkmn)
# target_link_libraries(teamgen PRIVATE pinyon)

add_executable(mcts src/mcts.cc)
add_dependencies(mcts build_zig_lib)
target_include_directories(mcts PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mcts PRIVATE libpkmn_client pinyon)

# add_executable(oak-test src/test.cc)
# add_dependencies(oak-test build_zig_lib)
# target_include_directories(oak-test PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include
# )
# target_link_libraries(oak-test PRIVATE libpkmn)
# target_link_libraries(oak-test PRIVATE pinyon)

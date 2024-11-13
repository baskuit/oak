cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.2)

set(CMAKE_CXX_STANDARD 23)

function(create_library target mode dir)
  add_library(${target} STATIC IMPORTED)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/${mode}/${dir}/libpkmn-showdown.a)
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include)
endfunction()

create_library(libpkmn_client release client)
create_library(libpkmn_chance release chance)
create_library(libpkmn_fast release fast)

add_custom_target(build_client_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/client/libpkmn-showdown.a)
add_custom_target(build_chance_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/chance/libpkmn-showdown.a)
add_custom_target(build_fast_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/fast/libpkmn-showdown.a)
add_custom_target(build_client_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/client/libpkmn-showdown.a)
add_custom_target(build_chance_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/chance/libpkmn-showdown.a)
add_custom_target(build_fast_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/fast/libpkmn-showdown.a)

add_executable(randbat src/randbat.cc)
add_dependencies(randbat build_client_release)
target_include_directories(randbat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(randbat PRIVATE libpkmn_client)

add_executable(debug-log src/debug-log.cc)
add_dependencies(debug-log build_client_release)
target_include_directories(debug-log PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(debug-log PRIVATE libpkmn_client)

add_executable(mcts src/mcts.cc)
add_dependencies(mcts build_chance_release)
target_include_directories(mcts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mcts PRIVATE libpkmn_chance)

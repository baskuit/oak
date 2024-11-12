cmake_minimum_required(VERSION 3.8)
project(oak VERSION 0.0.2)

set(CMAKE_CXX_STANDARD 23)

function(create_library target dir)
  add_library(${target} STATIC IMPORTED)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/${dir}/libpkmn-showdown.a)
  target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/engine/zig-out/include)
endfunction()

create_library(libpkmn_client client)
create_library(libpkmn_chance chance)
create_library(libpkmn_fast fast)

add_subdirectory(extern/pinyon)

add_custom_target(build_client ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/client/libpkmn-showdown.a)
add_custom_target(build_chance ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/chance/libpkmn-showdown.a)
add_custom_target(build_fast ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/fast/libpkmn-showdown.a)

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

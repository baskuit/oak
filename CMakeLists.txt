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

target_compile_definitions(libpkmn_client INTERFACE LOG CHANCE CALC)
target_compile_definitions(libpkmn_chance INTERFACE CHANCE CALC)
# target_compile_definitions(libpkmn_fast INTERFACE)

add_custom_target(build_client_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/client/libpkmn-showdown.a)
add_custom_target(build_chance_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/chance/libpkmn-showdown.a)
add_custom_target(build_fast_debug ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/debug/fast/libpkmn-showdown.a)
add_custom_target(build_client_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/client/libpkmn-showdown.a)
add_custom_target(build_chance_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/chance/libpkmn-showdown.a)
add_custom_target(build_fast_release ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/release/fast/libpkmn-showdown.a)

add_executable(benchmark src/benchmark.cc)
add_dependencies(benchmark build_chance_release)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark PRIVATE libpkmn_chance)

add_executable(benchmark-eval src/benchmark-eval.cc)
add_dependencies(benchmark-eval build_chance_release)
target_include_directories(benchmark-eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark-eval PRIVATE libpkmn_chance)

add_library(lab_client lab/log.cc)
add_dependencies(lab_client build_client_release)
target_include_directories(lab_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lab_client PRIVATE libpkmn_client)

add_executable(lab lab/main.cc lab/sides.cc lab/games.cc)
add_dependencies(lab build_chance_release)
target_include_directories(lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lab)
target_link_libraries(lab PRIVATE libpkmn_chance readline lab_client)

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

add_executable(eval src/eval.cc)
add_dependencies(eval build_chance_release)
target_include_directories(eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(eval PRIVATE libpkmn_chance)

add_executable(nnue src/nnue.cc)
add_dependencies(nnue build_chance_release)
target_include_directories(nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nnue PRIVATE libpkmn_chance)
target_compile_options(nnue PRIVATE -mavx2)

add_executable(generate src/generate.cc)
add_dependencies(generate build_chance_release)
target_include_directories(generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(generate PRIVATE libpkmn_chance)
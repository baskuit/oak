#pragma once

#include <vector>

#include <filesystem>
#include <fstream>
#include <string>

#include "battle.h"

struct DebugLog {
  static constexpr auto header_size = 4 + PKMN_GEN1_BATTLE_SIZE;
  static constexpr auto frame_size = 64 + PKMN_GEN1_BATTLE_SIZE + 3;

  using Header = std::array<uint8_t, header_size>;
  using Frame = std::array<uint8_t, frame_size>;

  Header header;
  std::array<Frame, 1000> frames;
  int current_frame = 0;

  template <typename State> void set_header(const State &state) {
    header[0] = 1;
    header[1] = 1;
    header[2] = 64;
    header[3] = 0;
    memcpy(header.data() + 4, state.battle.bytes, PKMN_GEN1_BATTLE_SIZE);
  }

  template <typename State>
  void apply_actions(State &state, pkmn_choice p1_action,
                     pkmn_choice p2_action) {
    state.apply_actions(p1_action, p2_action);
    auto *frame_data = frames[current_frame++].data();
    memcpy(frame_data, state.options_data.log_buffer, 64);

    memcpy(frame_data + 64, state.battle.bytes, PKMN_GEN1_BATTLE_SIZE);
    frame_data[header_size] = state.result;
    frame_data[header_size + 1] = p1_action;
    frame_data[header_size + 2] = p2_action;
  }

  void save_data_to_path(std::string path) const {
    if (path.empty()) {
      const uint8_t *battle_prng_bytes = frames[0].data() + 376;
      const uint64_t *seed =
          reinterpret_cast<const uint64_t *>(battle_prng_bytes);
      const std::string cwd = std::filesystem::current_path();
      path = std::filesystem::current_path().string() + "/logs/" +
             std::to_string(*seed);
    }
    std::fstream file;
    file.open(path, std::ios::binary | std::ios::app);
    file.write(reinterpret_cast<const char *>(header.data()), header_size);
    for (int i = 0; i < current_frame; ++i) {
      file.write(reinterpret_cast<const char *>(frames[i].data()), frame_size);
    }
    file.close();
  }

  template <typename State, typename PRNG>
  void rollout_battle(State &&state, PRNG &device) {
    set_header(state);
    while (!state.is_terminal()) {
      state.get_actions();
      apply_actions(state, state.row_actions[device.random_int(state.rows())],
                    state.col_actions[device.random_int(state.cols())]);
    }
  }
};

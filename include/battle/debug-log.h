#pragma once

#include <vector>

#include <filesystem>
#include <fstream>
#include <string>

#include <pkmn.h>

template <size_t log_size = 64> struct DebugLog {
  static constexpr auto header_size = 4 + PKMN_GEN1_BATTLE_SIZE;
  static constexpr auto frame_size = log_size + PKMN_GEN1_BATTLE_SIZE + 3;

  using Header = std::array<uint8_t, header_size>;
  using Frame = std::array<uint8_t, frame_size>;

  Header header;
  std::vector<Frame> frames;

  void set_header(const pkmn_gen1_battle &battle) {
    header[0] = 1;
    header[1] = 1;
    header[2] = log_size % 256;
    header[3] = log_size / 256;
    memcpy(header.data() + 4, battle.bytes, PKMN_GEN1_BATTLE_SIZE);
  }

  void update_battle(pkmn_gen1_battle *battle,
                     pkmn_gen1_battle_options *options, pkmn_choice p1_choice,
                     pkmn_choice p2_choice) {

    frames.emplace_back();
    pkmn_gen1_log_options log_options{frames.back().data(), log_size};
    pkmn_gen1_battle_options_set(options, &log_options, nullptr, nullptr);

    auto result =
        pkmn_gen1_battle_update(battle, p1_choice, p2_choice, options);

    const auto *frame_data = frames.back().data();
    frame_data[header_size] = result;
    frame_data[header_size + 1] = p1_choice;
    frame_data[header_size + 2] = p2_choice;
  }

  void save_data_to_path(std::string path = "") const {
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
    for (const auto &frame : frames) {
      file.write(reinterpret_cast<const char *>(frame.data()), frame_size);
    }
    file.close();
  }
};

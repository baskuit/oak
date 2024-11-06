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

  pkmn_result update_battle(pkmn_gen1_battle *battle,
                            pkmn_gen1_battle_options *options, pkmn_choice c1,
                            pkmn_choice c2) {

    frames.emplace_back();
    auto *frame_data = frames.back().data();
    pkmn_gen1_log_options log_options{frame_data, log_size};
    pkmn_gen1_battle_options_set(options, &log_options, nullptr, nullptr);

    auto result = pkmn_gen1_battle_update(battle, c1, c2, options);
    frame_data += log_size;
    std::memcpy(frame_data, battle->bytes, PKMN_GEN1_BATTLE_SIZE);
    frame_data += PKMN_GEN1_BATTLE_SIZE;
    frame_data[0] = result;
    frame_data[1] = c1;
    frame_data[2] = c2;
    return result;
  }

  void save_data_to_path(std::string path = "") const {
    if (path.empty()) {
      const uint8_t *battle_prng_bytes = frames[0].data() + 376;
      const uint64_t *seed = std::bit_cast<const uint64_t *>(battle_prng_bytes);
      path = (std::filesystem::current_path() / "logs" / std::to_string(*seed))
                 .string();
    }
    std::fstream file;
    file.open(path, std::ios::binary | std::ios::app);
    file.write(std::bit_cast<const char *>(header.data()), header_size);
    for (const auto &frame : frames) {
      file.write(std::bit_cast<const char *>(frame.data()), frame_size);
    }
    file.close();
  }
};

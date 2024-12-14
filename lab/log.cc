#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pkmn.h>

#include <data/offsets.h>
#include <data/options.h>

#include "./log.h"

static_assert(Options::log && Options::chance && Options::calc);

void DebugLog::set_header(const pkmn_gen1_battle &battle) {
  header[0] = 1;
  header[1] = 1;
  header[2] = log_size % 256;
  header[3] = log_size / 256;
  std::memset(header.data() + 4, 0, 4);
  std::memcpy(header.data() + 8, battle.bytes, PKMN_GEN1_BATTLE_SIZE);
}

pkmn_result DebugLog::update(pkmn_gen1_battle &battle,
                                       const pkmn_choice c1,
                                       const pkmn_choice c2,
                                       pkmn_gen1_battle_options &options) {

  frames.emplace_back();
  auto *frame_data = frames.back().data();
  pkmn_gen1_log_options log_options{frame_data, log_size};
  pkmn_gen1_battle_options_set(&options, &log_options, nullptr, nullptr);
  const auto result = pkmn_gen1_battle_update(&battle, c1, c2, &options);

  frame_data += log_size;
  std::memcpy(frame_data, battle.bytes, PKMN_GEN1_BATTLE_SIZE);
  frame_data += PKMN_GEN1_BATTLE_SIZE;
  frame_data[0] = result;
  frame_data[1] = c1;
  frame_data[2] = c2;

  return result;
}

void DebugLog::save_data_to_path(std::string path) const {
  if (path.empty()) {
    const auto *battle_prng_bytes = frames.front().data() + Offsets::seed;
    const auto *seed = std::bit_cast<const uint64_t *>(battle_prng_bytes);
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

// template <size_t log_size>
// template <typename T>
// void DebugLog<log_size>::write(T& t) const {

// }

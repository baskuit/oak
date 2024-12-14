#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <pkmn.h>

struct DebugLog {
  static constexpr auto log_size = 256;
  static constexpr auto header_size = 8 + PKMN_GEN1_BATTLE_SIZE;
  static constexpr auto frame_size = log_size + PKMN_GEN1_BATTLE_SIZE + 3;

  using Header = std::array<uint8_t, header_size>;
  using Frame = std::array<uint8_t, frame_size>;

  Header header;
  std::vector<Frame> frames;

  void set_header(const pkmn_gen1_battle &battle);
  pkmn_result update(pkmn_gen1_battle &battle, const pkmn_choice c1,
                     const pkmn_choice c2, pkmn_gen1_battle_options &options);

  void save_data_to_path(std::string path = "") const;

//   template <typename T>
//   void write(T& t) const;
};

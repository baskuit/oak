#include <data/durations.h>
#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <util/print.h>

#include <iostream>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#pragma pack(push, 1)
struct Frame {
  std::array<uint8_t, Sizes::battle> battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  float eval;
  float score;
  uint32_t iter;

  Frame() = default;
  Frame(const pkmn_gen1_battle *const battle,
        const pkmn_gen1_chance_durations *const durations, pkmn_result result,
        float eval, auto iter)
      : result{result}, eval{eval}, iter{static_cast<uint32_t>(iter)} {
    std::memcpy(this->battle.data(), battle, Sizes::battle);
    std::memcpy(this->durations.bytes, durations->bytes, Sizes::durations);
  }
};
#pragma pack(pop)

static_assert(sizeof(Frame) == 405);

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Input: filename, number of frames" << std::endl;
        return -1;
    }

    std::string fn{argv[1]};
    std::cout << fn << std::endl;

    int fd = open(fn.data(), O_RDONLY);
    std::cout << "fd: " << fd << std::endl;
    if (fd == -1) {
      std::cerr << "Failed to open file" << std::endl;
      return -1;
    }
    size_t length = std::atoi(argv[2]);
    std::cout << "length: " << length <<std::endl;

    uint8_t* bytes = new uint8_t[length * sizeof(Frame)];
    const auto r = pread(fd, bytes, length * sizeof(Frame), 0);
    if (r == -1) {
      std::cerr << "pread Error." << std::endl;
      return -1;
    }

    for (auto i = 0; i < length; ++i) {
      const auto& frame= *reinterpret_cast<const Frame*>(bytes + (sizeof(Frame) * i));
      std::cout << i << ":\n";
      pkmn_gen1_battle battle;
      std::memcpy(battle.bytes, frame.battle.data(), Sizes::battle);
      std::cout << Strings::battle_to_string(battle) << '\n';
      std::cout << "eval: " << frame.eval << "; iter: " << frame.iter << "; score: " << frame.score << '\n';
    }
    return 0;
}

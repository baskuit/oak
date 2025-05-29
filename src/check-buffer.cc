#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>
#include <data/durations.h>
#include <util/print.h>

#include <iostream>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

void print_durations(const pkmn_gen1_chance_durations &dur) {
  for (auto p = 0; p < 2; ++p) {
    const auto &d = View::ref(dur).duration(p);
    std::cout << "Sleep: ";
    for (auto i = 0; i < 6; ++i) {
      std::cout << d.sleep(i) << ' ';
    }
    std::cout << '\n';
    std::cout << "Confusion: " << d.confusion() << ' ';
    std::cout << "Disable: " << d.disable() << ' ';
    std::cout << "Attacking: " << d.attacking() << ' ';
    std::cout << "Binding: " << d.binding() << '\n';
  }
}

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

  size_t start = 0;
  size_t end = 0;
  if (argc == 3) {
    start = std::atoi(argv[2]);
    end = start + 1;
  } else if (argc == 4) {
    start = std::atoi(argv[2]);
    end = std::atoi(argv[3]);
  } else {
    std::cerr << "Input: filename, then either {frame number} or {frame start, "
                 "frame end}."
              << std::endl;
    return -1;
  }

  std::string fn{argv[1]};
  int fd = open(fn.data(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "Failed to open file" << std::endl;
    return -1;
  }

  if (start > end) {
    std::cerr << "Invalid range: " << start << ' ' << end << std::endl;
    return -1;
  }

  const size_t length = end - start;
  uint8_t *bytes = new uint8_t[length * sizeof(Frame)];
  const auto r =
      pread(fd, bytes, length * sizeof(Frame), start * sizeof(Frame));
  if (r == -1) {
    std::cerr << "pread Error." << std::endl;
    return -1;
  }

  for (auto i = 0; i < length; ++i) {
    const auto &frame =
        *reinterpret_cast<const Frame *>(bytes + (sizeof(Frame) * i));
    std::cout << i + start << ":\n";
    pkmn_gen1_battle battle;
    std::memcpy(battle.bytes, frame.battle.data(), Sizes::battle);
    std::cout << Strings::battle_to_string(battle);
    print_durations(frame.durations);
    std::cout << "eval: " << frame.eval << "; iter: " << frame.iter
              << "; score: " << frame.score << '\n'
              << '\n';
  }
  return 0;
}

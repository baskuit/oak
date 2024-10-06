#pragma once

#include <pkmn.h>

namespace Chance {

enum class Observation { started, continuing, ended, overwritten };

enum class Confusion { started, continuing, ended, overwritten };

auto damage(const auto *bytes) {
  return bytes[0];
}
auto hit(const auto *bytes) {
  return bytes[1] & 3;
}
auto critical_hit(const auto *bytes) {
  return (bytes[1] >> 2) & 3;
}
auto secondary_chance(const auto *bytes) {
  return (bytes[1] >> 4) & 3;
}
auto speed_tie(const auto *bytes) {
  return bytes[1] >> 6;
}
auto confused(const auto *bytes) {
  return bytes[2] & 3;
}
auto paralyzed(const auto *bytes) {
  return (bytes[2] >> 2) & 3;
}
auto duration(const auto *bytes) {
  return bytes[2] >> 4;
}
namespace Duration {
auto sleep(const auto *bytes) {
  return bytes[3] & 7;
}
auto confusion(const auto *bytes) {
  return (bytes[3] >> 3) & 7;
}
auto disable(const auto *bytes) {
  return (bytes[3] >> 6) | ((bytes[4] & 3) << 2);
}
auto attacking(const auto *bytes) {
  return (bytes[4] >> 2) & 7;
}
auto binding(const auto *bytes) {
  return bytes[4] >> 5;
}
}; // namespace Duration
auto move_slot(const auto *bytes) {
  return bytes[5] & 15;
}
auto multi_hit(const auto *bytes) {
  return bytes[5] >> 4;
}
auto psywave(const auto *bytes) {
  return bytes[6];
}
auto metronome(const auto *bytes) {
  return bytes[7];
}

void display_side(const auto *bytes) {
  std::cout << "damage: " << static_cast<int>(damage(bytes)) << std::endl;
  std::cout << "hit: " << static_cast<int>(hit(bytes)) << std::endl;
  std::cout << "critical_hit: " << static_cast<int>(critical_hit(bytes))
            << std::endl;
  std::cout << "secondary_chance: "
            << static_cast<int>(secondary_chance(bytes)) << std::endl;
  std::cout << "speed_tie: " << static_cast<int>(speed_tie(bytes))
            << std::endl;
  std::cout << "confused: " << static_cast<int>(confused(bytes)) << std::endl;
  std::cout << "paralyzed: " << static_cast<int>(paralyzed(bytes))
            << std::endl;
  std::cout << "duration: " << static_cast<int>(duration(bytes)) << std::endl;
  std::cout << "Duration::sleep: " << static_cast<int>(Duration::sleep(bytes))
            << std::endl;
  std::cout << "Duration::confusion: "
            << static_cast<int>(Duration::confusion(bytes)) << std::endl;
  std::cout << "Duration::disable: "
            << static_cast<int>(Duration::disable(bytes)) << std::endl;
  std::cout << "Duration::attacking: "
            << static_cast<int>(Duration::attacking(bytes)) << std::endl;
  std::cout << "Duration::binding: "
            << static_cast<int>(Duration::binding(bytes)) << std::endl;
  std::cout << "move_slot: " << static_cast<int>(move_slot(bytes))
            << std::endl;
  std::cout << "multi_hit: " << static_cast<int>(multi_hit(bytes))
            << std::endl;
  std::cout << "psywave: " << static_cast<int>(psywave(bytes)) << std::endl;
  std::cout << "metronome: " << static_cast<int>(metronome(bytes))
            << std::endl;
}

void display(const pkmn_gen1_chance_actions* const actions) {
  std::cout << "P1" << std::endl;
  display_side(actions->bytes);
  std::cout << "P2" << std::endl;
  display_side(actions->bytes + 8);
}

}; // namespace Chance
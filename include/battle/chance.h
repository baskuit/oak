#pragma once

#include <pkmn.h>

namespace Chance {

enum class Observation { started, continuing, ended, overwritten };

enum class Confusion { started, continuing, ended, overwritten };

auto damage(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[0];
}
auto hit(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] & 3;
}
auto critical_hit(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[1] >> 2) & 3;
}
auto secondary_chance(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[1] >> 4) & 3;
}
auto speed_tie(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] >> 6;
}
auto confused(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[2] & 3;
}
auto paralyzed(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[2] >> 2) & 3;
}
auto duration(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[2] >> 4;
}
namespace Duration {
auto sleep(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[3] & 7;
}
auto confusion(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[3] >> 3) & 7;
}
auto disable(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[3] >> 6) | ((actions->bytes[4] & 3) << 2);
}
auto attacking(const pkmn_gen1_chance_actions *actions) {
  return (actions->bytes[4] >> 2) & 7;
}
auto binding(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[4] >> 5;
}
}; // namespace Duration
auto move_slot(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[5] & 15;
}
auto multi_hit(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[5] >> 4;
}
auto psywave(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[6];
}
auto metronome(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[7];
}

void display(const pkmn_gen1_chance_actions *actions) {
  std::cout << "damage: " << static_cast<int>(damage(actions)) << std::endl;
  std::cout << "hit: " << static_cast<int>(hit(actions)) << std::endl;
  std::cout << "critical_hit: " << static_cast<int>(critical_hit(actions))
            << std::endl;
  std::cout << "secondary_chance: "
            << static_cast<int>(secondary_chance(actions)) << std::endl;
  std::cout << "speed_tie: " << static_cast<int>(speed_tie(actions))
            << std::endl;
  std::cout << "confused: " << static_cast<int>(confused(actions)) << std::endl;
  std::cout << "paralyzed: " << static_cast<int>(paralyzed(actions))
            << std::endl;
  std::cout << "duration: " << static_cast<int>(duration(actions)) << std::endl;
  std::cout << "Duration::sleep: " << static_cast<int>(Duration::sleep(actions))
            << std::endl;
  std::cout << "Duration::confusion: "
            << static_cast<int>(Duration::confusion(actions)) << std::endl;
  std::cout << "Duration::disable: "
            << static_cast<int>(Duration::disable(actions)) << std::endl;
  std::cout << "Duration::attacking: "
            << static_cast<int>(Duration::attacking(actions)) << std::endl;
  std::cout << "Duration::binding: "
            << static_cast<int>(Duration::binding(actions)) << std::endl;
  std::cout << "move_slot: " << static_cast<int>(move_slot(actions))
            << std::endl;
  std::cout << "multi_hit: " << static_cast<int>(multi_hit(actions))
            << std::endl;
  std::cout << "psywave: " << static_cast<int>(psywave(actions)) << std::endl;
  std::cout << "metronome: " << static_cast<int>(metronome(actions))
            << std::endl;
}

}; // namespace Chance
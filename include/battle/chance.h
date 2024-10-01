#pragma once

#include <pkmn.h>

namespace ChanceHelpers {

enum class Observation { started, continuing, ended, overwritten };

enum class Confusion { started, continuing, ended, overwritten };

auto damage(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[0];
}
auto hit(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] & 3;
}
auto critical_hit(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] & 12;
}
auto secondary_chance(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] & 48;
}
auto speed_tie(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[1] & 192;
}
auto confused(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[2] & 3;
}
auto paralyzed(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[2] & 12;
}
auto duration(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[2] >> 4;
}

namespace Duration { // TODO
auto sleep(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[3] & 7;
}
auto confusion(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[3] & 56;
}

auto disable(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[3] & 48;
}
auto attacking(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[3] & 192;
}
auto binding(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[4] & 3;
}
}; // namespace Duration

auto move_slot(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[4] >> 4;
}
auto pp(const pkmn_gen1_chance_actions *actions) {
  return actions->bytes[5] & 31;
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

}; // namespace ChanceHelpers

void display(const pkmn_gen1_chance_actions *actions) {
  using namespace ChanceHelpers;
  std::cout << "damage: " << damage(actions) << std::endl;
  std::cout << "hit: " << hit(actions) << std::endl;
  std::cout << "critical_hit: " << critical_hit(actions) << std::endl;
  std::cout << "secondary_chance: " << secondary_chance(actions) << std::endl;
  std::cout << "speed_tie: " << speed_tie(actions) << std::endl;
  std::cout << "confused: " << confused(actions) << std::endl;
  std::cout << "paralyzed: " << paralyzed(actions) << std::endl;
  std::cout << "duration: " << duration(actions) << std::endl;
  std::cout << "move_slot: " << move_slot(actions) << std::endl;
  std::cout << "pp: " << pp(actions) << std::endl;
  std::cout << "multi_hit: " << multi_hit(actions) << std::endl;
  std::cout << "psywave: " << psywave(actions) << std::endl;
  std::cout << "metronome: " << metronome(actions) << std::endl;
}
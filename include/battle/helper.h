#pragma once

// Convenience functions for dev/debugging

#include <pkmn.h>

#include <vector>

namespace Helper {

auto get_choices(const pkmn_gen1_battle *const battle, pkmn_result result)
    -> std::pair<std::vector<pkmn_choice>, std::vector<pkmn_choice>> {
  std::vector<pkmn_choice> p1_choices;
  std::vector<pkmn_choice> p2_choices;
  p1_choices.resize(9);
  p2_choices.resize(9);
  auto m =
      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P1, pkmn_result_p1(result),
                               p1_choices.data(), PKMN_GEN1_MAX_CHOICES);
  auto n =
      pkmn_gen1_battle_choices(battle, PKMN_PLAYER_P2, pkmn_result_p2(result),
                               p2_choices.data(), PKMN_GEN1_MAX_CHOICES);
  p1_choices.resize(m);
  p2_choices.resize(n);
  return {p1_choices, p2_choices};
}

} // namespace Util
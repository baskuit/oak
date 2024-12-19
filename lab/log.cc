#include <data/options.h>

#include "./log.h"

static_assert(Options::log && Options::chance && Options::calc);

namespace Lab {
namespace Log {
pkmn_result update(pkmn_gen1_battle &battle, const pkmn_choice c1,
                   const pkmn_choice c2, pkmn_gen1_battle_options &options) {
  return pkmn_gen1_battle_update(&battle, c1, c2, &options);
}
void set(pkmn_gen1_battle_options &options, const pkmn_gen1_log_options *log,
         const pkmn_gen1_chance_options *chance,
         const pkmn_gen1_calc_options *calc) {
  pkmn_gen1_battle_options_set(&options, log, chance, calc);
}
} // namespace Log
} // namespace Lab
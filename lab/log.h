#pragma once

#include <pkmn.h>

namespace Lab {

namespace Log {

static constexpr auto log_size = 256;

pkmn_result update(pkmn_gen1_battle &battle, const pkmn_choice c1,
                   const pkmn_choice c2, pkmn_gen1_battle_options &options);
void set(pkmn_gen1_battle_options &options, const pkmn_gen1_log_options *log,
         const pkmn_gen1_chance_options *chance,
         const pkmn_gen1_calc_options *calc);

} // namespace Log

} // namespace Lab

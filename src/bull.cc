#include <pinyon.h>

#include <array>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

/*

Checklist:

* Output file name
* Move pool
* hp_1 upper bound for total solve

*/

std::ofstream OUTPUT_FILE{"fb-vs-standard.txt",
                          std::ios::out | std::ios::trunc};

std::ostringstream BUFFER;

const int MAX_HP = 353;

const int BURN_DMG = MAX_HP / 16;

const mpq_class CRIT{55, 256};
const mpq_class NO_CRIT{201, 256};

struct Roll {
  int dmg;
  int n;
};

struct Move {
  std::string id;
  // probabilities. I assume (1 - p) can't be optimized if we use libgmp, so I
  // double up
  mpq_class acc;
  mpq_class one_minus_acc;
  mpq_class eff;
  mpq_class one_minus_eff;

  // recharge gets one 0 dmg roll
  std::vector<Roll> rolls;
  std::vector<Roll> crit_rolls;
  std::vector<Roll> burned_rolls;
  bool must_recharge;
  bool may_freeze;
  bool may_burn;
  bool may_flinch;
};

const Move BODY_SLAM{
    "Body Slam",
    mpq_class{255, 256},
    mpq_class{1, 256},
    mpq_class{0, 1},
    mpq_class{1, 1},
    {{95, 2},
     {96, 2},
     {97, 3},
     {98, 2},
     {99, 2},
     {100, 2},
     {101, 3},
     {102, 2},
     {103, 2},
     {104, 3},
     {105, 2},
     {106, 2},
     {107, 2},
     {108, 3},
     {109, 2},
     {110, 2},
     {111, 2},
     {112, 1}},
    {{184, 1}, {185, 1}, {186, 1}, {187, 1}, {188, 2}, {189, 1}, {190, 1},
     {191, 1}, {192, 1}, {193, 1}, {194, 2}, {195, 1}, {196, 1}, {197, 1},
     {198, 1}, {199, 2}, {200, 1}, {201, 1}, {202, 1}, {203, 1}, {204, 1},
     {205, 2}, {206, 1}, {207, 1}, {208, 1}, {209, 1}, {210, 1}, {211, 2},
     {212, 1}, {213, 1}, {214, 1}, {215, 1}, {216, 1}, {217, 1}},
    {{48, 3},
     {49, 4},
     {50, 5},
     {51, 4},
     {52, 5},
     {53, 4},
     {54, 5},
     {55, 4},
     {56, 4},
     {57, 1}},
    false,
    false,
    false,
    false};

const Move HYPER_BEAM{
    "Hyper Beam",
    mpq_class{229, 256},
    mpq_class{27, 256},
    mpq_class{0, 1},
    mpq_class{1, 1},
    {{166, 1}, {167, 1}, {168, 1}, {169, 2}, {170, 1}, {171, 1}, {172, 2},
     {173, 1}, {174, 1}, {175, 1}, {176, 2}, {177, 1}, {178, 1}, {179, 2},
     {180, 1}, {181, 1}, {182, 2}, {183, 1}, {184, 1}, {185, 1}, {186, 2},
     {187, 1}, {188, 1}, {189, 2}, {190, 1}, {191, 1}, {192, 2}, {193, 1},
     {194, 1}, {195, 1}, {196, 1}},
    {{324, 1}, {325, 1}, {327, 1}, {328, 1}, {330, 1}, {331, 1}, {333, 1},
     {334, 1}, {336, 1}, {337, 1}, {339, 1}, {340, 1}, {342, 1}, {343, 1},
     {345, 1}, {346, 1}, {348, 1}, {349, 1}, {351, 1}, {352, 1}, {354, 1},
     {355, 1}, {357, 1}, {358, 1}, {360, 1}, {361, 1}, {363, 1}, {364, 1},
     {366, 1}, {367, 1}, {369, 1}, {370, 1}, {372, 1}, {373, 1}, {375, 1},
     {376, 1}, {378, 1}, {379, 1}, {381, 1}},
    {{84, 2},
     {85, 3},
     {86, 3},
     {87, 2},
     {88, 3},
     {89, 2},
     {90, 3},
     {91, 2},
     {92, 3},
     {93, 3},
     {94, 2},
     {95, 3},
     {96, 2},
     {97, 3},
     {98, 2},
     {99, 1}},
    true,
    false,
    false,
    false};

const Move BLIZZARD{"Blizzard",
                    mpq_class{229, 256},
                    mpq_class{27, 256},
                    mpq_class{27, 256},
                    mpq_class{229, 256},
                    {{86, 1},
                     {87, 2},
                     {88, 3},
                     {89, 2},
                     {90, 3},
                     {91, 2},
                     {92, 3},
                     {93, 2},
                     {94, 3},
                     {95, 2},
                     {96, 3},
                     {97, 2},
                     {98, 3},
                     {99, 2},
                     {100, 3},
                     {101, 2},
                     {102, 1}},
                    {{168, 1}, {169, 1}, {170, 2}, {171, 1}, {172, 1}, {173, 2},
                     {174, 1}, {175, 1}, {176, 1}, {177, 2}, {178, 1}, {179, 1},
                     {180, 2}, {181, 1}, {182, 1}, {183, 1}, {184, 2}, {185, 1},
                     {186, 1}, {187, 2}, {188, 1}, {189, 1}, {190, 1}, {191, 2},
                     {192, 1}, {193, 1}, {194, 2}, {195, 1}, {196, 1}, {197, 1},
                     {198, 1}},
                    {{86, 1},
                     {87, 2},
                     {88, 3},
                     {89, 2},
                     {90, 3},
                     {91, 2},
                     {92, 3},
                     {93, 2},
                     {94, 3},
                     {95, 2},
                     {96, 3},
                     {97, 2},
                     {98, 3},
                     {99, 2},
                     {100, 3},
                     {101, 2},
                     {102, 1}},
                    false,
                    true,
                    false,
                    false};

const Move EARTHQUAKE{
    "Earthquake",
    mpq_class{255, 256},
    mpq_class{1, 256},
    mpq_class{0, 1},
    mpq_class{1, 1},
    {{74, 1},
     {75, 3},
     {76, 3},
     {77, 3},
     {78, 2},
     {79, 3},
     {80, 3},
     {81, 3},
     {82, 3},
     {83, 3},
     {84, 3},
     {85, 3},
     {86, 3},
     {87, 2},
     {88, 1}},
    {{144, 1}, {145, 1}, {146, 2}, {147, 1}, {148, 2}, {149, 1}, {150, 2},
     {151, 1}, {152, 2}, {153, 1}, {154, 2}, {155, 1}, {156, 2}, {157, 1},
     {158, 2}, {159, 1}, {160, 2}, {161, 1}, {162, 2}, {163, 1}, {164, 2},
     {165, 1}, {166, 2}, {167, 1}, {168, 2}, {169, 1}, {170, 1}},
    {{38, 4}, {39, 6}, {40, 6}, {41, 5}, {42, 6}, {43, 6}, {44, 5}, {45, 1}},
    false,
    false,
    false,
    false};

const Move FIRE_BLAST{
    "Fire Blast",
    mpq_class{27, 32},
    mpq_class{5, 32},
    mpq_class{19, 64},
    mpq_class{45, 64},
    {{86, 1},
     {87, 2},
     {88, 3},
     {89, 2},
     {90, 3},
     {91, 2},
     {92, 3},
     {93, 2},
     {94, 3},
     {95, 2},
     {96, 3},
     {97, 2},
     {98, 3},
     {99, 2},
     {100, 3},
     {101, 2},
     {102, 1}},
    {{168, 1}, {169, 1}, {170, 2}, {171, 1}, {172, 1}, {173, 2}, {174, 1},
     {175, 1}, {176, 1}, {177, 2}, {178, 1}, {179, 1}, {180, 2}, {181, 1},
     {182, 1}, {183, 1}, {184, 2}, {185, 1}, {186, 1}, {187, 2}, {188, 1},
     {189, 1}, {190, 1}, {191, 2}, {192, 1}, {193, 1}, {194, 2}, {195, 1},
     {196, 1}, {197, 1}, {198, 1}},
    {{86, 1},
     {87, 2},
     {88, 3},
     {89, 2},
     {90, 3},
     {91, 2},
     {92, 3},
     {93, 2},
     {94, 3},
     {95, 2},
     {96, 3},
     {97, 2},
     {98, 3},
     {99, 2},
     {100, 3},
     {101, 2},
     {102, 1}},
    false,
    false,
    true,
    false};

const Move STOMP{
    "Stomp",
    mpq_class{255, 256},
    mpq_class{1, 256},
    mpq_class{0, 1},
    mpq_class{1, 1},
    {{74, 3},
     {75, 3},
     {76, 3},
     {77, 3},
     {78, 3},
     {79, 3},
     {80, 3},
     {81, 3},
     {82, 3},
     {83, 3},
     {84, 3},
     {85, 3},
     {86, 2},
     {87, 1}},
    {{141, 2}, {142, 1}, {143, 2}, {144, 1}, {145, 2}, {146, 1}, {147, 2},
     {148, 1}, {149, 2}, {150, 1}, {151, 2}, {152, 2}, {153, 1}, {154, 2},
     {155, 1}, {156, 2}, {157, 1}, {158, 2}, {159, 1}, {160, 2}, {161, 1},
     {162, 2}, {163, 1}, {164, 2}, {165, 1}, {166, 1}},
    {{38, 4}, {39, 6}, {40, 6}, {41, 5}, {42, 6}, {43, 6}, {44, 5}, {45, 1}},
    false,
    false,
    false,
    true};

const Move RECHARGE{"Recharge",      mpq_class{0, 1}, mpq_class{1, 1},
                    mpq_class{0, 1}, mpq_class{1, 1}, {{0, 39}},
                    {{0, 39}},       {{0, 39}},       false,
                    false,           false,           false};

const std::vector<const Move *> ALL_MOVES{&BODY_SLAM,  &HYPER_BEAM, &BLIZZARD,
                                          &EARTHQUAKE, &FIRE_BLAST, &STOMP,
                                          &RECHARGE};

// Currently recharge MUST be the last move
// but otherwise you can do whatever, really
const std::vector<const Move *> P1_MOVES{&BODY_SLAM, &HYPER_BEAM, &BLIZZARD,
                                         &FIRE_BLAST, &RECHARGE};

const std::vector<const Move *> P2_MOVES{&BODY_SLAM, &HYPER_BEAM, &BLIZZARD,
                                         // &EARTHQUAKE,
                                         &RECHARGE};

struct State {
  int hp_1;
  int hp_2;
  int freeze_clause_1;
  int freeze_clause_2;
  int burned_1;
  int burned_2;
  int recharge_1;
  int recharge_2;

  State() {}

  State(const int hp_1, const int hp_2, const int freeze_clause_1,
        const int freeze_clause_2, const int burned_1, const int burned_2,
        const int recharge_1, const int recharge_2)
      : hp_1{hp_1}, hp_2{hp_2}, freeze_clause_1{freeze_clause_1},
        freeze_clause_2{freeze_clause_2}, burned_1{burned_1},
        burned_2{burned_2}, recharge_1{recharge_1}, recharge_2{recharge_2} {}

  bool operator==(const State &other) const {
    return (hp_1 == other.hp_1) && (hp_2 == other.hp_2) &&
           (burned_1 == other.burned_1) && (burned_2 == other.burned_2) &&
           (recharge_1 == other.recharge_1) && (recharge_2 == other.recharge_2);
  }
};

void print_state(const State &state) {
  std::cout << state.hp_1 << ' ' << state.hp_2 << ' ' << state.burned_1 << ' '
            << state.burned_2 << ' ' << state.recharge_1 << ' '
            << state.recharge_2 << std::endl;
}

struct SolutionEntry {
  mpq_class value;
  float p1_strategy[4];
  float p2_strategy[4];
  float payoff_matrix[4][4];
};

struct Solution { // hp, freeze, burn, recharge
  SolutionEntry data[MAX_HP][MAX_HP][2][2][2][2][3];
};

SolutionEntry &get_entry(Solution &tables, const State &state) {
  int x;
  if (state.hp_1 < state.hp_2) {
    x = (state.recharge_1 << 1) + (state.recharge_2 << 0);
    return tables
        .data[state.hp_2 - 1][state.hp_1 - 1][state.freeze_clause_2]
             [state.freeze_clause_1][state.burned_2][state.burned_1][x % 3];
  } else {
    int x = (state.recharge_1 << 0) + (state.recharge_2 << 1);
    return tables
        .data[state.hp_1 - 1][state.hp_2 - 1][state.freeze_clause_1]
             [state.freeze_clause_2][state.burned_1][state.burned_2][x % 3];
  }
}

const SolutionEntry &get_entry(const Solution &tables, const State &state) {
  int x;
  if (state.hp_1 < state.hp_2) {
    x = (state.recharge_1 << 1) + (state.recharge_2 << 0);
    return tables
        .data[state.hp_2 - 1][state.hp_1 - 1][state.freeze_clause_2]
             [state.freeze_clause_1][state.burned_2][state.burned_1][x % 3];
  } else {
    int x = (state.recharge_1 << 0) + (state.recharge_2 << 1);
    return tables
        .data[state.hp_1 - 1][state.hp_2 - 1][state.freeze_clause_1]
             [state.freeze_clause_2][state.burned_1][state.burned_2][x % 3];
  }
}

mpq_class lookup_value(const Solution &tables, const State &state) {
  const SolutionEntry &entry = get_entry(tables, state);

  if (state.hp_1 < state.hp_2) {
    mpq_class answer = mpq_class{1} - entry.value;
    answer.canonicalize();
    return answer;
  } else {
    return entry.value;
  }
}

template <bool debug = false>
mpq_class q_value(const Solution &tables, const State &state,
                  const int move_1_idx, const int move_2_idx) {
  mpq_class value{0};
  mpq_class reflexive_prob{0};
  mpq_class total_prob{0};
  mpq_class total_prob_no_rolls{0};

  static mpq_class p2_frz_loss;
  static mpq_class p2_ko_loss;
  static mpq_class p1_brn_loss;
  static mpq_class p1_frz_loss;
  static mpq_class p1_ko_loss;
  static mpq_class p2_brn_loss;
  static mpq_class non_terminal;

  if constexpr (debug) {
    mpq_class p2_frz_loss = 0;
    mpq_class p2_ko_loss = 0;
    mpq_class p1_brn_loss = 0;
    mpq_class p1_frz_loss = 0;
    mpq_class p1_ko_loss = 0;
    mpq_class p2_brn_loss = 0;
    mpq_class non_terminal = 0;
  }

  for (int i = 0; i < 128; ++i) {
    // These booleans are all ordered w.r.t. turn. hit_1 is whether the first
    // attack of the turn goes off, hit_2 the second finally flipped is weather
    // the first attack is from Player 1 vs Player 2. P1/P2 is rougly speaking,
    // the order of args/params
    const bool hit_1 = i & 1;
    const bool hit_2 = i & 2;
    const bool crit_1 = i & 4;
    const bool crit_2 = i & 8;
    const bool proc_1 = i & 16;
    const bool proc_2 = i & 32;
    const bool flipped = i & 64;

    // also in turn order
    const int t1_hp = flipped ? state.hp_2 : state.hp_1;
    const int t2_hp = flipped ? state.hp_1 : state.hp_2;
    const int t1_already_burned = flipped ? state.burned_2 : state.burned_1;
    const int t2_already_burned = flipped ? state.burned_1 : state.burned_2;
    const int t1_freeze_clause =
        flipped ? state.freeze_clause_2 : state.freeze_clause_1;
    const int t2_freeze_clause =
        flipped ? state.freeze_clause_1 : state.freeze_clause_2;

    const Move &m1 = flipped ? *P2_MOVES[move_2_idx] : *P1_MOVES[move_1_idx];
    const Move &m2 = flipped ? *P1_MOVES[move_1_idx] : *P2_MOVES[move_2_idx];

    const bool flinch_t1 = hit_1 && proc_1 && m1.may_flinch;
    const bool frz_t1 = hit_1 && proc_1 && m1.may_freeze && !t2_freeze_clause;
    const bool frz_t2 =
        hit_2 && proc_2 && m2.may_freeze && !t1_freeze_clause && !flinch_t1;
    const bool brn_t1 = hit_1 && proc_1 && m1.may_burn;
    const bool brn_t2 = hit_2 && proc_2 && m2.may_burn && !flinch_t1;

    mpq_class t1_prob_no_roll =
        mpq_class{1, 2} * (hit_1 ? m1.acc : m1.one_minus_acc) *
        (crit_1 ? CRIT : NO_CRIT) * (proc_1 ? m1.eff : m1.one_minus_eff) *
        (hit_2 ? m2.acc : m2.one_minus_acc) * (crit_2 ? CRIT : NO_CRIT) *
        (proc_2 ? m2.eff : m2.one_minus_eff);
    t1_prob_no_roll.canonicalize();

    total_prob_no_rolls += t1_prob_no_roll;
    total_prob_no_rolls.canonicalize();

    if (frz_t1) {
      if (!flipped) {
        value += t1_prob_no_roll;
        value.canonicalize();

        if constexpr (debug) {
          p2_frz_loss += t1_prob_no_roll;
          p2_frz_loss.canonicalize();
        }
      } else {
        if constexpr (debug) {
          p1_frz_loss += t1_prob_no_roll;
          p1_frz_loss.canonicalize();
        }
      }
      total_prob += t1_prob_no_roll;
      total_prob.canonicalize();
      continue;
    }

    const std::vector<Roll> &rolls_t1 =
        hit_1 ? (crit_1 ? m1.crit_rolls
                        : (t1_already_burned ? m1.burned_rolls : m1.rolls))
              : RECHARGE.rolls;
    const std::vector<Roll> &rolls_t2 =
        (hit_2 && !flinch_t1)
            ? (crit_2 ? m2.crit_rolls
                      : ((t2_already_burned || brn_t1) ? m2.burned_rolls
                                                       : m2.rolls))
            : RECHARGE.rolls;

    for (const Roll roll_1 : rolls_t1) {
      mpq_class t1_roll_prob = t1_prob_no_roll * mpq_class{roll_1.n, 39};
      t1_roll_prob.canonicalize();

      const int t1_dmg_dealt = (hit_1 ? roll_1.dmg : 0);
      if (t2_hp <= t1_dmg_dealt) {

        if (!flipped) {
          value += t1_roll_prob;
          value.canonicalize();

          if constexpr (debug) {
            p2_ko_loss += t1_roll_prob;
            p2_ko_loss.canonicalize();
          }
        } else {
          if constexpr (debug) {
            p1_ko_loss += t1_roll_prob;
            p1_ko_loss.canonicalize();
          }
        }
        total_prob += t1_roll_prob;
        total_prob.canonicalize();
        continue;
      }

      // brn damage
      const int t1_brn_taken = (t1_already_burned ? BURN_DMG : 0);

      if (t1_hp <= t1_brn_taken) {

        if (!flipped) {
          if constexpr (debug) {
            p1_brn_loss += t1_roll_prob;
            p1_brn_loss.canonicalize();
          }
        } else {
          value += t1_roll_prob;
          value.canonicalize();

          if constexpr (debug) {
            p2_brn_loss += t1_roll_prob;
            p2_brn_loss.canonicalize();
          }
        }
        total_prob += t1_roll_prob;
        total_prob.canonicalize();
        continue;
      }

      if (frz_t2) {
        if (!flipped) {
          if constexpr (debug) {
            p1_frz_loss += t1_roll_prob;
            p1_frz_loss.canonicalize();
          }
        } else {
          value += t1_roll_prob;
          value.canonicalize();

          if constexpr (debug) {
            p2_frz_loss += t1_roll_prob;
            p2_frz_loss.canonicalize();
          }
        }
        total_prob += t1_roll_prob;
        total_prob.canonicalize();
        continue;
      }

      for (const Roll roll_2 : rolls_t2) {
        mpq_class t2_roll_prob = t1_roll_prob * mpq_class{roll_2.n, 39};
        t2_roll_prob.canonicalize();

        total_prob += t2_roll_prob;
        total_prob.canonicalize();

        const int t2_dmg_dealt = (hit_2 ? roll_2.dmg : 0);
        if (t1_hp <= (t1_brn_taken + t2_dmg_dealt)) {

          if (!flipped) {
            if constexpr (debug) {
              p1_ko_loss += t2_roll_prob;
              p1_ko_loss.canonicalize();
            }
          } else {
            value += t2_roll_prob;
            value.canonicalize();

            if constexpr (debug) {
              p2_ko_loss += t2_roll_prob;
              p2_ko_loss.canonicalize();
            }
          }
          continue;
        }

        // brn damage
        const int t2_brn_taken = ((t2_already_burned || brn_t1) ? BURN_DMG : 0);
        if (t2_hp <= (t2_brn_taken + t1_dmg_dealt)) {

          if (!flipped) {
            value += t2_roll_prob;
            value.canonicalize();

            if constexpr (debug) {
              p2_brn_loss += t2_roll_prob;
              p2_brn_loss.canonicalize();
            }
          } else {
            if constexpr (debug) {
              p1_brn_loss += t2_roll_prob;
              p1_brn_loss.canonicalize();
            }
          }
          value.canonicalize();
          continue;
        }

        // No KOs or freeze

        State child;
        if (!flipped) {
          child = State{t1_hp - (t2_dmg_dealt + t1_brn_taken),
                        t2_hp - (t1_dmg_dealt + t2_brn_taken),
                        state.freeze_clause_1,
                        state.freeze_clause_2,
                        t1_already_burned || brn_t2,
                        t2_already_burned || brn_t1,
                        hit_1 && m1.must_recharge,
                        hit_2 && m2.must_recharge};
        } else {
          child = State{t2_hp - (t1_dmg_dealt + t2_brn_taken),
                        t1_hp - (t2_dmg_dealt + t1_brn_taken),
                        state.freeze_clause_1,
                        state.freeze_clause_2,
                        t2_already_burned || brn_t1,
                        t1_already_burned || brn_t2,
                        hit_2 && m2.must_recharge,
                        hit_1 && m1.must_recharge};
        }

        if constexpr (debug) {
          non_terminal += t2_roll_prob;
          non_terminal.canonicalize();
        }

        if (child != state) {
          if constexpr (!debug) {
            mpq_class lv = lookup_value(tables, child);
            // faily good at catching bad lookups
            // I think in uninitialized mpq_class has 0 value by default
            // but sometimes, e.g. (1 1 1 0 1 0)
            // the value IS 0
            // if (lv == mpq_class{0})
            // {
            //     std::cout << '!' << std::endl;
            //     print_state(state);
            //     print_state(child);
            //     // std::cout << move_1.id << ' ' << move_2.id << std::endl;
            //     assert(false);
            //     exit(1);
            // }
            const mpq_class weighted_solved_value = t2_roll_prob * lv;
            value += weighted_solved_value;
            value.canonicalize();
          }
        } else {
          reflexive_prob += t2_roll_prob;
          reflexive_prob.canonicalize();
        }
      }
    }
  }

  if (total_prob != mpq_class(1)) {
    std::cout << total_prob.get_str() << std::endl;
    // std::cout << move_1.id << ' ' << move_2.id << std::endl;
    assert(false);
    exit(1);
  }

  if (reflexive_prob > mpq_class{0}) {
    if (state.recharge_1 || state.recharge_2 || state.burned_1 ||
        state.burned_2) {
      std::cout << "reflexive assert fail" << std::endl;
      std::cout << reflexive_prob.get_str() << std::endl;
      // std::cout << move_1.id << ' ' << move_2.id << std::endl;
      assert(false);
      exit(1);
    }
    // only S00 should have this
  }
  mpq_class real_value = value / (mpq_class{1} - reflexive_prob);
  real_value.canonicalize();

  // Drop this as well for unequal movesets
  // if ((state.hp_1 == state.hp_2) && (state.burned_1 == state.burned_2))
  // {
  //     if (P1_MOVES[move_1_idx] == P2_MOVES[move_2_idx])
  //     {
  //         if (real_value != mpq_class{1, 2})
  //         {
  //             std::cout << "mirror q fail" << std::endl;
  //             std::cout << real_value.get_str() << std::endl;
  //             assert(false);
  //             exit(1);
  //         }
  //     }
  // }

  if constexpr (debug) {
    std::cout << "p2 FRZ LOSS: " << p2_frz_loss.get_d() << std::endl;
    std::cout << "p2 KO LOSS: " << p2_ko_loss.get_d() << std::endl;
    std::cout << "p1 BRN LOSS: " << p1_brn_loss.get_d() << std::endl;
    std::cout << "p1 FRZ LOSS: " << p1_frz_loss.get_d() << std::endl;
    std::cout << "p1 KO LOSS: " << p1_ko_loss.get_d() << std::endl;
    std::cout << "p2 BRN LOSS: " << p2_brn_loss.get_d() << std::endl;
    std::cout << "non terminal " << non_terminal.get_d() << std::endl;
    std::cout << std::endl;
    std::cout << "p2 FRZ LOSS: " << p2_frz_loss.get_str() << std::endl;
    std::cout << "p2 KO LOSS: " << p2_ko_loss.get_str() << std::endl;
    std::cout << "p1 BRN LOSS: " << p1_brn_loss.get_str() << std::endl;
    std::cout << "p1 FRZ LOSS: " << p1_frz_loss.get_str() << std::endl;
    std::cout << "p1 KO LOSS: " << p1_ko_loss.get_str() << std::endl;
    std::cout << "p2 BRN LOSS: " << p2_brn_loss.get_str() << std::endl;
    std::cout << "non terminal " << non_terminal.get_str() << std::endl;

    mpq_class total_prob_2 = p2_frz_loss + p2_ko_loss + p1_brn_loss +
                             p1_frz_loss + p1_ko_loss + p2_brn_loss +
                             non_terminal;
    total_prob_2.canonicalize();
    std::cout << "TOTAL " << total_prob_2.get_str() << std::endl;
  }

  return real_value;
}
void solve_state(Solution &tables, const State &state) {
  // pinyon ftw!!!
  using Types =
      DefaultTypes<mpq_class, int, int, mpq_class, ConstantSum<1, 1>::Value>;

  // This is why RECHARGE must always be the last move in the moveset
  std::vector<int> p1_legal_no_recharge = {};
  for (int i = 0; i < P1_MOVES.size() - 1; ++i) {
    p1_legal_no_recharge.push_back(i);
  }
  std::vector<int> p1_legal_recharge = {static_cast<int>(P1_MOVES.size()) - 1};

  std::vector<int> p2_legal_no_recharge = {};
  for (int i = 0; i < P2_MOVES.size() - 1; ++i) {
    p2_legal_no_recharge.push_back(i);
  }
  std::vector<int> p2_legal_recharge = {static_cast<int>(P2_MOVES.size()) - 1};

  // get legal moves
  std::vector<int> p1_legal_moves =
      (state.recharge_1 > 0) ? p1_legal_recharge : p1_legal_no_recharge;
  std::vector<int> p2_legal_moves =
      (state.recharge_2 > 0) ? p2_legal_recharge : p2_legal_no_recharge;
  const size_t rows = p1_legal_moves.size();
  const size_t cols = p2_legal_moves.size();

  // get entry in Solution
  SolutionEntry &entry = get_entry(tables, state);

  // fill payoff matrix and add to entry
  Types::MatrixValue payoff_matrix{rows, cols};

  for (int row_idx = 0; row_idx < rows; ++row_idx) {
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
      payoff_matrix.get(row_idx, col_idx) = Types::Value{q_value<false>(
          tables, state, p1_legal_moves[row_idx], p2_legal_moves[col_idx])};
      entry.payoff_matrix[row_idx][col_idx] =
          payoff_matrix.get(row_idx, col_idx).get_row_value().get_d();
    }
  }

  // solve

  Types::VectorReal row_strategy{rows};
  Types::VectorReal col_strategy{cols};

  auto value = LRSNash::solve(payoff_matrix, row_strategy, col_strategy);
  entry.value = value.get_row_value();

  for (int row_idx = 0; row_idx < rows; ++row_idx) {
    entry.p1_strategy[row_idx] = row_strategy[row_idx].get_d();
  }

  for (int col_idx = 0; col_idx < cols; ++col_idx) {
    entry.p2_strategy[col_idx] = col_strategy[col_idx].get_d();
  }
}

void total_solve(Solution &tables) {

  for (int hp_1 = 1; hp_1 <= MAX_HP; ++hp_1) {
    for (int hp_2 = 1; hp_2 <= hp_1; ++hp_2) {
      std::cout << "HP: " << hp_1 << ' ' << hp_2 << std::endl;

      for (int freeze_clause_1 = 0; freeze_clause_1 < 2; ++freeze_clause_1) {
        for (int freeze_clause_2 = 0; freeze_clause_2 < 2; ++freeze_clause_2) {
          for (int burned_1 = 0; burned_1 < 1; ++burned_1) {
            for (int burned_2 = 0; burned_2 < 2; ++burned_2) {
              // Solve

              const State state_00{
                  hp_1,     hp_2,     freeze_clause_1, freeze_clause_2,
                  burned_1, burned_2, false,           false};
              const State state_01{hp_1,
                                   hp_2,
                                   freeze_clause_1,
                                   freeze_clause_2,
                                   burned_1,
                                   burned_2,
                                   true,
                                   false};
              const State state_10{
                  hp_1,     hp_2,     freeze_clause_1, freeze_clause_2,
                  burned_1, burned_2, false,           true};

              solve_state(tables, state_00);
              solve_state(tables, state_01);
              solve_state(tables, state_10);

              SolutionEntry *entries =
                  tables.data[hp_1 - 1][hp_2 - 1][freeze_clause_1]
                             [freeze_clause_2][burned_1][burned_2];

              // assert

              // Not relevent if the movesets are not the same

              // if ((hp_1 == hp_2) && (burned_1 == 0) && (burned_2 == 0))
              // {
              //     if ((entries[0].value != mpq_class{1, 2}))
              //     {
              //         // std::cout << "s00 not 1/2 for same hp" << std::endl;
              //         // assert(false);
              //         // exit(1);
              //     }
              //     if (entries[1].value + entries[2].value != mpq_class{1})
              //     {
              //         std::cout << "s01 doesnt mirror s10" << std::endl;
              //         assert(false);
              //         exit(1);
              //     }
              // }

              // file output

              for (int r = 0; r < 3; ++r) {
                const int recharge_1 = r % 2;
                const int recharge_2 = r / 2;

                BUFFER << "HP: " << hp_1 << ' ' << hp_2
                       << " FRZ CLAUSE: " << freeze_clause_1 << ' '
                       << freeze_clause_2 << " BRN: " << burned_1 << ' '
                       << burned_2 << " RECHARGE: " << recharge_1 << ' '
                       << recharge_2 << std::endl;
                BUFFER << "VALUE: " << entries[r].value.get_d() << " = "
                       << entries[r].value.get_str() << std::endl;
                BUFFER << "PAYOFF MATRIX:" << std::endl;

                // always adds the full 4x4, hopefully the initialized mpq_class
                // values are just 0...
                for (int row_idx = 0; row_idx < 4; ++row_idx) {
                  for (int col_idx = 0; col_idx < 4; ++col_idx) {
                    BUFFER << entries[r].payoff_matrix[row_idx][col_idx] << ' ';
                  }
                  BUFFER << std::endl;
                }

                BUFFER << "STRATEGIES:" << std::endl;
                if (recharge_1 == 0) {
                  BUFFER << "P1: ";
                  for (int i = 0; i < P1_MOVES.size() - 1; ++i) {
                    BUFFER << P1_MOVES[i]->id << " : "
                           << entries[r].p1_strategy[i] << ", ";
                  }
                  BUFFER << std::endl;
                }
                if (recharge_2 == 0) {
                  BUFFER << "P2: ";
                  for (int i = 0; i < P2_MOVES.size() - 1; ++i) {
                    BUFFER << P2_MOVES[i]->id << " : "
                           << entries[r].p2_strategy[i] << ", ";
                  }
                  BUFFER << std::endl;
                }
              }

              // end hp, burn loop
            }
          }
        }
      }
    }
  }
}

void move_rolls_assert() {
  for (const Move *move : ALL_MOVES) {
    int a = 0;
    int b = 0;
    int c = 0;
    for (const auto roll : move->rolls) {
      a += roll.n;
    }
    for (const auto roll : move->crit_rolls) {
      b += roll.n;
    }
    for (const auto roll : move->burned_rolls) {
      c += roll.n;
    }
    if ((a != 39) || (b != 39) || (c != 39)) {
      assert(false);
      exit(1);
    }
  }
}

int main() {
  move_rolls_assert();

  Solution *tables_ptr = new Solution();
  Solution &tables = *tables_ptr;

  const size_t table_size_bytes = sizeof(tables);
  std::cout << "SOLUTION TABLE SIZE (MB): " << (table_size_bytes >> 20)
            << std::endl
            << std::endl;

  total_solve(tables);

  OUTPUT_FILE << BUFFER.str();
  OUTPUT_FILE.close();

  return 0;
}
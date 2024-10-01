#include <array>

#include "util.h"
#include <bit>

#include <pkmn.h>

namespace Bucketing {

// round hp up to the nearest eighth
enum class HP {
  KO,
  ONE,
  TWO,
  THREE,
  FOUR,
  FIVE,
  SIX,
  SEVEN,
};

// number of observed turns spent asleep. REST_2 is identical to SLP_6
enum class Status {
  None,
  PAR,
  BRN,
  FRZ,
  PSN,
  SLP_0,
  SLP_1,
  SLP_2,
  SLP_3,
  SLP_4,
  SLP_5,
  SLP_6,
  REST_0,
  REST_1, // 14
};

struct Bench {
  HP hp;
  Status status;

  Bench() : hp{SEVEN}, Status{None} {}
  Bench(const uin8_t const *bytes) {
  }

  constexpr void get_from_bytes(const uint8_t const *bytes) {}
};

struct Active {

  Active(const uin8_t const *bytes) {
  }

  constexpr void reset () noexcept {
  }

  constexpr void get_from_bytes(const uint8_t const *bytes) {}
};

struct Side {
  Active active;
  std::array<Bench, 6> bench;

  Side(const uin8_t const *bytes) {
    this->get_from_bytes(bytes);
  }

  constexpr void get_from_bytes(const uint8_t const *bytes) {
    for (int p = 0; p < 6; ++p) {
      bench[p].get_from_bytes(bytes + 24 * p);
    }
    active.get_from_bytes(bytes); // TODO
  }
};

struct Battle {
  std::array<Side, 2> sides;
  constexpr void get_from_bytes(const uint8_t const *bytes) {
    sides[0].get_from_bytes(bytes);
    sides[1].get_from_bytes(bytes + 184);
  }
};

};


// A battle that also caches information to make calculating the MatchupTable entries and hash easier.
template <typename State>
struct StateWithBucketing {
  // using State::State;
  State state;
  Bucketing::Battle battle;

  Bucketing::BaseStatData base_data;
  uint64_t pre_hashes[7][2];
  // this gets sigmoided and updated incrementally
  float pre_value{};

  template <typename ...Args>
  StateWithBucketing(Args ... &&args) : state{args}, battle{state.battle().bytes} {}

  void apply_actions (auto p1_action, auto p2_action) {
    state.apply_actions(p1_action, p2_action);
    const auto obs = state.obs();
    if (p1_action == 0) {
      // pass, no change must have occured to their side?. Not necessarily true, toxic/leach
    }
    if (p1_action.switch) {
      // reset_active
      sides[0].active.reset();
      // get active again cus stuff could have happened
      // leave old bench alone, it must have same status and hp
    } else {

    }

    // make sure incremental updates to bucketed battle are working
    assert(sides == sides{state.battle().bytes});
  }

  auto hash () const noexcept {
    uint64_t result = 0;
    for (auto side = 0; side < 2; ++side) {
      for (auto index = 0; index < 7; ++index) {
        result ^= pre_hashes[side][index];
      }
    }
    return result;
  }

  float get_value () const noexcept {
    return 0;
    // sigmoid on pre_value
  }

};
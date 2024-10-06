#include <iostream>
#include <mutex>

#include <types/random.h>
#include <types/vector.h>

#include <battle/battle.h>
#include <battle/sides.h>
#include <battle/tests.h>
#include <battle/util.h>

struct Types {
  using Real = float;
  template <typename T> using Vector = ArrayBasedVector<9>::Vector<T, uint32_t>;
  using PRNG = prng;
  using Mutex = std::mutex;
  using State = Battle<64, true>;
  struct Model {
    float inference(State &&state) {
      PRNG device{*(reinterpret_cast<uint64_t *>(state.battle().bytes + 376))};
      while (!state.is_terminal()) {
        state.get_actions();
        state.apply_actions(state.row_actions[0], state.col_actions[0]);
      }
      return state.payoff();
    }
  };
};

void obs_sanity_test() {
  Types::State state1{sides[0], sides[1]};
  Types::State state2{sides[0], sides[1]};

  Types::PRNG device;

  state1.apply_actions(0, 0);
  state2.apply_actions(0, 0);
  assert(state1.obs() == state2.obs());
}

int chance_test(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "Expects number of chance tries and seed." << std::endl;
    return 1;
  }
  size_t tries = std::atoi(argv[1]);
  uint64_t seed = std::atoi(argv[2]);

  Types::PRNG device{seed};

  for (size_t t = 0; t < 1; ++t) {
    // const int i = device.random_int(100);
    // const int j = device.random_int(100);
    const int i = 0;
    const int j = 1;

    Types::State state{sides[i], sides[j]};
    state.apply_actions(0, 0);
    state.get_actions();

    const auto p1_action = state.row_actions[device.random_int(state.rows())];
    const auto p2_action = state.col_actions[device.random_int(state.cols())];

    const std::string p1_string =
        side_choice_string(state.battle().bytes, p1_action);
    const std::string p2_string =
        side_choice_string(state.battle().bytes + 184, p2_action);
    std::cout << p1_string << ", " << p2_string << std::endl;

    const auto map = actions_test(device, state, p1_action, p2_action, tries);
    display_actions_test_map(map);
  }
  return 0;
}

int main(int argc, char **argv) { return chance_test(argc, argv); }
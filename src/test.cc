#include <iostream>

#include <battle.h>
#include <debug-log.h>
#include <sides.h>
#include <tests.h>

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

int main() {

  obs_sanity_test();

  int i = 0;
  int j = 1;

  Types::State state{sides[i], sides[j]};
  Types::PRNG device;
  state.apply_actions(0, 0);
  state.get_actions();

  print_moves(state.battle().bytes + 184);
  print_species(state.battle().bytes + 184);

  const auto map = actions_test(
      device, state, state.row_actions[device.random_int(state.rows())],
      state.col_actions[device.random_int(state.cols())], 10000);
  display_actions_test_map(map);

  copy_rollout_test(device, state);

  return 0;
}
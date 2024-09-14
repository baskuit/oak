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
  using State = Battle<64, false>;
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

int main() {

  int i = 0;
  int j = 1;

  Types::State state{sides[i], sides[j]};
  Types::PRNG device{203492839};

  state.get_actions();
  copy_rollout_test(device, state);
}
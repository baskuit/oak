#include <iostream>
#include <mutex>

#include <types/random.h>
#include <types/vector.h>

#include <battle.h>
#include <chance.h>
#include <debug-log.h>
#include <sides.h>

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

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Provide two team indices and a seed" << std::endl;
    return 1;
  }

  int i = std::atoi(argv[1]);
  int j = std::atoi(argv[2]);
  int seed = std::atoi(argv[3]);

  Types::State state{sides[i], sides[j]};
  Types::PRNG device{static_cast<uint64_t>(seed)};

  DebugLog<Types::State::log_buffer_size> debug_log{};
  debug_log.rollout_battle(std::move(state), device);
  debug_log.save_data_to_path("");

  return 0;
}
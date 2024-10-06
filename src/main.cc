#include <iostream>
#include <mutex>

#include <types/random.h>
#include <types/vector.h>

#include <battle/battle.h>
#include <battle/chance.h>
#include <battle/data/data.h>
#include <battle/debug-log.h>
#include <battle/sides.h>

struct Set {
  Data::Species species;
  std::vector<Data::Moves> moves;
};

struct Types {
  using Real = float;
  template <typename T> using Vector = ArrayBasedVector<9>::Vector<T, uint32_t>;
  using PRNG = prng;
  using Mutex = std::mutex;
  using State = Battle<64, false>;
  struct Model {
    float inference(State &&state) {
      PRNG device{*(reinterpret_cast<uint64_t *>(state.battle().bytes + 376))};
      while (!state.terminal()) {
        state.get_actions();
        state.apply_actions(state.row_actions[0], state.col_actions[0]);
      }
      return state.payoff();
    }
  };
};

int main(int argc, char **argv) {
  using Data::Species;
  using Data::Moves;

  if (argc != 4) {
    std::cout << "Provide two team indices (from include/battle/sides.h) and a seed" << std::endl;
    return 1;
  }

  Battle<0, false, false> Battle
  {
    std::vector<Set>{{Species::Tauros, {Moves::BodySlam}}},
    {}
  };

  int i = std::atoi(argv[1]);
  int j = std::atoi(argv[2]);
  int seed = std::atoi(argv[3]);

  Types::State state{sides[i], sides[j]};
  Types::PRNG device{static_cast<uint64_t>(seed)};

  DebugLog<Types::State::log_buffer_size> debug_log{};
  debug_log.rollout_battle(std::move(state), device);
  // save here until we can get pkmn-debug working everywhere
  debug_log.save_data_to_path("./extern/engine/logs");

  return 0;
}

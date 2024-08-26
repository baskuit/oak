#include "./battle.h"
#include "./sides.h"

template <typename State, size_t N_GAMES> struct ActorPool {

  std::vector<State> env_pool{};
  std::atomic<size_t> index{};
  std::atomic<size_t> total{};

  ActorPool(prng &device) {
    env_pool.reserve(N_GAMES);
    for (size_t i{}; i < N_GAMES; ++i) {
      env_pool.emplace_back(sides[device.random_int(n_sides)],
                            sides[device.random_int(n_sides)]);
    }
  }

  void act(prng &device) {
    const int i = this->index.fetch_add(1) % N_GAMES;
    State &state = env_pool[i];
    state.apply_actions(
        state.row_actions[device.random_int(state.row_actions.size())],
        state.col_actions[device.random_int(state.col_actions.size())]);
    if (state.is_terminal()) {
      total.fetch_add(1);
      state = State{sides[device.random_int(n_sides)],
                    sides[device.random_int(n_sides)]};
    } else {
      state.get_actions();
    }
  }
};
#pragma once

template <typename TT, typename State, typename Model, typename PRNG>
void run(TT &tt, PRNG &device, Model &model, const State &state) {}

template <typename TT, typename State, typename Model, typename PRNG,
          typename Outcome>
void run_iteration(TT &tt, PRNG &device, Model &model, const State &state) {

  using UCBNode = decltype(TT::get_node(0));
  

  if (state.is_terminal()) {
    state.payoff();
  }
}
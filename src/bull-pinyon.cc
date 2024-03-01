// #include <pinyon.hh>

// #include "./src/battle.hh"
// #include "./src/print.hh"
// #include "./src/mcm-no-switch.hh"
// #include "./src/monte-carlo-average.hh"

// void rollout_and_save()
// {
//     using S = BattleTypes<true>::State;
//     S state{0, 0};
//     // state.print_log = true;

//     prng device{0};

//     while (!state.is_terminal())
//     {
//         const int row_idx = device.random_int(state.row_actions.size());
//         const int col_idx = device.random_int(state.col_actions.size());
//         const auto row_action = state.row_actions[row_idx];
//         const auto col_action = state.col_actions[col_idx];
//         state.apply_actions(row_action, col_action);
//         state.get_actions();
//     }

//     state.save_debug_log();
// }

int main()
{

    // rollout_and_save();

    return 0;
}

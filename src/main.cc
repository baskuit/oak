#include "../include/battle.hh"

#include "../include/sides.hh"

struct DebugLog
{
    static constexpr size_t frame_size = 64 + 384 + 3;

    std::vector<std::array<uint8_t, frame_size>> frames{};
};

template <typename State>
void rollout_with_debug(State &state, DebugLog &debug_log)
{
    prng device{};

    int frame = 0;

    while (!state.is_terminal())
    {
        const int row_idx = device.random_int(state.row_actions.size());
        const int col_idx = device.random_int(state.col_actions.size());
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];
        // state.apply_actions(row_action, col_action);
        debug_log.frames.emplace_back();
        uint8_t *data = debug_log.frames[frame].data();

        apply_actions_with_log(state, row_action, col_action, data);
        state.get_actions();

        ++frame;
    }
}

int main()
{
    DebugLog debug_log{};
    Battle<64, 0, ChanceObs, float, float>::State
        state{sides[0], sides[0]};
    rollout_with_debug(state, debug_log);

    return 0;
}
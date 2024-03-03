#include "../include/battle.hh"
#include "../include/old-battle.hh"

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
        std::cout << "frame: " << frame << std::endl;
        get_active_hp(state);

        const int row_idx = device.random_int(state.row_actions.size());
        const int col_idx = device.random_int(state.col_actions.size());
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];
        // state.apply_actions(row_action, col_action);
        debug_log.frames.emplace_back();
        uint8_t *data = debug_log.frames[frame].data();

        apply_actions_with_log(state, row_action, col_action, data);
        // state.apply_actions(row_action, col_action);
        state.get_actions();

        ++frame;
    }

    get_active_hp(state);
}

int main()
{
    using U = MonteCarloModel<
        Battle<64, 0, ChanceObs, float, float>>;
    DebugLog debug_log{};
    U::State
        state{sides[0], sides[0]};
    state.apply_actions(0, 0);
    state.get_actions();

    U::Model model{0};
    U::ModelOutput output{};
    // model.inference(std::move(state), output);

    // prng device{};
    // state.randomize_transition(device);
    // state.apply_actions(state.row_actions[0], state.col_actions[0]);
    // state.get_actions();
    // get_active_hp(state);

    rollout_with_debug(state, debug_log);

    return 0;
}
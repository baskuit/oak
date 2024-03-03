#pragma once

#include "./battle.hh"

#include <filesystem>
#include <fstream>

template <typename State>
struct DebugLog
{
    static constexpr size_t header_size = SIZE_BATTLE_WITH_PRNG + 4;
    static constexpr size_t frame_size = State::log_size + SIZE_BATTLE_WITH_PRNG + 3;

    std::array<uint8_t, header_size> header{};
    std::vector<std::array<uint8_t, frame_size>> frames{};

    DebugLog(const State &state)
    {
        header[0] = uint8_t{1};
        header[1] = uint8_t{1};
        header[2] = uint8_t{State::log_size % 256};
        header[3] = uint8_t{State::log_size / 256};
        memcpy(header.data() + 4, state.battle.bytes, SIZE_BATTLE_WITH_PRNG);
    }

    void save(const State &state) const
    {
        const uint8_t *battle_prng_bytes = state.battle.bytes + SIZE_BATTLE_NO_PRNG;
        const uint64_t *seed = reinterpret_cast<const uint64_t *>(battle_prng_bytes);
        const std::string cwd = std::filesystem::current_path();
        const std::string path = cwd + "/logs/" + std::to_string(*seed) + ".log";
        std::fstream file;
        file.open(path, std::ios::binary | std::ios::app);

        file.write(reinterpret_cast<const char *>(header.data()), header_size);

        for (const auto &frame : frames)
        {
            file.write(reinterpret_cast<const char *>(frame.data()), frame_size);
        }

        file.close();
    }

    void print() const
    {
        std::cout << "HEADER: " << std::endl;
        for (int i = 0; i < header_size; ++i)
        {
            std::cout << (int)header[i] << ' ';
        }
        std::cout << std::endl;

        for (const auto &frame : frames)
        {
            std::cout << "FRAME: " << std::endl;
            for (int i = 0; i < frame_size; ++i)
            {
                std::cout << (int)frame[i] << ' ';
            }
            std::cout << std::endl;
        }
    }
};

template <typename State>
void rollout_with_debug(State &state, DebugLog<State> &debug_log)
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
        debug_log.frames.emplace_back();
        uint8_t *data = debug_log.frames[frame].data();
        apply_actions_with_log(state, row_action, col_action, data);
        state.get_actions();
        ++frame;
    }

    get_active_hp(state);
}

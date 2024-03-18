#pragma once

#include <filesystem>
#include <fstream>

#include "./battle.hh"

template <typename State>
struct EvalLog {
    static constexpr size_t header_size = SIZE_BATTLE_WITH_PRNG + 4;
    // assumes matrix data is null
    static constexpr size_t frame_size = State::log_size + SIZE_BATTLE_WITH_PRNG + 3 + 2 + 2 + 2 * 4 * (19);

    std::array<uint8_t, header_size> header{};
    std::vector<std::array<uint8_t, frame_size>> frames{};
    std::vector<int> frame_sizes{};

    EvalLog(const State &state) {
        header[0] = uint8_t{1};
        header[1] = uint8_t{1};
        header[2] = uint8_t{State::log_size % 256};
        header[3] = uint8_t{State::log_size / 256};
        memcpy(header.data() + 4, state.battle.bytes, SIZE_BATTLE_WITH_PRNG);
    }

    void save(const State &state) const {
        const uint8_t *battle_prng_bytes = state.battle.bytes + SIZE_BATTLE_NO_PRNG;
        const uint64_t *seed = reinterpret_cast<const uint64_t *>(battle_prng_bytes);
        const std::string cwd = std::filesystem::current_path();
        const std::string path = cwd + "/logs/" + std::to_string(*seed) + ".log";
        std::fstream file;
        file.open(path, std::ios::binary | std::ios::app);

        file.write(reinterpret_cast<const char *>(header.data()), header_size);

        for (int i = 0; i < frames.size(); ++i) {
            const auto &frame = frames[i];
            file.write(reinterpret_cast<const char *>(frame.data()), frame_sizes[i]);
        }

        file.close();
    }

    void print() const {
        std::cout << "HEADER: " << std::endl;
        for (int i = 0; i < header_size; ++i) {
            std::cout << (int)header[i] << ' ';
        }
        std::cout << std::endl;

        for (const auto &frame : frames) {
            std::cout << '{' << std::endl;
            this->process_frame(frame);
            std::cout << '}' << std::endl << std::endl;
        }
    }

    void process_frame(const std::array<uint8_t, frame_size> &frame) const {
        int index = 0;
        std::cout << "LOG: " << std::endl;
        for (int i = 0; i < State::log_size; ++i) {
            std::cout << (int)frame[index++] << ' ';
        }
        std::cout << std::endl;

        std::cout << "BATTLE: " << std::endl;
        for (int i = 0; i < SIZE_BATTLE_WITH_PRNG; ++i) {
            std::cout << (int)frame[index++] << ' ';
        }
        std::cout << std::endl;

        std::cout << "RESULT: " << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << (int)frame[index++] << ' ';
        }
        std::cout << std::endl;

        int rows = frame[index++];
        int cols = frame[index++];
        std::cout << "ROWS: " << rows << " COLS: " << cols << std::endl;

        const float *data_f = reinterpret_cast<const float *>(frame.data() + index);
        int float_index = 0;
        std::cout << "  ROW DATA: " << std::endl;
        {
            std::cout << "ROW'S VALUE: " << *(data_f + (float_index++)) << std::endl;
            std::cout << "ROW'S ROW POLICY:" << std::endl;
            for (int i = 0; i < rows; ++i) {
                std::cout << *(data_f + (float_index++)) << ' ';
            }
            std::cout << std::endl;
            std::cout << "ROW'S COL POLICY:" << std::endl;
            for (int i = 0; i < cols; ++i) {
                std::cout << *(data_f + (float_index++)) << ' ';
            }
            std::cout << std::endl;

            index += 4 * float_index;
            std::cout << "N ROW MATRICES: " << (int)frame[index++] << std::endl;
        }

        data_f = reinterpret_cast<const float *>(frame.data() + index);
        float_index = 0;
        std::cout << "  COL DATA: " << std::endl;
        {
            std::cout << "COL'S VALUE: " << *(data_f + (float_index++)) << std::endl;
            std::cout << "COL'S ROW POLICY:" << std::endl;
            for (int i = 0; i < rows; ++i) {
                std::cout << *(data_f + (float_index++)) << ' ';
            }
            std::cout << std::endl;
            std::cout << "COL'S COL POLICY:" << std::endl;
            for (int i = 0; i < cols; ++i) {
                std::cout << *(data_f + (float_index++)) << ' ';
            }
            std::cout << std::endl;

            index += 4 * float_index;
            std::cout << "N ROW MATRICES: " << (int)frame[index++] << std::endl;
        }
    }
};

template <typename State>
void rollout_with_debug(State &state, EvalLog<State> &debug_log) {
    prng device{};
    int frame = 0;
    while (!state.is_terminal()) {
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

template <typename State, typename RowModelTypes, typename ColModelTypes>
void rollout_with_eval_debug(State &state, typename RowModelTypes::Model &row_model,
                             typename ColModelTypes::Model &col_model, EvalLog<State> &debug_log) {
    prng device{};
    int frame = 0;
    typename RowModelTypes::ModelOutput row_output{};
    typename ColModelTypes::ModelOutput col_output{};

    while (!state.is_terminal()) {
        row_model.inference(State{state}, row_output);
        col_model.inference(State{state}, col_output);

        const int row_idx = device.sample_pdf(row_output.row_policy);
        const int col_idx = device.sample_pdf(col_output.col_policy);
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];

        debug_log.frames.emplace_back();

        uint8_t *data = debug_log.frames[frame].data();
        const int frame_size =
            apply_actions_with_eval_log(state, row_action, col_action, &row_output, &col_output, data);
        debug_log.frame_sizes.push_back(frame_size);

        state.get_actions();
        ++frame;
    }
}

template <typename State, typename ModelTypes>
void self_play_rollout_with_eval_debug(prng &device, State &state, typename ModelTypes::Model &model,
                                       EvalLog<State> &debug_log) {
    typename ModelTypes::ModelOutput output{};
    while (!state.is_terminal()) {
        const auto start = std::chrono::high_resolution_clock::now();
        model.inference(State{state}, output);
        const auto end = std::chrono::high_resolution_clock::now();
        const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        const int row_idx = device.sample_pdf(output.row_policy);
        const int col_idx = device.sample_pdf(output.col_policy);
        const auto row_action = state.row_actions[row_idx];
        const auto col_action = state.col_actions[col_idx];

        debug_log.frames.emplace_back();
        uint8_t *data = debug_log.frames.back().data();
        const int frame_size = apply_actions_with_eval_log(
            state, row_action, col_action, &output, static_cast<typename ModelTypes::ModelOutput *>(nullptr), data);
        debug_log.frame_sizes.push_back(frame_size);

        state.get_actions();
    }
}

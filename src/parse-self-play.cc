#include <fstream>
#include <iostream>
#include <vector>

#include "../include/battle.hh"
#include "../include/eval-log.hh"
#include "../include/old-battle.hh"
#include "../include/sides.hh"

enum MoveKind { Move, Switch };

struct Frame {
    std::array<uint8_t, SIZE_BATTLE_WITH_PRNG> battle{};
    pkmn_choice row_action, col_action;
    std::vector<float> row_policy{}, col_policy{};
    float row_value;
};

struct Trajectory {
    float row_value;
    std::vector<Frame> frames{};
};

Trajectory get_trajectory(const char* data, const int size) {
    static size_t SIZE_SELF_PLAY_FRAME = SIZE_BATTLE_WITH_PRNG + 3;

    char result;

    Trajectory trajectory{};

    trajectory.frames.emplace_back();
    memcpy(trajectory.frames[0].battle.data(), data + 4, SIZE_BATTLE_WITH_PRNG);

    // header info + battle
    int index{4 + SIZE_BATTLE_WITH_PRNG};
    while (index < size) {
        Frame& frame = trajectory.frames.back();

        // add 'next frame' to park the battle data
        trajectory.frames.emplace_back();

        // battle copied to next frame
        // other info added to current frame - battles are 'ahead' by 1...
        memcpy(trajectory.frames.back().battle.data(), data + index, SIZE_BATTLE_WITH_PRNG);
        index += SIZE_BATTLE_WITH_PRNG;

        // result
        result = *(data + index);
        // std::cout << "result: " << (int)result << std::endl;
        index += 1;

        // row action
        frame.row_action = *(data + index);
        // std::cout << "row_action: " << (int)frame.row_action << std::endl;
        index += 1;

        // col action
        frame.col_action = *(data + index);
        // std::cout << "col_action: " << (int)frame.col_action << std::endl;
        index += 1;

        const int rows = *(data + index);
        // std::cout << "rows: " << (int)rows << std::endl;
        frame.row_policy.resize(rows);
        index += 1;

        const int cols = *(data + index);
        // std::cout << "cols: " << (int)cols << std::endl;
        frame.col_policy.resize(cols);
        index += 1;

        const float* float_ptr = reinterpret_cast<const float*>(data + index);
        frame.row_value = *float_ptr;
        ++float_ptr;
        for (int row_idx{}; row_idx < frame.row_policy.size(); ++row_idx) {
            frame.row_policy[row_idx] = *float_ptr;
            ++float_ptr;
        }
        for (int col_idx{}; col_idx < frame.col_policy.size(); ++col_idx) {
            frame.col_policy[col_idx] = *float_ptr;
            ++float_ptr;
        }

        // for both players: value + row_policy + col_policy
        const int total_floats = 2 * (1 + frame.row_policy.size() + frame.col_policy.size());
        // skip bytes for floats plus both n_matrix bytes (zero'd)
        const int total_bytes_skipped = 4 * total_floats + 2;
        index += total_bytes_skipped;
    }

    switch (pkmn_result_kind(result)) {
        case PKMN_RESULT_WIN: {
            trajectory.row_value = 1.0;
        }
        case PKMN_RESULT_LOSE: {
            trajectory.row_value = 1.0;
        }
        case PKMN_RESULT_TIE: {
            trajectory.row_value = 0.5;
        }
        case PKMN_RESULT_ERROR: {
            std::exception();
        }
    }

    return trajectory;
}

void open_file_and_get_trajectory(std::string file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    // Get the size of the file
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file into a vector
    std::vector<char> fileData(fileSize);
    file.read(fileData.data(), fileSize);

    // Get a pointer to the file data
    char* data = fileData.data();

    // Output the number of bytes in the file
    std::cout << "Number of bytes in the file: " << fileSize << std::endl;

    // You can now use 'fileDataPtr' to access the bytes of the file

    // Don't forget to close the file
    file.close();

    get_trajectory(data, fileSize);
}

int main() {
    std::string demo_path = "/home/user/oak/logs/12463111002853059008.log";
    open_file_and_get_trajectory(demo_path);

    return 0;
}
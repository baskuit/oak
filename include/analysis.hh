#include <filesystem>
#include <fstream>
#include <pinyon.hh>

#include "./battle.hh"

struct AnalysisData {
  // row + cols + value + policy
  constexpr static size_t max_size = 2 + 4 * (1 + 9 + 9);

  using Frame = std::array<uint8_t, SIZE_BATTLE_WITH_PRNG + max_size>;
  std::vector<Frame> frames{};

  // use inference
  template <typename Types>
  void push(Types::State &&state, Types::Model &model) {

    const int rows = state.row_actions.size();
    const int cols = state.col_actions.size();

    typename Types::ModelOutput output{};
    model.inference(std::move(state), output);

    frames.emplace_back();
    Frame &frame = frames.back();
    memcpy(frame.data(), state.battle.bytes, SIZE_BATTLE_WITH_PRNG);
    int index{SIZE_BATTLE_WITH_PRNG};
    frame[index++] = static_cast<uint8_t>(rows);
    frame[index++] = static_cast<uint8_t>(cols);
    write_real_as_float(output.value.get_row_value(), frame.data(), index);
    for (int row_idx{}; row_idx < rows; ++row_idx) {
      write_real_as_float(output.row_policy[row_idx], frame.data(), index);
    }
    for (int col_idx{}; col_idx < cols; ++col_idx) {
      write_real_as_float(output.col_policy[col_idx], frame.data(), index);
    }

    std::cout << "cheat" << std::endl;
    math::print(output.row_policy);
    math::print(output.col_policy);
  }

  void print() const {
    for (const auto &frame : frames) {
      print_frame(frame);
    }
  }

  void print_frame(const Frame &frame) const {
    int index = 0;
    std::cout << "BATTLE: " << std::endl;
    for (int i = 0; i < SIZE_BATTLE_WITH_PRNG; ++i) {
      std::cout << (int)frame[index++] << ' ';
    }
    std::cout << std::endl;

    const int rows = frame[index++];
    const int cols = frame[index++];
    std::cout << "ROWS: " << rows << " COLS: " << cols << std::endl;

    const float *data_f = reinterpret_cast<const float *>(frame.data() + index);
    std::cout << "EVAL DATA: " << std::endl;
    {
      std::cout << "VALUE: " << *(data_f++) << std::endl;
      std::cout << "ROW POLICY:" << std::endl;
      for (int i = 0; i < rows; ++i) {
        std::cout << *(data_f++) << ' ';
      }
      std::cout << std::endl;
      std::cout << "COL POLICY:" << std::endl;
      for (int i = 0; i < cols; ++i) {
        std::cout << *(data_f++) << ' ';
      }
      std::cout << std::endl;
    }
  }
};
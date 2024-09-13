#pragma once

#include <cstring>
#include <mutex>

#include <pkmn.h>

#include <types/array.h>
#include <types/random.h>

#include <exception>

// stick to log and chance only for now

template <size_t log_size, bool chance> struct OptionsData;

template <> struct OptionsData<0, false> {
  pkmn_gen1_battle_options options;
  void set () {
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  }
};

template <> struct OptionsData<0, true> {
  pkmn_gen1_battle_options options;
  pkmn_gen1_chance_options chance_options;
  void set () {
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  }
};

template <size_t log_size> struct OptionsData<log_size, false> {
  pkmn_gen1_battle_options options;
  uint8_t log_buffer[log_size];
  void set() {
    pkmn_gen1_log_options log_options = {.buf = log_buffer,
                                         .len = log_size};
    pkmn_gen1_battle_options_set(&options, &log_options, nullptr, nullptr);
  }
};

template <size_t log_size> struct OptionsData<log_size, true> {
  pkmn_gen1_battle_options options;
  uint8_t log_buffer[log_size];
  pkmn_gen1_chance_options chance_options;
  void set () {
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  }
};

template <size_t log_size, bool chance> struct Battle {
  pkmn_gen1_battle battle;
  OptionsData<log_size, chance> options_data;

  pkmn_result result;
  uint8_t _rows;
  uint8_t _cols;
  std::array<uint8_t, 9> row_actions;
  std::array<uint8_t, 9> col_actions;

  Battle(const uint8_t *p1_side, const uint8_t *p2_side) {
    std::memcpy(this->battle.bytes, p1_side, 184);
    std::memcpy(this->battle.bytes + 184, p2_side, 184);
    std::memset(this->battle.bytes + 2 * 184, 0, 376 - 2 * 184);
    options_data.set();
  }

  template <typename PRNG> void randomize_transition(PRNG &device) noexcept {
    uint8_t *const battle_prng_bytes = battle.bytes + 376;
    *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
  }

  void apply_actions(pkmn_choice p1_action, pkmn_choice p2_action) {
    options_data.set();
    result = pkmn_gen1_battle_update(&battle, p1_action, p2_action, &options_data.options);
  };

  float payoff() const noexcept {
    switch (pkmn_result_type(this->result)) {
    case PKMN_RESULT_NONE: {
      return .5;
    }
    case PKMN_RESULT_WIN: {
      return 1;
    }
    case PKMN_RESULT_LOSE: {
      return 0;
    }
    default: {
      std::cerr << "battle error" << std::endl;
      std::terminate();
    }
    }
  }

  bool is_terminal() const noexcept { return pkmn_result_type(this->result); }

  void get_actions() {
    this->_rows = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P1, pkmn_result_p1(result), row_actions.data(),
        PKMN_CHOICES_SIZE);
    this->_cols = pkmn_gen1_battle_choices(
        &battle, PKMN_PLAYER_P2, pkmn_result_p2(result), col_actions.data(),
        PKMN_CHOICES_SIZE);
  }

  auto rows() const noexcept { return this->_rows; }

  auto cols() const noexcept { return this->_cols; }
};
#pragma once

#include <array>
#include <cstring>
#include <exception>
#include <iostream>

#include <pkmn.h>

template <size_t log_size, bool chance> struct OptionsData;

template <> struct OptionsData<0, false> {
  pkmn_gen1_battle_options options;
  void set() noexcept {
    // pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  }
  using Obs = std::array<uint8_t, PKMN_GEN1_BATTLE_SIZE>;
};

template <> struct OptionsData<0, true> {
  pkmn_gen1_battle_options options;
  void set() noexcept {
    pkmn_gen1_battle_options_set(&options, nullptr, nullptr, nullptr);
  }
  using Obs = std::array<uint8_t, PKMN_GEN1_CHANCE_ACTIONS_SIZE>;
  const Obs &obs() const noexcept {
    return *reinterpret_cast<Obs *>(
        pkmn_gen1_battle_options_chance_actions(&options)->bytes);
  }
};

template <size_t log_size> struct OptionsData<log_size, false> {
  pkmn_gen1_battle_options options;
  uint8_t log_buffer[log_size];
  void set() noexcept {
    const pkmn_gen1_log_options log_options{.buf = log_buffer, .len = log_size};
    pkmn_gen1_battle_options_set(&options, &log_options, nullptr, nullptr);
  }
  using Obs = std::array<uint8_t, log_size>;
  const Obs &obs() const noexcept {
    return *reinterpret_cast<Obs *>(log_buffer);
  }
};

template <size_t log_size> struct OptionsData<log_size, true> {
  pkmn_gen1_battle_options options;
  uint8_t log_buffer[log_size];
  void set() noexcept {
    const pkmn_gen1_log_options log_options{.buf = log_buffer, .len = log_size};
    pkmn_gen1_chance_options chance_options{};
    pkmn_rational_init(&chance_options.probability);
    pkmn_gen1_battle_options_set(&options, &log_options, &chance_options,
                                 nullptr);
  }
  using Obs = std::array<uint8_t, PKMN_GEN1_CHANCE_ACTIONS_SIZE>;
  const Obs &obs() const noexcept {
    return *reinterpret_cast<Obs *>(
        pkmn_gen1_battle_options_chance_actions(&options)->bytes);
  }
};

template <size_t log_size, bool chance, bool clamp = false> class Battle {

private:
  pkmn_gen1_battle _battle;
  OptionsData<log_size, chance> _options_data;
  pkmn_result _result;
  uint8_t _rows;
  uint8_t _cols;

public:
  static constexpr size_t log_buffer_size{log_size};

  std::array<uint8_t, 9> row_actions;
  std::array<uint8_t, 9> col_actions;

  Battle(const uint8_t *p1_side, const uint8_t *p2_side) : _result{} {
    std::memcpy(_battle.bytes, p1_side, 184);
    std::memcpy(_battle.bytes + 184, p2_side, 184);
    std::memset(_battle.bytes + 2 * 184, 0, 376 - 2 * 184);
    _options_data.set();
  }

  Battle(const Battle &other)
      : _battle{other._battle}, _result{other._result}, _rows{other._rows},
        _cols{other._cols} {
    std::copy(other.row_actions.begin(), other.row_actions.begin() + _rows,
              row_actions.begin());
    std::copy(other.col_actions.begin(), other.col_actions.begin() + _cols,
              col_actions.begin());
    // not required to copy log buffer for search - even if they are used as the
    // obs()
    if constexpr (log_size > 0) {
      std::copy(other._options_data.log_buffer,
                other._options_data.log_buffer + log_size,
                _options_data.log_buffer);
    }
    if constexpr (chance) {
      // pkmn_gen1_chance_options chance_options;
      // pkmn_gen1_battle_options_set(&_options_data.options, nullptr,
      // &other.obs(), nullptr);
    }
    _options_data.set();
  }

  // Battle() = default;
  ~Battle() = default;
  Battle(Battle &&) = default;

  template <typename PRNG> void randomize_transition(PRNG &device) noexcept {
    uint8_t *const battle_prng_bytes = _battle.bytes + 376;
    *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = device.uniform_64();
  }

  void randomize_transition(uint64_t seed) noexcept {
    uint8_t *const battle_prng_bytes = _battle.bytes + 376;
    *(reinterpret_cast<uint64_t *>(battle_prng_bytes)) = seed;
  }

  void apply_actions(pkmn_choice p1_action, pkmn_choice p2_action) {
    if constexpr (clamp) {
      pkmn_gen1_calc_options calc_options{};
      calc_options.overrides.bytes[0] = 217 + 19 * (_battle.bytes[383] % 3);
      calc_options.overrides.bytes[8] = 217 + 19 * (_battle.bytes[382] % 3);
      pkmn_gen1_battle_options_set(&options(), NULL, NULL, &calc_options);
    } else {
      pkmn_gen1_battle_options_set(&options(), nullptr, nullptr, nullptr);
    }
    _result =
        pkmn_gen1_battle_update(&_battle, p1_action, p2_action, &options());
  };

  std::pair<size_t, size_t> prob() const {
    auto *rat = pkmn_gen1_battle_options_chance_probability(&options());
    pkmn_rational_reduce(rat);
    return {static_cast<size_t>(pkmn_rational_numerator(rat)),
            static_cast<size_t>(pkmn_rational_denominator(rat))};
  }

  float payoff() const {
    switch (pkmn_result_type(_result)) {
    case PKMN_RESULT_NONE: {
      return .5;
    }
    case PKMN_RESULT_WIN: {
      return 1;
    }
    case PKMN_RESULT_LOSE: {
      return 0;
    }
    case PKMN_RESULT_TIE: {
      return .5;
    }
    default: {
      std::cerr << "battle error" << std::endl;
      throw std::exception();
      return .5;
    }
    }
  }

  bool terminal() const noexcept { return pkmn_result_type(_result); }

  void get_actions() {
    _rows = pkmn_gen1_battle_choices(&_battle, PKMN_PLAYER_P1,
                                     pkmn_result_p1(_result),
                                     row_actions.data(), PKMN_CHOICES_SIZE);
    _cols = pkmn_gen1_battle_choices(&_battle, PKMN_PLAYER_P2,
                                     pkmn_result_p2(_result),
                                     col_actions.data(), PKMN_CHOICES_SIZE);
  }

  const auto &obs() const noexcept {
    if constexpr (log_size > 0 || chance) {
      return _options_data.obs();
    } else {
      return *reinterpret_cast<OptionsData<log_size, chance>::Obs *>(
          _battle.bytes);
    }
  }

  auto rows() const noexcept { return _rows; }
  auto cols() const noexcept { return _cols; }
  pkmn_gen1_battle_options &options() noexcept {
    return _options_data.options;
  };
  const pkmn_gen1_battle_options &options() const noexcept {
    return _options_data.options;
  };
  const auto &options_data() const noexcept { return _options_data; }
  auto &options_data() noexcept { return _options_data; }
  pkmn_gen1_battle &battle() noexcept { return _battle; }
  const pkmn_gen1_battle &battle() const noexcept { return _battle; }
  pkmn_result &result() noexcept { return _result; }
  const pkmn_result &result() const noexcept { return _result; }
};

// using ClientBattle = Battle<256, false, false>;
// using SearchBattle = Battle<0, true, true>;
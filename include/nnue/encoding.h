#include <battle/view.h>
#include <data/status.h>

#include <bit>

constexpr auto get_status_index(uint8_t status, uint8_t sleeps) {
  if (!status) {
    return 0;
  }
  auto index = 0;
  if (!Data::is_sleep(status)) {
    index = std::countr_zero(status) - 4;
    assert((index >= 0) && (index < 4));
  } else {
    if (!Data::self(status)) {
      index = 4 + sleeps;
      assert((index >= 4) && (index < 12));
    } else {
      const auto s = status & 7;
      index = 12 + (s - 1);
      assert((index >= 12) && (index < 14));
    }
  }
  return index + 1;
}

void write_and_update_pointer(auto *&ptr, auto value, auto step = 1) {
  ptr[0] = value;
  ptr += step;
}

auto *write_one_hot(auto *data, auto n, auto max) {
  assert(n < max);
  data[n] = 1;
  return data + max;
}

// Order symmertry of bench
void shuffle_pokemon();

using PokemonInput = std::array<float, 151 + 166 + 14>;
using ActiveInput = std::array<float, 151 + 166 + 14>;

struct BattleInput {
  struct Side {
    ActiveInput active;
    std::array<PokemonInput, 5>;
  };

  Side p1;
  Side p2;

  const auto inference(const auto &pokemon_net, const auto &)
};

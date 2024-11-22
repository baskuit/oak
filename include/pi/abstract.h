#include <array>

#include <data/offsets.h>

#include <bit>
#include <cmath>

#include <pkmn.h>

namespace Abstract {

enum class HP : std::underlying_type_t<std::byte> {
  KO,
  ONE,
  TWO,
  THREE,
  FOUR,
  FIVE,
  SIX,
  SEVEN,
};

enum class Status : std::underlying_type_t<std::byte> {
  CLR = 0b00000000,
  PSN = 0b00001000,
  BRN = 0b00010000,
  FRZ = 0b00100000,
  PAR = 0b01000000,
  TOX = 0b10001000,
  SP0 = 0b00000001,
  SP1 = 0b00000010,
  SP2 = 0b00000011,
  SP3 = 0b00000100,
  SP4 = 0b00000101,
  SP5 = 0b00000110,
  SP6 = 0b00000111,
  RS0 = 0b10000001,
  RS1 = 0b10000010,
  RS2 = 0b10000011,
};

// 2 byte
#pragma pack(push, 1)
struct Bench {
  HP hp;
  Status status;

  static constexpr HP get_hp(const uint8_t *const bytes) {
    const auto cur = std::bit_cast<const uint8_t *const>(bytes)[9];
    const auto max = std::bit_cast<const uint8_t *const>(bytes)[0];
    return static_cast<HP>(std::min(7, 8 * (cur + max / 8) / max));
  }

  static constexpr Status get_status(const uint8_t byte, const uint8_t dur) {
    if (byte & 0b00000111) {
      if (!(byte & 0b10000000)) {
        return static_cast<Status>(static_cast<uint8_t>(Status::SP0) + dur);
      }
    }
    return static_cast<Status>(byte);
  }

  constexpr Bench() : hp{HP::SEVEN}, status{Status::CLR} {}
  constexpr Bench(const uint8_t *const bytes) : hp{get_hp(bytes)}, status{} {}
  constexpr Bench(const uint8_t *const bytes, const uint8_t dur)
      : hp{get_hp(bytes)}, status{get_status(bytes[20], dur)} {}
};
#pragma pack(pop)

constexpr int8_t stat_log_ratio(uint16_t current, uint16_t base) {
  const float f = std::log(4 * current / base);
  return f;
}

#pragma pack(push, 1)
struct Active {
  std::array<int8_t, 5> stats;
  uint8_t reflect;
  uint8_t light_screen;
  uint8_t slot;
  uint8_t padding[20 - 8];

  Active() = default;
  constexpr Active(const uint8_t *bytes)
      : stats{}, reflect{bytes[32]}, light_screen{bytes[31]},
        slot{bytes[176 - 144]}, padding{} {}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Side {
  Active active;
  std::array<Bench, 6> bench;

  Side() = default;
  constexpr Side(const uint8_t *const bytes)
      : active{bytes + Offsets::active},
        bench{Bench{bytes + 0 * Offsets::pokemon},
              Bench{bytes + 1 * Offsets::pokemon},
              Bench{bytes + 2 * Offsets::pokemon},
              Bench{bytes + 3 * Offsets::pokemon},
              Bench{bytes + 4 * Offsets::pokemon},
              Bench{bytes + 5 * Offsets::pokemon}} {}

  void update(const uint8_t *const side) noexcept {}
};
#pragma pack(pop)

struct Battle {
  std::array<Side, 2> sides;

  Battle() = default;

  constexpr Battle(const pkmn_gen1_battle &battle) noexcept
      : sides{battle.bytes, battle.bytes + Offsets::side} {}

  void update(const pkmn_gen1_battle *const battle) noexcept {}
};

// namespace Test {
// static_assert(sizeof(Bench) == 2);
// using PokemonBuffer = std::array<uint8_t, 24>;
// constexpr PokemonBuffer mon{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                             0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
// constexpr PokemonBuffer mon_low{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
// constexpr PokemonBuffer mon_kod{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// static_assert(Bench{mon.data()}.hp == HP::SEVEN);
// static_assert(Bench{mon_low.data()}.hp == HP::ONE);
// static_assert(Bench{mon_kod.data()}.hp == HP::KO);
// } // namespace Test

static_assert(sizeof(Active) == 20);
static_assert(sizeof(Bench) == 2);
static_assert(sizeof(Side) == 32);
static_assert(sizeof(Battle) == 64);

}; // namespace Abstract
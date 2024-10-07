#include <array>

#include <battle/util.h>
#include <bit>

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

// number of observed turns spent asleep. REST_2 is identical to SLP_6
enum class Status : std::underlying_type_t<std::byte> {
  None,
  PAR,
  BRN,
  FRZ,
  PSN,
  SLP_0,
  SLP_1,
  SLP_2,
  SLP_3,
  SLP_4,
  SLP_5,
  SLP_6,
  REST_0,
  REST_1, // 14
};

// 2 byte
struct Bench {
  HP hp;
  Status status;

  static constexpr uint8_t get_hp_bucket(const uint8_t *const bytes) {
    const uint16_t current = bytes[18] + 256 * bytes[19];
    const uint16_t max = bytes[0] + 256 * bytes[1];
    return std::min(7, 8 * (current + max / 8) / max);
  }

  constexpr Bench() : hp{HP::SEVEN}, status{Status::None} {}
  constexpr Bench(const uint8_t *bytes)
      : hp{get_hp_bucket(bytes)}, status{bytes[20]} {}
};

namespace BenchTest{
constexpr std::array<uint8_t, 24> mon{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
constexpr std::array<uint8_t, 24> mon_low{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
constexpr std::array<uint8_t, 24> mon_kod{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static_assert(Bench{mon.data()}.hp == HP::SEVEN);
static_assert(Bench{mon_low.data()}.hp == HP::ONE);
static_assert(Bench{mon_kod.data()}.hp == HP::KO);
};

constexpr int8_t stat_log_ratio(uint16_t current, uint16_t base) {
  const float f = std::log(4 * current / base);
  return f;
}

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

// 20 + 12 = 32 byte
struct Side {
  Active active;
  std::array<Bench, 6> bench;

  Side() = default;
  constexpr Side(const uint8_t *bytes) noexcept
      : active{bytes + 144},
        bench{Bench{bytes},      Bench{bytes + 24}, Bench{bytes + 48},
              Bench{bytes + 72}, Bench{bytes + 96}, Bench{bytes + 120}} {}
};

// 64 byte
struct Battle {
  std::array<Side, 2> sides;

  Battle() = default;

  constexpr Battle(const uint8_t *bytes) noexcept : sides{bytes, bytes + 184} {}
};

static_assert(sizeof(Battle) == 64);

}; // namespace Abstract

template <typename State> class WithAbstractBattle : public State {
public:
  Abstract::Battle abstract;

  // template <typename... Args>
  // WithAbstractBattle(Args... args)
  //     : State{args}, abstract{static_cast<State>(*this).battle().bytes} {}

  void apply_actions() {}
};
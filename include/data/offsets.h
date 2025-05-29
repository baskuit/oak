#pragma once

#include <cstddef>

namespace Offsets {

constexpr size_t seed{376};
constexpr size_t side{184};

constexpr size_t pokemon{24};
constexpr size_t species{21};
constexpr size_t order{176};
constexpr size_t active{144};
constexpr size_t active_moves{active + 24};

constexpr size_t moves{10};
constexpr size_t status{20};

} // namespace Offsets

namespace Sizes {
constexpr size_t battle{384};
constexpr size_t durations{8};
}; // namespace Sizes
